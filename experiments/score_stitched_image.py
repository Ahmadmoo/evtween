"""Reconstruct one full-frame event stream from ranked, overlapping 256px tiles."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import score_image_tiles as tiles

PMSE = ("pmse_2_v2ce_adapted", "pmse_4_v2ce_adapted")


def ownership(width, height, means):
    """Assign each pixel to its covering tile with the lowest mean Chamfer."""
    layout = tiles.tile_layout(width, height)
    order = sorted(layout, key=lambda tile: (means.get(tile.name, float("inf")), tile.index))
    owner = np.full((height, width), -1, dtype=np.int32)
    coverage = np.zeros_like(owner)
    for tile in reversed(order):
        x, y, w, h = tile.crop
        owner[y:y+h, x:x+w] = tile.index
        coverage[y:y+h, x:x+w] += 1
    if np.any(owner < 0):
        raise ValueError("Tile layout leaves pixels uncovered.")
    return order, owner, coverage


def stitch_events(paths, source, info, order, owner, args):
    """Keep all timestamps and polarities from the single owner of each pixel."""
    parts = {key: [] for key in "txyp"}
    for tile in sorted(order, key=lambda item: item.index):
        file = paths[tile.name]
        meta = tiles.tile_window(source, info, tile, {}).geometry[1]
        tiles.bench.verify_metadata(file.with_name("input.json"), meta, args, source.name)
        events = tiles.bench.read_events(file, meta)
        x = events["x"].astype(np.int32) + tile.x
        y = events["y"].astype(np.int32) + tile.y
        keep = owner[y, x] == tile.index
        parts["x"].append(x[keep])
        parts["y"].append(y[keep])
        for key in "tp":
            parts[key].append(events[key][keep])
    merged = {key: np.concatenate(arrays) for key, arrays in parts.items()}
    indices = np.argsort(merged["t"], kind="stable")
    return {key: array[indices] for key, array in merged.items()}


def pmse_chunked(prediction, reference, kernel, *, edges, height, width, bin_ms,
                 max_voxels=4_000_000):
    """Existing PMSE definition in bounded time chunks with intact pooling groups."""
    bins = len(edges) - 1
    step = max(1, max_voxels // (2 * height * width * kernel)) * kernel
    streams = (prediction, reference)
    cuts = [np.searchsorted(stream["t"], edges) for stream in streams]
    error = 0.
    for first in range(0, bins, step):
        stop = min(first + step, bins)
        # A preceding bin can contribute by interpolation. A following edge
        # lets the final bin of an interior chunk send its weight onwards.
        base = max(0, first - 1)
        domain = edges[base:min(stop + 2, len(edges))]
        pooled = []
        for stream, offsets in zip(streams, cuts):
            clip = {key: array[offsets[base]:offsets[stop]] for key, array in stream.items()}
            grid = tiles.voxels.event_voxels(clip, domain, height, width,
                                             interpolation="forward", bin_ms=bin_ms)
            means, weights = tiles.voxels._pool_full(grid[:, first-base:stop-base], kernel)
            pooled.append(means)
            del grid
        error += np.sum((pooled[0] - pooled[1])**2 * weights)
    return float(error / (2 * bins * height * width))


def score_full_image(events, window, metrics, bin_ms, threads):
    scores = tiles.score_prediction(events, window, [m for m in metrics if m not in PMSE],
                                    bin_ms, threads)
    for metric in PMSE:
        if metric not in metrics:
            continue
        _, meta = window.geometry
        edges = tiles.voxels.time_edges(0., meta["duration"], bin_ms)
        try:
            value = pmse_chunked(events, window.gt, int(metric.split("_")[1]), edges=edges,
                                 height=meta["height"], width=meta["width"], bin_ms=bin_ms)
            scores[metric] = dict(value=value, status="ok")
        except Exception as exc:
            scores[metric] = dict(value=None, status="error", error=str(exc))
    return scores


def score_sequence(source, paths_by_method, rankings, folder, metrics, bin_ms, threads, args):
    tiles.fast_scoring.limit_threads(threads)
    info = tiles.bench.v2.sequence_info(source)
    files, stamps, _, meta, (width, height) = info
    window = tiles.Window((source, files, stamps, meta), np.arange(len(stamps)), {})
    gt_paths = [source / "frame_ts.npy", *(source / f"ev_{key}.npy" for key in "txyp")]
    rows, computed, reused = [], 0, 0
    for method, paths in paths_by_method.items():
        order, owner, _ = ownership(width, height, rankings.get((method, width, height), {}))
        expected = [paths.get(tile.name) for tile in order]
        missing = [tile.name for tile, path in zip(order, expected) if path is None or not path.is_file()]
        if missing:
            scores = {m: dict(value=None, status="missing_prediction",
                              error=f"Missing {', '.join(missing)}") for m in metrics}
        else:
            inputs = [*gt_paths, files[0]]
            for file in expected:
                inputs.extend([file, file.with_name("input.json")])
            signature = tiles.bench.digest([tiles.bench.stamp(p) for p in inputs])
            cache = folder / "scores" / source.name / f"{method}.json"
            scores = tiles.cached_scores(cache, signature)
            pending = [m for m in metrics if m not in scores]
            reused += len(metrics) - len(pending)
            if pending:
                print(f"Scoring stitched frame: {source.name}/{method}", flush=True)
                try:
                    events = stitch_events(paths, source, info, order, owner, args)
                    scores.update(score_full_image(events, window, pending, bin_ms, threads))
                    del events
                    if signature != tiles.bench.digest([tiles.bench.stamp(p) for p in inputs]):
                        raise RuntimeError("Input files changed during scoring")
                    with tiles.bench.atomic_file(cache) as output:
                        json.dump(dict(signature=signature, scores=scores), output, allow_nan=False)
                    computed += len(pending)
                except Exception as exc:
                    scores = {m: dict(value=None, status="error", error=str(exc)) for m in metrics}
        rows.extend(dict(dataset=args.dataset, split=args.split, sequence=source.name, method=method,
                         source_width=width, source_height=height, metric=m, **scores[m])
                    for m in metrics)
    tiles.fast_scoring.forget_reference()
    return rows, computed, reused


def evaluate(tile_records, rankings, *, dataset, split, metrics, report_root, tile_report,
             rank_sequences, voxel_bin_ms=10., workers=1, chamfer_threads=4):
    """Score each reconstructed sequence/method against its original-size GT."""
    files = [__file__, tiles.__file__, tiles.bench.__file__, tiles.voxels.__file__,
             tiles.bench.m.__file__, tiles.fast_scoring.__file__]
    args = SimpleNamespace(dataset=dataset, split=split)
    by_sequence = defaultdict(lambda: defaultdict(dict))
    priorities = {}
    for row in tile_records:
        key = row["method"], row["source_width"], row["source_height"]
        means = rankings.get(key, {})
        if key not in priorities:
            method, width, height = key
            order, _, _ = ownership(width, height, means)
            priorities[key] = [tile.name for tile in order]
        by_sequence[Path(row["source"])][row["method"]][row["tile"]] = Path(row["prediction"])
    if not by_sequence or min(workers, chamfer_threads) < 1:
        raise ValueError("Select evaluation sequences and positive worker/thread counts.")
    config = dict(dataset=dataset, split=split, voxel_bin_ms=float(voxel_bin_ms),
                  metrics=list(metrics), ownership="lowest_mean_chamfer_then_tile_index",
                  tile_report=str(tile_report), evaluation_sequences=sorted(str(p) for p in by_sequence),
                  rank_sequences=sorted(rank_sequences),
                  priorities=[dict(method=m, width=w, height=h, order=priorities[(m, w, h)])
                              for m, w, h in sorted(priorities)],
                  code_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files})
    folder = Path(report_root).resolve() / tiles.bench.digest(config)[:12]
    with tiles.bench.atomic_file(folder / "config.json") as output:
        json.dump(config, output, indent=2)
    tasks = [(source, dict(paths), rankings, folder, metrics, voxel_bin_ms, chamfer_threads, args)
             for source, paths in sorted(by_sequence.items())]
    rows = []

    def collect(result):
        data, computed, reused = result
        rows.extend(data)
        print(f"{data[0]['sequence']}: {computed:,} stitched metrics scored, {reused:,} cached",
              flush=True)

    if workers == 1 or len(tasks) == 1:
        for task in tasks:
            collect(score_sequence(*task))
    else:
        with ProcessPoolExecutor(min(workers, len(tasks)), mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(score_sequence, *task) for task in tasks]):
                collect(future.result())
    rows.sort(key=lambda r: (r["sequence"], r["method"], r["metric"]))
    tiles.write_csv(folder / "stitched_results.csv", rows)
    return rows, folder
