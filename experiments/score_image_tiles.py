"""Score saved 256x256 event streams against the matching tiles of recorded GT."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks import voxel_metric_v2 as voxels
from . import fast_scoring
from .generate_tiled_event_streams import METHODS, ROOT, TILE_SIZE, path_component, tile_layout
from .score_windows import Window, bench, write_csv

VOXEL_METRICS = (
    "voxel_mae", "voxel_rmse", "voxel_relative_l1",
    "pmse_2_v2ce_adapted", "pmse_4_v2ce_adapted",
)
ISI_STATS = ("isi_median_ms", "isi_mean_ms", "isi_cv")
METRICS = (
    "chamfer_full", "event_count", "event_count_ratio", "event_rate", "on_percentage",
    *ISI_STATS, "iei_distance", "same_ts", "fano_factor", *VOXEL_METRICS,
)
OPTIONS = {
    "event_rate": {"bin_s": .01},
    "same_ts": {"window_s": .05, "resolution_s": 1e-9},
    "fano_factor": {"window_s": .05, "bin_s": .002, "bin_px": 8, "eps": .001},
}
VALID_STATUS = {"ok", "partial_support"}


def tile_window(source, info, tile, raw):
    files, stamps, _, base, size = info
    meta = dict(base, width=TILE_SIZE, height=TILE_SIZE, source_size=list(size),
                crop=tile.crop, tile_index=tile.index, tile_row=tile.row, tile_column=tile.column)
    window = Window((source, files, stamps, meta), np.arange(len(stamps)), raw)
    # The generator leaves split implicit in the source and output paths.
    window.geometry[1].pop("split")
    return window


def score_prediction(events, window, metrics, bin_ms, threads):
    edges, meta = window.geometry
    gt, width, height = window.gt, meta["width"], meta["height"]
    duration = float(edges[-1])
    intervals, voxel_edges = None, None
    records = {}
    for metric in metrics:
        status = "ok"
        try:
            if metric == "chamfer_full":
                results = fast_scoring.chamfer_full(events, gt, width, duration, workers=threads)
                value = results["value"]
                if not all(np.isfinite(results[key]) for key in ("chamfer_on", "chamfer_off")):
                    status = "partial_polarity"
            elif metric in VOXEL_METRICS:
                if voxel_edges is None:
                    voxel_edges = voxels.time_edges(0., duration, bin_ms)
                value = voxels.score(metric, events, gt, edges=voxel_edges, height=height,
                                     width=width, bin_ms=bin_ms)["value"]
            elif metric in ISI_STATS:
                if intervals is None:
                    intervals = bench.m.isi_distribution(events, width, duration)
                if not intervals.size:
                    value, status = None, "no_intervals"
                elif metric == "isi_cv":
                    value = intervals.std() / max(intervals.mean(), 1e-9)
                else:
                    value = np.mean(intervals) if metric == "isi_mean_ms" else np.median(intervals)
            else:
                value, _, status = bench.score(metric, events, window, OPTIONS.get(metric, {}))
            if value is not None and not np.isfinite(value):
                value, status = None, "undefined"
            records[metric] = dict(value=None if value is None else float(value), status=status)
        except Exception as exc:
            records[metric] = dict(value=None, status="error", error=str(exc))
    return records


def cached_scores(path, signature):
    try:
        saved = json.loads(path.read_text())
        if saved["signature"] == signature:
            return {metric: score for metric, score in saved["scores"].items()
                    if score["status"] != "error"}
    except (OSError, ValueError, KeyError):
        pass
    return {}


def score_sequence(task):
    source, pred, folder, methods, metrics, bin_ms, threads, args = task
    fast_scoring.limit_threads(threads)
    name = source.name
    info = bench.v2.sequence_info(source)
    size = info[-1]
    gt_paths = [source / "frame_ts.npy", *(source / f"ev_{key}.npy" for key in "txyp")]
    raw, rows = {}, []
    scored, reused = 0, 0
    for tile in tile_layout(*size):
        window = tile_window(source, info, tile, raw)
        for method in methods:
            file = pred / method / name / tile.name / "events.npz"
            metadata = file.with_name("input.json")
            cache = folder / "scores" / name / tile.name / f"{method}.json"
            paths = [file, metadata, *gt_paths]
            signature = bench.digest([bench.stamp(path) for path in paths])
            scores = cached_scores(cache, signature)
            pending = [metric for metric in metrics if metric not in scores]
            reused += len(metrics) - len(pending)
            if pending and not file.is_file():
                scores.update({metric: dict(value=None, status="missing_prediction") for metric in pending})
            elif pending:
                try:
                    meta = window.geometry[1]
                    bench.verify_metadata(metadata, meta, args, name)
                    events = bench.read_events(file, meta)
                    scores.update(score_prediction(events, window, pending, bin_ms, threads))
                    del events
                    if signature != bench.digest([bench.stamp(path) for path in paths]):
                        raise RuntimeError("Prediction or GT changed during scoring")
                    with bench.atomic_file(cache) as stream:
                        json.dump(dict(signature=signature, scores=scores), stream, allow_nan=False)
                    scored += len(pending)
                except Exception as exc:
                    scores = {metric: dict(value=None, status="error", error=str(exc)) for metric in metrics}
            for metric in metrics:
                rows.append(dict(dataset=args.dataset, split=args.split, sequence=name, method=method,
                                 tile=tile.name, tile_row=tile.row, tile_column=tile.column,
                                 x0=tile.x, y0=tile.y, source_width=size[0], source_height=size[1],
                                 source=str(source), prediction=str(file), metric=metric, **scores[metric]))
        fast_scoring.forget_reference()
    return rows, scored, reused


def find_source(name, methods, pred, data_root, args):
    if args.data_root is None:
        sources = set()
        for method in methods:
            for meta in (pred / method / name).glob("tile_*/input.json"):
                try:
                    source = Path(json.loads(meta.read_text())["source"])
                except (OSError, ValueError, TypeError, KeyError):
                    continue
                if source.parts[-3:] == (args.dataset, args.split, name) and source.is_dir():
                    sources.add(source.resolve())
        if len(sources) > 1:
            raise ValueError(f"{name} points to multiple GT folders. Set DATA_ROOT explicitly.")
        if sources:
            return next(iter(sources))
    return data_root / args.dataset / args.split / name


def evaluate(dataset, split="test", *, methods=None, sequences=None, metrics=METRICS,
             data_root=None, pred_root=None, report_root=None, voxel_bin_ms=10.,
             workers=1, chamfer_threads=4):
    """Score missing entries, reuse valid inputs, and export a long per-tile CSV."""
    path_component(dataset)
    if split not in {"test", "train"} or min(workers, chamfer_threads) < 1:
        raise ValueError("Choose train/test and positive worker and thread counts.")
    if not np.isfinite(voxel_bin_ms) or voxel_bin_ms <= 0:
        raise ValueError("VOXEL_BIN_MS must be positive.")
    metrics = tuple(dict.fromkeys(metrics))
    if not metrics or set(metrics) - set(METRICS):
        raise ValueError(f"Unknown metrics. Choose from {METRICS}.")
    pred = Path(pred_root or ROOT / "data_test_exp").resolve() / f"{dataset}_{TILE_SIZE}" / split
    if not pred.is_dir():
        raise FileNotFoundError(f"No tiled predictions at {pred}. Run generate_tiled_event_streams first.")
    missing = set(methods or ()) - set(METHODS)
    if missing:
        raise ValueError(f"Unknown tiled methods: {', '.join(sorted(missing))}")
    found = {method: {path.parent.parent.name for path in (pred / method).glob("*/tile_*/events.npz")}
             for method in METHODS}
    methods = tuple(dict.fromkeys(methods)) if methods is not None else tuple(m for m in METHODS if found[m])
    if not methods or not any(found[m] for m in methods):
        raise ValueError(f"No events.npz files for the selected methods in {pred}.")
    names = sorted({path_component(name) for name in sequences}) if sequences is not None else sorted(
        set().union(*(found[m] for m in methods)))
    if not names:
        raise ValueError("Select at least one sequence.")
    args = SimpleNamespace(dataset=dataset, split=split, data_root=data_root)
    data = Path(data_root or bench.v2.old.setting("EVTWEEN_DATA", bench.v2.old.LAB / "evtween/data")).resolve()
    files = (__file__, bench.__file__, bench.m.__file__, voxels.__file__, fast_scoring.__file__)
    config = dict(dataset=dataset, split=split, prediction_root=str(pred), data_root=str(data),
                  voxel_bin_ms=float(voxel_bin_ms), options=OPTIONS, metrics=metrics, methods=methods,
                  sequences=names, interval="[first frame, last frame)", tile_size=TILE_SIZE,
                  code_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files})
    root = Path(report_root or ROOT / "reports" / "experiments" / f"{dataset}_{TILE_SIZE}" /
                split / "watch_img_size" / "runs").resolve()
    folder = root / bench.digest(config)[:12]
    jobs = []
    for name in names:
        source = find_source(name, methods, pred, data, args)
        if not (source / "frame_ts.npy").is_file():
            raise FileNotFoundError(f"No source frame timestamps for {name} at {source}. Check DATA_ROOT.")
        jobs.append((source, pred, folder, methods, metrics, voxel_bin_ms, chamfer_threads, args))
    with bench.atomic_file(folder / "config.json") as stream:
        json.dump(config, stream, indent=2)
    rows = []

    def collect(result):
        scored_rows, scored, reused = result
        rows.extend(scored_rows)
        print(f"{scored_rows[0]['sequence']}: {scored:,} scored, {reused:,} loaded from cache", flush=True)

    if workers == 1 or len(jobs) == 1:
        for job in jobs:
            collect(score_sequence(job))
    else:
        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(score_sequence, job) for job in jobs]):
                collect(future.result())
    rows.sort(key=lambda row: (row["sequence"], row["tile"], row["method"], row["metric"]))
    write_csv(folder / "results.csv", rows)
    return rows, methods, folder


def load_example(row):
    """Read the selected prediction and the same GT used to generate its score."""
    source = Path(row["source"])
    info = bench.v2.sequence_info(source)
    tile = next(tile for tile in tile_layout(*info[-1]) if tile.name == row["tile"])
    window = tile_window(source, info, tile, {})
    prediction = Path(row["prediction"])
    args = SimpleNamespace(dataset=row["dataset"], split=row["split"])
    bench.verify_metadata(prediction.with_name("input.json"), window.geometry[1], args, row["sequence"])
    return window.gt, bench.read_events(prediction, window.geometry[1]), window.geometry[1]
