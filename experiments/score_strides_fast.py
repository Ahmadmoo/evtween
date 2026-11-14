"""Parallel version of compare_sequence_strides.ipynb's run_scores (same scores, same cache).

options, measure and input_dir are the notebook's code unchanged; run_scores keeps its
cache keys, records and row order, so the notebook's existing cache is reused and plot_only
reads the same per_sequence.csv. What changes:
  * (sequence, stride) pairs are scored in parallel processes (processes=...);
  * chamfer_full reuses the GT KD-trees across the methods of a pair
    (experiments/fast_scoring.py; identical values).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import multiprocessing
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from score_windows import Window, bench  # noqa: E402
import fast_scoring  # noqa: E402

sys.path.insert(0, str(ROOT / "benchmarks"))
from voxel_metric_v2 import PROTOCOL, score as voxel_score, time_edges  # noqa: E402
import voxel_metric_v2  # noqa: E402

VOXEL_METRICS = ("voxel_mae", "voxel_rmse", "voxel_relative_l1", "pmse_2_v2ce_adapted", "pmse_4_v2ce_adapted")
UNITS = {**bench.UNITS, **{m: "squared events/voxel" for m in VOXEL_METRICS if m.startswith("pmse")}}


# ---------------------------------------------------------------- the notebook's code, unchanged
def options(metric, DEVICE, WORKERS, VOXEL_BINS, VOXEL_BIN_MS):
    if metric in VOXEL_METRICS:
        return dict(metric=metric, options=dict(protocol=PROTOCOL, bin_ms=VOXEL_BIN_MS),
                    code_sha256=hashlib.sha256(Path(voxel_metric_v2.__file__).read_bytes()).hexdigest())
    args = SimpleNamespace(dataset="eds", split="test", metric=metric, device=DEVICE, workers=WORKERS,
                           chunk_size=131072 if metric in bench.FULL_CHAMFERS else 4096, max_points=20000,
                           voxel_temporal_bins=VOXEL_BINS, legacy_frame_voxels=False, voxel_block_size=1,
                           count_weight=1., rate_bin_s=.01)
    return bench.settings(args)


def measure(metric, events, window, opts):
    edges, meta = window.geometry
    if metric in VOXEL_METRICS:
        bins = time_edges(0., edges[-1], opts["bin_ms"])
        result = voxel_score(metric, events, window.gt, edges=bins, height=meta["height"],
                             width=meta["width"], bin_ms=opts["bin_ms"])
        numbers = {k: float(v) if np.isfinite(v) else None for k, v in result.items()}
        numbers.update(bin_ms=opts["bin_ms"], temporal_bins=len(bins)-1, last_bin_ms=1000*np.diff(bins)[-1])
        return numbers, {}, "ok" if numbers["value"] is not None else "undefined"
    if metric in bench.FULL_CHAMFERS:
        kw = {key: opts[key] for key in ("device", "workers", "chunk_size", "v_px_per_ms")}
        if metric == "chamfer_count":
            result = bench.metrics_our.chamfer_count(events, window.gt, meta["width"], edges,
                                                     count_weight=opts["count_weight"], **kw)
        else:
            if metric == "chamfer_penalty":
                kw["height"] = meta["height"]
            result = getattr(bench.m, metric)(events, window.gt, meta["width"], edges[-1], **kw)
        numbers = {k: float(v) if np.isfinite(v) else None for k, v in result.items()}
        status = "ok" if all(numbers[k] is not None for k in ("chamfer_on", "chamfer_off")) else "partial_polarity"
        return numbers, {}, status if numbers["value"] is not None else "undefined"
    value, arrays, status = bench.score(metric, events, window, opts)
    if metric == "isi_distribution" and arrays["isi_ms"].size:
        value = np.median(arrays["isi_ms"])
    return {"value": float(value) if value is not None and np.isfinite(value) else None}, arrays, status


def input_dir(method, name, step, indices, EXPERIMENTS):
    if step == 1:
        return ROOT / "data_test_evt/eds/test" / method / name
    return (ROOT / "data_test_exp/eds" / method / name / EXPERIMENTS[step] /
            f"k_{len(indices)}" / f"m_{step-1}" / f"frames_0_{indices[-1]}")


# ---------------------------------------------------------------- one (sequence, stride) pair
def score_pair(task):
    """The notebook's loop body for one sequence and one stride."""
    fast_scoring.limit_threads(task["threads"])
    fast_scoring.install(bench)
    name, step, p = task["name"], task["step"], task["params"]
    configs, code_id, REPORT = task["configs"], task["code_id"], Path(p["REPORT"])
    METRICS, METHODS, EXPERIMENTS = p["METRICS"], p["METHODS"], p["EXPERIMENTS"]
    source = Path(p["DATA"]) / "eds/test" / name
    files, ts, _, base, _ = bench.v2.sequence_info(source)
    sets = {s: np.arange(0, len(ts), s) for s in EXPERIMENTS}
    last = min(ids[-1] for ids in sets.values())
    if last < 1:
        raise ValueError(f"No common frame interval for {name}")
    window = Window((source, files, ts, base), np.arange(last+1), {})
    stop = window.geometry[0][-1]
    gt_stamps = [bench.stamp(source / f) for f in ("frame_ts.npy", "ev_t.npy", "ev_x.npy", "ev_y.npy", "ev_p.npy")]
    ids = sets[step]
    records, reused, computed = [], 0, 0
    for method in [*METHODS, "ground_truth"]:
        folder = input_dir(method, name, step, ids, EXPERIMENTS)
        path = folder / ("voxels.npz" if method == "v2v" else "events.npz")
        metadata = folder / "input.json"
        ground = method == "ground_truth"
        stamps = gt_stamps if ground else [*gt_stamps, bench.stamp(metadata), bench.stamp(path)]
        events = None
        for metric in METRICS:
            if ground and metric not in bench.DESCRIPTIVE:
                continue
            key = bench.digest([code_id, configs[metric], stamps, base["crop"], int(last), step, method])
            cache = REPORT / "cache" / name / str(step) / method / metric / key
            record = cache / "score.json"
            row = dict(sequence=name, stride=step, m=step-1, selected_frames=len(ids),
                       last_scored_frame=int(last), duration_s=stop, method=method, metric=metric,
                       unit=UNITS[metric], status="ok", value=None)
            if record.exists():
                saved = json.loads(record.read_text())
                if not saved.get("array_path") or (REPORT / saved["array_path"]).is_file():
                    records.append(saved)
                    reused += 1
                    continue
            try:
                if not ground and not path.exists():
                    row["status"] = "missing_prediction"
                elif method == "v2v":
                    row["status"] = "not_applicable"
                else:
                    if events is None:
                        if ground:
                            events = window.gt
                        else:
                            meta = json.loads(metadata.read_text())
                            if (Path(meta["source"]).parts[-3:] != ("eds", "test", name)
                                or meta["indices"] != ids.tolist() or meta["crop"] != base["crop"]
                                or meta["absolute_t0"] != ts[0] or meta["duration"] != ts[ids[-1]]-ts[0]
                                or (step > 1 and not np.array_equal(meta["frame_times_s"], ts[ids]-ts[0]))):
                                raise ValueError("Prediction metadata differs from selected frames or GT")
                            stream = bench.read_events(path, meta)
                            end = np.searchsorted(stream["t"], stop)
                            events = {k: v[:end] for k, v in stream.items()}
                    nums, arrays, status = measure(metric, events, window, configs[metric]["options"])
                    row.update(nums, status=status)
                    if arrays:
                        with bench.atomic_file(cache / "arrays.npz", binary=True) as f:
                            np.savez_compressed(f, **arrays)
                        row["array_path"] = str((cache / "arrays.npz").relative_to(REPORT))
                    computed += 1
                if row["status"] != "missing_prediction":
                    with bench.atomic_file(record) as f:
                        json.dump(row, f, allow_nan=False)
            except Exception as exc:
                row.update(value=None, status="error", error=str(exc))
                print(name, step, method, metric, exc, flush=True)
            records.append(row)
    fast_scoring.forget_reference()
    return name, step, records, reused, computed


def run_scores(*, METRICS, METHODS, SEQUENCES, EXPERIMENTS, DATA, DEVICE, WORKERS, VOXEL_BINS,
               VOXEL_BIN_MS, REPORT, processes=None):
    """The notebook's run_scores, with (sequence, stride) pairs in parallel. Same DataFrame."""
    configs = {metric: options(metric, DEVICE, WORKERS, VOXEL_BINS, VOXEL_BIN_MS) for metric in METRICS}
    code_id = [1, hashlib.sha256((ROOT / "experiments/score_windows.py").read_bytes()).hexdigest()]
    params = dict(METRICS=list(METRICS), METHODS=list(METHODS), EXPERIMENTS=dict(EXPERIMENTS),
                  DATA=str(DATA), REPORT=str(REPORT))
    pairs = [(name, step) for name in SEQUENCES for step in EXPERIMENTS]
    cpus = fast_scoring.allocated_cpus()
    processes = max(1, min(processes or cpus, len(pairs)))
    threads = max(1, cpus // processes)
    tasks = [dict(name=name, step=step, params=params, configs=configs, code_id=code_id, threads=threads)
             for name, step in pairs]
    print(f"Scoring {len(pairs)} sequence/stride pairs: {processes} processes x {threads} threads", flush=True)
    results = {}
    if processes == 1:
        for task in tasks:
            name, step, *rest = score_pair(task)
            results[name, step] = rest
    else:
        with ProcessPoolExecutor(processes, mp_context=multiprocessing.get_context("spawn")) as pool:
            for name, step, *rest in pool.map(score_pair, tasks):
                results[name, step] = rest
    records = []
    for name in SEQUENCES:
        reused = computed = 0
        for step in EXPERIMENTS:
            rows, r, c = results[name, step]
            records += rows
            reused, computed = reused + r, computed + c
        print(f"{name}: {reused} cached, {computed} computed", flush=True)
    return pd.DataFrame(records)
