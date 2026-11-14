"""Fast score_windows.py: several metrics in one run, windows in parallel, same reports.

Every score is computed by the original code (run_metric.evaluate_one with its Window and
read_events), so values, statuses, saved arrays, per-window scores.sqlite and the report
folders (same config.json, same folder id) are those score_windows.py would write; either
scorer resumes the other's results. What changes:
  * all requested metrics are scored in one pass: each prediction file is read once per
    window (not once per metric), GT arrays are checked once per sequence;
  * windows are scored in parallel processes (--processes), each with its own query threads;
  * chamfer_full (and chamfer_penalty/chamfer_count, which call it) reuse the GT KD-trees
    across methods of a window (experiments/fast_scoring.py; identical values);
  * with --voxel-bin-ms, voxel_* and pmse_*_v2ce_adapted use voxel_metric_v2 (fixed-duration
    bins, the definition used by compare_sequence_strides); without it they use the v1 path;
  * scores already in this report are skipped (as before), and scores saved by a report that
    differs only in its method or sequence list are copied instead of recomputed, when their
    input-file signature matches (--no-reuse turns this off). Only new scores are computed.
The last line printed is JSON {metric: report folder}.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sqlite3
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import score_windows as sw  # noqa: E402  (puts benchmarks on sys.path)
import fast_scoring  # noqa: E402

bench = sw.bench
import voxel_metric_v2 as v2vox  # noqa: E402  (benchmarks/voxel_metric_v2.py)

# Metrics scored with the fixed-duration voxel_metric_v2 when --voxel-bin-ms is given
# (as in compare_sequence_strides.ipynb); without it voxel_* use run_metric's v1 path.
VOXEL_V2 = ("voxel_mae", "voxel_rmse", "voxel_relative_l1", "pmse_2_v2ce_adapted", "pmse_4_v2ce_adapted")
UNITS = {**bench.UNITS, "pmse_2_v2ce_adapted": "squared events/voxel", "pmse_4_v2ce_adapted": "squared events/voxel"}


def uses_v2(args, metric):
    return args.voxel_bin_ms is not None and metric in VOXEL_V2


def v2_config(args, metric):
    """Settings recorded for a voxel_metric_v2 report (changing any gives a new report folder)."""
    files = (Path(v2vox.__file__), Path(bench.__file__), Path(__file__))
    return dict(dataset=args.dataset, split=args.split, metric=metric, unit=UNITS[metric],
                interval="[first frame, last frame)", crop="full v3 crop",
                options=dict(protocol=v2vox.PROTOCOL, bin_ms=float(args.voxel_bin_ms), voxel_version=2),
                numpy_version=np.__version__,
                code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def evaluate_v2(args, win, name, method, file, folder, db, config):
    """run_metric.evaluate_one's bookkeeping, scoring with voxel_metric_v2.score as the strides notebook does."""
    metric, bin_ms = args.metric, config["options"]["bin_ms"]
    ground = method == "ground_truth"
    metadata = file.parent / "input.json"
    sign = signature(args, win, method, file)
    previous = db.execute("SELECT signature,record FROM scores WHERE sequence=? AND method=?",
                          (name, method)).fetchone()
    if previous and previous[0] == sign and reusable(previous[1], folder):
        return "SKIP"
    row = dict(sequence=name, method=method, value=None, status="ok", array_path="")
    try:
        if not ground and not file.is_file():
            row["status"] = "missing_prediction"
        elif method == "v2v":
            row["status"] = "not_applicable"  # v2: native signed V2V arrays are not applicable
        elif not win.source.is_dir():
            row["status"] = "missing_source"
        elif not all((win.source / f"ev_{k}.npy").is_file() for k in "txyp"):
            row["status"] = "missing_gt"
        else:
            edges, meta = win.geometry
            if not ground:
                bench.verify_metadata(metadata, meta, args, name)
            events = win.gt if ground else bench.read_events(file, meta)
            bins = v2vox.time_edges(0., float(edges[-1]), bin_ms)
            result = v2vox.score(metric, events, win.gt, edges=bins, height=meta["height"],
                                 width=meta["width"], bin_ms=bin_ms)
            row.update({k: float(v) if np.isfinite(v) else None for k, v in result.items()})
            row.update(bin_ms=bin_ms, temporal_bins=len(bins) - 1, last_bin_ms=1000 * float(np.diff(bins)[-1]))
            if row["value"] is None:
                row["status"] = "undefined"
            if signature(args, win, method, file) != sign:
                raise RuntimeError("Input changed during scoring; rerun this metric")
    except Exception as exc:
        row.update(value=None, status="error", array_path="")
        print(f"  {name}/{method}: {exc}", flush=True)
    db.execute("INSERT OR REPLACE INTO scores VALUES (?,?,?,?)",
               (name, method, sign, json.dumps(row, allow_nan=False)))
    return row["status"]


def metric_args(args, metric):
    """The namespace score_windows.py builds for one metric."""
    one = copy.copy(args)
    one.metric = metric
    one.chunk_size = 131072 if metric in bench.FULL_CHAMFERS else 4096
    one.max_points, one.count_weight, one.rate_bin_s, one.voxel_block_size = 20000, 1., .01, 1
    return one


def signature(args, win, method, file):
    """The input signature run_metric.evaluate_one computes for this score."""
    ground = method == "ground_truth"
    paths = [win.source / "frame_ts.npy"]
    if not ground:
        paths.extend([file.parent / "input.json", file])
    if ground or args.metric not in bench.DESCRIPTIVE:
        paths.extend(win.source / f"ev_{k}.npy" for k in "txyp")
    return bench.digest([bench.stamp(p) for p in paths])


def reusable(record, folder):
    """evaluate_one's own condition for skipping a saved score."""
    row = json.loads(record)
    return row["status"] not in ("error", "missing_source", "missing_gt", "missing_prediction") and (
        not row.get("array_path") or (folder / row["array_path"]).is_file())


def adopt_saved(db, folder, donors, name, method, sign, rel_window):
    """Copy a score saved by another report with the same settings and the same input files."""
    row = db.execute("SELECT signature,record FROM scores WHERE sequence=? AND method=?", (name, method)).fetchone()
    if row and row[0] == sign and reusable(row[1], folder):
        return False
    for donor in donors:
        other = Path(donor) / rel_window
        if not (other / "scores.sqlite").is_file():
            continue
        with closing(sqlite3.connect(other / "scores.sqlite", timeout=60)) as source:
            try:
                saved = source.execute("SELECT signature,record FROM scores WHERE sequence=? AND method=?",
                                       (name, method)).fetchone()
            except sqlite3.Error:
                continue
        if not saved or saved[0] != sign or not reusable(saved[1], other):
            continue
        array = json.loads(saved[1]).get("array_path")
        if array:
            (folder / array).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(other / array, folder / array)
        db.execute("INSERT OR REPLACE INTO scores VALUES (?,?,?,?)", (name, method, sign, saved[1]))
        return True
    return False


def score_sequence_windows(task):
    """Worker: score every metric and method on some windows of one sequence."""
    fast_scoring.limit_threads(task["threads"])
    fast_scoring.install(bench)
    cache = {}
    original_read = bench.read_events

    def read_events(path, meta):  # one decompression per prediction file and window
        key = (str(path), json.dumps(meta, sort_keys=True))
        if key not in cache:
            cache.clear()
            cache[key] = original_read(path, meta)
        return cache[key]

    bench.read_events = read_events
    source, name = Path(task["source"]), task["name"]
    files, stamps, _, base, _ = bench.v2.sequence_info(source)
    raw = {}
    if task["gt_checked"]:
        raw.update({k: np.load(source / f"ev_{k}.npy", mmap_mode="r") for k in "txyp"})
    pred = Path(task["pred"])
    jobs = task["jobs"]  # per metric: args, config, report, methods, donors
    rows, reused = [], 0
    for first, indices in task["windows"]:
        indices = np.asarray(indices)
        label = f"frames_{first}_{indices[-1]}"
        win = sw.Window((source, files, stamps, base), indices, raw)
        fast_scoring.forget_reference()
        methods = list(dict.fromkeys(m for job in jobs.values() for m in job["methods"]))
        for method in methods:
            cache.clear()
            for metric, job in jobs.items():
                if method not in job["methods"]:
                    continue
                args = job["args"]
                report = Path(job["report"])
                folder = report / "windows" / name / label
                folder.mkdir(parents=True, exist_ok=True)
                file = pred / method / name / task["variant"] / label / ("voxels.npz" if method == "v2v" else "events.npz")
                with closing(sqlite3.connect(folder / "scores.sqlite", timeout=60)) as db, db:
                    db.execute("CREATE TABLE IF NOT EXISTS scores (sequence TEXT, method TEXT, signature TEXT, "
                               "record TEXT, PRIMARY KEY(sequence,method))")
                    if job["donors"]:
                        adopted = adopt_saved(db, folder, job["donors"], name, method,
                                              signature(args, win, method, file), Path("windows") / name / label)
                        reused += adopted
                    evaluate = evaluate_v2 if uses_v2(args, metric) else bench.evaluate_one
                    evaluate(args, win, name, method, file, folder, db, job["config"])
                    record, = db.execute("SELECT record FROM scores WHERE sequence=? AND method=?",
                                         (name, method)).fetchone()
                row = json.loads(record)
                if row.get("array_path"):
                    row["array_path"] = (folder / row["array_path"]).relative_to(report).as_posix()
                rows.append((metric, first, job["methods"].index(method), dict(
                    dataset=args.dataset, split=args.split, experiment=args.experiment_name,
                    k=args.k, m=args.m, start_stride=args.start_stride, first=first,
                    last=int(indices[-1]), duration_s=win.geometry[1]["duration"],
                    metric=metric, unit=job["config"]["unit"], **row)))
    fast_scoring.forget_reference()
    return name, rows, reused


def evaluate_many(args):
    """score_windows.evaluate for each metric in args.metrics; returns ({metric: report}, failed)."""
    for name in [args.dataset, args.experiment_name, *(args.sequences or [])]:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid name: {name!r}")
    if args.k < 2 or args.m < 0 or min(args.start_stride, args.workers, args.voxel_temporal_bins) < 1:
        raise ValueError("Invalid sampling or scoring parameters")
    metrics = list(dict.fromkeys(args.metrics))
    if args.voxel_bin_ms is not None and not (np.isfinite(args.voxel_bin_ms) and args.voxel_bin_ms > 0):
        raise ValueError("--voxel-bin-ms must be positive")
    adapted = [m for m in metrics if m.endswith("_adapted")]
    if adapted and args.voxel_bin_ms is None:
        raise ValueError(f"{adapted} are voxel_metric_v2 metrics; give --voxel-bin-ms")
    if args.legacy_frame_voxels and any(not metric.startswith("voxel_") for metric in metrics):
        raise ValueError("--legacy-frame-voxels requires voxel_* metrics only")
    pred = ROOT / "data_test_exp" / args.dataset
    data = args.data_root or bench.v2.old.setting("EVTWEEN_DATA", bench.v2.old.LAB / "evtween/data")
    split_dir = data / args.dataset / args.split
    names = sorted(set(args.sequences or [p.name for p in split_dir.iterdir() if p.is_dir()]))
    variant = Path(args.experiment_name) / f"k_{args.k}"
    if args.m:
        variant /= f"m_{args.m}"
    found = list(dict.fromkeys(args.methods or [method for method in bench.v2.old.METHODS
                 if any((pred / method / n / variant).is_dir() for n in names)]))
    if not found:
        raise ValueError(f"No saved methods found for {variant} in {pred}")
    evaluator = bench.digest(Path(sw.__file__).read_text())
    jobs = {}
    for metric in metrics:
        one = metric_args(args, metric)
        methods = (["ground_truth"] if metric in bench.DESCRIPTIVE else []) + found
        config = v2_config(one, metric) if uses_v2(one, metric) else bench.settings(one)
        config.update(experiment=args.experiment_name, k=args.k, m=args.m, start_stride=args.start_stride,
                      sequences=names, methods=methods, evaluator=evaluator)
        report = (ROOT / "reports" / "experiments" / args.dataset / args.split / variant / metric /
                  bench.digest(config)[:12])
        with bench.atomic_file(report / "config.json") as f:
            json.dump(config, f, indent=2)
        # Reports of this metric that differ only in their method or sequence lists hold valid
        # scores for the windows and methods they share (a score is reused only if its input
        # signature also matches, exactly as evaluate_one decides to skip).
        same = {key: value for key, value in config.items() if key not in ("methods", "sequences")}
        donors = []
        if args.reuse:
            for other in sorted(report.parent.glob("*/config.json")):
                if other.parent == report:
                    continue
                try:
                    saved = json.loads(other.read_text())
                except (OSError, ValueError):
                    continue
                if {key: value for key, value in saved.items() if key not in ("methods", "sequences")} == same:
                    donors.append(str(other.parent))
        jobs[metric] = dict(args=one, config=config, report=str(report), methods=methods, donors=donors)

    cpus = fast_scoring.allocated_cpus()
    tasks = []
    for name in names:
        source = split_dir / name
        files, _, _, _, _ = bench.v2.sequence_info(source)
        offsets = np.arange(args.k) * (args.m + 1)
        windows = [(first, (first + offsets).tolist())
                   for first in range(0, len(files)-int(offsets[-1]), args.start_stride)]
        needs_gt = any(metric not in bench.DESCRIPTIVE or "ground_truth" in job["methods"]
                       for metric, job in jobs.items())
        checked = needs_gt and fast_scoring.validated_gt(source) is not None
        tasks.append(dict(source=str(source), name=name, windows=windows, gt_checked=checked))
    processes = max(1, min(args.processes or cpus, sum(len(t["windows"]) for t in tasks) or 1))
    threads = max(1, cpus // processes)
    chunks = []
    for task in tasks:  # split long sequences so all processes stay busy
        size = max(1, -(-len(task["windows"]) * len(tasks) // (processes * 4)))
        for s in range(0, len(task["windows"]), size):
            chunks.append(dict(task, windows=task["windows"][s:s + size], jobs=jobs, pred=str(pred),
                               variant=str(variant), threads=threads))
    print(f"Scoring {len(metrics)} metrics on {sum(len(t['windows']) for t in tasks)} windows: "
          f"{processes} processes x {threads} threads", flush=True)
    rows_by_metric = defaultdict(list)
    done, reused = Counter(), 0
    if processes == 1:
        results = map(score_sequence_windows, chunks)
        pool = None
    else:
        pool = ProcessPoolExecutor(processes, mp_context=multiprocessing.get_context("spawn"))
        results = pool.map(score_sequence_windows, chunks)
    try:
        for name, rows, adopted in results:
            reused += adopted
            for metric, first, order, row in rows:
                rows_by_metric[metric].append(((names.index(name), first, order), row))
            done[name] += 1
            if done[name] == sum(c["name"] == name for c in chunks):
                print(f"Scored {name}", flush=True)
    finally:
        if pool:
            pool.shutdown()

    if reused:
        print(f"Reused {reused} scores saved by reports with other method/sequence lists", flush=True)
    reports, failed = {}, False
    for metric, job in jobs.items():
        report = Path(job["report"])
        rows = [row for _, row in sorted(rows_by_metric[metric], key=lambda item: item[0])]
        if not rows:
            raise ValueError("No complete windows for the requested settings")
        sw.write_csv(report / "per_window.csv", rows)
        groups = defaultdict(list)
        for row in rows:
            groups[row["method"]].append(row)
        summary = []
        for method, group in groups.items():
            valid = [r for r in group if r["status"] == "ok" and r["value"] is not None]
            summary.append(dict(dataset=args.dataset, split=args.split, experiment=args.experiment_name,
                                k=args.k, m=args.m, start_stride=args.start_stride, method=method, metric=metric,
                                unit=job["config"]["unit"], mean=np.mean([r["value"] for r in valid]) if valid else None,
                                n_valid=len(valid), n_total=len(group),
                                status_counts=json.dumps(Counter(r["status"] for r in group), sort_keys=True)))
        sw.write_csv(report / "summary.csv", summary)
        print(f"{metric}: {report}", flush=True)
        reports[metric] = str(report)
        failed |= any(r["status"] == "error" for r in rows)
    return reports, failed


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--split", choices=("train", "test"), default="test")
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--k", required=True, type=int)
    p.add_argument("--m", default=0, type=int)
    p.add_argument("--start-stride", default=50, type=int)
    p.add_argument("--metrics", nargs="+", choices=UNITS, required=True)
    p.add_argument("--methods", nargs="+", choices=bench.v2.old.METHODS)
    p.add_argument("--sequences", nargs="+")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--workers", type=int, default=1,
                   help="recorded in the report config as in score_windows.py; query threads come from --processes")
    p.add_argument("--processes", type=int, help="parallel worker processes (default: allocated CPUs)")
    p.add_argument("--no-reuse", dest="reuse", action="store_false",
                   help="do not copy matching scores from reports with other method/sequence lists")
    p.add_argument("--voxel-temporal-bins", type=int, default=160, help="v1 voxel metrics only")
    p.add_argument("--voxel-bin-ms", type=float,
                   help="score voxel_mae/rmse/relative_l1 and pmse_*_v2ce_adapted with voxel_metric_v2 "
                        "(fixed bins of this many ms, as in compare_sequence_strides); omit for v1")
    p.add_argument("--legacy-frame-voxels", action="store_true")
    args = p.parse_args()
    reports, failed = evaluate_many(args)
    print(json.dumps(reports), flush=True)
    sys.exit(int(failed))
