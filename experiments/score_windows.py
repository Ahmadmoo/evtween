"""Score saved independent frame windows and compute equal-window method means."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from contextlib import closing
from functools import cached_property
from pathlib import Path
import sqlite3
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import run_metric as bench


class Window:
    def __init__(self, sample, indices, raw):
        self.source, files, stamps, base = sample
        self.indices = indices
        self.raw = raw
        selected = stamps[indices]
        self.bounds = selected[[0, -1]]
        edges = selected - selected[0]
        gaps = np.diff(edges)
        meta = dict(base, indices=indices.tolist(), absolute_t0=float(selected[0]),
                    duration=float(edges[-1]), frame_times_s=edges.tolist(),
                    split=self.source.parent.name,
                    uniform_timestamps=bool(np.allclose(gaps, gaps.mean(), rtol=1e-3, atol=1e-7)))
        if indices[1] - indices[0] > 1:
            meta["m"] = int(indices[1] - indices[0] - 1)
        self.geometry = edges, meta

    @cached_property
    def gt(self):
        if not self.raw:
            arrays = {k: np.load(self.source / f"ev_{k}.npy", mmap_mode="r") for k in "txyp"}
            t = arrays["t"]
            if t.ndim != 1 or any(v.shape != t.shape for v in arrays.values()):
                raise ValueError("Invalid GT array shapes")
            if not np.all(np.isfinite(t)) or np.any(t[1:] < t[:-1]):
                raise ValueError("GT timestamps must be finite and sorted")
            self.raw.update(arrays)
        a, b = np.searchsorted(self.raw["t"], self.bounds)
        x, y, w, h = self.geometry[1]["crop"]
        ex, ey = self.raw["x"][a:b], self.raw["y"][a:b]
        keep = (ex >= x) & (ex < x+w) & (ey >= y) & (ey < y+h)
        pol = self.raw["p"][a:b][keep]
        if not np.isin(pol, (-1, 0, 1)).all():
            raise ValueError("Invalid GT polarity")
        return dict(t=self.raw["t"][a:b][keep].astype(float) - self.bounds[0],
                    x=ex[keep].astype(np.int64)-x, y=ey[keep].astype(np.int64)-y,
                    p=np.where(pol > 0, 1, -1).astype(np.int8))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with bench.atomic_file(path) as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    for name in [args.dataset, args.experiment_name, *(args.sequences or [])]:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid name: {name!r}")
    if args.k < 2 or args.m < 0 or min(args.start_stride, args.workers, args.voxel_temporal_bins) < 1:
        raise ValueError("Invalid sampling or scoring parameters")
    pred = ROOT / "data_test_exp" / args.dataset
    data = args.data_root or bench.v2.old.setting("EVTWEEN_DATA", bench.v2.old.LAB / "evtween/data")
    split_dir = data / args.dataset / args.split
    names = sorted(set(args.sequences or [p.name for p in split_dir.iterdir() if p.is_dir()]))
    variant = Path(args.experiment_name) / f"k_{args.k}"
    if args.m:
        variant /= f"m_{args.m}"
    methods = list(dict.fromkeys(args.methods or [method for method in bench.v2.old.METHODS
                  if any((pred / method / n / variant).is_dir() for n in names)]))
    if not methods:
        raise ValueError(f"No saved methods found for {variant} in {pred}")
    if args.metric in bench.DESCRIPTIVE:
        methods.insert(0, "ground_truth")
    config = bench.settings(args)
    config.update(experiment=args.experiment_name, k=args.k, m=args.m, start_stride=args.start_stride,
                  sequences=names, methods=methods, evaluator=bench.digest(Path(__file__).read_text()))
    report = ROOT / "reports" / "experiments" / args.dataset / args.split / variant / args.metric / bench.digest(config)[:12]
    with bench.atomic_file(report / "config.json") as f:
        json.dump(config, f, indent=2)
    rows = []
    for name in names:
        source = split_dir / name
        files, stamps, _, base, _ = bench.v2.sequence_info(source)
        raw = {}
        offsets = np.arange(args.k) * (args.m + 1)
        for first in range(0, len(files)-int(offsets[-1]), args.start_stride):
            indices = first + offsets
            label = f"frames_{first}_{indices[-1]}"
            win = Window((source, files, stamps, base), indices, raw)
            folder = report / "windows" / name / label
            folder.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(folder / "scores.sqlite", timeout=60)) as db, db:
                db.execute("CREATE TABLE IF NOT EXISTS scores (sequence TEXT, method TEXT, signature TEXT, "
                           "record TEXT, PRIMARY KEY(sequence,method))")
                for method in methods:
                    file = pred / method / name / variant / label / ("voxels.npz" if method == "v2v" else "events.npz")
                    bench.evaluate_one(args, win, name, method, file, folder, db, config)
                    record, = db.execute("SELECT record FROM scores WHERE sequence=? AND method=?", (name, method)).fetchone()
                    row = json.loads(record)
                    if row.get("array_path"):
                        row["array_path"] = (folder / row["array_path"]).relative_to(report).as_posix()
                    rows.append(dict(dataset=args.dataset, split=args.split, experiment=args.experiment_name,
                                     k=args.k, m=args.m, start_stride=args.start_stride, first=first,
                                     last=int(indices[-1]), duration_s=win.geometry[1]["duration"],
                                     metric=args.metric, unit=config["unit"], **row))
        print(f"Scored {name}", flush=True)
    if not rows:
        raise ValueError("No complete windows for the requested settings")
    write_csv(report / "per_window.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    summary = []
    for method, group in groups.items():
        valid = [r for r in group if r["status"] == "ok" and r["value"] is not None]
        summary.append(dict(dataset=args.dataset, split=args.split, experiment=args.experiment_name,
                            k=args.k, m=args.m, start_stride=args.start_stride, method=method, metric=args.metric,
                            unit=config["unit"], mean=np.mean([r["value"] for r in valid]) if valid else None,
                            n_valid=len(valid), n_total=len(group),
                            status_counts=json.dumps(Counter(r["status"] for r in group), sort_keys=True)))
    write_csv(report / "summary.csv", summary)
    print(report, flush=True)
    return int(any(r["status"] == "error" for r in rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset")
    p.add_argument("--split", choices=("train", "test"), default="test")
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--k", required=True, type=int)
    p.add_argument("--m", default=0, type=int)
    p.add_argument("--start-stride", default=50, type=int)
    p.add_argument("--metric", choices=bench.UNITS, required=True)
    p.add_argument("--methods", nargs="+", choices=bench.v2.old.METHODS)
    p.add_argument("--sequences", nargs="+")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--voxel-temporal-bins", type=int, default=160)
    p.add_argument("--legacy-frame-voxels", action="store_true")
    args = p.parse_args()
    if args.legacy_frame_voxels and not args.metric.startswith("voxel_"):
        p.error("--legacy-frame-voxels requires a voxel_* metric")
    args.chunk_size = 131072 if args.metric in bench.FULL_CHAMFERS else 4096
    args.max_points, args.count_weight, args.rate_bin_s, args.voxel_block_size = 20000, 1., .01, 1
    sys.exit(evaluate(args))
