import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(description="Exact bidirectional Chamfer distance in (x, y, v*t) per polarity, "
                             "between each predicted event stream and the recorded events, per dataset.")
ap.add_argument("datasets", nargs="+", type=Path, help="one prediction folder per dataset, e.g. <predictions>/hqevfi")
ap.add_argument("--seq", nargs="+", help="score only these sequences and skip the summary (one Slurm task per sequence)")
ap.add_argument("--summary", action="store_true", help="only collect the scores already computed")
ap.add_argument("--run", default="s1-seed0", help="prediction subfolder of each sequence")
ap.add_argument("--v", type=float, nargs="+", default=[1.0], help="pixels per millisecond (METRICS.md default 1)")
ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
ap.add_argument("--chunk", type=int, default=1 << 20, help="query points per KD-tree call")
a = ap.parse_args()


def points(t, x, y, on, sign, t0, v):
    k = on == sign
    xyz = np.empty((int(k.sum()), 3))
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = x[k], y[k], (t[k] - t0) * (1000 * v)
    return xyz


def tree(xyz):
    return cKDTree(xyz, balanced_tree=False, compact_nodes=False)


def mean_nn(query, target):
    total = 0.0
    for i in range(0, len(query), a.chunk):
        total += float(target.query(query[i:i + a.chunk], k=1, workers=a.workers)[0].sum())
    return total / len(query)


def check(x, y, w, h, what):
    if len(x) and (x.min() < 0 or y.min() < 0 or x.max() >= w or y.max() >= h):
        raise ValueError(f"{what}: events outside the {w}x{h} frame")


def score(run):
    files = sorted(run.glob("*.npz"))
    inputs = {}
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            if "metadata" in z:  # baselines; evtween's generate.py stores only t, x, y, p
                inputs[f.stem] = (json.loads(str(z["metadata"]))["sequence"], z["frame_indices"], z["frame_ts"],
                                  int(z["width"]), int(z["height"]))
    if not inputs:
        raise ValueError(f"{run}: needs at least one baseline file for the GT path and the frame times")
    src, idx, ts, w, h = next(iter(inputs.values()))
    if any(s != src or not np.array_equal(i, idx) or (ww, hh) != (w, h) for s, i, _, ww, hh in inputs.values()):
        raise ValueError(f"{run}: the methods were run on different inputs")
    out = run / "chamfer.json"
    scores = json.loads(out.read_text()) if out.exists() else {}
    if all(f.stem in scores.get(f"v{v:g}", {}) for v in a.v for f in files):
        return
    t0, t1 = float(ts[0]), float(ts[-1])  # scoring interval [first, last] input frame time, as run.py keeps events
    ev = {k: np.load(Path(src) / f"ev_{k}.npy", mmap_mode="r") for k in "txyp"}
    i0, i1 = np.searchsorted(ev["t"], t0, "left"), np.searchsorted(ev["t"], t1, "right")
    gt = [np.asarray(ev[k][i0:i1]) for k in "txyp"]
    gt = [gt[0].astype(np.float64), gt[1], gt[2], gt[3] > 0]
    check(gt[1], gt[2], w, h, f"GT {src}")
    for v in a.v:
        key = f"v{v:g}"
        pending = [f for f in files if f.stem not in scores.get(key, {})]
        if not pending:
            continue
        g = {s: points(*gt, s, t0, v) for s in (True, False)}
        gtree = {s: tree(g[s]) if len(g[s]) else None for s in g}
        for f in pending:
            start = time.time()
            with np.load(f, allow_pickle=False) as z:
                t, x, y, p = (z[k] for k in "txyp")
            keep = (t >= t0) & (t <= t1)
            t, x, y, on = t[keep].astype(np.float64), x[keep], y[keep], p[keep] > 0
            check(x, y, w, h, str(f))
            r = dict(n_pred=len(t), n_gt=len(gt[0]), outside=int((~keep).sum()))
            for s, name in ((True, "on"), (False, "off")):
                q = points(t, x, y, on, s, t0, v)
                r[f"chamfer_{name}"] = (0.5 * (mean_nn(q, gtree[s]) + mean_nn(g[s], tree(q)))
                                        if len(q) and len(g[s]) else None)
            parts = [r[k] for k in ("chamfer_on", "chamfer_off") if r[k] is not None]
            r["chamfer"] = float(np.mean(parts)) if parts else None
            scores.setdefault(key, {})[f.stem] = r
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(scores, indent=1))
            os.replace(tmp, out)
            print(f"{run.parent.name} {f.stem} {key}: chamfer {r['chamfer']}, {r['n_pred']:,} vs {r['n_gt']:,} GT events, "
                  f"{time.time() - start:.0f} s", flush=True)


def summary(ds):
    rows = []
    for d in sorted(ds.iterdir()):
        if (d / a.run / "chamfer.json").is_file():
            for key, methods in json.loads((d / a.run / "chamfer.json").read_text()).items():
                rows += [dict(sequence=d.name, v=float(key[1:]), method=m, **r) for m, r in methods.items()]
    if not rows:
        print(f"{ds}: no scores")
        return
    fields = ["sequence", "v", "method", "chamfer", "chamfer_on", "chamfer_off", "n_pred", "n_gt", "outside"]
    with (ds / f"chamfer_{a.run}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    table = []
    for v in sorted({r["v"] for r in rows}):
        rv = [r for r in rows if r["v"] == v]
        methods, seqs = sorted({r["method"] for r in rv}), sorted({r["sequence"] for r in rv})
        done = {(r["sequence"], r["method"]): r for r in rv if r["chamfer"] is not None}
        common = [s for s in seqs if all((s, m) in done for m in methods)]  # same sequences for every method
        for m in methods:
            row = dict(v=v, method=m, sequences=len(common), of=len(seqs))
            for k in ("chamfer", "chamfer_on", "chamfer_off"):
                vals = np.array([done[s, m][k] for s in common if done[s, m][k] is not None], dtype=float)
                row[k], row[k + "_std"] = (vals.mean(), vals.std()) if len(vals) else (None, None)
            table.append(row)
    fields = ["v", "method", "sequences", "of", "chamfer", "chamfer_std", "chamfer_on", "chamfer_on_std",
              "chamfer_off", "chamfer_off_std"]
    with (ds / f"chamfer_{a.run}_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(table)
    print(f"\n{ds.name} ({a.run}), mean over sequences scored for every method, pixel-equivalent units")
    print(f"{'v px/ms':>8} {'method':<16} {'seqs':>7} {'chamfer':>16} {'ON':>9} {'OFF':>9}")
    for r in table:
        fmt = lambda k: "-" if r[k] is None else f"{r[k]:.3f}"
        print(f"{r['v']:>8g} {r['method']:<16} {r['sequences']:>3}/{r['of']:<3} {fmt('chamfer'):>9} ± {fmt('chamfer_std'):<5} "
              f"{fmt('chamfer_on'):>9} {fmt('chamfer_off'):>9}")


for ds in a.datasets:
    if not a.summary:
        for name in a.seq or sorted(d.name for d in ds.iterdir() if (d / a.run).is_dir()):
            score(ds / name / a.run)
    if not a.seq:
        summary(ds)
