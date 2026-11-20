import argparse
import csv
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

METRICS = ["count", "on", "iei", "fano", "pmse4", "chamfer"]
ap = argparse.ArgumentParser(description="Event-stream metrics of each predicted stream against the recorded events, "
                             "one table per dataset. Every value is an error: 0 is best.")
ap.add_argument("datasets", nargs="+", type=Path, help="one prediction folder per dataset, e.g. <predictions>/hqevfi")
ap.add_argument("--seq", nargs="+", help="score only these sequences, no summary (one Slurm task per sequence)")
ap.add_argument("--summary", action="store_true", help="only collect the scores already computed")
ap.add_argument("--run", default="s1-seed0", help="prediction subfolder of each sequence")
ap.add_argument("--metrics", nargs="+", default=METRICS, choices=METRICS)
ap.add_argument("--v", type=float, nargs="+", default=[1.0], help="Chamfer: pixels per ms")
ap.add_argument("--fano-bin-ms", type=float, nargs="+", default=[2.0])
ap.add_argument("--fano-window-ms", type=float, nargs="+", default=[50.0])
ap.add_argument("--fano-tile", type=int, nargs="+", default=[8])
ap.add_argument("--fano-cells", nargs="+", default=["both"], choices=["both", "union"],
                help="tile-windows scored: active in both streams, or in either (a missing side counts as Fano 0)")
ap.add_argument("--pmse-bin-ms", type=float, nargs="+", default=[5.0])
ap.add_argument("--pmse-norm", nargs="+", default=["none"], choices=["none", "count"],
                help="none: raw counts (V2CE); count: each grid divided by its stream's mean density")
ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
ap.add_argument("--chunk", type=int, default=1 << 20, help="Chamfer query points per KD-tree call")
a = ap.parse_args()

settings = [(m, m, {}) for m in ("count", "on", "iei") if m in a.metrics]
if "fano" in a.metrics:
    for b, w, tile, cells in itertools.product(a.fano_bin_ms, a.fano_window_ms, a.fano_tile, a.fano_cells):
        if not np.isclose(w / b, round(w / b)):
            ap.error(f"The Fano window ({w} ms) must hold a whole number of bins ({b} ms).")
        settings.append((f"fano bin={b:g}ms window={w:g}ms tile={tile} cells={cells}", "fano",
                         dict(bin=b / 1000, window=w / 1000, tile=tile, cells=cells)))
if "pmse4" in a.metrics:
    settings += [(f"pmse4 bin={b:g}ms norm={n}", "pmse4", dict(bin=b / 1000, norm=n))
                 for b, n in itertools.product(a.pmse_bin_ms, a.pmse_norm)]
if "chamfer" in a.metrics:
    settings += [(f"chamfer v={v:g}px/ms", "chamfer", dict(v=v)) for v in a.v]


def stream(t, x, y, p, t0, t1):
    # events in [t0, t1], times relative to t0, sorted by time
    keep = (t >= t0) & (t <= t1)
    s = dict(t=t[keep].astype(np.float64) - t0, x=x[keep].astype(np.int64), y=y[keep].astype(np.int64), on=p[keep] > 0)
    if np.any(s["t"][1:] < s["t"][:-1]):
        order = np.argsort(s["t"], kind="stable")
        s = {k: v[order] for k, v in s.items()}
    return s, int((~keep).sum())


def log_iei(s, w):
    # FracEvent Eq. S3: positive same-pixel intervals, both polarities pooled, log10 seconds
    pixel = s["y"] * w + s["x"]
    order = np.argsort(pixel, kind="stable")
    pixel, t = pixel[order], s["t"][order]
    d = np.diff(t)[pixel[1:] == pixel[:-1]]
    return np.log10(np.maximum(d[d > 0], 1e-6))


def fano_map(s, w, h, T, bin, window, tile):
    # variance / mean of the counts in the 2 ms bins of each 8x8 tile and 50 ms window; inactive cells are 0
    ns, nw = round(window / bin), max(1, int(np.ceil(T / window - 1e-12)))
    nx, ny = -(-w // tile), -(-h // tile)
    win = np.minimum((s["t"] // window).astype(np.int64), nw - 1)
    sub = np.clip(((s["t"] - win * window) // bin).astype(np.int64), 0, ns - 1)
    cell = (win * ny + s["y"] // tile) * nx + s["x"] // tile
    c = np.bincount(cell * ns + sub, minlength=nw * ny * nx * ns).reshape(-1, ns)
    total, squares = c.sum(1), (c * c).sum(1)
    return np.divide(squares, total, out=np.zeros(len(total)), where=total > 0) - total / ns, total > 0


def pmse4(P, G, w, h, T, bin, norm, k=4):
    # V2CE PMSE: ON/OFF voxel grids with forward linear time weights, mean of k x k x k blocks (valid, no padding)
    nb = int(np.ceil(T / bin - 1e-12))
    nt, nh, nw = nb // k, h // k, w // k
    if min(nt, nh, nw) < 1 or (norm == "count" and not (len(P["t"]) and len(G["t"]))):
        return None
    step, total = max(1, 25_000_000 // (2 * nh * nw)), 0.0
    for c0 in range(0, nt, step):
        c1 = min(nt, c0 + step)
        grids = []
        for s in (P, G):
            lo, hi = np.searchsorted(s["t"], [(c0 * k - 2) * bin, (c1 * k + 1) * bin])
            t, x, y, on = (s[q][lo:hi] for q in ("t", "x", "y", "on"))
            u = t / bin
            j = np.floor(u).astype(np.int64)
            grid = np.zeros(2 * (c1 - c0) * nh * nw)
            for jj, weight in ((j, 1 - (u - j)), (j + 1, u - j)):
                m = (jj >= c0 * k) & (jj < c1 * k) & (x < nw * k) & (y < nh * k)
                idx = (((~on[m]) * (c1 - c0) + jj[m] // k - c0) * nh + y[m] // k) * nw + x[m] // k
                grid += np.bincount(idx, weights=weight[m], minlength=grid.size)
            # count: in units of the stream's mean block value, so 1 = an error as large as the mean event density
            grids.append(grid / k ** 3 / (len(s["t"]) / (2 * nt * nh * nw * k ** 3) if norm == "count" else 1))
        total += float(np.square(grids[0] - grids[1]).sum())
    return total / (2 * nt * nh * nw)


def points(s, sign, v):
    m = s["on"] == sign
    xyz = np.empty((int(m.sum()), 3))
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = s["x"][m], s["y"][m], s["t"][m] * (1000 * v)
    return xyz


def tree(xyz):
    return cKDTree(xyz, balanced_tree=False, compact_nodes=False)


def mean_nn(query, target):
    total = 0.0
    for i in range(0, len(query), a.chunk):
        total += float(target.query(query[i:i + a.chunk], k=1, workers=a.workers)[0].sum())
    return total / len(query)


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
    out = run / "metrics.json"
    scores = json.loads(out.read_text()) if out.exists() else {}
    todo = {f.stem: [st for st in settings if f.stem not in scores.get(st[0], {})] for f in files}
    if not any(todo.values()):
        return
    t0, t1 = float(ts[0]), float(ts[-1])  # scoring interval [first, last] input frame time, as run.py keeps events
    T = t1 - t0
    ev = {k: np.load(Path(src) / f"ev_{k}.npy", mmap_mode="r") for k in "txyp"}
    i0, i1 = np.searchsorted(ev["t"], t0, "left"), np.searchsorted(ev["t"], t1, "right")
    G, _ = stream(*(np.asarray(ev[k][i0:i1]) for k in "txyp"), t0, t1)
    cache = {}

    def save(key, method, r, start):
        scores.setdefault(key, {})[method] = r
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(scores, indent=1))
        os.replace(tmp, out)
        print(f"{run.parent.name} {method} {key}: {r['value']} ({time.time() - start:.0f} s)", flush=True)

    def load(f):
        with np.load(f, allow_pickle=False) as z:
            P, outside = stream(*(z[k] for k in "txyp"), t0, t1)
        for c, n in ((P["x"], w), (P["y"], h), (G["x"], w), (G["y"], h)):
            if len(c) and (c.min() < 0 or c.max() >= n):
                raise ValueError(f"{f}: events outside the {w}x{h} frame")
        return P, outside

    for f in files:  # every metric except Chamfer: one pass per method
        pending = [st for st in todo[f.stem] if st[1] != "chamfer"]
        if not pending:
            continue
        P, outside = load(f)
        nP, nG = len(P["t"]), len(G["t"])
        for key, metric, q in pending:
            start = time.time()
            if metric == "count":
                r = dict(value=abs(np.log(nP / nG)) if nP and nG else None, ratio=nP / max(nG, 1),
                         n_pred=nP, n_gt=nG, outside=outside)
            elif metric == "on":
                on, gt_on = 100 * P["on"].sum() / max(nP, 1), 100 * G["on"].sum() / max(nG, 1)
                r = dict(value=float(abs(on - gt_on)), on_pct=float(on), gt_on_pct=float(gt_on))
            elif metric == "iei":
                if "iei" not in cache:
                    cache["iei"] = log_iei(G, w)
                lp = log_iei(P, w)
                r = dict(value=float(wasserstein_distance(lp, cache["iei"])) if len(lp) and len(cache["iei"]) else None)
            elif metric == "fano":
                q = {k: v for k, v in q.items() if k != "cells"}
                if str(q) not in cache:
                    cache[str(q)] = fano_map(G, w, h, T, **q)
                (fp, ap_), (fg, ag) = fano_map(P, w, h, T, **q), cache[str(q)]
                both = ap_ & ag
                cells = both if "cells=both" in key else ap_ | ag
                r = dict(value=float(np.abs(np.log((fp[cells] + 1e-3) / (fg[cells] + 1e-3))).mean()) if cells.any() else None,
                         shared_cells=int(both.sum()), pred_only_cells=int((ap_ & ~ag).sum()),
                         gt_only_cells=int((ag & ~ap_).sum()))
            elif metric == "pmse4":
                r = dict(value=pmse4(P, G, w, h, T, q["bin"], q["norm"]))
            save(key, f.stem, r, start)

    for key, metric, q in settings:  # Chamfer: GT trees built once per v
        pending = [f for f in files if (key, metric, q) in todo[f.stem]]
        if metric != "chamfer" or not pending:
            continue
        g = {s: points(G, s, q["v"]) for s in (True, False)}
        gtree = {s: tree(g[s]) if len(g[s]) else None for s in g}
        for f in pending:
            start = time.time()
            P, _ = load(f)
            r = {}
            for s, name in ((True, "on"), (False, "off")):
                p = points(P, s, q["v"])
                r[f"chamfer_{name}"] = (0.5 * (mean_nn(p, gtree[s]) + mean_nn(g[s], tree(p)))
                                        if len(p) and len(g[s]) else None)
            parts = [x for x in r.values() if x is not None]
            r = dict(value=float(np.mean(parts)) if parts else None, **r)
            save(key, f.stem, r, start)


def summary(ds):
    rows = []
    for d in sorted(ds.iterdir()):
        if (d / a.run / "metrics.json").is_file():
            for key, methods in json.loads((d / a.run / "metrics.json").read_text()).items():
                rows += [dict(sequence=d.name, metric=key, method=m, **r) for m, r in methods.items()]
    if not rows:
        print(f"{ds}: no scores")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (ds / f"metrics_{a.run}.csv").open("w", newline="") as f:
        out = csv.DictWriter(f, fields)
        out.writeheader()
        out.writerows(rows)
    table = []
    for key in dict.fromkeys(r["metric"] for r in rows):
        rk = [r for r in rows if r["metric"] == key]
        methods, seqs = sorted({r["method"] for r in rk}), sorted({r["sequence"] for r in rk})
        done = {(r["sequence"], r["method"]): r for r in rk if r["value"] is not None}
        common = [s for s in seqs if all((s, m) in done for m in methods)]  # same sequences for every method
        for m in methods:
            for field in [k for k in done.get((common[0], m), {}) if k not in ("sequence", "metric", "method")] if common else ["value"]:
                vals = np.array([done[s, m][field] for s in common if done[s, m][field] is not None], dtype=float)
                table.append(dict(metric=key, method=m, field=field, sequences=len(common), of=len(seqs),
                                  mean=vals.mean() if len(vals) else None, std=vals.std() if len(vals) else None))
    with (ds / f"metrics_{a.run}_summary.csv").open("w", newline="") as f:
        out = csv.DictWriter(f, ["metric", "method", "field", "sequences", "of", "mean", "std"])
        out.writeheader()
        out.writerows(table)
    print(f"\n{ds.name} ({a.run}): mean ± std over the sequences scored for every method; all errors, 0 is best")
    for key in dict.fromkeys(r["metric"] for r in table):
        print(f"\n{key}")
        for r in table:
            if r["metric"] == key and r["field"] == "value":
                value = "-" if r["mean"] is None else f"{r['mean']:.4g} ± {r['std']:.2g}"
                print(f"  {r['method']:<16} {value:<22} {r['sequences']}/{r['of']} sequences")


for ds in a.datasets:
    if not a.summary:
        for name in a.seq or sorted(d.name for d in ds.iterdir() if (d / a.run).is_dir()):
            score(ds / name / a.run)
    if not a.seq:
        summary(ds)
