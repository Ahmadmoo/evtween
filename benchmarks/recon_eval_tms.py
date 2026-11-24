"""Frame reconstruction with fixed-duration windows ('tms'): a downstream test of event timing.

Same pretrained E2VID and FireNet, model loading, padding, output normalisation and metrics as
recon_eval.py (imported from it), but the event stream is cut into back-to-back windows of fixed
length instead of windows between frames:

  * --bin-ms D [D ...]: each run uses windows of 5 x D ms (5 = the networks' input bins, fixed by the
    pretrained weights). Windows start at the first frame (t = 0) for every event source, so window i
    covers the same [i*5D, (i+1)*5D) ms for real events and every method; each event is used once.
    The tail shorter than one window is not used.
  * --bin-time ms|events (both by default):
      ms      bins have fixed edges: bin b covers [b*D, (b+1)*D) ms of the window; each event is split
              bilinearly between the two nearest bin centres (b+0.5)*D, events in the outer half-bins go
              to the edge bin. Timing errors of a simulator therefore move events between bins.
      events  EVREAL's fixed-duration mode: time inside the window is scaled by the window's first and
              last event (as recon_eval.py does between frames).
    Voxel normalisation (non-zero mean 0 / std 1) is the same in both.

Metrics (windows ending at or after --warmup-s seconds)
  agreement_mse/ssim/lpips  reconstruction from a method's events vs reconstruction from the real events,
                            same window and network. Main result; needs no ground-truth frame.
  gt_mse/ssim/lpips         only for windows whose end lies within --gt-tol-ms of a ground-truth frame
                            (EVREAL uses 1 ms); the matched frame and the offset are recorded.

Saved windows (--save)
  hybrid (default): 40 windows per sequence and configuration, the same indices for every method and both
    networks: 10 with the best and 10 with the worst agreement with the real reconstruction (mean over
    methods and networks; LPIPS, or SSIM with --skip-lpips), 10 with the most real events (activity), and
    10 spread evenly over the rest, without duplicates; plus one contact sheet per window.
  samples: --samples evenly spaced windows; none.

Outputs
  data_test_recon_tms/<DATASET>/<SPLIT>/<config>/<network>/<method>/<sequence>/w{i:06d}.png
  reports/recon_tms/<DATASET>/<SPLIT>/<timestamp>/
      config.json, summary_all.csv, comparison.csv, wilcoxon_all.csv, selection.csv
      <config>/ per_window/, sanity/, selection/, contact_sheets/, per_window.csv, per_sequence.csv,
                summary.csv, sanity.csv, wilcoxon.csv
  where <config> is e.g. bin5ms_ms (D = 5 ms, fixed-ms bins) or bin5ms_events.

Example (one A100, 32 CPUs):
  python benchmarks/recon_eval_tms.py eds --bin-ms 2 5 10 20 --devices cuda:0 cuda:0
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image

if __package__:
    from . import recon_eval as r
else:
    import recon_eval as r

torch, F = r.torch, r.F
ROOT, REAL, METRICS = r.ROOT, r.REAL, r.METRICS
BIN_TIMES = ("ms", "events")
VALUE_COLUMNS = tuple(f"{kind}_{m}" for kind in ("agreement", "gt") for m in METRICS)
WINDOW_COLUMNS = ("dataset", "split", "sequence", "config", "bin_ms", "bin_time", "network", "method", "window",
                  "t_start_s", "t_end_s", "gt_frame", "gt_offset_ms", *VALUE_COLUMNS)
SANITY_COLUMNS = ("dataset", "split", "sequence", "config", "method", "status", "events", "windows",
                  "empty_windows", "mean_events_per_window")
SELECTION_COLUMNS = ("dataset", "split", "sequence", "config", "window", "t_start_s", "t_end_s", "category",
                     "mean_agreement", "agreement_metric", "real_events", "gt_frame", "gt_offset_ms")
CATEGORY_ORDER = ("best_agreement", "worst_agreement", "high_activity", "temporal")


# ---------------------------------------------------------------- configurations and windows (no torch)

def config_name(bin_ms, bin_time):
    return f"bin{bin_ms:g}ms_{bin_time}"


def configurations(args):
    return [(float(d), bt) for d in args.bin_ms for bt in args.bin_time]


def window_edges(duration_s, window_s):
    """Back-to-back windows [i*T, (i+1)*T) from t = 0; only whole windows inside [0, duration)."""
    n = int(math.floor(duration_s / window_s + 1e-9))
    return np.arange(n + 1, dtype=np.float64) * window_s


def match_frames(window_ends, frame_ts, tol_ms):
    """Nearest ground-truth frame for each window end; -1 where it is further than tol_ms."""
    frame_ts = np.asarray(frame_ts, np.float64)
    pos = np.clip(np.searchsorted(frame_ts, window_ends), 1, len(frame_ts) - 1)
    left, right = frame_ts[pos - 1], frame_ts[pos]
    nearest = np.where(np.abs(window_ends - left) <= np.abs(right - window_ends), pos - 1, pos)
    offset_ms = (frame_ts[nearest] - window_ends) * 1000.0
    matched = np.where(np.abs(offset_ms) <= tol_ms + 1e-9, nearest, -1)  # 1e-9 ms: float rounding
    return matched, offset_ms, nearest


def voxel_ms_numpy(t, x, y, p, start, bin_s, bins, height, width):
    """Fixed-ms voxel grid of one window (before normalisation): bin centres at start + (b + 0.5) * bin_s,
    bilinear between neighbouring centres, clamped to the edge bins."""
    grid = np.zeros((bins, height, width), np.float64)
    if t.size == 0:
        return grid
    tn = np.clip((t - start) / bin_s - 0.5, 0, bins - 1)
    for b in range(bins):
        np.add.at(grid[b], (y, x), p * np.maximum(0.0, 1.0 - np.abs(tn - b)))
    return grid


def windows_for(data, bin_ms, args):
    window_s = args.bins * bin_ms / 1000.0
    edges = window_edges(float(data.edges[-1]), window_s)
    return window_s, edges


def recon_base(args):
    return Path(args.recon_root) if args.recon_root else ROOT / "data_test_recon_tms" / args.dataset / args.split


def recon_dir(args, cfg, network, method, name):
    return recon_base(args) / cfg / network / method / name


def sample_windows(n_windows, first, count):
    if count <= 0 or first > n_windows - 1:
        return set()
    return {int(round(v)) for v in np.linspace(first, n_windows - 1, min(count, n_windows - first))}


# ---------------------------------------------------------------- GPU parts

def _normalize(flat):
    """EVREAL normalize_event_tensor per row (non-zero entries to mean 0, population std 1)."""
    nz = flat != 0
    n = nz.sum(1).clamp(min=1)
    mean = flat.sum(1) / n
    std = ((flat * flat).sum(1) / n - mean * mean).clamp(min=0).sqrt().clamp(min=1e-6)
    return torch.where(nz, (flat - mean[:, None]) / std[:, None], torch.zeros_like(flat))


def voxelize_ms(ev, edges, j0, j1, bins, bin_s, height, width, device):
    """Normalised fixed-ms voxel grids of windows j0..j1-1: (j1-j0, bins, H, W). Matches voxel_ms_numpy."""
    count = j1 - j0
    out = torch.zeros(count * bins * height * width, dtype=torch.float32, device=device)
    cuts = np.asarray(ev["cuts"][j0:j1 + 1], np.int64)
    a, b = int(cuts[0]), int(cuts[-1])
    if b > a:
        local = torch.from_numpy(cuts - a).to(device)
        t = torch.from_numpy(ev["t"][a:b]).to(device, non_blocking=True)
        x = torch.from_numpy(ev["x"][a:b]).to(device, non_blocking=True)
        y = torch.from_numpy(ev["y"][a:b]).to(device, non_blocking=True)
        p = torch.from_numpy(ev["p"][a:b]).to(device, non_blocking=True).float()
        w = torch.searchsorted(local[1:], torch.arange(b - a, device=device), right=True)
        start = torch.from_numpy(np.ascontiguousarray(edges[j0:j1])).to(device)[w]
        tn = ((t - start) / bin_s - 0.5).clamp(0, bins - 1)
        lo = tn.floor().long().clamp_(0, bins - 1)
        frac = (tn - lo).float()
        pix = y * width + x
        base = w * bins
        out.index_add_(0, (base + lo) * (height * width) + pix, p * (1 - frac))
        if bins > 1:
            hi = lo + 1
            ok = hi < bins
            out.index_add_(0, ((base + hi) * (height * width) + pix)[ok], (p * frac)[ok])
    return _normalize(out.view(count, -1)).view(count, bins, height, width)


def voxelize_window(ev, edges, j0, j1, bins, bin_s, bin_time, height, width, device):
    if bin_time == "ms":
        return voxelize_ms(ev, edges, j0, j1, bins, bin_s, height, width, device)
    return r.voxelize(ev, j0, j1, bins, height, width, device)  # EVREAL scaling by first/last event


@r._inference
def process_config(data, bin_ms, bin_time, networks, scorer, args, run_dir, io_pool, tag="", score=True,
                   samples=None, categories=None):
    """One sequence, one configuration. score=True writes <config>/per_window/ and sanity/ CSVs; samples are
    window indices to save. score=False replays up to the last sample and only saves (same computation)."""
    cfg = config_name(bin_ms, bin_time)
    out_dir = run_dir / cfg
    device = scorer.device
    names = [n for n in (REAL, *args._methods) if n in data.events]
    has_real = REAL in data.events
    h, w = data.meta["height"], data.meta["width"]
    window_s, edges = windows_for(data, bin_ms, args)
    bin_s = bin_ms / 1000.0
    n_win = len(edges) - 1
    if n_win <= 0:
        print(f"{tag} {data.name} {cfg}: sequence shorter than one {window_s * 1000:g} ms window; skipped", flush=True)
        if score:
            r._write_csv(out_dir / "per_window" / f"{data.name}.csv", WINDOW_COLUMNS, [])
        return 0
    views = {n: dict(t=data.events[n]["t"], x=data.events[n]["x"], y=data.events[n]["y"], p=data.events[n]["p"],
                     cuts=np.searchsorted(data.events[n]["t"], edges)) for n in names}
    first = int(math.ceil(args.warmup_s / window_s - 1e-9)) - 1 if args.warmup_s > 0 else 0
    first = max(first, 0)  # first scored window: the one ending at or after warmup_s
    matched, offset_ms, _ = match_frames(edges[1:], data.edges, args.gt_tol_ms)
    if samples is None:
        samples = sample_windows(n_win, first, args.samples) if args.save == "samples" else set()
    samples = {int(s) for s in samples}
    stop = n_win if score else min(n_win, max(samples, default=-1) + 1)
    for net in networks:
        net.reset()
    rows, futures, frame_jobs = [], [], {}

    def request(j0, j1):
        if not score:
            return
        for j in range(j0, j1):
            k = int(matched[j])
            if j >= first and k >= 0 and k not in frame_jobs:
                frame_jobs[k] = io_pool.submit(r.load_gray, data.frames[k], w, h)

    request(0, min(stop, args.chunk))
    started = last_report = time.monotonic()
    for j0 in range(0, stop, args.chunk):
        now = time.monotonic()
        if j0 and now - last_report >= args.progress_every:
            rate = j0 / (now - started)
            print(f"{tag} {data.name} {cfg}: {'window' if score else 'replay, window'} {j0}/{stop} "
                  f"({100 * j0 / stop:.0f}%), {rate:.1f} windows/s, ETA {r._clock((stop - j0) / rate)}", flush=True)
            last_report = now
        j1 = min(stop, j0 + args.chunk)
        request(j1, min(stop, j1 + args.chunk))
        vox = torch.stack([voxelize_window(views[n], edges, j0, j1, args.bins, bin_s, bin_time, h, w, device)
                           for n in names])
        pending = []
        for c in range(j1 - j0):
            j = j0 + c
            for net in networks:
                img = r.postprocess(net.step(vox[:, c]), net.name, args.norm)
                if score and j >= first:
                    pending.append((j, net.name, img))
                if j in samples:
                    arr = img.cpu().numpy()
                    futures += [io_pool.submit(r._save_png, recon_dir(args, cfg, net.name, m, data.name) / f"w{j:06d}.png",
                                               arr[s, 0]) for s, m in enumerate(names)]
        del vox
        if not pending:
            continue
        preds, refs, tags = [], [], []
        for j, net_name, img in pending:
            if has_real and len(names) > 1:
                preds.append(img[1:])
                refs.append(img[:1].expand(len(names) - 1, -1, -1, -1))
                tags += [(j, net_name, n, "agreement") for n in names[1:]]
            k = int(matched[j])
            if k >= 0:
                gt = torch.from_numpy(frame_jobs[k].result()).to(device)[None, None]
                preds.append(img)
                refs.append(gt.expand(len(names), -1, -1, -1))
                tags += [(j, net_name, n, "gt") for n in names]
        table = {}
        for j, net_name, _ in pending:
            k = int(matched[j])
            for n in names:
                table[(net_name, n, j)] = dict(
                    dataset=args.dataset, split=args.split, sequence=data.name, config=cfg, bin_ms=bin_ms,
                    bin_time=bin_time, network=net_name, method=n, window=j, t_start_s=float(edges[j]),
                    t_end_s=float(edges[j + 1]), gt_frame=k if k >= 0 else "",
                    gt_offset_ms=float(offset_ms[j]) if k >= 0 else "", **{c: "" for c in VALUE_COLUMNS})
        if preds:
            scores = scorer(torch.cat(preds), torch.cat(refs))
            scores = {m: v.cpu().numpy() for m, v in scores.items()}
            for i, (j, net_name, method, kind) in enumerate(tags):
                for m in METRICS:
                    table[(net_name, method, j)][f"{kind}_{m}"] = float(scores[m][i])
        rows += table.values()
        for k in {int(matched[j]) for j, _, _ in pending if matched[j] >= 0}:
            if not any(int(matched[jj]) == k for jj in range(j1, min(stop, j1 + args.chunk))):
                frame_jobs.pop(k, None)
    for fut in frame_jobs.values():
        fut.cancel()
    for fut in futures:
        fut.result()
    saved = sorted(s for s in samples if s < stop)
    if saved:
        for net in networks:
            for m in names:
                meta = dict(dataset=args.dataset, split=args.split, sequence=data.name, config=cfg, network=net.name,
                            method=m, source=str(data.source), bin_ms=bin_ms, bin_time=bin_time, bins=args.bins,
                            window_ms=window_s * 1000, windows=saved,
                            t_start_s={str(s): float(edges[s]) for s in saved}, run=str(run_dir))
                if categories:
                    meta["categories"] = {str(s): categories[s] for s in saved if s in categories}
                (recon_dir(args, cfg, net.name, m, data.name) / "recon.json").write_text(json.dumps(meta, indent=2))
    if not score:
        return 0
    rows.sort(key=lambda row: (row["network"], names.index(row["method"]), row["window"]))
    key = "agreement_lpips" if not args.skip_lpips else "agreement_ssim"
    for net in networks:
        means = []
        for m in names[1:] if has_real else []:
            vals = [row[key] for row in rows if row["network"] == net.name and row["method"] == m and row[key] != ""]
            if vals:
                means.append(f"{m} {np.mean(vals):.3f}")
        if means:
            print(f"{tag} {data.name} {cfg}: {net.name} mean {key}: " + " | ".join(means), flush=True)
    r._write_csv(out_dir / "per_window" / f"{data.name}.csv", WINDOW_COLUMNS, rows)
    sanity = []
    for n in names:
        counts = np.diff(views[n]["cuts"])
        sanity.append(dict(dataset=args.dataset, split=args.split, sequence=data.name, config=cfg, method=n,
                           status="ok", events=int(counts.sum()), windows=int(counts.size),
                           empty_windows=int((counts == 0).sum()), mean_events_per_window=float(counts.mean())))
    sanity += [dict(dataset=args.dataset, split=args.split, sequence=data.name, config=cfg, method=n, status="missing")
               for n in data.missing]
    r._write_csv(out_dir / "sanity" / f"{data.name}.csv", SANITY_COLUMNS, sanity)
    n_gt = int(sum(1 for j in range(first, n_win) if matched[j] >= 0))
    print(f"{tag} {data.name} {cfg}: {n_win - first} scored windows of {window_s * 1000:g} ms, "
          f"{n_gt} within {args.gt_tol_ms:g} ms of a GT frame", flush=True)
    return len(rows)


# ---------------------------------------------------------------- hybrid selection and contact sheets (no torch)

def selection_counts(args):
    return dict(best_agreement=args.select_best, worst_agreement=args.select_worst,
                high_activity=args.select_activity, temporal=args.select_temporal)


def select_windows(windows, activity, counts, total, metric="agreement_lpips"):
    """One shared set of window indices for every method and network of a sequence/configuration.

    windows: per_window rows (pandas). activity: {window: real events}. Categories are filled in
    CATEGORY_ORDER without duplicates: best and worst mean agreement with the real reconstruction (over
    methods and networks; LPIPS lower = better, SSIM higher = better), most real events, then windows spread
    evenly over the rest; 'temporal' gets at least its count and tops up to `total`.
    Returns ({window: category}, mean agreement per window)."""
    import pandas as pd
    higher_better = r.HIGHER_IS_BETTER[metric.split("_", 1)[1]]
    sim = windows[(windows["method"] != REAL) & windows[metric].notna()]
    score = sim.groupby("window")[metric].mean().sort_index()
    if score.empty:  # no agreement (no real events): fall back to all scored windows
        score = pd.Series(np.nan, index=sorted(windows["window"].unique()))
    candidates = [int(s) for s in score.index]
    chosen = {}

    def take(order, n, category):
        for s in order:
            if n <= 0:
                break
            if int(s) not in chosen:
                chosen[int(s)] = category
                n -= 1

    valid = score.dropna()
    take(valid.sort_values(ascending=not higher_better, kind="stable").index, counts["best_agreement"], "best_agreement")
    take(valid.sort_values(ascending=higher_better, kind="stable").index, counts["worst_agreement"], "worst_agreement")
    act = pd.Series(activity, dtype=float).reindex(candidates).dropna()
    take(act.sort_values(ascending=False, kind="stable").index, counts["high_activity"], "high_activity")
    remaining = [s for s in candidates if s not in chosen]
    n = min(len(remaining), max(counts["temporal"], total - len(chosen)))
    if n > 0:
        for i in np.unique(np.round(np.linspace(0, len(remaining) - 1, n)).astype(int)):
            chosen[remaining[i]] = "temporal"
    return dict(sorted(chosen.items())), score


def contact_sheets_tms(run_dir, args, cfg, name, selection, windows, frame_paths, frame_ts, networks, methods,
                       mean_agreement, activity, metric):
    """One sheet per selected window: rows = networks; columns = nearest GT frame, real, every method."""
    out = []
    windows = windows.set_index(["network", "method", "window"])
    for s, category in selection.items():
        sub = windows.xs(s, level="window") if s in windows.index.get_level_values("window") else None
        t_end = float(sub["t_end_s"].iloc[0]) if sub is not None else float("nan")
        k = int(np.clip(np.searchsorted(frame_ts, t_end), 0, len(frame_ts) - 1))
        if k > 0 and abs(frame_ts[k - 1] - t_end) < abs(frame_ts[k] - t_end):
            k -= 1
        gt = Image.open(frame_paths[k]).convert("L")
        rows = []
        for net in networks:
            tiles = [(["nearest GT frame", f"frame {k}", f"offset {(frame_ts[k] - t_end) * 1000:+.1f} ms"], gt)]
            for m in methods:
                png = recon_dir(args, cfg, net, m, name) / f"w{s:06d}.png"
                if not png.is_file():
                    continue
                caption = [m]
                if (net, m, s) in windows.index:
                    row = windows.loc[(net, m, s)]
                    if m != REAL and r.pd_notna(row.get("agreement_lpips")):
                        caption.append(f"vs real LPIPS {row['agreement_lpips']:.3f}")
                    if m != REAL and r.pd_notna(row.get("agreement_ssim")):
                        caption.append(f"vs real SSIM {row['agreement_ssim']:.3f}")
                    if r.pd_notna(row.get("gt_mse")):
                        caption.append(f"vs GT PSNR {float(r.psnr_from_mse(row['gt_mse'])):.2f}")
                tiles.append((caption, Image.open(png)))
            rows.append((net, tiles))
        title = (f"{args.dataset} / {name} / {cfg} / window {s} ({category})   ends {t_end:.4f} s   "
                 f"mean {metric} {mean_agreement.get(s, float('nan')):.3f}   real events {activity.get(s, 0)}")
        out.append(r.contact_sheet(run_dir / cfg / "contact_sheets" / name / f"w{s:06d}_{category}.png", title, rows,
                                   args.sheet_tile_width))
    return out


def render_selection(data, bin_ms, bin_time, networks, scorer, args, run_dir, io_pool, tag=""):
    import pandas as pd
    cfg = config_name(bin_ms, bin_time)
    windows = pd.read_csv(run_dir / cfg / "per_window" / f"{data.name}.csv")
    if windows.empty:
        r._write_csv(run_dir / cfg / "selection" / f"{data.name}.csv", SELECTION_COLUMNS, [])
        return {}
    metric = "agreement_lpips" if windows["agreement_lpips"].notna().any() else "agreement_ssim"
    _, edges = windows_for(data, bin_ms, args)
    activity = {}
    if REAL in data.events:
        counts = np.diff(np.searchsorted(data.events[REAL]["t"], edges))
        activity = {int(s): int(counts[s]) for s in windows["window"].unique()}
    selection, mean_agreement = select_windows(windows, activity, selection_counts(args), args.select_total, metric)
    process_config(data, bin_ms, bin_time, networks, scorer, args, run_dir, io_pool, tag, score=False,
                   samples=set(selection), categories=selection)
    methods = [n for n in (REAL, *args._methods) if n in data.events]
    contact_sheets_tms(run_dir, args, cfg, data.name, selection, windows, data.frames, data.edges,
                       [n.name for n in networks], methods, mean_agreement, activity, metric)
    first = windows.drop_duplicates("window").set_index("window")
    rows = [dict(dataset=args.dataset, split=args.split, sequence=data.name, config=cfg, window=s,
                 t_start_s=float(edges[s]), t_end_s=float(edges[s + 1]), category=c,
                 mean_agreement=float(mean_agreement.get(s, float("nan"))), agreement_metric=metric,
                 real_events=activity.get(s, ""), gt_frame=first.loc[s, "gt_frame"] if s in first.index else "",
                 gt_offset_ms=first.loc[s, "gt_offset_ms"] if s in first.index else "")
            for s, c in selection.items()]
    r._write_csv(run_dir / cfg / "selection" / f"{data.name}.csv", SELECTION_COLUMNS, rows)  # last: marks done
    per_cat = ", ".join(f"{c} {sum(v == c for v in selection.values())}" for c in CATEGORY_ORDER)
    print(f"{tag} {data.name} {cfg}: saved {len(selection)} windows ({per_cat}) -> {cfg}/contact_sheets/{data.name}/",
          flush=True)
    return selection


# ---------------------------------------------------------------- workers

def _done(run_dir, cfg, name, args):
    ok = (run_dir / cfg / "per_window" / f"{name}.csv").is_file()
    if args.save == "hybrid":
        ok = ok and (run_dir / cfg / "selection" / f"{name}.csv").is_file()
    return ok


def worker(device, names, args, run_dir, worker_id):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_num_threads(max(1, args.io_workers // 2))
    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    networks = [r.Network(n, r.build_model(n, r.checkpoint_path(n, args)[0], args.evreal_root), device, args.amp)
                for n in args.networks]
    scorer = r.Scorer(device, use_lpips=not args.skip_lpips, batch=args.metric_batch)
    done, errors = 0, []
    tag = f"[w{worker_id} {device}]"
    with ThreadPoolExecutor(args.io_workers) as io_pool, ThreadPoolExecutor(1) as prefetch:
        loads = {}
        for i, name in enumerate(names):
            for j in range(i, min(len(names), i + args.prefetch + 1)):
                if j not in loads:
                    loads[j] = prefetch.submit(r.load_sequence, names[j], args, args._methods, io_pool)
            start = time.monotonic()
            try:
                data = loads.pop(i).result()
                counts = ", ".join(f"{k} {v['events'] / 1e6:.2f}M" for k, v in data.stats.items())
                print(f"{tag} {name}: loaded {data.meta['width']}x{data.meta['height']}, "
                      f"{float(data.edges[-1]):.2f} s in {time.monotonic() - start:.1f}s; events: {counts}"
                      + (f"; missing: {', '.join(data.missing)}" if data.missing else ""), flush=True)
                if REAL not in data.events:
                    print(f"{tag} WARNING {name}: no real events in {data.source}; agreement metrics will be empty",
                          flush=True)
                for bin_ms, bin_time in configurations(args):
                    cfg = config_name(bin_ms, bin_time)
                    if _done(run_dir, cfg, name, args):
                        continue
                    if not (run_dir / cfg / "per_window" / f"{name}.csv").is_file():
                        process_config(data, bin_ms, bin_time, networks, scorer, args, run_dir, io_pool, tag)
                    if args.save == "hybrid":
                        render_selection(data, bin_ms, bin_time, networks, scorer, args, run_dir, io_pool, tag)
                del data
                done += 1
                print(f"{tag} {name}: DONE in {r._clock(time.monotonic() - start)}", flush=True)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                print(f"{tag} ERROR {name}: {exc}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
    return done, errors


def _worker_entry(device, names, args, run_dir, worker_id):
    return worker(device, names, args, run_dir, worker_id)


def run(args):
    if torch is None:
        raise ImportError("PyTorch is required for reconstruction")
    if not getattr(args, "io_workers", None):
        args.io_workers = max(2, (os.cpu_count() or 4) // len(args.devices))
    methods, names = r.discover(args)
    args._methods = methods
    run_dir = Path(args.run_dir) if args.run_dir else \
        ROOT / "reports" / "recon_tms" / args.dataset / args.split / datetime.now().strftime("%Y%m%d-%H%M%S")
    cfgs = [config_name(d, bt) for d, bt in configurations(args)]
    for cfg in cfgs:
        for sub in ("per_window", "sanity", "selection"):
            (run_dir / cfg / sub).mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not all(_done(run_dir, cfg, n, args) for cfg in cfgs)]
    commit, ckpts = r.setup(args)
    config = dict(
        dataset=args.dataset, split=args.split, events_root=str(r.events_root(args)), methods=methods,
        excluded={m: r.EXCLUDED[m] for m in r.EXCLUDED if (r.events_root(args) / m).is_dir()},
        sequences=names, networks=list(args.networks), configs=cfgs, bin_ms=[float(d) for d in args.bin_ms],
        bin_time=list(args.bin_time), bins=args.bins,
        windows="back-to-back windows of bins x bin_ms from t = 0 (first frame), same for every source",
        bin_time_rules={"ms": "fixed bin edges every bin_ms; bilinear between bin centres; edge bins clamped",
                        "events": "EVREAL fixed-duration mode: time scaled by the window's first and last event"},
        warmup_s=args.warmup_s, gt_tol_ms=args.gt_tol_ms, norm=args.norm, amp=args.amp, tf32=args.tf32,
        lpips=not args.skip_lpips, save=args.save, samples=args.samples, recon_root=str(recon_base(args)),
        selection=dict(selection_counts(args), total=args.select_total) if args.save == "hybrid" else None,
        evreal=dict(root=args.evreal_root, commit=commit, pinned=r.EVREAL_COMMIT),
        checkpoints={n: dict(path=str(p), sha256=sha, verified_against_pin=ok) for n, (p, sha, ok) in ckpts.items()},
        torch=str(torch.__version__), devices=list(args.devices), code_sha256=r.file_sha256(Path(__file__)))
    path = run_dir / "config.json"
    if path.is_file():
        old = json.loads(path.read_text())
        keep = ("methods", "networks", "bins", "warmup_s", "gt_tol_ms", "norm", "amp", "lpips")
        changed = [k for k in keep if old.get(k) != config[k]]
        if {n: c["sha256"] for n, c in old.get("checkpoints", {}).items()} != \
                {n: c["sha256"] for n, c in config["checkpoints"].items()}:
            changed.append("checkpoints")
        if changed:
            raise ValueError(f"Resuming {run_dir} with different settings: {changed}")
        config["configs"] = sorted(set(old.get("configs", [])) | set(cfgs))
    path.write_text(json.dumps(config, indent=2))
    print(f"{len(methods)} methods, {len(names)} sequences, configs {', '.join(cfgs)} "
          f"({len(names) - len(todo)} sequences already done) -> {run_dir}", flush=True)
    errors = []
    if todo:
        devices = list(args.devices)
        if len(devices) == 1:
            _, errors = worker(devices[0], todo, args, run_dir, 0)
        else:
            shares = r.balance(todo, args, methods, len(devices))
            with ProcessPoolExecutor(len(devices), mp_context=mp.get_context("spawn")) as pool:
                futures = [pool.submit(_worker_entry, d, s, args, run_dir, i)
                           for i, (d, s) in enumerate(zip(devices, shares)) if s]
                for fut in futures:
                    errors += fut.result()[1]
    summarize(run_dir)
    print(f"Summary: {run_dir / 'summary_all.csv'}; across bin lengths: {run_dir / 'comparison.csv'}", flush=True)
    if errors:
        raise RuntimeError("Some sequences failed (rerun with --run-dir to resume):\n" + "\n".join(errors))
    return run_dir


# ---------------------------------------------------------------- summary (no torch)

def summarize_config(cfg_dir):
    import pandas as pd
    files = sorted((cfg_dir / "per_window").glob("*.csv"))
    frames = [pd.read_csv(f) for f in files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return None
    windows = pd.concat(frames, ignore_index=True)
    windows.to_csv(cfg_dir / "per_window.csv", index=False)
    value_cols = [c for c in VALUE_COLUMNS if c in windows.columns]
    per_seq = (windows.groupby(["dataset", "split", "config", "bin_ms", "bin_time", "network", "method", "sequence"],
                               sort=False)[value_cols].mean().reset_index())
    per_seq.to_csv(cfg_dir / "per_sequence.csv", index=False)
    order = {m: i for i, m in enumerate((REAL, *r.METHODS))}
    rows = []
    for (net, method), g in per_seq.groupby(["network", "method"], sort=False):
        w = windows[(windows["network"] == net) & (windows["method"] == method)]
        row = dict(config=g["config"].iloc[0], bin_ms=g["bin_ms"].iloc[0], bin_time=g["bin_time"].iloc[0],
                   network=net, method=method, sequences=len(g), windows=len(w),
                   gt_windows=int(w["gt_mse"].notna().sum()))
        for c in value_cols:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std(ddof=1) if g[c].notna().sum() > 1 else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["_o"] = summary["method"].map(order)
    summary = summary.sort_values(["network", "_o"]).drop(columns="_o").reset_index(drop=True)
    for c in value_cols:
        summary[f"{c}_rank"] = (summary.groupby("network")[f"{c}_mean"]
                                .rank(ascending=not r.HIGHER_IS_BETTER[c.split("_", 1)[1]], method="min"))
    summary.to_csv(cfg_dir / "summary.csv", index=False)
    sanity = [pd.read_csv(f) for f in sorted((cfg_dir / "sanity").glob("*.csv"))]
    if sanity:
        pd.concat(sanity, ignore_index=True).to_csv(cfg_dir / "sanity.csv", index=False)
    wx = r.wilcoxon(per_seq, summary)
    if not wx.empty:
        wx.insert(0, "config", summary["config"].iloc[0])
    wx.to_csv(cfg_dir / "wilcoxon.csv", index=False)
    return summary, wx


def summarize(run_dir):
    import pandas as pd
    run_dir = Path(run_dir)
    summaries, tests, selections = [], [], []
    for cfg_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / "per_window").is_dir()):
        out = summarize_config(cfg_dir)
        if out is not None:
            summaries.append(out[0])
            tests.append(out[1])
        sel = [pd.read_csv(f) for f in sorted((cfg_dir / "selection").glob("*.csv"))]
        selections += [s for s in sel if not s.empty]
    if not summaries:
        return None
    summary = pd.concat(summaries, ignore_index=True).sort_values(["bin_time", "bin_ms", "network"], kind="stable")
    summary.to_csv(run_dir / "summary_all.csv", index=False)
    pd.concat(tests, ignore_index=True).to_csv(run_dir / "wilcoxon_all.csv", index=False)
    if selections:
        pd.concat(selections, ignore_index=True).to_csv(run_dir / "selection.csv", index=False)
    # how each method holds up as bins get shorter: one row per (network, bin_time, method), one column per bin_ms
    cols = [c for c in ("agreement_lpips_mean", "agreement_ssim_mean", "agreement_mse_mean")
            if c in summary and summary[c].notna().any()]
    parts = []
    for c in cols:
        piv = summary[summary["method"] != REAL].pivot_table(index=["network", "bin_time", "method"],
                                                             columns="bin_ms", values=c)
        piv.columns = [f"{c.replace('_mean', '')}@{b:g}ms" for b in piv.columns]
        parts.append(piv)
    if parts:
        pd.concat(parts, axis=1).reset_index().to_csv(run_dir / "comparison.csv", index=False)
    return summary


# ---------------------------------------------------------------- CLI

def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="Folder name under data_test_evt/ (e.g. eds, hqevfi, bsergb)")
    p.add_argument("--split", default="test")
    p.add_argument("--bin-ms", nargs="+", type=float, default=[2, 5, 10, 20],
                   help="Bin lengths in ms; each gives windows of --bins x bin_ms")
    p.add_argument("--bin-time", nargs="+", choices=BIN_TIMES, default=list(BIN_TIMES),
                   help="ms: fixed bin edges; events: EVREAL fixed-duration scaling (first/last event)")
    p.add_argument("--warmup-s", type=float, default=0.5, help="Windows ending before this time are not scored")
    p.add_argument("--gt-tol-ms", type=float, default=1.0,
                   help="Score against a GT frame only if it is within this many ms of the window end (EVREAL: 1)")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--sequences", nargs="+")
    p.add_argument("--networks", nargs="+", choices=r.NETWORKS, default=list(r.NETWORKS))
    p.add_argument("--events-root"), p.add_argument("--data-root"), p.add_argument("--recon-root")
    p.add_argument("--run-dir", help="Resume a report folder; finished sequence/configuration pairs are skipped")
    p.add_argument("--evreal-root")
    p.add_argument("--no-auto-setup", dest="auto_setup", action="store_false")
    p.add_argument("--e2vid-ckpt"), p.add_argument("--firenet-ckpt")
    p.add_argument("--bins", type=int, default=5, help="Network input bins (5 for the pretrained weights)")
    p.add_argument("--norm", choices=("evreal", "minmax", "none"), default="evreal")
    p.add_argument("--save", choices=("hybrid", "samples", "none"), default="hybrid")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--select-best", type=int, default=10)
    p.add_argument("--select-worst", type=int, default=10)
    p.add_argument("--select-activity", type=int, default=10)
    p.add_argument("--select-temporal", type=int, default=10)
    p.add_argument("--select-total", type=int, default=40)
    p.add_argument("--sheet-tile-width", type=int, default=320)
    p.add_argument("--devices", nargs="+", default=["cuda:0"])
    p.add_argument("--io-workers", type=int)
    p.add_argument("--prefetch", type=int, default=1)
    p.add_argument("--chunk", type=int, default=32, help="Windows voxelised per GPU step")
    p.add_argument("--metric-batch", type=int, default=64)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-tf32", dest="tf32", action="store_false")
    p.add_argument("--skip-lpips", action="store_true")
    p.add_argument("--progress-every", type=float, default=30)
    p.add_argument("--summarize-only", metavar="RUN_DIR")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.summarize_only:
        print(summarize(args.summarize_only))
        return Path(args.summarize_only)
    if args.bins != 5:
        print("WARNING: the pretrained E2VID and FireNet weights expect --bins 5", flush=True)
    run_dir = run(args)
    print(f"Done: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
