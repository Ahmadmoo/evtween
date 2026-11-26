"""Why can two event streams with the same counts give different reconstructions? Look at what E2VID adds up.

E2VID reads each event as "this pixel's log brightness moved by one threshold" and adds ON minus OFF per pixel.
Event counts (the rate curve, PMSE with ON and OFF pooled separately) do not see that sum. This script shows,
for one sequence, the camera frames next to the real events and each simulator:

  counts_vs_net.png         row 1: where events fire (ON + OFF per pixel); row 2: what they say the brightness
                            did (ON - OFF per pixel) next to the camera's change in log brightness; row 3 (if
                            saved by recon_eval.py): the E2VID reconstruction at that frame
  net_over_windows.png      ON - OFF over the last 1, 5, 20 frame gaps before the frame, next to the camera
  agreement_vs_pooling.png  correlation at 1..32 px pooling, over many windows of the sequence: counts with the
                            real events, net polarity with the real events, net polarity with the camera
  pixel_traces.png          single pixels: the camera's log brightness over time against each stream's running
                            C_on * ON - C_off * OFF (C fitted per stream, so no threshold is assumed)
  agreement.csv, agreement_summary.csv, thresholds.csv, summary.txt, config.json

The windows end at --frame (the frame E2VID reconstructs: frame k uses the events up to frame k), so
--frame 45 matches a recon_eval figure of frame 45. Time alignment and event loading are recon_eval.py's.

    python benchmarks/event_diagnosis.py eds 14_ziggy_in_the_arena --frame 45
    python benchmarks/event_diagnosis.py hqevfi 2024-01-05-15-04_1 --methods dvs_voltmeter evtween_v8.2_student evtween_v8_student

Output: reports/event_diagnosis/<dataset>/<split>/<sequence>/frame<k>_<time>/ (or --out).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

if __package__:
    from . import recon_eval as r
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import recon_eval as r

REAL = r.REAL
DEFAULT_METHODS = ("dvs_voltmeter", "evtween_v8.2_student")

# Colors follow the method, never its position (validated categorical palette; real events in primary ink).
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {REAL: INK, "evtween_v8.2_student": "#2a78d6", "dvs_voltmeter": "#eb6834", "v2e": "#1baf7a",
          "iebcs": "#eda100", "evis_core": "#e87ba4", "fracevent": "#008300",
          "evtween_v8_student": "#6250d6", "evtween_v7_student": "#e34948"}
MARKERS = {REAL: "o", "evtween_v8.2_student": "s", "dvs_voltmeter": "^", "v2e": "D", "iebcs": "v",
           "evis_core": "P", "fracevent": "X", "evtween_v8_student": "<", "evtween_v7_student": ">"}
DIVERGING = ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f4a7a6", "#e34948", "#8f2324"]  # OFF / darker .. ON / brighter
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def label(name):
    return "real events" if name == REAL else name


def color(name):
    return COLORS.get(name, MUTED)


# ---------------------------------------------------------------- data

def load(args):
    """recon_eval's discovery and loading: same sequences, same time base, same event arrays."""
    ns = SimpleNamespace(dataset=args.dataset, split=args.split, methods=list(args.methods),
                         sequences=[args.sequence], events_root=args.events_root, data_root=args.data_root)
    methods, _ = r.discover(ns)
    with ThreadPoolExecutor(4) as pool:
        data = r.load_sequence(args.sequence, ns, methods, pool)
    if data.missing:
        raise FileNotFoundError(f"No events for {data.missing} in {args.sequence}")
    streams = [s for s in (REAL, *methods) if s in data.events]
    return data, streams


class Frames:
    """Camera frames as log brightness: log(gray ** gamma + eps), gray in [0, 1]."""
    def __init__(self, data, gamma, eps, keep=48):
        self.data, self.gamma, self.eps, self.keep, self.cache = data, gamma, eps, keep, {}

    def gray(self, k):
        if k in self.cache:
            self.cache[k] = self.cache.pop(k)  # most recently used last
        else:
            m = self.data.meta
            self.cache[k] = r.load_gray(self.data.frames[k], m["width"], m["height"]).astype(np.float64)
            while len(self.cache) > self.keep:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[k]

    def log(self, k):
        return np.log(self.gray(k) ** self.gamma + self.eps)

    def change(self, a, b):
        return self.log(b) - self.log(a)


def window(ev, a, b):
    """The events between frames a and b (recon_eval's cuts: frame k is reconstructed from the events before it)."""
    i, j = int(ev["cuts"][a]), int(ev["cuts"][b])
    return ev["t"][i:j], ev["x"][i:j], ev["y"][i:j], ev["p"][i:j]


def count_maps(ev, a, b, h, w):
    _, x, y, p = window(ev, a, b)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    idx, on = y[ok] * w + x[ok], p[ok] > 0
    count = lambda sel: np.bincount(idx[sel], minlength=h * w).reshape(h, w).astype(np.float64)
    return count(on), count(~on)


def pool(m, k):
    """Sum over non-overlapping k x k blocks (the edge rows / columns that do not fill a block are dropped)."""
    if k == 1:
        return m
    h, w = (m.shape[0] // k) * k, (m.shape[1] // k) * k
    return m[:h, :w].reshape(h // k, k, w // k, k).sum((1, 3))


def corr(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def _mean(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - b) ** 2)))


# ---------------------------------------------------------------- numbers

def window_stats(maps, change, streams, pools):
    """Per stream and pool size: agreement with the real events (counts, net polarity) and with the camera."""
    rows = []
    ref = maps.get(REAL)
    for s in streams:
        on, off = maps[s]
        net = on - off
        for k in pools:
            row = dict(stream=s, pool_px=k, events=int(on.sum() + off.sum()),
                       camera_r=corr(pool(net, k), pool(change, k)))
            if ref is not None and s != REAL:
                ron, roff = ref
                pk = lambda m: pool(m, k) / (k * k)  # block means, as PMSE
                row.update(
                    count_r=_mean([corr(pool(on, k), pool(ron, k)), corr(pool(off, k), pool(roff, k))]),
                    total_r=corr(pool(on + off, k), pool(ron + roff, k)),
                    net_r=corr(pool(net, k), pool(ron - roff, k)),
                    count_rmse=float(np.sqrt((rmse(pk(on), pk(ron)) ** 2 + rmse(pk(off), pk(roff)) ** 2) / 2)),
                    net_rmse=rmse(pk(net), pk(ron - roff)))
            rows.append(row)
    return rows


class ThresholdFit:
    """Least squares camera change = C_on * ON - C_off * OFF over all pixels of all windows (one fit per stream)."""
    def __init__(self):
        self.A, self.b, self.yy, self.y, self.n = np.zeros((2, 2)), np.zeros(2), 0.0, 0.0, 0

    def add(self, on, off, change):
        X = np.stack([on.ravel(), -off.ravel()], 1)
        y = change.ravel()
        self.A += X.T @ X
        self.b += X.T @ y
        self.yy += float(y @ y)
        self.y += float(y.sum())
        self.n += y.size

    def result(self):
        try:
            c = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan"), float("nan")
        sse = self.yy - 2 * c @ self.b + c @ self.A @ c
        sst = self.yy - self.y ** 2 / max(self.n, 1)
        return float(c[0]), float(c[1]), float(1 - sse / sst) if sst > 0 else float("nan")


class StepSize:
    """Log brightness per event from the camera, ignoring which way each event points: C_on = the camera's total rise
    (summed frame to frame) / ON events, C_off = its total fall / OFF events. A stream with the right counts gets the
    right steps even if its polarities were random, so the traces cannot be made to look right by the fit."""
    def __init__(self):
        self.rise = self.fall = self.on = self.off = 0.0

    def add(self, frames, a, b, on, off):
        for j in range(a, b):
            dl = frames.change(j, j + 1)
            self.rise += float(np.maximum(dl, 0).sum())
            self.fall += float(np.maximum(-dl, 0).sum())
        self.on += float(on.sum())
        self.off += float(off.sum())

    def result(self):
        return (self.rise / self.on if self.on else float("nan"), self.fall / self.off if self.off else float("nan"))


def end_frames(n_frames, gaps, count):
    """count window ends spread over the sequence, each with room for gaps frame gaps before it."""
    if n_frames - 1 < gaps:
        return []
    return sorted({int(round(v)) for v in np.linspace(gaps, n_frames - 1, count)})


def busiest(data, streams, gaps, candidates=None):
    """The window end with the most events (real events if present) in its last gaps frame gaps."""
    ev = data.events[streams[0]]
    ends = np.arange(gaps, len(data.edges)) if candidates is None else np.asarray(sorted(candidates))
    ends = ends[ends >= gaps]
    if not ends.size:
        return None
    counts = ev["cuts"][ends] - ev["cuts"][ends - gaps]
    return int(ends[int(np.argmax(counts))])


def saved_recon(args, network, streams, name):
    """Frames recon_eval.py saved for every stream of this sequence (empty set if none)."""
    ns = SimpleNamespace(recon_root=args.recon_root, dataset=args.dataset, split=args.split)
    found = None
    for s in streams:
        folder = r.recon_dir(ns, network, s, name)
        have = {int(p.stem) for p in folder.glob("*.png") if p.stem.isdigit()} if folder.is_dir() else set()
        found = have if found is None else found & have
    return found or set()


def pick_pixels(frames, data, end, gaps, count, spacing=8, border=2):
    """Pixels where the camera brightness moves most over the window (total variation), spread apart."""
    h, w = data.meta["height"], data.meta["width"]
    tv = sum(np.abs(frames.change(j, j + 1)) for j in range(end - gaps, end))
    if REAL in data.events:  # only pixels where the real camera fired too
        on, off = count_maps(data.events[REAL], end - gaps, end, h, w)
        tv = np.where(on + off > 0, tv, 0)
    tv[:border], tv[-border:], tv[:, :border], tv[:, -border:] = 0, 0, 0, 0
    chosen = []
    for i in np.argsort(tv, axis=None)[::-1]:
        y, x = divmod(int(i), w)
        if tv[y, x] <= 0 or len(chosen) == count:
            break
        if all(max(abs(x - cx), abs(y - cy)) >= spacing for cx, cy in chosen):
            chosen.append((x, y))
    return chosen


def trace(ev, a, b, x, y, c_on, c_off, t0):
    """Times (ms from t0) and running C_on * ON - C_off * OFF at one pixel, starting at 0."""
    t, ex, ey, p = window(ev, a, b)
    sel = (ex == x) & (ey == y)
    t, p = t[sel], p[sel]
    steps = np.where(p > 0, c_on, -c_off)
    return np.concatenate([[0.0], (t - t0) * 1e3]), np.concatenate([[0.0], np.cumsum(steps)])


# ---------------------------------------------------------------- figures

def _plt():
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("matplotlib is needed for the figures: pip install matplotlib") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9,
                         "axes.titlecolor": INK, "figure.facecolor": "white", "savefig.facecolor": "white"})
    cmaps = dict(div=LinearSegmentedColormap.from_list("net", DIVERGING),
                 seq=LinearSegmentedColormap.from_list("count", SEQUENTIAL))
    return plt, cmaps


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def grid(path, suptitle, rows, aspect, note=None):
    """rows: [(row label, [panel or None], colorbar label or None)]; panel: dict(img, cmap, vmin, vmax, title)."""
    plt, _ = _plt()
    ncol = max(len(p) for _, p, _ in rows)
    pw = 2.7
    fig, axes = plt.subplots(len(rows), ncol, figsize=(pw * ncol + 0.9, len(rows) * (pw * aspect + 0.75) + 0.7),
                             layout="constrained", squeeze=False)
    for i, (row_label, panels, cbar) in enumerate(rows):
        mappable = None
        for j in range(ncol):
            ax = axes[i, j]
            _bare(ax)
            p = panels[j] if j < len(panels) else None
            if p is None:
                ax.set_visible(j == 0)
            else:
                im = ax.imshow(p["img"], cmap=p["cmap"], vmin=p.get("vmin"), vmax=p.get("vmax"),
                               interpolation="nearest")
                ax.set_title(p.get("title", ""), loc="left", fontsize=8.5)
                for x, y, n in p.get("marks", []):
                    ax.plot(x, y, "o", ms=9, mfc="none", mec="#e34948", mew=1.5)
                    ax.annotate(str(n), (x, y), xytext=(6, 6), textcoords="offset points", color="#e34948",
                                fontsize=8, fontweight="bold")
                if j >= 1 and mappable is None:
                    mappable = im
        axes[i, 0].set_ylabel(row_label, fontsize=9, color=INK2)
        if cbar and mappable is not None:
            cb = fig.colorbar(mappable, ax=list(axes[i, 1:]), fraction=0.02, pad=0.01)
            cb.set_label(cbar, color=INK2)
            cb.outline.set_edgecolor(BASE)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=11, color=INK)
    if note:
        fig.text(0.01, 0.0, note, ha="left", va="bottom", fontsize=8, color=INK2)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save(fig, path):
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.15)


def _r(v):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.2f}"


def figure_counts_vs_net(path, data, frames, streams, end, gaps, maps, stats, recon, pixels, title):
    _, cm = _plt()
    change = frames.change(end - gaps, end)
    ref = maps.get(REAL) or maps[streams[0]]
    cmax = max(np.percentile(ref[0] + ref[1], 99.5), 1)
    nmax = max(np.percentile(np.abs(ref[0] - ref[1]), 99.5), 1)
    dmax = max(np.percentile(np.abs(change), 99.5), 1e-3)
    by = {(s["stream"], s["pool_px"]): s for s in stats}
    count_row = [dict(img=frames.gray(end), cmap="gray", vmin=0, vmax=1, title=f"camera frame {end}",
                      marks=[(x, y, n + 1) for n, (x, y) in enumerate(pixels)])]
    net_row = [dict(img=change, cmap=cm["div"], vmin=-dmax, vmax=dmax,
                    title=f"camera: change in log brightness\nframes {end - gaps} to {end} (scale ±{dmax:.2f})")]
    for s in streams:
        on, off = maps[s]
        one, four = by.get((s, 1), {}), by.get((s, 4), {})
        extra = "" if s == REAL else f"\nr vs real {_r(one.get('count_r'))} (1 px), {_r(four.get('count_r'))} (4 px)"
        count_row.append(dict(img=on + off, cmap=cm["seq"], vmin=0, vmax=cmax,
                              title=f"{label(s)}\n{int(on.sum() + off.sum()):,} events{extra}"))
        extra = "" if s == REAL else f"\nvs real {_r(one.get('net_r'))} (1 px), {_r(four.get('net_r'))} (4 px)"
        net_row.append(dict(img=on - off, cmap=cm["div"], vmin=-nmax, vmax=nmax,
                            title=f"{label(s)}\nvs camera {_r(one.get('camera_r'))} (1 px), "
                                  f"{_r(four.get('camera_r'))} (4 px){extra}"))
    rows = [("Where events fire", count_row, "ON + OFF events per pixel"),
            ("What they say the brightness did", net_row, "ON − OFF per pixel (red brighter, blue darker)")]
    if recon:
        rec = [dict(img=frames.gray(end), cmap="gray", vmin=0, vmax=1, title=f"camera frame {end}")]
        rec += [dict(img=recon[s], cmap="gray", vmin=0, vmax=1, title=f"{label(s)}\n{recon['network']} at frame {end}")
                if s in recon else None for s in streams]
        rows.append((f"{recon['network']} reconstruction", rec, None))
    aspect = data.meta["height"] / data.meta["width"]
    grid(path, title, rows, aspect,
         note="r = Pearson correlation over pixels (1 px) or 4×4 px blocks. Counts: ON and OFF maps compared "
              "separately, as PMSE does. Net: ON − OFF, the sum E2VID integrates.")


def figure_net_over_windows(path, data, frames, streams, end, gaps_list, title):
    _, cm = _plt()
    h, w = data.meta["height"], data.meta["width"]
    rows = []
    for g in gaps_list:
        change = frames.change(end - g, end)
        maps = {s: count_maps(data.events[s], end - g, end, h, w) for s in streams}
        ref = maps.get(REAL) or maps[streams[0]]
        nmax = max(np.percentile(np.abs(ref[0] - ref[1]), 99.5), 1)
        dmax = max(np.percentile(np.abs(change), 99.5), 1e-3)
        ms = (data.edges[end] - data.edges[end - g]) * 1e3
        panels = [dict(img=change, cmap=cm["div"], vmin=-dmax, vmax=dmax, title=f"camera (scale ±{dmax:.2f})")]
        for s in streams:
            net = maps[s][0] - maps[s][1]
            panels.append(dict(img=net, cmap=cm["div"], vmin=-nmax, vmax=nmax,
                               title=f"{label(s)}\nr vs camera {_r(corr(net, change))}"))
        rows.append((f"last {g} frame gap{'s' if g > 1 else ''} ({ms:.0f} ms)", panels, "ON − OFF per pixel"))
    grid(path, title, rows, data.meta["height"] / data.meta["width"],
         note="Each row: the events between frame end−n and the frame, next to the camera's change in log brightness "
              "over the same frames. Red brighter, blue darker.")


def figure_pooling(path, summary, streams, gaps, n_windows, title):
    plt, _ = _plt()
    panels = [("count_r", "a  Event counts vs real\n(ON and OFF separately, like PMSE)", False),
              ("net_r", "b  Net polarity ON − OFF vs real\n(what E2VID adds up)", False),
              ("camera_r", "c  Net polarity vs the camera's change\nin log brightness", True)]
    if REAL not in streams:
        panels = panels[2:]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.9), layout="constrained", squeeze=False)
    for ax, (key, name, with_real) in zip(axes[0], panels):
        ends = []
        for s in streams:
            if s == REAL and not with_real:
                continue
            pts = sorted((row["pool_px"], row[key]) for row in summary
                         if row["stream"] == s and row["gaps"] == gaps and np.isfinite(row.get(key, np.nan)))
            if not pts:
                continue
            x, y = zip(*pts)
            ax.plot(x, y, color=color(s), marker=MARKERS.get(s, "o"), ms=6, lw=2, label=label(s),
                    zorder=3 if s == REAL else 2)
            ends.append((y[-1], x[-1]))
        placed = []
        for yv, xv in sorted(ends):  # value labels at the line ends, nudged apart when lines end together
            ty = max([yv] + [q + 0.045 for q in placed])
            placed.append(ty)
            ax.annotate(f"{yv:.2f}", (xv, yv), xytext=(xv * 1.08, ty), textcoords="data", va="center",
                        fontsize=8, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([row for row in sorted({row["pool_px"] for row in summary})])
        ax.set_xticklabels([str(k) for k in sorted({row["pool_px"] for row in summary})])
        ax.set_xlabel("pooling block (px)")
        ax.set_ylabel("Pearson r (mean over windows)")
        ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.02)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(name, loc="left")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels), frameon=False)
    fig.suptitle(f"{title}\n{n_windows} windows of {gaps} frame gaps spread over the sequence",
                 x=0.01, ha="left", fontsize=10.5)
    _save(fig, path)
    plt.close(fig)


def figure_traces(path, data, frames, streams, end, gaps, pixels, steps, title):
    plt, _ = _plt()
    if not pixels:
        return False
    ncol = min(3, len(pixels))
    nrow = -(-len(pixels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.0 * nrow + 0.8), layout="constrained",
                             squeeze=False)
    a, t0 = end - gaps, data.edges[end - gaps]
    times = (data.edges[a:end + 1] - t0) * 1e3
    for n, (x, y) in enumerate(pixels):
        ax = axes.flat[n]
        cam = np.array([frames.log(j)[y, x] for j in range(a, end + 1)]) - frames.log(a)[y, x]
        ax.plot(times, cam, color=MUTED, lw=1, ls="--", marker="o", ms=4, label="camera frames", zorder=4)
        for s in streams:
            c_on, c_off = steps[s]
            if not (np.isfinite(c_on) and np.isfinite(c_off)):
                continue
            tt, yy = trace(data.events[s], a, end, x, y, c_on, c_off, t0)
            tt = np.append(tt, times[-1])
            yy = np.append(yy, yy[-1])
            ax.step(tt, yy, where="post", color=color(s), lw=2 if s == REAL else 1.5, label=label(s),
                    zorder=3 if s == REAL else 2)
        ax.set_title(f"pixel {n + 1}  (x={x}, y={y})", loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("ms")
        ax.set_ylabel("change in log brightness")
    for ax in list(axes.flat)[len(pixels):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels), frameon=False)
    fig.suptitle(f"{title}\nEach line adds up its events at one pixel: ON +C_on, OFF −C_off. C = the camera's brightness "
                 f"rise (fall) per ON (OFF) event of that stream, so polarity plays no part in it.\n"
                 f"The pixels are marked in counts_vs_net.png.", x=0.01, ha="left", fontsize=10)
    _save(fig, path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- run

def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        out.writeheader()
        for row in rows:
            out.writerow({k: ("" if isinstance(row.get(k), float) and not np.isfinite(row[k]) else row.get(k, ""))
                          for k in columns})


def summarize(rows):
    keys = ("count_r", "total_r", "net_r", "camera_r", "count_rmse", "net_rmse", "events")
    groups = {}
    for row in rows:
        groups.setdefault((row["gaps"], row["stream"], row["pool_px"]), []).append(row)
    out = []
    for (g, s, k), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        row = dict(gaps=g, stream=s, pool_px=k, windows=len(items))
        for key in keys:
            row[key] = _mean([i[key] for i in items if key in i])
        out.append(row)
    return out


def run(args):
    data, streams = load(args)
    h, w = data.meta["height"], data.meta["width"]
    n_frames = len(data.edges)
    frames = Frames(data, args.gamma, args.eps)
    gaps_list = sorted({g for g in args.gaps if 0 < g < n_frames})
    main = args.main_gaps
    if not gaps_list or not 0 < main < n_frames:
        raise ValueError(f"{args.sequence} has {n_frames} frames; --gaps / --main-gaps do not fit")
    saved = saved_recon(args, args.network, streams, data.name)
    end = args.frame
    if end is None:  # a frame recon_eval saved if possible, else the busiest window
        end = busiest(data, streams, max(gaps_list + [main]), saved or None) or busiest(data, streams, main)
    if not 0 < end < n_frames:
        raise ValueError(f"--frame {end} outside 1..{n_frames - 1}")
    if end - main < 0:
        raise ValueError(f"--frame {end} is too early for --main-gaps {main}")
    dropped = [g for g in gaps_list if end - g < 0]
    gaps_list = [g for g in gaps_list if end - g >= 0]
    out = Path(args.out) if args.out else (r.ROOT / "reports" / "event_diagnosis" / args.dataset / args.split /
                                           data.name / f"frame{end}_{datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    head = f"{args.dataset}/{args.split}/{data.name}"
    print(f"{head}: {n_frames} frames {w}x{h}, streams {', '.join(streams)}; window ends at frame {end}", flush=True)
    if dropped:
        print(f"  gaps {dropped} do not fit before frame {end}; skipped", flush=True)

    # numbers over many windows of the sequence (and the threshold fit on the main length)
    rows, fits, steps = [], {s: ThresholdFit() for s in streams}, {s: StepSize() for s in streams}
    for g in sorted(set(gaps_list + [main])):
        ends = end_frames(n_frames, g, args.windows)
        for e in ends:
            change = frames.change(e - g, e)
            maps = {s: count_maps(data.events[s], e - g, e, h, w) for s in streams}
            for row in window_stats(maps, change, streams, args.pools):
                rows.append(dict(dataset=args.dataset, split=args.split, sequence=data.name, gaps=g, end_frame=e, **row))
            if g == main:
                for s in streams:
                    fits[s].add(*maps[s], change)
                    steps[s].add(frames, e - g, e, *maps[s])
        print(f"  {len(ends)} windows of {g} gaps scored", flush=True)
    fits = {s: f.result() for s, f in fits.items()}
    steps = {s: f.result() for s, f in steps.items()}
    summary = summarize(rows)

    # the window that ends at the chosen frame
    maps = {s: count_maps(data.events[s], end - main, end, h, w) for s in streams}
    stats = window_stats(maps, frames.change(end - main, end), streams, sorted(set(args.pools) | {1, 4}))
    longest = max(gaps_list)
    pixels = pick_pixels(frames, data, end, longest, args.pixels)
    recon = None
    if end in saved:
        ns = SimpleNamespace(recon_root=args.recon_root, dataset=args.dataset, split=args.split)
        recon = {s: np.asarray(Image.open(r.recon_dir(ns, args.network, s, data.name) / f"{end:06d}.png")
                               .convert("L"), np.float64) / 255 for s in streams}
        recon["network"] = args.network
    elif saved:
        print(f"  no saved {args.network} frame {end}; saved frames: {sorted(saved)[:40]}", flush=True)

    ms = (data.edges[end] - data.edges[end - main]) * 1e3
    title = f"{head} · frames {end - main}–{end} ({main} gaps, {ms:.0f} ms)"
    figure_counts_vs_net(out / "counts_vs_net.png", data, frames, streams, end, main, maps, stats, recon, pixels, title)
    figure_net_over_windows(out / "net_over_windows.png", data, frames, streams, end, gaps_list,
                            f"{head} · window ends at frame {end}")
    n_windows = len(end_frames(n_frames, main, args.windows))
    figure_pooling(out / "agreement_vs_pooling.png", summary, streams, main, n_windows, head)
    figure_traces(out / "pixel_traces.png", data, frames, streams, end, longest, pixels, steps,
                  f"{head} · frames {end - longest}–{end}")

    columns = ("dataset", "split", "sequence", "gaps", "end_frame", "stream", "pool_px", "events", "count_r",
               "total_r", "net_r", "camera_r", "count_rmse", "net_rmse")
    write_csv(out / "agreement.csv", rows, columns)
    write_csv(out / "agreement_summary.csv", summary, ("gaps", "stream", "pool_px", "windows", *columns[7:]))
    write_csv(out / "thresholds.csv", [dict(stream=s, fit_c_on=c[0], fit_c_off=c[1], fit_r2=c[2],
                                            step_c_on=steps[s][0], step_c_off=steps[s][1]) for s, c in fits.items()],
              ("stream", "fit_c_on", "fit_c_off", "fit_r2", "step_c_on", "step_c_off"))
    (out / "config.json").write_text(json.dumps(dict(
        vars(args), out=str(out), frame=end, streams=streams, pixels=pixels, events_root=str(args.events_root or ""),
        data_root=str(args.data_root or ""), recon_root=str(args.recon_root or ""), source=str(data.source),
        recon_shown=recon is not None),
        indent=2, default=str))
    text = report(summary, fits, streams, main, n_windows, head)
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"Saved to {out}", flush=True)
    return out


def report(summary, fits, streams, gaps, n_windows, head):
    by = {(row["stream"], row["pool_px"]): row for row in summary if row["gaps"] == gaps}
    lines = [f"{head}: mean over {n_windows} windows of {gaps} frame gaps",
             f"{'stream':<24}{'counts r vs real':>18}{'net r vs real':>16}{'net r vs camera':>18}"
             f"{'C_on':>8}{'C_off':>8}{'R²':>7}",
             f"{'':<24}{'1 px / 4 px':>18}{'1 px / 4 px':>16}{'1 px / 4 px':>18}"]
    for s in streams:
        one, four = by.get((s, 1), {}), by.get((s, 4), {})
        pair = lambda key: f"{_r(one.get(key))} / {_r(four.get(key))}"
        c = fits[s]
        lines.append(f"{label(s):<24}{('—' if s == REAL else pair('count_r')):>18}"
                     f"{('—' if s == REAL else pair('net_r')):>16}{pair('camera_r'):>18}"
                     f"{_r(c[0]):>8}{_r(c[1]):>8}{_r(c[2]):>7}")
    lines.append("counts: ON and OFF maps compared separately (as PMSE); net: ON − OFF per pixel (what E2VID adds up); "
                 "camera: the frames' change in log brightness; C, R²: least squares camera change = "
                 "C_on·ON − C_off·OFF.")
    return "\n".join(lines) + "\n"


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("sequence")
    p.add_argument("--split", default="test")
    p.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    p.add_argument("--frame", type=int, default=None,
                   help="the frame the windows end at (= the reconstructed frame); default: a frame recon_eval saved, "
                        "else the busiest window")
    p.add_argument("--main-gaps", type=int, default=5, help="frame gaps in the main window (default 5)")
    p.add_argument("--gaps", type=int, nargs="+", default=[1, 5, 20], help="window lengths for net_over_windows.png")
    p.add_argument("--pools", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--windows", type=int, default=40, help="windows spread over the sequence for the numbers")
    p.add_argument("--pixels", type=int, default=6, help="pixels in pixel_traces.png")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="PNG value ** gamma = linear light (1.0 for linear frames, 2.2 for gamma-encoded ones)")
    p.add_argument("--eps", type=float, default=0.01, help="log(light + eps)")
    p.add_argument("--network", default="e2vid", choices=r.NETWORKS, help="saved reconstructions to show")
    p.add_argument("--events-root", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--recon-root", type=Path)
    p.add_argument("--out", type=Path)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
