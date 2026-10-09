import argparse
import glob
import os
import h5py
import numpy as np
from PIL import Image

# EDS (Event-aided Direct Sparse Odometry, CVPR'22): beam splitter, Prophesee Gen3 640x480 + RGB camera.
# Archive layout per sequence: events.h5, images/frame_*.png, images_timestamps.txt, times.txt, imu.csv, stamped_groundtruth.txt.
# No official split: every 4th sequence (02, 06, 10, 14) is test. Time units and h5 key names are found from the files.
# RGB frames are 640x480 like the events; check.py tells whether they are pixel-aligned and whether --mid-exposure helps.
SCALES = (1.0, 1e-3, 1e-6, 1e-9)

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description="EDS sequences (folders in this directory) -> evtween layout: <out>/{train,test}/<sequence>/")
ap.add_argument("--out", default="data/eds")
ap.add_argument("--raw", default=HERE, help="folder holding the sequence folders (00_peanuts_dark, ...)")
ap.add_argument("--copy", action="store_true", help="copy PNG frames instead of symlinking them")
ap.add_argument("--dry", action="store_true", help="only print the h5 layout, timestamp files, image size and time spans")
ap.add_argument("--offset-ms", type=float, default=0.0, help="added to frame times (after check.py)")
ap.add_argument("--mid-exposure", action="store_true", help="frame time = stamp + half the exposure in times.txt (stamp = exposure start)")
a = ap.parse_args()


def columns(f):
    # x, y, t, p datasets wherever they sit in the file (events/x or x, t or ts or timestamp, p or polarity)
    found = {}
    f.visititems(lambda n, o: None if not isinstance(o, h5py.Dataset) else found.setdefault(n.rsplit("/", 1)[-1], o) and None)
    pick = lambda *names: next(found[n] for n in names if n in found)
    off = found["t_offset"][()] if "t_offset" in found else 0
    return pick("x"), pick("y"), pick("t", "ts", "timestamp", "timestamps"), pick("p", "polarity", "polarities"), off


def frame_times(seq):
    t = np.loadtxt(os.path.join(seq, "images_timestamps.txt"), comments="#", ndmin=2)[:, -1].astype(np.float64)
    ts = t * next(s for s in SCALES if 5e-4 <= np.median(np.diff(t)) * s <= 1.0) + a.offset_ms * 1e-3
    if a.mid_exposure:  # times.txt: id, timestamp [s], exposure [ms], gain [dB], filename
        ts = ts + np.loadtxt(os.path.join(seq, "times.txt"), comments="#", usecols=2, ndmin=1)[:len(ts)] * 0.5e-3
    return ts, t


def first(t, v):
    # first index with t >= v, by bisection on the h5 dataset (the full time array never sits in memory)
    lo, hi = 0, len(t)
    while lo < hi:
        m = (lo + hi) // 2
        lo, hi = (m + 1, hi) if t[m] < v else (lo, m)
    return lo


seqs = sorted(d for d in glob.glob(os.path.join(a.raw, "*")) if os.path.exists(os.path.join(d, "events.h5")))
test = set(seqs[2::4])
print(f"{len(seqs)} sequences, {len(test)} test: {[os.path.basename(s) for s in sorted(test)]}")
for seq in seqs:
    name, imgs = os.path.basename(seq), sorted(glob.glob(os.path.join(seq, "images", "*.png")))
    ts, ts_raw = frame_times(seq)
    with h5py.File(os.path.join(seq, "events.h5"), "r") as f:
        if a.dry:
            print(f"{name}:")
            f.visititems(lambda n, o: print(f"  h5 {n}: {getattr(o, 'shape', '')} {getattr(o, 'dtype', '')}"
                                            + (f" attrs {dict(o.attrs)}" if len(o.attrs) else "")))
        x, y, t, p, off = columns(f)
        if a.dry:
            for txt in ("images_timestamps.txt", "times.txt"):
                lines = open(os.path.join(seq, txt)).read().splitlines()
                print(f"  {txt}: {len(lines)} lines, first: {lines[:3]}")
            im = Image.open(imgs[0])
            print(f"  images: {len(imgs)} x {im.size[0]}x{im.size[1]} {im.mode}, timestamps {ts_raw[0]:.0f} .. {ts_raw[-1]:.0f} "
                  f"-> {ts[-1] - ts[0]:.1f} s @ {1 / np.median(np.diff(ts)):.1f} fps")
            print(f"  events: {len(t)}, t {t[0] + off} .. {t[-1] + off}, x max {x[:1000000].max()}, y max {y[:1000000].max()}, "
                  f"p values {np.unique(p[:1000000])}")
            continue
        sample = t[::max(1, len(t) // 1000)].astype(np.float64) + off
        se = min(SCALES, key=lambda s: abs(np.log((np.median(sample) * s + 1e-12) / np.median(ts))))  # event unit -> seconds
        n = min(len(imgs), len(ts))
        keep = (ts[:n] >= (t[0] + off) * se) & (ts[:n] <= (t[-1] + off) * se)  # frames inside the event stream
        fr, ts = [imgs[i] for i in np.flatnonzero(keep)], ts[:n][keep]
        base = int(round(ts[0] / se)) - off  # first kept frame in raw event units: times stay exact integers until here
        lo, hi = first(t, base), first(t, int(np.ceil(ts[-1] / se)) - off)
        dst = os.path.join(a.out, "test" if seq in test else "train", name)
        os.makedirs(os.path.join(dst, "frames"), exist_ok=True)
        np.save(os.path.join(dst, "frame_ts.npy"), ts - ts[0])
        out = {k: np.lib.format.open_memmap(os.path.join(dst, f"ev_{k}.npy"), "w+", dt, (hi - lo,))
               for k, dt in (("t", np.float64), ("x", np.int16), ("y", np.int16), ("p", np.int8))}
        for i in range(lo, hi, 50_000_000):  # 50M events at a time
            j, o = min(i + 50_000_000, hi), slice(i - lo, min(i + 50_000_000, hi) - lo)
            out["t"][o] = (t[i:j] - base) * se
            out["x"][o], out["y"][o] = x[i:j], y[i:j]
            out["p"][o] = np.where(p[i:j] > 0, 1, -1)
        for m in out.values():
            m.flush()
        del out
        gap = (t[lo] - base) * se * 1e3
    for k, src in enumerate(fr):
        out = os.path.join(dst, "frames", f"{k:06d}.png")
        if a.copy:
            Image.open(src).convert("RGB").save(out)
        else:
            os.path.lexists(out) or os.symlink(os.path.abspath(src), out)
    W, H = Image.open(fr[0]).size
    print(f"  -> {dst}: {len(fr)}/{n} frames {W}x{H} @ {1 / np.median(np.diff(ts)):.1f} fps, {hi - lo} events, "
          f"event time unit {se:g} s, first event {gap:.2f} ms after the first frame")
