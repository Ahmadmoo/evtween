import argparse
import glob
import json
import os
import re
import shutil
import tarfile
import zipfile
import h5py
import numpy as np
from PIL import Image

# EDS (Event-aided Direct Sparse Odometry, CVPR'22): beam splitter, Prophesee Gen3 640x480 events + 640x480 RGB at 75 Hz
# (exposure ~10 ms). Input: the per-sequence "Archive file" downloads (or the folders they were extracted to):
# <seq>/events.h5 (x, y, t in us, p), images/frame_*.png, images_timestamps.txt (us), times.txt (id, timestamp [s],
# exposure [ms], gain, file), imu.csv, stamped_groundtruth.txt. Archives are unpacked one at a time to --work and removed.
# Calibration recordings are skipped. No official split: sequences 02, 06, 10, 14 (every 4th) are test.
URL = "https://rpg.ifi.uzh.ch/eds.html"
ARCHIVE = r"\.(zip|tar|tgz|tar\.gz|tar\.bz2|tar\.xz)$"
SCALES = (1.0, 1e-3, 1e-6, 1e-9)
STEP = 50_000_000  # events copied at a time (a sequence holds up to ~1e9)

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description="EDS archives or extracted folders -> <out>/{train,test}/<sequence>/")
ap.add_argument("--src", nargs="+", default=sorted(glob.glob(os.path.join(HERE, "*"))), help="archives and/or folders")
ap.add_argument("--out", default="data/eds")
ap.add_argument("--work", default=None, help="archives are unpacked one at a time into <work>/.tmp (default work: <out>)")
ap.add_argument("--mid-exposure", action="store_true", help="frame time = stamp + half the exposure in times.txt (stamp = exposure start)")
ap.add_argument("--offset-ms", type=float, default=0.0, help="added to frame times")
ap.add_argument("--warp", default=None, help="s,dy,dx: frame pixel = s * (event pixel - center) + center + (dy, dx); frames are "
                "resampled onto the event pixels and both are cut to the part the frames cover (check_align.py finds s, dy, dx)")
ap.add_argument("--dry", action="store_true", help="only list the sequences and their split")
a = ap.parse_args()
work = os.path.join(a.work or a.out, ".tmp")  # only this subfolder is created and removed


def columns(f):
    # x, y, t, p datasets wherever they sit in the file (events/x or x, t or ts or timestamp, p or polarity)
    found = {}
    f.visititems(lambda n, o: None if not isinstance(o, h5py.Dataset) else found.setdefault(n.rsplit("/", 1)[-1], o) and None)
    pick = lambda *names: next(found[n] for n in names if n in found)
    off = np.int64(found["t_offset"][()]) if "t_offset" in found else np.int64(0)
    return pick("x"), pick("y"), pick("t", "ts", "timestamp", "timestamps"), pick("p", "polarity", "polarities"), off


def first(t, v):
    # first index with t >= v, by bisection on the h5 dataset (the full time array never sits in memory)
    lo, hi = 0, len(t)
    while lo < hi:
        m = (lo + hi) // 2
        lo, hi = (m + 1, hi) if t[m] < v else (lo, m)
    return lo


def out_xy(dst, x0, y0, x1, y1):
    # keep the events inside the box the frames cover, coordinates relative to its corner (done in place, 50M at a time)
    ev = {k: np.load(os.path.join(dst, f"ev_{k}.npy"), mmap_mode="r") for k in "txyp"}
    n, m = len(ev["t"]), 0
    out = {k: np.lib.format.open_memmap(os.path.join(dst, f"ev_{k}.tmp.npy"), "w+", v.dtype, (n,)) for k, v in ev.items()}
    for i in range(0, n, STEP):
        x, y = np.asarray(ev["x"][i:i + STEP]), np.asarray(ev["y"][i:i + STEP])
        ok = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        j = m + int(ok.sum())
        out["t"][m:j], out["p"][m:j] = np.asarray(ev["t"][i:i + STEP])[ok], np.asarray(ev["p"][i:i + STEP])[ok]
        out["x"][m:j], out["y"][m:j], m = x[ok] - x0, y[ok] - y0, j
    for k in "txyp":
        final = np.lib.format.open_memmap(os.path.join(dst, f"ev_{k}.new.npy"), "w+", out[k].dtype, (m,))
        for i in range(0, m, STEP):
            final[i:i + STEP] = out[k][i:min(i + STEP, m)]
        final.flush()
        del final, out[k]
        os.replace(os.path.join(dst, f"ev_{k}.new.npy"), os.path.join(dst, f"ev_{k}.npy"))
        os.remove(os.path.join(dst, f"ev_{k}.tmp.npy"))
    return x1 - x0 + 1, y1 - y0 + 1, m / max(n, 1)


def convert(seq, dst, move):
    imgs = sorted(glob.glob(os.path.join(seq, "images", "*.png")))
    t_img = np.loadtxt(os.path.join(seq, "images_timestamps.txt"), comments="#", ndmin=2)[:, -1].astype(np.float64)
    ts = t_img * next(s for s in SCALES if 5e-4 <= np.median(np.diff(t_img)) * s <= 1.0) + a.offset_ms * 1e-3
    if a.mid_exposure:
        ts = ts + np.loadtxt(os.path.join(seq, "times.txt"), comments="#", usecols=2, ndmin=1)[:len(ts)] * 0.5e-3
    os.makedirs(os.path.join(dst, "frames"))
    with h5py.File(os.path.join(seq, "events.h5"), "r") as f:
        x, y, t, p, off = columns(f)
        sample = t[::max(1, len(t) // 1000)].astype(np.float64) + off
        se = min(SCALES, key=lambda s: abs(np.log((np.median(sample) * s + 1e-12) / np.median(ts))))  # event unit -> seconds
        n = min(len(imgs), len(ts))
        keep = (ts[:n] >= (t[0] + off) * se) & (ts[:n] <= (t[-1] + off) * se)  # frames inside the event stream
        fr, ts = [imgs[i] for i in np.flatnonzero(keep)], ts[:n][keep]
        base = int(round(ts[0] / se)) - off  # first kept frame in raw event units: times stay exact integers until here
        lo, hi = first(t, base), first(t, int(np.ceil(ts[-1] / se)) - off)
        np.save(os.path.join(dst, "frame_ts.npy"), ts - ts[0])
        out = {k: np.lib.format.open_memmap(os.path.join(dst, f"ev_{k}.npy"), "w+", dt, (hi - lo,))
               for k, dt in zip("txyp", (np.float64, np.int16, np.int16, np.int8))}
        for i in range(lo, hi, STEP):
            j, o = min(i + STEP, hi), slice(i - lo, min(i + STEP, hi) - lo)
            out["t"][o] = (t[i:j] - base) * se
            out["x"][o], out["y"][o] = x[i:j], y[i:j]
            out["p"][o] = np.where(p[i:j] > 0, 1, -1)
        for v in out.values():
            v.flush()
        del out
    if a.warp:  # RGB and event cameras see the scene at different scales: resample frames onto the event pixels
        s, dy, dx = map(float, a.warp.split(","))
        W, H = Image.open(fr[0]).size
        cx, cy = W / 2 + dx - s * W / 2, H / 2 + dy - s * H / 2  # frame pixel = s * event pixel + (cx, cy)
        x0, y0 = max(0, int(np.ceil(-cx / s))), max(0, int(np.ceil(-cy / s)))
        x1, y1 = min(W - 1, int((W - 1 - cx) / s)), min(H - 1, int((H - 1 - cy) / s))
        aff = (s, 0, s * x0 + cx + 0.5 - 0.5 * s, 0, s, s * y0 + cy + 0.5 - 0.5 * s)  # PIL maps pixel centers
        ex, ey, keep = out_xy(dst, x0, y0, x1, y1)
        for k, src in enumerate(fr):
            Image.open(src).convert("RGB").transform((x1 - x0 + 1, y1 - y0 + 1), Image.AFFINE, aff, Image.BICUBIC).save(
                os.path.join(dst, "frames", f"{k:06d}.png"))
        fr = []
        print(f"  warp s {s} dy {dy} dx {dx}: event pixels x {x0}..{x1}, y {y0}..{y1} kept ({100 * keep:.0f}% of the events)")
    for k, src in enumerate(fr):
        target = os.path.join(dst, "frames", f"{k:06d}.png")
        if move:
            os.replace(src, target)
        else:
            try:
                os.link(os.path.realpath(src), target)
            except OSError:
                shutil.copy(src, target)
    open(os.path.join(dst, "done"), "w").close()
    W, H = Image.open(os.path.join(dst, "frames", "000000.png")).size
    print(f"  -> {dst}: {len(glob.glob(os.path.join(dst, 'frames', '*.png')))}/{n} frames {W}x{H} @ {1 / np.median(np.diff(ts)):.1f} fps, {hi - lo} events, "
          f"event time unit {se:g} s", flush=True)


items = {}  # sequence name -> archive or folder (first one found wins)
for path in a.src:
    if os.path.isdir(path):
        for h in sorted(glob.glob(os.path.join(path, "**", "events.h5"), recursive=True)):
            items.setdefault(os.path.basename(os.path.dirname(h)), os.path.dirname(h))
    elif re.search(ARCHIVE, path):
        items.setdefault(re.sub(ARCHIVE, "", os.path.basename(path)), path)
os.makedirs(a.out, exist_ok=True)
json.dump(dict(name="EDS", paper="Event-aided Direct Sparse Odometry (CVPR 2022)", url=URL,
               events="Prophesee Gen3 640x480, beam splitter", frames="RGB 640x480, 75 Hz, ~10 ms exposure", cfa=None,
               frame_time="stamp + half exposure" if a.mid_exposure else "stamp as released",
               splits="every 4th sequence (02, 06, 10, 14) is test", converter="EDS/eds.py"),
          open(os.path.join(a.out, "info.json"), "w"), indent=1)
for name, path in sorted(items.items()):
    if re.search("calib|hand_eye", name):
        continue
    split = "test" if name[:2].isdigit() and int(name[:2]) % 4 == 2 else "train"
    dst = os.path.join(a.out, split, name)
    if os.path.exists(os.path.join(dst, "done")):
        continue
    if a.dry:
        print(f"  {split}: {name} ({'archive' if os.path.isfile(path) else 'folder'})")
        continue
    shutil.rmtree(dst, ignore_errors=True)
    tmp, move = os.path.join(work, name), os.path.isfile(path)  # frames are moved only out of our own unpacked copy
    try:
        if move:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"{name}: unpacking {os.path.basename(path)}", flush=True)
            if zipfile.is_zipfile(path):
                zipfile.ZipFile(path).extractall(tmp)
            else:
                with tarfile.open(path) as tf:
                    tf.extractall(tmp, filter="data")
            path = os.path.dirname(glob.glob(os.path.join(tmp, "**", "events.h5"), recursive=True)[0])
        convert(path, dst, move)
    except Exception as e:
        print(f"  ! {dst}: failed ({type(e).__name__}: {e})", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(work, ignore_errors=True)
