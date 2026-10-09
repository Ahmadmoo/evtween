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
    print(f"  -> {dst}: {len(fr)}/{n} frames {W}x{H} @ {1 / np.median(np.diff(ts)):.1f} fps, {hi - lo} events, "
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
