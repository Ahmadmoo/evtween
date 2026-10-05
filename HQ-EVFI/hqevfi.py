import argparse
import glob
import importlib.util
import os
import re
import shutil
import urllib.request
import numpy as np
from PIL import Image

# HQ-EVFI (TimeLens-XL, ECCV'24): beam splitter, Prophesee EVK4-HD + 142 fps RGB.
# Layout per sequence: visual_RGB/<n>_*.png and one event file per frame interval in RGB-EVS/*.npz (x, y, t, p);
# some sequences use RGB-EVS_EVSneg3ms with images shifted by one. Ranges and test split: TimeLens-XL dataset_dict.py
DRIVE_ID = "104ZMJ-M_frImOOCGfLk_HDb2FV1trveT"
LISTS = "https://raw.githubusercontent.com/OpenImagingLab/TimeLens-XL/main/dataset/RC_4816/dataset_dict.py"
FPS = 142.0
SCALES = (1.0, 1e-3, 1e-6, 1e-9)

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description="HQ-EVFI zip (put in this folder) -> evtween layout: <out>/{train,test}/<sequence>/")
ap.add_argument("--out", default="data/hqevfi")
ap.add_argument("--raw", default=os.path.join(HERE, "raw"), help="where the zip is extracted (skipped if it exists)")
ap.add_argument("--lists", default=LISTS, help="TimeLens-XL dataset_dict.py (url or local path)")
ap.add_argument("--copy", action="store_true", help="copy PNG frames instead of symlinking them")
a = ap.parse_args()
raw = a.raw

if not os.path.isdir(raw):
    zips = glob.glob(os.path.join(HERE, "*.zip"))
    assert zips, f"put the HQ-EVFI zip (https://drive.google.com/file/d/{DRIVE_ID}) in {HERE}"
    print(f"extracting {zips[0]} -> {raw}")
    shutil.unpack_archive(zips[0], raw)

lists = os.path.join(a.out, "dataset_dict.py")
if not os.path.exists(lists):
    os.makedirs(a.out, exist_ok=True)
    shutil.copy(a.lists, lists) if os.path.exists(a.lists) else urllib.request.urlretrieve(a.lists, lists)
spec = importlib.util.spec_from_file_location("dataset_dict", lists)
meta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta)
natural = lambda f: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", os.path.basename(f))]


def load(f):
    e = np.load(f)
    return [np.asarray(e[k]).reshape(-1) for k in "xytp"]


def frame_times(chunks):
    # frame i sits between event file i-1 and i; end frames one median interval out.
    # if event times restart in every file, frames are 1/FPS apart and events are shifted to absolute time
    lo = np.array([c[2].min() if len(c[2]) else np.nan for c in chunks], dtype=np.float64)
    hi = np.array([c[2].max() if len(c[2]) else np.nan for c in chunks], dtype=np.float64)
    i, ok = np.arange(len(chunks)), ~np.isnan(lo)
    lo, hi = np.interp(i, i[ok], lo[ok]), np.interp(i, i[ok], hi[ok])
    if np.median(np.diff(lo)) < 0.5 * np.median(hi - lo):
        s = next(s for s in SCALES if np.median(hi - lo) * s <= 1.5 / FPS)
        for k, c in enumerate(chunks):
            c[2] = k / FPS + c[2].astype(np.float64) * s
        return np.arange(len(chunks) + 1) / FPS
    mid = (hi[:-1] + lo[1:]) / 2
    step = np.median(np.diff(mid))
    return np.concatenate([mid[:1] - step, mid, mid[-1:] + step])


def write(dst, imgs, chunks):
    ts = frame_times(chunks)
    s = next(s for s in SCALES if 5e-4 <= np.median(np.diff(ts)) * s <= 1.0)  # time unit -> seconds
    W, H = Image.open(imgs[0]).size
    x, y, t, p = (np.concatenate([c[j] for c in chunks]) for j in range(4))
    o = np.argsort(t, kind="stable")
    x, y = np.round(x[o]), np.round(y[o])
    keep = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    os.makedirs(os.path.join(dst, "frames"), exist_ok=True)
    np.save(os.path.join(dst, "frame_ts.npy"), ts * s)
    np.save(os.path.join(dst, "ev_t.npy"), t[o][keep].astype(np.float64) * s)
    np.save(os.path.join(dst, "ev_x.npy"), x[keep].astype(np.int16))
    np.save(os.path.join(dst, "ev_y.npy"), y[keep].astype(np.int16))
    np.save(os.path.join(dst, "ev_p.npy"), np.where(p[o][keep] > 0, 1, -1).astype(np.int8))
    for k, f in enumerate(imgs):
        out = os.path.join(dst, "frames", f"{k:06d}.png")
        if f.lower().endswith(".png") and not a.copy:
            os.path.lexists(out) or os.symlink(os.path.abspath(f), out)
        else:
            Image.open(f).convert("RGB").save(out)
    print(f"  -> {dst}: {len(imgs)} frames {W}x{H} @ {1 / np.median(np.diff(ts * s)):.1f} fps, {keep.sum()} events")


seqs = sorted({os.path.dirname(d) for d in glob.glob(os.path.join(raw, "**", "visual_RGB"), recursive=True)})
print(f"{len(seqs)} sequences in {raw}")
for seq in seqs:
    name = os.path.basename(seq)
    if name not in meta.dataset_dict and name not in meta.EVSneg3:
        print(f"{name}: not in the TimeLens-XL lists, skipped")
        continue
    shifted = name in meta.EVSneg3 and os.path.isdir(os.path.join(seq, "RGB-EVS_EVSneg3ms"))
    imgs = sorted((f for f in glob.glob(os.path.join(seq, "visual_RGB", "*")) if f.lower().endswith((".png", ".jpg", ".jpeg"))), key=natural)
    efiles = sorted(glob.glob(os.path.join(seq, "RGB-EVS_EVSneg3ms" if shifted else "RGB-EVS", "*.npz")), key=natural)
    imgs = imgs[int(shifted):int(shifted) + len(efiles)]
    first = int(os.path.basename(imgs[0]).split("_")[0]) if name in meta.dataset_dict else 0
    r = meta.dataset_dict.get(name, [0, len(imgs)])
    for j in range(len(r) // 2):  # official ranges: images [s, e) with their event files, the last one ends past the range
        key = f"{name}_{j}" if j else name
        s, e = r[2 * j] - first, min(r[2 * j + 1] - first, len(imgs))
        split = "test" if key in meta.test_key else "train"
        write(os.path.join(a.out, split, key), imgs[s:e], [load(f) for f in efiles[s:e - 1]])
