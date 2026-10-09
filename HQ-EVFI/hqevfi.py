import argparse
import glob
import io
import json
import os
import re
import runpy
import shutil
import types
import urllib.request
import zipfile
import numpy as np
from PIL import Image

# HQ-EVFI (TimeLens-XL, ECCV'24): beam splitter with checkerboard calibration (pixel-aligned RGB and events),
# Prophesee EVK4-HD events + 142 fps RGB, 71 sequences. Input: the zip from Google Drive (or the folder it was extracted to).
# The zip holds one zip per sequence: <seq>/visual_RGB/<n>_*.png and one event file per frame interval in RGB-EVS/*.npz
# (x, y, t, p); sequences in the EVSneg3 list use RGB-EVS_EVSneg3ms (events corrected by 3 ms) with images shifted by one.
# Ranges, test split and that list come from TimeLens-XL dataset_dict.py. Inner zips are unpacked one at a time to --work.
DRIVE = "https://drive.google.com/file/d/104ZMJ-M_frImOOCGfLk_HDb2FV1trveT"
LISTS = "https://raw.githubusercontent.com/OpenImagingLab/TimeLens-XL/main/dataset/RC_4816/dataset_dict.py"
FPS = 142.0
SCALES = (1.0, 1e-3, 1e-6, 1e-9)

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description="HQ-EVFI zip or extracted folder -> <out>/{train,test}/<sequence>/")
ap.add_argument("--src", nargs="+", default=glob.glob(os.path.join(HERE, "*.zip")) or [os.path.join(HERE, "raw")],
                help=f"the zip from {DRIVE}, or the folder it was extracted to")
ap.add_argument("--out", default="data/hqevfi")
ap.add_argument("--work", default=None, help="inner zips are unpacked one at a time into <work>/.tmp (default work: <out>)")
ap.add_argument("--lists", default=LISTS, help="TimeLens-XL dataset_dict.py (url or local path)")
ap.add_argument("--dry", action="store_true", help="only list the sequences and their split")
a = ap.parse_args()
work = os.path.join(a.work or a.out, ".tmp")  # only this subfolder is created and removed
natural = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


class Src:
    # a zip or a folder, read the same way; frames come out as the original bytes (hard link when it is a folder)
    def __init__(self, path):
        self.path, self.z = path, zipfile.ZipFile(path) if os.path.isfile(path) else None
        self.stem = os.path.splitext(os.path.basename(path))[0]
        names = self.z.namelist() if self.z else [os.path.relpath(os.path.join(d, f), path)
                                                  for d, _, fs in os.walk(path, followlinks=True) for f in fs]
        self.names = [n for n in names if "__MACOSX" not in n and not os.path.basename(n).startswith("._")]

    def open(self, name):
        return self.z.open(name) if self.z else open(os.path.join(self.path, name), "rb")

    def read(self, name):
        with self.open(name) as f:
            return f.read()

    def put(self, name, dst):
        if not name.lower().endswith(".png"):
            Image.open(self.open(name)).convert("RGB").save(dst)
        elif self.z:
            with self.z.open(name) as f, open(dst, "wb") as g:
                shutil.copyfileobj(f, g)
        else:
            try:
                os.link(os.path.realpath(os.path.join(self.path, name)), dst)
            except OSError:
                shutil.copy(os.path.join(self.path, name), dst)


def load(S, f):
    e = np.load(io.BytesIO(S.read(f)))
    return [np.asarray(e[k]).reshape(-1) for k in "xytp"]


def frame_times(chunks):
    # frame i sits between event file i-1 and i; end frames one median interval out.
    # if event times restart in every file, frames are 1/FPS apart and events are shifted to absolute time
    lo = np.array([c[2].min() if len(c[2]) else np.nan for c in chunks], dtype=np.float64)
    hi = np.array([c[2].max() if len(c[2]) else np.nan for c in chunks], dtype=np.float64)
    i, ok = np.arange(len(chunks)), ~np.isnan(lo)  # empty files: inside interpolated, at the ends one file interval further out
    slope, edge = np.polyfit(i[ok], lo[ok], 1)[0] if ok.sum() > 1 else 0.0, i - np.clip(i, i[ok][0], i[ok][-1])
    lo, hi = np.interp(i, i[ok], lo[ok]) + slope * edge, np.interp(i, i[ok], hi[ok]) + slope * edge
    if np.median(np.diff(lo)) < 0.5 * np.median(hi - lo):
        s = next(s for s in SCALES if np.median(hi - lo) * s <= 1.5 / FPS)
        for k, c in enumerate(chunks):
            c[2] = k / FPS + c[2].astype(np.float64) * s
        return np.arange(len(chunks) + 1) / FPS
    mid = (hi[:-1] + lo[1:]) / 2
    step = np.median(np.diff(mid))
    return np.concatenate([mid[:1] - step, mid, mid[-1:] + step])


def write(dst, S, imgs, efiles):
    chunks = [load(S, f) for f in efiles]
    ts = frame_times(chunks)
    s = next(s for s in SCALES if 5e-4 <= np.median(np.diff(ts)) * s <= 1.0)  # time unit -> seconds
    W, H = Image.open(S.open(imgs[0])).size
    x, y, t, p = (np.concatenate([c[j] for c in chunks]) for j in range(4))
    o = np.argsort(t, kind="stable")
    x, y = np.round(x[o]), np.round(y[o])
    keep = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    os.makedirs(os.path.join(dst, "frames"))
    np.save(os.path.join(dst, "frame_ts.npy"), (ts - ts[0]) * s)
    np.save(os.path.join(dst, "ev_t.npy"), (t[o][keep] - ts[0]).astype(np.float64) * s)
    np.save(os.path.join(dst, "ev_x.npy"), x[keep].astype(np.int16))
    np.save(os.path.join(dst, "ev_y.npy"), y[keep].astype(np.int16))
    np.save(os.path.join(dst, "ev_p.npy"), np.where(p[o][keep] > 0, 1, -1).astype(np.int8))
    for k, f in enumerate(imgs):
        S.put(f, os.path.join(dst, "frames", f"{k:06d}.png"))
    open(os.path.join(dst, "done"), "w").close()
    print(f"  -> {dst}: {len(imgs)} frames {W}x{H} @ {1 / np.median(np.diff(ts * s)):.1f} fps, {keep.sum()} events", flush=True)


def keys(name):  # official ranges of a sequence: name, name_1, ...
    return [f"{name}_{j}" if j else name for j in range(len(meta.dataset_dict.get(name, [0, 0])) // 2)]


def done(key):
    return bool(glob.glob(os.path.join(a.out, "*", key, "done")))


def convert(S):
    # sequences stored as folders in this source: <...>/<seq>/visual_RGB/
    roots = sorted({("/" + n).split("/visual_RGB/")[0] for n in S.names if ("/" + n).count("/visual_RGB/")}, key=natural)
    for root in roots:
        name = os.path.basename(root) or S.stem
        if name not in meta.dataset_dict and name not in meta.EVSneg3:
            print(f"  {name}: not in the TimeLens-XL lists, skipped")
            continue
        if all(map(done, keys(name))):
            continue
        files = [n for n in S.names if ("/" + n).startswith(root + "/")]
        shifted = name in meta.EVSneg3 and any("/RGB-EVS_EVSneg3ms/" in "/" + n for n in files)
        evdir = "/RGB-EVS_EVSneg3ms/" if shifted else "/RGB-EVS/"
        imgs = sorted((n for n in files if "/visual_RGB/" in "/" + n and n.lower().endswith((".png", ".jpg", ".jpeg"))), key=natural)
        efiles = sorted((n for n in files if evdir in "/" + n and n.endswith(".npz")), key=natural)
        imgs = imgs[int(shifted):int(shifted) + len(efiles)]
        first = int(os.path.basename(imgs[0]).split("_")[0]) if name in meta.dataset_dict else 0
        r = meta.dataset_dict.get(name, [0, len(imgs)])
        for key, j in zip(keys(name), range(len(r) // 2)):  # official ranges: images [s, e) with their event files
            s, e = r[2 * j] - first, min(r[2 * j + 1] - first, len(imgs))
            dst = os.path.join(a.out, "test" if key in meta.test_key else "train", key)
            if done(key):
                continue
            if a.dry:
                print(f"  {os.path.basename(os.path.dirname(dst))}: {key}, {e - s} frames{', shifted' if shifted else ''}")
                continue
            shutil.rmtree(dst, ignore_errors=True)
            try:
                write(dst, S, imgs[s:e], efiles[s:e - 1])
            except Exception as ex:
                print(f"  ! {dst}: failed ({type(ex).__name__}: {ex})", flush=True)


os.makedirs(a.out, exist_ok=True)
lists = os.path.join(a.out, "dataset_dict.py")
if not os.path.exists(lists):
    shutil.copy(a.lists, lists) if os.path.exists(a.lists) else urllib.request.urlretrieve(a.lists, lists)
meta = types.SimpleNamespace(**runpy.run_path(lists))
json.dump(dict(name="HQ-EVFI", paper="TimeLens-XL (ECCV 2024)", url="https://github.com/OpenImagingLab/TimeLens-XL", download=DRIVE,
               events="Prophesee EVK4-HD, beam splitter, checkerboard-calibrated, pixel-aligned to RGB", frames="RGB, 142 fps",
               cfa=None, splits="official (TimeLens-XL dataset_dict.py: ranges, test_key)", converter="HQ-EVFI/hqevfi.py"),
          open(os.path.join(a.out, "info.json"), "w"), indent=1)
for path in a.src:
    S = Src(path)
    for inner in sorted((n for n in S.names if n.lower().endswith(".zip")), key=natural):  # one zip per sequence
        stem = os.path.splitext(os.path.basename(inner))[0]
        if a.dry:
            print(f"  {stem}: zip (not unpacked in --dry)")
            continue
        if (not S.z and os.path.isdir(os.path.join(path, inner[:-4]))) or (stem in meta.dataset_dict and all(map(done, keys(stem)))):
            continue  # already unpacked next to it (read as a folder below) or already converted
        os.makedirs(work, exist_ok=True)
        tmp = os.path.join(work, os.path.basename(inner))
        with S.open(inner) as f, open(tmp, "wb") as g:
            shutil.copyfileobj(f, g, 1 << 24)
        convert(Src(tmp))
        os.remove(tmp)
    convert(S)
shutil.rmtree(work, ignore_errors=True)
