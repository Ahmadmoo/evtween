import argparse
import glob
import os
import shutil
import subprocess
import numpy as np
from PIL import Image

# BS-ERGB (Time Lens++, CVPR'22): beam splitter, Prophesee Gen4M + FLIR RGB at ~28 fps, 970x625 after alignment.
# Layout: {1_TEST,2_VALIDATION,3_TRAINING}/<seq>/images/*.png and events/*.npz (x, y in 1/32 px, timestamp, polarity);
# events/i.npz holds the events between image i and i+1. Event files TimeLens-XL skips as broken split their sequence.
BAD = {"basket_09": [31, 32, 33, 34], "may29_rooftop_handheld_02": [17, 70],
       "may29_rooftop_handheld_03": [306], "may29_rooftop_handheld_05": [121]}
SPLITS = {"3_TRAINING": "train", "2_VALIDATION": "val", "1_TEST": "test"}
SCALES = (1.0, 1e-3, 1e-6, 1e-9)

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description="BS-ERGB (folders in this directory) -> evtween layout: <out>/{train,val,test}/<sequence>/")
ap.add_argument("--out", default="data/bsergb")
ap.add_argument("--archive", default=None, help="the BS-ERGB zip / tar, extracted once to --raw (default <this folder>/raw)")
ap.add_argument("--raw", default=None, help="folder holding 1_TEST, 2_VALIDATION, 3_TRAINING, searched below it too "
                "(default: this folder)")
ap.add_argument("--copy", action="store_true", help="copy PNG frames instead of symlinking them")
ap.add_argument("--min-frames", type=int, default=8, help="drop pieces shorter than this after splitting at broken files")
a = ap.parse_args()


def extract(z, dst):
    print(f"extracting {os.path.basename(z)} -> {dst}")
    os.makedirs(dst, exist_ok=True)
    try:
        shutil.unpack_archive(z, dst)
    except (shutil.ReadError, ValueError):  # zip64 / large archives python cannot read
        cmd = ["unzip", "-q", "-o", z, "-d", dst] if z.lower().endswith(".zip") else ["tar", "-xf", z, "-C", dst]
        subprocess.run(cmd, check=True)


if a.archive:
    a.raw = a.raw or os.path.join(HERE, "raw")
    os.path.isdir(a.raw) or extract(a.archive, a.raw)
a.raw = a.raw or HERE
found = sorted(glob.glob(os.path.join(a.raw, "**", "3_TRAINING"), recursive=True)) + \
    sorted(glob.glob(os.path.join(a.raw, "**", "1_TEST"), recursive=True))
assert found, f"no 1_TEST / 3_TRAINING folder under {a.raw}: pass --archive or --raw"
a.raw = os.path.dirname(found[0])


def load(f):
    e = np.load(f)
    return [np.asarray(e[k]).reshape(-1) for k in ("x", "y", "timestamp", "polarity")]


def frame_times(chunks):
    # frame i sits between event file i-1 and i; end frames one median interval out
    lo = np.array([c[2].min() if len(c[2]) else np.nan for c in chunks], dtype=np.float64)
    hi = np.array([c[2].max() if len(c[2]) else np.nan for c in chunks], dtype=np.float64)
    i, ok = np.arange(len(chunks)), ~np.isnan(lo)
    lo, hi = np.interp(i, i[ok], lo[ok]), np.interp(i, i[ok], hi[ok])
    mid = (hi[:-1] + lo[1:]) / 2
    step = np.median(np.diff(mid))
    return np.concatenate([mid[:1] - step, mid, mid[-1:] + step])


def write(dst, imgs, chunks):
    ts = frame_times(chunks)
    s = next(s for s in SCALES if 5e-4 <= np.median(np.diff(ts)) * s <= 1.0)  # time unit -> seconds
    W, H = Image.open(imgs[0]).size
    x, y, t, p = (np.concatenate([c[j] for c in chunks]) for j in range(4))
    o = np.argsort(t, kind="stable")
    sub = 32.0 if x.max() > 1.5 * W else 1.0  # coordinates stored in 1/32 px
    x, y = np.round(x[o] / sub), np.round(y[o] / sub)
    keep = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    os.makedirs(os.path.join(dst, "frames"), exist_ok=True)
    np.save(os.path.join(dst, "frame_ts.npy"), ts * s)
    np.save(os.path.join(dst, "ev_t.npy"), t[o][keep].astype(np.float64) * s)
    np.save(os.path.join(dst, "ev_x.npy"), x[keep].astype(np.int16))
    np.save(os.path.join(dst, "ev_y.npy"), y[keep].astype(np.int16))
    np.save(os.path.join(dst, "ev_p.npy"), np.where(p[o][keep] > 0, 1, -1).astype(np.int8))
    for k, f in enumerate(imgs):
        out = os.path.join(dst, "frames", f"{k:06d}.png")
        if a.copy:
            Image.open(f).convert("RGB").save(out)
        else:
            os.path.lexists(out) or os.symlink(os.path.abspath(f), out)
    print(f"  -> {dst}: {len(imgs)} frames {W}x{H} @ {1 / np.median(np.diff(ts * s)):.1f} fps, "
          f"{keep.sum()} events ({100 * (1 - keep.mean()):.2f}% outside), x/y unit 1/{sub:.0f} px")


for folder, split in SPLITS.items():
    seqs = sorted(d for d in glob.glob(os.path.join(a.raw, folder, "*")) if os.path.isdir(os.path.join(d, "events")))
    print(f"{folder}: {len(seqs)} sequences -> {split}")
    for seq in seqs:
        name = os.path.basename(seq)
        imgs = sorted(glob.glob(os.path.join(seq, "images", "*.png")))
        efiles = sorted(glob.glob(os.path.join(seq, "events", "*.npz")))
        n = min(len(imgs), len(efiles) + 1)
        cuts = [-1] + sorted(i for i in BAD.get(name, []) if i < n - 1) + [n - 1]  # a broken file i drops the gap i -> i+1
        for j, (s0, s1) in enumerate(zip(cuts[:-1], cuts[1:])):
            fr = list(range(s0 + 1, s1 + 1))
            if len(fr) < a.min_frames:
                continue
            key = f"{name}_{j}" if len(cuts) > 2 else name
            write(os.path.join(a.out, split, key), [imgs[i] for i in fr], [load(efiles[i]) for i in fr[:-1]])
