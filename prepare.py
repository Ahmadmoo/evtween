import argparse
import glob
import importlib.util
import os
import re
import numpy as np
from PIL import Image

# one image per frame + one event file per frame interval, folder and key names as in the official loaders
PRESETS = {
    "hsergb": dict(images="images_corrected", events="events_aligned", keys=("x", "y", "t", "p"), xy_scale=1),
    "bsergb": dict(images="images", events="events", keys=("x", "y", "timestamp", "polarity"), xy_scale=32),
    "erf": dict(images="processed_images", events="processed_events", keys=("x", "y", "t", "p"), xy_scale=128),
    "hqevfi": dict(images="visual_RGB", events="RGB-EVS", keys=("x", "y", "t", "p"), xy_scale=1,
                   alt=("RGB-EVS_EVSneg3ms", 1)),  # TimeLens-XL: 3 ms corrected events, images shifted by one
}
# BS-ERGB event files the TimeLens-XL loader skips (dataset/BSERGBloader/loader_bsergb.py); sequences are cut there
BSERGB_BAD = {"basket_09": [31, 32, 33, 34], "may29_rooftop_handheld_02": [17, 70],
              "may29_rooftop_handheld_03": [306], "may29_rooftop_handheld_05": [121]}
SCALES = (1.0, 1e-3, 1e-6, 1e-9)
natural = lambda f: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", os.path.basename(f))]

ap = argparse.ArgumentParser(description="raw RGB+event dataset -> evtween layout (out/<split>/<sequence>/)")
ap.add_argument("dataset", choices=PRESETS)
ap.add_argument("--raw", required=True, help="extracted dataset folder")
ap.add_argument("--out", required=True)
ap.add_argument("--timelensxl", help="HQ-EVFI only: clone of OpenImagingLab/TimeLens-XL, for its official ranges and test split")
ap.add_argument("--fps", type=float, help="only if event times restart in every file: frame rate of the RGB camera")
ap.add_argument("--copy", action="store_true", help="copy PNG frames instead of symlinking them")
a = ap.parse_args()
P = PRESETS[a.dataset]


def chunk(f):
    try:
        e = np.load(f)
        x, y, t, p = (np.asarray(e[k]).reshape(-1) for k in P["keys"])
        return x / P["xy_scale"], y / P["xy_scale"], t.astype(np.float64), p
    except Exception as err:
        print(f"  unreadable {f}: {err}")
        return None


def frame_times(seq, chunks):
    # timestamp file if the dataset has one, else the boundaries between consecutive event files
    txt = sorted(glob.glob(os.path.join(seq, P["images"], "*timestamp*.txt")) + glob.glob(os.path.join(seq, "*timestamp*.txt")))
    if txt:
        return np.loadtxt(txt[0]).reshape(-1)[:len(chunks) + 1]
    lo = np.array([c[2].min() if c is not None and len(c[2]) else np.nan for c in chunks])
    hi = np.array([c[2].max() if c is not None and len(c[2]) else np.nan for c in chunks])
    i, ok = np.arange(len(chunks)), ~np.isnan(lo)
    lo, hi = np.interp(i, i[ok], lo[ok]), np.interp(i, i[ok], hi[ok])
    if len(lo) > 1 and np.median(np.diff(lo)) < 0.5 * np.median(hi - lo):  # event times restart in every file
        assert a.fps, "event times restart in every file: pass --fps"
        s = next(s for s in SCALES if np.median(hi - lo) * s <= 1.5 / a.fps)
        for k, c in enumerate(chunks):
            if c is not None:
                chunks[k] = (c[0], c[1], k / a.fps + c[2] * s, c[3])
        return np.arange(len(chunks) + 1) / a.fps
    mid = (hi[:-1] + lo[1:]) / 2
    step = np.median(np.diff(mid)) if len(mid) > 1 else hi[0] - lo[0]
    ts = np.concatenate([mid[:1] - step, mid, mid[-1:] + step])  # end frames: one median interval out
    return ts


def write(dst, imgs, ts, chunks):
    # one continuous piece: frames imgs[0..n], events between them, everything in seconds
    s_ts = next(s for s in SCALES if 5e-4 <= np.median(np.diff(ts)) * s <= 1.0)
    t = np.concatenate([c[2] for c in chunks])
    tol = 2 * np.median(np.diff(ts)) * s_ts
    s_ev = s_ts if len(t) == 0 else max(SCALES, key=lambda s: np.mean((t[::97] * s >= ts[0] * s_ts - tol) & (t[::97] * s <= ts[-1] * s_ts + tol)))
    W, H = Image.open(imgs[0]).size
    x, y, p = (np.round(np.concatenate([c[i] for c in chunks])) for i in (0, 1, 3))
    o = np.argsort(t, kind="stable")
    keep = (x[o] >= 0) & (x[o] < W) & (y[o] >= 0) & (y[o] < H)
    o = o[keep]
    os.makedirs(os.path.join(dst, "frames"), exist_ok=True)
    np.save(os.path.join(dst, "frame_ts.npy"), ts * s_ts)
    np.save(os.path.join(dst, "ev_t.npy"), t[o] * s_ev)
    np.save(os.path.join(dst, "ev_x.npy"), x[o].astype(np.int16))
    np.save(os.path.join(dst, "ev_y.npy"), y[o].astype(np.int16))
    np.save(os.path.join(dst, "ev_p.npy"), np.where(p[o] > 0, 1, -1).astype(np.int8))
    for i, f in enumerate(imgs):
        out = os.path.join(dst, "frames", f"{i:06d}.png")
        if f.lower().endswith(".png") and not a.copy:
            os.path.lexists(out) or os.symlink(os.path.abspath(f), out)
        else:
            Image.open(f).convert("RGB").save(out)
    fps = 1 / np.median(np.diff(ts * s_ts))
    print(f"  -> {dst}: {len(imgs)} frames {W}x{H} @ {fps:.1f} fps, {len(o)} events ({keep.mean():.1%} in frame), "
          f"time units: frames x{s_ts:g}, events x{s_ev:g}")


meta = None
if a.dataset == "hqevfi" and a.timelensxl:
    spec = importlib.util.spec_from_file_location("dd", os.path.join(a.timelensxl, "dataset/RC_4816/dataset_dict.py"))
    meta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(meta)

ev_dirs = [P["events"], P["alt"][0]] if "alt" in P else [P["events"]]
seqs = sorted(d for d, subs, _ in os.walk(a.raw) if P["images"] in subs and any(e in subs for e in ev_dirs))
print(f"{len(seqs)} sequences in {a.raw}")
for seq in seqs:
    name, rel = os.path.basename(seq), os.path.relpath(seq, a.raw).lower()
    ev_dir, offset = P["alt"] if "alt" in P and os.path.isdir(os.path.join(seq, P["alt"][0])) else (P["events"], 0)
    imgs = sorted((f for f in glob.glob(os.path.join(seq, P["images"], "*")) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))), key=natural)
    efiles = sorted(glob.glob(os.path.join(seq, ev_dir, "*.npz")), key=natural)
    imgs = imgs[offset:offset + len(efiles) + 1]
    efiles = efiles[:len(imgs) - 1]
    print(f"{name}: {len(imgs)} images, {len(efiles)} event files")
    chunks = [chunk(f) for f in efiles]
    for k in BSERGB_BAD.get(name, []) if a.dataset == "bsergb" else []:
        if k < len(chunks):
            chunks[k] = None
    ts = frame_times(seq, chunks)
    imgs = imgs[:len(ts)]

    if meta is not None:  # official HQ-EVFI ranges (absolute frame numbers) and test keys
        if name not in meta.dataset_dict and name not in meta.EVSneg3:
            print("  not in the TimeLens-XL lists, skipped")
            continue
        first = int(os.path.basename(imgs[0]).split("_")[0]) if name in meta.dataset_dict else 0
        r = meta.dataset_dict.get(name, [first, first + len(imgs)])
        pieces = [(f"{name}_{i}" if i else name, r[2 * i] - first, r[2 * i + 1] - first) for i in range(len(r) // 2)]
        pieces = [(key, "test" if key in meta.test_key else "train", s, e) for key, s, e in pieces]
    else:
        split = "test" if "test" in rel else "val" if "valid" in rel else "train"
        prefix = "close_" if "close" in rel.split(os.sep) else "far_" if "far" in rel.split(os.sep) else ""
        pieces = [(prefix + name, split, 0, len(imgs))]

    for key, split, s, e in pieces:
        e = min(e, len(imgs))
        good = [k for k in range(s, e - 1) if chunks[k] is not None]  # cut where an event file is missing
        runs = np.split(np.array(good), np.where(np.diff(good) > 1)[0] + 1) if good else []
        for j, run in enumerate(r for r in runs if len(r) >= 8):
            sub = key if len(runs) == 1 else f"{key}_part{j}"
            write(os.path.join(a.out, split, sub), imgs[run[0]:run[-1] + 2], ts[run[0]:run[-1] + 2], [chunks[k] for k in run])
