import argparse
import glob
import os
import re
import shutil
import subprocess
import numpy as np
import yaml
from PIL import Image

# EDS (Event-aided Direct Sparse Odometry, CVPR'22): beam splitter, Prophesee Gen3.1 640x480 events + FLIR Blackfly S
# 640x480 RGB (up to 75 Hz). The two cameras are NOT pixel-aligned in the release: different lenses, radtan distortion,
# a small rotation. Events stay at their own sensor pixels (the per-pixel likelihood needs real pixels: no resampling,
# no holes, no duplicates); RGB frames are warped into the event camera's own (distorted) geometry:
#   event pixel -> undistort (event cam) -> rotate into the RGB camera -> distort (RGB cam) -> sample RGB.
# Then the largest centred rectangle that the RGB view fully covers is kept (the RGB lens sees a narrower field).
# Calibration: Kalibr camchain used for the dataset recordings (uzh-rpg/bundles-eds, config/data/dual_setup/03_calib);
# cam0 = RGB ("flip: True"), cam1 = events. Whether the stored RGB images still need that flip is decided from the data:
# every candidate flip is scored by how well event counts line up with image edges (--flip to force one).
# Input: the per-sequence archives (.tgz / .tar / .zip) or extracted folders; events from events.h5 (DSEC-style
# events/{x,y,t,p} + t_offset, or x/y/t/p, or xs/ys/ts/ps) or events.txt[.gz] (t x y p per line); frames from an image
# folder; frame times from a *timestamp* text file next to it (or from the frame file names).
# No official split: every 5th sequence (starting at the 3rd) is test.
CALIB = """
cam0:  # RGB
  distortion_coeffs: [-0.36965913545735024, 0.17414034009883844, 0.003915245015812422, 0.003666687416655559]
  intrinsics: [766.536025127154, 767.5749459126396, 291.0503512057777, 227.4060484950132]
  resolution: [640, 480]
cam1:  # events
  T_cn_cnm1:
  - [0.9998964430808897, -0.0020335804041023736, -0.014246672065022661, -0.00011238613157578769]
  - [0.001703024953250547, 0.9997299470300024, -0.023176123864880376, -0.0005981481496958399]
  - [0.014289955220253567, 0.02314946137886846, 0.9996298813149167, -0.004416681577516066]
  - [0.0, 0.0, 0.0, 1.0]
  distortion_coeffs: [-0.09776467241921379, 0.2143738428636279, -0.004710710105172864, -0.004215916089401789]
  intrinsics: [560.8520948927032, 560.6295819972383, 313.00733235019237, 217.32858679842997]
  resolution: [640, 480]
"""
SCALES = (1.0, 1e-3, 1e-6, 1e-9)
FLIPS = ("none", "lr", "ud", "both")
IMG = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
ARCH = (".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tbz2", ".tar.xz")

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description="EDS archives or folders -> evtween layout: <out>/{train,test}/<sequence>/")
ap.add_argument("--out", default="data/eds")
ap.add_argument("--archive", nargs="*", default=None,
                help="sequence archives (.tgz/.tar/.zip) or folders holding them; default: archives in this folder")
ap.add_argument("--raw", default=os.path.join(HERE, "raw"), help="where archives are extracted (once) and folders are searched")
ap.add_argument("--calib", default=None, help="Kalibr camchain yaml (default: the dataset calibration above)")
ap.add_argument("--rgb-cam", default="cam0")
ap.add_argument("--event-cam", default="cam1")
ap.add_argument("--flip", default="auto", choices=("auto",) + FLIPS, help="flip of the stored RGB images before the calibration")
ap.add_argument("--offset-ms", type=float, default=0.0, help="added to frame times (e.g. half the exposure)")
ap.add_argument("--test-every", type=int, default=5)
ap.add_argument("--only", nargs="*", default=None, help="convert only sequences whose name contains one of these")
ap.add_argument("--dry", action="store_true", help="only list what is found in each sequence (files, counts, time spans, units)")
a = ap.parse_args()
natural = lambda f: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", os.path.basename(f))]


def extract(z, dst):
    print(f"extracting {os.path.basename(z)} -> {dst}")
    os.makedirs(dst, exist_ok=True)
    try:
        shutil.unpack_archive(z, dst)
    except (shutil.ReadError, ValueError):  # zip64 / unknown suffix: let the system tools try
        cmd = ["unzip", "-q", "-o", z, "-d", dst] if z.lower().endswith(".zip") else ["tar", "-xf", z, "-C", dst]
        subprocess.run(cmd, check=True)


def stem(f):
    b = os.path.basename(f)
    return next((b[:-len(s)] for s in ARCH if b.lower().endswith(s)), b)


# ---------------------------------------------------------------- reading
def read_events(seq):
    h5 = sorted(f for f in glob.glob(os.path.join(seq, "**", "*.h5"), recursive=True) + glob.glob(os.path.join(seq, "**", "*.hdf5"), recursive=True)
                if "event" in os.path.basename(f).lower()) or sorted(glob.glob(os.path.join(seq, "**", "*.h5"), recursive=True))
    if h5:
        try:
            import hdf5plugin  # noqa: F401  (blosc-compressed event files)
        except ImportError:
            pass
        import h5py
        with h5py.File(h5[0], "r") as f:
            for keys in (("events/x", "events/y", "events/t", "events/p"), ("x", "y", "t", "p"), ("xs", "ys", "ts", "ps"),
                         ("events/xs", "events/ys", "events/ts", "events/ps")):
                if all(k in f for k in keys):
                    try:
                        x, y, t, p = (f[k][()] for k in keys)
                    except OSError as e:
                        raise OSError(f"{e} (blosc-compressed? pip install hdf5plugin)") from e
                    off = f["t_offset"][()] if "t_offset" in f else 0
                    return x, y, t.astype(np.float64) + float(off), p, os.path.relpath(h5[0], seq)
            raise KeyError(f"{h5[0]}: no x/y/t/p datasets, has {list(f.keys())}")
    txt = sorted(f for f in glob.glob(os.path.join(seq, "**", "*.txt*"), recursive=True) if os.path.basename(f).lower().startswith("events"))
    if txt:
        import gzip
        op = gzip.open if txt[0].endswith(".gz") else open
        with op(txt[0], "rt") as f:
            body = "".join(l for l in f if l.lstrip()[:1] in "0123456789.-+")  # drop header / comment lines
        v = np.array(body.split(), dtype=np.float64).reshape(-1, 4)  # t x y p
        return v[:, 1], v[:, 2], v[:, 0], v[:, 3], os.path.relpath(txt[0], seq)
    raise FileNotFoundError("no events.h5 / events.txt")


def find_images(seq):
    dirs = {}
    for f in glob.glob(os.path.join(seq, "**", "*"), recursive=True):
        if f.lower().endswith(IMG):
            dirs.setdefault(os.path.dirname(f), []).append(f)
    if not dirs:
        raise FileNotFoundError("no image folder")
    # prefer a folder called images / rgb / frames, else the one with most images
    d = max(dirs, key=lambda k: (any(n in os.path.basename(k).lower() for n in ("image", "rgb", "frame")), len(dirs[k])))
    return sorted(dirs[d], key=natural)


def read_times(seq, imgs):
    n = len(imgs)
    cands = sorted(glob.glob(os.path.join(seq, "**", "*.txt"), recursive=True) + glob.glob(os.path.join(seq, "**", "*.csv"), recursive=True))
    cands = [f for f in cands if any(k in os.path.basename(f).lower() for k in ("timestamp", "times", "stamps"))
             and not os.path.basename(f).lower().startswith("event")]
    cands.sort(key=lambda f: ("image" not in os.path.basename(f).lower(), len(f)))
    for f in cands:
        num = []
        for l in open(f):
            if l.lstrip().startswith(("#", "%")):
                continue
            v = []
            for tok in re.split(r"[\s,;]+", l.strip()):
                try:
                    v.append(float(tok))
                except ValueError:
                    pass  # file names, header words
            if v:
                num.append(v)
        width = min((len(v) for v in num), default=0)
        if width == 0 or abs(len(num) - n) > 1:
            continue
        m = np.array([v[:width] for v in num], dtype=np.float64)
        col = int(np.argmax(np.median(np.abs(m), 0)))  # the time column has the largest values (index columns are small)
        ts = m[:min(len(m), n), col]
        if np.all(np.diff(ts) > 0):
            return ts, os.path.relpath(f, seq)
    digits = [re.findall(r"\d+", os.path.basename(f)) for f in imgs]
    if all(digits):
        ts = np.array([float(d[-1]) for d in digits])
        if np.all(np.diff(ts) > 0) and np.median(np.diff(ts)) > 100:  # numbers that are times, not 0,1,2,...
            return ts, "file names"
    raise FileNotFoundError("no frame timestamps (a *timestamp*.txt next to the images, or times in the file names)")


# ---------------------------------------------------------------- geometry
def radtan(xn, yn, k):
    k1, k2, p1, p2 = k
    r2 = xn * xn + yn * yn
    rad = 1 + k1 * r2 + k2 * r2 * r2
    return xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn), yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn


def undistort(xd, yd, k, iters=30):
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):  # fixed point: x = xd - (distort(x) - x)
        dx, dy = radtan(x, y, k)
        x, y = xd - (dx - x), yd - (dy - y)
    return x, y


def rgb_map(cal, W, H):
    # for every event pixel: where it is in the stored RGB image (float px)
    e, r = cal[a.event_cam], cal[a.rgb_cam]
    fx, fy, cx, cy = e["intrinsics"]
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    xn, yn = undistort((u - cx) / fx, (v - cy) / fy, e["distortion_coeffs"])
    T = np.array(e.get("T_cn_cnm1", np.eye(4)), dtype=np.float64)  # RGB (cam n-1) -> events (cam n)
    R = T[:3, :3].T                                                 # events -> RGB; translation (mm, beam splitter) ignored
    X = np.einsum("ij,jhw->ihw", R, np.stack([xn, yn, np.ones_like(xn)]))
    xd, yd = radtan(X[0] / X[2], X[1] / X[2], r["distortion_coeffs"])
    fx, fy, cx, cy = r["intrinsics"]
    return fx * xd + cx, fy * yd + cy


def flipped(img, flip):
    if flip in ("lr", "both"):
        img = img[:, ::-1]
    if flip in ("ud", "both"):
        img = img[::-1]
    return img


def sample(img, mx, my):
    # bilinear lookup of img (H, W, C) at float coordinates
    H, W = img.shape[:2]
    x0, y0 = np.clip(np.floor(mx).astype(np.int64), 0, W - 2), np.clip(np.floor(my).astype(np.int64), 0, H - 2)
    fx, fy = (mx - x0)[..., None], (my - y0)[..., None]
    im = img.astype(np.float32)
    out = (im[y0, x0] * (1 - fx) * (1 - fy) + im[y0, x0 + 1] * fx * (1 - fy)
           + im[y0 + 1, x0] * (1 - fx) * fy + im[y0 + 1, x0 + 1] * fx * fy)
    return out.round().clip(0, 255).astype(np.uint8)


def valid_box(ok):
    # largest centred rectangle inside the valid mask (shrink all sides together, then each side alone)
    H, W = ok.shape
    box = [0, 0, H, W]
    full = lambda b: b[2] > b[0] and b[3] > b[1] and ok[b[0]:b[2], b[1]:b[3]].all()
    while not full(box) and box[2] - box[0] > 2 and box[3] - box[1] > 2:
        box = [box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1]
    for side, step in ((0, -1), (1, -1), (2, 1), (3, 1)):
        while True:
            b = list(box)
            b[side] += step
            if b[0] < 0 or b[1] < 0 or b[2] > H or b[3] > W or not full(b):
                break
            box = b
    return box


def rgb(f):
    im = np.asarray(Image.open(f))
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, 2)
    if im.dtype == np.uint16:
        im = (im >> 8).astype(np.uint8)
    return im[..., :3]


def edges(img):
    y = np.log(img.astype(np.float32).mean(-1) + 4)
    gx, gy = np.zeros_like(y), np.zeros_like(y)
    gx[:, 1:-1], gy[1:-1] = y[:, 2:] - y[:, :-2], y[2:] - y[:-2]
    return np.hypot(gx, gy)


def blur(m, r=2):
    c = np.cumsum(np.cumsum(np.pad(m, ((r + 1, r), (r + 1, r))), 0), 1)
    k = 2 * r + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / k ** 2


def score(counts, grads, shift=(0, 0)):
    # correlation of event counts and image edges in the inner region, events shifted by (dy, dx)
    dy, dx = shift
    m = 8
    c = [blur(np.roll(q, (dy, dx), (0, 1)))[m:-m, m:-m].ravel() for q in counts]
    g = [blur(q)[m:-m, m:-m].ravel() for q in grads]
    return float(np.mean([np.corrcoef(u, v)[0, 1] for u, v in zip(c, g) if u.std() > 0 and v.std() > 0] or [0.0]))


# ---------------------------------------------------------------- main
cal = yaml.safe_load(open(a.calib)) if a.calib else yaml.safe_load(CALIB)
is_event_file = lambda f: os.path.basename(f).lower().startswith("events") and f.lower().endswith((".h5", ".hdf5", ".txt", ".txt.gz"))
extracted = any(is_event_file(f) for f in glob.glob(os.path.join(a.raw, "**", "events*"), recursive=True))
archives = []
# without --archive, archives in this folder are used only while --raw holds no extracted sequences yet
for p in a.archive if a.archive is not None else ([] if extracted else [HERE]):
    if os.path.isdir(p):
        archives += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(ARCH))
    elif os.path.isfile(p):
        archives.append(p)
    else:
        raise FileNotFoundError(p)
for z in archives:
    dst = os.path.join(a.raw, stem(z))
    if os.path.isdir(dst):
        continue
    if a.dry:  # --dry never writes anything
        print(f"would extract {z} -> {dst}")
    else:
        extract(z, dst)
if not extracted and not a.dry:  # archives inside the freshly extracted ones (e.g. images.zip)
    for z in sorted(glob.glob(os.path.join(a.raw, "**", "*"), recursive=True)):
        if z.lower().endswith(ARCH) and not os.path.isdir(os.path.join(os.path.dirname(z), stem(z))):
            extract(z, os.path.join(os.path.dirname(z), stem(z)))

# a sequence = the folder holding an event file (its images may sit in a subfolder)
evfiles = [f for f in glob.glob(os.path.join(a.raw, "**", "*"), recursive=True)
           if os.path.basename(f).lower().startswith("events") and f.lower().endswith((".h5", ".hdf5", ".txt", ".txt.gz"))]
seqs = sorted({os.path.dirname(f) for f in evfiles}, key=natural)
test = set([os.path.basename(s) for s in seqs][2::a.test_every])  # split from all sequences, so --only does not change it
seqs = [s for s in seqs if not a.only or any(o in os.path.basename(s) for o in a.only)]
print(f"{len(seqs)} sequences in {a.raw}")
names = [os.path.basename(s) for s in seqs]

for seq, name in zip(seqs, names):
    try:
        imgs = find_images(seq)
        ts, tsrc = read_times(seq, imgs)
        x, y, t, p, esrc = read_events(seq)
    except Exception as e:
        print(f"{name}: skipped ({type(e).__name__}: {e})")
        continue
    imgs = imgs[:len(ts)]
    s = next((s for s in SCALES if 2e-3 <= np.median(np.diff(ts)) * s <= 0.5), None)  # 2-500 Hz
    if s is None:
        print(f"{name}: skipped (frame interval {np.median(np.diff(ts)):.3g} in no known time unit)")
        continue
    ts = ts * s
    # event time unit: same clock as the frames; pick the unit that makes the two time spans overlap best
    def overlap(se):
        lo, hi = max(t[0] * se, ts[0]), min(t[-1] * se, ts[-1])
        return (hi - lo) / (ts[-1] - ts[0])
    se = max(SCALES, key=overlap)
    W, H = Image.open(imgs[0]).size
    print(f"{name}: {len(imgs)} frames {W}x{H} @ {1 / np.median(np.diff(ts)):.1f} fps (times: {tsrc}, unit {s:g} s), "
          f"{len(t)} events (from {esrc}, unit {se:g} s), x {int(x.min())}-{int(x.max())} y {int(y.min())}-{int(y.max())}, "
          f"time overlap {100 * overlap(se):.0f}%"
          + f" -> {'test' if name in test else 'train'}")
    if a.dry:
        continue
    if overlap(se) < 0.5:
        print(f"  skipped: frames and events do not share a clock (overlap {100 * overlap(se):.0f}%)")
        continue

    t = t.astype(np.float64) * se
    if np.any(np.diff(t) < 0):
        o = np.argsort(t, kind="stable")
        x, y, t, p = x[o], y[o], t[o], p[o]
    ts = ts + a.offset_ms * 1e-3
    keep_f = (ts >= t[0]) & (ts <= t[-1])
    imgs, ts = [f for f, k in zip(imgs, keep_f) if k], ts[keep_f]
    eW, eH = cal[a.event_cam]["resolution"]
    mx, my = rgb_map(cal, eW, eH)
    rW, rH = cal[a.rgb_cam]["resolution"]
    if (W, H) != (rW, rH):  # stored images at another size than calibrated: scale the map
        mx, my = (mx + 0.5) * W / rW - 0.5, (my + 0.5) * H / rH - 0.5
    y0, x0, y1, x1 = valid_box((mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1))
    mx, my = mx[y0:y1, x0:x1], my[y0:y1, x0:x1]

    # flip of the stored RGB: events vs edges on a few frames spread over the sequence
    probe = np.linspace(1, len(imgs) - 2, min(8, len(imgs) - 2)).astype(int)
    dt = np.median(np.diff(ts))
    counts = []
    for k in probe:
        a_, b_ = np.searchsorted(t, [ts[k] - dt / 2, ts[k] + dt / 2])
        xe, ye = x[a_:b_].astype(np.int64) - x0, y[a_:b_].astype(np.int64) - y0
        m = (xe >= 0) & (xe < x1 - x0) & (ye >= 0) & (ye < y1 - y0)
        counts.append(np.bincount(ye[m] * (x1 - x0) + xe[m], minlength=(y1 - y0) * (x1 - x0)).reshape(y1 - y0, x1 - x0).astype(np.float32))
    frames = {k: rgb(imgs[k]) for k in probe}
    scores = {f: score(counts, [edges(sample(flipped(frames[k], f), mx, my)) for k in probe]) for f in FLIPS}
    flip = max(scores, key=scores.get) if a.flip == "auto" else a.flip
    grads = [edges(sample(flipped(frames[k], flip), mx, my)) for k in probe]
    shifts = {(dy, dx): score(counts, grads, (dy, dx)) for dy in range(-3, 4) for dx in range(-3, 4)}
    best = max(shifts, key=shifts.get)
    print(f"  alignment: flip {flip} (" + ", ".join(f"{f} {v:.3f}" for f, v in scores.items())
          + f"), best residual shift dy,dx = {best} ({shifts[best]:.3f} vs {shifts[(0, 0)]:.3f} at 0,0)"
          + ("" if best == (0, 0) else " ! check the calibration")
          + ("" if shifts[(0, 0)] > 0.1 else " ! weak match between events and edges: check the frames / calibration"))

    dst = os.path.join(a.out, "test" if name in test else "train", name)
    os.makedirs(os.path.join(dst, "frames"), exist_ok=True)
    for k, f in enumerate(imgs):
        Image.fromarray(sample(flipped(rgb(f), flip), mx, my)).save(os.path.join(dst, "frames", f"{k:06d}.png"))
    keep = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    t0 = ts[0]
    np.save(os.path.join(dst, "frame_ts.npy"), ts - t0)
    np.save(os.path.join(dst, "ev_t.npy"), t[keep] - t0)
    np.save(os.path.join(dst, "ev_x.npy"), (x[keep] - x0).astype(np.int16))
    np.save(os.path.join(dst, "ev_y.npy"), (y[keep] - y0).astype(np.int16))
    np.save(os.path.join(dst, "ev_p.npy"), np.where(p[keep] > 0, 1, -1).astype(np.int8))
    with open(os.path.join(dst, "align.yaml"), "w") as f:
        yaml.safe_dump(dict(flip=flip, flip_scores=scores, crop_yxyx=[int(v) for v in (y0, x0, y1, x1)],
                            residual_shift=list(best), score=shifts[(0, 0)]), f)
    print(f"  -> {dst}: {len(imgs)} frames {x1 - x0}x{y1 - y0}, {keep.sum()} events")
