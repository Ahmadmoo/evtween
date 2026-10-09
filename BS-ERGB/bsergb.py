import argparse
import glob
import io
import json
import os
import re
import shutil
import zipfile
import numpy as np
from PIL import Image

# BS-ERGB (Time Lens++, CVPR'22): beam splitter, Prophesee Gen4M events + RGB frames, 970x625 after alignment, ~20-28 fps.
# Input: the downloaded zip (or the folder it was extracted to), read as it is: nothing is unpacked to disk.
# Inside: {1_TEST,2_VALIDATION,3_TRAINING}/<seq>/images/*.png and events/*.npz (x, y in 1/32 px, timestamp in us, polarity).
# events/i.npz holds the events between image i and i+1, so frame times come from the boundaries between event files.
# Event files that TimeLens-XL skips as broken split their sequence into pieces. Output: official splits train / val / test.
URL = "https://github.com/uzh-rpg/timelens-pp"
BAD = {"basket_09": [31, 32, 33, 34], "may29_rooftop_handheld_02": [17, 70],
       "may29_rooftop_handheld_03": [306], "may29_rooftop_handheld_05": [121]}
SPLITS = {"3_TRAINING": "train", "2_VALIDATION": "val", "1_TEST": "test"}
SCALES = (1.0, 1e-3, 1e-6, 1e-9)
STEP = 50_000_000  # events per piece when the full array would not fit in memory

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description="BS-ERGB zip or extracted folder -> <out>/{train,val,test}/<sequence>/")
ap.add_argument("--src", nargs="+", default=glob.glob(os.path.join(HERE, "*.zip")) or [HERE], help="zip file(s) or folder(s)")
ap.add_argument("--out", default="data/bsergb")
ap.add_argument("--min-frames", type=int, default=8, help="drop pieces shorter than this after splitting at broken files")
ap.add_argument("--dry", action="store_true", help="only list the sequences and their split")
a = ap.parse_args()
natural = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


class Src:
    # a zip or a folder, read the same way; frames come out as the original bytes (hard link when it is a folder)
    def __init__(self, path):
        self.path, self.z = path, zipfile.ZipFile(path) if os.path.isfile(path) else None
        names = self.z.namelist() if self.z else [os.path.relpath(os.path.join(d, f), path)
                                                  for d, _, fs in os.walk(path, followlinks=True) for f in fs]
        self.names = [n for n in names if "__MACOSX" not in n and not os.path.basename(n).startswith("._")]

    def open(self, name):
        return self.z.open(name) if self.z else open(os.path.join(self.path, name), "rb")

    def read(self, name):
        with self.open(name) as f:
            return f.read()

    def put(self, name, dst):
        if self.z:
            with self.z.open(name) as f, open(dst, "wb") as g:
                shutil.copyfileobj(f, g)
            return
        try:
            os.link(os.path.realpath(os.path.join(self.path, name)), dst)
        except OSError:
            shutil.copy(os.path.join(self.path, name), dst)


class Npy:
    # .npy written piece by piece (a sequence can hold 1e9 events); the header gets its final length at the end
    def __init__(self, path, dtype):
        self.f, self.dtype, self.n = open(path, "wb"), np.dtype(dtype), 0
        self.f.write(b"\0" * 128)

    def add(self, v):
        self.f.write(np.ascontiguousarray(v, self.dtype).tobytes())
        self.n += len(v)

    def close(self):
        d = "{'descr': '%s', 'fortran_order': False, 'shape': (%d,), }" % (self.dtype.str, self.n)
        self.f.seek(0)
        self.f.write(b"\x93NUMPY\x01\x00" + (118).to_bytes(2, "little") + d.ljust(117).encode() + b"\n")
        self.f.close()


def write(dst, S, imgs, efiles):
    os.makedirs(os.path.join(dst, "frames"))
    W, H = Image.open(S.open(imgs[0])).size
    load = lambda f: np.load(io.BytesIO(S.read(f)))
    pick = lambda e, *keys: next(np.asarray(e[k]).reshape(-1) for k in keys if k in e)
    sub = 32.0 if max(load(f)["x"].max(initial=0) for f in efiles[:10]) > 1.5 * W else 1.0  # x, y stored in 1/32 px
    out = {k: Npy(os.path.join(dst, f"ev_{k}.npy"), dt) for k, dt in zip("txyp", (np.float64, np.int16, np.int16, np.int8))}
    lo, hi, base, last, ordered, n_all = [], [], None, -np.inf, True, 0
    for f in efiles:
        e = load(f)
        x, y, t, p = pick(e, "x"), pick(e, "y"), pick(e, "timestamp", "t"), pick(e, "polarity", "p")
        if np.any(np.diff(t) < 0):
            o = np.argsort(t, kind="stable")
            x, y, t, p = x[o], y[o], t[o], p[o]
        base = t[0] if base is None and len(t) else base
        t = (t - base).astype(np.float64) if len(t) else t.astype(np.float64)  # whole raw units: exact in float64
        lo.append(t[0] if len(t) else np.nan)
        hi.append(t[-1] if len(t) else np.nan)
        ordered, last, n_all = ordered and (not len(t) or t[0] >= last), max(last, t[-1] if len(t) else last), n_all + len(t)
        x, y = np.round(x / sub), np.round(y / sub)
        keep = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        for k, v in zip("txyp", (t, x, y, np.where(p > 0, 1, -1))):
            out[k].add(v[keep])
    for v in out.values():
        v.close()
    if not ordered:  # event files overlap in time: sort the whole sequence once
        o = np.argsort(np.load(os.path.join(dst, "ev_t.npy")), kind="stable")
        for k in "txyp":
            np.save(os.path.join(dst, f"ev_{k}.npy"), np.load(os.path.join(dst, f"ev_{k}.npy"))[o])
    # frame i sits between event file i-1 and i; the end frames one median interval out
    i, lo, hi = np.arange(len(lo)), np.array(lo), np.array(hi)
    ok = ~np.isnan(lo)  # empty event files: inside a piece interpolated, at its ends one file interval further out each
    slope, edge = np.polyfit(i[ok], lo[ok], 1)[0] if ok.sum() > 1 else 0.0, i - np.clip(i, i[ok][0], i[ok][-1])
    lo, hi = np.interp(i, i[ok], lo[ok]) + slope * edge, np.interp(i, i[ok], hi[ok]) + slope * edge
    mid = (hi[:-1] + lo[1:]) / 2
    step = np.median(np.diff(mid))
    ts = np.concatenate([mid[:1] - step, mid, mid[-1:] + step])
    s = next(s for s in SCALES if 5e-4 <= np.median(np.diff(ts)) * s <= 1.0)  # time unit -> seconds
    np.save(os.path.join(dst, "frame_ts.npy"), (ts - ts[0]) * s)
    t = np.load(os.path.join(dst, "ev_t.npy"), mmap_mode="r+")
    for j in range(0, len(t), STEP):
        t[j:j + STEP] = (t[j:j + STEP] - ts[0]) * s
    t.flush()
    for k, f in enumerate(imgs):
        S.put(f, os.path.join(dst, "frames", f"{k:06d}.png"))
    open(os.path.join(dst, "done"), "w").close()
    print(f"  -> {dst}: {len(imgs)} frames {W}x{H} @ {1 / np.median(np.diff(ts * s)):.1f} fps, {len(t)} events "
          f"({100 * (1 - len(t) / max(n_all, 1)):.2f}% outside), x/y unit 1/{sub:.0f} px", flush=True)


groups = {}
for S in map(Src, a.src):
    for n in S.names:
        m = re.search(r"(1_TEST|2_VALIDATION|3_TRAINING)/([^/]+)/(images|events)/[^/]+\.(png|npz)$", n)
        key = m and (SPLITS[m[1]], m[2])
        if m and groups.setdefault(key, (S, {"images": [], "events": []}))[0] is S:
            groups[key][1][m[3]].append(n)
os.makedirs(a.out, exist_ok=True)
json.dump(dict(name="BS-ERGB", paper="Time Lens++ (CVPR 2022)", url=URL, events="Prophesee Gen4M, beam splitter, aligned to RGB",
               frames="RGB, ~20-28 fps", cfa=None, splits="official (3_TRAINING / 2_VALIDATION / 1_TEST)",
               license="Time Lens++ Evaluation License: non-commercial internal evaluation, no derivatives",
               converter="BS-ERGB/bsergb.py"), open(os.path.join(a.out, "info.json"), "w"), indent=1)
print(f"{len(groups)} sequences: " + ", ".join(f"{sum(k[0] == s for k in groups)} {s}" for s in SPLITS.values()))
for (split, name), (S, f) in sorted(groups.items()):
    imgs, efiles = sorted(f["images"], key=natural), sorted(f["events"], key=natural)
    n = min(len(imgs), len(efiles) + 1)
    cuts = [-1] + sorted(i for i in BAD.get(name, []) if i < n - 1) + [n - 1]  # a broken file i drops the gap i -> i+1
    for j, (s0, s1) in enumerate(zip(cuts[:-1], cuts[1:])):
        fr = list(range(s0 + 1, s1 + 1))
        dst = os.path.join(a.out, split, f"{name}_{j}" if len(cuts) > 2 else name)
        if len(fr) < a.min_frames or os.path.exists(os.path.join(dst, "done")):
            continue
        if a.dry:
            print(f"  {split}: {os.path.basename(dst)}, {len(fr)} frames")
            continue
        shutil.rmtree(dst, ignore_errors=True)
        try:
            write(dst, S, [imgs[i] for i in fr], [efiles[i] for i in fr[:-1]])
        except Exception as e:
            print(f"  ! {dst}: failed ({type(e).__name__}: {e})", flush=True)
