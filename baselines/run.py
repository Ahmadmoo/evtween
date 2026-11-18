import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from fetch import FILM, ROOT, SOURCES, WEIGHTS, digest

EVENT = np.dtype([("t", "<f8"), ("x", "<i2"), ("y", "<i2"), ("p", "i1")])


class Sequence:
    def __init__(self, path, stride):
        files = sorted((path / "frames").glob("*.png"))
        t = np.load(path / "frame_ts.npy", allow_pickle=False).astype(np.float64)
        if t.shape != (len(files),) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("Expected one strictly increasing timestamp in seconds per PNG.")
        self.idx = np.arange(0, len(files), stride)
        if len(self.idx) < 2:
            raise ValueError("Need at least two frames after striding.")
        self.files = [files[i] for i in self.idx]
        self.ts = t[self.idx]
        self.time = self.ts - self.ts[0]
        self.h, self.w = cv2.imread(str(self.files[0]), cv2.IMREAD_UNCHANGED).shape[:2]
        if max(self.w, self.h) > 32767:
            raise ValueError("The pixel coordinates exceed int16.")

    def read(self, i, gray=False):
        mode = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
        img = cv2.imread(str(self.files[i]), mode | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None or img.dtype != np.uint8 or img.shape[:2] != (self.h, self.w):
            raise ValueError(f"Expected an 8-bit PNG at the sequence resolution: {self.files[i]}")
        return img if gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def microseconds(self):
        us = np.rint(self.time * 1e6).astype(np.int64)
        if np.any(np.diff(us) <= 0):
            raise ValueError("Two frame times coincide at microsecond precision.")
        return us


def prepare(seq, dest):
    # strided copy of the frames for evtween's generate.py (stride 1: use the sequence folder itself)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists.")
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        (Path(tmp) / "frames").mkdir()
        for i, f in enumerate(seq.files):
            shutil.copyfile(f, Path(tmp) / "frames" / f"{i:06d}.png")
        np.save(Path(tmp) / "frame_ts.npy", seq.ts)
        np.save(Path(tmp) / "frame_indices.npy", seq.idx)
        Path(tmp).rename(dest)
    print(dest)


def film(seq, dest):
    # ESIM stage 1, as vid2e's upsampling/upsample.py: FILM midpoints, bisected until the largest flow is <= 1 px.
    # Differences: the real frame clock instead of a constant fps; a largest flow below 1 px gives one midpoint
    # (upstream recurses without end); grayscale from RGB (upstream applies BGR2GRAY to RGB frames).
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    sys.path.insert(0, str(ROOT / "upstream" / "esim" / "upsampling" / "utils"))
    from interpolator import Interpolator
    it, half = Interpolator(str(FILM), None), np.full((1,), 0.5, np.float32)

    def split(I0, I1, t0, t1, n=-1):
        if n == 0:
            return []
        image, f01, f10 = it.interpolate(I0, I1, half)
        if n < 0:
            m = max(np.sqrt((f01 ** 2).sum(-1)).max(), np.sqrt((f10 ** 2).sum(-1)).max(), 1.0)
            n = int(np.ceil(np.log(m) / np.log(2)))
            if n == 0:
                return [(image, (t0 + t1) / 2)]
        tm = (t0 + t1) / 2
        return split(I0, image, t0, tm, n - 1) + [(image, tm)] + split(image, I1, tm, t1, n - 1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists.")
    load = lambda i: seq.read(i)[None].astype(np.float32) / 255
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        (Path(tmp) / "frames").mkdir()
        t = []
        def put(img, ti):
            gray = cv2.cvtColor(np.clip(img[0] * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            cv2.imwrite(str(Path(tmp) / "frames" / f"{len(t):08d}.png"), gray)
            t.append(ti)
        I0 = load(0)
        put(I0, seq.time[0])
        for i in range(1, len(seq.time)):
            I1 = load(i)
            for img, ti in split(I0, I1, seq.time[i - 1], seq.time[i]):
                put(img, ti)
            put(I1, seq.time[i])
            I0 = I1
            print(f"\r{i}/{len(seq.time) - 1} frame pairs, {len(t)} frames", end="", flush=True)
        np.save(Path(tmp) / "t.npy", np.array(t))
        np.save(Path(tmp) / "frame_indices.npy", seq.idx)
        Path(tmp).rename(dest)
    print(f"\n{dest}")


def save_events(path, seq, chunks, meta):
    count, outside, last, ordered = 0, 0, -np.inf, True
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        raw = Path(tmp) / "events.bin"
        with raw.open("wb") as f:
            for t, x, y, p in chunks:
                t, x, y, p = map(np.asarray, (t, x, y, p))
                if t.ndim != 1 or not (t.shape == x.shape == y.shape == p.shape):
                    raise ValueError("Unequal event column shapes.")
                if not np.isfinite(t).all() or not np.isin(p, [-1, 1]).all():
                    raise ValueError("Invalid event time or polarity.")
                if not ((x >= 0) & (x < seq.w) & (x == np.floor(x)) & (y >= 0) & (y < seq.h) & (y == np.floor(y))).all():
                    raise ValueError("Event outside the pixel grid.")
                keep = (t >= 0) & (t <= seq.time[-1])
                outside += int((~keep).sum())
                n = int(keep.sum())
                if n:
                    e = np.empty(n, dtype=EVENT)
                    e["t"], e["x"], e["y"], e["p"] = t[keep], x[keep], y[keep], p[keep]
                    ordered = ordered and e["t"][0] >= last and bool(np.all(e["t"][1:] >= e["t"][:-1]))
                    last = e["t"][-1]
                    e["t"] += seq.ts[0]
                    e.tofile(f)
                    count += n
        events = np.memmap(raw, mode="r+", dtype=EVENT, shape=(count,)) if count else np.empty(0, dtype=EVENT)
        if not ordered:
            events.sort(order="t", kind="quicksort")
        meta["events_outside_time_window"] = outside
        out = Path(tmp) / "events.npz"
        np.savez(out, **{k: events[k] for k in EVENT.names}, frame_indices=seq.idx, frame_ts=seq.ts,
                 width=seq.w, height=seq.h, metadata=json.dumps(meta))
        os.replace(out, path)
    print(f"{path}: {count:,} events, {seq.w}x{seq.h}, {len(seq.ts)} frames, {outside} events outside recording", flush=True)


ap = argparse.ArgumentParser(description="Generate raw events from an evtween-format RGB sequence at its full resolution.")
ap.add_argument("model", choices=[*SOURCES, "all", "prepare", "film"])
ap.add_argument("sequence", type=Path)
ap.add_argument("--out", type=Path, default=Path("predictions"))
ap.add_argument("--stride", type=int, default=1, help="step between input frames in the original sequence (1: the full video)")
ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--v2e-steps", type=int, default=0, help="0: auto SuperSloMo (v2e default), 1: sensor only, >=2: fixed substeps")
a = ap.parse_args()
if a.stride < 1 or not 0 <= a.seed < 2**32 or a.v2e_steps < 0:
    ap.error("Use stride >= 1, seed in [0, 2^32-1] and v2e-steps >= 0.")
seq = Sequence(a.sequence, a.stride)
base = a.out / a.sequence.resolve().name
film_dir = base / f"s{a.stride}" / "film"
if a.model == "prepare":
    prepare(seq, base / f"s{a.stride}" / "rgb")
elif a.model == "film":
    film(seq, film_dir)
elif a.model == "all":
    if not film_dir.is_dir():
        ap.error(f"ESIM needs its FILM frames: run `python run.py film {a.sequence} --stride {a.stride} --out {a.out}` "
                 "in the TensorFlow environment first.")
    opts = [str(a.sequence), "--out", str(a.out), "--stride", str(a.stride), "--device", a.device,
            "--seed", str(a.seed), "--v2e-steps", str(a.v2e_steps)]
    for name in SOURCES:
        subprocess.run([sys.executable, __file__, name, *opts], check=True)
else:
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from models import MODELS
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.model in ("dvs-voltmeter", "iebcs"):
        a.device = "cpu"
    if a.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA is unavailable. Select --device cpu or auto.")
    if a.model == "esim" and a.device != "cuda":
        ap.error("ESIM (esim_torch) needs CUDA.")
    repo, revision, _ = SOURCES[a.model]
    a.repo = ROOT / "upstream" / a.model
    if not a.repo.is_dir():
        ap.error(f"Run python fetch.py {a.model} first.")
    actual = subprocess.check_output(["git", "-C", str(a.repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        ap.error(f"Expected {repo} at {revision}, found {actual}.")
    sys.path.insert(0, str(a.repo / "src" if a.model == "iebcs" else a.repo))
    a.film = film_dir
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    meta = dict(model=a.model, repo=repo, commit=revision, sequence=str(a.sequence.resolve()), stride=a.stride,
                seed=a.seed, device=a.device, torch=str(torch.__version__), numpy=np.__version__)
    variant = a.model
    if a.model == "v2e":
        meta["slomo_steps"] = a.v2e_steps
        variant += "" if not a.v2e_steps else ("-sensor" if a.v2e_steps == 1 else f"-x{a.v2e_steps}")
    if a.model in WEIGHTS and (a.model != "v2e" or a.v2e_steps != 1):
        meta["weights_sha256"] = WEIGHTS[a.model][2] or digest(ROOT / "weights" / WEIGHTS[a.model][0])
    out = base / f"s{a.stride}-seed{a.seed}" / f"{variant}.npz"
    with torch.inference_mode():
        save_events(out, seq, MODELS[a.model](seq, a), meta)
