import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from fetch import ROOT, SOURCES, WEIGHTS

EVENT = np.dtype([("t", "<f8"), ("x", "<i2"), ("y", "<i2"), ("p", "i1")])


class Sequence:
    def __init__(self, path, stride, view):
        files = sorted((path / "frames").glob("*.png"))
        t = np.load(path / "frame_ts.npy", allow_pickle=False).astype(np.float64)
        if t.shape != (len(files),) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("Expected one strictly increasing timestamp in seconds per PNG.")
        self.idx = np.arange(0, len(files), stride)
        if len(self.idx) < 2:
            raise ValueError("Need at least two observed frames after striding.")
        self.files = [files[i] for i in self.idx]
        self.ts = t[self.idx]
        self.time = self.ts - self.ts[0]
        from PIL import Image
        with Image.open(self.files[0]) as f:
            self.source_w, self.source_h = f.size
        self.w, self.h = (346, 260) if view == "center346" else (self.source_w, self.source_h)
        if self.w > self.source_w or self.h > self.source_h:
            raise ValueError(f"Cannot take a 346x260 crop from {self.source_w}x{self.source_h}.")
        if max(self.w, self.h) > 32768:
            raise ValueError("The output pixel coordinates exceed int16.")
        self.x0, self.y0 = (self.source_w - self.w) // 2, (self.source_h - self.h) // 2

    def read(self, i, gray=False):
        path = self.files[i]
        mode = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
        img = cv2.imread(str(path), mode | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None or img.dtype != np.uint8 or img.shape[:2] != (self.source_h, self.source_w):
            raise ValueError(f"Expected an 8-bit PNG at the sequence resolution: {path}")
        img = img[self.y0:self.y0+self.h, self.x0:self.x0+self.w]
        return np.ascontiguousarray(img) if gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def microseconds(self):
        us = np.rint(self.time * 1e6).astype(np.int64)
        if np.any(np.diff(us) <= 0):
            raise ValueError("Two observed frame times coincide at microsecond precision.")
        return us


def prepare(seq, dest):
    from PIL import Image
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists. Use this prepared input or choose a different --out.")
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        rgb = Path(tmp) / "rgb"
        (rgb / "frames").mkdir(parents=True)
        for i in range(len(seq.files)):
            Image.fromarray(seq.read(i)).save(rgb / "frames" / f"{i:06d}.png")
        np.save(rgb / "frame_ts.npy", seq.ts)
        np.savez(rgb / "selection.npz", frame_indices=seq.idx, roi_xywh=[seq.x0, seq.y0, seq.w, seq.h],
                 source_wh=[seq.source_w, seq.source_h])
        rgb.rename(dest)
    print(dest)


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
                    raise ValueError("Event outside the chosen pixel grid.")
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
                 width=seq.w, height=seq.h, roi_xywh=[seq.x0, seq.y0, seq.w, seq.h],
                 source_wh=[seq.source_w, seq.source_h], metadata=json.dumps(meta))
        os.replace(out, path)
    print(f"{path}: {count:,} events, {seq.w}x{seq.h}, {len(seq.ts)} frames, {outside} events outside recording", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Generate raw events from an evtween-format RGB sequence.")
    ap.add_argument("model", choices=[*SOURCES, "all", "prepare"])
    ap.add_argument("sequence", type=Path)
    ap.add_argument("--out", type=Path, default=Path("predictions"))
    ap.add_argument("--view", choices=["native", "center346"], default="native")
    ap.add_argument("--stride", type=int, default=1, help="step between observed frames in the original sequence")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--evis-gamma", type=float, default=1.0, help="PNG to linear light for EVIS-core")
    ap.add_argument("--v2e-steps", type=int, default=0, help="0: auto SuperSloMo, 1: sensor only, >=2: fixed substeps")
    a = ap.parse_args()
    if a.stride < 1 or not 0 < a.seed < 2**32 or a.v2e_steps < 0 or not np.isfinite(a.evis_gamma) or a.evis_gamma <= 0:
        ap.error("Use stride >= 1, seed in [1, 2^32-1], v2e-steps >= 0 and positive finite gamma.")
    seq = Sequence(a.sequence, a.stride, a.view)
    if a.model == "prepare":
        prepare(seq, a.out / a.sequence.resolve().name / f"{a.view}-s{a.stride}" / "rgb")
        return
    if a.model in ("all", "v2ce") and ((seq.w, seq.h) != (346, 260) or len(seq.time) < 17):
        ap.error("For a six-way comparison use --view center346 (or native 346x260) and at least 17 observed frames.")
    if a.model == "all":
        opts = [str(a.sequence), "--out", str(a.out), "--view", a.view, "--stride", str(a.stride), "--device", a.device,
                "--seed", str(a.seed), "--evis-gamma", str(a.evis_gamma), "--v2e-steps", str(a.v2e_steps)]
        for name in SOURCES:
            subprocess.run([sys.executable, __file__, name, *opts], check=True)
        return
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
    repo, revision, _ = SOURCES[a.model]
    a.repo = ROOT / "upstream" / a.model
    if not a.repo.is_dir():
        ap.error(f"Run python fetch.py {a.model} first.")
    actual = subprocess.check_output(["git", "-C", str(a.repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        ap.error(f"Expected {repo} at {revision}, found {actual}.")
    sys.path.insert(0, str(a.repo / "src" if a.model == "iebcs" else a.repo))
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    meta = dict(model=a.model, repo=repo, commit=revision, sequence=str(a.sequence.resolve()), view=a.view, stride=a.stride,
                seed=a.seed, device=a.device, torch=str(torch.__version__), numpy=np.__version__)
    variant = a.model
    if a.model == "v2e":
        meta["slomo_steps"] = a.v2e_steps
        variant += "-auto" if not a.v2e_steps else ("-sensor" if a.v2e_steps == 1 else f"-x{a.v2e_steps}")
    if a.model == "evis-core":
        meta["gamma"] = a.evis_gamma
        variant += f"-g{a.evis_gamma:g}"
    if a.model in WEIGHTS and (a.model == "v2ce" or a.v2e_steps != 1):
        meta["weights_sha256"] = WEIGHTS[a.model][2]
    out = a.out / a.sequence.resolve().name / f"{a.view}-s{a.stride}-seed{a.seed}" / f"{variant}.npz"
    with torch.inference_mode():
        save_events(out, seq, MODELS[a.model](seq, a), meta)


if __name__ == "__main__":
    main()
