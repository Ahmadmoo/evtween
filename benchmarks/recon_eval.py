"""Frame reconstruction as a downstream test of simulated events (inference only, nothing is trained).

Pretrained E2VID and FireNet reconstruct intensity frames from the real events and from every
method's events in data_test_evt/<DATASET>/<SPLIT>/<method>/<sequence>/events.npz.

Protocol (identical for every event source; follows EVREAL's standard 'std' protocol)
  * Frame k (k >= 1) is reconstructed from the events in [t_{k-1}, t_k), so it lines up with
    frames/{k:06d}.png and frame_ts[k] of the source sequence (EVREAL 'between_frames').
  * Each window becomes a voxel grid with --bins temporal bins, built and normalised as in EVREAL
    (time scaled by the window's first and last event, bilinear weights, polarity +-1; non-zero
    entries to mean 0 / std 1), with EVREAL's centred zero padding.
  * All event sources of one sequence run as one batch, one recurrent state per source, so every
    method gets the same windows, voxel settings, padding and network state handling.
  * The network state is reset at the start of every sequence; frames k < --warmup are not scored.
  * Outputs are normalised as in EVREAL (E2VID robust 1-99 percentile, FireNet none) and clipped to [0, 1].
  * V2V is left out: it stores one summed voxel per frame interval, without the event timing
    inside the interval that the networks need.

Metrics (frames k >= --warmup, images in [0, 1])
  agreement_mse/ssim/lpips  reconstruction from a method's events vs reconstruction from the real
                            events, same frame and network. Main result; GT frames play no role.
  gt_mse/ssim/lpips         reconstruction vs the ground-truth frame. Secondary: the simulators were
                            given these frames. The 'real' row is the reference level.
  MSE lower is better, SSIM (as skimage: Gaussian 11x11, sigma 1.5) higher, LPIPS (AlexNet) lower.

Automatic setup (first run; --no-auto-setup turns it off)
  * Clones EVREAL (github.com/ercanburak/EVREAL) at a pinned commit into $EVREAL_ROOT or
    ~/.cache/evtween/EVREAL. It contains the model code and the authors' pretrained E2VID and FireNet
    weights; both files are checked against pinned SHA-256 hashes before use.
  * pip-installs missing packages the models and LPIPS need (lpips, opencv-python-headless, einops).
  * LPIPS downloads torchvision's AlexNet weights on first use.

Outputs
  data_test_recon/<DATASET>/<SPLIT>/<network>/<method>/<sequence>/{k:06d}.png   (saved frames)
  reports/.../<timestamp>/contact_sheets/<sequence>/{k:06d}_<category>.png         (--save hybrid)

Saved frames (--save)
  hybrid (default): 40 frames per sequence, the same indices for every method and both networks:
    10 highest and 10 lowest PSNR vs GT (from the stored gt_mse, averaged over all event-generation methods
    and networks), 10 with the largest GT frame-to-frame change, 10 spread evenly over the rest; no
    duplicates (--select-* change the counts). Chosen after scoring, then the sequence is replayed to save
    them. With --run-dir of a finished run the stored metrics are reused and only the frames are made.
  samples: --samples evenly spaced frames; all: every frame; none.
  reports/recon/<DATASET>/<SPLIT>/<timestamp>/
      config.json, per_frame/<sequence>.csv, sanity/<sequence>.csv          (written while running)
      selection/<sequence>.csv, contact_sheets/<sequence>/                  (--save hybrid)
      per_frame.csv, per_sequence.csv, summary.csv, sanity.csv, wilcoxon.csv, selection.csv

Example (one A100, 32 CPUs; two workers share the GPU so loading overlaps with compute):
  python benchmarks/recon_eval.py EDS --devices cuda:0 cuda:0
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
import csv
from datetime import datetime
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings
from types import SimpleNamespace

import numpy as np
from PIL import Image

try:  # torch is only needed for reconstruction; summarize() and the helpers below work without it
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = F = None

if __package__:
    from . import run_metric as rm
else:
    import run_metric as rm

v2 = rm.v2
ROOT = rm.ROOT
METHODS = v2.old.METHODS
EVTWEEN = tuple(v2.old.EVTWEEN)
REAL = "real"
NETWORKS = ("e2vid", "firenet")
EXCLUDED = {"v2v": "stores one summed voxel per frame interval; no event timing inside the interval"}
METRICS = ("mse", "ssim", "lpips")
HIGHER_IS_BETTER = {"mse": False, "ssim": True, "lpips": False}
FRAME_COLUMNS = ("dataset", "split", "sequence", "network", "method", "frame", "t_s",
                 *(f"{kind}_{m}" for kind in ("agreement", "gt") for m in METRICS))
SANITY_COLUMNS = ("dataset", "split", "sequence", "method", "status", "events", "duration_s", "rate_hz",
                  "on_fraction", "windows", "empty_windows")


def _inference(fn):
    """torch.inference_mode() as a decorator, a no-op when torch is not installed."""
    return torch.inference_mode()(fn) if torch is not None else fn


# ---------------------------------------------------------------- discovery (no torch)

def events_root(args):
    return Path(args.events_root) if args.events_root else ROOT / "data_test_evt" / args.dataset / args.split


def discover(args):
    """Methods (in METHODS order) and sequence names found under data_test_evt/<dataset>/<split>."""
    root = events_root(args)
    if not root.is_dir():
        hint = ""
        parent = root.parent.parent
        if not args.events_root and parent.is_dir():
            found = sorted(d.name for d in parent.iterdir() if (d / args.split).is_dir())
            close = [d for d in found if d.lower() == args.dataset.lower()]
            hint = (f" Did you mean '{close[0]}'? (names are case-sensitive)" if close
                    else f" Available datasets: {', '.join(found) or 'none'}")
        raise FileNotFoundError(f"No generated events at {root}.{hint}")
    present = [m for m in METHODS if (root / m).is_dir()]
    requested = list(args.methods) if args.methods else present
    unknown = set(requested) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    missing = [m for m in requested if m not in present]
    if missing and args.methods:
        raise FileNotFoundError(f"No outputs for {missing} under {root}")
    methods = [m for m in requested if m in present and m not in EXCLUDED]
    if not methods:
        raise ValueError("No method left to evaluate")
    names = sorted({s.name for m in methods for s in (root / m).iterdir()
                    if (s / "events.npz").is_file()})
    if args.sequences:
        absent = set(args.sequences) - set(names)
        if absent:
            raise FileNotFoundError(f"Sequences without events: {sorted(absent)}")
        names = [n for n in names if n in set(args.sequences)]
    if not names:
        raise ValueError("No sequences found")
    return methods, names


def source_of(name, args, methods):
    ns = SimpleNamespace(dataset=args.dataset, split=args.split, data_root=args.data_root)
    split_dir = (Path(args.data_root) / args.dataset / args.split) if args.data_root else events_root(args)
    return rm.source_for(name, split_dir, [events_root(args) / m for m in methods], ns)


def sample_frames(n_frames, warmup, count):
    """Evenly spaced scored frame indices to save in --save samples mode."""
    first = max(1, warmup)
    if count <= 0 or first > n_frames - 1:
        return set()
    return {int(round(v)) for v in np.linspace(first, n_frames - 1, min(count, n_frames - first))}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- loading one sequence (CPU threads)

class SequenceData:
    def __init__(self, name, source, edges, meta, frames, events, missing, stats):
        self.name, self.source, self.edges, self.meta = name, source, edges, meta
        self.frames, self.events, self.missing, self.stats = frames, events, missing, stats


def _canonical(t, x, y, p):
    return dict(t=np.ascontiguousarray(t, dtype=np.float64), x=np.ascontiguousarray(x, dtype=np.int64),
                y=np.ascontiguousarray(y, dtype=np.int64), p=np.where(np.asarray(p) > 0, 1, -1).astype(np.int8))


def _load_method(path, meta, args, name):
    rm.verify_metadata(path.parent / "input.json", meta,
                       SimpleNamespace(dataset=args.dataset, split=args.split), name)
    ev = rm.read_events(path, meta)
    return _canonical(ev["t"], ev["x"], ev["y"], ev["p"])


def _stats(ev, edges):
    cuts = np.searchsorted(ev["t"], edges)
    counts = np.diff(cuts)
    n = int(ev["t"].size)
    duration = float(edges[-1] - edges[0])
    return dict(events=n, duration_s=duration, rate_hz=n / duration if duration > 0 else float("nan"),
                on_fraction=float((ev["p"] > 0).mean()) if n else float("nan"),
                windows=int(counts.size), empty_windows=int((counts == 0).sum())), cuts


def load_sequence(name, args, methods, pool):
    source = source_of(name, args, methods)
    seq = rm.Sequence(source)
    edges, meta = seq.geometry
    frames = sorted((source / "frames").glob("*.png"))
    if len(frames) != len(edges):
        raise ValueError(f"{name}: {len(frames)} frames but {len(edges)} timestamps")
    root = events_root(args)
    jobs, missing = {}, []
    for m in methods:
        f = root / m / name / "events.npz"
        if f.is_file():
            jobs[m] = pool.submit(_load_method, f, meta, args, name)
        else:
            missing.append(m)
    events = {}
    if (source / "ev_t.npy").is_file():
        gt = seq.gt
        events[REAL] = _canonical(gt["t"], gt["x"], gt["y"], gt["p"])
    for m in methods:
        if m in jobs:
            events[m] = jobs[m].result()
    stats = {}
    for key, ev in events.items():
        stats[key], ev["cuts"] = _stats(ev, edges)
    return SequenceData(name, source, np.asarray(edges, np.float64), meta, frames, events, missing, stats)


def load_gray(path, width, height):
    with Image.open(path) as im:
        if im.size != (width, height):
            raise ValueError(f"Frame size {im.size} != events {(width, height)}: {path}")
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


# ---------------------------------------------------------------- automatic setup (EVREAL code + official weights)

EVREAL_URL = "https://github.com/ercanburak/EVREAL"
EVREAL_COMMIT = "398f03551b7f5d150ae10aa89e4e731557fce240"
# EVREAL redistributes the authors' pretrained weights (E2VID: uzh-rpg/rpg_e2vid; FireNet:
# cedric-scheerlinck/rpg_e2vid, branch cedric/firenet). SHA-256 and size of each file at EVREAL_COMMIT;
# the run stops if a default checkpoint does not match.
CHECKPOINTS = {
    "e2vid": ("pretrained/E2VID/model.pth",
              "db222f5cec147f8035006f4d56e11c15909d77991901693cfb1cfad286fea106", 42878166),
    "firenet": ("pretrained/FireNet/model.pth",
                "a64af3191a00e9f01cd47d086d9a88c35ff1e63756566ca44b4ff02301ca696e", 468053),
}
PACKAGES = {"lpips": "lpips", "cv2": "opencv-python-headless", "einops": "einops"}  # import name: pip name


def default_evreal_root():
    return Path(os.environ.get("EVREAL_ROOT") or Path.home() / ".cache" / "evtween" / "EVREAL").expanduser()


def ensure_packages(auto, need_lpips=True):
    """Install missing packages that the EVREAL models and LPIPS need (pip, current interpreter)."""
    import importlib.util
    import subprocess
    missing = [pip for mod, pip in PACKAGES.items()
               if (mod != "lpips" or need_lpips) and importlib.util.find_spec(mod) is None]
    if missing:
        if not auto:
            raise ImportError(f"Missing packages; run: pip install {' '.join(missing)}")
        print(f"Installing {' '.join(missing)} ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
        importlib.invalidate_caches()
    return missing


def ensure_evreal(root, auto):
    """Clone EVREAL at the pinned commit if root has no checkout. Returns the checked-out commit."""
    import subprocess
    root = Path(root)
    if not (root / "model" / "__init__.py").is_file():
        if not auto:
            raise FileNotFoundError(f"No EVREAL checkout at {root}; clone {EVREAL_URL} there "
                                    "or drop --no-auto-setup")
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"{root} exists but is not an EVREAL checkout; remove it or set --evreal-root")
        root.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning EVREAL at {EVREAL_COMMIT[:7]} into {root} ...", flush=True)
        subprocess.run(["git", "clone", "--quiet", EVREAL_URL, str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--quiet", EVREAL_COMMIT], check=True)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return head.stdout.strip() or None


def checkpoint_path(name, args):
    """(path, sha256, verified). A custom checkpoint (--<net>-ckpt / RECON_<NET>_CKPT) is used as given;
    the default one from EVREAL must match its pinned hash."""
    custom = getattr(args, f"{name}_ckpt", None) or os.environ.get(f"RECON_{name.upper()}_CKPT")
    if custom:
        path = Path(custom).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")
        return path, file_sha256(path), False
    rel, sha, size = CHECKPOINTS[name]
    path = Path(args.evreal_root) / rel
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; remove {args.evreal_root} so it is cloned again")
    digest = file_sha256(path)
    if digest != sha:
        raise ValueError(f"{path}: sha256 {digest} ({path.stat().st_size} bytes) does not match the pinned "
                         f"{sha} ({size} bytes). A Git LFS pointer or a changed file? Remove the folder to re-clone.")
    return path, digest, True


def setup(args):
    """Install missing packages, clone EVREAL if needed and verify the checkpoints (main process only)."""
    args.evreal_root = str(Path(args.evreal_root).expanduser() if args.evreal_root else default_evreal_root())
    ensure_packages(args.auto_setup, need_lpips=not args.skip_lpips)
    commit = ensure_evreal(args.evreal_root, args.auto_setup)
    return commit, {n: checkpoint_path(n, args) for n in args.networks}


# ---------------------------------------------------------------- reference implementations (numpy, for tests)

def voxel_grid_numpy(t, x, y, p, bins, height, width):
    """EVREAL/E2VID voxel grid of one window, before normalisation: time scaled by the first and last
    event of the window, bilinear weights max(0, 1 - |t_norm - b|), polarity +-1."""
    grid = np.zeros((bins, height, width), np.float64)
    if t.size == 0:
        return grid
    dt = t[-1] - t[0]
    tn = np.linspace(0, bins - 1, t.size) if dt < 1e-9 else (t - t[0]) / dt * (bins - 1)
    for b in range(bins):
        np.add.at(grid[b], (y, x), p * np.maximum(0.0, 1.0 - np.abs(tn - b)))
    return grid


def normalize_numpy(grid):
    """EVREAL normalize_event_tensor: non-zero entries to mean 0, (population) std 1."""
    nz = grid != 0
    n = nz.sum()
    if n == 0:
        return grid.copy()
    mean = grid.sum() / n
    std = max(math.sqrt(max((grid ** 2).sum() / n - mean ** 2, 0.0)), 1e-6)
    return nz * (grid - mean) / std


def crop_params(height, width, num_encoders):
    """EVREAL CropParameters: zero padding (left, right, top, bottom) to a multiple of 2**num_encoders,
    and the crop (y0, y1, x0, x1) that takes the output back to height x width."""
    m = 2 ** int(num_encoders)
    hc, wc = m * math.ceil(height / m), m * math.ceil(width / m)
    pad = (math.ceil((wc - width) / 2), math.floor((wc - width) / 2),
           math.ceil((hc - height) / 2), math.floor((hc - height) / 2))
    cx, cy = wc // 2, hc // 2
    crop = (cy - height // 2, cy + math.ceil(height / 2), cx - width // 2, cx + math.ceil(width / 2))
    return pad, crop


# ---------------------------------------------------------------- GPU parts

def voxelize(ev, j0, j1, bins, height, width, device):
    """Normalised voxel grids of windows j0..j1-1 for one source: (j1-j0, bins, H, W) float32.
    Same result as voxel_grid_numpy + normalize_numpy per window."""
    count = j1 - j0
    out = torch.zeros(count * bins * height * width, dtype=torch.float32, device=device)
    cuts = np.asarray(ev["cuts"][j0:j1 + 1], np.int64)
    a, b = int(cuts[0]), int(cuts[-1])
    if b > a:
        local = torch.from_numpy(cuts - a).to(device)
        t = torch.from_numpy(ev["t"][a:b]).to(device, non_blocking=True)
        x = torch.from_numpy(ev["x"][a:b]).to(device, non_blocking=True)
        y = torch.from_numpy(ev["y"][a:b]).to(device, non_blocking=True)
        p = torch.from_numpy(ev["p"][a:b]).to(device, non_blocking=True).float()
        idx = torch.arange(b - a, device=device)
        w = torch.searchsorted(local[1:], idx, right=True)  # window of each event
        first, stop = local[w], local[w + 1]
        dt = t[stop - 1] - t[first]
        by_time = (t - t[first]) / dt.clamp(min=1e-9) * (bins - 1)
        by_rank = (idx - first).double() / (stop - first - 1).clamp(min=1).double() * (bins - 1)
        tn = torch.where(dt >= 1e-9, by_time, by_rank)
        lo = tn.floor().long().clamp_(0, bins - 1)
        frac = (tn - lo).float()
        pix = y * width + x
        base = w * bins
        out.index_add_(0, (base + lo) * (height * width) + pix, p * (1 - frac))
        if bins > 1:
            hi = lo + 1
            ok = hi < bins
            out.index_add_(0, ((base + hi) * (height * width) + pix)[ok], (p * frac)[ok])
    flat = out.view(count, -1)
    nz = flat != 0
    n = nz.sum(1).clamp(min=1)
    mean = flat.sum(1) / n
    std = ((flat * flat).sum(1) / n - mean * mean).clamp(min=0).sqrt().clamp(min=1e-6)
    flat = torch.where(nz, (flat - mean[:, None]) / std[:, None], torch.zeros_like(flat))
    return flat.view(count, bins, height, width)


@contextmanager
def evreal_imports(root):
    """Import EVREAL's 'model'/'utils' packages without clashing with same-named modules (e.g. the repo's model.py)."""
    names = ("model", "utils", "parse_config")

    def clashes(key):
        return any(key == n or key.startswith(n + ".") for n in names)
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if clashes(k)}
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path.remove(str(root))
        for k in [k for k in sys.modules if clashes(k)]:
            sys.modules.pop(k)
        sys.modules.update(saved)


def build_model(name, checkpoint, evreal_root):
    """Instantiate the network exactly as EVREAL's eval.py does for 'E2VID' and 'FireNet'."""
    with evreal_imports(evreal_root):
        arch = importlib.import_module("model")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if name == "e2vid":
            kwargs = dict(ckpt["model"])
            kwargs["final_activation"] = "sigmoid"
            model = arch.E2VIDRecurrent(kwargs)
        elif name == "firenet":
            kwargs = dict(ckpt["config"]["model"])
            kwargs["final_activation"] = ""
            model = arch.FireNet_legacy(kwargs)
        else:
            raise ValueError(name)
        model.load_state_dict(ckpt["state_dict"])
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model


class Network:
    """EVREAL-style recurrent network: reset_states(), forward(voxels) -> {'image'}; one state per batch item."""
    def __init__(self, name, model, device, amp):
        self.name, self.model, self.device, self.amp = name, model.to(device).eval(), device, amp
        self.num_encoders = int(model.num_encoders)

    def reset(self):
        self.model.reset_states()

    def step(self, voxels):
        pad, (y0, y1, x0, x1) = crop_params(voxels.shape[-2], voxels.shape[-1], self.num_encoders)
        x = F.pad(voxels, pad) if any(pad) else voxels
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp and x.is_cuda):
            out = self.model(x)
        return out["image"][..., y0:y1, x0:x1].float()


def robust_normalize(img, q_lo, q_hi):
    """EVREAL normalize(): (img - P_lo) / (P_hi - P_lo) per image, percentiles with linear interpolation."""
    q = torch.tensor([q_lo / 100, q_hi / 100], device=img.device, dtype=torch.float32)
    out = torch.empty_like(img)
    for i in range(img.shape[0]):
        lo, hi = torch.quantile(img[i].flatten().float(), q)
        scale = torch.where(hi > lo, hi - lo, torch.ones_like(hi))
        out[i] = (img[i] - lo) / scale
    return out


def postprocess(img, network, norm):
    """Output normalisation before saving and scoring, then clip to [0, 1].
    evreal: EVREAL 'std' settings (E2VID robust 1-99 percentiles, FireNet none); minmax: per image 0-100; none."""
    if norm == "evreal" and network == "e2vid":
        img = robust_normalize(img, 1, 99)
    elif norm == "minmax":
        img = robust_normalize(img, 0, 100)
    return img.clamp(0, 1)


def gaussian_window(size=11, sigma=1.5, device=None):
    g = torch.exp(-((torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2) ** 2) / (2 * sigma ** 2))
    return g / g.sum()


class Scorer:
    """MSE, SSIM (as skimage: Gaussian 11x11, sigma 1.5, no sample covariance) and LPIPS (AlexNet, v0.1)
    on images in [0, 1], as in EVREAL."""
    def __init__(self, device, use_lpips=True, batch=64):
        self.device, self.batch = device, batch
        g = gaussian_window(device=device)
        self.gx, self.gy = g.view(1, 1, 1, -1), g.view(1, 1, -1, 1)
        self.lpips = None
        if use_lpips:
            import lpips
            with warnings.catch_warnings():
                # lpips loads AlexNet with torchvision's deprecated 'pretrained' argument and its own bundled
                # weights without weights_only; both are harmless, so hide them here only
                warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.models\._utils")
                warnings.filterwarnings("ignore", category=FutureWarning,
                                        message=r"You are using `torch.load` with `weights_only=False`")
                self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
            for prm in self.lpips.parameters():
                prm.requires_grad_(False)

    def ssim(self, x, y):
        def blur(z):
            return F.conv2d(F.conv2d(z, self.gx), self.gy)
        mx, my = blur(x), blur(y)
        sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
        return s.mean((1, 2, 3))

    @_inference
    def __call__(self, pred, ref):
        """pred, ref: (N, 1, H, W) in [0, 1]. Returns {metric: (N,) tensor}."""
        out = {m: [] for m in METRICS}
        for i in range(0, pred.shape[0], self.batch):
            p, r = pred[i:i + self.batch].clamp(0, 1), ref[i:i + self.batch].clamp(0, 1)
            out["mse"].append(((p - r) ** 2).mean((1, 2, 3)))
            out["ssim"].append(self.ssim(p, r))
            if self.lpips is not None:
                out["lpips"].append(self.lpips(p.expand(-1, 3, -1, -1) * 2 - 1,
                                               r.expand(-1, 3, -1, -1) * 2 - 1).flatten())
            else:
                out["lpips"].append(torch.full_like(out["mse"][-1], float("nan")))
        return {m: torch.cat(v) for m, v in out.items()}


# ---------------------------------------------------------------- one sequence on the GPU

def recon_base(args):
    return Path(args.recon_root) if args.recon_root else ROOT / "data_test_recon" / args.dataset / args.split


def recon_dir(args, network, method, name):
    return recon_base(args) / network / method / name


def _write_csv(path, columns, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _save_png(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)).save(path, compress_level=1)


@_inference
def _clock(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def process_sequence(data, networks, scorer, args, run_dir, io_pool, tag="", score=True, samples=None,
                     categories=None):
    """Reconstruct one sequence. score=True: compute metrics and write per_frame/ and sanity/ CSVs.
    samples: frame indices to save as PNG (default from --save samples|all). score=False only replays the
    reconstruction up to the last sample (same computation) and saves the samples."""
    device = scorer.device
    names = [n for n in (REAL, *args._methods) if n in data.events]
    has_real = REAL in data.events
    h, w = data.meta["height"], data.meta["width"]
    n_frames = len(data.edges)
    if samples is None:
        samples = (set(range(1, n_frames)) if args.save == "all"
                   else sample_frames(n_frames, args.warmup, args.samples) if args.save == "samples" else set())
    samples = {int(k) for k in samples}
    for net in networks:
        net.reset()
    rows, futures = [], []
    frame_jobs = {}

    def request(j0, j1):
        if not score:
            return
        for k in range(j0 + 1, j1 + 1):
            if k >= args.warmup and k not in frame_jobs:
                frame_jobs[k] = io_pool.submit(load_gray, data.frames[k], w, h)

    # frame k comes from window k-1; a replay can stop after the last sample (the state only depends on the past)
    windows = n_frames - 1 if score else min(n_frames - 1, max(samples, default=0))
    request(0, min(windows, args.chunk))
    started = last_report = time.monotonic()
    for j0 in range(0, windows, args.chunk):
        now = time.monotonic()
        if j0 and now - last_report >= args.progress_every:
            rate = j0 / (now - started)
            print(f"{tag} {data.name}: {'frame' if score else 'replay, frame'} {j0}/{windows} "
                  f"({100 * j0 / windows:.0f}%), {rate:.1f} frames/s, ETA {_clock((windows - j0) / rate)}", flush=True)
            last_report = now
        j1 = min(windows, j0 + args.chunk)
        request(j1, min(windows, j1 + args.chunk))  # prefetch the next chunk's frames
        vox = torch.stack([voxelize(data.events[n], j0, j1, args.bins, h, w, device) for n in names])
        pending = []  # (k, network, recon (S,1,H,W))
        for c in range(j1 - j0):
            k = j0 + c + 1
            for net in networks:
                img = postprocess(net.step(vox[:, c]), net.name, args.norm)
                if score and k >= args.warmup:
                    pending.append((k, net.name, img))
                if k in samples:
                    arr = img.cpu().numpy()
                    futures += [io_pool.submit(_save_png, recon_dir(args, net.name, method, data.name) / f"{k:06d}.png",
                                               arr[s, 0]) for s, method in enumerate(names)]
        del vox
        if not pending:
            continue
        gt = {k: torch.from_numpy(frame_jobs.pop(k).result()).to(device)[None, None]
              for k in sorted({k for k, _, _ in pending})}
        preds, refs, tags = [], [], []
        for k, net_name, img in pending:
            preds.append(img)
            refs.append(gt[k].expand(len(names), -1, -1, -1))
            tags += [(k, net_name, n, "gt") for n in names]
            if has_real and len(names) > 1:
                preds.append(img[1:])
                refs.append(img[:1].expand(len(names) - 1, -1, -1, -1))
                tags += [(k, net_name, n, "agreement") for n in names[1:]]
        scores = scorer(torch.cat(preds), torch.cat(refs))
        scores = {m: v.cpu().numpy() for m, v in scores.items()}
        table = {}
        for i, (k, net_name, method, kind) in enumerate(tags):
            row = table.setdefault((net_name, method, k), dict(
                dataset=args.dataset, split=args.split, sequence=data.name, network=net_name,
                method=method, frame=k, t_s=float(data.edges[k]),
                **{f"{kd}_{m}": "" for kd in ("agreement", "gt") for m in METRICS}))
            for m in METRICS:
                row[f"{kind}_{m}"] = float(scores[m][i])
        rows += table.values()
    for k, fut in frame_jobs.items():
        fut.cancel()
    for fut in futures:
        fut.result()
    saved = sorted(k for k in samples if k <= windows)
    if saved:
        for net in networks:
            for method in names:
                meta = dict(dataset=args.dataset, split=args.split, sequence=data.name, network=net.name,
                            method=method, source=str(data.source), bins=args.bins, warmup=args.warmup,
                            window="[t_{k-1}, t_k) -> frame k", saved=args.save, frames=saved, run=str(run_dir))
                if categories:
                    meta["categories"] = {str(k): categories[k] for k in saved if k in categories}
                (recon_dir(args, net.name, method, data.name) / "recon.json").write_text(json.dumps(meta, indent=2))
    if not score:
        return 0
    rows.sort(key=lambda r: (r["network"], names.index(r["method"]), r["frame"]))
    # one line per network: mean agreement with the real-event reconstruction (quick look; full tables at the end)
    key = "agreement_lpips" if not args.skip_lpips else "agreement_ssim"
    for net in networks:
        means = []
        for method in names[1:] if has_real else []:
            vals = [r[key] for r in rows if r["network"] == net.name and r["method"] == method and r[key] != ""]
            if vals:
                means.append(f"{method} {np.mean(vals):.3f}")
        if means:
            print(f"{tag} {data.name}: {net.name} mean {key}: " + " | ".join(means), flush=True)
    _write_csv(run_dir / "per_frame" / f"{data.name}.csv", FRAME_COLUMNS, rows)
    sanity = [dict(dataset=args.dataset, split=args.split, sequence=data.name, method=n, status="ok",
                   **data.stats[n]) for n in names]
    sanity += [dict(dataset=args.dataset, split=args.split, sequence=data.name, method=n, status="missing")
               for n in data.missing]
    _write_csv(run_dir / "sanity" / f"{data.name}.csv", SANITY_COLUMNS, sanity)
    return len(rows)


# ---------------------------------------------------------------- hybrid frame selection and contact sheets (no torch)

SELECTION_COLUMNS = ("dataset", "split", "sequence", "frame", "t_s", "category", "mean_gt_psnr", "motion")
CATEGORY_ORDER = ("high_psnr", "low_psnr", "high_motion", "temporal")


def psnr_from_mse(mse):
    """PSNR in dB of images in [0, 1], from the stored MSE (no new metric computation)."""
    return 10 * np.log10(1.0 / np.maximum(np.asarray(mse, np.float64), 1e-10))


def selection_counts(args):
    return dict(high_psnr=args.select_high_psnr, low_psnr=args.select_low_psnr,
                high_motion=args.select_motion, temporal=args.select_temporal)


def select_frames(frames, motion, counts, total):
    """Pick one shared set of frame indices for every method and network of a sequence.

    frames: per_frame rows of one sequence (pandas). motion: {frame: mean |GT_k - GT_{k-1}|}.
    Categories are filled in CATEGORY_ORDER without duplicates (a frame already taken is skipped and the
    next candidate is used): highest and lowest PSNR vs GT averaged over all event-generation methods and
    networks, highest GT motion, then frames spread evenly over the remaining ones. The temporal category
    gets at least counts['temporal'] frames and tops the selection up to `total`.
    Returns {frame: category} and the per-frame mean PSNR (pandas Series)."""
    sim = frames[frames["method"] != REAL]
    if sim.empty:
        sim = frames
    sim = sim[sim["gt_mse"].notna()]
    psnr = (sim.assign(psnr=psnr_from_mse(sim["gt_mse"].astype(float)))
            .groupby("frame")["psnr"].mean().sort_index())
    candidates = [int(f) for f in psnr.index]
    chosen = {}

    def take(order, n, category):
        for f in order:
            if n <= 0:
                break
            f = int(f)
            if f not in chosen:
                chosen[f] = category
                n -= 1

    take(psnr.sort_values(ascending=False, kind="stable").index, counts["high_psnr"], "high_psnr")
    take(psnr.sort_values(ascending=True, kind="stable").index, counts["low_psnr"], "low_psnr")
    import pandas as pd
    mot = pd.Series(motion, dtype=float).reindex(candidates).dropna()
    take(mot.sort_values(ascending=False, kind="stable").index, counts["high_motion"], "high_motion")
    remaining = [f for f in candidates if f not in chosen]
    n = min(len(remaining), max(counts["temporal"], total - len(chosen)))
    if n > 0:
        positions = np.unique(np.round(np.linspace(0, len(remaining) - 1, n)).astype(int))
        for i in positions:
            chosen[remaining[i]] = "temporal"
    return dict(sorted(chosen.items())), psnr


def gt_motion(frame_paths, ks, width, height, pool):
    """Mean absolute difference between consecutive ground-truth frames (grayscale, [0, 1]) for frame k."""
    def diff(k):
        return float(np.abs(load_gray(frame_paths[k], width, height)
                            - load_gray(frame_paths[k - 1], width, height)).mean())
    ks = [int(k) for k in ks if k >= 1]
    return dict(zip(ks, pool.map(diff, ks)))


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def contact_sheet(path, title, rows, tile_width=320):
    """rows: [(row label, [(caption lines, PIL image), ...]), ...] -> one PNG grid with captions."""
    from PIL import ImageDraw
    font, title_font = _font(13), _font(16)
    first = rows[0][1][0][1]
    tile_h = max(1, round(tile_width * first.height / first.width))
    lines = max(len(c) for _, tiles in rows for c, _ in tiles)
    cap_h, label_w, title_h, gap = 6 + 16 * lines, 90, 34, 6
    cols = max(len(tiles) for _, tiles in rows)
    sheet = Image.new("RGB", (label_w + cols * (tile_width + gap) + gap,
                              title_h + len(rows) * (tile_h + cap_h + gap) + gap), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 8), title, fill="black", font=title_font)
    for r, (label, tiles) in enumerate(rows):
        y = title_h + r * (tile_h + cap_h + gap)
        draw.text((gap, y + tile_h // 2 - 8), label, fill="black", font=title_font)
        for c, (caption, im) in enumerate(tiles):
            x = label_w + c * (tile_width + gap)
            sheet.paste(im.convert("RGB").resize((tile_width, tile_h), Image.BILINEAR), (x, y))
            for i, line in enumerate(caption):
                draw.text((x + 2, y + tile_h + 3 + 16 * i), line, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, optimize=False, compress_level=1)
    return path


def contact_sheets(run_dir, args, name, selection, frames, frame_paths, networks, methods, psnr, motion):
    """One sheet per selected frame: rows = networks, columns = GT, real, every method (same frame everywhere)."""
    out = []
    frames = frames.set_index(["network", "method", "frame"])
    for k, category in selection.items():
        gt = Image.open(frame_paths[k]).convert("L")
        rows = []
        for net in networks:
            tiles = [(["ground truth", f"frame {k}"], gt)]
            for method in methods:
                png = recon_dir(args, net, method, name) / f"{k:06d}.png"
                if not png.is_file():
                    continue
                caption = [method]
                if (net, method, k) in frames.index:
                    row = frames.loc[(net, method, k)]
                    caption += [f"vs GT  PSNR {float(psnr_from_mse(row['gt_mse'])):.2f}",
                                f"vs GT  SSIM {row['gt_ssim']:.3f}"]
                    if method != REAL and pd_notna(row.get("agreement_lpips")):
                        caption.append(f"vs real LPIPS {row['agreement_lpips']:.3f}")
                    if method != REAL and pd_notna(row.get("agreement_ssim")):
                        caption.append(f"vs real SSIM {row['agreement_ssim']:.3f}")
                tiles.append((caption, Image.open(png)))
            rows.append((net, tiles))
        title = (f"{args.dataset} / {name} / frame {k} ({category})   mean PSNR {psnr.get(k, float('nan')):.2f} dB"
                 f"   motion {motion.get(k, float('nan')):.4f}")
        out.append(contact_sheet(run_dir / "contact_sheets" / name / f"{k:06d}_{category}.png", title, rows,
                                 args.sheet_tile_width))
    return out


def pd_notna(value):
    try:
        return value is not None and value == value and value != ""
    except (TypeError, ValueError):
        return False


def render_selection(data, networks, scorer, args, run_dir, io_pool, tag=""):
    """--save hybrid: choose frames from per_frame/<seq>.csv, replay the reconstruction to save exactly
    those frames for every method and network, then write the contact sheets and selection/<seq>.csv."""
    import pandas as pd
    frames = pd.read_csv(run_dir / "per_frame" / f"{data.name}.csv")
    ks = sorted(int(k) for k in frames["frame"].unique())
    w, h = data.meta["width"], data.meta["height"]
    motion = gt_motion(data.frames, ks, w, h, io_pool)
    selection, psnr = select_frames(frames, motion, selection_counts(args), args.select_total)
    process_sequence(data, networks, scorer, args, run_dir, io_pool, tag, score=False,
                     samples=set(selection), categories=selection)
    methods = [n for n in (REAL, *args._methods) if n in data.events]
    contact_sheets(run_dir, args, data.name, selection, frames, data.frames, [n.name for n in networks],
                   methods, psnr, motion)
    rows = [dict(dataset=args.dataset, split=args.split, sequence=data.name, frame=k, t_s=float(data.edges[k]),
                 category=c, mean_gt_psnr=float(psnr.get(k, float("nan"))), motion=motion.get(k, float("nan")))
            for k, c in selection.items()]
    _write_csv(run_dir / "selection" / f"{data.name}.csv", SELECTION_COLUMNS, rows)  # written last: marks done
    per_cat = ", ".join(f"{c} {sum(v == c for v in selection.values())}" for c in CATEGORY_ORDER)
    print(f"{tag} {data.name}: saved {len(selection)} selected frames ({per_cat}) -> contact_sheets/{data.name}/",
          flush=True)
    return selection


# ---------------------------------------------------------------- workers

def worker(device, names, args, run_dir, worker_id):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_num_threads(max(1, args.io_workers // 2))
    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    networks = [Network(n, build_model(n, checkpoint_path(n, args)[0], args.evreal_root), device, args.amp)
                for n in args.networks]
    scorer = Scorer(device, use_lpips=not args.skip_lpips, batch=args.metric_batch)
    done, errors = 0, []
    with ThreadPoolExecutor(args.io_workers) as io_pool, ThreadPoolExecutor(1) as prefetch:
        loads = {}
        for i, name in enumerate(names):
            # keep the next --prefetch sequences loading on the CPU while this one runs on the GPU
            for j in range(i, min(len(names), i + args.prefetch + 1)):
                if j not in loads:
                    loads[j] = prefetch.submit(load_sequence, names[j], args, args._methods, io_pool)
            start = time.monotonic()
            try:
                data = loads.pop(i).result()
                tag = f"[w{worker_id} {device}]"
                counts = ", ".join(f"{k} {v['events'] / 1e6:.2f}M" for k, v in data.stats.items())
                print(f"{tag} {name}: loaded {len(data.edges)} frames {data.meta['width']}x{data.meta['height']} "
                      f"in {time.monotonic() - start:.1f}s; events: {counts}"
                      + (f"; missing: {', '.join(data.missing)}" if data.missing else ""), flush=True)
                if REAL not in data.events:
                    print(f"{tag} WARNING {name}: no real events (ev_*.npy) in {data.source}; "
                          "agreement metrics will be empty", flush=True)
                for k, v in data.stats.items():
                    if v["windows"] and v["empty_windows"] > 0.2 * v["windows"]:
                        print(f"{tag} WARNING {name}/{k}: {v['empty_windows']} of {v['windows']} "
                              "frame windows have no events", flush=True)
                if (run_dir / "per_frame" / f"{name}.csv").is_file():
                    rows = None
                    print(f"{tag} {name}: metrics already in per_frame/{name}.csv; selecting and saving frames only",
                          flush=True)
                else:
                    rows = process_sequence(data, networks, scorer, args, run_dir, io_pool, tag,
                                            samples=set() if args.save == "hybrid" else None)
                if args.save == "hybrid":
                    render_selection(data, networks, scorer, args, run_dir, io_pool, tag)
                del data
                done += 1
                print(f"{tag} {name}: DONE in {_clock(time.monotonic() - start)}"
                      + (f" ({rows} scored rows) -> per_frame/{name}.csv" if rows is not None else ""), flush=True)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                print(f"[w{worker_id}] ERROR {name}: {exc}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
    return done, errors


def _worker_entry(device, names, args, run_dir, worker_id):
    return worker(device, names, args, run_dir, worker_id)


def balance(names, args, methods, workers):
    """Longest sequences first, each to the least loaded worker (frame count as cost)."""
    cost = {}
    for n in names:
        try:
            cost[n] = len(list((source_of(n, args, methods) / "frames").glob("*.png")))
        except Exception:
            cost[n] = 0
    shares, load = [[] for _ in range(workers)], [0] * workers
    for n in sorted(names, key=lambda n: -cost[n]):
        i = load.index(min(load))
        shares[i].append(n)
        load[i] += cost[n]
    return shares


def run(args):
    if torch is None:
        raise ImportError("PyTorch is required for reconstruction")
    if not getattr(args, "io_workers", None):
        args.io_workers = max(2, (os.cpu_count() or 4) // len(args.devices))
    methods, names = discover(args)
    args._methods = methods
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = ROOT / "reports" / "recon" / args.dataset / args.split / datetime.now().strftime("%Y%m%d-%H%M%S")
    (run_dir / "per_frame").mkdir(parents=True, exist_ok=True)
    (run_dir / "sanity").mkdir(parents=True, exist_ok=True)
    (run_dir / "selection").mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (run_dir / "per_frame" / f"{n}.csv").is_file()
            or (args.save == "hybrid" and not (run_dir / "selection" / f"{n}.csv").is_file())]
    commit, ckpts = setup(args)
    config = dict(
        dataset=args.dataset, split=args.split, events_root=str(events_root(args)), methods=methods,
        excluded={m: EXCLUDED[m] for m in EXCLUDED if (events_root(args) / m).is_dir()},
        sequences=names, networks=list(args.networks), bins=args.bins, warmup=args.warmup, norm=args.norm,
        window="frame k <- events in [t_{k-1}, t_k) (EVREAL between_frames)",
        voxel="EVREAL: time scaled by first/last event of the window, bilinear, polarity +-1; "
              "non-zero entries mean 0 / population std 1",
        padding="EVREAL CropParameters: centred zero padding to a multiple of 2**num_encoders",
        postprocess={"evreal": "E2VID robust 1-99 percentile, FireNet none; clip [0, 1]",
                     "minmax": "per image 0-100 percentile; clip [0, 1]", "none": "clip [0, 1]"}[args.norm],
        amp=args.amp, tf32=args.tf32, lpips=not args.skip_lpips,
        save=args.save, samples=args.samples, devices=list(args.devices),
        selection=dict(selection_counts(args), total=args.select_total,
                       rule="per sequence, one shared frame set for all methods and networks; categories filled in "
                            "order high_psnr, low_psnr, high_motion, temporal without duplicates; PSNR from gt_mse, "
                            "averaged over event-generation methods and networks; motion = mean |GT_k - GT_k-1|; "
                            "temporal tops up to total") if args.save == "hybrid" else None,
        recon_root=str(recon_base(args)),
        evreal=dict(root=args.evreal_root, commit=commit, pinned=EVREAL_COMMIT, url=EVREAL_URL),
        checkpoints={n: dict(path=str(path), sha256=sha, verified_against_pin=ok)
                     for n, (path, sha, ok) in ckpts.items()},
        torch=str(torch.__version__), host=platform.node(),
        gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        code_sha256=file_sha256(Path(__file__)))
    cfg_path = run_dir / "config.json"
    if cfg_path.is_file():
        old = json.loads(cfg_path.read_text())
        keep = ("methods", "networks", "bins", "warmup", "norm", "amp", "lpips")
        changed = [k for k in keep if old.get(k) != config[k]]
        if {n: c["sha256"] for n, c in old.get("checkpoints", {}).items()} != \
                {n: c["sha256"] for n, c in config["checkpoints"].items()}:
            changed.append("checkpoints")
        if changed:
            raise ValueError(f"Resuming {run_dir} with different settings: {changed}")
    cfg_path.write_text(json.dumps(config, indent=2))
    print(f"{len(methods)} methods, {len(names)} sequences ({len(names)-len(todo)} already done) -> {run_dir}", flush=True)
    errors = []
    if todo:
        devices = list(args.devices)
        if len(devices) == 1:
            _, errors = worker(devices[0], todo, args, run_dir, 0)
        else:
            shares = balance(todo, args, methods, len(devices))
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(len(devices), mp_context=ctx) as pool:
                futures = [pool.submit(_worker_entry, d, s, args, run_dir, i)
                           for i, (d, s) in enumerate(zip(devices, shares)) if s]
                for fut in futures:
                    errors += fut.result()[1]
    summarize(run_dir)
    print(f"Summary tables: {run_dir / 'summary.csv'}, sanity.csv, wilcoxon.csv", flush=True)
    if args.save == "hybrid":
        files = sorted((run_dir / "selection").glob("*.csv"))
        if files:
            import pandas as pd
            pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(run_dir / "selection.csv", index=False)
        print(f"Selected frames: {run_dir / 'selection.csv'}; contact sheets: {run_dir / 'contact_sheets'}", flush=True)
    if errors:
        raise RuntimeError("Some sequences failed (rerun with --run-dir to resume):\n" + "\n".join(errors))
    return run_dir


# ---------------------------------------------------------------- summary (no torch)

def summarize(run_dir):
    import pandas as pd
    run_dir = Path(run_dir)
    files = sorted((run_dir / "per_frame").glob("*.csv"))
    if not files:
        return None
    frames = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frames.to_csv(run_dir / "per_frame.csv", index=False)
    value_cols = [c for c in frames.columns if c.startswith(("agreement_", "gt_"))]
    per_seq = (frames.groupby(["dataset", "split", "network", "method", "sequence"], sort=False)[value_cols]
               .mean().reset_index())
    per_seq.to_csv(run_dir / "per_sequence.csv", index=False)
    order = {m: i for i, m in enumerate((REAL, *METHODS))}
    rows = []
    for (net, method), g in per_seq.groupby(["network", "method"], sort=False):
        row = dict(network=net, method=method, sequences=len(g),
                   frames=int((frames["network"].eq(net) & frames["method"].eq(method)).sum()))
        for c in value_cols:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std(ddof=1) if len(g) > 1 else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["_o"] = summary["method"].map(order)
    summary = summary.sort_values(["network", "_o"]).drop(columns="_o").reset_index(drop=True)
    for c in value_cols:
        metric = c.split("_", 1)[1]
        summary[f"{c}_rank"] = (summary.groupby("network")[f"{c}_mean"]
                                .rank(ascending=not HIGHER_IS_BETTER[metric], method="min"))
    summary.to_csv(run_dir / "summary.csv", index=False)
    sanity_files = sorted((run_dir / "sanity").glob("*.csv"))
    if sanity_files:
        pd.concat([pd.read_csv(f) for f in sanity_files], ignore_index=True).to_csv(run_dir / "sanity.csv", index=False)
    wilcoxon(per_seq, summary).to_csv(run_dir / "wilcoxon.csv", index=False)
    return summary


def wilcoxon(per_seq, summary):
    """Best Evtween variant vs best baseline per network and metric, paired over sequences."""
    from scipy.stats import wilcoxon as test
    rows = []
    for net, s in summary.groupby("network"):
        for kind in ("agreement", "gt"):
            for metric in METRICS:
                col = f"{kind}_{metric}"
                if f"{col}_mean" not in s or s[f"{col}_mean"].isna().all():
                    continue
                pick = (lambda d: d.loc[d[f"{col}_mean"].idxmax() if HIGHER_IS_BETTER[metric]
                                        else d[f"{col}_mean"].idxmin(), "method"])
                ours = s[s["method"].isin(EVTWEEN)].dropna(subset=[f"{col}_mean"])
                base = s[~s["method"].isin((*EVTWEEN, REAL))].dropna(subset=[f"{col}_mean"])
                if ours.empty or base.empty:
                    continue
                a, b = pick(ours), pick(base)
                g = per_seq[per_seq["network"] == net].pivot(index="sequence", columns="method", values=col)
                pair = g[[a, b]].dropna()
                diff = pair[a] - pair[b]
                stat = p = float("nan")
                if len(pair) >= 2 and (diff != 0).any():
                    res = test(pair[a], pair[b])
                    stat, p = float(res.statistic), float(res.pvalue)
                better = (diff.mean() > 0) == HIGHER_IS_BETTER[metric] if len(pair) else None
                rows.append(dict(network=net, metric=col, evtween=a, baseline=b, sequences=len(pair),
                                 evtween_mean=pair[a].mean(), baseline_mean=pair[b].mean(),
                                 mean_difference=diff.mean(), evtween_better=better,
                                 statistic=stat, p_value=p,
                                 note="fewer than 6 sequences: two-sided p cannot go below 0.05"
                                 if len(pair) < 6 else ""))
    import pandas as pd
    return pd.DataFrame(rows)


def qualitative(run_dir, sequence, frame=None, network="e2vid", methods=None, out=None, columns=5):
    """Grid of the same frame reconstructed from every source, next to the GT frame. Returns the figure path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    base = Path(cfg["recon_root"]) / network
    sources = [REAL, *(methods or cfg["methods"])]
    found = sorted(int(p.stem) for p in (base / REAL / sequence).glob("*.png"))
    if not found:
        raise FileNotFoundError(f"No saved frames for {sequence} under {base / REAL}")
    frame = frame if frame is not None else found[len(found) // 2]
    meta = json.loads((base / REAL / sequence / "recon.json").read_text())
    gt = Image.open(sorted((Path(meta["source"]) / "frames").glob("*.png"))[frame]).convert("L")
    panels = [("ground truth", gt)] + [(s, Image.open(base / s / sequence / f"{frame:06d}.png"))
                                       for s in sources if (base / s / sequence / f"{frame:06d}.png").is_file()]
    rows_n = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows_n, columns, figsize=(3.2 * columns, 2.6 * rows_n), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, im) in zip(axes.flat, panels):
        ax.imshow(np.asarray(im), cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"{cfg['dataset']} / {sequence} / frame {frame} / {network}", fontsize=11)
    fig.tight_layout()
    out = Path(out) if out else run_dir / f"qualitative_{network}_{sequence}_{frame:06d}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- CLI

def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="Folder name under data_test_evt/ (e.g. EDS, HQ-EVFI, BS-ERGB, or any other)")
    p.add_argument("--split", default="test")
    p.add_argument("--methods", nargs="+", help="Default: every method found for the dataset (V2V excluded)")
    p.add_argument("--sequences", nargs="+", help="Default: all sequences")
    p.add_argument("--networks", nargs="+", choices=NETWORKS, default=list(NETWORKS))
    p.add_argument("--events-root", help="Override data_test_evt/<dataset>/<split>")
    p.add_argument("--data-root", help="Override the source sequences root (<root>/<dataset>/<split>/<seq>)")
    p.add_argument("--recon-root", help="Override data_test_recon/<dataset>/<split>")
    p.add_argument("--run-dir", help="Resume (or name) a report folder; finished sequences are skipped")
    p.add_argument("--evreal-root", help="EVREAL checkout (default: $EVREAL_ROOT or ~/.cache/evtween/EVREAL); "
                   "cloned at the pinned commit if missing")
    p.add_argument("--no-auto-setup", dest="auto_setup", action="store_false",
                   help="Do not clone EVREAL or pip-install missing packages; fail instead")
    p.add_argument("--e2vid-ckpt", help="Use another E2VID checkpoint in the same format instead of EVREAL's")
    p.add_argument("--firenet-ckpt", help="Use another FireNet checkpoint in the same format instead of EVREAL's")
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--warmup", type=int, default=10, help="Frames k < warmup are reconstructed but not scored")
    p.add_argument("--norm", choices=("evreal", "minmax", "none"), default="evreal",
                   help="Output normalisation before saving and scoring (evreal: E2VID robust 1-99 percentile, "
                        "FireNet none, as in EVREAL's standard protocol)")
    p.add_argument("--save", choices=("hybrid", "samples", "all", "none"), default="hybrid",
                   help="hybrid: frames chosen from the metrics (see --select-*), same frames for every method and "
                        "network, plus contact sheets; samples: --samples evenly spaced frames; all; none")
    p.add_argument("--samples", type=int, default=5, help="Frames saved per sequence with --save samples")
    p.add_argument("--select-high-psnr", type=int, default=10, help="hybrid: highest mean PSNR vs GT")
    p.add_argument("--select-low-psnr", type=int, default=10, help="hybrid: lowest mean PSNR vs GT")
    p.add_argument("--select-motion", type=int, default=10, help="hybrid: largest GT frame-to-frame change")
    p.add_argument("--select-temporal", type=int, default=10, help="hybrid: at least this many evenly spread frames")
    p.add_argument("--select-total", type=int, default=40,
                   help="hybrid: frames per sequence; evenly spread frames fill up to this number")
    p.add_argument("--sheet-tile-width", type=int, default=320, help="Width of one image in a contact sheet")
    p.add_argument("--devices", nargs="+", default=["cuda:0"],
                   help="One worker per entry; repeat a GPU (cuda:0 cuda:0) to run two workers on it")
    p.add_argument("--io-workers", type=int, help="Loader threads per worker (default: CPUs / workers)")
    p.add_argument("--prefetch", type=int, default=1, help="Sequences loaded ahead per worker")
    p.add_argument("--chunk", type=int, default=16, help="Frame windows voxelised per GPU step")
    p.add_argument("--metric-batch", type=int, default=64)
    p.add_argument("--amp", action="store_true", help="fp16 autocast for the networks (faster, slightly different)")
    p.add_argument("--no-tf32", dest="tf32", action="store_false")
    p.add_argument("--skip-lpips", action="store_true")
    p.add_argument("--progress-every", type=float, default=30, help="Seconds between progress lines")
    p.add_argument("--summarize-only", metavar="RUN_DIR", help="Rebuild the summary tables of a run")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.summarize_only:
        print(summarize(args.summarize_only))
        return Path(args.summarize_only)
    if not args.io_workers:
        args.io_workers = max(2, (os.cpu_count() or 4) // len(args.devices))
    run_dir = run(args)
    print(f"Done: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
