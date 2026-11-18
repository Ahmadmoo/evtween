import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from fetch import ROOT, WEIGHTS


def evis_core(seq, args):
    from dvs_gen.dvs import BatchedMultiCamProcessor
    recorder = SimpleNamespace(events=None)
    def record(camera, env, x, y, p, t):
        recorder.events = np.full(x.numel(), t), x.cpu().numpy(), y.cpu().numpy(), p.cpu().numpy()
    recorder.record = record
    sim = BatchedMultiCamProcessor(recorder, "camera", threshold=0.15)
    for i, t in enumerate(seq.time):
        rgb = torch.as_tensor(seq.read(i), device=args.device).float().div_(255).pow_(args.evis_gamma)
        recorder.events = None
        sim(rgb[None], float(t))
        if recorder.events is not None:
            yield recorder.events


def fracevent(seq, args):
    from core import FracEventSimulator, load_config
    cfg = load_config(args.repo / "configs" / "davis346.yaml")
    cfg.solver.device, cfg.noise.seed = args.device, args.seed
    sim = FracEventSimulator(cfg)
    prev = seq.read(0).astype(np.float32) / 255
    for i in range(1, len(seq.time)):
        image = seq.read(i).astype(np.float32) / 255
        for e in sim.process_frame_pair(prev, image, seq.time[i-1], seq.time[i]):
            yield e.t, e.x, e.y, e.p
        prev = image


def iebcs(seq, args):
    from dvs_sensor import DvsSensor
    sim = DvsSensor("camera")
    portrait = seq.h > seq.w  # Upstream indexes the noise table with y * height + x.
    w, h = (seq.h, seq.w) if portrait else (seq.w, seq.h)
    sim.initCamera(w, h, lat=100, jit=10, ref=100, tau=40, th_pos=0.4, th_neg=0.4,
                   th_noise=0.01, bgnp=0.1, bgnn=0.01)
    sim.init_bgn_hist(str(args.repo / "data/noise_pos_161lux.npy"), str(args.repo / "data/noise_neg_161lux.npy"))
    us = seq.microseconds()
    for i, t in enumerate(us):
        flux = cv2.cvtColor(seq.read(i), cv2.COLOR_RGB2LUV)[..., 0].astype(np.float64) * (10000 / 255)
        if portrait:
            flux = flux.T
        if i == 0:
            sim.init_image(flux)
            continue
        if not np.any(flux):
            raise ValueError(f"IEBCS cannot advance through a zero-flux frame: {seq.files[i]}")
        e = sim.update(flux, int(t - us[i-1]))
        x, y = (e.get_y(), e.get_x()) if portrait else (e.get_x(), e.get_y())
        yield e.get_ts().astype(np.float64) * 1e-6, x, y, e.get_p().astype(np.int8) * 2 - 1


def dvs_voltmeter(seq, args):
    from src.config import cfg
    from src.simulator import EventSim
    us = seq.microseconds()
    if us[-1] > np.iinfo(np.int32).max:
        raise ValueError("The released DVS-Voltmeter stores signed 32-bit microseconds (35.8 minutes).")
    sim = EventSim(cfg, output_folder=".", video_name="sequence")
    for i, t in enumerate(us):
        e = sim.generate_events(seq.read(i, gray=True), int(t))
        if e is not None:
            yield e[:, 0].astype(np.float64) * 1e-6, e[:, 1], e[:, 2], e[:, 3] * 2 - 1


def slomo_frames(seq, args):
    from PIL import Image
    from v2ecore.slomo import SuperSloMo
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
    weights = str(ROOT / "weights" / WEIGHTS["v2e"][0])
    sim = SuperSloMo(weights, args.v2e_steps == 0, max(2, args.v2e_steps), batch_size=1)
    h, w = seq.h, seq.w
    yield seq.read(0, gray=True), seq.time[0]
    for start in range(0, len(seq.time) - 1, 8):
        end = min(start + 8, len(seq.time) - 1)
        with tempfile.TemporaryDirectory() as tmp:
            inputs, mids = Path(tmp) / "input", Path(tmp) / "mid"
            inputs.mkdir()
            mids.mkdir()
            for j in range(max(3, end - start + 1)):  # Upstream needs two pairs. The extra pair is never emitted.
                im = seq.read(min(start+j, end), gray=True)
                padded = np.pad(im, ((0, -h % 32), (0, -w % 32)), mode="edge")
                np.save(inputs / f"{j:03d}.npy", padded)
            with torch.serialization.safe_globals([datetime]):
                u, _ = sim.interpolate(str(inputs), str(mids), padded.shape[::-1])
            for j in range(end - start):
                indices = np.flatnonzero((u >= j) & (u < j+1))
                t0, t1 = seq.time[start+j:start+j+2]
                for k, index in enumerate(indices):
                    im = cv2.imread(str(mids / f"{index}.png"), cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        raise IOError(f"Missing SuperSloMo output {index}")
                    yield im[:h, :w], t0 + (k + 0.5) / len(indices) * (t1 - t0)
                yield seq.read(start+j+1, gray=True), t1


def v2e(seq, args):
    from v2ecore.emulator import EventEmulator
    sim = EventEmulator(pos_thres=0.2, neg_thres=0.2, sigma_thres=0.03, cutoff_hz=300, leak_rate_hz=0.01,
                        shot_noise_rate_hz=0.001, refractory_period_s=0.0005, seed=args.seed, device=args.device,
                        output_width=seq.w, output_height=seq.h)
    frames = ((seq.read(i, gray=True), t) for i, t in enumerate(seq.time)) if args.v2e_steps == 1 else slomo_frames(seq, args)
    for im, t in frames:
        e = sim.generate_events(np.ascontiguousarray(im), float(t))
        if e is not None:
            times = np.where(e[:, 0] == np.float32(t), t, e[:, 0].astype(np.float64))
            yield times, e[:, 1], e[:, 2], e[:, 3]


def v2ce(seq, args):
    from scripts.v2ce_3d import V2ce3d
    from scripts.LDATI import sample_voxel_statistical, y_relocate
    n = len(seq.time) - 1
    model = V2ce3d().to(args.device).eval()
    weights = ROOT / "weights" / WEIGHTS["v2ce"][0]
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    for first in range(0, n, 16):
        start = min(first, n - 16)  # Match the released overlapping last window; keep each interval once.
        frames = torch.from_numpy(np.stack([seq.read(i, gray=True) for i in range(start, start+17)])).to(args.device)
        images = (frames.float() / 255 - 0.153) / 0.165
        pairs = torch.stack((images[:-1], images[1:]), dim=1)[None]
        voxels = model(pairs).reshape(16, 2, 10, seq.h, seq.w)
        for i in range(first, start+16):
            v = voxels[i-start:i-start+1]
            # The released sampler takes max() on an empty tensor for a zero-count interval.
            if not torch.any(y_relocate(v[0])[0] > 0):
                continue
            dt = seq.time[i+1] - seq.time[i]
            e = sample_voxel_statistical(v, fps=1.0 / dt, additional_events_strategy="slope")[0]
            t = seq.time[i] + e["timestamp"].astype(np.float64) * 1e-6
            yield t, e["x"], e["y"], e["polarity"].astype(np.int8) * 2 - 1


MODELS = {"evis-core": evis_core, "fracevent": fracevent, "iebcs": iebcs,
          "v2e": v2e, "v2ce": v2ce, "dvs-voltmeter": dvs_voltmeter}
