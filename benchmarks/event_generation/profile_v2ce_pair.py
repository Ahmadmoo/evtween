"""Per-frame-pair inference profile of V2CE, measured like profile_pair.py.

The upstream network runs on a whole clip at once (video_to_voxels), so its cost per
pair is the clip time divided by the number of pairs; LDATI sampling runs pair by pair
exactly as in v2ce_interval_worker. Stages:

  network   frames from disk + V2CE network, per pair (video_to_voxels / pairs)
  simulate  LDATI event sampling of one pair (sample_voxel_statistical + time decoding)

Usage: profile_v2ce_pair.py ASSETS SEQUENCE PAIRS WARMUP OUT_JSON
SEQUENCE holds frames/*.png and frame_ts.npy with at least 17 frames.
"""
import importlib.util
import json
import logging
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    assets, seq_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    pairs, warmup, out = int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]).resolve()
    import torch
    sys.path.insert(0, str(HERE))
    from v2ce_interval_worker import decode_times
    if not torch.cuda.is_available():
        raise RuntimeError("The installed V2CE network requires CUDA")
    sync = torch.cuda.synchronize

    start = time.perf_counter()
    source = assets / "third_party/v2ce"
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("_v2ce_upstream", source / "v2ce.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.logger = logging.getLogger("V2CE")
    torch.manual_seed(123)
    np.random.seed(123)
    model = module.get_trained_mode(str(assets / "weights/v2ce_3d.pt"))
    sync()
    load_s = time.perf_counter() - start

    ts = np.load(seq_dir / "frame_ts.npy")
    paths = sorted(map(str, (seq_dir / "frames").glob("*.png")))
    if len(paths) != len(ts) or len(ts) < max(17, warmup + pairs + 1):
        raise ValueError(f"Need at least {max(17, warmup + pairs + 1)} frames with timestamps")
    run = lambda: module.video_to_voxels(model, image_paths=paths, batch_size=1)

    def sample(voxels, i):
        ev = module.sample_voxel_statistical(torch.as_tensor(voxels[i:i + 1], device="cuda"), fps=30.,
                                             bidirectional=False, additional_events_strategy="slope")[0]
        t, keep = decode_times(ev["timestamp"], ts[i], ts[i + 1])
        return int(keep.sum())

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        voxels = run()  # warm-up of the network (cuDNN autotuning, caches)
        for i in range(warmup):
            sample(voxels, i)
        clip_ms = []
        for _ in range(3):
            sync()
            a = time.perf_counter()
            voxels = run()
            sync()
            clip_ms.append(1e3 * (time.perf_counter() - a))
        per_pair = float(np.median(clip_ms)) / (len(ts) - 1)
        times = {"network": [], "simulate": []}
        events = []
        for i in range(warmup, warmup + pairs):
            sync()
            a = time.perf_counter()
            events.append(sample(voxels, i))
            sync()
            times["simulate"].append(1e3 * (time.perf_counter() - a))
            times["network"].append(per_pair)

    flops, flop_note = None, ""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        flop_note = f"torch {torch.__version__} has no torch.utils.flop_counter (needs >= 2.1)"
    else:
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            run()
        flops = {"network": int(counter.get_total_flops()) // (len(ts) - 1)}
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            sample(voxels, warmup)
        flops["simulate"] = int(counter.get_total_flops())
        flop_note = "matmul/conv/attention only; network FLOPs are the clip total divided by its pairs"

    params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else None
    record = dict(mode="v2ce", device="cuda", gpu=torch.cuda.get_device_name(),
                  cpu=platform.processor() or platform.machine(), torch=torch.__version__,
                  clip_frames=len(ts), clip_ms=clip_ms, pairs=pairs, warmup=warmup, load_s=load_s,
                  times_ms=times, events=events, peak_memory_gb=torch.cuda.max_memory_allocated() / 2**30,
                  params={"network": params} if params is not None else {}, flops=flops, flop_note=flop_note,
                  loaded_but_unused=[])
    out.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
