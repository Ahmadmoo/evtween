"""Check kernel_v2_fast on a GPU against kernel.py on the CPU: same scores, and how fast.

    python -m MMD.check_device_v2_fast                  # 2M synthetic events per stream
    python -m MMD.check_device_v2_fast --events 5000000

Scores one generator with v2_fast on the device, fused (torch.compile) and element-wise
(--no-compile path), and with kernel.py on the CPU. Prints each score, the relative
differences (should be ~1e-12 or less), seconds and kernel pairs per second.
Exit code 1 if any disagrees. The first fused call includes compile time (a warm-up
on small data runs first).
"""
import argparse
import os
import time

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np  # noqa: E402

from . import kernel, kernel_v2_fast  # noqa: E402
from .check_device import PairCounter, synthetic  # noqa: E402

KEYS = ("count_distance", "prob_distance", "count_empty_baseline")


def difference(a, b):
    return max(abs(a[k] - b[k]) / abs(b[k]) for k in KEYS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--events", type=int, default=2_000_000, help="GT events")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--sigma-spatial-px", type=float, default=2.0)
    parser.add_argument("--sigma-temporal-ms", type=float, default=1.0)
    parser.add_argument("--threads", type=int,
                        default=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                        else os.cpu_count())
    parser.add_argument("--skip-cpu", action="store_true", help="Do not time kernel.py on CPU")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(0)
    shape = args.width, args.height, args.duration_s
    sigmas = dict(sigma_spatial_px=args.sigma_spatial_px,
                  sigma_temporal_ms=args.sigma_temporal_ms)
    worst = 0.0

    small_gt, small_gen = synthetic(rng, 20_000, *shape), synthetic(rng, 18_000, *shape)
    exact = kernel.MMDReference(small_gt, algorithm="exact", **sigmas).score(small_gen)
    for compile in (True, False):
        warm = kernel_v2_fast.MMDReference(small_gt, device=args.device, compile=compile,
                                           threads=args.threads, **sigmas)
        scores = warm.score(small_gen)  # also compiles and warms up the GPU
        worst = max(worst, difference(scores, exact))
        print(f"Small check vs exact (20k events), {warm.device}: "
              f"relative difference {difference(scores, exact):.1e}")

    gt = synthetic(rng, args.events, *shape)
    generated = synthetic(rng, int(args.events * 0.9), *shape)
    print(f"{args.events:,} GT and {len(generated):,} generated events, "
          f"{args.width}x{args.height}, {args.duration_s} s, sigma {args.sigma_spatial_px} px "
          f"/ {args.sigma_temporal_ms} ms")
    results = {}
    for name, make in (
            ("v2 fused", lambda: kernel_v2_fast.MMDReference(
                gt, device=args.device, compile=True, threads=args.threads, **sigmas)),
            ("v2 element-wise", lambda: kernel_v2_fast.MMDReference(
                gt, device=args.device, compile=False, threads=args.threads, **sigmas)),
            ("v1 cpu", None if args.skip_cpu else lambda: kernel.MMDReference(
                gt, threads=args.threads, **sigmas))):
        if make is None:
            continue
        scorer = make()
        scorer.progress = counter = PairCounter()
        start = time.perf_counter()
        scores = scorer.score(generated)
        seconds = time.perf_counter() - start
        results[name] = scores
        print(f"  {name:>16}: {seconds:8.1f} s  {counter.pairs / seconds:9.2g} pairs/s  "
              f"count {scores['count_distance']:.10g}  prob {scores['prob_distance']:.10g}  "
              f"[{scorer.device}]")
    reference = results.get("v1 cpu", results["v2 element-wise"])
    for name, scores in results.items():
        if scores is not reference:
            worst = max(worst, difference(scores, reference))
            print(f"  {name} vs {'v1 cpu' if 'v1 cpu' in results else 'v2 element-wise'}: "
                  f"relative difference {difference(scores, reference):.1e}")
    ok = worst < 1e-9
    print("OK" if ok else "MISMATCH: do not use this device for scoring")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
