"""Check a GPU against the CPU on synthetic events: same scores, and how much faster.

Run this first on a new machine or GPU type:

    python -m MMD.check_device                    # cuda vs cpu, 2M events per stream
    python -m MMD.check_device --events 5000000   # bigger, closer to real sequences

It prints both scores, their relative difference (should be ~1e-12 or less) and the
time of each. Exit code 1 if they disagree.
"""
import argparse
import os
import time

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np  # noqa: E402

from .events import Events  # noqa: E402
from .kernel import MMDReference  # noqa: E402


def synthetic(rng, n, width, height, duration):
    """Events clustered around a moving vertical edge, roughly like a real scene."""
    t = np.sort(rng.uniform(0, duration, n))
    x = ((t * width / 2 + rng.normal(0, width / 16, n)) % width).astype(np.int64)
    y = rng.integers(0, height, n)
    return Events(x, y, t, rng.choice([-1, 1], n).astype(np.int8))


class PairCounter:
    """Progress hook that only adds up the kernel pairs evaluated."""

    def __init__(self):
        self.done, self.current = 0, 0

    def stage(self, label, events):
        self.done, self.current = self.done + self.current, 0

    def __call__(self, done, total, pairs):
        self.current = pairs

    @property
    def pairs(self):
        return self.done + self.current


def timed(function):
    start = time.perf_counter()
    value = function()
    return value, time.perf_counter() - start


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
    parser.add_argument("--skip-cpu", action="store_true", help="Only time the device")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(0)
    shape = args.width, args.height, args.duration_s
    gt = synthetic(rng, args.events, *shape)
    generated = synthetic(rng, int(args.events * 0.9), *shape)
    sigmas = dict(sigma_spatial_px=args.sigma_spatial_px,
                  sigma_temporal_ms=args.sigma_temporal_ms)

    small_gt, small_gen = synthetic(rng, 20_000, *shape), synthetic(rng, 18_000, *shape)
    exact = MMDReference(small_gt, algorithm="exact", **sigmas).score(small_gen)
    device = MMDReference(small_gt, device=args.device, threads=args.threads, **sigmas)
    warm = device.score(small_gen)  # also warms up the GPU
    print(f"Device: {device.device}")
    worst = max(abs(warm[k] - exact[k]) / exact[k] for k in ("count_distance", "prob_distance"))
    print(f"Small check vs exact (20k events): relative difference {worst:.1e}")

    print(f"{args.events:,} GT and {len(generated):,} generated events, "
          f"{args.width}x{args.height}, {args.duration_s} s")
    on_device = MMDReference(gt, device=args.device, threads=args.threads, **sigmas)
    on_device.progress = counter = PairCounter()
    scores, seconds = timed(lambda: on_device.score(generated))
    print(f"  {args.device:>8}: {seconds:8.1f} s   count {scores['count_distance']:.10g}   "
          f"prob {scores['prob_distance']:.10g}   ({counter.pairs / seconds:.2g} pairs/s)")
    if not args.skip_cpu:
        reference, cpu_seconds = timed(lambda: MMDReference(gt, threads=args.threads, **sigmas)
                                       .score(generated))
        print(f"  {'cpu':>8}: {cpu_seconds:8.1f} s   count {reference['count_distance']:.10g}   "
              f"prob {reference['prob_distance']:.10g}   ({args.threads} threads)")
        difference = max(abs(scores[k] - reference[k]) / reference[k]
                         for k in ("count_distance", "prob_distance"))
        worst = max(worst, difference)
        print(f"Relative difference: {difference:.1e}; speedup x{cpu_seconds / seconds:.1f}")
    ok = worst < 1e-9
    print("OK" if ok else "MISMATCH: do not use this device for scoring")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
