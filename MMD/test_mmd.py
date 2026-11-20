"""Analytical cases, a dense reference, fast/exact agreement and the benchmark runner.

    python -m unittest MMD.test_mmd -v
"""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .events import Events
from .kernel import MMDReference


def events(*tuples):
    return Events(*np.asarray(tuples, dtype=np.float64).reshape(-1, 4).T)


def random_events(rng, n, width=50, height=40, duration=0.05):
    return Events(rng.integers(0, width, n), rng.integers(0, height, n),
                  np.sort(rng.uniform(0, duration, n)), rng.choice([-1, 1], n).astype(np.int8))


def dense(gt, gen, sigma_s, sigma_t_ms, probability):
    """Independent O(n^2) formula on the raw (non-deduplicated) events."""
    total = 0.0
    for polarity in (1, -1):
        a, b = ([np.column_stack((e.t[e.p == polarity], e.x[e.p == polarity],
                                  e.y[e.p == polarity])).astype(float)] for e in (gt, gen))
        points = np.vstack(a + b)
        scale_gt = 1 / len(gt) if probability else 1
        scale_gen = 1 / len(gen) if probability else 1
        w = np.r_[np.full(len(a[0]), scale_gt), np.full(len(b[0]), -scale_gen)]
        offsets = (points[:, None] - points[None]) / [sigma_t_ms / 1000, sigma_s, sigma_s]
        total += w @ np.exp(-0.5 * (offsets ** 2).sum(-1)) @ w
    return math.sqrt(max(total, 0.0))


def scorer(reference, **options):
    return MMDReference(reference, sigma_spatial_px=2, sigma_temporal_ms=1, **options)


class AnalyticalTests(unittest.TestCase):
    def test_single_event_displacements_and_polarity(self):
        for algorithm in ("exact", "fast"):
            ref = scorer(events((10, 10, 0.1, 1)), algorithm=algorithm)
            for generated, expected in (
                ((10, 10, 0.1, 1), 0),
                ((11, 10, 0.1, 1), math.sqrt(-2 * math.expm1(-0.125))),
                ((12, 10, 0.1, 1), math.sqrt(-2 * math.expm1(-0.5))),
                ((10, 10, 0.101, 1), math.sqrt(-2 * math.expm1(-0.5))),
                ((10, 10, 0.1, -1), math.sqrt(2)),
            ):
                with self.subTest(algorithm=algorithm, generated=generated):
                    score = ref.score(events(generated))
                    self.assertAlmostEqual(score["count_distance"], expected)
                    # One event each: both weightings coincide.
                    self.assertAlmostEqual(score["prob_distance"], expected)

    def test_tiny_timestamp_shift_survives_cancellation(self):
        for algorithm in ("exact", "fast"):
            ref = scorer(events((0, 0, 0, 1)), algorithm=algorithm, block_size=1)
            score = ref.score(events((0, 0, 1e-12, 1)))
            self.assertAlmostEqual(score["count_distance"] / 1e-9, 1)

    def test_empty_cases_and_count_vs_probability(self):
        on, off = (4, 4, 0.1, 1), (4, 4, 0.1, -1)
        ref = scorer(events(on, off))
        doubled = ref.score(events(on, on, off, off))
        self.assertAlmostEqual(doubled["count_distance"], math.sqrt(2))
        self.assertAlmostEqual(doubled["prob_distance"], 0)  # same shape, twice the events
        empty = ref.score(events())
        self.assertEqual(empty["count_relative_to_empty"], 1)
        self.assertIsNone(empty["prob_distance"])
        self.assertIsNone(scorer(events()).score(events(on))["count_relative_to_empty"])
        self.assertEqual(scorer(events()).score(events())["count_distance"], 0)

    def test_weighting_selection(self):
        only = scorer(events((1, 1, 0.0, 1)), weightings=("probability",))
        self.assertEqual(set(only.score(events((2, 1, 0.0, 1)))),
                         {"prob_distance", "prob_distance_on", "prob_distance_off",
                          "prob_truncation_bound"})


class AgreementTests(unittest.TestCase):
    def test_exact_and_fast_match_dense_formula(self):
        rng = np.random.default_rng(7)
        for trial in range(8):
            gt = random_events(rng, int(rng.integers(50, 400)))
            gen = random_events(rng, int(rng.integers(1, 400)))
            sigma_s, sigma_t = float(rng.choice([0.5, 2])), float(rng.choice([0.5, 3]))
            expected = {p: dense(gt, gen, sigma_s, sigma_t, p) for p in (False, True)}
            for options in (dict(algorithm="exact", block_size=17),
                            dict(algorithm="fast", block_size=13, chunk_size=29, threads=3),
                            dict(algorithm="fast", cutoff_sigma=4, tile_px=2, block_size=9)):
                score = MMDReference(gt, sigma_spatial_px=sigma_s, sigma_temporal_ms=sigma_t,
                                     **options).score(gen)
                for key, probability in (("count", False), ("prob", True)):
                    with self.subTest(trial=trial, options=options, key=key):
                        error = abs(score[f"{key}_distance"] - expected[probability])
                        allowed = score[f"{key}_truncation_bound"] + 1e-9 * expected[probability]
                        self.assertLessEqual(error, allowed + 1e-12)

    def test_threads_do_not_change_the_score(self):
        rng = np.random.default_rng(3)
        gt, gen = random_events(rng, 3000), random_events(rng, 2500)
        one = scorer(gt, block_size=64).score(gen)
        many = scorer(gt, block_size=64, threads=4).score(gen)
        for key in ("count_distance", "prob_distance"):
            self.assertAlmostEqual(one[key], many[key], delta=1e-12 * one[key])


class RunnerTests(unittest.TestCase):
    def test_runner_alignment_and_resume(self):
        from PIL import Image

        from . import run
        from .benchmark_io import benchmark

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data" / "eds" / "test" / "seq"
            (source / "frames").mkdir(parents=True)
            for i in range(2):
                Image.new("RGB", (8, 6)).save(source / "frames" / f"{i:04}.png")
            np.save(source / "frame_ts.npy", [20.0, 21.0])
            raw = dict(x=np.array([1, 3, 5]), y=np.array([0, 2, 4]),
                       t=np.array([20.25, 20.5, 20.75]), p=np.array([0, 1, 1]))
            for key, array in raw.items():
                np.save(source / f"ev_{key}.npy", array)
            _, _, _, metadata, _ = benchmark.v2.sequence_info(source)
            for model, origin, shift in (("match", 20.0, 0), ("moved", 20.0, 1),
                                         ("wrong_clock", 19.0, 0)):
                path = root / "pred" / "eds" / "test" / model / "seq"
                path.mkdir(parents=True)
                (path / "input.json").write_text(json.dumps(dict(metadata, absolute_t0=origin)))
                np.savez(path / "events.npz", x=raw["x"] + shift, y=raw["y"], t=raw["t"] - 20,
                         p=np.array([-1, 1, 1]))
            command = ["eds", "test", "--sigma-spatial-px", "1", "--sigma-temporal-ms", "1",
                       "--data-root", str(root / "data"), "--pred-root", str(root / "pred"),
                       "--output", str(root / "run"), "--jobs", "1", "--threads", "1"]
            self.assertEqual(run.main(command), 1)  # wrong_clock is unavailable
            with (root / "run" / "results.csv").open(newline="") as stream:
                rows = {r["generator"]: r for r in csv.DictReader(stream)}
            self.assertEqual(rows["match"]["status"], "ok")
            self.assertEqual(float(rows["match"]["count_distance"]), 0)
            self.assertEqual(rows["match"]["gt_source"], str(source))
            self.assertAlmostEqual(float(rows["moved"]["count_distance"]),
                                   math.sqrt(3 * -2 * math.expm1(-0.5)))
            self.assertEqual(rows["wrong_clock"]["status"], "alignment_unavailable")
            stamp = (root / "run" / "rows" / "seq" / "match.json").stat().st_mtime_ns
            run.main(command)
            self.assertEqual((root / "run" / "rows" / "seq" / "match.json").stat().st_mtime_ns,
                             stamp)  # resumed, not rescored
            changed = list(command)
            changed[3] = "2"  # --sigma-spatial-px
            with self.assertRaisesRegex(SystemExit, "different settings"):
                run.main(changed)


if __name__ == "__main__":
    unittest.main()
