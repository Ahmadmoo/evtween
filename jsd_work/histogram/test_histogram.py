"""Numerical checks and a focused test of the benchmark entry point."""
import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from .config import Config, DISTRIBUTIONS
from .distributions import Grid, histograms
from .events import prepare_events
from .metrics import count_error, divergences


def events(x=(), y=(), t=(), p=(), **options):
    arrays = dict(x=np.asarray(x, dtype=np.int32), y=np.asarray(y, dtype=np.int32),
                  t=np.asarray(t, dtype=np.float64), p=np.asarray(p, dtype=np.int8))
    return prepare_events(arrays, 5, 3, unit=options.pop("unit", "s"), **options)


class HistogramTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(5, 3, 0, 1, 1, 0.25)
        self.gt = events([0, 4, 4], [0, 2, 2], [0, 0.25, 0.75], [1, -1, 1])
        self.gen = events([1, 4, 4], [0, 2, 2], [0.0, 0.25, 0.75], [-1, -1, 1])

    def test_identity_and_reproducibility(self):
        for name in DISTRIBUTIONS:
            p = histograms(self.gt, self.grid, [name])[name]
            q = histograms(self.gt, self.grid, [name])[name]
            assert_array_equal(p.ids, q.ids)
            assert_array_equal(p.counts, q.counts)
            config = Config(reverse_kl=True)
            score = divergences(p, q, config)
            self.assertEqual((score["jsd"], score["kl_gt_gen"], score["kl_gen_gt"]), (0, 0, 0))
            self.assertEqual(score, divergences(p, q, config))

    def test_support_symmetry_and_independent_reference(self):
        for name in DISTRIBUTIONS:
            p = histograms(self.gt, self.grid, [name])[name]
            q = histograms(self.gen, self.grid, [name])[name]
            config = Config(reverse_kl=True, normalize_jsd=False)
            first = divergences(p, q, config)
            second = divergences(q, p, config)
            a, b = (np.zeros(math.prod(p.shape)) for _ in range(2))
            a[p.ids] = p.probabilities()
            b[q.ids] = q.probabilities()
            self.assertAlmostEqual(first["jsd"], second["jsd"])
            self.assertAlmostEqual(first["jsd"], jensenshannon(a, b) ** 2)
            mix = config.kl_epsilon
            a, b = ((1 - mix) * v + mix / v.size for v in (a, b))
            self.assertAlmostEqual(first["kl_gt_gen"], entropy(a, b))
            self.assertAlmostEqual(first["kl_gen_gt"], second["kl_gt_gen"])

    def test_disjoint_support_and_unsmoothed_direction(self):
        p = histograms(events([0], [0], [0], [-1]), self.grid, ["polarity"])["polarity"]
        q = histograms(events([0], [0], [0], [1]), self.grid, ["polarity"])["polarity"]
        self.assertEqual(divergences(p, q, Config())["jsd"], 1)
        self.assertEqual(divergences(p, q, Config(kl_epsilon=0))["kl_gt_gen"], math.inf)
        r = histograms(self.gen, self.grid, ["polarity"])["polarity"]
        asymmetric = divergences(p, r, Config(kl_epsilon=0, reverse_kl=True, log_base="2"))
        self.assertAlmostEqual(asymmetric["kl_gt_gen"], -math.log2(2 / 3))
        self.assertEqual(asymmetric["kl_gen_gt"], math.inf)

    def test_pixel_resolution_and_mass(self):
        h = histograms(self.gt, self.grid, DISTRIBUTIONS)
        self.assertEqual(h["spatial"].shape, (3, 5))
        assert_array_equal(h["spatial"].ids, [0, 14])
        assert_array_equal(h["spatial"].counts, [1, 2])
        for histogram in h.values():
            self.assertAlmostEqual(histogram.probabilities().sum(), 1)
        grid = replace(self.grid, spatial_bin_size=2)
        self.assertEqual(grid.shape("spatial"), (2, 3))
        assert_array_equal(grid.spatial_positions(np.array([5])), [[4.5, 2.5]])

    def test_shared_half_open_edges_and_partial_last_bin(self):
        grid = Grid(5, 3, 0.125, 0.8, 4, 0.25)
        a = events([0] * 5, [0] * 5, [0.124, 0.125, 0.374, 0.375, 0.8], [1] * 5)
        b = events([0] * 3, [0] * 3, [0.25, 0.375, 0.799], [1] * 3)
        p, q = (histograms(s.select(grid.start_s, grid.stop_s), grid, ["temporal"])["temporal"]
                for s in (a, b))
        self.assertEqual(p.shape, q.shape)
        assert_array_equal(p.ids, [0, 1])
        assert_array_equal(q.ids, [0, 1, 2])
        assert_array_equal(p.counts, [2, 1])
        np.testing.assert_allclose(grid.temporal_positions(q.ids), [0.25, 0.5, 0.7125])
        self.assertEqual(Grid(5, 3, 0, 0.9, 1, 0.3).nt, 3)

    def test_sparse_with_huge_logical_grid(self):
        grid = Grid(10_000, 10_000, 0, 3600, 1, 0.001)
        h = histograms(self.gt, grid, ["spatiotemporal"])["spatiotemporal"]
        self.assertEqual(math.prod(h.shape), 360_000_000_000_000)
        self.assertEqual(h.ids.size, 3)
        self.assertLess(h.nbytes, 100)

    def test_joint_structure_with_identical_marginals(self):
        a = events([0, 4], [0, 0], [0.1, 0.6], [1, -1])
        b = events([4, 0], [0, 0], [0.1, 0.6], [1, -1])
        p, q = (histograms(s, self.grid, DISTRIBUTIONS) for s in (a, b))
        scores = {name: divergences(p[name], q[name], Config())["jsd"]
                  for name in DISTRIBUTIONS}
        self.assertEqual(scores, dict(spatial=0, temporal=0, polarity=0, spatiotemporal=1, joint=1))

    def test_empty_stream_and_independent_count(self):
        empty = histograms(events(), self.grid, ["temporal"])["temporal"]
        gt = histograms(self.gt, self.grid, ["temporal"])["temporal"]
        self.assertEqual(empty.probabilities().size, 0)
        score = divergences(gt, empty, Config(comparisons=("jsd", "kl", "wasserstein")))
        self.assertNotIn("jsd", score)
        self.assertNotIn("kl_gt_gen", score)
        self.assertIsNone(score["wasserstein"])
        self.assertEqual(score["wasserstein_status"], "empty_stream")
        self.assertEqual(count_error(0, 0, 1e-12)["relative_count_error"], 0)
        self.assertEqual(count_error(0, 3, 1e-12)["relative_count_error"], 3e12)
        repeated = events([0] * 3, [0] * 3, [0] * 3, [1] * 3)
        singleton = events([0], [0], [0], [1])
        p, q = (histograms(s, self.grid, ["spatiotemporal"])["spatiotemporal"]
                for s in (singleton, repeated))
        self.assertEqual(divergences(p, q, Config())["jsd"], 0)
        self.assertEqual(count_error(1, 3, 1e-12)["relative_count_error"], 2)

    def test_1d_and_2d_wasserstein_units_and_guard(self):
        p, q = (histograms(s, self.grid, DISTRIBUTIONS)
                for s in (events([0], [0], [0], [1]), events([3], [2], [0.5], [1])))
        config = Config(comparisons=("wasserstein",))
        score = lambda name, settings=config: divergences(p[name], q[name], settings)
        self.assertAlmostEqual(score("spatial")["wasserstein"], math.sqrt(13))
        self.assertAlmostEqual(score("temporal")["wasserstein"], 0.5)
        self.assertEqual(score("spatiotemporal")["wasserstein_status"], "not_applicable")
        multi = histograms(self.gt, self.grid, ["spatial"])["spatial"]
        limited = divergences(multi, multi, replace(config, max_spatial_transport_pairs=1))
        self.assertEqual(limited["wasserstein_status"], "spatial_transport_limit")

    def test_preprocessing_clock_polarity_and_exact_duplicates(self):
        raw = events([1, 0, 1, 0], [0] * 4, [100, 100, 100, 200], [1, 0, 1, 0],
                     unit="ms", origin=100, polarity="binary")
        assert_array_equal(raw.t, [0, 0, 0, 0.1])
        assert_array_equal(raw.p, [1, -1, 1, -1])
        self.assertEqual(raw.duplicates(), 1)
        for invalid in (
            dict(x=[5], y=[0], t=[0], p=[1]),
            dict(x=[0], y=[0], t=[math.nan], p=[1]),
            dict(x=[0, 0], y=[0, 0], t=[0, 0], p=[0, -1], polarity="benchmark"),
        ):
            with self.assertRaises(ValueError):
                events(**invalid)


class BenchmarkTests(unittest.TestCase):
    def test_verified_pair_and_metadata_mismatch(self):
        from PIL import Image

        from .benchmark_io import benchmark
        from .run_benchmark import run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data" / "eds" / "test" / "test_sequence"
            frames = source / "frames"
            frames.mkdir(parents=True)
            for i in range(2):
                Image.new("RGB", (5, 3)).save(frames / f"{i:04}.png")
            np.save(source / "frame_ts.npy", [20.0, 21.0])
            raw = dict(x=np.array([1, 3]), y=np.array([0, 2]), t=np.array([20.25, 20.75]),
                       p=np.array([0, 1]))
            for key, array in raw.items():
                np.save(source / f"ev_{key}.npy", array)
            _, _, _, metadata, _ = benchmark.v2.sequence_info(source)
            for model, origin in (("match", 20.0), ("wrong_clock", 19.0), ("cropped", 20.0)):
                path = root / "pred" / "eds" / "test" / model / "test_sequence"
                path.mkdir(parents=True)
                sidecar = dict(metadata, absolute_t0=origin)
                prediction = dict(x=raw["x"], y=raw["y"], t=raw["t"] - 20, p=np.array([-1, 1]))
                if model == "cropped":
                    sidecar.update(crop=[1, 0, 2, 3], width=2)
                    prediction = {key: array[:1] for key, array in prediction.items()}
                    prediction["x"] = prediction["x"] - 1
                (path / "input.json").write_text(json.dumps(sidecar))
                np.savez(path / "events.npz", **prediction)
            args = argparse.Namespace(dataset="eds", split="test", data_root=root / "data",
                                      pred_root=root / "pred", output=root / "run",
                                      sequence=["test_sequence"], methods=["match", "wrong_clock", "cropped"])
            rows = run(args, Config(save_distributions=True))
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(rows[0]["spatiotemporal_jsd"], 0)
            self.assertEqual(rows[0]["n_gt"], 2)
            self.assertEqual(rows[1]["status"], "alignment_unavailable")
            self.assertIn("absolute_t0", rows[1]["detail"])
            self.assertEqual(rows[2]["status"], "alignment_unavailable")
            with (args.output / "results.csv").open(newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 3)
            with np.load(args.output / rows[0]["distribution_path"]) as saved:
                self.assertEqual(saved["spatiotemporal_gt_counts"].sum(), 2)
            args.output = root / "cropped-run"
            args.methods = ["cropped"]
            (row,) = run(args, Config(allow_benchmark_crop=True))
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["spatiotemporal_jsd"], 0)
            self.assertEqual(row["gt_outside_crop"], 1)


if __name__ == "__main__":
    unittest.main()
