"""kernel_v2_fast against kernel.py, its PyTorch backends, and run_v2_fast end to end.

    python -m unittest MMD.test_v2_fast -v
"""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import kernel, kernel_v2_fast
from .events import Events
from .test_mmd import MOVED, events, has_torch, make_benchmark, random_events, read_results

KEYS = ("count_distance", "count_distance_on", "count_distance_off", "count_empty_baseline",
        "count_relative_to_empty", "prob_distance", "prob_distance_on", "prob_distance_off")
SIGMAS = dict(sigma_spatial_px=2, sigma_temporal_ms=1)


def close(a, b, tolerance=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tolerance * max(abs(b), 1e-300)


def overlapping(rng, n):
    """GT and a prediction that shares half of its events exactly."""
    gt = random_events(rng, n)
    extra = random_events(rng, n // 2)
    keep = n // 2
    order = np.argsort(np.r_[gt.t[:keep], extra.t], kind="stable")
    return gt, Events(*(np.r_[a[:keep], b][order] for a, b in
                        zip((gt.x, gt.y, gt.t, gt.p), (extra.x, extra.y, extra.t, extra.p))))


class AgreementTests(unittest.TestCase):
    def assert_same(self, options, cases):
        for index, (gt, gen) in enumerate(cases):
            expected = kernel.MMDReference(gt, **SIGMAS).score(gen)
            got = kernel_v2_fast.MMDReference(gt, **SIGMAS, **options).score(gen)
            self.assertEqual(set(got), set(expected) | set(kernel_v2_fast.SUM_COLUMNS))
            for key in KEYS:
                with self.subTest(options=options, case=index, key=key):
                    self.assertTrue(close(got[key], expected[key]), (got[key], expected[key]))

    def cases(self):
        rng = np.random.default_rng(21)
        return [(random_events(rng, 3000), random_events(rng, 2500)),
                overlapping(rng, 3000),
                (random_events(rng, 1500), random_events(rng, 1))]

    def test_matches_kernel_py_on_cpu(self):
        self.assert_same(dict(), self.cases())
        self.assert_same(dict(threads=3, tile_px=4, block_size=33, chunk_size=97), self.cases())

    def test_matches_kernel_py_on_device_path(self):
        self.assert_same(dict(device="numpy"), self.cases())
        self.assert_same(dict(device="numpy", tile_px=8, block_size=40, chunk_size=150),
                         self.cases())

    @unittest.skipUnless(has_torch(), "PyTorch is not installed")
    def test_torch_backends(self):
        import torch

        devices = [("torch-cpu", {}), ("torch-cpu-compiled", {})]
        if torch.cuda.is_available():
            devices += [("cuda", dict(compile=False)), ("cuda", dict(compile=True))]
        failed = []
        for device, extra in devices:
            options = dict(device=device, **extra, block_size=64, chunk_size=512)
            self.assert_same(options, self.cases())
            name = kernel_v2_fast.MMDReference(events((1, 1, 0.0, 1)), **SIGMAS,
                                               **options).device
            if "compile failed" in name:
                failed.append(name)
        if failed:
            self.skipTest(f"Scores agree, but torch.compile fell back: {failed}")

    def test_identical_and_empty(self):
        rng = np.random.default_rng(5)
        gt = random_events(rng, 4000)
        for options in (dict(), dict(device="numpy")):
            score = kernel_v2_fast.MMDReference(gt, **SIGMAS, **options).score(gt)
            self.assertEqual((score["count_distance"], score["prob_distance"]), (0, 0))
        empty = Events(*(a[:0] for a in (gt.x, gt.y, gt.t, gt.p)))
        for a, b in ((gt, empty), (empty, gt), (empty, empty)):
            expected = kernel.MMDReference(a, **SIGMAS).score(b)
            got = kernel_v2_fast.MMDReference(a, **SIGMAS).score(b)
            for key in expected:
                self.assertTrue(close(got[key], expected[key]), (key, got[key], expected[key]))

    def test_each_channel_takes_the_cheaper_way(self):
        rng = np.random.default_rng(8)
        gt = random_events(rng, 4000)
        keep = np.arange(len(gt)) % 10 != 0  # 90% of GT: rows together are cheaper
        subset = Events(*(a[keep] for a in (gt.x, gt.y, gt.t, gt.p)))
        half = Events(*(a[::2] for a in (gt.x, gt.y, gt.t, gt.p)))  # 50%: sums are cheaper
        on_only = gt.p == 1
        mixed = Events(*(np.r_[a[on_only], b] for a, b in zip(
            (gt.x, gt.y, gt.t, gt.p), (np.array([3]), np.array([3]), np.array([0.01]),
                                       np.array([-1], dtype=np.int8)))))
        order = np.argsort(mixed.t, kind="stable")
        mixed = Events(*(a[order] for a in (mixed.x, mixed.y, mixed.t, mixed.p)))
        for generated, paths in ((random_events(rng, 3500), ["sums", "sums"]),
                                 (subset, ["together", "together"]),
                                 (half, ["sums", "sums"]),
                                 (mixed, ["identical", "sums"])):
            for options in (dict(), dict(device="numpy")):
                scorer = kernel_v2_fast.MMDReference(gt, **SIGMAS, **options)
                got = scorer.score(generated)
                expected = kernel.MMDReference(gt, **SIGMAS).score(generated)
                with self.subTest(paths=paths, options=options):
                    self.assertEqual(scorer.paths, paths)
                    for key in KEYS:
                        self.assertTrue(close(got[key], expected[key]),
                                        (key, got[key], expected[key]))

    def test_sums_and_similarity_on_every_path(self):
        """The three sums against a dense O(n^2) sum, for each way a channel is scored."""
        rng = np.random.default_rng(9)

        def dense(a, b, polarity):
            pa, pb = ([np.column_stack((e.t[e.p == polarity], e.x[e.p == polarity],
                                        e.y[e.p == polarity])).astype(float)] for e in (a, b))
            offsets = (pa[0][:, None] - pb[0][None]) / [0.001, 2, 2]
            return np.exp(-0.5 * (offsets ** 2).sum(-1)).sum()

        gt = random_events(rng, 800)
        keep = np.arange(len(gt)) % 10 != 0
        on = gt.p == 1
        cases = ((random_events(rng, 700), ["sums", "sums"]),
                 (Events(*(a[keep] for a in (gt.x, gt.y, gt.t, gt.p))),
                  ["together", "together"]),
                 (Events(*(a[on] for a in (gt.x, gt.y, gt.t, gt.p))), ["identical", "sums"]))
        for generated, paths in cases:
            for options in (dict(), dict(device="numpy", tile_px=4, block_size=30,
                                         chunk_size=70)):
                scorer = kernel_v2_fast.MMDReference(gt, **SIGMAS, **options)
                score = scorer.score(generated)
                self.assertEqual(scorer.paths, paths)
                totals = np.zeros(3)
                for polarity, tag in ((1, "on"), (-1, "off")):
                    want = [dense(gt, gt, polarity), dense(generated, generated, polarity),
                            dense(gt, generated, polarity)]
                    totals += want
                    got = [score[f"sum_gt_gt_{tag}"], score[f"sum_gen_gen_{tag}"],
                           score[f"sum_gt_gen_{tag}"]]
                    for g, w in zip(got, want):
                        with self.subTest(paths=paths, options=options, tag=tag):
                            self.assertTrue(close(g, w, 1e-12) or (w == 0 and abs(g) < 1e-12),
                                            (g, w))
                similarity = totals[2] / np.sqrt(totals[0] * totals[1])
                self.assertAlmostEqual(score["similarity"], similarity, delta=1e-12)
                self.assertTrue(0 <= score["similarity"] <= 1 + 1e-12)
        self.assertEqual(kernel_v2_fast.MMDReference(gt, **SIGMAS).score(gt)["similarity"], 1)
        empty = Events(*(a[:0] for a in (gt.x, gt.y, gt.t, gt.p)))
        score = kernel_v2_fast.MMDReference(gt, **SIGMAS).score(empty)
        self.assertIsNone(score["similarity"])
        self.assertEqual((score["sum_gen_gen_on"], score["sum_gt_gen_on"]), (0, 0))
        self.assertGreater(score["sum_gt_gt_on"], 0)

    def test_single_weighting(self):
        rng = np.random.default_rng(6)
        gt, gen = random_events(rng, 500), random_events(rng, 400)
        for weightings in (("count",), ("probability",)):
            expected = kernel.MMDReference(gt, **SIGMAS, weightings=weightings).score(gen)
            got = kernel_v2_fast.MMDReference(gt, **SIGMAS, weightings=weightings).score(gen)
            self.assertEqual(set(got), set(expected) | set(kernel_v2_fast.SUM_COLUMNS))

    def test_progress_and_gt_sum_once(self):
        rng = np.random.default_rng(7)
        gt = random_events(rng, 2000)

        class Recorder:
            def __init__(self):
                self.stages, self.last = [], {}

            def stage(self, label, events):
                self.stages.append(label)

            def __call__(self, done, total, pairs):
                self.last[self.stages[-1]] = (done, total)

        scorer = kernel_v2_fast.MMDReference(gt, **SIGMAS, device="numpy")
        scorer.progress = first = Recorder()
        scorer.score(random_events(rng, 1500))
        scorer.progress = second = Recorder()
        scorer.score(random_events(rng, 1500))
        sums = ["ON with itself", "ON with GT", "OFF with itself", "OFF with GT"]
        self.assertEqual(second.stages, ["ON merge", sums[0], sums[1], "OFF merge", sums[2],
                                         sums[3]])
        self.assertEqual(first.stages[:4], ["GT ON merge", "GT ON with GT", "GT OFF merge",
                                            "GT OFF with GT"])
        for label in sums:
            done, total = second.last[label]
            self.assertAlmostEqual(done, total)


class RunnerTests(unittest.TestCase):
    def test_runner_resume_and_separate_folder(self):
        from . import run_v2_fast

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_benchmark(root, ("seq",), (("match", 20.0, 0), ("moved", 20.0, 1),
                                            ("wrong_clock", 19.0, 0)))
            command = ["eds", "test", "--sigma-spatial-px", "1", "--sigma-temporal-ms", "1",
                       "--data-root", str(root / "data"), "--pred-root", str(root / "pred"),
                       "--jobs", "1", "--threads", "1"]
            with mock.patch.object(run_v2_fast, "HERE", root):
                self.assertEqual(run_v2_fast.main(command), 1)  # wrong_clock is unavailable
                (folder,) = (root / "runs").iterdir()
                self.assertIn("-v2-", folder.name)
                config = json.loads((folder / "config.json").read_text())
                self.assertEqual((config["engine"], config["output_version"]),
                                 ("v2_fast", kernel_v2_fast.OUTPUT_VERSION))
                rows = {r["generator"]: r for r in read_results(folder / "results.csv")}
                self.assertEqual((rows["match"]["status"], rows["match"]["algorithm"]),
                                 ("ok", "v2_fast"))
                self.assertEqual(float(rows["match"]["count_distance"]), 0)
                self.assertEqual(float(rows["match"]["similarity"]), 1)
                self.assertAlmostEqual(float(rows["moved"]["count_distance"]), MOVED)
                self.assertAlmostEqual(float(rows["moved"]["similarity"]), math.exp(-0.5))
                self.assertGreater(float(rows["moved"]["sum_gt_gt_on"]), 0)
                self.assertEqual(rows["wrong_clock"]["status"], "alignment_unavailable")
                try:
                    import pandas  # noqa: F401
                except ImportError:
                    pass
                else:
                    from . import summarize_v2_fast
                    self.assertEqual(summarize_v2_fast.main([str(folder), "--no-plot"]), 0)
                    self.assertTrue((folder / "wins_similarity.csv").is_file())
                stamp = (folder / "rows" / "seq" / "match.json").stat().st_mtime_ns
                run_v2_fast.main(command)
                self.assertEqual((folder / "rows" / "seq" / "match.json").stat().st_mtime_ns,
                                 stamp)  # resumed

    def test_gpu_scheduling_with_stand_in_devices(self):
        from . import run_v2_fast

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_benchmark(root, ("s1", "s2"), (("match", 20.0, 0), ("moved", 20.0, 1)))
            command = ["eds", "test", "--sigma-spatial-px", "1", "--sigma-temporal-ms", "1",
                       "--data-root", str(root / "data"), "--pred-root", str(root / "pred"),
                       "--output", str(root / "run"), "--device", "cuda", "--jobs", "2",
                       "--threads", "1"]
            with mock.patch.object(run_v2_fast, "visible_gpus", return_value=["numpy", "numpy"]):
                self.assertEqual(run_v2_fast.main(command), 0)
            rows = read_results(root / "run" / "results.csv")
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual((row["status"], row["device"]), ("ok", "numpy"))
                expected = 0 if row["generator"] == "match" else MOVED
                self.assertAlmostEqual(float(row["count_distance"]), expected)


if __name__ == "__main__":
    unittest.main()
