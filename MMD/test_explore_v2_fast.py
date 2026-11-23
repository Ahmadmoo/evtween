"""explore_v2_fast helpers; the plots are built with real plotly where it is installed.

    python -m unittest MMD.test_explore_v2_fast -v
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import explore_v2_fast as ex
from .benchmark_io import Reference, load_prediction
from .kernel_v2_fast import MMDReference

HAS_PLOTLY = importlib.util.find_spec("plotly") is not None


def make_runs(root, sigmas=(1, 2)):
    """A small benchmark (2 sequences, 2 generators) scored by run_v2_fast at each sigma."""
    from PIL import Image

    from . import run_v2_fast
    from .benchmark_io import benchmark

    rng = np.random.default_rng(0)
    for index, sequence in enumerate(("s1", "s2")):
        source = root / "data" / "eds" / "test" / sequence
        (source / "frames").mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (40, 30)).save(source / "frames" / f"{i:04}.png")
        np.save(source / "frame_ts.npy", [5.0, 5.2])
        n = 3000 + 500 * index
        t = np.sort(rng.uniform(5, 5.2, n))
        x, y, p = rng.integers(0, 40, n), rng.integers(0, 30, n), rng.integers(0, 2, n)
        for key, array in dict(x=x, y=y, t=t, p=p).items():
            np.save(source / f"ev_{key}.npy", array)
        _, _, _, metadata, _ = benchmark.v2.sequence_info(source)
        for generator, jitter, save in (("near", 0.3, np.savez),
                                        ("far", 3.0, np.savez_compressed)):
            folder = root / "pred" / "eds" / "test" / generator / sequence
            folder.mkdir(parents=True)
            (folder / "input.json").write_text(json.dumps(metadata))
            times = np.clip(t - 5 + rng.normal(0, jitter * 1e-3, n), 0, 0.2 - 1e-9)
            order = np.argsort(times, kind="stable")
            save(folder / "events.npz", x=x[order], y=y[order], t=times[order],
                 p=np.where(p[order] > 0, 1, -1))
    folders = []
    for sigma in sigmas:
        folders.append(root / "runs" / f"eds-s{sigma}")
        run_v2_fast.main(["eds", "test", "--sigma-spatial-px", str(sigma), "--sigma-temporal-ms",
                          "1", "--data-root", str(root / "data"), "--pred-root",
                          str(root / "pred"), "--output", str(folders[-1]), "--jobs", "1",
                          "--threads", "1", "--progress-every", "0"])
    return folders


class ExploreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.folders = make_runs(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_runs_scores_and_tables(self):
        runs = ex.runs_from_folders(self.folders, "eds", "test")
        self.assertEqual(list(runs.sigma_spatial_px), [1.0, 2.0])
        scores = ex.load_scores(runs)
        self.assertEqual(len(scores), 8)
        self.assertLess(ex.check_terms(scores), 1e-12)
        self.assertEqual(ex.metric_table(scores).shape, (2, 9))
        self.assertEqual(ex.terms_table(scores).shape, (4, 6))
        with self.assertRaisesRegex(ValueError, "not hqevfi"):
            ex.runs_from_folders(self.folders, "hqevfi", "test")
        with self.assertRaisesRegex(ValueError, "Same sigmas"):
            ex.runs_from_folders(self.folders + self.folders[:1], "eds", "test")

    def test_window_equals_full_loading(self):
        reference = Reference(self.root / "data" / "eds" / "test" / "s2")
        for generator in ("near", "far"):  # stored and compressed .npz
            window = ex.load_window(self.folders[0], "s2", generator, 0.05, 40)
            prediction = self.root / "pred" / "eds" / "test" / generator / "s2" / "events.npz"
            metadata = reference.align(prediction, type("Args", (), dict(dataset="eds",
                                                                         split="test")), "s2")
            full = (reference.select(tuple(metadata["crop"]), 0.05, 0.09),
                    load_prediction(prediction, metadata).select(0.05, 0.09))
            for got, want in zip((window["gt"], window["generated"]), full):
                for a, b in zip((got.x, got.y, got.t, got.p), (want.x, want.y, want.t, want.p)):
                    np.testing.assert_array_equal(a, b)

    def test_event_terms_match_kernel(self):
        window = ex.load_window(self.folders[0], "s1", "near", None, 50)
        terms = ex.event_terms(window, 2, 1, cutoff_sigma=10)
        score = MMDReference(window["gt"], sigma_spatial_px=2, sigma_temporal_ms=1).score(
            window["generated"])
        n, m = len(window["gt"].t), len(window["generated"].t)
        expected = dict(a=(score["sum_gt_gt_on"] + score["sum_gt_gt_off"]) / n ** 2,
                        b=(score["sum_gen_gen_on"] + score["sum_gen_gen_off"]) / m ** 2,
                        c=(score["sum_gt_gen_on"] + score["sum_gt_gen_off"]) / (n * m))
        for key, value in expected.items():
            self.assertAlmostEqual(terms[key], value, delta=1e-12 * value)
        self.assertAlmostEqual(terms["similarity"], score["similarity"], delta=1e-12)
        total = terms["gt_difference"].sum() / n - terms["gen_difference"].sum() / m
        self.assertAlmostEqual(total, terms["a"] + terms["b"] - 2 * terms["c"],
                               delta=1e-12 * terms["a"])

    def test_row_sums_against_dense(self):
        rng = np.random.default_rng(1)
        rows, cols = (np.column_stack((np.sort(rng.uniform(0, 0.01, n)), rng.integers(0, 30, n),
                                       rng.integers(0, 30, n))).astype(float) for n in (300, 250))
        scales = np.array([0.001, 2.0, 2.0])
        dense = np.exp(-0.5 * (((rows[:, None] - cols[None]) / scales) ** 2).sum(-1)).sum(1)
        got = ex.row_sums(rows, cols, scales, cutoff_sigma=10, block_size=17, chunk_size=23)
        np.testing.assert_allclose(got, dense, rtol=1e-12)

    @unittest.skipUnless(HAS_PLOTLY, "plotly is not installed")
    def test_plots_build_with_plotly(self):
        for folders, kind in ((self.folders, "scatter"), (self.folders[:1], "bar")):
            scores = ex.load_scores(ex.runs_from_folders(folders, "eds", "test"))
            for fig in (ex.metric_vs_sigma(scores), ex.terms_vs_sigma(scores)):
                fig.to_html(full_html=False, include_plotlyjs=False)  # validates properties
                self.assertEqual({trace.type for trace in fig.data}, {kind})
                for button in fig.layout.updatemenus[0].buttons:
                    self.assertEqual(len(button.args[0]["visible"]), len(fig.data))
        window = ex.load_window(self.folders[1], "s1", "far", None, 30)
        fig, _ = ex.point_cloud(window, sigmas_px=(1, 2), max_points=500)
        fig.to_html(full_html=False, include_plotlyjs=False)
        self.assertEqual(len(fig.layout.updatemenus[0].buttons), 8)


if __name__ == "__main__":
    unittest.main()
