"""Analytical cases and an independent dense calculation of the kernel distance."""

import math
import unittest

import numpy as np

from jsd_work.histogram.events import Events

from .metric import KernelReference


def events(*tuples):
    return Events(*np.asarray(tuples, dtype=np.float64).reshape(-1, 4).T)


def scorer(reference, *, block_size=1024):
    return KernelReference(
        reference, sigma_spatial_px=2, sigma_temporal_ms=1, block_size=block_size,
    )


class KernelDistanceTests(unittest.TestCase):
    def test_single_event_displacements_and_polarity(self):
        ref = scorer(events((10, 10, 0.1, 1)))
        for generated, expected in (
            ((10, 10, 0.1, 1), 0),
            ((11, 10, 0.1, 1), math.sqrt(-2 * math.expm1(-0.125))),
            ((12, 10, 0.1, 1), math.sqrt(-2 * math.expm1(-0.5))),
            ((10, 10, 0.101, 1), math.sqrt(-2 * math.expm1(-0.5))),
            ((10, 10, 0.1, -1), math.sqrt(2)),
        ):
            with self.subTest(generated=generated):
                self.assertAlmostEqual(ref.score(events(generated)).distance, expected)

    def test_tiny_timestamp_shift_survives_float64_cancellation(self):
        ref = scorer(events((0, 0, 0, 1)), block_size=1)
        score = ref.score(events((0, 0, 1e-12, 1)))
        self.assertAlmostEqual(score.distance / 1e-9, 1)

    def test_multiplicity_empty_cases_and_polarity_components(self):
        on, off = (4, 4, 0.1, 1), (4, 4, 0.1, -1)
        ref = scorer(events(on, off))
        extra = ref.score(events(off, on, on, on))
        self.assertEqual((extra.distance, extra.distance_on, extra.distance_off), (2, 2, 0))
        self.assertEqual(ref.score(events(off)).distance, 1)
        self.assertEqual(ref.score(events()).relative_to_empty, 1)
        self.assertEqual(scorer(events()).score(events(on)).distance, 1)
        self.assertIsNone(scorer(events()).score(events(on)).relative_to_empty)
        self.assertEqual(scorer(events()).score(events()).distance, 0)

    def test_blocks_match_dense_formula_including_large_time_gaps(self):
        rng = np.random.default_rng(15)
        gt, generated = (
            np.column_stack((rng.integers(0, 10, (size, 2)),
                             rng.choice([0.001, 0.002, 0.004, 0.11], size),
                             rng.choice([-1, 1], size)))
            for size in (31, 47)
        )

        def kernel_sum(left, right):
            offsets = ((left[:, None, :3] - right[None, :, :3]) / [2, 2, 0.001])
            same_polarity = left[:, None, 3] == right[None, :, 3]
            return np.sum(np.exp(-0.5 * np.sum(offsets ** 2, axis=-1)) * same_polarity)

        expected = math.sqrt(kernel_sum(gt, gt) + kernel_sum(generated, generated)
                             - 2 * kernel_sum(gt, generated))
        for block in (1, 8, 1024):
            with self.subTest(block=block):
                reference = scorer(events(*gt), block_size=block)
                self.assertAlmostEqual(reference.score(events(*generated)).distance, expected)
                swapped = scorer(events(*generated), block_size=block)
                self.assertAlmostEqual(swapped.score(events(*gt)).distance, expected)


if __name__ == "__main__":
    unittest.main()
