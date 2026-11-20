"""Gaussian kernel distance between event counting measures."""

from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np
from scipy.spatial.distance import cdist

from jsd_work.histogram.events import Events


@dataclass(frozen=True)
class KernelScore:
    distance: float
    distance_on: float
    distance_off: float
    empty_baseline: float
    relative_to_empty: float | None


def _points(events, polarity):
    keep = events.p == polarity
    return np.column_stack((events.t[keep], events.x[keep], events.y[keep]))


def _support(reference, generated=None):
    """Combine duplicate weights and cancel exact cross-stream matches."""
    if generated is None:
        return np.unique(reference, axis=0, return_counts=True)
    points, inverse = np.unique(
        np.concatenate((reference, generated)), axis=0, return_inverse=True
    )
    inverse = inverse.reshape(-1)
    counts = (
        np.bincount(inverse[:len(reference)], minlength=len(points))
        - np.bincount(inverse[len(reference):], minlength=len(points))
    )
    keep = counts != 0
    return points[keep], counts[keep]


def _kernel_distance(points, counts, scales, block_size):
    if not len(points):
        return 0.0
    # np.unique sorts time first. Use a local origin to preserve timestamp precision.
    scaled = (points - points[0]) / scales
    if not np.isfinite(scaled).all():
        raise ValueError("Scaled event coordinates must be finite")
    weights = counts.astype(np.float64)
    signed = bool(np.any(counts < 0) and np.any(counts > 0))

    def block_contributions():
        for i in range(0, len(scaled), block_size):
            left = scaled[i:i + block_size]
            left_weights = weights[i:i + block_size]
            for j in range(i, len(scaled), block_size):
                right = scaled[j:j + block_size]
                # These Gaussian terms round to zero in float64.
                if right[0, 0] - left[-1, 0] >= 40:
                    break
                right_weights = weights[j:j + block_size]
                factor = 1 if i == j else 2
                exponent = cdist(left, right, metric="sqeuclidean")
                exponent *= -0.5
                if signed and exponent.min() >= -0.5:
                    # Separate the constant kernel to preserve tiny displacements.
                    yield factor * float(left_weights.sum() * right_weights.sum())
                    np.expm1(exponent, out=exponent)
                else:
                    np.exp(exponent, out=exponent)
                exponent *= left_weights[:, None]
                exponent *= right_weights
                yield factor * float(exponent.sum())

    energy = math.fsum(block_contributions())
    if energy < 0:
        tolerance = 8 * np.finfo(np.float64).eps * float(np.abs(weights).sum()) ** 2
        if energy < -tolerance:
            raise ArithmeticError("Kernel energy is negative beyond float64 roundoff")
    return math.sqrt(max(0.0, energy))


class KernelReference:
    """Score prepared Events on a shared clock in seconds and pixel coordinates.

    Event polarities must be signed (-1 for OFF, +1 for ON). Prepare both
    streams and select the common interval before scoring.
    """

    def __init__(
        self, reference: Events, *, sigma_spatial_px: float,
        sigma_temporal_ms: float, block_size: int = 1024,
    ):
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        time_s = sigma_temporal_ms / 1000
        if not all(math.isfinite(v) and v > 0 for v in (sigma_spatial_px, time_s)):
            raise ValueError("Gaussian scales must be finite and positive")
        self._scales = np.array([time_s, sigma_spatial_px, sigma_spatial_px])
        self._block_size = block_size
        self._reference = self._channels(reference)

    @staticmethod
    def _channels(events):
        if not np.all((events.p == -1) | (events.p == 1)):
            raise ValueError("Expected signed event polarities (-1 or +1)")
        return _points(events, 1), _points(events, -1)

    def _distance(self, points, counts):
        return _kernel_distance(points, counts, self._scales, self._block_size)

    @cached_property
    def _empty_distances(self):
        return tuple(self._distance(*_support(channel)) for channel in self._reference)

    def score(self, generated: Events) -> KernelScore:
        """Return the joint and per-polarity distances plus the empty baseline."""
        predicted = self._channels(generated)
        if len(generated):
            on, off = (
                self._distance(*_support(reference, prediction))
                for reference, prediction in zip(self._reference, predicted)
            )
        else:
            on, off = self._empty_distances
        distance = math.hypot(on, off)
        baseline = math.hypot(*self._empty_distances)
        return KernelScore(distance, on, off, baseline, distance / baseline if baseline else None)
