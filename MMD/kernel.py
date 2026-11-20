"""Gaussian kernel distance (MMD) between event streams.

For one polarity channel with signed point weights w (GT positive, generated negative)
the squared distance is the kernel energy

    E = sum_i sum_j w_i w_j exp(-0.5 * ((dx/s)^2 + (dy/s)^2 + (dt/st)^2)).

Opposite polarities have zero kernel, so the joint distance is hypot(on, off).

Weightings
    count        w = +1 per GT event, -1 per generated event (van Rossum style; the
                 score depends on event counts and placement).
    probability  w = +1/N_gt, -1/N_gen with N the total count over both polarities
                 (standard biased MMD between the two normalized distributions;
                 bounded by sqrt(2); count differences are left to count metrics).

Algorithms
    exact  every pair, blocks swept in time and stopped where the Gaussian
           underflows (40 temporal sigmas). Reference implementation.
    fast   pairs whose scaled offset exceeds `cutoff_sigma` along any axis are
           skipped using spatial tiles and a time sweep, and tiles run on threads.
           Each skipped pair has kernel < exp(-cutoff^2/2); the reported
           truncation bound covers all of them.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np

WEIGHTINGS = ("count", "probability")
EXACT_TIME_CUTOFF = 40.0


@dataclass(frozen=True)
class Channel:
    points: np.ndarray  # (n, 3): t [s], x [px], y [px]; unique rows, sorted by t, x, y
    gt: np.ndarray      # GT multiplicity of each row
    gen: np.ndarray     # generated multiplicity of each row


def _points(events, polarity):
    keep = events.p == polarity
    return np.column_stack((events.t[keep], events.x[keep], events.y[keep])).astype(np.float64)


def _channel(reference, generated):
    both = np.concatenate((reference, generated))
    if not len(both):
        empty = np.zeros(0, dtype=np.int64)
        return Channel(np.zeros((0, 3)), empty, empty)
    points, inverse = np.unique(both, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    gt = np.bincount(inverse[:len(reference)], minlength=len(points))
    gen = np.bincount(inverse[len(reference):], minlength=len(points))
    return Channel(points, gt, gen)


def _weights(channel, columns, n_gt, n_gen):
    """Stack one weight column per weighting and drop rows that are zero in all."""
    stack = []
    for name in columns:
        if name == "count":
            stack.append((channel.gt - channel.gen).astype(np.float64))
        elif name == "probability":
            stack.append(channel.gt / n_gt - channel.gen / n_gen)
        elif name == "gt_count":
            stack.append(channel.gt.astype(np.float64))
        else:
            raise ValueError(f"Unknown weighting {name!r}")
    weights = np.column_stack(stack) if stack else np.zeros((len(channel.points), 0))
    keep = np.any(weights != 0, axis=1)
    return channel.points[keep], weights[keep]


def _block(left, right, wl, wr):
    """Return (constant, kernel) contributions of sum_ij wl_i K_ij wr_j per column."""
    exponent = np.subtract.outer(left[:, 0], right[:, 0])
    exponent *= exponent
    for axis in (1, 2):
        offset = np.subtract.outer(left[:, axis], right[:, axis])
        offset *= offset
        exponent += offset
    exponent *= -0.5
    constant = None
    if exponent.min() >= -0.5:
        # All pairs are within one sigma: split K = 1 + expm1 to keep tiny displacements.
        constant = wl.sum(axis=0) * wr.sum(axis=0)
        np.expm1(exponent, out=exponent)
    else:
        np.exp(exponent, out=exponent)
    return constant, np.einsum("ik,ik->k", wl, exponent @ wr)


def _collect(sums, constant, kernel, factor=1):
    for k in range(len(sums)):
        if constant is not None:
            sums[k].append(factor * float(constant[k]))
        sums[k].append(factor * float(kernel[k]))


def exact_energy(points, weights, scales, block_size=1024):
    """All pairs, symmetric blocks; matches the original JSD_v2 metric."""
    k = weights.shape[1]
    if not len(points):
        return np.zeros(k)
    scaled = (points - points[0]) / scales
    sums = [[] for _ in range(k)]
    for i in range(0, len(scaled), block_size):
        left, wl = scaled[i:i + block_size], weights[i:i + block_size]
        for j in range(i, len(scaled), block_size):
            right = scaled[j:j + block_size]
            if right[0, 0] - left[-1, 0] >= EXACT_TIME_CUTOFF:
                break
            constant, kernel = _block(left, right, wl, weights[j:j + block_size])
            _collect(sums, constant, kernel, 1 if i == j else 2)
    return np.array([math.fsum(s) for s in sums])


def fast_energy(points, weights, scales, *, cutoff_sigma=10.0, tile_px=32,
                block_size=512, chunk_size=4096, span_factor=1.0, executor=None):
    """Ordered pairs within a box of half-width cutoff_sigma sigmas on every axis."""
    k = weights.shape[1]
    if not len(points):
        return np.zeros(k)
    scaled = (points - points[0]) / scales
    times = scaled[:, 0]
    halo_px = cutoff_sigma * scales[1]
    span = cutoff_sigma * span_factor
    tile = max(int(tile_px), math.ceil(halo_px), 1)
    x = points[:, 1].astype(np.int64)
    y = points[:, 2].astype(np.int64)
    columns = int(x.max()) // tile + 1
    key = (y // tile) * columns + x // tile
    # Stable: rows of each tile stay in time order (row index order).
    order = np.argsort(key, kind="stable")
    tile_ids, starts = np.unique(key[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    ranges = dict(zip(tile_ids.tolist(), zip(starts.tolist(), ends.tolist())))

    def run_tile(tile_id):
        start, end = ranges[tile_id]
        owned = order[start:end]
        a, b = tile_id % columns, tile_id // columns
        parts = [order[slice(*ranges[n])] for db in (-1, 0, 1) for da in (-1, 0, 1)
                 if 0 <= a + da < columns and (n := (b + db) * columns + a + da) in ranges]
        near = np.concatenate(parts)
        nx, ny = points[near, 1], points[near, 2]
        keep = ((nx >= a * tile - halo_px) & (nx < (a + 1) * tile + halo_px)
                & (ny >= b * tile - halo_px) & (ny < (b + 1) * tile + halo_px))
        near = np.sort(near[keep])  # row order is time order
        near_t = times[near]
        sums = [[] for _ in range(k)]
        owned_t = times[owned]
        i = 0
        while i < len(owned):
            # Cap the block's time span too, so sparse tiles do not pull in far neighbours.
            end = min(i + block_size,
                      int(np.searchsorted(owned_t, owned_t[i] + span, "right")))
            rows = owned[i:max(end, i + 1)]
            i += len(rows)
            left, wl = scaled[rows], weights[rows]
            lo = np.searchsorted(near_t, times[rows[0]] - cutoff_sigma, "left")
            hi = np.searchsorted(near_t, times[rows[-1]] + cutoff_sigma, "right")
            for j in range(lo, hi, chunk_size):
                cols = near[j:min(j + chunk_size, hi)]
                _collect(sums, *_block(left, scaled[cols], wl, weights[cols]))
        return sums

    # Largest tiles first so threads finish together.
    tasks = sorted(ranges, key=lambda t: ranges[t][0] - ranges[t][1])
    results = executor.map(run_tile, tasks) if executor else map(run_tile, tasks)
    sums = [[] for _ in range(k)]
    for part in results:
        for column, values in zip(sums, part):
            column.extend(values)
    return np.array([math.fsum(s) for s in sums])


def _distance(energy, weights, bound):
    """sqrt of a kernel energy, rejecting negatives beyond roundoff and truncation."""
    if energy < 0:
        tolerance = 8 * np.finfo(np.float64).eps * float(np.abs(weights).sum()) ** 2 + bound
        if energy < -tolerance:
            raise ArithmeticError("Kernel energy is negative beyond float64 roundoff")
    return math.sqrt(max(0.0, energy))


class MMDReference:
    """Score prepared Events against one GT on a shared clock (seconds, pixels).

    Polarities must be signed (-1 OFF, +1 ON). Select the common interval first.
    """

    def __init__(self, reference, *, sigma_spatial_px, sigma_temporal_ms,
                 weightings=WEIGHTINGS, algorithm="fast", cutoff_sigma=10.0, tile_px=32,
                 block_size=512, chunk_size=4096, threads=1):
        time_s = sigma_temporal_ms / 1000
        if not all(math.isfinite(v) and v > 0 for v in (sigma_spatial_px, time_s)):
            raise ValueError("Gaussian scales must be finite and positive")
        weightings = tuple(dict.fromkeys(weightings))
        if not weightings or any(w not in WEIGHTINGS for w in weightings):
            raise ValueError(f"weightings must be a non-empty subset of {WEIGHTINGS}")
        if algorithm not in ("fast", "exact"):
            raise ValueError("algorithm must be 'fast' or 'exact'")
        for name, value in (("block_size", block_size), ("chunk_size", chunk_size),
                            ("tile_px", tile_px), ("threads", threads)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(cutoff_sigma) or cutoff_sigma < 3:
            raise ValueError("cutoff_sigma must be finite and at least 3")
        self.weightings = weightings
        self.algorithm = algorithm
        self.cutoff_sigma = cutoff_sigma if algorithm == "fast" else EXACT_TIME_CUTOFF
        self._scales = np.array([time_s, sigma_spatial_px, sigma_spatial_px])
        self._options = dict(block_size=block_size)
        if algorithm == "fast":
            self._options.update(cutoff_sigma=cutoff_sigma, tile_px=tile_px,
                                 chunk_size=chunk_size)
        self._threads = threads
        self._reference = self._channels(reference)
        self.n_gt = len(reference)

    @staticmethod
    def _channels(events):
        if not np.all((events.p == -1) | (events.p == 1)):
            raise ValueError("Expected signed event polarities (-1 or +1)")
        return _points(events, 1), _points(events, -1)

    def _energies(self, points, weights):
        if self.algorithm == "exact":
            return exact_energy(points, weights, self._scales, **self._options)
        if self._threads == 1 or len(points) < 4 * self._options["block_size"]:
            return fast_energy(points, weights, self._scales, **self._options)
        with ThreadPoolExecutor(self._threads) as pool:
            return fast_energy(points, weights, self._scales, executor=pool, **self._options)

    def _bound(self, weights):
        """Upper bound on |energy| of the pairs the fast algorithm skips."""
        if self.algorithm == "exact":
            return 0.0
        return math.exp(-0.5 * self.cutoff_sigma ** 2) * float(np.abs(weights).sum()) ** 2

    def _score_channels(self, generated, columns, n_gen):
        energies, bounds = {c: [] for c in columns}, {c: [] for c in columns}
        for reference, prediction in zip(self._reference, generated):
            channel = _channel(reference, prediction)
            points, weights = _weights(channel, columns, self.n_gt, n_gen)
            values = self._energies(points, weights)
            for index, column in enumerate(columns):
                bound = self._bound(weights[:, index])
                energies[column].append(_distance(values[index], weights[:, index], bound))
                bounds[column].append(bound)
        return energies, bounds

    @cached_property
    def empty_baseline(self):
        """Count distance from GT to an empty prediction, (joint, on, off, bound)."""
        empty = (np.zeros((0, 3)), np.zeros((0, 3)))
        distances, bounds = self._score_channels(empty, ("gt_count",), 0)
        on, off = distances["gt_count"]
        return math.hypot(on, off), on, off, math.sqrt(sum(bounds["gt_count"]))

    def score(self, generated):
        """Return a flat dict of count_* and prob_* columns (None where undefined)."""
        predicted = self._channels(generated)
        n_gen = len(generated)
        result = {}
        columns = [w for w in self.weightings
                   if w == "count" or (self.n_gt and n_gen)]
        if "count" in columns and not n_gen:
            columns.remove("count")
            joint, on, off, bound = self.empty_baseline
            result.update(count_distance=joint, count_distance_on=on, count_distance_off=off,
                          count_truncation_bound=bound)
        if columns:
            distances, bounds = self._score_channels(predicted, columns, n_gen)
            prefix = {"count": "count", "probability": "prob"}
            for column in columns:
                on, off = distances[column]
                p = prefix[column]
                result.update({f"{p}_distance": math.hypot(on, off), f"{p}_distance_on": on,
                               f"{p}_distance_off": off,
                               f"{p}_truncation_bound": math.sqrt(sum(bounds[column]))})
        if "count" in self.weightings:
            baseline = self.empty_baseline[0]
            result["count_empty_baseline"] = baseline
            result["count_relative_to_empty"] = (
                result["count_distance"] / baseline if baseline else None)
        if "probability" in self.weightings and "prob_distance" not in result:
            result.update(prob_distance=None, prob_distance_on=None, prob_distance_off=None,
                          prob_truncation_bound=None)
        return result
