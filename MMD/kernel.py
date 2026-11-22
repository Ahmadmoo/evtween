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
           underflows (40 temporal sigmas). Reference implementation, CPU only.
    fast   pairs whose scaled offset exceeds `cutoff_sigma` along any axis are
           skipped using spatial tiles and a time sweep, and tiles run on threads.
           Each skipped pair has kernel < exp(-cutoff^2/2); the reported
           truncation bound covers all of them.
           device="cpu" runs the blocks with NumPy threads; device="cuda" runs the same
           blocks in float64 on a GPU through PyTorch (device_energy). The GPU path has
           no expm1 split for blocks within one sigma, which only matters for
           displacements below ~1e-7 sigma.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np

WEIGHTINGS = ("count", "probability")
EXACT_TIME_CUTOFF = 40.0
# Bump when a change alters scores; run folders are keyed on it instead of on the code.
METRIC_VERSION = 1
# Tile and block sizes per backend: they change speed and memory, never the pairs scored.
SIZES = {"cpu": dict(tile_px=32, block_size=512, chunk_size=4096),
         "device": dict(tile_px=64, block_size=4096, chunk_size=32768)}


def make_ops(device):
    """None for the threaded NumPy path ("cpu"); otherwise the device_energy backend."""
    if device == "cpu":
        return None
    if device == "numpy":
        return NumpyOps()
    if device == "torch-cpu":
        return TorchOps("cpu")
    if device == "cuda" or device.startswith("cuda:"):
        return TorchOps(device)
    raise ValueError(f"Unknown device {device!r}; use cpu, cuda or cuda:N")


@dataclass(frozen=True)
class Channel:
    points: np.ndarray  # (n, 3): t [s], x [px], y [px]; unique rows, sorted by t, x, y
    gt: np.ndarray      # GT multiplicity of each row
    gen: np.ndarray     # generated multiplicity of each row


def _points(events, polarity):
    keep = events.p == polarity
    return np.column_stack((events.t[keep], events.x[keep], events.y[keep])).astype(np.float64)


def _channel(reference, generated):
    """Merge identical (t, x, y) rows, sorted by t, then x, then y, with multiplicities.

    Same result as np.unique(axis=0) but much faster on time-sorted streams: a stable sort
    on t (two sorted runs), then on one integer key (t rank, x, y) that is nearly sorted.
    """
    both = np.concatenate((reference, generated))
    if not len(both):
        empty = np.zeros(0, dtype=np.int64)
        return Channel(np.zeros((0, 3)), empty, empty)
    order = np.argsort(both[:, 0], kind="stable")
    times = both[order, 0]
    rank = np.zeros(len(times), dtype=np.int64)
    np.cumsum(times[1:] != times[:-1], out=rank[1:])
    x = both[order, 1].astype(np.int64)
    y = both[order, 2].astype(np.int64)
    height = int(y.max()) + 1
    key = rank * ((int(x.max()) + 1) * height) + x * height + y
    order = order[np.argsort(key, kind="stable")]
    ordered = both[order]
    new = np.ones(len(ordered), dtype=bool)
    new[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    starts = np.flatnonzero(new)
    gt = np.add.reduceat((order < len(reference)).astype(np.int64), starts)
    gen = np.diff(np.append(starts, len(ordered))) - gt
    return Channel(ordered[starts], gt, gen)


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


class _Plan:
    """Spatial tiles of the fast algorithm (only to find neighbours; every event is scored).

    Each row is owned by one tile. A tile's neighbours are the rows of the 3 x 3 tiles
    around it that lie within `cutoff_sigma` spatial sigmas of the tile, so no pair closer
    than the cutoff is missed at a tile border.
    """

    def __init__(self, points, scales, cutoff_sigma, tile_px):
        self.scaled = (points - points[0]) / scales
        self.times = self.scaled[:, 0]
        self.cutoff = cutoff_sigma
        self.halo_px = cutoff_sigma * scales[1]
        self.tile = max(int(tile_px), math.ceil(self.halo_px), 1)
        self.x, self.y = points[:, 1], points[:, 2]
        key_x = self.x.astype(np.int64) // self.tile
        self.columns = int(key_x.max()) + 1
        key = (self.y.astype(np.int64) // self.tile) * self.columns + key_x
        # Stable: rows of each tile stay in time order (row index order).
        self.order = np.argsort(key, kind="stable")
        ids, starts = np.unique(key[self.order], return_index=True)
        ends = np.append(starts[1:], len(self.order))
        self.ranges = dict(zip(ids.tolist(), zip(starts.tolist(), ends.tolist())))

    def tiles(self):
        """Largest tiles first so parallel workers finish together."""
        return sorted(self.ranges, key=lambda t: self.ranges[t][0] - self.ranges[t][1])

    def members(self, tile_id):
        """(owned rows, neighbour rows), both in time order."""
        start, end = self.ranges[tile_id]
        a, b = tile_id % self.columns, tile_id // self.columns
        parts = [self.order[slice(*self.ranges[n])] for db in (-1, 0, 1) for da in (-1, 0, 1)
                 if 0 <= a + da < self.columns
                 and (n := (b + db) * self.columns + a + da) in self.ranges]
        near = np.concatenate(parts)
        nx, ny, tile, halo = self.x[near], self.y[near], self.tile, self.halo_px
        keep = ((nx >= a * tile - halo) & (nx < (a + 1) * tile + halo)
                & (ny >= b * tile - halo) & (ny < (b + 1) * tile + halo))
        return self.order[start:end], np.sort(near[keep])

    def blocks(self, owned, near, block_size, chunk_size, span_factor):
        """(row start, row end, column start, column end) into owned / near."""
        owned_t, near_t = self.times[owned], self.times[near]
        span = self.cutoff * span_factor
        i = 0
        while i < len(owned):
            # Cap the block's time span too, so sparse tiles do not pull in far neighbours.
            end = max(i + 1, min(i + block_size,
                                 int(np.searchsorted(owned_t, owned_t[i] + span, "right"))))
            lo = int(np.searchsorted(near_t, owned_t[i] - self.cutoff, "left"))
            hi = int(np.searchsorted(near_t, owned_t[end - 1] + self.cutoff, "right"))
            for j in range(lo, hi, chunk_size):
                yield i, end, j, min(j + chunk_size, hi)
            i = end


def fast_energy(points, weights, scales, *, cutoff_sigma=10.0, tile_px=32,
                block_size=512, chunk_size=4096, span_factor=1.0, executor=None):
    """CPU: ordered pairs within a box of half-width cutoff_sigma sigmas on every axis."""
    k = weights.shape[1]
    if not len(points):
        return np.zeros(k)
    plan = _Plan(points, scales, cutoff_sigma, tile_px)

    def run_tile(tile_id):
        owned, near = plan.members(tile_id)
        left, wl = plan.scaled[owned], weights[owned]
        right, wr = plan.scaled[near], weights[near]
        sums = [[] for _ in range(k)]
        for r0, r1, c0, c1 in plan.blocks(owned, near, block_size, chunk_size, span_factor):
            _collect(sums, *_block(left[r0:r1], right[c0:c1], wl[r0:r1], wr[c0:c1]))
        return sums

    tasks = plan.tiles()
    results = executor.map(run_tile, tasks) if executor else map(run_tile, tasks)
    sums = [[] for _ in range(k)]
    for part in results:
        for column, values in zip(sums, part):
            column.extend(values)
    return np.array([math.fsum(s) for s in sums])


class NumpyOps:
    """Array backend of device_energy on the CPU (reference for the GPU backend)."""

    name = "numpy"

    def put(self, array):
        return np.ascontiguousarray(array, dtype=np.float64)

    def kernel_rows(self, left, right, wr):
        exponent = np.subtract.outer(left[:, 0], right[:, 0])
        exponent *= exponent
        for axis in (1, 2):
            offset = np.subtract.outer(left[:, axis], right[:, axis])
            offset *= offset
            exponent += offset
        exponent *= -0.5
        np.exp(exponent, out=exponent)
        return exponent @ wr

    def block_sum(self, wl, rows):
        return np.einsum("ik,ik->k", wl, rows)

    def stack(self, parts, k):
        return np.array(parts, dtype=np.float64).reshape(-1, k)

    def concatenate(self, parts, k):
        return np.concatenate(parts) if parts else np.zeros((0, k))


class TorchOps(NumpyOps):
    """PyTorch backend: the same blocks in float64 on a GPU ("cuda", "cuda:1") or "cpu"."""

    def __init__(self, device):
        try:
            import torch
        except ImportError:
            raise RuntimeError(f"device={device!r} needs PyTorch in this environment") from None
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
            if self.device.index is not None:
                torch.cuda.set_device(self.device)
            self.name = torch.cuda.get_device_name(self.device)
        else:
            self.name = f"torch-{self.device.type}"

    def put(self, array):
        array = np.ascontiguousarray(array, dtype=np.float64)
        return self.torch.from_numpy(array).to(self.device)

    def kernel_rows(self, left, right, wr):
        # Exact differences (no |a|^2 + |b|^2 - 2ab shortcut), squared and exponentiated in place.
        kernel = self.torch.cdist(left, right, compute_mode="donot_use_mm_for_euclid_dist")
        kernel.square_().mul_(-0.5).exp_()
        return kernel @ wr

    def block_sum(self, wl, rows):
        return (wl * rows).sum(dim=0)

    def stack(self, parts, k):
        if not parts:
            return self.torch.zeros((0, k), dtype=self.torch.float64, device=self.device)
        return self.torch.stack(parts)

    def concatenate(self, parts, k):
        if not parts:
            return np.zeros((0, k))
        return self.torch.cat(parts).cpu().numpy()


def _ahead(pool, function, items, depth):
    """pool.map that prepares at most `depth` items ahead of the consumer."""
    pending = deque()
    for item in items:
        pending.append(pool.submit(function, item))
        if len(pending) > depth:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def device_energy(points, weights, scales, *, ops, cutoff_sigma=10.0, tile_px=64,
                  block_size=4096, chunk_size=32768, span_factor=1.0, prefetch=2):
    """Same pairs as fast_energy; blocks run through `ops` (a GPU with TorchOps).

    CPU threads find each tile's rows and block bounds ahead of the device; every block's
    sum stays on the device until the end and is then added exactly (math.fsum).
    """
    k = weights.shape[1]
    if not len(points):
        return np.zeros(k)
    plan = _Plan(points, scales, cutoff_sigma, tile_px)

    def prepare(tile_id):
        owned, near = plan.members(tile_id)
        bounds = list(plan.blocks(owned, near, block_size, chunk_size, span_factor))
        return plan.scaled[owned], weights[owned], plan.scaled[near], weights[near], bounds

    parts = []
    with ThreadPoolExecutor(max(1, prefetch)) as pool:
        for left, wl, right, wr, bounds in _ahead(pool, prepare, plan.tiles(), max(1, prefetch)):
            left, wl, right, wr = (ops.put(a) for a in (left, wl, right, wr))
            sums = [ops.block_sum(wl[r0:r1], ops.kernel_rows(left[r0:r1], right[c0:c1], wr[c0:c1]))
                    for r0, r1, c0, c1 in bounds]
            parts.append(ops.stack(sums, k))
    values = ops.concatenate(parts, k)
    return np.array([math.fsum(values[:, c]) for c in range(k)])


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
                 weightings=WEIGHTINGS, algorithm="fast", cutoff_sigma=10.0, tile_px=None,
                 block_size=None, chunk_size=None, threads=1, device="cpu"):
        time_s = sigma_temporal_ms / 1000
        if not all(math.isfinite(v) and v > 0 for v in (sigma_spatial_px, time_s)):
            raise ValueError("Gaussian scales must be finite and positive")
        weightings = tuple(dict.fromkeys(weightings))
        if not weightings or any(w not in WEIGHTINGS for w in weightings):
            raise ValueError(f"weightings must be a non-empty subset of {WEIGHTINGS}")
        if algorithm not in ("fast", "exact"):
            raise ValueError("algorithm must be 'fast' or 'exact'")
        if algorithm == "exact" and device != "cpu":
            raise ValueError("The exact algorithm runs on the CPU only")
        sizes = SIZES["cpu" if device == "cpu" else "device"]
        tile_px = sizes["tile_px"] if tile_px is None else tile_px
        block_size = sizes["block_size"] if block_size is None else block_size
        chunk_size = sizes["chunk_size"] if chunk_size is None else chunk_size
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
        self._ops = make_ops(device) if algorithm == "fast" else None
        self.device = self._ops.name if self._ops else "cpu"
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
        if self._ops is not None:
            return device_energy(points, weights, self._scales, ops=self._ops,
                                 prefetch=min(self._threads, 4), **self._options)
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
