"""v2_fast: the same Gaussian kernel distance as kernel.py, with less work per generator.

kernel.py scores GT and prediction together, so every generator recomputes all GT-GT
pairs. Here each polarity uses three sums of kernel values over merged identical events
(c = multiplicity, k = Gaussian of kernel.py):

    S_gg = sum_ij c_i c_j k(g_i, g_j)    GT with GT        (once per sequence)
    S_pp = sum_ij c_i c_j k(p_i, p_j)    prediction with itself
    S_gp = sum_ij c_i c_j k(g_i, p_j)    GT with prediction

    count:        D_p^2 = S_gg + S_pp - 2 S_gp
    probability:  D_p^2 = S_gg / N^2 + S_pp / M^2 - 2 S_gp / (N M)

with N, M the GT and generated event counts over both polarities. This is the formula of
kernel.py, regrouped; per generator about half the pairs are evaluated. S_gg is also the
squared empty baseline. When a prediction channel equals the GT channel the distance is
exactly 0.

If a prediction shares many events exactly with GT (e.g. a subset of GT), kernel.py's way
is cheaper, because identical events cancel before any pair is evaluated. Each channel
takes the way with fewer estimated pairs: (unique GT + prediction rows)^2 for kernel.py's,
M^2 + N M for the sums above (S_gg is already known). Otherwise the result matches kernel.py to roundoff of the sums (~1e-12
relative), which only matters for predictions far closer to GT than any real generator.

On a GPU, `compile=True` fuses distance, exp and the weighted row sum into one kernel with
torch.compile, so no block x chunk matrix is written to GPU memory. If compiling fails,
it falls back to the element-wise kernel of kernel.py and says so in `device`.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
import math
import warnings

import numpy as np

from .kernel import (SIZES, WEIGHTINGS, NumpyOps, TorchOps, _ahead, _block, _channel,
                     _distance, _points, _weights, device_energy, fast_energy)

ENGINE = "v2_fast"
SIZES_V2 = {"cpu": SIZES["cpu"], "device": SIZES["device"],
            # Fused kernel: no block x chunk matrix in memory, so take bigger chunks.
            "compiled": dict(tile_px=64, block_size=4096, chunk_size=131072)}


def unique_rows(points):
    """Identical (t, x, y) rows merged: unique rows sorted by t, x, y, and multiplicities."""
    if not len(points):
        return np.zeros((0, 3)), np.zeros(0)
    order = np.argsort(points[:, 0], kind="stable")
    times = points[order, 0]
    rank = np.zeros(len(times), dtype=np.int64)
    np.cumsum(times[1:] != times[:-1], out=rank[1:])
    x = points[order, 1].astype(np.int64)
    y = points[order, 2].astype(np.int64)
    height = int(y.max()) + 1
    key = rank * ((int(x.max()) + 1) * height) + x * height + y
    order = order[np.argsort(key, kind="stable")]
    ordered = points[order]
    new = np.ones(len(ordered), dtype=bool)
    new[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    starts = np.flatnonzero(new)
    counts = np.diff(np.append(starts, len(ordered))).astype(np.float64)
    return ordered[starts], counts


class PairPlan:
    """Spatial tiles for pairs (row point, column point) from two time-sorted point sets.

    Each row belongs to one tile; its candidate columns are the columns of the 3 x 3 tiles
    around it within `cutoff_sigma` spatial sigmas, so no pair inside the cutoff is missed.
    """

    def __init__(self, rows, cols, scales, cutoff_sigma, tile_px):
        origin = rows[0] if rows[0, 0] <= cols[0, 0] else cols[0]
        self.rows = (rows - origin) / scales
        self.cols = (cols - origin) / scales
        self.cutoff = cutoff_sigma
        self.halo_px = cutoff_sigma * scales[1]
        self.tile = max(int(tile_px), math.ceil(self.halo_px), 1)
        self.columns = int(max(rows[:, 1].max(), cols[:, 1].max())) // self.tile + 1
        self.cols_x, self.cols_y = cols[:, 1], cols[:, 2]
        self.row_order, self.row_ranges = self._index(rows)
        self.col_order, self.col_ranges = self._index(cols)

    def _index(self, points):
        key = ((points[:, 2].astype(np.int64) // self.tile) * self.columns
               + points[:, 1].astype(np.int64) // self.tile)
        order = np.argsort(key, kind="stable")  # time order kept inside each tile
        ids, starts = np.unique(key[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        return order, dict(zip(ids.tolist(), zip(starts.tolist(), ends.tolist())))

    def _around(self, tile_id):
        a, b = tile_id % self.columns, tile_id // self.columns
        return [(b + db) * self.columns + a + da for db in (-1, 0, 1) for da in (-1, 0, 1)
                if 0 <= a + da < self.columns]

    def work(self):
        """Rough relative cost of each row tile, for progress."""
        counts = {t: end - start for t, (start, end) in self.col_ranges.items()}
        return {t: float(end - start) * sum(counts.get(n, 0) for n in self._around(t))
                for t, (start, end) in self.row_ranges.items()}

    def tiles(self):
        return sorted(self.row_ranges, key=lambda t: self.row_ranges[t][0] - self.row_ranges[t][1])

    def members(self, tile_id):
        """(row indices, column indices), both in time order."""
        start, end = self.row_ranges[tile_id]
        parts = [self.col_order[slice(*self.col_ranges[n])]
                 for n in self._around(tile_id) if n in self.col_ranges]
        if not parts:
            return self.row_order[start:end], np.zeros(0, dtype=np.int64)
        near = np.concatenate(parts)
        a, b = tile_id % self.columns, tile_id // self.columns
        nx, ny, tile, halo = self.cols_x[near], self.cols_y[near], self.tile, self.halo_px
        keep = ((nx >= a * tile - halo) & (nx < (a + 1) * tile + halo)
                & (ny >= b * tile - halo) & (ny < (b + 1) * tile + halo))
        return self.row_order[start:end], np.sort(near[keep])

    def blocks(self, owned, near, block_size, chunk_size, span_factor):
        row_t, col_t = self.rows[owned, 0], self.cols[near, 0]
        span = self.cutoff * span_factor
        i = 0
        while i < len(owned):
            end = max(i + 1, min(i + block_size,
                                 int(np.searchsorted(row_t, row_t[i] + span, "right"))))
            lo = int(np.searchsorted(col_t, row_t[i] - self.cutoff, "left"))
            hi = int(np.searchsorted(col_t, row_t[end - 1] + self.cutoff, "right"))
            for j in range(lo, hi, chunk_size):
                yield i, end, j, min(j + chunk_size, hi)
            i = end


def pair_sum(rows, row_weights, cols, col_weights, scales, *, ops=None, executor=None,
             cutoff_sigma=10.0, tile_px=32, block_size=512, chunk_size=4096, span_factor=1.0,
             prefetch=2, report=None):
    """sum_i sum_j w_i w_j k(row_i, col_j) over pairs within cutoff_sigma on every axis.

    ops=None: NumPy blocks on CPU threads (executor). Otherwise blocks go through `ops`
    (kernel.NumpyOps / TorchOps / CompiledTorchOps). Block sums are added exactly.
    report(done, total, pairs) follows the estimated fraction of the work.
    """
    if not len(rows) or not len(cols):
        return 0.0
    plan = PairPlan(rows, cols, scales, cutoff_sigma, tile_px)
    row_w, col_w = row_weights[:, None], col_weights[:, None]
    work = plan.work()
    total, done, pairs = sum(work.values()), 0.0, 0
    values = []

    def bounds_of(owned, near):
        return list(plan.blocks(owned, near, block_size, chunk_size, span_factor))

    if ops is None:
        def run_tile(tile_id):
            owned, near = plan.members(tile_id)
            left, wl = plan.rows[owned], row_w[owned]
            right, wr = plan.cols[near], col_w[near]
            sums, count = [], 0
            for r0, r1, c0, c1 in bounds_of(owned, near):
                constant, kernel = _block(left[r0:r1], right[c0:c1], wl[r0:r1], wr[c0:c1])
                if constant is not None:
                    sums.append(float(constant[0]))
                sums.append(float(kernel[0]))
                count += (r1 - r0) * (c1 - c0)
            return sums, count

        if executor:
            futures = {executor.submit(run_tile, t): t for t in plan.tiles()}
            results = ((futures[f], f.result()) for f in as_completed(futures))
        else:
            results = ((t, run_tile(t)) for t in plan.tiles())
        for tile_id, (sums, count) in results:
            values.extend(sums)
            done, pairs = done + work[tile_id], pairs + count
            if report:
                report(done, total, pairs)
        return math.fsum(values)

    def prepare(tile_id):
        owned, near = plan.members(tile_id)
        return (tile_id, plan.rows[owned], row_w[owned], plan.cols[near], col_w[near],
                bounds_of(owned, near))

    parts = []
    with ThreadPoolExecutor(max(1, prefetch)) as pool:
        for tile_id, *arrays, bounds in _ahead(pool, prepare, plan.tiles(), max(1, prefetch)):
            if bounds:
                left, wl, right, wr = (ops.put(a) for a in arrays)
                sums = [ops.block_sum(wl[r0:r1],
                                      ops.kernel_rows(left[r0:r1], right[c0:c1], wr[c0:c1]))
                        for r0, r1, c0, c1 in bounds]
                parts.append(ops.stack(sums, 1))
            done += work[tile_id]
            pairs += sum((r1 - r0) * (c1 - c0) for r0, r1, c0, c1 in bounds)
            if report:
                report(done, total, pairs)
    return math.fsum(ops.concatenate(parts, 1)[:, 0])


class CompiledTorchOps(TorchOps):
    """TorchOps with distance, exp and weighted row sum fused by torch.compile (float64)."""

    def __init__(self, device):
        super().__init__(device)
        torch = self.torch

        def rows(left, right, wr):
            squared = ((left[:, 0:1] - right[:, 0]) ** 2 + (left[:, 1:2] - right[:, 1]) ** 2
                       + (left[:, 2:3] - right[:, 2]) ** 2)
            return (torch.exp(squared * -0.5) * wr[:, 0]).sum(dim=1, keepdim=True)

        self.problem = None
        try:
            self._fused = torch.compile(rows, dynamic=True, fullgraph=True)
            self.name = f"{self.name} (compiled)"
        except Exception as exc:  # no torch.compile in this PyTorch
            self.problem = f"{type(exc).__name__}: {exc}"
            self.name = f"{self.name} (eager; compile failed)"

    def kernel_rows(self, left, right, wr):
        if self.problem is None and wr.shape[1] == 1:
            try:
                return self._fused(left, right, wr)
            except Exception as exc:  # compiler or Triton missing / unsupported: fall back
                self.problem = f"{type(exc).__name__}: {exc}"
                self.name = self.name.replace(" (compiled)", " (eager; compile failed)")
                warnings.warn(f"torch.compile failed, using the element-wise kernel: "
                              f"{self.problem}", RuntimeWarning)
        return super().kernel_rows(left, right, wr)


def make_ops(device, compile=True):
    if device == "cpu":
        return None
    if device == "numpy":
        return NumpyOps()
    if device == "torch-cpu":
        return TorchOps("cpu")
    if device == "torch-cpu-compiled":
        return CompiledTorchOps("cpu")
    if device == "cuda" or device.startswith("cuda:"):
        return CompiledTorchOps(device) if compile else TorchOps(device)
    raise ValueError(f"Unknown device {device!r}; use cpu, cuda or cuda:N")


def _clamp(squared, magnitude, bound):
    """sqrt of D^2 = sum of terms of total size `magnitude`, rejecting real negatives."""
    if squared < 0:
        if squared < -(64 * np.finfo(np.float64).eps * magnitude + bound):
            raise ArithmeticError("Kernel distance squared is negative beyond roundoff")
        return 0.0
    return math.sqrt(squared)


class MMDReference:
    """v2_fast drop-in for kernel.MMDReference (fast algorithm only); same score() columns.

    Score prepared Events (signed polarity, shared clock in seconds, common interval).
    """

    def __init__(self, reference, *, sigma_spatial_px, sigma_temporal_ms,
                 weightings=WEIGHTINGS, cutoff_sigma=10.0, tile_px=None, block_size=None,
                 chunk_size=None, threads=1, device="cpu", compile=True):
        time_s = sigma_temporal_ms / 1000
        if not all(math.isfinite(v) and v > 0 for v in (sigma_spatial_px, time_s)):
            raise ValueError("Gaussian scales must be finite and positive")
        weightings = tuple(dict.fromkeys(weightings))
        if not weightings or any(w not in WEIGHTINGS for w in weightings):
            raise ValueError(f"weightings must be a non-empty subset of {WEIGHTINGS}")
        if not math.isfinite(cutoff_sigma) or cutoff_sigma < 3:
            raise ValueError("cutoff_sigma must be finite and at least 3")
        self._ops = make_ops(device, compile)
        kind = ("cpu" if self._ops is None else
                "compiled" if isinstance(self._ops, CompiledTorchOps) else "device")
        sizes = dict(SIZES_V2[kind])
        for name, value in (("tile_px", tile_px), ("block_size", block_size),
                            ("chunk_size", chunk_size)):
            if value is not None:
                sizes[name] = value
        for name, value in (*sizes.items(), ("threads", threads)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.weightings = weightings
        self.algorithm = "fast"
        self.cutoff_sigma = cutoff_sigma
        self._scales = np.array([time_s, sigma_spatial_px, sigma_spatial_px])
        self._options = dict(sizes, cutoff_sigma=cutoff_sigma)
        self._threads = threads
        self._polarities = self._channels(reference)
        self.n_gt = len(reference)
        # Optional: an object with stage(label, events) and __call__(done, total, pairs).
        self.progress = None

    @property
    def device(self):
        return self._ops.name if self._ops is not None else "cpu"

    @staticmethod
    def _channels(events):
        if not np.all((events.p == -1) | (events.p == 1)):
            raise ValueError("Expected signed event polarities (-1 or +1)")
        return _points(events, 1), _points(events, -1)

    def _stage(self, label, events):
        if self.progress:
            self.progress.stage(label, events)

    def _unique(self, points, label):
        self._stage(f"{label} merge", len(points))
        return unique_rows(points)

    def _sum(self, rows, row_weights, cols, col_weights, label):
        self._stage(label, len(rows) + (len(cols) if cols is not rows else 0))
        options = dict(self._options, report=self.progress)
        if self._ops is not None:
            return pair_sum(rows, row_weights, cols, col_weights, self._scales, ops=self._ops,
                            prefetch=min(self._threads, 4), **options)
        if self._threads == 1 or len(rows) < 4 * options["block_size"]:
            return pair_sum(rows, row_weights, cols, col_weights, self._scales, **options)
        with ThreadPoolExecutor(self._threads) as pool:
            return pair_sum(rows, row_weights, cols, col_weights, self._scales, executor=pool,
                            **options)

    def _together(self, points, weights, label):
        """kernel.py's way: GT and prediction rows together, signed weights per column."""
        self._stage(label, len(points))
        options = dict(self._options, report=self.progress)
        if self._ops is not None:
            options["chunk_size"] = min(options["chunk_size"], SIZES["device"]["chunk_size"])
            return device_energy(points, weights, self._scales, ops=self._ops,
                                 prefetch=min(self._threads, 4), **options)
        if self._threads == 1 or len(points) < 4 * options["block_size"]:
            return fast_energy(points, weights, self._scales, **options)
        with ThreadPoolExecutor(self._threads) as pool:
            return fast_energy(points, weights, self._scales, executor=pool, **options)

    def _bound(self, total_weight):
        return math.exp(-0.5 * self.cutoff_sigma ** 2) * total_weight ** 2

    @cached_property
    def _gt(self):
        """Per polarity: (unique GT points, multiplicities, S_gg)."""
        merged = []
        for name, points in zip(("ON", "OFF"), self._polarities):
            unique, counts = self._unique(points, f"GT {name}")
            merged.append((unique, counts,
                           self._sum(unique, counts, unique, counts, f"GT {name} with GT")))
        return merged

    @cached_property
    def empty_baseline(self):
        """Count distance from GT to an empty prediction, (joint, on, off, bound)."""
        on, off = (math.sqrt(max(0.0, s_gg)) for _, _, s_gg in self._gt)
        bound = math.sqrt(sum(self._bound(len(p)) for p in self._polarities))
        return math.hypot(on, off), on, off, bound

    def score(self, generated):
        """Return a flat dict of count_* and prob_* columns (None where undefined)."""
        predicted = self._channels(generated)
        n_gt, n_gen = self.n_gt, len(generated)
        result = {}
        want_count = "count" in self.weightings
        want_prob = "probability" in self.weightings and n_gt and n_gen
        if n_gen == 0:
            if want_count:
                joint, on, off, bound = self.empty_baseline
                result.update(count_distance=joint, count_distance_on=on,
                              count_distance_off=off, count_truncation_bound=bound)
        else:
            count, prob, count_bound, prob_bound = [], [], 0.0, 0.0
            self.paths = []
            for index, (name, (gt_points, gt_counts, s_gg), points) in enumerate(
                    zip(("ON", "OFF"), self._gt, predicted)):
                reference = self._polarities[index]
                n_pol, m_pol = len(reference), len(points)
                bound = self._bound(n_pol + m_pol)
                p_bound = self._bound(n_pol / n_gt + m_pol / n_gen) if want_prob else 0.0
                count_bound, prob_bound = count_bound + bound, prob_bound + p_bound
                self._stage(f"{name} merge", n_pol + m_pol)
                both = _channel(reference, points)  # GT and prediction rows, multiplicities
                mine = both.gen > 0
                unique, counts = both.points[mine], both.gen[mine].astype(np.float64)
                if np.array_equal(both.gt, both.gen):
                    self.paths.append("identical")  # same events: exactly zero
                    count.append(0.0)
                    if want_prob:  # same events, different totals N, M over polarities
                        prob.append(math.sqrt(max(0.0, s_gg)) * abs(1 / n_gt - 1 / n_gen))
                    continue
                rows_together, rows_gt, rows_gen = len(both.points), len(gt_points), len(unique)
                if rows_together ** 2 <= rows_gen ** 2 + rows_gt * rows_gen:
                    self.paths.append("together")
                    columns = ["count"] + (["probability"] if want_prob else [])
                    together, weights = _weights(both, columns, n_gt, n_gen)
                    energies = self._together(together, weights, f"{name} GT+prediction")
                    count.append(_distance(energies[0], weights[:, 0], bound))
                    if want_prob:
                        prob.append(_distance(energies[1], weights[:, 1], p_bound))
                    continue
                self.paths.append("sums")
                s_pp = self._sum(unique, counts, unique, counts, f"{name} with itself")
                s_gp = self._sum(gt_points, gt_counts, unique, counts, f"{name} with GT")
                count.append(_clamp(s_gg + s_pp - 2 * s_gp, s_gg + s_pp + 2 * abs(s_gp), bound))
                if want_prob:
                    a, b, c = s_gg / n_gt ** 2, s_pp / n_gen ** 2, s_gp / (n_gt * n_gen)
                    prob.append(_clamp(a + b - 2 * c, a + b + 2 * abs(c), p_bound))
            if want_count:
                result.update(count_distance=math.hypot(*count), count_distance_on=count[0],
                              count_distance_off=count[1],
                              count_truncation_bound=math.sqrt(count_bound))
            if want_prob:
                result.update(prob_distance=math.hypot(*prob), prob_distance_on=prob[0],
                              prob_distance_off=prob[1],
                              prob_truncation_bound=math.sqrt(prob_bound))
        if want_count:
            baseline = self.empty_baseline[0]
            result["count_empty_baseline"] = baseline
            result["count_relative_to_empty"] = (
                result["count_distance"] / baseline if baseline else None)
        if "probability" in self.weightings and "prob_distance" not in result:
            result.update(prob_distance=None, prob_distance_on=None, prob_distance_off=None,
                          prob_truncation_bound=None)
        return result
