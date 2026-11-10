"""Event-stream PMSE and backward-compatible voxel-array error functions.

Pass aligned arrays of shape (intervals, height, width), using identical physical
time edges and crop for prediction and GT. These legacy array functions do not
choose time bins. The runner defaults to uniform full-interval bins; native V2V
is available only with explicit legacy frame-pair scoring. support optionally
selects a common boolean (height, width) mask.
Inactive pixels within support are included. No per-stream normalization.
"""
import numpy as np


def _event_voxels(events, *, start, stop, temporal_bins, height, width):
    """Benchmark representation, not a metric: forward linear temporal weights.

    Shared [start, stop) domain; ON/OFF are separate nonnegative planes.
    A unit-duration box starting at an event contributes 1-f to its bin and
    f to the following bin (V2CE II-B). Contributions beyond stop are clipped.
    No event-count, per-stream or nonzero-voxel normalization is applied.
    """
    if (not np.isfinite([start, stop]).all() or stop <= start or
            any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                or v < 1 for v in (temporal_bins, height, width))):
        raise ValueError("Expected finite increasing bounds and positive integer dimensions")
    t, x, y, p = (np.asarray(events[k]) for k in "txyp")
    if any(a.ndim != 1 or a.shape != t.shape or a.dtype.kind not in "iuf"
           or not np.isfinite(a).all() for a in (t, x, y, p)):
        raise ValueError("Events must have equal finite real 1D txyp arrays")
    if (np.any(x != np.floor(x)) or np.any(y != np.floor(y)) or
            np.any((x < 0) | (x >= width) | (y < 0) | (y >= height)) or
            not np.isin(p, (-1, 0, 1)).all()):
        raise ValueError("Invalid event coordinates or polarity")
    keep = (t >= start) & (t < stop)
    u = (t[keep] - start) / (stop - start) * temporal_bins
    # Protect floor against rounding of a timestamp immediately below stop.
    u = np.minimum(u, np.nextafter(float(temporal_bins), -np.inf))
    b = np.floor(u).astype(np.int64)
    f = u - b
    pol = (p[keep] <= 0).astype(np.int64)
    xx, yy = x[keep].astype(np.int64), y[keep].astype(np.int64)
    grid = np.zeros((2, temporal_bins, height, width), dtype=np.float64)
    np.add.at(grid, (pol, b, yy, xx), 1-f)
    valid = b + 1 < temporal_bins
    np.add.at(grid, (pol[valid], b[valid]+1, yy[valid], xx[valid]), f[valid])
    return grid


def _pmse_v2ce(prediction, reference, kernel, **domain):
    """V2CE III-A PMSE formula; see VOXEL_METRIC_PROVENANCE.md for adaptations."""
    pred = _event_voxels(prediction, **domain)
    gt = _event_voxels(reference, **domain)
    _, t, h, w = pred.shape
    if min(t, h, w) < kernel:
        raise ValueError("Each pooled dimension must be at least the PMSE kernel")
    def pool(a):
        # Valid, unpadded, floor pooling; polarity is a channel, never pooled.
        nt, nh, nw = t//kernel, h//kernel, w//kernel
        return a[:, :nt*kernel, :nh*kernel, :nw*kernel].reshape(
            2, nt, kernel, nh, kernel, nw, kernel).mean(axis=(2, 4, 6))
    return float(np.mean(np.square(pool(pred) - pool(gt))))


def pmse_2_v2ce(prediction, reference, *, start, stop, temporal_bins, height, width):
    """Pooling MSE (k=stride=2), from two txyp event streams on one domain.

    Formula follows V2CE III-A; full-interval protocol is a benchmark adaptation.
    https://arxiv.org/html/2309.08891v2#S3.SS1
    """
    return _pmse_v2ce(prediction, reference, 2, start=start, stop=stop,
                     temporal_bins=temporal_bins, height=height, width=width)


def pmse_4_v2ce(prediction, reference, *, start, stop, temporal_bins, height, width):
    """Pooling MSE (k=stride=4); same domain and conventions as pmse_2_v2ce."""
    return _pmse_v2ce(prediction, reference, 4, start=start, stop=stop,
                     temporal_bins=temporal_bins, height=height, width=width)


def _errors(prediction, reference, support):
    pred, gt = np.asarray(prediction), np.asarray(reference)
    if pred.ndim != 3 or pred.shape != gt.shape:
        raise ValueError("Expected equal (intervals, height, width) shapes")
    if pred.dtype.kind not in "iuf" or gt.dtype.kind not in "iuf":
        raise ValueError("Voxel arrays must contain real signed counts")
    if support is not None:
        support = np.asarray(support)
        if support.dtype != np.bool_ or support.shape != gt.shape[1:]:
            raise ValueError("support must be a boolean (height, width) mask")
    absolute = squared = mass = 0.0
    count = 0
    for p, g in zip(pred, gt):
        if support is not None:
            p, g = p[support], g[support]
        g = g.astype(np.float64, copy=False)
        diff = np.subtract(p, g, dtype=np.float64)
        if not np.isfinite(diff).all():
            raise ValueError("Voxels must be finite on the scoring support")
        absolute += np.abs(diff).sum()
        squared += np.square(diff).sum()
        mass += np.abs(g).sum()
        count += diff.size
    return absolute, squared, mass, count


def voxel_mae(prediction, reference, support=None):
    """Mean absolute signed-count error per voxel; empty support gives NaN."""
    absolute, _, _, count = _errors(prediction, reference, support)
    return float(absolute / count) if count else np.nan


def voxel_rmse(prediction, reference, support=None):
    """Root mean squared signed-count error per voxel; empty support gives NaN."""
    _, squared, _, count = _errors(prediction, reference, support)
    return float(np.sqrt(squared / count)) if count else np.nan


def voxel_relative_l1(prediction, reference, support=None):
    """Sum of absolute errors / sum of absolute GT counts; zero GT mass gives NaN."""
    absolute, _, mass, _ = _errors(prediction, reference, support)
    return float(absolute / mass) if mass else np.nan


def _polarity_scores(prediction, reference, squared):
    # Each input yields one (2, H, W) array per frame interval, ON then OFF.
    from itertools import zip_longest

    totals = np.zeros(3, dtype=np.float64)
    count = 0
    spatial_shape = None
    for pred, gt in zip_longest(prediction, reference):
        if pred is None or gt is None:
            raise ValueError("Prediction and reference must have the same number of intervals")
        pred, gt = np.asarray(pred), np.asarray(gt)
        if pred.ndim != 3 or pred.shape[0] != 2 or pred.shape != gt.shape:
            raise ValueError("Expected matching (2, height, width) ON/OFF planes")
        if spatial_shape is not None and pred.shape[1:] != spatial_shape:
            raise ValueError("Spatial dimensions must be constant across intervals")
        spatial_shape = pred.shape[1:]
        if any(a.dtype.kind not in "iuf" or not np.isfinite(a).all() or np.any(a < 0)
               for a in (pred, gt)):
            raise ValueError("ON/OFF counts must be finite and nonnegative")
        on, off = np.subtract(pred, gt, dtype=np.float64)
        for i, error in enumerate((on - off, on, off)):
            totals[i] += np.square(error).sum() if squared else np.abs(error).sum()
        count += on.size
    means = totals / count if count else np.full(3, np.nan)
    values = np.sqrt(means) if squared else means
    return dict(zip(("value", "value_on", "value_off"), map(float, values)))


def voxel_mae_by_polarity(prediction, reference):
    """Signed, ON and OFF MAE over aligned (interval, 2, H, W) counts, ON first."""
    return _polarity_scores(prediction, reference, squared=False)


def voxel_rmse_by_polarity(prediction, reference):
    """Signed, ON and OFF RMSE over aligned (interval, 2, H, W) counts, ON first."""
    return _polarity_scores(prediction, reference, squared=True)
