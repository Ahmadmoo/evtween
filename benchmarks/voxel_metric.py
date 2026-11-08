"""Signed-count voxel errors from the ALLTestSet_Eval comparison protocol.

Pass aligned arrays of shape (intervals, height, width), using identical physical
frame edges and crop for prediction and GT. For V2V use the native 'voxels'
array. support optionally selects a common boolean (height, width) mask.
Inactive pixels within support are included. No per-stream normalization.
"""
import numpy as np


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
