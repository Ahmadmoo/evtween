"""Shared helpers for the *_fast scorers. No metric is redefined here.

chamfer_full below is benchmarks/metrics.chamfer_full line for line, except that the
polarity point sets and KD-trees of the reference (GT) stream are kept for the next call
with the same reference object (e.g. the next method on the same window). A KD-tree built
from the same points answers the same nearest-neighbour queries, and the chunking and
summation order are unchanged, so every value is identical to the original.
Its `workers` argument only sets query threads (it does not change results); the fast
scorers replace it with their own per-process thread count.
"""
import os

import numpy as np

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
QUERY_THREADS = None  # set by the fast scorers in each worker process
_REFERENCE = {"object": None, "parts": {}}


def allocated_cpus():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def limit_threads(threads):
    """Call before numpy/scipy start their thread pools (i.e. at worker start)."""
    global QUERY_THREADS
    QUERY_THREADS = max(1, int(threads))
    for key in THREAD_VARS:
        os.environ[key] = str(QUERY_THREADS)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(QUERY_THREADS)
    except ImportError:
        pass


def forget_reference():
    _REFERENCE["object"], _REFERENCE["parts"] = None, {}


def chamfer_full(events, reference, width, duration_s, *, start_s=0.0,
                 v_px_per_ms=1.0, device="cpu", workers=1, chunk_size=131072):
    """benchmarks.metrics.chamfer_full with the reference's trees reused (identical values)."""
    from metrics import _positive, _times, _coordinates, _polarity  # benchmarks/metrics.py
    _positive(duration_s, "duration_s")
    _positive(v_px_per_ms, "v_px_per_ms")
    if not np.isfinite(start_s) or chunk_size < 1 or workers < 1:
        raise ValueError("Expected finite start_s and positive chunk_size and workers")
    if device == "cpu":
        from scipy.spatial import cKDTree as Tree
        xp = np
        query_options = {"workers": QUERY_THREADS or workers}
    elif device == "cuda":
        import cupy as xp
        from cupyx.scipy.spatial import KDTree as Tree
        query_options = {}
    else:
        raise ValueError("device must be cpu or cuda")

    def columns(stream):
        t = _times(stream)
        if np.any(t < start_s) or np.any(t >= start_s + duration_s):
            raise ValueError("Events outside the common half-open scoring interval")
        x, y = _coordinates(stream, t, width)
        return x, y, t, _polarity(stream, t)

    def points(data, sign):
        x, y, t, p = data
        keep = p == sign
        xyz = np.empty((np.count_nonzero(keep), 3), dtype=np.float64)
        xyz[:, 0] = x[keep]
        xyz[:, 1] = y[keep]
        xyz[:, 2] = (t[keep] - start_s) * (1000.0 * v_px_per_ms)
        return xyz

    def directed(query, target, tree=None):
        tree = Tree(xp.asarray(target)) if tree is None else tree
        total = xp.zeros((), dtype=xp.float64)
        for offset in range(0, len(query), chunk_size):
            block = xp.asarray(query[offset:offset + chunk_size])
            distances, _ = tree.query(block, k=1, p=2, eps=0.0, **query_options)
            total += distances.sum(dtype=xp.float64)
        return float(total.item()) / len(query)

    pred = columns(events)
    # Reference parts depend on these settings; anything else starts a new cache.
    key = (int(width), float(start_s), float(duration_s), float(v_px_per_ms), device)
    if _REFERENCE["object"] is not reference or _REFERENCE["parts"].get("key") != key:
        forget_reference()
        _REFERENCE["parts"] = {"key": key, "columns": columns(reference)}
        _REFERENCE["object"] = reference
    cached = _REFERENCE["parts"]

    def reference_points(sign):
        if ("points", sign) not in cached:
            cached["points", sign] = points(cached["columns"], sign)
        return cached["points", sign]

    def reference_tree(sign):
        if ("tree", sign) not in cached:
            cached["tree", sign] = Tree(xp.asarray(reference_points(sign)))
        return cached["tree", sign]

    parts = {}
    for sign, name in ((1, "chamfer_on"), (-1, "chamfer_off")):
        a, b = points(pred, sign), reference_points(sign)
        parts[name] = (0.5 * (directed(a, b, reference_tree(sign)) + directed(b, a))
                       if len(a) and len(b) else np.nan)
        del a, b
    shared = [v for v in parts.values() if np.isfinite(v)]
    return dict(value=float(np.mean(shared)) if shared else np.nan, **parts)


def install(bench):
    """Route every caller of metrics.chamfer_full (also chamfer_penalty/chamfer_count) here."""
    bench.m.chamfer_full = chamfer_full
    bench.metrics_our.chamfer_full = chamfer_full


def validated_gt(source):
    """The checks Window.gt / Sequence.gt run on the raw GT arrays, once per sequence.

    Returns the memory-mapped arrays, or None if a check fails (the original code then
    raises the same error per window, which is left to happen unchanged).
    """
    try:
        arrays = {k: np.load(source / f"ev_{k}.npy", mmap_mode="r") for k in "txyp"}
        t = arrays["t"]
        if t.ndim != 1 or any(v.shape != t.shape for v in arrays.values()):
            return None
        if not np.all(np.isfinite(t)) or np.any(t[1:] < t[:-1]):
            return None
        return arrays
    except (OSError, ValueError):
        return None
