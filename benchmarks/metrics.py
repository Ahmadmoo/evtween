"""Metrics for aligned v3 event streams with arrays t (seconds), x, y and p.

Pass an events.npz mapping opened with numpy.load or a dict of arrays. Inputs
must already share the evaluation crop and time origin. Clip them to the same
half-open scoring interval before calling. Chamfer and ISI follow ALLTestSet_Eval
(Ahmadmoo/evtween, 835f6bff). Same-ts and Fano follow evt_metric_eval.
"""
import numpy as np


def _array(events, key):
    a = np.asarray(events[key])
    if a.ndim != 1:
        raise ValueError(f"{key} must be a one-dimensional array")
    return a


def _times(events):
    t = _array(events, "t").astype(np.float64, copy=False)
    if not np.isfinite(t).all():
        raise ValueError("Event timestamps must be finite seconds")
    return t


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _coordinates(events, t, width, height=None):
    x, y = (_array(events, k) for k in ("x", "y"))
    for a, limit in ((x, width), (y, height)):
        if limit is not None and (not isinstance(limit, (int, np.integer)) or limit <= 0):
            raise ValueError("Sensor dimensions must be positive integers")
        if a.shape != t.shape or a.dtype.kind not in "iu" or np.any(a < 0):
            raise ValueError("Coordinates must be nonnegative integer arrays matching t")
        if limit is not None and np.any(a >= limit):
            raise ValueError("Coordinates outside sensor dimensions")
    return x.astype(np.int64, copy=False), y.astype(np.int64, copy=False)


def _polarity(events, t):
    p = _array(events, "p")
    if p.shape != t.shape or not np.isin(p, (-1, 0, 1)).all():
        raise ValueError("Expected one polarity (-1/1 or 0/1) per timestamp")
    return np.where(p > 0, 1, -1)


def _layout(events, width, duration_s, start_s):
    _positive(duration_s, "duration_s")
    t = _times(events)
    if not np.isfinite(start_s) or np.any(t < start_s) or np.any(t >= start_s + duration_s):
        raise ValueError("Events must be inside the common half-open scoring interval")
    x, y = _coordinates(events, t, width)
    pixel = y * width + x
    tau = ((t - start_s) / duration_s).astype(np.float32)
    order = np.lexsort((tau, pixel))
    return pixel[order], tau[order], _polarity(events, t)[order]


def event_count(events):
    """Count events in the supplied scoring interval."""
    return len(_array(events, "t"))


def event_count_ratio(events, reference):
    """Return predicted / reference count, using the notebook's denominator floor of 1."""
    return event_count(events) / max(event_count(reference), 1)


def on_percentage(events):
    """Return the ON percentage (0 for an empty stream)."""
    t = _array(events, "t")
    return 100.0 * np.count_nonzero(_polarity(events, t) > 0) / max(len(t), 1)


def chamfer(events, reference, width, duration_s, *, start_s=0.0, n=20000,
            v_px_per_ms=1.0, chunk_size=4096, device=None):
    """Symmetric Euclidean Chamfer in pixels, equally weighted by shared polarity.

    Uses the reference pixel/time ordering and seeded CPU Torch permutations.
    A polarity missing from either stream is skipped; no shared polarity gives NaN.
    duration_s is the common scoring duration, not the last event timestamp.
    """
    import torch

    if min(n, chunk_size) < 1:
        raise ValueError("n and chunk_size must be positive")
    _positive(v_px_per_ms, "v_px_per_ms")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    layouts = [_layout(e, width, duration_s, start_s) for e in (events, reference)]
    rng = torch.Generator(device="cpu").manual_seed(0)

    def sample(layout, sign):
        pixel, tau, polarity = layout
        candidates = np.flatnonzero(polarity == sign)
        indices = candidates[torch.randperm(len(candidates), generator=rng)[:n].numpy()]
        pixel = pixel[indices]
        time = tau[indices] * (duration_s * 1000) * v_px_per_ms
        xyz = np.column_stack((pixel % width, pixel // width, time)).astype(np.float32)
        return torch.as_tensor(xyz, device=device)

    def directed(a, b):
        nearest = [torch.cdist(block, b).amin(dim=1) for block in a.split(chunk_size)]
        return torch.cat(nearest).mean()

    scores = []
    for sign in (1, -1):
        if all(np.any(layout[2] == sign) for layout in layouts):
            a, b = (sample(layout, sign) for layout in layouts)
            scores.append(((directed(a, b) + directed(b, a)) / 2).item())
    return float(np.mean(scores)) if scores else np.nan



def chamfer_full(events, reference, width, duration_s, *, start_s=0.0,
                 v_px_per_ms=1.0, device="cpu", workers=1, chunk_size=131072):
    """Exact bidirectional 1-NN Chamfer over every event of each shared polarity.

    Return value, chamfer_on and chamfer_off in pixel-equivalent units. The
    aggregate weights available polarities equally; missing components are NaN.
    Each batch queries the entire opposite polarity-matched target in float64.
    CPU uses SciPy cKDTree and workers; CUDA uses CuPy's spatial KDTree.
    """
    _positive(duration_s, "duration_s")
    _positive(v_px_per_ms, "v_px_per_ms")
    if not np.isfinite(start_s) or chunk_size < 1 or workers < 1:
        raise ValueError("Expected finite start_s and positive chunk_size and workers")
    if device == "cpu":
        from scipy.spatial import cKDTree as Tree
        xp = np
        query_options = {"workers": workers}
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

    def directed(query, target):
        tree = Tree(xp.asarray(target))
        total = xp.zeros((), dtype=xp.float64)
        for offset in range(0, len(query), chunk_size):
            block = xp.asarray(query[offset:offset + chunk_size])
            distances, _ = tree.query(block, k=1, p=2, eps=0.0, **query_options)
            total += distances.sum(dtype=xp.float64)
        return float(total.item()) / len(query)

    pred, gt = columns(events), columns(reference)
    parts = {}
    for sign, key in ((1, "chamfer_on"), (-1, "chamfer_off")):
        a, b = points(pred, sign), points(gt, sign)
        parts[key] = (0.5 * (directed(a, b) + directed(b, a))
                      if len(a) and len(b) else np.nan)
        del a, b
    shared = [v for v in parts.values() if np.isfinite(v)]
    return dict(value=float(np.mean(shared)) if shared else np.nan, **parts)


def chamfer_penalty(events, reference, width, duration_s, *, height, start_s=0.0,
                    v_px_per_ms=1.0, device="cpu", workers=1, chunk_size=131072):
    """Full-stream Chamfer with the scoring-domain diameter for one-sided absence.

    Keep raw ON/OFF distances undefined when a nearest neighbour does not exist.
    Average over polarities present in at least one stream. Two empty streams
    have no score. The penalty is in the same pixel-equivalent units as Chamfer.
    """
    if not isinstance(height, (int, np.integer)) or height < 1:
        raise ValueError("Height must be a positive integer")
    for stream in (events, reference):
        if np.any(_array(stream, "y") >= height):
            raise ValueError("Coordinates outside sensor height")

    scores = chamfer_full(events, reference, width, duration_s, start_s=start_s,
                          v_px_per_ms=v_px_per_ms, device=device, workers=workers,
                          chunk_size=chunk_size)
    diameter = float(np.hypot(np.hypot(width - 1, height - 1),
                              1000.0 * duration_s * v_px_per_ms))
    terms = []
    for on, key in ((True, "chamfer_on"), (False, "chamfer_off")):
        if np.isfinite(scores[key]):
            terms.append(scores[key])
        else:
            for stream in (events, reference):
                p = _array(stream, "p")
                if np.any(p > 0 if on else p <= 0):
                    terms.append(diameter)
                    break
    scores.update(value=float(np.mean(terms)) if terms else np.nan,
                  missing_polarity_cost=diameter)
    return scores


def isi_distribution(events, width, duration_s, *, start_s=0.0):
    """Return all same-pixel inter-event intervals in ms, including zero ties."""
    pixel, tau, _ = _layout(events, width, duration_s, start_s)
    return np.diff(tau)[pixel[1:] == pixel[:-1]] * (duration_s * 1000)


def isi_median_ms(events, width, duration_s, *, start_s=0.0):
    """Median per-pixel interval in ms, pooled across pixels; empty gives 0."""
    delta = isi_distribution(events, width, duration_s, start_s=start_s)
    return float(np.median(delta)) if delta.size else 0.0


def isi_cv(events, width, duration_s, *, start_s=0.0):
    """Population standard deviation / mean of the pooled ISIs; empty gives 0."""
    delta = isi_distribution(events, width, duration_s, start_s=start_s)
    return float(delta.std() / max(delta.mean(), 1e-9)) if delta.size else 0.0


def event_rate(events, edges_s):
    """Return events/s for each half-open bin given its real-time edges in seconds."""
    edges = np.asarray(edges_s, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("edges_s must contain at least two increasing finite timestamps")
    t = _times(events)
    counts, _ = np.histogram(t[t < edges[-1]], bins=edges)
    return counts / np.diff(edges)


def same_ts(events, *, start_s=0.0, window_s=0.05, resolution_s=1e-9):
    """Fraction of events sharing a quantized timestamp in the same window, across pixels."""
    _positive(window_s, "window_s")
    _positive(resolution_s, "resolution_s")
    t = _times(events) - start_s
    if not np.isfinite(start_s) or np.any(t < 0):
        raise ValueError("start_s must be finite and precede or equal all timestamps")
    if not t.size:
        return np.nan
    windows = np.floor(t / window_s)
    ticks = np.rint(t / resolution_s)
    if max(ticks.max(), windows.max()) >= 2**63:
        raise ValueError("Timestamp index exceeds int64")
    keys = np.column_stack((windows.astype(np.int64), ticks.astype(np.int64)))
    _, counts = np.unique(keys, axis=0, return_counts=True)
    return float(counts[counts > 1].sum() / len(t))


def fano_factor(events, width, height, duration_s, *, reference=None, start_s=0.0,
                window_s=0.05, bin_s=0.002, bin_px=8, eps=1e-3, return_maps=False):
    """Return window-by-spatial-tile population variance/mean of temporal counts.

    Inactive cells have factor zero. With reference supplied, return the benchmark
    mean absolute log ratio over jointly active cells (NaN for no overlap).
    With reference and return_maps=True, also return both maps and active masks.
    The last partial window is zero-padded as in the reference implementation.
    """
    for value, name in ((duration_s, "duration_s"), (window_s, "window_s"),
                        (bin_s, "bin_s"), (eps, "eps")):
        _positive(value, name)
    if not isinstance(bin_px, (int, np.integer)) or bin_px <= 0:
        raise ValueError("bin_px must be a positive integer")
    ns = round(window_s / bin_s)
    if ns < 1 or not np.isclose(ns, window_s / bin_s, rtol=0, atol=1e-9):
        raise ValueError("window_s must contain a whole number of time bins")
    nw = max(1, int(np.ceil(duration_s / window_s - 1e-12)))

    def values(stream):
        t = _times(stream) - start_s
        if not np.isfinite(start_s) or np.any(t < 0) or np.any(t >= duration_s):
            raise ValueError("Events must lie in the specified half-open interval")
        x, y = _coordinates(stream, t, width, height)
        nx, ny = (width + bin_px - 1) // bin_px, (height + bin_px - 1) // bin_px
        window = np.minimum(np.floor(t / window_s).astype(np.int64), nw - 1)
        sub = np.clip(np.floor((t - window * window_s) / bin_s).astype(np.int64), 0, ns - 1)
        cell = (window * ny + y // bin_px) * nx + x // bin_px
        # Count occupied sub-bins only. Zero-count sub-bins still contribute to ns.
        ids, counts = np.unique(cell * ns + sub, return_counts=True)
        tile = ids // ns
        total = np.bincount(tile, weights=counts, minlength=nw * ny * nx)
        squares = np.bincount(tile, weights=counts.astype(float)**2, minlength=total.size)
        factor = np.divide(squares, total, out=np.zeros(total.shape), where=total > 0) - total / ns
        return factor.reshape(nw, ny, nx), (total > 0).reshape(nw, ny, nx)

    pred, active = values(events)
    if reference is None:
        return pred
    gt, gt_active = values(reference)
    shared = active & gt_active
    score = (float(np.abs(np.log((pred[shared] + eps) / (gt[shared] + eps))).mean())
             if shared.any() else np.nan)
    if return_maps:
        return score, dict(pred_fano=pred, gt_fano=gt, pred_active=active, gt_active=gt_active)
    return score
