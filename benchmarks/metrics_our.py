"""Event-stream metrics proposed for the v3 benchmark."""
import numpy as np

if __package__:
    from .metrics import chamfer_full
else:
    from metrics import chamfer_full


def _frame_counts(stream, edges):
    """Count ON and OFF events in each half-open interval of a sorted stream."""
    t = np.asarray(stream["t"])
    p = np.asarray(stream["p"])
    if np.any(t[1:] < t[:-1]):
        raise ValueError("Frame counts require sorted event timestamps")
    cuts = np.searchsorted(t, edges, side="left")
    if cuts[0] != 0 or cuts[-1] != len(t):
        raise ValueError("Frame intervals must cover every event")
    on = np.fromiter((np.count_nonzero(p[a:b] > 0) for a, b in zip(cuts, cuts[1:])),
                     dtype=np.int64, count=len(edges) - 1)
    return np.column_stack((on, np.diff(cuts) - on))


def chamfer_count(events, reference, width, frame_edges_s, *, count_weight=1.0,
                  v_px_per_ms=1.0, device="cpu", workers=1, chunk_size=131072):
    """Add weighted count error per frame interval and polarity to full Chamfer.

    Use sorted streams and common RGB frame timestamps in seconds. Excess and
    deficit are normalized by the combined event count. Their sum lies in [0, 1].
    count_weight is in pixel-equivalent units. With no shared polarity, geometry
    and the combined score are undefined. Two empty streams have undefined ratios.
    """
    edges = np.asarray(frame_edges_s, dtype=np.float64)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)):
        raise ValueError("Expected at least two finite, increasing frame timestamps")
    if not np.isfinite(count_weight) or count_weight < 0:
        raise ValueError("count_weight must be finite and nonnegative")
    scores = chamfer_full(events, reference, width, edges[-1] - edges[0],
                          start_s=edges[0], v_px_per_ms=v_px_per_ms, device=device,
                          workers=workers, chunk_size=chunk_size)
    delta = _frame_counts(events, edges) - _frame_counts(reference, edges)
    total = len(events["t"]) + len(reference["t"])
    excess = float(np.maximum(delta, 0).sum()) / total if total else np.nan
    deficit = float(np.maximum(-delta, 0).sum()) / total if total else np.nan
    penalty = excess + deficit
    geometry = scores["value"]
    scores.update(chamfer_geometry=geometry, count_excess=excess, count_deficit=deficit,
                  count_penalty=penalty, value=geometry + count_weight * penalty)
    return scores
