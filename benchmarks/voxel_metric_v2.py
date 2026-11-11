"""Full-image, fixed-duration voxel errors. See VOXEL_METRIC_V2.md.

PMSE variants are explicitly adapted, not exact reproductions of V2CE tables.
"""
import numpy as np

PROTOCOL = "fixed-duration-full-coverage-v2"


def scoring_window(source_edges, absolute_t0, start=0., stop=None):
    """Edges must be relative to the first source frame and start exactly at zero.

    start/stop use that same origin, never offsets from a pretrimmed subwindow.
    """
    source_edges=np.asarray(source_edges,dtype=np.float64)
    if (source_edges.ndim != 1 or len(source_edges)<2 or not np.isfinite(source_edges).all()
            or np.any(np.diff(source_edges)<=0) or source_edges[0] != 0.):
        raise ValueError("source_edges must increase from zero relative to the first source frame")
    if not np.isfinite(absolute_t0):
        raise ValueError("absolute_t0 must be finite")
    lower,upper=float(source_edges[0]),float(source_edges[-1])
    stop=upper if stop is None else float(stop)
    tolerance=8*np.spacing(max(1.,abs(float(absolute_t0)),abs(upper)))
    if not np.isfinite([start,stop]).all() or start>=stop:
        raise ValueError("Require finite start < stop")
    if start < lower-tolerance or stop > upper+tolerance:
        raise ValueError("Requested interval is outside the common source interval")
    start=max(float(start),lower);stop=min(stop,upper)
    if start>=stop:
        raise ValueError("Empty scoring interval")
    return start,stop


def time_edges(start, stop, bin_ms):
    """Nominal-width bins in seconds; retain a shorter final bin."""
    if not np.isfinite([start, stop, bin_ms]).all() or stop <= start or bin_ms <= 0:
        raise ValueError("Require finite start < stop and positive bin_ms")
    step = bin_ms / 1000.
    if start + step <= start:
        raise ValueError("Bin width is below timestamp precision")
    n = int(np.ceil((stop-start)/step))
    edges = start + np.arange(n, dtype=np.float64)*step
    edges = edges[edges < stop]
    # Remove a floating-point sliver at an intended exact multiple.
    if len(edges) > 1 and abs(edges[-1]-stop) <= 8*np.spacing(max(abs(start), abs(stop), step)):
        edges = edges[:-1]
    return np.append(edges, stop)


def event_voxels(events, edges, height, width, *, interpolation="histogram", bin_ms=None):
    """Separate ON/OFF counts; no spatial resizing or per-stream normalization.

    Forward weights use nominal bin_ms. Any right-boundary overflow is folded
    into the last bin, so each included event contributes exactly one unit.
    """
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Require finite increasing time edges")
    if any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 1
           for d in (height, width)):
        raise ValueError("Require positive integer image dimensions")
    if interpolation not in ("histogram", "forward"):
        raise ValueError("Unknown interpolation")
    t,x,y,p = (np.asarray(events[k]) for k in "txyp")
    if any(a.ndim != 1 or a.shape != t.shape or a.dtype.kind not in "iuf" or not np.isfinite(a).all()
           for a in (t,x,y,p)):
        raise ValueError("Require matching finite real txyp vectors")
    if (np.any(x != np.floor(x)) or np.any(y != np.floor(y)) or
        np.any((x < 0) | (x >= width)) or np.any((y < 0) | (y >= height)) or
        not np.isin(p, (-1,0,1)).all()):
        raise ValueError("Invalid coordinates or polarity")
    keep = (t >= edges[0]) & (t < edges[-1])
    t,x,y,p = (a[keep] for a in (t,x,y,p))
    bins = len(edges)-1
    b = np.searchsorted(edges, t, side="right")-1
    pixel = y.astype(np.int64)*width+x.astype(np.int64)
    pol = (p <= 0).astype(np.int64)
    ids = (pol*bins+b)*height*width+pixel
    size = 2*bins*height*width
    if interpolation == "histogram":
        return np.bincount(ids, minlength=size).reshape(2,bins,height,width).astype(np.float64)
    if bin_ms is None or not np.isfinite(bin_ms) or bin_ms <= 0:
        raise ValueError("Forward interpolation requires positive bin_ms")
    step = bin_ms/1000.
    if np.any(np.diff(edges) > step*(1+1e-10)) or not np.allclose(np.diff(edges)[:-1], step, rtol=1e-10, atol=1e-14):
        raise ValueError("Forward interpolation requires nominal-width bins and an optional short final bin")
    current = np.clip((edges[b+1]-t)/step, 0, 1)
    current[b == bins-1] = 1.  # Explicit mass-preserving boundary convention.
    grid = np.bincount(ids, weights=current, minlength=size)
    valid = b+1 < bins
    grid += np.bincount(ids[valid]+height*width, weights=1-current[valid], minlength=size)
    return grid.reshape(2,bins,height,width)


def _pool_full(a, kernel):
    """Partial-block means and original-cell counts; nothing is discarded."""
    sizes = []
    for axis in (1,2,3):
        n=a.shape[axis]; starts=np.arange(0,n,kernel)
        lengths=np.diff(np.append(starts,n)); sizes.append(lengths)
        shape=[1]*4; shape[axis]=len(lengths)
        a=np.add.reduceat(a,starts,axis=axis)/lengths.reshape(shape)
    weights=sizes[0][:,None,None]*sizes[1][None,:,None]*sizes[2][None,None,:]
    return a,weights


def _pmse(prediction, reference, kernel, *, edges, height, width, bin_ms):
    pred=event_voxels(prediction,edges,height,width,interpolation="forward",bin_ms=bin_ms)
    pp,weights=_pool_full(pred,kernel)
    del pred
    gt=event_voxels(reference,edges,height,width,interpolation="forward",bin_ms=bin_ms)
    gg,_=_pool_full(gt,kernel)
    return float(np.sum((pp-gg)**2*weights)/(2*weights.sum()))


def pmse_2_v2ce_adapted(prediction, reference, **domain):
    """V2CE-inspired 3D pooling MSE; full-coverage, cell-weighted edge adaptation."""
    return _pmse(prediction,reference,2,**domain)


def pmse_4_v2ce_adapted(prediction, reference, **domain):
    return _pmse(prediction,reference,4,**domain)


def score(metric, prediction, reference, *, edges, height, width, bin_ms):
    """Return report fields. Counts have equal cell weight, including a short final bin."""
    edges=np.asarray(edges,dtype=np.float64)
    if edges.ndim != 1 or len(edges)<2 or not np.isfinite(edges).all() or np.any(np.diff(edges)<=0):
        raise ValueError("Require finite increasing time edges")
    if metric in ("pmse_2_v2ce_adapted", "pmse_4_v2ce_adapted"):
        return {"value": globals()[metric](prediction,reference,edges=edges,height=height,width=width,bin_ms=bin_ms)}
    if metric not in ("voxel_mae", "voxel_rmse", "voxel_relative_l1"):
        raise ValueError("Unknown metric")
    from .voxel_metric import voxel_mae_by_polarity, voxel_rmse_by_polarity
    # Bin streaming avoids building full-resolution grids for legacy count errors.
    def planes(events):
        # Validate the whole stream once; avoid an O(events*bins) scan.
        event_voxels(events,[edges[0],edges[-1]],height,width)
        t,x,y,p=(np.asarray(events[k]) for k in 'txyp')
        if np.any(t[1:] < t[:-1]):
            order=np.argsort(t,kind='stable'); t,x,y,p=(a[order] for a in (t,x,y,p))
        cuts=np.searchsorted(t,edges,side='left')
        for a,b in zip(cuts[:-1],cuts[1:]):
            ids=(p[a:b]<=0).astype(np.int64)*height*width
            ids+=y[a:b].astype(np.int64)*width+x[a:b].astype(np.int64)
            yield np.bincount(ids,minlength=2*height*width).reshape(2,height,width)
    if metric == "voxel_relative_l1":
        absolute=mass=0.
        for p,g in zip(planes(prediction),planes(reference)):
            ps,gs=p[0]-p[1],g[0]-g[1]
            absolute+=np.abs(ps-gs).sum(); mass+=np.abs(gs).sum()
        return {"value":float(absolute/mass) if mass else np.nan}
    compare=voxel_mae_by_polarity if metric == "voxel_mae" else voxel_rmse_by_polarity
    return compare(planes(prediction),planes(reference))
