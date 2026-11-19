"""Validate stream declarations, convert units and select one shared interval."""
from dataclasses import dataclass

import numpy as np


SECONDS_PER_UNIT = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


@dataclass(frozen=True)
class Events:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray

    def __len__(self):
        return self.t.size

    def select(self, start_s, stop_s):
        left, right = np.searchsorted(self.t, [start_s, stop_s])
        return Events(*(array[left:right] for array in (self.x, self.y, self.t, self.p)))

    def duplicates(self):
        """Count exact repeated tuples after their first occurrence."""
        repeated = self.t[1:] == self.t[:-1]
        if not np.any(repeated):
            return 0
        indices = np.flatnonzero(np.r_[repeated, False] | np.r_[False, repeated])
        x, y, t, p = (array[indices] for array in (self.x, self.y, self.t, self.p))
        order = np.lexsort((p, x, y, t))
        equal = np.ones(indices.size - 1, dtype=bool)
        for array in (t, x, y, p):
            sorted_array = array[order]
            equal &= sorted_array[1:] == sorted_array[:-1]
        return int(np.count_nonzero(equal))


def prepare_events(raw, width, height, *, unit, origin=0, polarity="signed"):
    """Convert declared raw units and origin into seconds on the common clock.

    Origin uses the raw timestamp unit and maps to zero on the evaluation clock.
    Callers supply provenance for resolution, clock and polarity declarations.
    """
    if unit not in SECONDS_PER_UNIT or not np.isfinite(origin):
        raise ValueError("A supported timestamp unit and finite clock origin are required")
    x, y, t, p = (np.asarray(raw[key]) for key in "xytp")
    if t.ndim != 1 or any(a.shape != t.shape for a in (x, y, p)):
        raise ValueError("Event arrays must have the same one-dimensional shape")
    for array, limit, axis in ((x, width, "x"), (y, height, "y")):
        if array.dtype.kind not in "iu" or np.any(array < 0) or np.any(array >= limit):
            raise ValueError(f"Invalid {axis} coordinates for sensor dimension {limit}")
    if t.dtype.kind not in "iuf" or not np.isfinite(t).all() or np.any(t[1:] < t[:-1]):
        raise ValueError("Timestamps must be numeric, finite and nondecreasing")
    if polarity == "signed":
        valid = (p == -1) | (p == 1)
    elif polarity == "binary":
        valid = (p == 0) | (p == 1)
    elif polarity == "benchmark":
        # The converted GT may use either convention. Positive-only is shared.
        has_zero = np.any(p == 0)
        valid = (p == (0 if has_zero else -1)) | (p == 1)
    else:
        raise ValueError("Polarity convention must be 'signed', 'binary' or 'benchmark'")
    if p.dtype.kind not in "biuf" or not valid.all():
        raise ValueError(f"Events violate the {polarity} polarity convention")
    # Use extended precision for integer clocks with large absolute origins.
    precision = np.longdouble if t.dtype.kind in "iu" and origin else np.float64
    times = t.astype(precision, copy=False)
    if origin:
        times = times - origin
    if unit != "s":
        times = times * SECONDS_PER_UNIT[unit]
    times = times.astype(np.float64, copy=False)
    if not np.isfinite(times).all():
        raise ValueError("Converted timestamps must be finite")
    return Events(x, y, times, np.where(p > 0, np.int8(1), np.int8(-1)))
