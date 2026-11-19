"""Shared histogram geometry and sparse occupancy counts."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    start_s: float
    stop_s: float
    spatial_bin_size: int
    temporal_bin_s: float

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in (self.width, self.height, self.spatial_bin_size)):
            raise ValueError("Image dimensions and spatial_bin_size must be positive integers")
        if not all(map(math.isfinite, (self.start_s, self.stop_s, self.temporal_bin_s))):
            raise ValueError("Time grid parameters must be finite")
        if self.stop_s <= self.start_s or self.temporal_bin_s <= 0:
            raise ValueError("Time grid requires a positive duration and temporal bin size")
        if (self.start_s + self.temporal_bin_s <= self.start_s
                or self.stop_s - self.temporal_bin_s >= self.stop_s):
            raise ValueError("Temporal bin size is below floating-point resolution on this clock")
        ratio = (self.stop_s - self.start_s) / self.temporal_bin_s
        if not math.isfinite(ratio) or ratio >= 2**52:
            raise ValueError("Too many temporal bins to index at float64 precision")

    @property
    def nx(self):
        return (self.width + self.spatial_bin_size - 1) // self.spatial_bin_size

    @property
    def ny(self):
        return (self.height + self.spatial_bin_size - 1) // self.spatial_bin_size

    @property
    def nt(self):
        ratio = (self.stop_s - self.start_s) / self.temporal_bin_s
        # A one-ULP overshoot of an integer is floating-point roundoff at the stop.
        return max(1, math.ceil(math.nextafter(ratio, -math.inf)))

    def shape(self, name):
        return {
            "spatial": (self.ny, self.nx),
            "temporal": (self.nt,),
            "spatiotemporal": (self.nt, self.ny, self.nx),
            "polarity": (2,),
            "joint": (self.nt, self.ny, self.nx, 2),
        }[name]

    def time_indices(self, times):
        if np.any(times < self.start_s) or np.any(times >= self.stop_s):
            raise ValueError("Histogram input must already be inside the shared interval")
        index = np.floor((times - self.start_s) / self.temporal_bin_s).astype(np.int64)
        # Correct roundoff at an edge using the same generated boundaries for both streams.
        left = self.start_s + index * self.temporal_bin_s
        right = self.start_s + (index + 1) * self.temporal_bin_s
        index += (times >= right).astype(np.int64) - (times < left).astype(np.int64)
        return index.clip(0, self.nt - 1)

    def temporal_positions(self, ids):
        left = self.start_s + ids * self.temporal_bin_s
        right = np.where(ids == self.nt - 1, self.stop_s,
                         self.start_s + (ids + 1) * self.temporal_bin_s)
        return left + (right - left) / 2

    def spatial_positions(self, ids):
        y, x = np.divmod(ids, self.nx)
        left = np.column_stack((x, y)).astype(np.float64) * self.spatial_bin_size
        right = np.minimum(left + self.spatial_bin_size, (self.width, self.height))
        return (left + right) / 2


@dataclass(frozen=True)
class Histogram:
    grid: Grid
    name: str
    ids: np.ndarray
    counts: np.ndarray
    n_events: int

    @property
    def shape(self):
        return self.grid.shape(self.name)

    @property
    def nbytes(self):
        return self.ids.nbytes + self.counts.nbytes

    def probabilities(self):
        if not self.n_events:
            return np.empty(0, dtype=np.float64)
        return self.counts / self.n_events


def histograms(events, grid, names):
    """Store only the sorted linear IDs and counts of occupied cells."""
    names = tuple(names)
    if any(math.prod(grid.shape(name)) > np.iinfo(np.int64).max for name in names):
        raise ValueError("Histogram index domain exceeds int64")
    if any(name in names for name in ("spatial", "spatiotemporal", "joint")):
        size = grid.spatial_bin_size
        space = (events.y.astype(np.int64, copy=False) // size * grid.nx
                 + events.x.astype(np.int64, copy=False) // size)
    if any(name in names for name in ("temporal", "spatiotemporal", "joint")):
        temporal = grid.time_indices(events.t)
    if "spatiotemporal" in names or "joint" in names:
        spacetime = temporal * (grid.nx * grid.ny) + space
    polarity = events.p > 0

    result = {}
    for name in names:
        if name == "spatial":
            indices = space
        elif name == "temporal":
            indices = temporal
        elif name == "spatiotemporal":
            indices = spacetime
        elif name == "polarity":
            indices = polarity.astype(np.int64)
        else:
            indices = 2 * spacetime + polarity
        occupied, counts = np.unique(indices, return_counts=True)
        result[name] = Histogram(grid, name, occupied, counts, len(events))
    return result
