"""Configuration for histogram extraction and comparison."""
from dataclasses import dataclass
import math


DISTRIBUTIONS = ("spatial", "temporal", "spatiotemporal", "polarity", "joint")
COMPARISONS = ("jsd", "kl", "wasserstein")


@dataclass(frozen=True)
class Config:
    spatial_bin_size: int = 1
    temporal_bin_s: float = 0.01
    time_window_s: tuple[float, float] | None = None
    distributions: tuple[str, ...] = DISTRIBUTIONS[:-1]
    comparisons: tuple[str, ...] = ("jsd", "kl")
    normalize_jsd: bool = True
    log_base: str = "natural"
    kl_epsilon: float = 1e-12
    reverse_kl: bool = False
    count_epsilon: float = 1e-12
    max_spatial_transport_pairs: int = 50_000
    allow_benchmark_crop: bool = False
    save_distributions: bool = False
    save_results: bool = True
    save_plots: bool = False

    def __post_init__(self):
        for name in ("spatial_bin_size", "max_spatial_transport_pairs"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("temporal_bin_s", "count_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.kl_epsilon < 1:
            raise ValueError("kl_epsilon must be in [0, 1)")
        if self.log_base not in ("natural", "2"):
            raise ValueError("log_base must be 'natural' or '2'")
        for name, allowed in (("distributions", DISTRIBUTIONS), ("comparisons", COMPARISONS)):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
                raise ValueError(f"{name} must be a nonempty list of unique values from {allowed}")
            object.__setattr__(self, name, values)
        if self.time_window_s is not None:
            window = tuple(self.time_window_s)
            if len(window) != 2 or not all(map(math.isfinite, window)) or window[1] <= window[0]:
                raise ValueError("time_window_s must contain two increasing finite times")
            object.__setattr__(self, "time_window_s", window)
