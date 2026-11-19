"""Adapter for the repository's saved v3 benchmark streams."""
from functools import cached_property
import json

import numpy as np

from benchmarks import run_metric as benchmark

from .distributions import Grid, histograms
from .events import Events, prepare_events


class AlignmentError(ValueError):
    """The prediction and reference do not have a verified common domain."""


class Reference:
    def __init__(self, source, config):
        self.source = source
        self.config = config
        self.cache = {}

    @cached_property
    def geometry(self):
        _, _, _, metadata, size = benchmark.v2.sequence_info(self.source)
        return metadata, size

    @cached_property
    def events(self):
        metadata, (width, height) = self.geometry
        raw = {key: np.load(self.source / f"ev_{key}.npy", mmap_mode="r",
                            allow_pickle=False) for key in "xytp"}
        return prepare_events(raw, width, height, unit="s", origin=metadata["absolute_t0"],
                              polarity="benchmark")

    def align(self, prediction, args, sequence):
        sidecar = prediction.with_name("input.json")
        if not sidecar.is_file():
            raise AlignmentError("Missing input.json with the prediction geometry and clock")
        expected, (width, height) = self.geometry
        full_frame = [0, 0, width, height]
        try:
            recorded = json.loads(sidecar.read_text())
            crop = recorded["crop"]
            if not isinstance(crop, list) or len(crop) != 4 or any(type(n) is not int for n in crop):
                raise AlignmentError("Prediction crop must be four integers")
            x, y, w, h = crop
            if x < 0 or y < 0 or w < 1 or h < 1 or x + w > width or y + h > height:
                raise AlignmentError("Prediction crop is outside the original frame")
            if crop != full_frame and not self.config.allow_benchmark_crop:
                raise AlignmentError("Prediction uses a crop. Use full-resolution events or explicitly enable allow_benchmark_crop")
            if recorded.get("timestamp_unit", "s") != "s":
                raise AlignmentError("Saved v3 prediction timestamps must be in seconds")
            expected = dict(expected, width=w, height=h, crop=crop)
            benchmark.verify_metadata(sidecar, expected, args, sequence)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise AlignmentError(str(exc)) from exc
        return expected

    def select(self, metadata, names):
        crop = tuple(metadata["crop"])
        if crop in self.cache:
            return self.cache[crop]
        x, y, width, height = crop
        duration = metadata["duration"]
        start, stop = self.config.time_window_s or (0.0, duration)
        if not 0 <= start < stop <= duration:
            raise AlignmentError(f"Requested window [{start}, {stop}) exceeds the shared [0, {duration}) interval")
        grid = Grid(width, height, start, stop, self.config.spatial_bin_size, self.config.temporal_bin_s)
        events = self.events.select(start, stop)
        before_crop = len(events)
        if crop != (0, 0, *self.geometry[1]):
            # The matching crop is explicitly permitted and recorded in input.json.
            keep = ((events.x >= x) & (events.x < x + width)
                    & (events.y >= y) & (events.y < y + height))
            events = Events(events.x[keep].astype(np.int64) - x,
                            events.y[keep].astype(np.int64) - y, events.t[keep], events.p[keep])
        prepared = (grid, len(events), events.duplicates(), before_crop - len(events),
                    histograms(events, grid, names))
        self.cache[crop] = prepared
        return prepared


def load_prediction(path, metadata):
    raw = benchmark.read_events(path, metadata)
    return prepare_events(raw, metadata["width"], metadata["height"], unit="s", polarity="signed")
