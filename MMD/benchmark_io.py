"""Read saved v3 benchmark streams through the repository's own benchmark code.

GT geometry, metadata checks and prediction reading come from benchmarks/run_metric.py,
the same code every other benchmark metric uses.
"""
import argparse
from functools import cached_property
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import run_metric as benchmark  # noqa: E402

from .events import Events, prepare_events  # noqa: E402


class AlignmentError(ValueError):
    """The prediction and reference do not have a verified common domain."""


def component(value):
    if Path(value).name != value or value in ("", ".", ".."):
        raise argparse.ArgumentTypeError("Expected one directory name")
    return value


class Reference:
    """One converted GT sequence: geometry, events and prediction alignment."""

    def __init__(self, source, allow_benchmark_crop=False):
        self.source = Path(source)
        self.allow_benchmark_crop = allow_benchmark_crop

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
            if crop != full_frame and not self.allow_benchmark_crop:
                raise AlignmentError("Prediction uses a crop. Use full-resolution events or "
                                     "pass --allow-benchmark-crop")
            if recorded.get("timestamp_unit", "s") != "s":
                raise AlignmentError("Saved v3 prediction timestamps must be in seconds")
            expected = dict(expected, width=w, height=h, crop=crop)
            benchmark.verify_metadata(sidecar, expected, args, sequence)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise AlignmentError(str(exc)) from exc
        return expected

    def select(self, crop, start, stop):
        """GT events in [start, stop) inside the crop, in crop coordinates."""
        events = self.events.select(start, stop)
        x, y, width, height = crop
        if tuple(crop) == (0, 0, *self.geometry[1]):
            return events
        keep = ((events.x >= x) & (events.x < x + width)
                & (events.y >= y) & (events.y < y + height))
        return Events(events.x[keep].astype(np.int64) - x, events.y[keep].astype(np.int64) - y,
                      events.t[keep], events.p[keep])


def load_prediction(path, metadata):
    raw = benchmark.read_events(path, metadata)
    return prepare_events(raw, metadata["width"], metadata["height"], unit="s", polarity="signed")
