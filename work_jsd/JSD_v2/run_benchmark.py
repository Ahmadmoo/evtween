"""Score saved Evtween v3 event streams against their ground truth."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from tempfile import mkdtemp
from zipfile import BadZipFile

from jsd_work.histogram.benchmark_io import AlignmentError, Reference, benchmark, load_prediction
from jsd_work.histogram.events import Events
from jsd_work.histogram.run_benchmark import component

from .metric import KernelReference


COLUMNS = (
    "dataset", "split", "sequence", "generator", "status", "detail",
    "start_s", "stop_s", "absolute_t0_s", "width", "height", "crop_x", "crop_y",
    "sigma_spatial_px", "sigma_temporal_ms", "n_gt", "n_gen",
    "distance", "distance_on", "distance_off", "empty_baseline", "relative_to_empty",
)
COMPLETED = {"ok", "empty_gt", "empty_generated", "empty_both"}


def _select_reference(source, crop, start, stop):
    events = source.events.select(start, stop)
    x, y, width, height = crop
    if crop == (0, 0, *source.geometry[1]):
        return events
    keep = ((events.x >= x) & (events.x < x + width)
            & (events.y >= y) & (events.y < y + height))
    return Events(events.x[keep].astype("int64") - x, events.y[keep].astype("int64") - y,
                  events.t[keep], events.p[keep])


def _score(path, args, sequence, source, prepared, row):
    if not path.is_file():
        if path.with_name("voxels.npz").is_file():
            return dict(row, status="not_applicable", detail="Only native voxels are available")
        return dict(row, status="missing_prediction", detail=str(path))
    if not all((source.source / f"ev_{key}.npy").is_file() for key in "xytp"):
        return dict(row, status="missing_gt", detail=str(source.source))

    metadata = source.align(path, args, sequence)
    start, stop = args.time_window_s or (0.0, metadata["duration"])
    if not 0 <= start < stop <= metadata["duration"]:
        raise AlignmentError(f"Requested interval [{start}, {stop}) is outside the shared scoring domain")
    crop = tuple(metadata["crop"])
    predicted = load_prediction(path, metadata).select(start, stop)
    if crop not in prepared:
        gt = _select_reference(source, crop, start, stop)
        prepared[crop] = (len(gt), KernelReference(
            gt, sigma_spatial_px=args.sigma_spatial_px,
            sigma_temporal_ms=args.sigma_temporal_ms, block_size=args.block_size,
        ))
    n_gt, reference = prepared[crop]
    n_gen = len(predicted)
    status = {
        (True, True): "ok",
        (True, False): "empty_generated",
        (False, True): "empty_gt",
        (False, False): "empty_both",
    }[bool(n_gt), bool(n_gen)]
    return dict(
        row, status=status, start_s=start, stop_s=stop, absolute_t0_s=metadata["absolute_t0"],
        width=crop[2], height=crop[3], crop_x=crop[0], crop_y=crop[1], n_gt=n_gt, n_gen=n_gen,
        **asdict(reference.score(predicted)),
    )


def _rows(args):
    data_root = args.data_root or benchmark.v2.old.setting(
        "EVTWEEN_DATA", benchmark.v2.old.LAB / "evtween/data",
    )
    split = data_root / args.dataset / args.split
    predictions = args.pred_root / args.dataset / args.split
    methods = args.methods or sorted(
        folder.name for folder in predictions.iterdir() if folder.is_dir() and not folder.name.startswith(".")
    )
    if not methods:
        raise ValueError(f"No generator directories found in {predictions}")
    folders = [predictions / method for method in dict.fromkeys(methods)]
    for sequence in dict.fromkeys(args.sequence):
        prepared = {}
        try:
            source = Reference(benchmark.source_for(sequence, split, folders, args), args)
        except ValueError as exc:
            source = None
            source_error = str(exc)
        for folder in folders:
            row = dict(dataset=args.dataset, split=args.split, sequence=sequence, generator=folder.name,
                       sigma_spatial_px=args.sigma_spatial_px, sigma_temporal_ms=args.sigma_temporal_ms)
            path = folder / sequence / "events.npz"
            try:
                if source is None:
                    raise AlignmentError(source_error)
                row = _score(path, args, sequence, source, prepared, row)
            except AlignmentError as exc:
                row.update(status="alignment_unavailable", detail=str(exc))
            except (OSError, ArithmeticError, ValueError, TypeError, KeyError, BadZipFile) as exc:
                row.update(status="error", detail=str(exc))
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=component)
    parser.add_argument("split", choices=("train", "test"))
    parser.add_argument("--sequence", type=component, nargs="+", required=True)
    parser.add_argument("--methods", type=component, nargs="+")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pred-root", type=Path, default=benchmark.ROOT / "data_test_evt")
    parser.add_argument("--time-window-s", type=float, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--allow-benchmark-crop", action="store_true")
    parser.add_argument("--sigma-spatial-px", type=float, required=True)
    parser.add_argument("--sigma-temporal-ms", type=float, required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, help="Path to a new results CSV")
    args = parser.parse_args()
    try:
        output = args.output
        if output is None:
            parent = Path(__file__).resolve().parent / "runs"
            parent.mkdir(exist_ok=True)
            output = Path(mkdtemp(prefix="run-", dir=parent)) / "results.csv"
        statuses = set()
        with output.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            for row in _rows(args):
                writer.writerow(row)
                stream.flush()
                statuses.add(row["status"])
                print(f"{row['status']} {row['sequence']}/{row['generator']} {row.get('detail', '')}", flush=True)
        print(f"Output: {output}")
        return int(not statuses & COMPLETED or bool(statuses & {"error", "alignment_unavailable"}))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
