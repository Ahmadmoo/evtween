"""Score saved v3 event streams using shared spatial and temporal histograms."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from tempfile import mkdtemp
from zipfile import BadZipFile

import numpy as np
import scipy

from .benchmark_io import AlignmentError, Reference, benchmark, load_prediction
from .config import Config, DISTRIBUTIONS
from .distributions import histograms
from .metrics import count_error, divergences


BASE_COLUMNS = (
    "dataset", "split", "sequence", "generator", "status", "detail", "start_s", "stop_s",
    "absolute_t0_s", "width", "height", "crop_x", "crop_y", "spatial_bin_size", "temporal_bin_s",
    "n_gt", "n_gen", "absolute_count_difference", "relative_count_error", "gt_duplicates",
    "gen_duplicates", "gt_outside_crop", "alignment", "distribution_path", "plot_path",
)


def score_prediction(args, config, reference, sequence, generator, path, output):
    row = dict(dataset=args.dataset, split=args.split, sequence=sequence, generator=generator)
    if not path.is_file():
        if path.with_name("voxels.npz").is_file():
            return dict(row, status="not_applicable", detail="Only native voxels are available")
        return dict(row, status="missing_prediction", detail=str(path))
    if not reference.source.is_dir():
        return dict(row, status="missing_source", detail=str(reference.source))
    if not all((reference.source / f"ev_{key}.npy").is_file() for key in "xytp"):
        return dict(row, status="missing_gt", detail=str(reference.source))

    metadata = reference.align(path, args, sequence)
    names = config.distributions
    if config.save_plots:
        names = tuple(dict.fromkeys((*names, *DISTRIBUTIONS[:-1])))
    grid, n_gt, gt_duplicates, gt_outside_crop, gt = reference.select(metadata, names)
    predicted = load_prediction(path, metadata).select(grid.start_s, grid.stop_s)
    generated = histograms(predicted, grid, names)
    if n_gt and len(predicted):
        status = "ok"
    elif n_gt:
        status = "empty_generated"
    elif len(predicted):
        status = "empty_gt"
    else:
        status = "empty_both"

    row.update(status=status, start_s=grid.start_s, stop_s=grid.stop_s,
               absolute_t0_s=metadata["absolute_t0"], width=grid.width, height=grid.height,
               crop_x=metadata["crop"][0], crop_y=metadata["crop"][1],
               spatial_bin_size=grid.spatial_bin_size, temporal_bin_s=grid.temporal_bin_s,
               gt_duplicates=gt_duplicates, gen_duplicates=predicted.duplicates(),
               gt_outside_crop=gt_outside_crop, alignment="v3_metadata_verified")
    row.update(count_error(n_gt, len(predicted), config.count_epsilon))
    for name in config.distributions:
        scores = divergences(gt[name], generated[name], config)
        row.update({f"{name}_{key}": value for key, value in scores.items()})

    if config.save_distributions or config.save_plots:
        artifacts = output / "pairs" / sequence / generator
        artifacts.mkdir(parents=True, exist_ok=True)
    if config.save_distributions:
        payload = {}
        for name in config.distributions:
            payload[f"{name}_shape"] = gt[name].shape
            for label, histogram in (("gt", gt[name]), ("gen", generated[name])):
                payload[f"{name}_{label}_ids"] = histogram.ids
                payload[f"{name}_{label}_counts"] = histogram.counts
        file = artifacts / "distributions.npz"
        np.savez_compressed(file, **payload)
        row["distribution_path"] = file.relative_to(output).as_posix()
    if config.save_plots and status == "ok":
        from .plots import plot_comparison

        file = artifacts / "comparison.png"
        plot_comparison(file, gt, generated, f"{args.dataset} / {sequence} · {generator}")
        row["plot_path"] = file.relative_to(output).as_posix()
    return row


def write_results(file, rows, config):
    stats = ["grid_bins", "gt_nonzero_bins", "gen_nonzero_bins", "histogram_bytes"]
    if "jsd" in config.comparisons:
        stats.append("jsd")
    if "kl" in config.comparisons:
        stats.append("kl_gt_gen")
        if config.reverse_kl:
            stats.append("kl_gen_gt")
    if "wasserstein" in config.comparisons:
        stats.extend(("wasserstein", "wasserstein_status"))
    columns = (*BASE_COLUMNS, *(f"{name}_{key}" for name in config.distributions for key in stats))
    with file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(args, config):
    data_root = args.data_root or benchmark.v2.old.setting(
        "EVTWEEN_DATA", benchmark.v2.old.LAB / "evtween/data")
    source_split = data_root / args.dataset / args.split
    predictions = args.pred_root / args.dataset / args.split
    folders = sorted(p for p in predictions.glob("*") if p.is_dir() and not p.name.startswith("."))
    generators = list(dict.fromkeys(args.methods or [p.name for p in folders]))
    if not generators:
        raise ValueError(f"No generator directories found in {predictions}")
    folders = [predictions / generator for generator in generators
               if (predictions / generator).is_dir()]
    sequences = {p.name for folder in (*folders, source_split) for p in folder.glob("*")
                 if p.is_dir() and not p.name.startswith(".")}
    if args.sequence:
        missing = set(args.sequence) - sequences
        if missing:
            raise ValueError(f"Unknown sequences: {', '.join(sorted(missing))}")
        sequences = set(args.sequence)
    if not sequences:
        raise ValueError(f"No sequences found in {source_split} or {predictions}")

    output = args.output
    if output is None:
        parent = Path(__file__).resolve().parent / "runs"
        parent.mkdir(exist_ok=True)
        output = Path(mkdtemp(prefix="run-", dir=parent))
    else:
        output.mkdir(parents=True, exist_ok=False)

    (output / "config.json").write_text(json.dumps({
        "histogram": asdict(config),
        "dataset": args.dataset, "split": args.split,
        "sequences": sorted(sequences), "generators": generators,
        "data_root": str(data_root.resolve()), "pred_root": str(args.pred_root.resolve()),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
    }, indent=2) + "\n", encoding="utf-8")

    rows = []
    for sequence in sorted(sequences):
        try:
            source = benchmark.source_for(sequence, source_split, folders, args)
            reference = Reference(source, config)
        except ValueError as exc:
            reference = None
            source_error = str(exc)
        for generator in generators:
            row = dict(dataset=args.dataset, split=args.split, sequence=sequence, generator=generator)
            path = predictions / generator / sequence / "events.npz"
            try:
                if reference is None:
                    raise AlignmentError(source_error)
                row = score_prediction(args, config, reference, sequence, generator, path, output)
            except AlignmentError as exc:
                row.update(status="alignment_unavailable", detail=str(exc))
            except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, BadZipFile) as exc:
                row.update(status="error", detail=str(exc))
            rows.append(row)
            detail = f" · {row['detail']}" if row.get("detail") else ""
            print(f"{row['status']} {sequence}/{generator}{detail}", flush=True)
    if config.save_results:
        write_results(output / "results.csv", rows, config)
    print(f"Output: {output}")
    return rows


def component(value):
    if Path(value).name != value or value in ("", ".", ".."):
        raise argparse.ArgumentTypeError("Expected one directory name")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=component)
    parser.add_argument("split", choices=("train", "test"))
    parser.add_argument("--sequence", type=component, nargs="+")
    parser.add_argument("--methods", type=component, nargs="+")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pred-root", type=Path, default=benchmark.ROOT / "data_test_evt")
    parser.add_argument("--config", type=Path, help="JSON overrides to histogram Config")
    parser.add_argument("--output", type=Path, help="New output directory")
    args = parser.parse_args()
    try:
        config = Config(**json.loads(args.config.read_text())) if args.config else Config()
        rows = run(args, config)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    completed = {"ok", "empty_gt", "empty_generated", "empty_both"}
    return int(not any(row["status"] in completed for row in rows)
               or any(row["status"] in {"error", "alignment_unavailable"} for row in rows))


if __name__ == "__main__":
    sys.exit(main())
