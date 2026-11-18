"""Score complete saved v3 sequences. Example: python benchmarks/run_metric.py eds test --metric event_count"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import contextmanager
from functools import cached_property

import numpy as np

if __package__:
    from . import metrics as m, metrics_our, voxel_metric
    from .event_generation import generate_event_streams_v2 as v2
else:
    import metrics as m
    import metrics_our
    import voxel_metric
    from event_generation import generate_event_streams_v2 as v2

ROOT = v2.old.PROJECT
UNITS = dict(event_count="events", event_count_ratio="ratio", on_percentage="percent",
             chamfer="pixel-equivalent", chamfer_full="pixel-equivalent", chamfer_penalty="pixel-equivalent",
             chamfer_count="pixel-equivalent", isi_distribution="ms", isi_median_ms="ms",
             iei_distance="log10 interval distance", isi_cv="ratio", event_rate="events/s", same_ts="fraction",
             fano_factor="mean absolute log ratio", voxel_mae="events/voxel",
             voxel_rmse="events/voxel", voxel_relative_l1="ratio")
FULL_CHAMFERS = {"chamfer_full", "chamfer_penalty", "chamfer_count"}
PMSE = {"pmse_2_v2ce", "pmse_4_v2ce"}
UNITS.update({name: "squared events/voxel" for name in PMSE})
ARRAYS = {"isi_distribution", "event_rate", "fano_factor"}
DESCRIPTIVE = {"event_count", "on_percentage", "isi_distribution", "isi_median_ms",
               "isi_cv", "event_rate", "same_ts"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def settings(args):
    metric = args.metric
    options = {}
    if metric == "chamfer":
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        options.update(n=args.max_points, chunk_size=args.chunk_size, device=device,
                       v_px_per_ms=1.0, seed=0, torch_version=str(torch.__version__))
    elif metric in FULL_CHAMFERS:
        device = args.device or "cpu"
        if device == "cpu":
            import scipy
            from scipy.spatial import cKDTree
            version = scipy.__version__
        else:
            import cupy
            from cupyx.scipy.spatial import KDTree
            if not cupy.cuda.runtime.getDeviceCount():
                raise ValueError("CUDA requested but no GPU is visible to CuPy")
            version = cupy.__version__
        options.update(device=device, backend_version=version, workers=args.workers,
                       chunk_size=args.chunk_size, v_px_per_ms=1.0, sampling="all", eps=0.0)
        if metric == "chamfer_penalty":
            options["one_sided_absence"] = "scoring_domain_diameter"
        if metric == "chamfer_count":
            options.update(count_weight=args.count_weight, count_bins="RGB frame intervals by polarity")
    elif metric in PMSE or metric.startswith("voxel_"):
        options.update(temporal_bins=args.voxel_temporal_bins,
                       temporal_domain=("legacy frame pairs" if args.legacy_frame_voxels else "uniform full shared interval"),
                       legacy_frame_voxels=args.legacy_frame_voxels,
                       normalization="none")
        if metric in PMSE:
            options.update(voxelization="forward linear box integration; clipped at stop",
                           pooling="valid 3D; kernel=stride; polarity separate",
                           provenance="https://arxiv.org/html/2309.08891v2; III-A PMSE, II-B representation",
                           protocol="benchmark adaptation; not author-code/table reproduction",
                           kernel=int(metric.split("_")[1]))
        else:
            if args.legacy_frame_voxels:
                options["block_size"] = args.voxel_block_size
            options["native_v2v_protocol"] = "legacy frame edges; signed counts only"
    elif metric == "event_rate":
        options["bin_s"] = args.rate_bin_s
    elif metric == "same_ts":
        options.update(window_s=0.05, resolution_s=1e-9)
    elif metric == "fano_factor":
        options.update(window_s=0.05, bin_s=0.002, bin_px=8, eps=0.001)
    files = (Path(__file__), Path(m.__file__), Path(voxel_metric.__file__), Path(v2.__file__))
    if metric == "chamfer_count":
        files += (Path(metrics_our.__file__),)
    return dict(dataset=args.dataset, split=args.split, metric=metric, unit=UNITS[metric],
                interval="[first frame, last frame)", crop="original resolution (no crop)", options=options,
                numpy_version=np.__version__, code_sha256={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def stamp(path):
    try:
        stat = path.stat()
        return str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except FileNotFoundError:
        return str(path), None


class Sequence:
    """Read v3 geometry once and load aligned GT only if a pending score needs it."""
    def __init__(self, source):
        self.source = source

    @cached_property
    def geometry(self):
        _, _, edges, meta, _ = v2.sequence_info(self.source)
        return edges, meta

    @cached_property
    def gt(self):
        edges, meta = self.geometry
        raw = {k: np.load(self.source / f"ev_{k}.npy", mmap_mode="r",
                          allow_pickle=False) for k in "txyp"}
        t = raw["t"]
        if t.ndim != 1 or any(a.shape != t.shape for a in raw.values()):
            raise ValueError("GT arrays must have the same 1D shape")
        if not np.isfinite(t).all() or np.any(t[1:] < t[:-1]):
            raise ValueError("GT timestamps must be finite and sorted")
        ts = np.load(self.source / "frame_ts.npy", mmap_mode="r")
        a, b = np.searchsorted(t, [ts[0], ts[-1]])
        x, y, w, h = meta["crop"]
        ex, ey = raw["x"][a:b], raw["y"][a:b]
        keep = (ex >= x) & (ex < x+w) & (ey >= y) & (ey < y+h)
        p = raw["p"][a:b][keep]
        if not np.isin(p, (-1, 0, 1)).all():
            raise ValueError("Invalid GT polarity")
        return dict(t=t[a:b][keep].astype(np.float64) - meta["absolute_t0"],
                    x=ex[keep].astype(np.int64)-x, y=ey[keep].astype(np.int64)-y,
                    p=np.where(p > 0, 1, -1).astype(np.int8))


def source_for(name, split_dir, predictions, args):
    fallback = split_dir / name
    if args.data_root:
        return fallback
    sources = set()
    for folder in predictions:
        metadata = folder / name / "input.json"
        if metadata.is_file():
            try:
                path = Path(json.loads(metadata.read_text())["source"])
            except (OSError, ValueError, TypeError, KeyError):
                continue
            if path.parts[-3:] == (args.dataset, args.split, name) and path.is_dir():
                sources.add(path.resolve())
    if len(sources) > 1:
        raise ValueError(f"Conflicting source directories for {name}; specify --data-root")
    return next(iter(sources), fallback)


def verify_metadata(path, expected, args, name):
    recorded = json.loads(path.read_text())
    if Path(recorded["source"]).parts[-3:] != (args.dataset, args.split, name):
        raise ValueError("Prediction metadata names a different source")
    for key in expected.keys() - {"source"}:
        if recorded.get(key) != expected[key]:
            raise ValueError(f"Prediction metadata disagrees with source: {key}")


def read_events(path, meta):
    with np.load(path, allow_pickle=False) as archive:
        events = {k: archive[k] for k in "txyp"}
    t = m._times(events)
    m._coordinates(events, t, meta["width"], meta["height"])
    m._polarity(events, t)
    if np.any(t[1:] < t[:-1]) or np.any(t < 0) or np.any(t >= meta["duration"]):
        raise ValueError("Prediction timestamps outside the sorted v3 scoring domain")
    return events


def voxel_planes(events, edges, h, w, *, by_polarity=False):
    cuts = np.searchsorted(events["t"], edges)
    for a, b in zip(cuts[:-1], cuts[1:]):
        pixels = events["y"][a:b].astype(np.int64)*w + events["x"][a:b]
        on = events["p"][a:b] > 0
        if by_polarity:
            ids = pixels + (~on) * (h * w)
            yield np.bincount(ids, minlength=2*h*w).reshape(2, h, w)
        else:
            yield np.bincount(pixels, weights=np.where(on, 1, -1), minlength=h*w).reshape(h, w)


@contextmanager
def native_planes(path, edges, h, w):
    # Stream the compressed member one frame interval at a time.
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z["edges_s"], edges):
            raise ValueError("Native voxel edges disagree with source frame timestamps")
    with zipfile.ZipFile(path) as archive, archive.open("voxels.npy") as stream:
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError("Unsupported native voxel NPY version")
        shape, fortran, dtype = readers[version](stream)
        if shape != (len(edges)-1, h, w) or dtype.kind not in "iuf":
            raise ValueError("Invalid native voxel shape or dtype")
        if fortran:
            # A Fortran-order member interleaves intervals. Extract to disk for mmap.
            with tempfile.TemporaryDirectory() as tmp:
                file = Path(tmp) / "voxels.npy"
                with archive.open("voxels.npy") as src, file.open("wb") as dest:
                    shutil.copyfileobj(src, dest)
                array = np.load(file, mmap_mode="r", allow_pickle=False)
                try:
                    yield array
                finally:
                    del array
        else:
            def planes():
                for _ in range(shape[0]):
                    data = stream.read(h*w*dtype.itemsize)
                    yield np.frombuffer(data, dtype=dtype).reshape(h, w)
            yield planes()


def block_means(planes, block_size):
    """Average counts per pixel in each non-overlapping block of each frame gap."""
    if block_size == 1:
        yield from planes
        return
    for plane in planes:
        h, w = plane.shape[-2:]
        rows = np.arange(0, h, block_size)
        cols = np.arange(0, w, block_size)
        sums = np.add.reduceat(plane, rows, axis=-2, dtype=np.float64)
        sums = np.add.reduceat(sums, cols, axis=-1)
        areas = np.diff(np.append(rows, h))[:, None] * np.diff(np.append(cols, w))
        yield sums / areas


def voxel_score(metric, prediction, gt, edges, h, w, block_size=1):
    # Accumulate sufficient statistics, so inactive pixels and all intervals
    # receive the same weight as in voxel_metric on the full dense tensor.
    totals = np.zeros(4, dtype=np.float64)
    reference = voxel_planes(gt, edges, h, w)
    for pred, ref in zip(block_means(prediction, block_size), block_means(reference, block_size)):
        totals += voxel_metric._errors(pred[None], ref[None], None)
    absolute, squared, mass, count = totals
    if metric == "voxel_relative_l1":
        return absolute / mass if mass else np.nan
    return absolute/count if metric == "voxel_mae" else np.sqrt(squared/count)


def score(metric, events, seq, options):
    edges, meta = seq.geometry
    duration, w, h = float(edges[-1]), meta["width"], meta["height"]
    arrays = {}
    status = "ok"
    if metric in PMSE:
        value = getattr(voxel_metric, metric)(events, seq.gt, start=float(edges[0]),
                    stop=float(edges[-1]), temporal_bins=options["temporal_bins"],
                    height=h, width=w)
    elif metric == "event_rate":
        bins = np.arange(0, duration, options["bin_s"])
        bins = np.append(bins[bins < duration], duration)
        arrays = dict(edges_s=bins, rate_hz=m.event_rate(events, bins))
        value = len(events["t"])/duration
    elif metric == "iei_distance":
        value = m.iei_distance(events, seq.gt, w, duration)
        if np.isnan(value):
            return None, arrays, "no_intervals"
    elif metric.startswith("isi_"):
        intervals = m.isi_distribution(events, w, duration)
        if metric == "isi_distribution":
            arrays["isi_ms"] = intervals
            value = None
        elif metric == "isi_median_ms":
            value = np.median(intervals) if intervals.size else 0.0
        else:
            value = intervals.std()/max(intervals.mean(), 1e-9) if intervals.size else 0.0
        if not intervals.size:
            status = "no_intervals"
    elif metric == "event_count_ratio":
        value = m.event_count_ratio(events, seq.gt)
        if not seq.gt["t"].size:
            status = "empty_gt"
    elif metric == "chamfer":
        value = m.chamfer(events, seq.gt, w, duration, **{
            k: options[k] for k in ("device", "chunk_size", "n", "v_px_per_ms")})
        if not all(np.any(events["p"] == p) and np.any(seq.gt["p"] == p) for p in (1,-1)):
            status = "partial_polarity"
    elif metric == "fano_factor":
        value, arrays = m.fano_factor(events, w, h, duration, reference=seq.gt,
                                      return_maps=True, **options)
        if np.any(arrays["pred_active"] != arrays["gt_active"]):
            status = "partial_support"
    elif metric.startswith("voxel_"):
        if not options.get("legacy_frame_voxels", False):
            edges = np.linspace(edges[0], edges[-1], options["temporal_bins"] + 1)
        value = voxel_score(metric, voxel_planes(events, edges, h, w), seq.gt, edges, h, w,
                            options.get("block_size", 1))
    else:
        value = getattr(m, metric)(events, **options)
    if value is not None and not np.isfinite(value):
        return None, arrays, "undefined"
    return value, arrays, status


@contextmanager
def atomic_file(path, binary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {} if binary else {"encoding": "utf-8", "newline": ""}
    with tempfile.NamedTemporaryFile(mode="wb" if binary else "w", dir=path.parent,
                                     delete=False, **kwargs) as f:
        temp = Path(f.name)
        try:
            yield f
            f.close()
            os.replace(temp, path)
        finally:
            f.close()
            temp.unlink(missing_ok=True)


def export_csv(db, folder, metric):
    fields = ["sequence", "method", "value", "status"]
    if metric in FULL_CHAMFERS:
        fields.extend(["chamfer_on", "chamfer_off"])
    if metric == "chamfer_penalty":
        fields.append("missing_polarity_cost")
    if metric == "chamfer_count":
        fields.extend(["chamfer_geometry", "count_excess", "count_deficit", "count_penalty"])
    if metric in ("voxel_mae", "voxel_rmse"):
        fields.extend(["value_on", "value_off"])
    if metric in ARRAYS:
        fields.append("array_path")
    with atomic_file(folder / "results.csv") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        for (record,) in db.execute("SELECT record FROM scores ORDER BY sequence, method"):
            writer.writerow(json.loads(record))


def score_paths(source, method, file, metric):
    """Inputs whose identity determines whether a saved score is reusable."""
    ground = method == "ground_truth"
    needs_gt = ground or metric not in DESCRIPTIVE
    paths = [source / "frame_ts.npy"]
    if not ground:
        paths.extend([file.parent / "input.json", file])
    if needs_gt:
        paths.extend(source / f"ev_{k}.npy" for k in "txyp")
    return paths


def evaluate_one(args, seq, name, method, file, folder, db, config, *, read_prediction=None):
    metric = args.metric
    native = method == "v2v"
    ground = method == "ground_truth"
    metadata = file.parent / "input.json"
    needs_gt = ground or metric not in DESCRIPTIVE
    read_prediction = read_prediction or read_events
    paths = score_paths(seq.source, method, file, metric)
    signature = digest([stamp(p) for p in paths])
    previous = db.execute("SELECT signature,record FROM scores WHERE sequence=? AND method=?",
                          (name, method)).fetchone()
    if previous and previous[0] == signature:
        row = json.loads(previous[1])
        if row["status"] not in ("error", "missing_source", "missing_gt", "missing_prediction") and (
                not row.get("array_path") or (folder / row["array_path"]).is_file()):
            return "SKIP"
    row = dict(sequence=name, method=method, value=None, status="ok", array_path="")
    try:
        if not ground and not file.is_file():
            row["status"] = "missing_prediction"
        elif native and (not metric.startswith("voxel_") or
                         not config["options"].get("legacy_frame_voxels", False)):
            row["status"] = "not_applicable"
        elif not seq.source.is_dir():
            row["status"] = "missing_source"
        elif needs_gt and not all((seq.source / f"ev_{k}.npy").is_file() for k in "txyp"):
            row["status"] = "missing_gt"
        else:
            edges, meta = seq.geometry
            if not ground:
                verify_metadata(metadata, meta, args, name)
            arrays = {}
            if native:
                h, w = meta["height"], meta["width"]
                with native_planes(file, edges, h, w) as planes:
                    value = voxel_score(metric, planes, seq.gt, edges, h, w,
                                        config["options"].get("block_size", 1))
                if not np.isfinite(value):
                    value, row["status"] = None, "undefined"
            else:
                events = seq.gt if ground else read_prediction(file, meta)
                if metric in FULL_CHAMFERS:
                    options = config["options"]
                    kwargs = {k: options[k] for k in ("device", "workers", "chunk_size", "v_px_per_ms")}
                    if metric == "chamfer_count":
                        scores = metrics_our.chamfer_count(events, seq.gt, meta["width"], edges,
                                                          count_weight=options["count_weight"], **kwargs)
                    else:
                        if metric == "chamfer_penalty":
                            kwargs["height"] = meta["height"]
                        scores = getattr(m, metric)(events, seq.gt, meta["width"], meta["duration"], **kwargs)
                    for key, number in scores.items():
                        row[key] = float(number) if np.isfinite(number) else None
                    value = row["value"]
                    if value is None:
                        row["status"] = "undefined"
                    elif row["chamfer_on"] is None or row["chamfer_off"] is None:
                        row["status"] = "partial_polarity"
                elif metric in ("voxel_mae", "voxel_rmse"):
                    w, h = meta["width"], meta["height"]
                    compare = getattr(voxel_metric, metric + "_by_polarity")
                    size = config["options"].get("block_size", 1)
                    if not config["options"].get("legacy_frame_voxels", False):
                        edges = np.linspace(edges[0], edges[-1], config["options"]["temporal_bins"] + 1)
                    pred = voxel_planes(events, edges, h, w, by_polarity=True)
                    gt = voxel_planes(seq.gt, edges, h, w, by_polarity=True)
                    parts = compare(block_means(pred, size), block_means(gt, size))
                    value = parts["value"]
                    row.update(parts)
                else:
                    value, arrays, row["status"] = score(metric, events, seq, config["options"])
            if digest([stamp(p) for p in paths]) != signature:
                raise RuntimeError("Input changed during scoring; rerun this metric")
            row["value"] = float(value) if value is not None else None
            if metric == "event_count":
                row["value"] = int(value)
            if arrays:
                rel = Path("arrays") / method / f"{name}-{signature[:12]}.npz"
                with atomic_file(folder / rel, binary=True) as stream:
                    np.savez_compressed(stream, **arrays)
                row["array_path"] = rel.as_posix()
    except Exception as exc:
        row.update(value=None, status="error", array_path="")
        if metric in ("voxel_mae", "voxel_rmse"):
            row.update(value_on=None, value_off=None)
        if metric in FULL_CHAMFERS:
            row.update(chamfer_on=None, chamfer_off=None)
        print(f"  {name}/{method}: {exc}", flush=True)
    db.execute("INSERT OR REPLACE INTO scores VALUES (?,?,?,?)",
               (name, method, signature, json.dumps(row, allow_nan=False)))
    return row["status"]


def run(args):
    config = settings(args)
    data = args.data_root or v2.old.setting("EVTWEEN_DATA", v2.old.LAB / "evtween/data")
    source_split = data / args.dataset / args.split
    pred = args.pred_root / args.dataset / args.split
    dirs = [pred / method for method in v2.old.METHODS if (pred / method).is_dir()]
    names = {p.name for d in dirs for p in d.iterdir() if p.is_dir()}
    if source_split.is_dir():
        names.update(p.name for p in source_split.iterdir() if p.is_dir())
    if args.sequence:
        if args.sequence not in names:
            raise ValueError(f"Unknown sequence {args.sequence} in {source_split} or {pred}")
        names = {args.sequence}
    if not names:
        raise ValueError(f"No sequences found in {source_split} or {pred}")
    methods = list(dict.fromkeys(args.methods or v2.old.METHODS))
    if args.metric in DESCRIPTIVE:
        methods.insert(0, "ground_truth")
    folder = args.report_root / args.dataset / args.metric / f"{args.split}-full-{digest(config)[:12]}"
    folder.mkdir(parents=True, exist_ok=True)
    with atomic_file(folder / "config.json") as stream:
        json.dump(config, stream, indent=2)
    print(f"Results: {folder / 'results.csv'}", flush=True)
    db = sqlite3.connect(folder / ".resume.sqlite", timeout=3600)
    failed = False
    try:
        db.execute("CREATE TABLE IF NOT EXISTS scores (sequence TEXT, method TEXT, signature TEXT, "
                   "record TEXT, PRIMARY KEY(sequence, method))")
        for name in sorted(names):
            seq = Sequence(source_for(name, source_split, dirs, args))
            for method in methods:
                file = pred / method / name / ("voxels.npz" if method == "v2v" else "events.npz")
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    status = evaluate_one(args, seq, name, method, file, folder, db, config)
                    if status != "SKIP":
                        export_csv(db, folder, args.metric)
                print(f"{status} {name}/{method}", flush=True)
                failed |= status == "error"
        with db:
            db.execute("BEGIN IMMEDIATE")
            export_csv(db, folder, args.metric)
    finally:
        db.close()
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("split", choices=("train", "test"))
    parser.add_argument("--metric", required=True, choices=tuple(UNITS))
    parser.add_argument("--methods", nargs="+", choices=v2.old.METHODS)
    parser.add_argument("--sequence")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Chamfer device (sampled: auto, full: CPU)")
    parser.add_argument("--max-points", type=int, help="Sampled Chamfer only: cap per polarity (default 20000)")
    parser.add_argument("--chunk-size", type=int, help="Queries per batch (sampled: 4096, full: 131072)")
    parser.add_argument("--workers", type=int, default=1, help="CPU query threads for full-stream Chamfer")
    parser.add_argument("--rate-bin-s", type=float, default=0.01, help="Event-rate bin duration in seconds")
    parser.add_argument("--data-root", type=Path, help="Override the original dataset root containing dataset/split")
    parser.add_argument("--pred-root", type=Path, default=ROOT / "data_test_evt")
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports")
    parser.add_argument("--count-weight", type=float, help="Count penalty weight for chamfer_count (default: 1.0 pixel-equivalent)")
    parser.add_argument("--voxel-block-size", type=int, default=1, metavar="N",
                        help="Historical frame-pair mode only: spatial averaging (requires --legacy-frame-voxels)")
    parser.add_argument("--voxel-temporal-bins", type=int, default=160,
                        help="Uniform bins over the entire shared interval (default: 160)")
    parser.add_argument("--legacy-frame-voxels", action="store_true",
                        help="Reproduce old voxel_* frame-pair scores, including native V2V; incompatible with PMSE")
    args = parser.parse_args()
    if args.legacy_frame_voxels and not args.metric.startswith("voxel_"):
        parser.error("--legacy-frame-voxels applies only to legacy voxel_* metrics")
    if args.voxel_temporal_bins < 1:
        parser.error("--voxel-temporal-bins must be positive")
    if args.metric in PMSE and args.voxel_temporal_bins < int(args.metric.split('_')[1]):
        parser.error("PMSE requires at least kernel-size temporal bins")
    if args.voxel_block_size < 1:
        parser.error("--voxel-block-size must be positive")
    if args.voxel_block_size != 1 and not args.legacy_frame_voxels:
        parser.error("Spatial block averaging is disabled for full-interval metrics; use the full image and temporal bins")
    if args.voxel_block_size != 1 and args.metric not in ("voxel_mae", "voxel_rmse"):
        parser.error("--voxel-block-size applies to voxel_mae and voxel_rmse")
    if args.count_weight is not None:
        if args.metric != "chamfer_count" or not np.isfinite(args.count_weight) or args.count_weight < 0:
            parser.error("--count-weight requires chamfer_count and a finite nonnegative value")
    else:
        args.count_weight = 1.0
    if Path(args.dataset).name != args.dataset or args.dataset in ("", ".", ".."):
        parser.error("Expected a single dataset directory name")
    if args.metric in FULL_CHAMFERS and args.max_points is not None:
        parser.error(f"{args.metric} uses every event; --max-points applies to chamfer only")
    if args.max_points is None:
        args.max_points = 20000
    if args.chunk_size is None:
        args.chunk_size = 131072 if args.metric in FULL_CHAMFERS else 4096
    if min(args.workers, args.chunk_size, args.max_points) < 1 or not np.isfinite(args.rate_bin_s) or args.rate_bin_s <= 0:
        parser.error("Sample cap, chunk size and bin duration must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

