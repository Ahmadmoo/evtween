"""Score saved v3 event streams against GT with the Gaussian kernel distance (MMD).

One process per sequence (--jobs), each scoring its generators one after another with
--threads threads. GT is loaded once per sequence and its empty baseline is reused.
With --device cuda the kernel blocks run on GPUs (PyTorch, float64): by default one
sequence per visible GPU, each GPU used by one sequence at a time.
Defaults split the CPUs this process may use (Slurm allocation / taskset), not the
machine's core count. Every finished (sequence, generator) is saved at once, so
rerunning the same command skips finished pairs whose inputs are unchanged.

    python -m MMD.run eds test --sigma-spatial-px 2 --sigma-temporal-ms 1
    python -m MMD.run eds test --sigma-spatial-px 2 --sigma-temporal-ms 1 --dry-run
    python -m MMD.run eds test --sigma-spatial-px 2 --sigma-temporal-ms 1 --device cuda
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from zipfile import BadZipFile

# Parallelism comes from our own processes and threads; keep BLAS single-threaded.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "MMD"

from .benchmark_io import AlignmentError, Reference, benchmark, component, load_prediction  # noqa: E402
from .kernel import METRIC_VERSION, WEIGHTINGS, MMDReference  # noqa: E402

HERE = Path(__file__).resolve().parent
COMPLETED = {"ok", "empty_gt", "empty_generated", "empty_both", "not_applicable"}
SCORES = ("count_distance", "count_distance_on", "count_distance_off", "count_empty_baseline",
          "count_relative_to_empty", "count_truncation_bound",
          "prob_distance", "prob_distance_on", "prob_distance_off", "prob_truncation_bound")
COLUMNS = ("dataset", "split", "sequence", "generator", "status", "detail",
           "start_s", "stop_s", "absolute_t0_s", "width", "height", "crop_x", "crop_y",
           "sigma_spatial_px", "sigma_temporal_ms", "algorithm", "cutoff_sigma", "device",
           "n_gt", "n_gen", *SCORES, "seconds", "gt_source", "prediction")


def allocated_cpus():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def fingerprint(paths):
    return benchmark.digest([benchmark.stamp(Path(p)) for p in paths])


def gt_files(source):
    return [source / f"ev_{k}.npy" for k in "xytp"] + [source / "frame_ts.npy", source / "frames"]


def visible_gpus(device):
    """[] on cpu; the GPUs to use for cuda / cuda:N, checked before any work starts."""
    if device == "cpu":
        return []
    try:
        import torch
    except ImportError:
        raise SystemExit("--device cuda needs PyTorch in this environment") from None
    if not torch.cuda.is_available():
        raise SystemExit("--device cuda: PyTorch sees no CUDA GPU (check the job's GPU request "
                         "and CUDA_VISIBLE_DEVICES)")
    count = torch.cuda.device_count()
    if device == "cuda":
        return [f"cuda:{i}" for i in range(count)]
    if int(device[5:]) >= count:
        raise SystemExit(f"{device} is not visible; {count} GPU(s) available")
    return [device]


def duration(seconds):
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


class Progress:
    """Where one (sequence, generator) pair is: printed every `every` seconds.

    Stages: "<ON|OFF> merge" (sorting the events), "<ON|OFF>" (the kernel sums), and the
    same with "GT baseline" once per sequence. The percentage is an estimate of the
    kernel work, from event counts per spatial tile.
    """

    def __init__(self, prefix, every):
        self.prefix, self.every = prefix, every
        self.label, self.start, self.last = "", time.perf_counter(), 0.0

    def stage(self, label, events):
        self.label = label
        self.start = self.last = time.perf_counter()
        if events >= 5_000_000:
            print(f"{self.prefix} {label}: {events:,} events", flush=True)

    def __call__(self, done, total, pairs):
        now = time.perf_counter()
        finished = done >= total * (1 - 1e-9)
        elapsed = now - self.start
        if (now - self.last < self.every and not finished) or (finished and elapsed < self.every):
            return
        self.last = now
        fraction = min(1.0, done / total) if total else 1.0
        left = "done" if finished else (
            f"~{duration(elapsed * (1 - fraction) / fraction)} left" if fraction else "")
        print(f"{self.prefix} {self.label}: {fraction:6.1%}  {duration(elapsed)} elapsed, {left}, "
              f"{pairs / max(elapsed, 1e-9):.2g} pairs/s", flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    os.replace(temporary, path)


def row_file(rows_dir, sequence, generator):
    return rows_dir / sequence / f"{generator}.json"


def finished(rows_dir, sequence, generator, inputs):
    try:
        saved = json.loads(row_file(rows_dir, sequence, generator).read_text())
    except (OSError, ValueError):
        return False
    return saved.get("status") in COMPLETED and saved.get("_inputs") == inputs


def score_sequence(sequence, source, generators, settings, rows_dir, threads, device="cpu",
                   free_gpus=None):
    """Worker: score every pending generator of one sequence. Returns statuses.

    With free_gpus (a queue), take one GPU for this sequence and give it back afterwards.
    """
    if free_gpus is None:
        return _score_sequence(sequence, source, generators, settings, rows_dir, threads, device)
    device = free_gpus.get()
    try:
        return _score_sequence(sequence, source, generators, settings, rows_dir, threads, device)
    finally:
        free_gpus.put(device)


def _score_sequence(sequence, source, generators, settings, rows_dir, threads, device):
    args = argparse.Namespace(**settings)
    reference = Reference(Path(source), args.allow_benchmark_crop)
    scorers, statuses = {}, []
    for number, (generator, path, inputs) in enumerate(generators, 1):
        path = Path(path)
        row = dict(dataset=args.dataset, split=args.split, sequence=sequence, generator=generator,
                   sigma_spatial_px=args.sigma_spatial_px, sigma_temporal_ms=args.sigma_temporal_ms,
                   algorithm=args.algorithm, gt_source=str(source), prediction=str(path))
        began = time.perf_counter()
        try:
            if not path.is_file():
                if path.with_name("voxels.npz").is_file():
                    row.update(status="not_applicable", detail="Only native voxels are available")
                else:
                    row.update(status="missing_prediction", detail=str(path))
            elif not all((Path(source) / f"ev_{k}.npy").is_file() for k in "xytp"):
                row.update(status="missing_gt", detail=str(source))
            else:
                metadata = reference.align(path, args, sequence)
                start, stop = args.time_window_s or (0.0, metadata["duration"])
                if not 0 <= start < stop <= metadata["duration"]:
                    raise AlignmentError(f"Interval [{start}, {stop}) is outside the shared "
                                         f"[0, {metadata['duration']}) scoring domain")
                crop = tuple(metadata["crop"])
                if crop not in scorers:
                    gt = reference.select(crop, start, stop)
                    scorers[crop] = MMDReference(
                        gt, sigma_spatial_px=args.sigma_spatial_px,
                        sigma_temporal_ms=args.sigma_temporal_ms, weightings=args.weighting,
                        algorithm=args.algorithm, cutoff_sigma=args.cutoff_sigma,
                        tile_px=args.tile_px, block_size=args.block_size,
                        chunk_size=args.chunk_size, threads=threads, device=device)
                scorer = scorers[crop]
                if args.progress_every:
                    prefix = f"  [{sequence}/{generator}]"
                    print(f"{prefix} start: generator {number}/{len(generators)} pending here",
                          flush=True)
                    scorer.progress = Progress(prefix, args.progress_every)
                predicted = load_prediction(path, metadata).select(start, stop)
                n_gt, n_gen = scorer.n_gt, len(predicted)
                status = {(True, True): "ok", (True, False): "empty_generated",
                          (False, True): "empty_gt", (False, False): "empty_both"}[n_gt > 0, n_gen > 0]
                row.update(status=status, start_s=start, stop_s=stop,
                           absolute_t0_s=metadata["absolute_t0"], width=crop[2], height=crop[3],
                           crop_x=crop[0], crop_y=crop[1], cutoff_sigma=scorer.cutoff_sigma,
                           device=scorer.device,
                           n_gt=n_gt, n_gen=n_gen, **scorer.score(predicted))
        except AlignmentError as exc:
            row.update(status="alignment_unavailable", detail=str(exc))
        except (OSError, ArithmeticError, ValueError, TypeError, KeyError, BadZipFile) as exc:
            row.update(status="error", detail=f"{type(exc).__name__}: {exc}")
        row["seconds"] = round(time.perf_counter() - began, 3)
        row["_inputs"] = inputs
        write_json(row_file(rows_dir, sequence, generator), row)
        statuses.append(row["status"])
        print(f"{row['status']:>22} {sequence}/{generator} {row['seconds']:.1f}s "
              f"{row.get('detail', '')}", flush=True)
    return statuses


def export_csv(output, rows_dir, pairs):
    rows = []
    for sequence, generator in pairs:
        try:
            rows.append(json.loads(row_file(rows_dir, sequence, generator).read_text()))
        except (OSError, ValueError):
            continue
    temporary = output / ".results.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output / "results.csv")
    return rows


def arguments(argv):
    cpus = allocated_cpus()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=component)
    parser.add_argument("split", choices=("train", "test"))
    parser.add_argument("--sigma-spatial-px", type=float, required=True)
    parser.add_argument("--sigma-temporal-ms", type=float, required=True)
    parser.add_argument("--weighting", nargs="+", choices=WEIGHTINGS, default=list(WEIGHTINGS),
                        help="count, probability or both (default: both, one pass)")
    parser.add_argument("--sequence", type=component, nargs="+")
    parser.add_argument("--methods", type=component, nargs="+")
    parser.add_argument("--data-root", type=Path,
                        help="Root of converted dataset/split/sequence GT (default: input.json "
                             "source of the predictions, else $EVTWEEN_DATA or LAB/evtween/data)")
    parser.add_argument("--pred-root", type=Path, default=benchmark.ROOT / "data_test_evt")
    parser.add_argument("--output", type=Path,
                        help="Run directory (default: MMD/runs/<dataset>-<split>-<settings hash>)")
    parser.add_argument("--time-window-s", type=float, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--allow-benchmark-crop", action="store_true")
    parser.add_argument("--algorithm", choices=("fast", "exact"), default="fast")
    parser.add_argument("--cutoff-sigma", type=float, default=10.0,
                        help="fast: skip pairs farther than this many sigmas on an axis")
    parser.add_argument("--device", default="cpu",
                        help="cpu (default), cuda (all visible GPUs) or cuda:N (one GPU)")
    parser.add_argument("--tile-px", type=int, help="Default 32 on cpu, 64 on cuda")
    parser.add_argument("--block-size", type=int, help="Default 512 on cpu, 4096 on cuda")
    parser.add_argument("--chunk-size", type=int, help="Default 4096 on cpu, 32768 on cuda")
    parser.add_argument("--jobs", type=int,
                        help="Sequences scored at once (default: CPUs/8 on cpu, GPUs on cuda)")
    parser.add_argument("--threads", type=int,
                        help="Threads per sequence (on cuda: CPU threads preparing tiles)")
    parser.add_argument("--progress-every", type=float, default=60.0, metavar="SECONDS",
                        help="Progress line inside each pair this often (0: off; default 60)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sources, predictions and pending pairs, then stop")
    args = parser.parse_args(argv)
    if args.cutoff_sigma < 3:
        parser.error("--cutoff-sigma must be at least 3")
    device = args.device
    if device != "cpu" and not (device == "cuda" or (device.startswith("cuda:")
                                                     and device[5:].isdigit())):
        parser.error("--device must be cpu, cuda or cuda:N")
    if device != "cpu" and args.algorithm == "exact":
        parser.error("--algorithm exact runs on the CPU only")
    for name in ("tile_px", "block_size", "chunk_size", "jobs", "threads"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    if args.time_window_s and not args.time_window_s[0] < args.time_window_s[1]:
        parser.error("--time-window-s needs START < STOP")
    args.weighting = list(dict.fromkeys(args.weighting))
    args.cpus = cpus
    return args


def settings_of(args, data_root):
    """Everything that changes a score. Device, tiles, blocks, jobs and threads do not."""
    return dict(
        dataset=args.dataset, split=args.split, sigma_spatial_px=args.sigma_spatial_px,
        sigma_temporal_ms=args.sigma_temporal_ms, weighting=args.weighting,
        algorithm=args.algorithm,
        cutoff_sigma=args.cutoff_sigma if args.algorithm == "fast" else None,
        time_window_s=args.time_window_s, allow_benchmark_crop=args.allow_benchmark_crop,
        data_root=str(data_root) if args.data_root else None,
        pred_root=str(args.pred_root.resolve()),
        metric_version=METRIC_VERSION,
    )


def main(argv=None):
    args = arguments(argv)
    data_root = args.data_root or benchmark.v2.old.setting(
        "EVTWEEN_DATA", benchmark.v2.old.LAB / "evtween/data")
    split_dir = data_root / args.dataset / args.split
    predictions = args.pred_root / args.dataset / args.split
    if not predictions.is_dir():
        raise SystemExit(f"Prediction directory not found: {predictions}")
    methods = list(dict.fromkeys(args.methods or sorted(
        p.name for p in predictions.iterdir() if p.is_dir() and not p.name.startswith("."))))
    if not methods:
        raise SystemExit(f"No generator directories in {predictions}")
    folders = [predictions / m for m in methods]
    names = set(args.sequence or ())
    if not names:
        names = {p.name for f in folders if f.is_dir() for p in f.iterdir() if p.is_dir()}
        if split_dir.is_dir():
            names |= {p.name for p in split_dir.iterdir() if p.is_dir()}
    if not names:
        raise SystemExit("No sequences found; check --data-root and --pred-root")

    settings = settings_of(args, data_root)
    tag = benchmark.digest(settings)[:12]
    output = args.output or HERE / "runs" / f"{args.dataset}-{args.split}-{tag}"
    rows_dir = output / "rows"
    config_file = output / "config.json"
    if config_file.is_file() and json.loads(config_file.read_text()) != settings:
        raise SystemExit(f"{output} holds a run with different settings; choose another --output")

    print(f"Predictions: {predictions}", flush=True)
    print(f"GT split   : {split_dir}{'' if args.data_root else ' (fallback; input.json sources win)'}",
          flush=True)
    tasks, pairs, skipped = [], [], 0
    for sequence in sorted(names):
        try:
            source = benchmark.source_for(sequence, split_dir, folders, args)
        except ValueError as exc:
            source, problem = None, str(exc)
        pending = []
        for method in methods:
            pairs.append((sequence, method))
            path = predictions / method / sequence / "events.npz"
            if source is None:
                write_json(row_file(rows_dir, sequence, method), dict(
                    dataset=args.dataset, split=args.split, sequence=sequence, generator=method,
                    status="alignment_unavailable", detail=problem, prediction=str(path)))
                continue
            inputs = fingerprint([path, path.with_name("input.json"), *gt_files(source)])
            if finished(rows_dir, sequence, method, inputs):
                skipped += 1
            else:
                pending.append((method, str(path), inputs))
        if source is not None:
            print(f"  {sequence}: GT {source}  ({len(pending)} pending)", flush=True)
        if pending:
            tasks.append((sequence, str(source), pending))

    gpus = visible_gpus(args.device)
    default_jobs = len(gpus) if gpus else args.cpus // 8 or 1
    jobs = max(1, min(args.jobs or default_jobs, len(tasks) or 1))
    threads = args.threads or max(1, args.cpus // jobs)
    if jobs * threads > args.cpus:
        print(f"Warning: {jobs} jobs x {threads} threads exceeds the {args.cpus} allocated CPUs",
              flush=True)
    if gpus and jobs > len(gpus):
        print(f"Warning: {jobs} jobs share {len(gpus)} GPU(s); each GPU will run several "
              "sequences at once", flush=True)
    where = f"GPUs: {', '.join(gpus)}" if gpus else f"{args.cpus} CPUs allocated"
    print(f"Output: {output}\n{skipped} finished pairs reused; {sum(len(t[2]) for t in tasks)} "
          f"pending in {len(tasks)} sequences; {jobs} jobs x {threads} threads ({where})",
          flush=True)
    if args.dry_run:
        return 0
    output.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(settings, indent=2))

    worker_settings = {k: v for k, v in vars(args).items()
                       if k in ("dataset", "split", "sigma_spatial_px", "sigma_temporal_ms",
                                "weighting", "algorithm", "cutoff_sigma", "tile_px", "block_size",
                                "chunk_size", "time_window_s", "allow_benchmark_crop",
                                "data_root", "pred_root", "progress_every")}
    statuses = []
    if jobs == 1:
        for task in tasks:
            statuses += score_sequence(*task, worker_settings, rows_dir, threads,
                                       gpus[0] if gpus else "cpu")
            export_csv(output, rows_dir, pairs)
    else:
        context = multiprocessing.get_context("spawn")
        free_gpus = None
        if gpus:
            manager = context.Manager()
            free_gpus = manager.Queue()
            for slot in range(jobs):
                free_gpus.put(gpus[slot % len(gpus)])
        pool = ProcessPoolExecutor(jobs, mp_context=context)
        try:
            # Longest GT first so the big sequences do not start last.
            tasks.sort(key=lambda t: -sum(os.path.getsize(f) for f in gt_files(Path(t[1]))[:3]
                                          if os.path.isfile(f)))
            futures = [pool.submit(score_sequence, *task, worker_settings, rows_dir, threads,
                                   "cpu", free_gpus) for task in tasks]
            for future in as_completed(futures):
                statuses += future.result()
                export_csv(output, rows_dir, pairs)
        except BaseException:
            for process in list((pool._processes or {}).values()):
                if process.is_alive():
                    process.terminate()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        finally:
            if gpus:
                manager.shutdown()
    rows = export_csv(output, rows_dir, pairs)
    print(f"Results: {output / 'results.csv'} ({len(rows)} rows)", flush=True)
    return int(any(s in ("error", "alignment_unavailable") for s in statuses))


if __name__ == "__main__":
    raise SystemExit(main())
