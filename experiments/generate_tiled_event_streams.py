"""Generate an independent, full-sequence event stream for each 256x256 tile."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
import math
import os
from pathlib import Path
import sys
import tempfile
import threading

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.event_generation import generate_event_streams_v3_fast as fast  # noqa: E402

TILE_SIZE = 256
METHODS = tuple(method for method in fast.old.METHODS if method not in {"v2ce", "v2v"})


@dataclass(frozen=True)
class Tile:
    index: int
    row: int
    column: int
    x: int
    y: int

    @property
    def name(self):
        return f"tile_{self.index:04d}"

    @property
    def crop(self):
        return [self.x, self.y, TILE_SIZE, TILE_SIZE]


def tile_starts(length):
    last = length - TILE_SIZE
    if last < 0:
        raise ValueError(f"Image dimension {length} cannot fit a {TILE_SIZE}-pixel tile")
    starts = list(range(0, last + 1, TILE_SIZE))
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_layout(width, height):
    xs = tile_starts(width)
    return [Tile(row * len(xs) + col, row, col, x, y)
            for row, y in enumerate(tile_starts(height))
            for col, x in enumerate(xs)]


def crop_frames(frames, size, tile_dirs):
    """Decode one source PNG, then save all its requested lossless RGB tiles."""
    for index, path in frames:
        filename = f"{index:06d}.png"
        with Image.open(path) as image:
            if image.size != size:
                raise ValueError(f"Frame resolution changed in {path}: {image.size} != {size}")
            rgb = image.convert("RGB")
            for tile, folder in tile_dirs:
                x, y, w, h = tile.crop
                rgb.crop((x, y, x + w, y + h)).save(folder / filename, compress_level=1)


def prepare_ground_truth(sample, t0, t1, duration, tiles):
    errors = {}
    # One cut at a time per sequence to bound memory on dense event streams.
    for tile, path in tiles:
        try:
            fast.cut_ground_truth(sample, t0, t1, tile.crop, duration, path)
        except (OSError, ValueError) as exc:
            errors[tile.index] = f"{sample.name}/{tile.name}: ground truth: {exc}"
    return errors


def prepare_batch(batch, scratch, jobs):
    """Build the fast runner's job records, with crops shared by all methods."""
    prepared, errors = [], []
    prep_jobs = min(jobs, sum(len(files) + 1 for _, files, *_ in batch))
    with ProcessPoolExecutor(prep_jobs) as pool:
        for sample, files, size, times, t1, tiles in batch:
            tile_dirs, teacher_tiles, records = [], [], []
            for tile, meta, targets in tiles:
                folder = scratch / sample.name / tile.name
                frames = folder / "frames"
                frames.mkdir(parents=True)
                tile_dirs.append((tile, frames))
                gt_path = folder / "ground_truth.npz"
                if any(method.endswith("_teacher") for method, _ in targets):
                    teacher_tiles.append((tile, gt_path))
                common = dict(meta=meta, times=times, gt=None,
                              frames=[str(frames / f"{i:06d}.png") for i in range(len(files))])
                records.append((tile, common, targets, gt_path))
            indexed = list(enumerate(files))
            step = max(1, math.ceil(len(indexed) / (prep_jobs * 4)))
            crop_tasks = [pool.submit(crop_frames, indexed[start:start + step], size, tile_dirs)
                          for start in range(0, len(indexed), step)]
            gt_task = None
            if teacher_tiles:
                meta = tiles[0][1]
                gt_task = pool.submit(prepare_ground_truth, sample, meta["absolute_t0"],
                                      t1, times[-1], teacher_tiles)
            prepared.append((sample, records, crop_tasks, gt_task))

        by_method = {}
        for sample, records, crop_tasks, gt_task in prepared:
            failures = []
            for task in crop_tasks:
                try:
                    task.result()
                except (OSError, ValueError) as exc:
                    failures.append(str(exc))
            if failures:
                errors.append(f"{sample.name}: frame preparation failed: {failures[0]}")
            gt_errors = gt_task.result() if gt_task else {}
            errors.extend(gt_errors.values())
            if failures:
                continue
            for tile, common, targets, gt_path in records:
                for method, target in targets:
                    teacher = method.endswith("_teacher")
                    if teacher and tile.index in gt_errors:
                        continue
                    job = dict(common, target=str(target), gt=str(gt_path) if teacher else None)
                    by_method.setdefault(method, []).append(job)
    return by_method, errors


def run_batch(by_method, scratch, dataset, assets, jobs, gpu_jobs, gpu_methods, cpus):
    cpu_plans, gpu_plans = [], []
    for method, items in by_method.items():
        common = dict(method=method, dataset=dataset, assets=str(assets), scratch=str(scratch))
        if fast.persistent(method):
            shards = fast.balanced(items, min(gpu_jobs, len(items)))
            gpu_plans.extend(dict(common, jobs=shard) for shard in shards)
        else:
            plans = gpu_plans if method in gpu_methods else cpu_plans
            plans.extend(dict(common, jobs=[item]) for item in items)
    cpu_plans.sort(key=lambda plan: -len(plan["jobs"][0]["frames"]))
    slots = min(jobs, len(cpu_plans)) + min(gpu_jobs, len(gpu_plans))
    env = fast.thread_env(max(1, cpus // max(1, slots)))
    counter = dict(n=0, lock=threading.Lock(), stop=False, children=set())
    errors = []
    with ThreadPoolExecutor(jobs) as cpu_pool, ThreadPoolExecutor(gpu_jobs) as gpu_pool:
        futures = [gpu_pool.submit(fast.launch, plan, scratch, env, counter) for plan in gpu_plans]
        futures += [cpu_pool.submit(fast.launch, plan, scratch, env, counter) for plan in cpu_plans]
        try:
            for future in futures:
                errors.extend(future.result())
        except KeyboardInterrupt:
            with counter["lock"]:
                counter["stop"] = True
                children = list(counter["children"])
            for child in children:
                child.terminate()
            raise
    return errors


def path_component(name):
    if name in {"", ".", ".."} or Path(name).name != name:
        raise ValueError(f"Expected a single folder name, got {name!r}")
    return name


def pending_sequences(samples, methods, dest, verify_existing, errors):
    for sample in samples:
        try:
            files, stamps, times, base, size = fast.v2.sequence_info(sample)
            layout = tile_layout(*size)
        except (OSError, ValueError) as exc:
            errors.append(f"{sample.name}: {exc}")
            continue
        frame_times = times.tolist()
        base = dict(base, width=TILE_SIZE, height=TILE_SIZE, source_size=list(size),
                    frame_times_s=frame_times)
        pending, completed = [], 0
        with fast.skip_testzip(not verify_existing):
            for tile in layout:
                meta = dict(base, crop=tile.crop, tile_index=tile.index,
                            tile_row=tile.row, tile_column=tile.column)
                targets = []
                for method in methods:
                    target = dest / method / sample.name / tile.name / "events.npz"
                    if fast.v2.complete(target, meta, times):
                        completed += 1
                    else:
                        targets.append((method, target))
                if targets:
                    pending.append((tile, meta, targets))
        print(f"{sample.name}: {len(files)} frames, {len(layout)} tiles, "
              f"{completed}/{len(layout) * len(methods)} outputs already complete", flush=True)
        if pending:
            yield sample, files, size, frame_times, float(stamps[-1]), pending


def generate(dataset, split, *, data_root, sequences, methods, jobs, gpu_jobs, gpu_methods,
             sequences_per_batch, verify_existing, dry_run):
    data_root = (data_root or fast.old.setting("EVTWEEN_DATA", fast.old.LAB / "evtween/data")).resolve()
    source = data_root / path_component(dataset) / split
    samples = ([source / path_component(name) for name in sequences] if sequences is not None else
               sorted(path for path in source.iterdir() if path.is_dir()))
    if not samples or len(set(samples)) != len(samples) or any(not path.is_dir() for path in samples):
        raise ValueError(f"Select one or more distinct existing sequences under {source}")
    if len(set(methods)) != len(methods):
        raise ValueError("Select distinct methods")
    dest = ROOT / "data_test_exp" / f"{dataset}_{TILE_SIZE}" / split
    print(f"Output: {dest}", flush=True)
    if dry_run:
        total_tiles = 0
        for sample in samples:
            files, _, _, _, size = fast.v2.sequence_info(sample)
            count = len(tile_layout(*size))
            total_tiles += count
            print(f"{sample.name}: {size[0]}x{size[1]}, {len(files)} frames, {count} tiles")
        print(f"{total_tiles * len(methods)} planned event-stream files across {len(methods)} methods")
        return

    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    jobs = jobs or max(1, cpus - gpu_jobs)
    if "fracevent" in gpu_methods and "fracevent" in methods:
        os.environ.setdefault("FRACEVENT_DEVICE", "cuda")
    assets = fast.old.setting("BENCHMARK_INSTALL", fast.old.LAB / "evtween/evtween-benchmark/benchmarks/event_generation")
    dest.mkdir(parents=True, exist_ok=True)
    errors = []
    # Each fast worker locks its output directory and rechecks completeness
    # before writing, so runs on the same split can safely share the destination.
    plans = pending_sequences(samples, methods, dest, verify_existing, errors)
    while batch := list(islice(plans, sequences_per_batch)):
        # Publication uses atomic rename, so scratch lives on the output filesystem.
        with tempfile.TemporaryDirectory(prefix=".tiled-", dir=dest) as tmp:
            scratch = Path(tmp)
            by_method, prep_errors = prepare_batch(batch, scratch, jobs)
            errors.extend(prep_errors)
            errors.extend(run_batch(by_method, scratch, dataset, assets, jobs, gpu_jobs, gpu_methods, cpus))
    if errors:
        raise RuntimeError("\n".join(errors))


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("split", choices=("train", "test"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sequences", nargs="+", help="sequence folders; default: all in the split")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--jobs", type=positive_int, help="CPU workers (default: allocated CPUs minus GPU workers)")
    parser.add_argument("--gpu-jobs", type=positive_int, default=2, help="concurrent GPU workers (default: 2)")
    parser.add_argument("--gpu-methods", nargs="*", choices=METHODS, default=("fracevent",),
                        help="additional methods to schedule on GPU slots")
    parser.add_argument("--sequences-per-batch", type=positive_int, default=1,
                        help="stage this many pending sequences together to share model loads (default: 1)")
    parser.add_argument("--verify-existing", action="store_true", help="fully check saved NPZ archives before skipping")
    parser.add_argument("--dry-run", action="store_true", help="list coverage without generating or writing files")
    try:
        generate(**vars(parser.parse_args()))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
