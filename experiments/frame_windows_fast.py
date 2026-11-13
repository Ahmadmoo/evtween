"""Fast, parallel version of frame_windows.py: same windows, same output paths and files.

Differences from frame_windows.py (outputs are meant to be the same; only scheduling changes):
  * Every frame a window needs is cropped once per sequence (in parallel), not once per window.
  * Ground-truth events for teacher methods are cut once per window, in parallel, and kept uncompressed.
  * Methods that load a model (evtween_*, v2ce) run in --gpu-jobs worker processes per method; each loads
    the model once and handles its share of the windows. Every window resets the RNG (rng_reset), so the
    split across workers does not change the outputs.
  * All other methods run one window per task, --jobs tasks at a time; methods listed in --gpu-methods
    (fracevent by default) share the --gpu-jobs slots instead.
  * FracEvent runs on CUDA unless FRACEVENT_DEVICE is set (frame_windows.py defaults to cpu).
  * Existing outputs are accepted without decompressing them (outputs are published atomically);
    --verify-existing restores the full check.
  * Each task gets OMP/MKL/torch threads = CPUs / concurrent tasks, so processes do not oversubscribe the CPU.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import frame_windows as fw  # noqa: E402  (also puts benchmarks/event_generation on sys.path)

v3, runner = fw.v3, fw.runner
old = runner.old
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def persistent(method):
    return method in old.EVTWEEN or method == "v2ce"


def interpreter(method, assets):
    if method == "v2ce":
        return json.loads((Path(assets) / "interpreters.json").read_text())[method]
    if method in old.EVTWEEN:
        return str(old.setting("EVTWEEN_PYTHON", sys.executable))
    return sys.executable


@contextmanager
def skip_testzip(enabled):
    """runner.complete without its last step (decompressing the whole archive)."""
    if not enabled:
        yield
        return
    original = zipfile.ZipFile.testzip
    zipfile.ZipFile.testzip = lambda self: None
    try:
        yield
    finally:
        zipfile.ZipFile.testzip = original


# ---------------------------------------------------------------- preparation (same results as runner.prepare)

def crop_frames(jobs, size, crop):
    """jobs: (source png, target png). Same rule as runner.prepare: link untouched frames, else crop + save."""
    x, y, w, h = crop
    for source, target in jobs:
        source, target = Path(source), Path(target)
        with Image.open(source) as im:
            if tuple(im.size) != tuple(size):
                raise ValueError(f"Frame resolution changed: {source}")
            if im.mode == "RGB" and not im.info and im.size == (w, h):
                target.symlink_to(source.resolve())
            else:
                staging = target.with_name(target.name + ".tmp.png")
                im.convert("RGB").crop((x, y, x + w, y + h)).save(staging, compress_level=1)
                os.replace(staging, target)


def cut_ground_truth(seq, t0, t1, crop, duration, out):
    """runner.prepare's ground truth for one window, saved uncompressed (an intermediate file)."""
    x, y, w, h = crop
    seq = Path(seq)
    stamps = np.load(seq / "ev_t.npy", mmap_mode="r")
    a, b = np.searchsorted(stamps, [t0, t1])
    ex, ey, pol = (np.load(seq / f"ev_{k}.npy", mmap_mode="r")[a:b] for k in "xyp")
    mask = (ex >= x) & (ex < x + w) & (ey >= y) & (ey < y + h)
    t, ex, ey, p = stamps[a:b][mask] - t0, ex[mask] - x, ey[mask] - y, pol[mask]
    # benchmark.save_events, without compression
    t, ex, ey, p = map(np.asarray, (t, ex, ey, p))
    if not np.all(np.isfinite(t)):
        raise ValueError("Malformed events")
    if not np.all(np.isin(p, [-1, 0, 1])):
        raise ValueError("Unknown polarity encoding")
    keep = (t >= 0) & (t < duration) & (ex >= 0) & (ex < w) & (ey >= 0) & (ey < h)
    order = np.argsort(t[keep], kind="stable")
    staging = Path(out).with_name(Path(out).name + ".tmp.npz")
    np.savez(staging, t=t[keep][order].astype("float64"), x=ex[keep][order].astype("int32"),
             y=ey[keep][order].astype("int32"), p=np.where(p[keep][order] > 0, 1, -1).astype("int8"))
    os.replace(staging, out)


def build_run(run, job):
    """The run folder runner.prepare would make, from the shared crops (each task gets its own folder)."""
    inputs = run / "input"
    frames = inputs / "frames"
    frames.mkdir(parents=True)
    for n, path in enumerate(job["frames"]):
        (frames / f"{n:06d}.png").symlink_to(Path(path).resolve())
    np.save(inputs / "frame_ts.npy", np.asarray(job["times"], dtype=np.float64))
    (run / "input.json").write_text(json.dumps(job["meta"]))
    (inputs / "evaluation_meta.json").write_text(json.dumps(
        {"evaluation_window_in_run_s": [0, job["meta"]["duration"]]}))
    if job.get("gt"):
        (run / "ground_truth.npz").symlink_to(job["gt"])


# ---------------------------------------------------------------- one window (worker side)

def run_job(job, plan, generate):
    method, dataset = plan["method"], plan["dataset"]
    assets, scratch = Path(plan["assets"]), Path(plan["scratch"])
    target = Path(job["target"])
    meta, times = job["meta"], np.asarray(job["times"], dtype=np.float64)
    with fw.output_lock(target.parent):
        if runner.complete(target, meta, times):  # another runner may have finished it meanwhile
            print(f"SKIP {target}", flush=True)
            return generate
        previous_dir = Path.cwd()
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            work = Path(tmp)
            run = work / "run"
            build_run(run, job)
            print(f"RUN  {target}", flush=True)
            log = target.parent / "run.log"
            if persistent(method):
                with log.open("w", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                    try:
                        raw, generate = v3.learned_output(method, dataset, assets, run, work, generate)
                    except Exception:
                        traceback.print_exc()
                        raise
                    finally:
                        os.chdir(previous_dir)
            else:
                raw = old.generate_one(method, dataset, run, assets, work, log)
            runner.publish(method, raw, target, meta, times)
            record = work / "input.json"
            record.write_text(json.dumps(meta, indent=2))
            os.replace(record, target.parent / "input.json")
    return generate


def worker(plan_path, report_path):
    plan = json.loads(Path(plan_path).read_text())
    threads = os.environ.get("OMP_NUM_THREADS")
    if threads and persistent(plan["method"]):
        try:
            import torch
            torch.set_num_threads(int(threads))
        except ImportError:
            pass
    generate, errors = None, []
    for job in plan["jobs"]:
        try:
            generate = run_job(job, plan, generate)
        except Exception as exc:
            errors.append(f"{job['target']}: {exc}")
            print(errors[-1], file=sys.stderr, flush=True)
    Path(report_path).write_text(json.dumps(errors))


# ---------------------------------------------------------------- scheduling (parent side)

def launch(plan, scratch, env, counter):
    """Run one worker process on a list of windows; return its errors."""
    with counter["lock"]:
        counter["n"] += 1
        name = f"{plan['method']}-{counter['n']}"
    plan_path, report = scratch / f"plan-{name}.json", scratch / f"report-{name}.json"
    plan_path.write_text(json.dumps(plan))
    python = interpreter(plan["method"], plan["assets"])
    result = subprocess.run([python, "-u", str(Path(__file__).resolve()), "--worker", str(plan_path), str(report)],
                            env=env)
    errors = json.loads(report.read_text()) if report.is_file() else []
    if result.returncode:
        errors.append(f"{name}: worker exited {result.returncode}")
    return errors


def thread_env(threads):
    env = dict(os.environ)
    for key in THREAD_VARS:
        env[key] = str(threads)
    return env


def run_experiment(dataset, *, k, start_stride, methods, experiment_name, sequences=None, split="test",
                   data_root=None, m=0, jobs=None, gpu_jobs=4, gpu_methods=("fracevent",),
                   verify_existing=False):
    """Same arguments and return value as frame_windows.run_experiment, plus the scheduling options."""
    fw.path_component(dataset)
    fw.path_component(experiment_name)
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(old.METHODS):
        raise ValueError(f"Select distinct methods from {old.METHODS}")
    cpus = os.cpu_count() or 1
    gpu_jobs = max(1, int(gpu_jobs))
    jobs = max(1, int(jobs or cpus - gpu_jobs))
    os.environ.setdefault("FRACEVENT_DEVICE", "cuda")
    data = Path(data_root or old.setting("EVTWEEN_DATA", old.LAB / "evtween/data")).resolve()
    source = data / dataset / split
    samples = (sorted(p for p in source.iterdir() if p.is_dir()) if sequences is None
               else [source / fw.path_component(name) for name in sequences])
    if not samples or len(set(samples)) != len(samples):
        raise ValueError("Select one or more distinct sequences")
    dest = ROOT / "data_test_exp" / dataset
    dest.mkdir(parents=True, exist_ok=True)
    assets = old.setting("BENCHMARK_INSTALL", old.LAB / "evtween/evtween-benchmark/benchmarks/event_generation")

    # 1. every (method, window) and whether it is already done
    results, pending, windows = [], {method: [] for method in methods}, []
    for sample in samples:
        files, stamps, _, base, size = runner.sequence_info(sample)
        if len(files) < 1 + (k - 1) * (m + 1):
            print(f"SKIP {sample.name}: only {len(files)} frames for k={k}, m={m}", flush=True)
        for window in fw.frame_windows(len(files), k, start_stride, m):
            first, last = window.start, window.stop - 1
            ts = stamps[window]
            times = ts - ts[0]
            dt = np.diff(times)
            meta = dict(base, split=split, indices=list(range(first, last + 1, window.step)),
                        absolute_t0=float(ts[0]), duration=float(times[-1]),
                        frame_times_s=times.tolist(),
                        uniform_timestamps=bool(np.allclose(dt, dt.mean(), rtol=1e-3, atol=1e-7)))
            sub = Path(experiment_name) / f"k_{k}"
            if m:
                meta["m"] = m
                sub = sub / f"m_{m}"
            sub = sub / f"frames_{first}_{last}"
            windows.append(dict(sample=sample, files=files, size=size, ts=ts, times=times, meta=meta, sub=sub))
            for method in methods:
                folder = dest / method / sample.name / sub
                target = folder / ("voxels.npz" if method == "v2v" else "events.npz")
                metadata = folder / "input.json"
                if metadata.exists() and json.loads(metadata.read_text()).get("source") != str(sample):
                    raise ValueError(f"{folder} belongs to another source. Choose another experiment name.")
                results.append(target)
                pending[method].append((len(windows) - 1, target))

    def done(item):
        index, target = item
        w = windows[index]
        return runner.complete(target, w["meta"], w["times"])

    with skip_testzip(not verify_existing), ThreadPoolExecutor(min(32, cpus * 2)) as pool:
        for method in methods:
            flags = list(pool.map(done, pending[method]))
            for (index, target), flag in zip(pending[method], flags):
                if flag:
                    print(f"SKIP {target.relative_to(dest)}", flush=True)
            pending[method] = [item for item, flag in zip(pending[method], flags) if not flag]
    todo = sum(map(len, pending.values()))
    print(f"{len(results) - todo} outputs already complete, {todo} to generate", flush=True)
    if not todo:
        return results

    errors = []
    with fw.dataset_access(dest), tempfile.TemporaryDirectory(dir=dest) as tmp:
        scratch = Path(tmp)
        # 2. crop each needed frame once, and cut ground truth once per window for teacher methods
        needed = {}
        gt_windows = set()
        for method, items in pending.items():
            for index, _ in items:
                w = windows[index]
                needed.setdefault(w["sample"], set()).update(w["meta"]["indices"])
                if method.endswith("_teacher") and (w["sample"] / "ev_t.npy").is_file():
                    gt_windows.add(index)
        crops, crop_jobs = {}, []
        for index, w in enumerate(windows):
            crops.setdefault(w["sample"], (w["files"], w["size"], w["meta"]["crop"]))
        for sample, indices in needed.items():
            files, size, crop = crops[sample]
            folder = scratch / "frames" / sample.name
            folder.mkdir(parents=True)
            batch = [(str(files[i]), str(folder / f"{i:06d}.png")) for i in sorted(indices)]
            step = max(1, math.ceil(len(batch) / (jobs * 4)))
            crop_jobs += [(batch[s:s + step], size, crop) for s in range(0, len(batch), step)]
        (scratch / "gt").mkdir()
        with ProcessPoolExecutor(jobs) as pool:
            futures = [pool.submit(crop_frames, *job) for job in crop_jobs]
            gt_futures = {}
            for index in sorted(gt_windows):
                w = windows[index]
                out = scratch / "gt" / f"{index}.npz"
                gt_futures[index] = pool.submit(cut_ground_truth, str(w["sample"]), float(w["ts"][0]),
                                                float(w["ts"][-1]), w["meta"]["crop"], float(w["times"][-1]), str(out))
            for future in futures:
                future.result()
            for index, future in gt_futures.items():
                try:
                    future.result()
                except Exception as exc:
                    errors.append(f"ground truth for {windows[index]['sample'].name}/{windows[index]['sub']}: {exc}")
        print(f"Prepared {sum(map(len, needed.values()))} frames and {len(gt_windows)} ground-truth windows", flush=True)

        def job(index, target):
            w = windows[index]
            gt = scratch / "gt" / f"{index}.npz"
            return dict(target=str(target), meta=w["meta"], times=w["times"].tolist(), gt=str(gt) if gt.is_file() else None,
                        frames=[str(scratch / "frames" / w["sample"].name / f"{i:06d}.png") for i in w["meta"]["indices"]])

        # 3. run: model methods as --gpu-jobs long workers each; the rest one window per task
        cpu_tasks, gpu_tasks = [], []
        for method, items in pending.items():
            if not items:
                continue
            plan = dict(method=method, dataset=dataset, assets=str(assets), scratch=str(scratch))
            jobs_list = [job(*item) for item in items]
            if persistent(method):
                shards = min(gpu_jobs, len(jobs_list))
                gpu_tasks += [dict(plan, jobs=jobs_list[s::shards]) for s in range(shards)]
            elif method in gpu_methods:
                gpu_tasks += [dict(plan, jobs=[j]) for j in jobs_list]
            else:
                cpu_tasks += [dict(plan, jobs=[j]) for j in jobs_list]
        threads = max(1, cpus // (jobs + gpu_jobs))
        env = thread_env(threads)
        counter = dict(n=0, lock=threading.Lock())
        print(f"Running {len(cpu_tasks)} CPU tasks ({jobs} at a time) and {len(gpu_tasks)} GPU tasks "
              f"({gpu_jobs} at a time), {threads} threads each", flush=True)
        with ThreadPoolExecutor(jobs) as cpu_pool, ThreadPoolExecutor(gpu_jobs) as gpu_pool:
            futures = ([gpu_pool.submit(launch, plan, scratch, env, counter) for plan in gpu_tasks] +
                       [cpu_pool.submit(launch, plan, scratch, env, counter) for plan in cpu_tasks])
            for future in futures:
                errors.extend(future.result())
    if errors:
        raise RuntimeError("\n".join(errors))
    return results


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
        sys.exit(0)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, default=0, help="Source frames skipped between selected frames")
    parser.add_argument("--start-stride", type=int, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--methods", nargs="+", choices=old.METHODS, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--jobs", type=int, help="concurrent CPU tasks (default: CPUs - gpu-jobs)")
    parser.add_argument("--gpu-jobs", type=int, default=4,
                        help="concurrent GPU processes: workers per model method, and slots for --gpu-methods")
    parser.add_argument("--gpu-methods", nargs="*", default=["fracevent"],
                        help="per-window methods that use the GPU and share its slots (e.g. add v2e if yours runs on CUDA)")
    parser.add_argument("--verify-existing", action="store_true",
                        help="fully decompress existing outputs before skipping them (slow; frame_windows.py behaviour)")
    run_experiment(**vars(parser.parse_args()))
