"""Fast, parallel version of generate_event_streams_v3.py: same outputs, same paths, same resume rule.

Built on the scheduling of experiments/frame_windows_fast.py, for whole sequences. Only the
scheduling changes; every output is made by the same code as in v3 (v2.prepare's rules for frames
and ground truth, old.generate_one, v3.learned_output, v2.publish):

  * Existing outputs are checked in parallel and accepted without decompressing them (outputs
    are published atomically); --verify-existing restores v3's full check.
  * Frames of every pending sequence are cropped once, in parallel, and ground truth for teacher
    methods is cut once per sequence, in parallel, and kept uncompressed (an intermediate file).
  * Methods that load a model (evtween_*, v2ce) run in --gpu-jobs worker processes per method;
    each loads the model once and handles its share of the sequences (longest first, balanced
    by frame count). Every sequence restores the RNG state (rng_reset), so how the sequences are
    split across workers does not change the outputs.
  * In those workers, publishing (sorting + compressing) one sequence overlaps with generating
    the next, and Sequence.frame is cached so each frame is decoded once instead of once per pair
    that uses it as context (--no-frame-cache turns this off).
  * All other methods run one sequence per task, --jobs tasks at a time, longest sequences first;
    methods listed in --gpu-methods (fracevent by default) share the --gpu-jobs slots instead.
  * FracEvent runs on CUDA unless FRACEVENT_DEVICE is set (v3 defaults to cpu). Set
    FRACEVENT_DEVICE=cpu and leave fracevent out of --gpu-methods to keep v3's behaviour.
  * Each task gets OMP/MKL/torch threads = CPUs / concurrent tasks, so processes do not
    oversubscribe the CPU.

Example (32 CPUs, one A100):  python generate_event_streams_v3_fast.py DATASET test --gpu-jobs 4
"""
import argparse
from collections import OrderedDict
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
import time
import traceback
import zipfile

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import generate_event_streams_v3 as v3  # noqa: E402

v2 = v3.v2
old = v3.old
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def persistent(method):
    return method in old.EVTWEEN or method == "v2ce"


def interpreter(method, assets):
    if method == "v2ce":
        return json.loads((Path(assets) / "interpreters.json").read_text())[method]
    if method in old.EVTWEEN:
        return str(old.setting("EVTWEEN_PYTHON", sys.executable))
    return sys.executable


def output_name(method):
    return "voxels.npz" if method == "v2v" else "events.npz"


# ---------------------------------------------------------------- locks

@contextmanager
def split_access(dest):
    """Shared with other fast runners; excluded while v2/v3 hold their exclusive lock (and vice versa)."""
    import fcntl
    with (dest / ".generation-v2.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"A v2/v3 runner holds the exclusive lock on {dest}; "
                               "finish or stop it before starting the fast runner.") from exc
        yield


def acquire(folder):
    """Exclusive lock on one output folder; released by closing the returned file (any thread)."""
    import fcntl
    folder.mkdir(parents=True, exist_ok=True)
    stream = (folder / ".generation.lock").open("a")
    fcntl.flock(stream, fcntl.LOCK_EX)
    return stream


@contextmanager
def skip_testzip(enabled):
    """v2.complete without its last step (decompressing the whole archive)."""
    if not enabled:
        yield
        return
    original = zipfile.ZipFile.testzip
    zipfile.ZipFile.testzip = lambda self: None
    try:
        yield
    finally:
        zipfile.ZipFile.testzip = original


# ---------------------------------------------------------------- preparation (same results as v2.prepare)

def crop_frames(jobs, size, crop):
    """jobs: (source png, target png). v2.prepare's rule: link untouched frames, else crop + save."""
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
    """v2.prepare's ground truth (benchmark.save_events arrays), saved uncompressed."""
    x, y, w, h = crop
    seq = Path(seq)
    stamps = np.load(seq / "ev_t.npy", mmap_mode="r")
    a, b = np.searchsorted(stamps, [t0, t1])
    ex, ey, pol = (np.load(seq / f"ev_{k}.npy", mmap_mode="r")[a:b] for k in "xyp")
    mask = (ex >= x) & (ex < x + w) & (ey >= y) & (ey < y + h)
    t, ex, ey, p = map(np.asarray, (stamps[a:b][mask] - t0, ex[mask] - x, ey[mask] - y, pol[mask]))
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
    """The run folder v2.prepare would make, from the shared crops (each task gets its own folder)."""
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


# ---------------------------------------------------------------- model workers

def cache_frames(module, checkout, size=64):
    """Decode each frame once per sequence; callers get a copy, so the outputs cannot change."""
    Sequence = getattr(module, "Sequence", None)
    if Sequence is None or getattr(Sequence, "_fast_frame_cache", False):
        return False
    if Path(module.__file__).resolve().parent != Path(checkout).resolve():
        return False
    original = Sequence.frame

    def frame(self, i):
        store = self.__dict__.setdefault("_fast_frames", OrderedDict())
        key = int(i)
        if key in store:
            store.move_to_end(key)
        else:
            store[key] = original(self, i)
            if len(store) > size:
                store.popitem(last=False)
        value = store[key]
        return value.clone() if hasattr(value, "clone") else value.copy()

    Sequence.frame = frame
    Sequence._fast_frame_cache = True
    return True


def load_learned(method, dataset, assets, first_output, frame_cache):
    """The generator v3.learned_output would load, plus the optional frame cache."""
    if method == "v2ce":
        from v2ce_interval_worker import load_generator
        return load_generator(Path(assets))
    from evtween_worker import load_generator
    version, mode = old.EVTWEEN[method]
    checkout = old.evtween_checkout(version)
    weights = old.setting("EVTWEEN_WEIGHTS", old.LAB / "evtween-tides-compare/benchmarks/event_generation/checkpoints/evtween_versions")
    ckpt = weights / version / ("model.pt" if mode == "teacher" else "student.pt")
    generate = load_generator(checkout, ckpt, dataset, 123, mode, first_output)
    if frame_cache and cache_frames(sys.modules.get("data"), checkout):
        print("frame cache on", flush=True)
    return generate


def publish_and_release(method, raw, target, meta, times, tmp, lock, start, console):
    """v3's publication step; returns an error message or None. Always frees the folder and lock."""
    try:
        v2.publish(method, raw, target, meta, times)
        record = tmp / "input.json"
        record.write_text(json.dumps(meta, indent=2))
        os.replace(record, target.parent / "input.json")
        print(f"DONE {target} ({time.monotonic() - start:.1f}s)", file=console, flush=True)
        return None
    except Exception as exc:
        return f"{target}: {exc}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        lock.close()


def worker(plan_path, report_path):
    plan = json.loads(Path(plan_path).read_text())
    method, dataset = plan["method"], plan["dataset"]
    assets, scratch = Path(plan["assets"]), Path(plan["scratch"])
    model = persistent(method)
    threads = os.environ.get("OMP_NUM_THREADS")
    if threads and model:
        try:
            import torch
            torch.set_num_threads(int(threads))
        except ImportError:
            pass
    console, errors_out = sys.stdout, sys.stderr
    errors, generate, previous = [], None, None
    publisher = ThreadPoolExecutor(1) if model else None

    def collect(future):
        message = future.result()
        if message:
            errors.append(message)
            print(message, file=errors_out, flush=True)

    for job in plan["jobs"]:
        target = Path(job["target"])
        meta, times = job["meta"], np.asarray(job["times"], dtype=np.float64)
        lock, tmp, handed = None, None, False
        cwd = Path.cwd()
        try:
            lock = acquire(target.parent)
            if v2.complete(target, meta, times):  # another runner may have finished it meanwhile
                print(f"SKIP {target}", file=console, flush=True)
                continue
            tmp = Path(tempfile.mkdtemp(dir=scratch))
            run = tmp / "run"
            build_run(run, job)
            print(f"RUN  {target}", file=console, flush=True)
            start = time.monotonic()
            log = target.parent / "run.log"
            if model:
                with log.open("w", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                    try:
                        if generate is None:
                            generate = load_learned(method, dataset, assets,
                                                    scratch / f"{Path(plan_path).stem}-load.npz",
                                                    plan.get("frame_cache", True))
                        raw, generate = v3.learned_output(method, dataset, assets, run, tmp, generate)
                    except Exception:
                        traceback.print_exc()
                        raise
                    finally:
                        os.chdir(cwd)
                if previous is not None:  # at most one publication in flight
                    collect(previous)
                previous = publisher.submit(publish_and_release, method, raw, target, meta, times,
                                            tmp, lock, start, console)
                handed = True
            else:
                raw = old.generate_one(method, dataset, run, assets, tmp, log)
                handed = True
                message = publish_and_release(method, raw, target, meta, times, tmp, lock, start, console)
                if message:
                    errors.append(message)
                    print(message, file=errors_out, flush=True)
        except Exception as exc:
            errors.append(f"{target}: {exc}")
            print(errors[-1], file=errors_out, flush=True)
        finally:
            os.chdir(cwd)
            if not handed:
                if tmp is not None:
                    shutil.rmtree(tmp, ignore_errors=True)
                if lock is not None:
                    lock.close()
    if previous is not None:
        collect(previous)
    if publisher:
        publisher.shutdown()
    Path(report_path).write_text(json.dumps(errors))


# ---------------------------------------------------------------- scheduling (parent side)

def launch(plan, scratch, env, counter):
    """Run one worker process on a list of sequences; return its errors."""
    with counter["lock"]:
        counter["n"] += 1
        name = f"{plan['method']}-{counter['n']}"
    plan_path, report = scratch / f"plan-{name}.json", scratch / f"report-{name}.json"
    plan_path.write_text(json.dumps(plan))
    try:
        python = interpreter(plan["method"], plan["assets"])
        result = subprocess.run([python, "-u", str(Path(__file__).resolve()), "--worker",
                                 str(plan_path), str(report)], env=env)
    except (OSError, KeyError, ValueError) as exc:
        return [f"{name}: {exc}"]
    errors = json.loads(report.read_text()) if report.is_file() else []
    if result.returncode:
        errors.append(f"{name}: worker exited {result.returncode}; see the sequence logs")
    return errors


def thread_env(threads):
    env = dict(os.environ)
    for key in THREAD_VARS:
        env[key] = str(threads)
    return env


def balanced(jobs_list, shards):
    """Longest-first assignment to the least loaded shard (load = frames)."""
    loads = [[0, []] for _ in range(shards)]
    for job in sorted(jobs_list, key=lambda j: -len(j["frames"])):
        slot = min(loads, key=lambda s: s[0])
        slot[0] += len(job["frames"])
        slot[1].append(job)
    return [jobs for _, jobs in loads if jobs]


def generate_event_streams(dataset, split, *, methods=None, data_root=None, jobs=None, gpu_jobs=4,
                           gpu_methods=("fracevent",), verify_existing=False, frame_cache=True):
    """Same arguments and return value as v3.generate_event_streams, plus the scheduling options."""
    if not dataset or Path(dataset).name != dataset or dataset in (".", "..") or split not in ("train", "test"):
        raise ValueError("Expected a dataset name and train or test")
    methods = tuple(old.METHODS if methods is None else methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(old.METHODS):
        raise ValueError(f"Select distinct methods from {old.METHODS}")
    gpu_methods = tuple(gpu_methods or ())
    # CPUs this process may use (a Slurm/taskset allocation), not every core of the machine
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    gpu_jobs = max(1, int(gpu_jobs))
    jobs = max(1, int(jobs or cpus - gpu_jobs))
    if "fracevent" in gpu_methods:
        os.environ.setdefault("FRACEVENT_DEVICE", "cuda")
    data = Path(data_root or old.setting("EVTWEEN_DATA", old.LAB / "evtween/data"))
    sequences = sorted(p for p in (data / dataset / split).iterdir() if p.is_dir())
    if not sequences:
        raise ValueError("No sequences found")
    dest = old.PROJECT / "data_test_evt" / dataset / split
    dest.mkdir(parents=True, exist_ok=True)
    assets = old.setting("BENCHMARK_INSTALL", old.LAB / "evtween/evtween-benchmark/benchmarks/event_generation")

    # 1. every (method, sequence) and whether it is already done
    errors, results, infos, candidates = [], [], {}, []
    for seq in sequences:
        try:
            files, ts, times, meta, size = v2.sequence_info(seq)
        except Exception as exc:
            errors.append(f"{seq.name}: {exc}")
            print(errors[-1], file=sys.stderr, flush=True)
            continue
        infos[seq] = dict(files=files, ts=ts, times=times, meta=meta, size=size)
        for method in methods:
            target = dest / method / seq.name / output_name(method)
            results.append(target)
            candidates.append((method, seq, target))

    def done(item):
        method, seq, target = item
        return v2.complete(target, infos[seq]["meta"], infos[seq]["times"])

    with skip_testzip(not verify_existing), ThreadPoolExecutor(min(32, cpus * 2)) as pool:
        flags = list(pool.map(done, candidates))
    pending = {method: [] for method in methods}
    for (method, seq, target), flag in zip(candidates, flags):
        if flag:
            print(f"SKIP {seq.name}/{method}", flush=True)
        else:
            pending[method].append((seq, target))
    todo = sum(map(len, pending.values()))
    print(f"{len(candidates) - todo} outputs already complete, {todo} to generate", flush=True)
    if not todo:
        if errors:
            raise RuntimeError("\n".join(errors))
        return results

    with split_access(dest), tempfile.TemporaryDirectory(dir=dest) as tmp:
        scratch = Path(tmp).resolve()
        # 2. crop each pending sequence's frames once, and cut ground truth once for teacher methods
        needed, gt_seqs = set(), set()
        for method, items in pending.items():
            for seq, _ in items:
                needed.add(seq)
                if method.endswith("_teacher") and (seq / "ev_t.npy").is_file():
                    gt_seqs.add(seq)
        (scratch / "frames").mkdir()
        (scratch / "gt").mkdir()
        failed = {}
        with ProcessPoolExecutor(min(jobs, cpus)) as pool:
            crop_futures = []
            for seq in sorted(needed):
                info = infos[seq]
                folder = scratch / "frames" / seq.name
                folder.mkdir()
                batch = [(str(f), str(folder / f"{i:06d}.png")) for i, f in enumerate(info["files"])]
                step = max(1, math.ceil(len(batch) / 16))
                crop_futures += [(seq, pool.submit(crop_frames, batch[s:s + step], info["size"], info["meta"]["crop"]))
                                 for s in range(0, len(batch), step)]
            gt_futures = [(seq, pool.submit(cut_ground_truth, str(seq), float(infos[seq]["ts"][0]),
                                            float(infos[seq]["ts"][-1]), infos[seq]["meta"]["crop"],
                                            float(infos[seq]["times"][-1]), str(scratch / "gt" / f"{seq.name}.npz")))
                          for seq in sorted(gt_seqs)]
            for seq, future in crop_futures + gt_futures:
                try:
                    future.result()
                except Exception as exc:
                    failed.setdefault(seq, f"{seq.name}: input preparation failed: {exc}")
        for message in failed.values():
            errors.append(message)
            print(message, file=sys.stderr, flush=True)
        print(f"Prepared {len(needed) - len(failed)} sequences "
              f"({sum(len(infos[s]['files']) for s in needed)} frames, {len(gt_seqs)} with ground truth)", flush=True)

        def job(seq, target):
            info = infos[seq]
            gt = scratch / "gt" / f"{seq.name}.npz"
            return dict(target=str(target), meta=info["meta"], times=info["times"].tolist(),
                        gt=str(gt) if gt.is_file() else None,
                        frames=[str(scratch / "frames" / seq.name / f"{i:06d}.png") for i in range(len(info["files"]))])

        # 3. run: model methods as --gpu-jobs long workers each; the rest one sequence per task
        cpu_tasks, gpu_tasks = [], []
        for method, items in pending.items():
            jobs_list = [job(seq, target) for seq, target in items if seq not in failed]
            if not jobs_list:
                continue
            plan = dict(method=method, dataset=dataset, assets=str(assets), scratch=str(scratch),
                        frame_cache=frame_cache)
            if persistent(method):
                gpu_tasks += [dict(plan, jobs=shard) for shard in balanced(jobs_list, min(gpu_jobs, len(jobs_list)))]
            elif method in gpu_methods:
                gpu_tasks += [dict(plan, jobs=[j]) for j in jobs_list]
            else:
                cpu_tasks += [dict(plan, jobs=[j]) for j in jobs_list]
        cpu_tasks.sort(key=lambda p: -len(p["jobs"][0]["frames"]))
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
    parser.add_argument("split", choices=("train", "test"))
    parser.add_argument("--methods", nargs="+", choices=old.METHODS)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--jobs", type=int, help="concurrent CPU tasks (default: allocated CPUs - gpu-jobs)")
    parser.add_argument("--gpu-jobs", type=int, default=4,
                        help="concurrent GPU processes: workers per model method, and slots for --gpu-methods")
    parser.add_argument("--gpu-methods", nargs="*", default=["fracevent"],
                        help="per-sequence methods that use the GPU and share its slots (e.g. add v2e if yours runs on CUDA)")
    parser.add_argument("--verify-existing", action="store_true",
                        help="fully decompress existing outputs before skipping them (slow; v3 behaviour)")
    parser.add_argument("--no-frame-cache", dest="frame_cache", action="store_false",
                        help="decode frames on every use in Evtween workers, exactly as generate.py does")
    generate_event_streams(**vars(parser.parse_args()))
