"""Generate a dataset split method by method, resuming v2 outputs."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

if __package__:
    from . import generate_event_streams_v2 as v2
else:
    import generate_event_streams_v2 as v2

old = v2.old


def learned_output(method, dataset, assets, run, work, generate):
    if method == "v2ce":
        from v2ce_interval_worker import load_generator
        (run / method).mkdir()
        if generate is None:
            generate = load_generator(assets)
        generate(run)
        raw = run / method / "events.npz"
    else:
        from evtween_worker import load_generator
        version, mode = old.EVTWEEN[method]
        seq_input = work / "sequence"
        seq_input.mkdir()
        for name in ("frames", "frame_ts.npy", "evaluation_meta.json"):
            (seq_input / name).symlink_to(run / "input" / name)
        if mode == "teacher":
            gt = run / "ground_truth.npz"
            if not gt.is_file():
                raise ValueError("Teacher requires recorded ground truth")
            (seq_input / "teacher_gt.npz").symlink_to(gt)
        raw = work / "raw.npz"
        if generate is None:
            if version == "v5.2":
                checkout = old.setting("EVTWEEN_V52_ROOT", old.LAB / "evtween_bnchmark-v5.2")
            else:
                checkout = old.setting("EVTWEEN_V53_ROOT", old.PROJECT)
            weights = old.setting("EVTWEEN_WEIGHTS", old.LAB / "evtween-tides-compare/benchmarks/event_generation/checkpoints/evtween_versions")
            ckpt = weights / version / ("model.pt" if mode == "teacher" else "student.pt")
            generate = load_generator(checkout, ckpt, dataset, 123, mode, raw)
        generate(seq_input, raw)
    return raw, generate


def execute_method(plan):
    method, dataset = plan["method"], plan["dataset"]
    cache, assets = Path(plan["cache"]), Path(plan["assets"])
    generate = None
    errors = []
    for job in plan["jobs"]:
        seq, target = Path(job["seq"]), Path(job["target"])
        prepared = cache / seq.name
        previous_dir = Path.cwd()
        start = time.monotonic()
        print(f"RUN  {seq.name}/{method}", flush=True)
        try:
            info = v2.sequence_info(seq)
            _, _, times, meta, _ = info
            with tempfile.TemporaryDirectory(dir=cache.parent) as tmp:
                work = Path(tmp)
                if not prepared.exists():
                    staging = work / "prepared"
                    v2.prepare(staging, seq, *info, job["need_gt"])
                    staging.rename(prepared)
                run = work / "run"
                run.mkdir()
                for item in prepared.iterdir():
                    (run / item.name).symlink_to(item)
                target.parent.mkdir(parents=True, exist_ok=True)
                log = target.parent / "run.log"
                if method in old.EVTWEEN or method == "v2ce":
                    with log.open("w", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                        try:
                            raw, generate = learned_output(method, dataset, assets, run, work, generate)
                        except Exception:
                            traceback.print_exc()
                            raise
                else:
                    raw = old.generate_one(method, dataset, run, assets, work, log)
                v2.publish(method, raw, target, meta, times)
                (work / "input.json").write_text(json.dumps(meta, indent=2))
                os.replace(work / "input.json", target.parent / "input.json")
            print(f"DONE {seq.name}/{method} ({time.monotonic()-start:.1f}s)", flush=True)
        except Exception as exc:
            errors.append(f"{seq.name}/{method}: {exc}")
            print(errors[-1], file=sys.stderr, flush=True)
        finally:
            os.chdir(previous_dir)
            if job["last"] and prepared.is_dir():
                shutil.rmtree(prepared)
    return errors


def generate_event_streams(dataset, split, *, methods=None, data_root=None):
    if not dataset or Path(dataset).name != dataset or dataset in (".", "..") or split not in ("train", "test"):
        raise ValueError("Expected a dataset name and train or test")
    methods = tuple(old.METHODS if methods is None else methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(old.METHODS):
        raise ValueError(f"Select distinct methods from {old.METHODS}")
    data = Path(data_root or old.setting("EVTWEEN_DATA", old.LAB / "evtween/data"))
    sequences = sorted(p for p in (data / dataset / split).iterdir() if p.is_dir())
    if not sequences:
        raise ValueError("No sequences found")
    dest = old.PROJECT / "data_test_evt" / dataset / split
    dest.mkdir(parents=True, exist_ok=True)
    assets = old.setting("BENCHMARK_INSTALL", old.LAB / "evtween/evtween-benchmark/benchmarks/event_generation")
    errors, results = [], []
    jobs = {method: [] for method in methods}
    with v2.split_lock(dest), tempfile.TemporaryDirectory(dir=dest) as tmp:
        root = Path(tmp).resolve()
        cache = root / "inputs"
        cache.mkdir()
        for seq in sequences:
            try:
                _, _, times, meta, _ = v2.sequence_info(seq)
                pending = {}
                for method in methods:
                    name = "voxels.npz" if method == "v2v" else "events.npz"
                    target = dest / method / seq.name / name
                    results.append(target)
                    if v2.complete(target, meta, times):
                        print(f"SKIP {seq.name}/{method}", flush=True)
                    else:
                        pending[method] = target
                need_gt = any(m.endswith("_teacher") for m in pending)
                for method, target in pending.items():
                    jobs[method].append(dict(seq=str(seq), target=str(target), need_gt=need_gt,
                                             last=method == next(reversed(pending))))
            except Exception as exc:
                errors.append(f"{seq.name}: {exc}")
                print(errors[-1], file=sys.stderr, flush=True)

        for method, tasks in jobs.items():
            if not tasks:
                continue
            plan = dict(method=method, dataset=dataset, jobs=tasks, cache=str(cache), assets=str(assets))
            if method not in old.EVTWEEN and method != "v2ce":
                errors.extend(execute_method(plan))
                continue
            try:
                if method == "v2ce":
                    python = json.loads((assets / "interpreters.json").read_text())[method]
                else:
                    python = old.setting("EVTWEEN_PYTHON", sys.executable)
                manifest = root / "plan.json"
                manifest.write_text(json.dumps(plan))
                result = subprocess.run([python, "-u", Path(__file__).resolve(), "--worker", manifest])
                if result.returncode:
                    errors.append(f"{method}: worker exited {result.returncode}; see sequence logs")
            except (OSError, KeyError, ValueError) as exc:
                errors.append(f"{method}: {exc}")
    if errors:
        raise RuntimeError("\n".join(errors))
    return results


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        sys.exit(bool(execute_method(json.loads(Path(sys.argv[2]).read_text()))))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("split", choices=("train", "test"))
    parser.add_argument("--methods", nargs="+", choices=old.METHODS)
    parser.add_argument("--data-root", type=Path)
    generate_event_streams(**vars(parser.parse_args()))
