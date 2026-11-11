"""Generate benchmark outputs independently on regularly spaced frame windows."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import subprocess
import traceback
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmarks' / 'event_generation'))
import generate_event_streams_v3 as v3

runner = v3.v2


def path_component(name):
    if not isinstance(name, str) or not name.strip() or name in {".", ".."} or any(
        char in name for char in ("/", "\\", "\x00")
    ):
        raise ValueError(f"Expected a single path component, got {name!r}")
    return name


def frame_windows(n_frames, k, start_stride):
    """Yield slices of k consecutive frames, omitting an incomplete tail."""
    if k < 2 or start_stride < 1:
        raise ValueError("k must be at least 2 and start_stride must be positive")
    return (slice(s, s + k) for s in range(0, n_frames - k + 1, start_stride))


def run_experiment(dataset, *, k, start_stride, methods, experiment_name,
                   sequences=None, split="test", data_root=None):
    """Return one path per selected method and complete window, including resumed outputs."""
    path_component(dataset)
    path_component(experiment_name)
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    methods = tuple(methods)
    old = runner.old
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(old.METHODS):
        raise ValueError(f"Select distinct methods from {old.METHODS}")
    data = Path(data_root or old.setting("EVTWEEN_DATA", old.LAB / "evtween/data")).resolve()
    source = data / dataset / split
    samples = (sorted(p for p in source.iterdir() if p.is_dir()) if sequences is None
               else [source / path_component(name) for name in sequences])
    if not samples or len(set(samples)) != len(samples):
        raise ValueError("Select one or more distinct sequences")
    dest = ROOT / "data_test_exp" / dataset
    dest.mkdir(parents=True, exist_ok=True)
    assets = old.setting("BENCHMARK_INSTALL", old.LAB / "evtween/evtween-benchmark/benchmarks/event_generation")
    results, errors = [], []
    with runner.split_lock(dest), tempfile.TemporaryDirectory(dir=dest) as tmp:
        cache = Path(tmp) / "inputs" if len(methods) > 1 else None
        if cache:
            cache.mkdir()
        for method in methods:
            options = dict(dataset=dataset, split=split, k=k, start_stride=start_stride,
                           samples=[str(p) for p in samples], method=method,
                           experiment_name=experiment_name, assets=str(assets), dest=str(dest),
                           cache=str(cache) if cache else None, last_method=method == methods[-1],
                           need_gt=any(m.endswith('_teacher') for m in methods))
            if method in old.EVTWEEN or method == "v2ce":
                plan = Path(tmp) / "plan.json"
                report = Path(tmp) / "results.json"
                report.unlink(missing_ok=True)
                plan.write_text(json.dumps(options))
                python = (json.loads((assets / "interpreters.json").read_text())[method]
                          if method == "v2ce" else old.setting("EVTWEEN_PYTHON", sys.executable))
                result = subprocess.run([python, "-u", Path(__file__).resolve(), "--worker", plan, report])
                if result.returncode:
                    errors.append(f"{method}: worker exited {result.returncode}")
                    continue
                paths, failures = json.loads(report.read_text())
            else:
                paths, failures = run_method(**options)
            results.extend(Path(p) for p in paths)
            errors.extend(failures)
    if errors:
        raise RuntimeError("\n".join(errors))
    return results


def run_method(dataset, split, k, start_stride, samples, method, experiment_name, assets, dest, cache=None,
               last_method=True, need_gt=False):
    dest, assets = Path(dest), Path(assets)
    generate = None
    results, errors = [], []
    for sample in map(Path, samples):
        files, stamps, _, base, size = runner.sequence_info(sample)
        windows = frame_windows(len(files), k, start_stride)
        if len(files) < k:
            print(f"SKIP {sample.name}: only {len(files)} frames for k={k}", flush=True)
        for window in windows:
            first, last = window.start, window.stop - 1
            ts = stamps[window]
            times = ts - ts[0]
            dt = np.diff(times)
            meta = dict(base, split=split, indices=list(range(first, last + 1)),
                        absolute_t0=float(ts[0]), duration=float(times[-1]),
                        frame_times_s=times.tolist(),
                        uniform_timestamps=bool(np.allclose(dt, dt.mean(), rtol=1e-3, atol=1e-7)))
            folder = dest / method / sample.name / experiment_name / f"k_{k}" / f"frames_{first}_{last}"
            target = folder / ("voxels.npz" if method == "v2v" else "events.npz")
            metadata = folder / "input.json"
            if metadata.exists() and json.loads(metadata.read_text()).get("source") != str(sample):
                raise ValueError(f"{folder} belongs to another source. Choose another experiment name.")
            results.append(str(target))
            prepared = Path(cache) / sample.name / folder.name if cache else None
            if runner.complete(target, meta, times):
                if last_method and prepared and prepared.exists():
                    shutil.rmtree(prepared)
                print(f"SKIP {target.relative_to(dest)}", flush=True)
                continue
            previous_dir = Path.cwd()
            try:
                with tempfile.TemporaryDirectory(dir=dest) as tmp:
                    work = Path(tmp)
                    run = work / "run"
                    if prepared is None:
                        runner.prepare(run, sample, files[window], ts, times, meta, size, need_gt)
                    else:
                        if not prepared.exists():
                            staging = work / "prepared"
                            runner.prepare(staging, sample, files[window], ts, times, meta, size, need_gt)
                            prepared.parent.mkdir(parents=True, exist_ok=True)
                            staging.rename(prepared)
                        run.mkdir()
                        for item in prepared.iterdir():
                            (run / item.name).symlink_to(item, target_is_directory=item.is_dir())
                    print(f"RUN  {target.relative_to(dest)}", flush=True)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    log = target.parent / "run.log"
                    if method in v3.old.EVTWEEN or method == "v2ce":
                        with log.open("w", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                            try:
                                raw, generate = v3.learned_output(method, dataset, assets, run, work, generate)
                            except Exception:
                                traceback.print_exc()
                                raise
                            finally:
                                os.chdir(previous_dir)
                    else:
                        raw = v3.old.generate_one(method, dataset, run, assets, work, log)
                    runner.publish(method, raw, target, meta, times)
                    record = work / "input.json"
                    record.write_text(json.dumps(meta, indent=2))
                    os.replace(record, target.parent / "input.json")
            except Exception as exc:
                errors.append(f"{target}: {exc}")
                print(errors[-1], file=sys.stderr, flush=True)
            finally:
                if last_method and prepared and prepared.exists():
                    shutil.rmtree(prepared)
    return results, errors


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        options = json.loads(Path(sys.argv[2]).read_text())
        Path(sys.argv[3]).write_text(json.dumps(run_method(**options)))
        sys.exit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--start-stride", type=int, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--methods", nargs="+", choices=runner.old.METHODS, required=True)
    parser.add_argument("--data-root", type=Path)
    run_experiment(**vars(parser.parse_args()))
