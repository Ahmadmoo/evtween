"""Generate benchmark outputs independently on regularly spaced frame windows."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.event_generation import generate_event_streams_v2 as runner


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
    with runner.split_lock(dest):
        for sample in samples:
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
                pending = {}
                for method in methods:
                    folder = dest / method / sample.name / experiment_name / f"k_{k}" / f"frames_{first}_{last}"
                    target = folder / ("voxels.npz" if method == "v2v" else "events.npz")
                    metadata = folder / "input.json"
                    if metadata.exists() and json.loads(metadata.read_text()).get("source") != str(sample):
                        raise ValueError(f"{folder} belongs to another source. Choose another experiment name.")
                    results.append(target)
                    if runner.complete(target, meta, times):
                        print(f"SKIP {target.relative_to(dest)}", flush=True)
                    else:
                        pending[method] = target
                if not pending:
                    continue
                with tempfile.TemporaryDirectory(dir=dest) as tmp:
                    work = Path(tmp)
                    run = work / "run"
                    runner.prepare(run, sample, files[window], ts, times, meta, size,
                                   any(m.endswith("_teacher") for m in pending))
                    for method, target in pending.items():
                        print(f"RUN  {target.relative_to(dest)}", flush=True)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        scratch = work / method
                        scratch.mkdir()
                        try:
                            raw = old.generate_one(method, dataset, run, assets, scratch, target.parent / "run.log")
                            runner.publish(method, raw, target, meta, times)
                            record = scratch / "input.json"
                            record.write_text(json.dumps(meta, indent=2))
                            os.replace(record, target.parent / "input.json")
                        except Exception as exc:
                            errors.append(f"{target}: {exc}")
                            print(errors[-1], file=sys.stderr, flush=True)
    if errors:
        raise RuntimeError("\n".join(errors))
    return results


if __name__ == "__main__":
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
