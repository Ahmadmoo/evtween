"""Generate all requested representations for a complete converted dataset split."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

if __package__:
    from .benchmark import prepare, save_events
else:
    from benchmark import prepare, save_events


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
LAB = PROJECT.parent
BASELINES = ("v2e", "iebcs", "dvs_voltmeter", "v2ce", "v2v", "evis_core")
EVTWEEN = {f"evtween_{v}_{mode}": (v, mode)
           for v in ("v5.2", "v5.3") for mode in ("teacher", "student")}
METHODS = (*BASELINES, "fracevent", *EVTWEEN)


def setting(key, default):
    return Path(os.environ.get(key, default)).expanduser().resolve()


def run_logged(command, log):
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 3000))
            detail = stream.read().decode(errors="replace")
        raise RuntimeError(f"Exit {result.returncode}: {detail}")


def generate_one(method, dataset, run, assets, work, log):
    inputs = run / "input"
    raw = work / "raw.npz"
    if method in BASELINES:
        python = json.loads((assets / "interpreters.json").read_text())[method]
        folder = run / method
        folder.mkdir()
        if method == "v2ce":
            command = [python, HERE / "v2ce_interval_worker.py", assets, run]
        else:
            command = [python, assets / "worker.py", method, run, '{"seed":123}']
        run_logged(command, log)
        return folder / ("voxels.npz" if method == "v2v" else "events.npz")
    if method == "fracevent":
        simulator = setting("FRACEVENT_ROOT", LAB / "FracEvent")
        python = setting("FRACEVENT_PYTHON", sys.executable)
        command = [python, simulator / "main.py", "simulate", "--config",
                   setting("FRACEVENT_CONFIG", simulator / "configs/davis346.yaml"),
                   "--input", inputs / "frames", "--timestamps", inputs / "frame_ts.npy",
                   "--device", os.getenv("FRACEVENT_DEVICE", "cpu"), "--output", raw]
    else:
        version, mode = EVTWEEN[method]
        checkout = (setting("EVTWEEN_V52_ROOT", LAB / "evtween_bnchmark-v5.2")
                    if version == "v5.2" else setting("EVTWEEN_V53_ROOT", PROJECT))
        weights = setting("EVTWEEN_WEIGHTS", LAB / "evtween-tides-compare" /
                          "benchmarks/event_generation/checkpoints/evtween_versions")
        checkpoint = weights / version / ("model.pt" if mode == "teacher" else "student.pt")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        seq = work / "sequence"
        seq.mkdir()
        for name in ("frames", "frame_ts.npy", "evaluation_meta.json"):
            (seq / name).symlink_to(inputs / name)
        if mode == "teacher":
            gt = run / "ground_truth.npz"
            if not gt.is_file():
                raise ValueError("Teacher requires recorded ground truth")
            (seq / "teacher_gt.npz").symlink_to(gt)
        command = [setting("EVTWEEN_PYTHON", sys.executable), HERE / "evtween_worker.py",
                   checkout, checkpoint, seq, raw, dataset, "123", mode]
    run_logged(command, log)
    return raw


def publish(source, output, meta, times):
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as z:
        if output.name == "voxels.npz":
            if not np.array_equal(z["edges_s"], times) or z["voxels"].shape != (
                    len(times)-1, meta["height"], meta["width"]):
                raise ValueError("V2V voxel shape or time edges do not match the input")
            os.replace(source, output)
            return
        normalized = source.parent / "normalized.npz"
        arrays = [z[key] for key in "txyp"]
        if any(a.ndim != 1 or a.shape != arrays[0].shape or not np.isfinite(a).all()
               for a in arrays):
            raise ValueError("Invalid event arrays")
        save_events(normalized, *arrays, meta["duration"], meta["width"], meta["height"])
    os.replace(normalized, output)


def generate_event_streams(dataset, split, *, methods=None, data_root=None):
    """Run every sequence, record failures independently, and return artifact paths."""
    if not dataset or Path(dataset).name != dataset or dataset in (".", ".."):
        raise ValueError("Expected a dataset name")
    if split not in ("test", "train"):
        raise ValueError("Expected test or train")
    methods = tuple(METHODS if methods is None else methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError(f"Select distinct methods from {METHODS}")
    data = Path(data_root or setting("EVTWEEN_DATA", LAB / "evtween/data"))
    sequences = sorted(p for p in (data / dataset / split).iterdir() if p.is_dir())
    if not sequences:
        raise ValueError("No sequences found")
    dest = PROJECT / "data_test_evt" / dataset / split
    dest.mkdir(parents=True, exist_ok=True)
    assets = setting("BENCHMARK_INSTALL", LAB / "evtween/evtween-benchmark/benchmarks/event_generation")
    results, errors = [], []
    for sequence in sequences:
        try:
            with tempfile.TemporaryDirectory(dir=dest) as tmp:
                run = Path(tmp) / "run"
                count = len(np.load(sequence / "frame_ts.npy", mmap_mode="r"))
                first = next(iter(sorted((sequence / "frames").glob("*.png"))), None)
                if first is None:
                    raise ValueError("No PNG frames")
                with Image.open(first) as image:
                    width, height = image.size
                # Match the historical shared center crop required by V2CE.
                prepare(sequence, run, count=count, crop_size=(min(height,260), min(width,346)))
                meta = json.loads((run / "input.json").read_text())
                times = np.load(run / "input/frame_ts.npy")
                (run / "input/evaluation_meta.json").write_text(json.dumps(
                    {"evaluation_window_in_run_s": [0, meta["duration"]]}))
                for method in methods:
                    folder = dest / method / sequence.name
                    folder.mkdir(parents=True, exist_ok=True)
                    print(f"{dataset}/{split}/{method}/{sequence.name}", flush=True)
                    name = "voxels.npz" if method == "v2v" else "events.npz"
                    try:
                        with tempfile.TemporaryDirectory(dir=tmp) as work:
                            source = generate_one(method, dataset, run, assets, Path(work),
                                                  folder / "run.log")
                            publish(source, folder / name, meta, times)
                        (folder / "input.json").write_text(json.dumps(meta, indent=2))
                        results.append(folder / name)
                    except Exception as exc:
                        errors.append(f"{sequence.name}/{method}: {exc}")
                        print(errors[-1], file=sys.stderr, flush=True)
        except Exception as exc:
            errors.append(f"{sequence.name}: {exc}")
            print(errors[-1], file=sys.stderr, flush=True)
    if errors:
        raise RuntimeError(f"{len(results)} outputs generated; failures:\n" + "\n".join(errors))
    return results


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("dataset")
    cli.add_argument("split", choices=("train", "test"))
    cli.add_argument("--methods", nargs="+", choices=METHODS)
    cli.add_argument("--data-root", type=Path)
    generate_event_streams(**vars(cli.parse_args()))
