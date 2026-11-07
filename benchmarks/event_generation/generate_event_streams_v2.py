"""Resumable Rails split runner. Use with the same inputs, weights and settings as the original run."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile
import zlib

import numpy as np
from PIL import Image

if __package__:
    from . import generate_event_streams as old
else:
    import generate_event_streams as old


@contextmanager
def split_lock(dest):
    import fcntl
    with (dest / ".generation-v2.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another v2 runner is using {dest}") from exc
        yield


def complete(path, meta, times):
    """Accept legacy atomic publications only with matching input metadata and intact arrays."""
    try:
        if json.loads((path.parent / "input.json").read_text()) != meta:
            return False
        with zipfile.ZipFile(path) as archive:
            if path.name == "voxels.npz":
                layouts = {"voxels": (len(times)-1, meta["height"], meta["width"]),
                           "edges_s": (len(times),)}
            else:
                layouts = dict.fromkeys("txyp")
            count = None
            for key, expected in layouts.items():
                member = archive.getinfo(key + ".npy")
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        return False
                    offset = stream.tell()
                dtype = np.dtype(dtype)
                if dtype.kind not in "iuf" or offset + dtype.itemsize * int(np.prod(shape)) != member.file_size:
                    return False
                if expected is None:
                    if len(shape) != 1 or (count is not None and shape[0] != count):
                        return False
                    count = shape[0]
                elif shape != expected:
                    return False
            if "edges_s" in layouts:
                with archive.open("edges_s.npy") as stream:
                    if not np.array_equal(np.lib.format.read_array(stream), times):
                        return False
            return archive.testzip() is None
    except (OSError, ValueError, TypeError, KeyError, SyntaxError, EOFError,
            zipfile.BadZipFile, zlib.error):
        return False


def sequence_info(seq):
    bayer = seq / "bayer.txt"
    if bayer.is_file() and bayer.read_text().strip():
        raise ValueError("Bayer/CED requires a sensor-specific protocol")
    files = sorted((seq / "frames").glob("*.png"))
    ts = np.load(seq / "frame_ts.npy", allow_pickle=False).astype(np.float64)
    if ts.shape != (len(files),) or len(files) < 2 or not np.isfinite(ts).all() or not (np.diff(ts) > 0).all():
        raise ValueError("Frame count/timestamps are invalid")
    with Image.open(files[0]) as im:
        width, height = im.size
    w, h = min(width, 346), min(height, 260)
    crop = [(width-w)//2, (height-h)//2, w, h]
    times = ts - ts[0]
    dt = np.diff(times)
    meta = dict(source=str(seq), indices=list(range(len(files))), absolute_t0=float(ts[0]),
                duration=float(times[-1]), height=h, width=w, crop=crop, interpolation="none",
                input_protocol="shared selected frames; no hidden intermediate frames",
                uniform_timestamps=bool(np.allclose(dt, dt.mean(), rtol=1e-3, atol=1e-7)))
    return files, ts, times, meta, (width, height)


def prepare(run, seq, files, ts, times, meta, size, need_gt):
    inputs = run / "input"
    frames = inputs / "frames"
    frames.mkdir(parents=True)
    x, y, w, h = meta["crop"]
    for index, source in enumerate(files):
        target = frames / f"{index:06d}.png"
        with Image.open(source) as im:
            if im.size != size:
                raise ValueError(f"Frame resolution changed: {source}")
            if im.mode == "RGB" and not im.info and im.size == (w, h):
                target.symlink_to(source.resolve())
            else:
                im.convert("RGB").crop((x, y, x+w, y+h)).save(target, compress_level=1)
    np.save(inputs / "frame_ts.npy", times)
    (run / "input.json").write_text(json.dumps(meta))
    (inputs / "evaluation_meta.json").write_text(json.dumps(
        {"evaluation_window_in_run_s": [0, meta["duration"]]}))
    if need_gt and (seq / "ev_t.npy").is_file():
        stamps = np.load(seq / "ev_t.npy", mmap_mode="r")
        a, b = np.searchsorted(stamps, [ts[0], ts[-1]])
        ex, ey, pol = (np.load(seq / f"ev_{k}.npy", mmap_mode="r")[a:b] for k in "xyp")
        mask = (ex >= x) & (ex < x+w) & (ey >= y) & (ey < y+h)
        old.save_events(run / "ground_truth.npz", stamps[a:b][mask]-ts[0],
                        ex[mask]-x, ey[mask]-y, pol[mask], times[-1], w, h)


def publish(method, source, target, meta, times):
    # Baseline workers normally already produce canonical event arrays. Avoid
    # sorting, compressing and writing them twice when this can be verified.
    if method in old.BASELINES and method != "v2v":
        with np.load(source, allow_pickle=False) as z:
            if set(z.files) == set("txyp"):
                t, x, y, p = (z[k] for k in "txyp")
                canonical = (t.ndim == 1 and t.shape == x.shape == y.shape == p.shape
                        and (t.dtype, x.dtype, y.dtype, p.dtype) ==
                        (np.float64, np.int32, np.int32, np.int8)
                        and (not t.size or (0 <= t[0] and t[-1] < meta["duration"]))
                        and np.all(t[1:] >= t[:-1])
                        and np.all((x >= 0) & (x < meta["width"]))
                        and np.all((y >= 0) & (y < meta["height"]))
                        and np.all((p == -1) | (p == 1)))
            else:
                canonical = False
        if canonical:
            os.replace(source, target)
            return
    if method == "v2v":
        with np.load(source, allow_pickle=False) as z:
            if (not np.array_equal(z["edges_s"], times) or
                    z["voxels"].shape != (len(times)-1, meta["height"], meta["width"])):
                raise ValueError("V2V output geometry or time edges do not match input")
        os.replace(source, target)
        return
    old.publish(source, target, meta, times)


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
    results, errors = [], []
    with split_lock(dest):
        for seq in sequences:
            try:
                files, ts, times, meta, size = sequence_info(seq)
                pending = {}
                for method in methods:
                    name = "voxels.npz" if method == "v2v" else "events.npz"
                    target = dest / method / seq.name / name
                    if complete(target, meta, times):
                        results.append(target)
                        print(f"SKIP {seq.name}/{method}", flush=True)
                    else:
                        pending[method] = target
                if not pending:
                    continue
                with tempfile.TemporaryDirectory(dir=dest) as tmp:
                    run = Path(tmp) / "run"
                    prepare(run, seq, files, ts, times, meta, size,
                            any(m.endswith("_teacher") for m in pending))
                    for method, target in pending.items():
                        folder = target.parent
                        folder.mkdir(parents=True, exist_ok=True)
                        print(f"RUN  {seq.name}/{method}", flush=True)
                        start = time.monotonic()
                        try:
                            with tempfile.TemporaryDirectory(dir=tmp) as work:
                                source = old.generate_one(method, dataset, run, assets,
                                                          Path(work), folder / "run.log")
                                publish(method, source, target, meta, times)
                                tmp_meta = Path(work) / "input.json"
                                tmp_meta.write_text(json.dumps(meta, indent=2))
                                os.replace(tmp_meta, folder / "input.json")
                            results.append(target)
                            print(f"DONE {seq.name}/{method} ({time.monotonic()-start:.1f}s)", flush=True)
                        except Exception as exc:
                            errors.append(f"{seq.name}/{method}: {exc}")
                            print(errors[-1], file=sys.stderr, flush=True)
            except Exception as exc:
                errors.append(f"{seq.name}: {exc}")
                print(errors[-1], file=sys.stderr, flush=True)
    if errors:
        raise RuntimeError("\n".join(errors))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("split", choices=("train", "test"))
    parser.add_argument("--methods", nargs="+", choices=old.METHODS)
    parser.add_argument("--data-root", type=Path)
    generate_event_streams(**vars(parser.parse_args()))
