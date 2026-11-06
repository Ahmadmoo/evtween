"""Generate FracEvent streams from prepared dataset/split/sequence folders."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def generate_fracevent(dataset, split, *, data_root, simulator, config=None,
                       device="cpu", sequence=None):
    """Read frames/ and frame_ts.npy (seconds) for every selected sequence."""
    if not dataset or Path(dataset).name != dataset or dataset in {".", ".."}:
        raise ValueError("Expected a dataset name")
    if split not in {"train", "test"}:
        raise ValueError("Expected train or test")
    simulator = Path(simulator).resolve()
    config = Path(config or simulator / "configs/davis346.yaml").resolve()
    split_dir = Path(data_root).resolve() / dataset / split
    sequences = sorted(p for p in split_dir.iterdir()
                       if p.is_dir() and (sequence is None or p.name == sequence))
    if not sequences:
        raise ValueError(f"No matching sequences in {split_dir}")
    for seq in sequences:
        frames = seq / "frames"
        images = sorted(p for p in frames.iterdir() if p.suffix.lower() in IMAGE_TYPES)
        times = np.load(seq / "frame_ts.npy", allow_pickle=False).reshape(-1)
        if len(times) != len(images) or len(times) < 2:
            raise ValueError(f"{seq}: expected one timestamp per frame and at least two frames")
        if not np.isfinite(times).all() or not (np.diff(times.astype(float)) > 0).all():
            raise ValueError(f"{seq}: expected strictly increasing timestamps in seconds")
        order = images
        if all(p.stem.isdigit() for p in images):
            order = sorted(images, key=lambda p: int(p.stem))
        dest = PROJECT / "data_test_evt" / dataset / split / "fracevent" / seq.name
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Generating {dataset}/{split}/{seq.name} ({len(times)} frames)", flush=True)
        with TemporaryDirectory(dir=dest) as directory:
            temp = Path(directory)
            source = frames
            if order != images:
                source = temp / "frames"
                source.mkdir()
                for index, path in enumerate(order):
                    (source / f"{index:010d}{path.suffix}").symlink_to(path)
            output = temp / "events.npz"
            subprocess.run([
                sys.executable, str(simulator / "main.py"), "simulate",
                "--config", str(config), "--input", str(source),
                "--timestamps", str(seq / "frame_ts.npy"), "--device", device,
                "--output", str(output),
            ], cwd=simulator, check=True)
            with np.load(output, allow_pickle=False) as events:
                arrays = [events[key] for key in ("x", "y", "t", "p")]
                if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
                    raise ValueError(f"{seq}: malformed event arrays")
            os.replace(output, dest / "events.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("split", choices=["train", "test"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sequence", help="Run only this sequence for a first check")
    generate_fracevent(**vars(parser.parse_args()))
