"""GoPro sharp frames -> the event generator's dataset layout, for training reconstruction networks.

Input: the frames extracted by download_gopro.py, <src>/train/<sequence>/*.png (1280x720 RGB, 240 fps).
Output: <out>/<sequence>/frames/000000.png ... and frame_ts.npy (seconds), plus prepare.json,
where <out> defaults to $EVTWEEN_DATA/gopro/train (the generator's data root; without EVTWEEN_DATA:
<lab>/evtween/data/gopro/train, lab = the folder above this repository).

Every simulator then gets exactly the same input:
  * centre crop to 4:3 (1280x720 -> 960x720) so nothing is stretched (--crop none: whole frame);
  * downscale to 240x180 by pixel averaging (an exact 4x4 average after the crop; --size to change);
  * grayscale (ITU-R 601 luma, as PIL), stored with three equal channels so no simulator's own colour
    conversion can make inputs differ;
  * timestamps k * stride / fps; --stride N keeps every N-th frame (e.g. 4 -> 60 fps, for a lower-rate
    input experiment).
Sequences are written to a temporary folder and renamed when complete; finished sequences with the
same settings are skipped, so an interrupted run can be resumed by rerunning the command.

Example:
  python benchmarks/recon_train/prepare_gopro.py --src /u/marmary/airlabv2e/evtween/data
  python benchmarks/event_generation/generate_event_streams_v3_fast.py gopro train \\
      --methods v2e dvs_voltmeter evtween_v8.2_student --gpu-jobs 4
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
LAB = PROJECT.parent


def default_out(split="train"):
    data = Path(os.environ.get("EVTWEEN_DATA", LAB / "evtween" / "data")).expanduser()
    return data / "gopro" / split


def crop_box(width, height, mode, aspect):
    """Centre crop to aspect (w/h); 'none' keeps the whole frame."""
    if mode == "none":
        return (0, 0, width, height)
    if width / height > aspect:
        w = int(round(height * aspect))
        x = (width - w) // 2
        return (x, 0, x + w, height)
    h = int(round(width / aspect))
    y = (height - h) // 2
    return (0, y, width, y + h)


def convert_frame(job):
    src, dst, box, size = job
    with Image.open(src) as im:
        gray = im.convert("RGB").crop(box).resize(size, Image.Resampling.BOX).convert("L")
    Image.merge("RGB", (gray, gray, gray)).save(dst, compress_level=1)
    return dst


def settings(args):
    return dict(size=list(args.size), crop=args.crop, aspect=args.size[0] / args.size[1], fps=args.fps,
                stride=args.stride, resample="box (pixel averaging)", gray="PIL convert('L'): ITU-R 601-2 luma",
                channels="gray replicated to RGB")


def prepare_sequence(seq, out, args, pool):
    files = sorted(p for p in seq.iterdir() if p.suffix.lower() == ".png")[::args.stride]
    if len(files) < 2:
        raise ValueError(f"{seq}: fewer than 2 frames")
    target = out / seq.name
    stamp = dict(settings(args), source=str(seq), frames=len(files))
    meta_path = target / "prepare.json"
    if meta_path.is_file() and json.loads(meta_path.read_text()).get("settings") == stamp and not args.force:
        return target, len(files), True
    with Image.open(files[0]) as im:
        width, height = im.size
    box = crop_box(width, height, args.crop, args.size[0] / args.size[1])
    tmp = out.parent / f".{out.name}_tmp" / seq.name  # outside <out>: the generator lists every folder in <out>
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "frames").mkdir(parents=True)
    jobs = [(f, tmp / "frames" / f"{i:06d}.png", box, tuple(args.size)) for i, f in enumerate(files)]
    for _ in pool.map(convert_frame, jobs, chunksize=32):
        pass
    np.save(tmp / "frame_ts.npy", np.arange(len(files), dtype=np.float64) * args.stride / args.fps)
    (tmp / "prepare.json").write_text(json.dumps(dict(
        settings=stamp, source_size=[width, height], crop_box=list(box),
        source_files=[f.name for f in files[:3]] + ["..."] + [files[-1].name],
        time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    if target.exists():
        shutil.rmtree(target)
    os.replace(tmp, target)
    return target, len(files), False


def find_sequences(src, split):
    root = Path(src).expanduser()
    for candidate in (root / split, root / "GOPRO_Large_all" / split, root):
        seqs = sorted(p for p in candidate.iterdir() if p.is_dir() and any(p.glob("*.png"))) if candidate.is_dir() else []
        if seqs:
            return seqs
    raise FileNotFoundError(f"No <sequence>/*.png folders under {root}/{split}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Folder given to download_gopro.py --dest")
    p.add_argument("--split", default="train", choices=("train", "test"))
    p.add_argument("--out", help="Output folder (default: $EVTWEEN_DATA/gopro/<split>)")
    p.add_argument("--size", nargs=2, type=int, default=[240, 180], metavar=("W", "H"))
    p.add_argument("--crop", choices=("center", "none"), default="center",
                   help="center: crop to the output's aspect ratio first (no stretching); none: whole frame")
    p.add_argument("--fps", type=float, default=240.0, help="Frame rate of the source frames")
    p.add_argument("--stride", type=int, default=1, help="Keep every N-th frame (input rate = fps / N)")
    p.add_argument("--sequences", nargs="+", help="Only these sequences")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--force", action="store_true", help="Redo sequences that are already prepared")
    args = p.parse_args(argv)
    out = Path(args.out).expanduser() if args.out else default_out(args.split)
    out.mkdir(parents=True, exist_ok=True)
    seqs = find_sequences(args.src, args.split)
    if args.sequences:
        seqs = [s for s in seqs if s.name in set(args.sequences)]
    print(f"{len(seqs)} sequences -> {out} ({args.size[0]}x{args.size[1]}, crop {args.crop}, "
          f"{args.fps / args.stride:g} fps)", flush=True)
    total = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for seq in seqs:
            start = time.monotonic()
            target, n, skipped = prepare_sequence(seq, out, args, pool)
            total += n
            print(f"  {seq.name}: {n} frames " + ("(already prepared)" if skipped else
                                                  f"in {time.monotonic() - start:.1f}s"), flush=True)
    print(f"Done: {total} frames in {len(seqs)} sequences under {out}")
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
