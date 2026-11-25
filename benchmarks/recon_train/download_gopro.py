"""Download the GoPro dataset (all sharp 240 fps frames) for training reconstruction networks.

Source: the dataset authors' Hugging Face mirror, listed on https://seungjunnah.github.io/Datasets/gopro
(Nah, Kim, Lee, CVPR 2017; CC BY 4.0). GOPRO_Large_all.zip holds every sharp frame that the blurry
GOPRO_Large images were made from, i.e. the high-frame-rate frames we feed to the event simulators.

  * resumable: an interrupted download continues where it stopped (rerun the same command);
  * verified: size and SHA-256 must match the values published on Hugging Face;
  * extracts only the requested splits (default: train), then lists sequences and frame counts.

Example:
  python benchmarks/recon_train/download_gopro.py --dest /u/marmary/airlab_v2e/evtween/data
  -> <dest>/GOPRO_Large_all.zip and <dest>/GOPRO_Large_all/train/<sequence>/*.png
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile

FILES = {
    "GOPRO_Large_all": dict(
        url="https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large_all.zip",
        size=37929656970,
        sha256="e56908dbd6e01eccc47a206bc1b786fd526a0a499a67341ad3b2f507ef7ef1ca"),
}
CHUNK = 8 << 20


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def download(url, dest, size, retries=20, timeout=60, report_every=30.0):
    """Resumable HTTP download into dest (via dest.part). Returns dest."""
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size == size:
        print(f"{dest.name}: already downloaded")
        return dest
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.is_file() else 0
        if have == size:
            break
        if have > size:
            part.unlink()
            have = 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-", "User-Agent": "evtween-benchmark"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                if have and resp.status != 206:  # server ignored the range: start over
                    print("server does not support resume; restarting", flush=True)
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab"
                print(f"{dest.name}: {'resuming at ' + human(have) if have else 'starting'} "
                      f"({human(size)} total)", flush=True)
                start, last, got = time.monotonic(), time.monotonic(), 0
                with part.open(mode) as out:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        out.write(block)
                        got += len(block)
                        now = time.monotonic()
                        if now - last >= report_every:
                            rate = got / (now - start)
                            done = have + got
                            eta = (size - done) / rate if rate else float("inf")
                            print(f"  {human(done)} / {human(size)} ({100 * done / size:.1f}%), {human(rate)}/s, "
                                  f"ETA {int(eta // 3600)}:{int(eta % 3600 // 60):02d}:{int(eta % 60):02d}", flush=True)
                            last = now
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            wait = min(60, 5 * attempt)
            print(f"  interrupted ({exc}); retry {attempt}/{retries} in {wait}s", flush=True)
            time.sleep(wait)
            continue
        if part.stat().st_size == size:
            break
    else:
        raise RuntimeError(f"Download incomplete after {retries} attempts; rerun to resume ({part})")
    os.replace(part, dest)
    return dest


def verify(path, size, sha256):
    actual = path.stat().st_size
    if actual != size:
        raise ValueError(f"{path}: {actual} bytes, expected {size}")
    print(f"{path.name}: checking SHA-256 (a few minutes for large files) ...", flush=True)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    if h.hexdigest() != sha256:
        raise ValueError(f"{path}: SHA-256 {h.hexdigest()} does not match {sha256}; delete it and download again")
    print(f"{path.name}: verified", flush=True)


def split_of(member):
    """'GOPRO_Large_all/train/GOPR0372_07_00/000001.png' -> 'train' (the first 'train'/'test' path part)."""
    for part in Path(member).parts:
        if part in ("train", "test"):
            return part
    return None


def extract(zip_path, dest, splits):
    """Extract only files under the requested splits; skip files already extracted with the right size."""
    written = skipped = 0
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist() if not m.is_dir() and split_of(m.filename) in splits]
        if not members:
            raise ValueError(f"No files for splits {splits} in {zip_path}")
        for i, m in enumerate(members, 1):
            target = dest / m.filename
            if target.is_file() and target.stat().st_size == m.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            written += 1
            if i % 5000 == 0:
                print(f"  extracted {i}/{len(members)} files", flush=True)
    print(f"extracted {written} files ({skipped} already there)", flush=True)
    return members


def summarize(dest, members):
    """Sequences and frame counts per split, and the frame size of the first image."""
    seqs = {}
    for m in members:
        p = Path(m.filename)
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            seqs.setdefault((split_of(m.filename), p.parent), 0)
            seqs[(split_of(m.filename), p.parent)] += 1
    for split in sorted({s for s, _ in seqs}):
        rows = sorted((folder, n) for (s, folder), n in seqs.items() if s == split)
        print(f"{split}: {len(rows)} folders, {sum(n for _, n in rows)} frames")
        for folder, n in rows:
            print(f"   {folder}  {n}")
    first = next((dest / m.filename for m in members if Path(m.filename).suffix.lower() == ".png"), None)
    if first is not None and first.is_file():
        from PIL import Image
        with Image.open(first) as im:
            print(f"frame size: {im.size[0]}x{im.size[1]} ({im.mode})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True, help="Folder for the zip and the extracted frames")
    p.add_argument("--splits", nargs="+", default=["train"], choices=("train", "test"))
    p.add_argument("--keep-zip", action="store_true", help="Keep the 38 GB zip after extracting (default: delete)")
    p.add_argument("--no-extract", action="store_true", help="Only download and verify")
    p.add_argument("--skip-sha", action="store_true", help="Skip the SHA-256 check (size is still checked)")
    p.add_argument("--file", default="GOPRO_Large_all", choices=sorted(FILES))
    args = p.parse_args(argv)
    info = FILES[args.file]
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{args.file}.zip"
    need = 0 if zip_path.is_file() else info["size"] - (zip_path.with_name(zip_path.name + ".part").stat().st_size
                                                         if zip_path.with_name(zip_path.name + ".part").is_file() else 0)
    need += 0 if args.no_extract else info["size"]  # extracted frames: about the zip's size again (estimate)
    free = shutil.disk_usage(dest).free
    print(f"free space in {dest}: {human(free)}; needed about {human(need)}", flush=True)
    if free < need:
        raise SystemExit(f"Not enough free space in {dest}: {human(free)} free, about {human(need)} needed")
    download(info["url"], zip_path, info["size"])
    if args.skip_sha:
        if zip_path.stat().st_size != info["size"]:
            raise ValueError(f"{zip_path}: wrong size")
    else:
        verify(zip_path, info["size"], info["sha256"])
    if args.no_extract:
        return zip_path
    members = extract(zip_path, dest, set(args.splits))
    summarize(dest, members)
    if not args.keep_zip:
        zip_path.unlink()
        print(f"deleted {zip_path.name} (use --keep-zip to keep it)")
    print(f"Done: frames under {dest}")
    return dest


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
