"""Gate check for Phase 1a: are the generated training events sane before any training?

Reads every simulator's events for a dataset/split (data_test_evt/<dataset>/<split>/<method>/<sequence>/
events.npz) and the frames they were made from, and writes:
  sanity.csv      per method and sequence: events, duration, event rate, events per pixel per second,
                  ON fraction, and per training-window spacing the median / 5th / 95th percentile number
                  of events per window and the share of empty windows
  per_method.csv  the same, summed / averaged per method
  samples/*.png   the same random windows for every simulator: target frame next to each simulator's
                  events in that window (red = ON, blue = OFF), event count under each image
Training windows: events between frame k - s and frame k, where s is the spacing in frames
(--spacings; at 240 fps, 10 frames = 41.7 ms, about one DAVIS240C frame interval).

Example:
  python benchmarks/recon_train/check_training_events.py gopro --methods v2e dvs_voltmeter evtween_v8.2_student
  -> reports/recon_train/check_events/gopro/train/<time>/
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import recon_eval as r  # noqa: E402

PROJECT = HERE.parents[1]


def load_events(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in "txyp"}


def frame_info(folder):
    """Prepared frames and their times (relative to the first frame) for one sequence output folder."""
    meta = json.loads((folder / "input.json").read_text())
    src = Path(meta["source"])
    frames = sorted((src / "frames").glob("*.png"))
    ts = np.load(src / "frame_ts.npy").astype(np.float64)
    return frames, ts - ts[0], meta


def window_counts(t, times, spacing):
    """Events in [times[k - s], times[k]) for every k >= s."""
    cuts = np.searchsorted(t, times)
    return cuts[spacing:] - cuts[:-spacing]


def event_image(ev, a, b, width, height):
    """Red = ON, blue = OFF on dark gray; brightness grows with the count per pixel."""
    img = np.full((height, width, 3), 40, np.float32)
    if b > a:
        x, y, p = ev["x"][a:b], ev["y"][a:b], ev["p"][a:b]
        on = np.zeros((height, width), np.float32)
        off = np.zeros((height, width), np.float32)
        np.add.at(on, (y[p > 0], x[p > 0]), 1)
        np.add.at(off, (y[p <= 0], x[p <= 0]), 1)
        scale = max(np.percentile(np.maximum(on, off)[np.maximum(on, off) > 0], 99), 1.0)
        on_s, off_s = np.clip(on / scale, 0, 1), np.clip(off / scale, 0, 1)
        img[..., 0] += on_s * 215
        img[..., 2] += off_s * 215
        img[..., 1] += 0
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", default="gopro")
    p.add_argument("--split", default="train")
    p.add_argument("--methods", nargs="+", help="Default: every method folder found")
    p.add_argument("--events-root", help="Default: data_test_evt/<dataset>/<split>")
    p.add_argument("--spacings", nargs="+", type=int, default=[1, 6, 10, 14],
                   help="Training-window lengths in frames to report")
    p.add_argument("--samples", type=int, default=8, help="Random windows to draw (same for every method)")
    p.add_argument("--sample-spacing", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tile-width", type=int, default=240)
    p.add_argument("--out", help="Report folder (default: reports/recon_train/check_events/<dataset>/<split>/<time>)")
    args = p.parse_args(argv)
    root = Path(args.events_root) if args.events_root else PROJECT / "data_test_evt" / args.dataset / args.split
    methods = args.methods or sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    missing = [m for m in methods if not (root / m).is_dir()]
    if missing:
        raise FileNotFoundError(f"No events for {missing} under {root}")
    seqs = sorted({s.name for m in methods for s in (root / m).iterdir() if (s / "events.npz").is_file()})
    if not seqs:
        raise FileNotFoundError(f"No events.npz under {root}")
    out = Path(args.out) if args.out else (PROJECT / "reports" / "recon_train" / "check_events" / args.dataset /
                                           args.split / datetime.now().strftime("%Y%m%d-%H%M%S"))
    (out / "samples").mkdir(parents=True, exist_ok=True)
    rows, events, frames = [], {}, {}
    for seq in seqs:
        for m in methods:
            path = root / m / seq / "events.npz"
            if not path.is_file():
                rows.append(dict(method=m, sequence=seq, status="missing"))
                continue
            ev = load_events(path)
            files, times, meta = frame_info(path.parent)
            frames[seq] = (files, times, meta)
            events[(m, seq)] = ev
            n, dur = int(ev["t"].size), float(times[-1])
            pixels = meta["width"] * meta["height"]
            row = dict(method=m, sequence=seq, status="ok", frames=len(files), duration_s=round(dur, 4), events=n,
                       rate_mev_s=n / dur / 1e6, ev_per_px_s=n / dur / pixels,
                       on_fraction=float((ev["p"] > 0).mean()) if n else float("nan"))
            for s in args.spacings:
                if s < len(times):
                    c = window_counts(ev["t"], times, s)
                    row[f"win{s}_median"] = float(np.median(c))
                    row[f"win{s}_p5"] = float(np.percentile(c, 5))
                    row[f"win{s}_p95"] = float(np.percentile(c, 95))
                    row[f"win{s}_empty_pct"] = float(100 * (c == 0).mean())
            rows.append(row)
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "sanity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    import pandas as pd
    df = pd.DataFrame([row for row in rows if row.get("status") == "ok"])
    agg = {"events": "sum", "duration_s": "sum", "on_fraction": "mean"}
    agg.update({c: "median" for c in df.columns if c.startswith("win") and c.endswith(("_median", "_p5", "_p95"))})
    agg.update({c: "mean" for c in df.columns if c.endswith("_empty_pct")})
    per = df.groupby("method", sort=False).agg(agg)
    per.insert(2, "rate_mev_s", per["events"] / per["duration_s"] / 1e6)
    per.to_csv(out / "per_method.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(per.round(3).to_string())
    rng = np.random.default_rng(args.seed)
    s = args.sample_spacing
    for i in range(args.samples):
        seq = seqs[int(rng.integers(len(seqs)))]
        files, times, meta = frames[seq]
        if len(times) <= s:
            continue
        k = int(rng.integers(s, len(times)))
        a_t, b_t = times[k - s], times[k]
        tiles = [(["target frame", f"frame {k}", f"{(b_t - a_t) * 1000:.1f} ms window"], Image.open(files[k]))]
        for m in methods:
            ev = events.get((m, seq))
            if ev is None:
                continue
            a, b = np.searchsorted(ev["t"], [a_t, b_t])
            on = int((ev["p"][a:b] > 0).sum())
            tiles.append(([m, f"{b - a} events", f"ON {on} / OFF {b - a - on}"],
                          event_image(ev, a, b, meta["width"], meta["height"])))
        r.contact_sheet(out / "samples" / f"{i:02d}_{seq}_{k:06d}.png",
                        f"{args.dataset}/{seq}: events in frames {k - s}..{k} (red ON, blue OFF)", [("", tiles)],
                        args.tile_width)
    print(f"Report: {out} (sanity.csv, per_method.csv, samples/)")
    return out


if __name__ == "__main__":
    main()
