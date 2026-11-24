"""Cross-check with EVREAL itself, unchanged: fixed-duration (and between-frames) reconstruction scores.

recon_eval_tms.py is the study (same windows for every source, agreement with the real reconstruction,
fixed-ms bins). This script runs EVREAL's own eval.py on the same events, so the ground-truth scores
can be compared with EVREAL's published protocol. Nothing in EVREAL is modified.

Steps
  1. Export: every event source (real events and each method in data_test_evt/<DATASET>/<SPLIT>/) is
     written in EVREAL's format, one EVREAL "dataset" per source:
       <export-root>/<source>/<sequence>/events_ts.npy (s), events_xy.npy (N,2 uint16), events_p.npy (0/1),
       image_event_indices.npy, metadata.json, images.npy and images_ts.npy (links to one shared copy).
  2. Configs: EVREAL dataset configs (config/dataset/evtween__<dataset>__<split>__<source>.json) and
     evaluation configs (config/eval/evtween_t<T>ms.json for windows of T = 5 x bin_ms, EVREAL's 't_seconds'
     mode; evtween_std.json = EVREAL 'std' between frames) are added to the EVREAL checkout. They are
     copies of EVREAL's own t*ms/std configs with save_images off and the chosen ts_tol_ms.
  3. Run: python eval.py -m E2VID FireNet -c <configs> -d <datasets> -qm mse ssim lpips, in the EVREAL
     checkout (pinned commit, cloned automatically like recon_eval.py).
  4. Collect: EVREAL's per-frame score files (outputs/<config>/<dataset>/<sequence>/<method>/*.txt) become
     reports/recon_evreal/<DATASET>/<SPLIT>/<timestamp>/evreal_per_frame.csv, evreal_summary.csv
     (EVREAL's frame-weighted mean, plus mean +- std over sequences and ranks) and comparison.csv.

What EVREAL does differently from recon_eval_tms.py (by design of EVREAL, reported, not changed):
  * fixed-duration windows start at each stream's own first event (first_event_offsets.csv lists them);
  * time inside a window is scaled by its first and last event (no fixed-ms bins);
  * only ground-truth scores (frames within ts_tol_ms of a window end); no agreement with real events;
  * LPIPS from pyiqa, MSE/SSIM from scikit-image.

EVREAL needs extra packages (pyiqa, tabulate, yachalk, tqdm, scikit-image, ffmpeg-python, pretrainedmodels,
opencv, einops). They are pip-installed into the Python that runs EVREAL unless --no-auto-setup; use
--evreal-python to run EVREAL in a separate environment.

Example:
  python benchmarks/recon_evreal_tms.py eds --bin-ms 2 5 10 20
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

if __package__:
    from . import recon_eval as r
else:
    import recon_eval as r

ROOT, REAL = r.ROOT, r.REAL
EVREAL_METHODS = {"e2vid": "E2VID", "firenet": "FireNet"}
EVREAL_PACKAGES = {"pyiqa": "pyiqa", "tabulate": "tabulate", "yachalk": "yachalk", "tqdm": "tqdm",
                   "skimage": "scikit-image", "ffmpeg": "ffmpeg-python", "pretrainedmodels": "pretrainedmodels",
                   "cv2": "opencv-python-headless", "einops": "einops"}
EXPORT_VERSION = 1


# ---------------------------------------------------------------- names

def dataset_name(args, source):
    return f"evtween__{args.dataset}__{args.split}__{source}"


def eval_config_names(args):
    names = [f"evtween_t{args.bins * d:g}ms" for d in args.bin_ms]
    return (["evtween_std"] if args.with_std else []) + names


def export_root(args):
    return Path(args.export_root) if args.export_root else ROOT / "data_evreal" / args.dataset / args.split


# ---------------------------------------------------------------- setup

def ensure_evreal_packages(python, auto):
    """Install EVREAL's requirements into the interpreter that runs it (current one by default)."""
    probe = "import importlib.util,sys;print(','.join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))"
    out = subprocess.run([python, "-c", probe, *EVREAL_PACKAGES], capture_output=True, text=True, check=True)
    missing = [EVREAL_PACKAGES[m] for m in out.stdout.strip().split(",") if m]
    if missing:
        if not auto:
            raise ImportError(f"EVREAL needs: {python} -m pip install {' '.join(missing)}")
        print(f"Installing EVREAL requirements into {python}: {' '.join(missing)}", flush=True)
        subprocess.run([python, "-m", "pip", "install", *missing], check=True)
    return missing


def verify_evreal_checkpoints(root, networks):
    """EVREAL loads its own pretrained/<Method>/model.pth; check them against the pinned hashes."""
    out = {}
    for net in networks:
        rel, sha, _ = r.CHECKPOINTS[net]
        path = Path(root) / rel
        digest = r.file_sha256(path) if path.is_file() else None
        if digest != sha:
            raise ValueError(f"{path}: sha256 {digest} does not match the pinned {sha}; remove {root} to re-clone")
        out[net] = dict(path=str(path), sha256=digest)
    return out


# ---------------------------------------------------------------- export (no torch)

def _write_frames(data, dest):
    """Shared EVREAL images.npy (N, H, W, 1) uint8 grayscale and images_ts.npy (N, 1) seconds."""
    dest.mkdir(parents=True, exist_ok=True)
    h, w = data.meta["height"], data.meta["width"]
    tmp = dest / "images.tmp.npy"
    images = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(data.frames), h, w, 1))
    for i, path in enumerate(data.frames):
        with Image.open(path) as im:
            images[i, :, :, 0] = np.asarray(im.convert("L"), dtype=np.uint8)
    images.flush()
    del images
    os.replace(tmp, dest / "images.npy")
    np.save(dest / "images_ts.npy", np.asarray(data.edges, np.float64)[:, None])


def _link(target, link):
    if link.is_symlink() or link.exists():
        link.unlink()
    try:
        link.symlink_to(os.path.relpath(target, link.parent))
    except OSError:  # file systems without symlinks: copy
        shutil.copy2(target, link)


def _write_source(ev, data, dest, frames_dir):
    dest.mkdir(parents=True, exist_ok=True)
    t = np.ascontiguousarray(ev["t"], np.float64)
    np.save(dest / "events_ts.npy", t)
    np.save(dest / "events_xy.npy", np.stack([ev["x"], ev["y"]], axis=-1).astype(np.uint16))
    np.save(dest / "events_p.npy", (np.asarray(ev["p"]) > 0).astype(np.uint8))
    np.save(dest / "image_event_indices.npy", np.searchsorted(t, data.edges).astype(np.int64)[:, None])
    (dest / "metadata.json").write_text(json.dumps({"sensor_resolution": [data.meta["height"], data.meta["width"]]}))
    _link(frames_dir / "images.npy", dest / "images.npy")
    _link(frames_dir / "images_ts.npy", dest / "images_ts.npy")
    return float(t[0]) if t.size else float("nan"), int(t.size)


def export(args, methods, names):
    """Write every source of every sequence in EVREAL's format (skipped when already exported)."""
    root = export_root(args)
    offsets = []
    with ThreadPoolExecutor(args.io_workers) as pool:
        for name in names:
            marker = root / "_frames" / name / "export.json"
            stamp = dict(version=EXPORT_VERSION, methods=methods)
            if marker.is_file() and json.loads(marker.read_text()).get("stamp") == stamp and not args.re_export:
                offsets += json.loads(marker.read_text())["offsets"]
                print(f"{name}: already exported", flush=True)
                continue
            data = r.load_sequence(name, args, methods, pool)
            frames_dir = root / "_frames" / name
            _write_frames(data, frames_dir)
            jobs = {s: pool.submit(_write_source, ev, data, root / s / name, frames_dir)
                    for s, ev in data.events.items()}
            rows = []
            for s, job in jobs.items():
                first, count = job.result()
                rows.append(dict(sequence=name, source=s, first_event_s=first,
                                 first_event_offset_ms=first * 1000.0, events=count))
            marker.write_text(json.dumps(dict(stamp=stamp, offsets=rows), indent=2))
            offsets += rows
            print(f"{name}: exported {', '.join(jobs)}", flush=True)
    return offsets


def write_configs(args, sources_by_seq):
    """EVREAL dataset and evaluation configs inside the EVREAL checkout (new files only)."""
    root = Path(args.evreal_root)
    datasets = []
    for source in sorted({s for seqs in sources_by_seq.values() for s in seqs}, key=lambda s: (s != REAL, s)):
        seqs = sorted(n for n, srcs in sources_by_seq.items() if source in srcs)
        name = dataset_name(args, source)
        cfg = {"root_path": str(export_root(args) / source), "sequences": {n: {} for n in seqs}}
        (root / "config" / "dataset" / f"{name}.json").write_text(json.dumps(cfg, indent=4))
        datasets.append(name)
    base = {"save_images": False, "histeq": "none", "eval_infer_all": False, "ts_tol_ms": args.gt_tol_ms,
            "create_video": False}
    configs = {}
    if args.with_std:
        configs["evtween_std"] = {"dataset_kwargs": {"num_bins": args.bins, "voxel_method": {"method": "between_frames"},
                                                     "keep_ratio": 1.0}, **base}
    for d in args.bin_ms:
        configs[f"evtween_t{args.bins * d:g}ms"] = {"dataset_kwargs": {"num_bins": args.bins, "voxel_method": {
            "method": "t_seconds", "t": args.bins * d / 1000.0, "sliding_window_t": 0}}, **base}
    for name, cfg in configs.items():
        (root / "config" / "eval" / f"{name}.json").write_text(json.dumps(cfg, indent=4))
    return datasets, list(configs)


def run_evreal(args, datasets, configs, log_path):
    cmd = [args.evreal_python, "eval.py", "-m", *[EVREAL_METHODS[n] for n in args.networks], "-c", *configs,
           "-d", *datasets, "-qm", *args.metrics]
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("Running EVREAL:", " ".join(cmd), flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=args.evreal_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
    if code:
        raise RuntimeError(f"EVREAL exited with code {code}; see {log_path}")


# ---------------------------------------------------------------- collect (no torch)

def _read_scores(path):
    out = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                out[int(parts[0])] = float(parts[1])
    return out


def collect(args, datasets, configs, report_dir):
    import pandas as pd
    rows = []
    sources = {dataset_name(args, s): s for s in [REAL, *r.METHODS]}
    for cfg in configs:
        window_ms = "between_frames" if cfg == "evtween_std" else cfg.removeprefix("evtween_t").removesuffix("ms")
        for ds in datasets:
            base = Path(args.evreal_root) / "outputs" / cfg / ds
            if not base.is_dir():
                continue
            for seq_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                for net, method_name in EVREAL_METHODS.items():
                    if net not in args.networks:
                        continue
                    d = seq_dir / method_name
                    ts = _read_scores(d / "timestamps.txt")
                    scores = {m: _read_scores(d / f"{m}.txt") for m in args.metrics}
                    for idx in sorted(set().union(*[set(v) for v in scores.values()])):
                        rows.append(dict(config=cfg, window_ms=window_ms, network=net, method=sources.get(ds, ds),
                                         sequence=seq_dir.name, index=idx, t_s=ts.get(idx, float("nan")),
                                         **{m: scores[m].get(idx, float("nan")) for m in args.metrics}))
    frames = pd.DataFrame(rows)
    frames.to_csv(report_dir / "evreal_per_frame.csv", index=False)
    if frames.empty:
        print("No EVREAL scores found (check the log)", flush=True)
        return frames
    keys = ["config", "window_ms", "network", "method"]
    per_seq = frames.groupby(keys + ["sequence"], sort=False)[list(args.metrics)].mean().reset_index()
    per_seq.to_csv(report_dir / "evreal_per_sequence.csv", index=False)
    summary = frames.groupby(keys, sort=False).agg(frames_scored=("index", "size"),
                                                   sequences=("sequence", "nunique"),
                                                   **{f"{m}_frame_weighted": (m, "mean") for m in args.metrics})
    seq_stats = per_seq.groupby(keys, sort=False)[list(args.metrics)].agg(["mean", "std"])
    seq_stats.columns = [f"{m}_seq_{s}" for m, s in seq_stats.columns]
    summary = summary.join(seq_stats).reset_index()
    for m in args.metrics:
        higher = r.HIGHER_IS_BETTER.get(m, False)
        summary[f"{m}_rank"] = summary.groupby(["config", "network"])[f"{m}_frame_weighted"].rank(
            ascending=not higher, method="min")
    order = {m: i for i, m in enumerate((REAL, *r.METHODS))}
    summary = summary.sort_values(["config", "network", "method"], key=lambda c: c.map(order) if c.name == "method"
                                  else c, kind="stable")
    summary.to_csv(report_dir / "evreal_summary.csv", index=False)
    parts = []
    for m in args.metrics:
        piv = summary.pivot_table(index=["network", "method"], columns="window_ms", values=f"{m}_frame_weighted",
                                  aggfunc="first")
        piv.columns = [f"{m}@{c}" if c == "between_frames" else f"{m}@{c}ms" for c in piv.columns]
        parts.append(piv)
    pd.concat(parts, axis=1).reset_index().to_csv(report_dir / "comparison.csv", index=False)
    return summary


# ---------------------------------------------------------------- main

def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="Folder name under data_test_evt/")
    p.add_argument("--split", default="test")
    p.add_argument("--bin-ms", nargs="+", type=float, default=[2, 5, 10, 20],
                   help="Windows of --bins x bin_ms, as in recon_eval_tms.py (EVREAL scales time inside each window)")
    p.add_argument("--no-std", dest="with_std", action="store_false",
                   help="Skip EVREAL's 'std' between-frames run (a cross-check of recon_eval.py)")
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--gt-tol-ms", type=float, default=1.0, help="EVREAL ts_tol_ms")
    p.add_argument("--metrics", nargs="+", default=["mse", "ssim", "lpips"])
    p.add_argument("--networks", nargs="+", choices=r.NETWORKS, default=list(r.NETWORKS))
    p.add_argument("--methods", nargs="+"), p.add_argument("--sequences", nargs="+")
    p.add_argument("--events-root"), p.add_argument("--data-root")
    p.add_argument("--export-root", help="Default: data_evreal/<dataset>/<split>")
    p.add_argument("--re-export", action="store_true", help="Export again even if already done")
    p.add_argument("--evreal-root"), p.add_argument("--no-auto-setup", dest="auto_setup", action="store_false")
    p.add_argument("--evreal-python", default=sys.executable, help="Python that runs EVREAL (another env is fine)")
    p.add_argument("--gpu", help="Value for CUDA_VISIBLE_DEVICES when running EVREAL")
    p.add_argument("--io-workers", type=int, default=max(2, os.cpu_count() or 4))
    p.add_argument("--report-dir")
    p.add_argument("--skip-run", action="store_true", help="Export and write configs only")
    p.add_argument("--collect-only", action="store_true", help="Only collect existing EVREAL outputs")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.evreal_root = str(Path(args.evreal_root).expanduser() if args.evreal_root else r.default_evreal_root())
    methods, names = r.discover(args)
    report_dir = Path(args.report_dir) if args.report_dir else \
        ROOT / "reports" / "recon_evreal" / args.dataset / args.split / datetime.now().strftime("%Y%m%d-%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    commit = r.ensure_evreal(args.evreal_root, args.auto_setup)
    ckpts = verify_evreal_checkpoints(args.evreal_root, args.networks)
    if not args.collect_only:
        offsets = export(args, methods, names)
        import pandas as pd
        pd.DataFrame(offsets).to_csv(report_dir / "first_event_offsets.csv", index=False)
    sources_by_seq = {}
    for name in names:
        sources_by_seq[name] = [s for s in [REAL, *methods] if (export_root(args) / s / name / "events_ts.npy").is_file()]
    datasets, configs = write_configs(args, sources_by_seq)
    (report_dir / "config.json").write_text(json.dumps(dict(
        dataset=args.dataset, split=args.split, methods=methods, sequences=names, datasets=datasets, configs=configs,
        bin_ms=args.bin_ms, bins=args.bins, ts_tol_ms=args.gt_tol_ms, metrics=args.metrics, networks=args.networks,
        export_root=str(export_root(args)), evreal=dict(root=args.evreal_root, commit=commit, pinned=r.EVREAL_COMMIT,
                                                         python=args.evreal_python), checkpoints=ckpts), indent=2))
    if not args.skip_run and not args.collect_only:
        ensure_evreal_packages(args.evreal_python, args.auto_setup)
        run_evreal(args, datasets, configs, report_dir / "evreal_log.txt")
    if not args.skip_run:
        summary = collect(args, datasets, configs, report_dir)
        print(summary.to_string(index=False) if summary is not None and not summary.empty else "", flush=True)
    print(f"Done: {report_dir}", flush=True)
    return report_dir


if __name__ == "__main__":
    main()
