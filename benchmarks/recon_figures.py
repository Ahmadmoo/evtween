"""Side-by-side comparison images of reconstructed frames, built from a run's saved frames.

Used by the notebooks. One image per frame (recon_eval.py) or window (recon_eval_tms.py):
rows = networks; columns = [ground truth], real, every method; under each image its scores
(vs real, and vs GT when show_gt). Images are saved inside the run's report folder:
  recon_eval.py run:      <run>/figures/<name>.png
  recon_eval_tms.py run:  <run>/<setting>/figures/<name>.png
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

if __package__:
    from . import recon_eval as r
else:
    import recon_eval as r

REAL = r.REAL


def run_config(run_dir):
    return json.loads((Path(run_dir) / "config.json").read_text())


def recon_root(run_dir, kind):
    """Where a run saved its frames (recorded in config.json; standard location otherwise)."""
    cfg = run_config(run_dir)
    if cfg.get("recon_root"):
        return Path(cfg["recon_root"])
    base = "data_test_recon" if kind == "frames" else "data_test_recon_tms"
    return r.ROOT / base / cfg["dataset"] / cfg.get("split", "test")


def _source(folder):
    meta = Path(folder) / "recon.json"
    return Path(json.loads(meta.read_text())["source"]) if meta.is_file() else None


def gt_frame(source, k):
    return sorted((Path(source) / "frames").glob("*.png"))[int(k)]


def nearest_gt(source, t_end):
    """Nearest camera frame to a window end: (path, frame index, offset in ms)."""
    edges, _ = r.rm.Sequence(Path(source)).geometry
    edges = np.asarray(edges, np.float64)
    k = int(np.argmin(np.abs(edges - t_end)))
    return gt_frame(source, k), k, (edges[k] - t_end) * 1000.0


def _caption(method, row, show_gt):
    lines = [method]
    if row is None:
        return lines
    if method != REAL:
        if r.pd_notna(row.get("agreement_lpips")):
            lines.append(f"vs real LPIPS {row['agreement_lpips']:.3f}")
        if r.pd_notna(row.get("agreement_ssim")):
            lines.append(f"vs real SSIM {row['agreement_ssim']:.3f}")
        if r.pd_notna(row.get("agreement_mse")):
            lines.append(f"vs real MSE {row['agreement_mse']:.4f}")
    if show_gt and r.pd_notna(row.get("gt_mse")):
        lines.append(f"vs GT PSNR {float(r.psnr_from_mse(row['gt_mse'])):.2f}")
        if r.pd_notna(row.get("gt_ssim")):
            lines.append(f"vs GT SSIM {row['gt_ssim']:.3f}")
    return lines


def _figure(out, title, networks, methods, png, scores, gt, show_gt, tile_width):
    rows = []
    for net in networks:
        tiles = []
        for m in methods:
            path = png(net, m)
            if path.is_file():
                row = scores.get((net, m))
                tiles.append((_caption(m, row, show_gt), Image.open(path)))
        if not tiles:  # nothing reconstructed was saved for this network: no row (never a GT-only row)
            continue
        if show_gt and gt is not None:
            tiles.insert(0, (gt[1], Image.open(gt[0]).convert("L")))
        rows.append((net, tiles))
    if not rows:
        return None
    return r.contact_sheet(Path(out), title, rows, tile_width)


def frame_figure(run_dir, sequence, frame, name, show_gt=True, methods=None, networks=None, title="",
                 tile_width=320):
    """Comparison image of one frame of a recon_eval.py run, saved as <run>/figures/<name>.png."""
    import pandas as pd
    run_dir = Path(run_dir)
    cfg = run_config(run_dir)
    root = recon_root(run_dir, "frames")
    networks = networks or cfg.get("networks", list(r.NETWORKS))
    methods = [REAL] + [m for m in (methods or cfg["methods"]) if m != REAL]
    pf = pd.read_csv(run_dir / "per_frame" / f"{sequence}.csv")
    pf = pf[pf["frame"] == int(frame)]
    scores = {(row["network"], row["method"]): row for _, row in pf.iterrows()}
    source = _source(root / networks[0] / REAL / sequence)
    gt = (gt_frame(source, frame), ["ground truth", f"frame {int(frame)}"]) if (show_gt and source) else None

    def png(net, m):
        return root / net / m / sequence / f"{int(frame):06d}.png"
    head = f"{cfg['dataset']} / {sequence} / frame {int(frame)} / between frames" + (f"   {title}" if title else "")
    return _figure(run_dir / "figures" / f"{name}.png", head, networks, methods, png, scores, gt, show_gt, tile_width)


def window_figure(run_dir, setting, sequence, window, name, show_gt=True, methods=None, networks=None, title="",
                  tile_width=320):
    """Comparison image of one window of a recon_eval_tms.py run, saved as <run>/<setting>/figures/<name>.png."""
    import pandas as pd
    run_dir = Path(run_dir)
    cfg = run_config(run_dir)
    root = recon_root(run_dir, "windows") / setting
    networks = networks or cfg.get("networks", list(r.NETWORKS))
    methods = [REAL] + [m for m in (methods or cfg["methods"]) if m != REAL]
    pw = pd.read_csv(run_dir / setting / "per_window" / f"{sequence}.csv")
    pw = pw[pw["window"] == int(window)]
    scores = {(row["network"], row["method"]): row for _, row in pw.iterrows()}
    t_end = float(pw["t_end_s"].iloc[0]) if not pw.empty else None
    gt = None
    if show_gt and t_end is not None:
        source = _source(root / networks[0] / REAL / sequence)
        if source:
            path, k, offset = nearest_gt(source, t_end)
            gt = (path, ["nearest GT frame", f"frame {k}", f"offset {offset:+.1f} ms"])

    def png(net, m):
        return root / net / m / sequence / f"w{int(window):06d}.png"
    head = (f"{cfg['dataset']} / {sequence} / {setting} / window {int(window)}"
            + (f" (ends {t_end:.4f} s)" if t_end is not None else "") + (f"   {title}" if title else ""))
    return _figure(run_dir / setting / "figures" / f"{name}.png", head, networks, methods, png, scores, gt, show_gt,
                   tile_width)
