"""Tests for recon_figures.py (no torch needed):  python -m pytest benchmarks/test_recon_figures.py -q"""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recon_eval as r  # noqa: E402
import recon_eval_tms as tms  # noqa: E402
import recon_figures as f  # noqa: E402

H, W, N = 24, 32, 12
METHODS = ["v2e", "evtween_v8_student"]


def _png(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((H, W), value, np.uint8)).save(path)


def _source(tmp):
    src = tmp / "data" / "eds" / "test" / "seqA"
    for k in range(N):
        _png(src / "frames" / f"{k:06d}.png", 10 * k)
    np.save(src / "frame_ts.npy", 50.0 + np.arange(N) * 0.05)
    return src


def _rows(columns, base, methods, **values):
    rows = []
    for net in r.NETWORKS:
        for m in [r.REAL] + methods:
            row = {c: "" for c in columns}
            row.update(base, network=net, method=m, gt_mse=0.01, gt_ssim=0.5, gt_lpips=0.3)
            if m != r.REAL:
                row.update(values)
            rows.append(row)
    return rows


def _write(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def test_frame_figure_with_and_without_gt(tmp_path):
    src = _source(tmp_path)
    run = tmp_path / "reports" / "recon" / "eds" / "test" / "run"
    root = tmp_path / "recon"
    (run).mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(dict(dataset="eds", split="test", methods=METHODS,
                                                     networks=list(r.NETWORKS), recon_root=str(root))))
    _write(run / "per_frame" / "seqA.csv", r.FRAME_COLUMNS,
           _rows(r.FRAME_COLUMNS, dict(dataset="eds", split="test", sequence="seqA", frame=5, t_s=0.25), METHODS,
                 agreement_lpips=0.2, agreement_ssim=0.8, agreement_mse=0.01))
    for net in r.NETWORKS:
        for m in [r.REAL] + METHODS:
            _png(root / net / m / "seqA" / "000005.png", 120)
            (root / net / m / "seqA" / "recon.json").write_text(json.dumps(dict(source=str(src))))
    with_gt = f.frame_figure(run, "seqA", 5, "a", show_gt=True, tile_width=64)
    without = f.frame_figure(run, "seqA", 5, "b", show_gt=False, tile_width=64)
    assert with_gt == run / "figures" / "a.png" and without == run / "figures" / "b.png"
    with Image.open(with_gt) as a, Image.open(without) as b:
        assert a.width > b.width                       # one more column (ground truth)
    assert f.frame_figure(run, "seqA", 7, "missing", tile_width=64) is None  # no saved frame 7


def test_window_figure_nearest_gt(tmp_path):
    src = _source(tmp_path)
    run = tmp_path / "reports" / "recon_tms" / "eds" / "test" / "run"
    root = tmp_path / "recon_tms"
    cfg = tms.config_name(2.0, "ms")
    (run / cfg).mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(dict(dataset="eds", split="test", methods=METHODS,
                                                     networks=list(r.NETWORKS), recon_root=str(root))))
    _write(run / cfg / "per_window" / "seqA.csv", tms.WINDOW_COLUMNS,
           _rows(tms.WINDOW_COLUMNS, dict(dataset="eds", split="test", sequence="seqA", config=cfg, bin_ms=2.0,
                                          bin_time="ms", window=10, t_start_s=0.1, t_end_s=0.11), METHODS,
                 agreement_lpips=0.3, agreement_ssim=0.7, agreement_mse=0.02))
    for net in r.NETWORKS:
        for m in [r.REAL] + METHODS:
            _png(root / cfg / net / m / "seqA" / "w000010.png", 90)
            (root / cfg / net / m / "seqA" / "recon.json").write_text(json.dumps(dict(source=str(src))))
    path, k, offset = f.nearest_gt(src, 0.11)
    assert k == 2 and np.isclose(offset, -10.0)          # frames every 50 ms: 0.10 s is frame 2
    out = f.window_figure(run, cfg, "seqA", 10, "w", show_gt=True, tile_width=64)
    assert out == run / cfg / "figures" / "w.png" and out.is_file()
