"""Tests for recon_evreal_tms.py with a fake EVREAL eval.py (no torch needed):
    python -m pytest benchmarks/test_recon_evreal_tms.py -q
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recon_eval as r  # noqa: E402
import recon_evreal_tms as ev  # noqa: E402
from test_recon_eval import DATASET, SPLIT, toy  # noqa: E402,F401  (toy is a fixture)

FAKE_EVAL = r'''
import argparse, json, os
p = argparse.ArgumentParser()
p.add_argument("-c", nargs="+"); p.add_argument("-m", nargs="+"); p.add_argument("-d", nargs="+")
p.add_argument("-qm", nargs="+")
a = p.parse_args()
for cfg in a.c:
    json.load(open(os.path.join("config", "eval", cfg + ".json")))
    for ds in a.d:
        dcfg = json.load(open(os.path.join("config", "dataset", ds + ".json")))
        for seq in dcfg["sequences"]:
            path = os.path.join(dcfg["root_path"], seq)
            for f in ("events_ts.npy", "events_xy.npy", "events_p.npy", "images.npy", "images_ts.npy",
                      "image_event_indices.npy", "metadata.json"):
                assert os.path.exists(os.path.join(path, f)), (path, f)
            for m in a.m:
                out = os.path.join("outputs", cfg, ds, seq, m)
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(out, "timestamps.txt"), "w") as f:
                    f.writelines(f"{i} {i * 0.05:.15f}\n" for i in range(5))
                for q in a.qm:
                    val = 0.1 + 0.01 * len(ds) + (0.05 if m == "FireNet" else 0)
                    with open(os.path.join(out, q + ".txt"), "w") as f:
                        f.writelines(f"{i} {val:.5f}\n" for i in range(1, 5))
print("fake EVREAL done")
'''


def fake_evreal(root, monkeypatch):
    (root / "model").mkdir(parents=True)
    (root / "model" / "__init__.py").write_text("")
    (root / "config" / "dataset").mkdir(parents=True)
    (root / "config" / "eval").mkdir(parents=True)
    (root / "eval.py").write_text(FAKE_EVAL)
    fake = {}
    for net, (rel, _, size) in r.CHECKPOINTS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(net.encode())
        fake[net] = (rel, r.file_sha256(root / rel), size)
    monkeypatch.setattr(r, "CHECKPOINTS", fake)
    monkeypatch.setattr(ev, "EVREAL_PACKAGES", {"json": "json"})


def test_names(toy):  # noqa: F811
    args = ev.parser().parse_args([DATASET, "--bin-ms", "2", "5"])
    assert ev.eval_config_names(args) == ["evtween_std", "evtween_t10ms", "evtween_t25ms"]
    assert ev.dataset_name(args, "v2e") == f"evtween__{DATASET}__{SPLIT}__v2e"


def test_export_run_collect(toy, monkeypatch):  # noqa: F811
    import pandas as pd
    root = toy.tmp / "evreal"
    fake_evreal(root, monkeypatch)
    a = toy.args
    report = toy.tmp / "report"
    ev.main([DATASET, "--events-root", a.events_root, "--data-root", a.data_root, "--evreal-root", str(root),
             "--export-root", str(toy.tmp / "export"), "--report-dir", str(report), "--bin-ms", "2", "10",
             "--no-auto-setup", "--io-workers", "2"])
    # export: EVREAL layout, shared frames, timestamps relative to the first frame, polarity 0/1
    seq0 = toy.tmp / "export" / r.REAL / "seq0"
    t = np.load(seq0 / "events_ts.npy")
    xy = np.load(seq0 / "events_xy.npy")
    assert t.dtype == np.float64 and np.all(np.diff(t) >= 0) and t[0] >= 0
    assert xy.shape == (t.size, 2) and xy.dtype == np.uint16
    assert set(np.unique(np.load(seq0 / "events_p.npy"))) <= {0, 1}
    images = np.load(seq0 / "images.npy")
    assert images.shape[3] == 1 and images.dtype == np.uint8 and (seq0 / "images.npy").is_symlink()
    idx = np.load(seq0 / "image_event_indices.npy")
    ts = np.load(seq0 / "images_ts.npy")[:, 0]
    assert idx.shape == (len(ts), 1) and np.array_equal(idx[:, 0], np.searchsorted(t, ts))
    assert not (toy.tmp / "export" / "v2e" / "seq1").exists()   # v2e has no output for seq1
    assert not (toy.tmp / "export" / "v2v").exists()            # V2V excluded
    # configs written into the EVREAL checkout
    t10 = json.loads((root / "config" / "eval" / "evtween_t10ms.json").read_text())
    assert t10["dataset_kwargs"]["voxel_method"] == {"method": "t_seconds", "t": 0.01, "sliding_window_t": 0}
    assert t10["save_images"] is False and t10["ts_tol_ms"] == 1.0
    std = json.loads((root / "config" / "eval" / "evtween_std.json").read_text())
    assert std["dataset_kwargs"]["voxel_method"] == {"method": "between_frames"}
    dreal = json.loads((root / "config" / "dataset" / f"evtween__{DATASET}__{SPLIT}__real.json").read_text())
    assert set(dreal["sequences"]) == {"seq0", "seq1"}
    dv2e = json.loads((root / "config" / "dataset" / f"evtween__{DATASET}__{SPLIT}__v2e.json").read_text())
    assert set(dv2e["sequences"]) == {"seq0"}
    # collected tables
    frames = pd.read_csv(report / "evreal_per_frame.csv")
    assert set(frames.config) == {"evtween_std", "evtween_t10ms", "evtween_t50ms"}
    assert set(frames.method) == {r.REAL, "v2e", "evtween_v5.2_teacher"} and set(frames.network) == {"e2vid", "firenet"}
    summary = pd.read_csv(report / "evreal_summary.csv")
    row = summary[(summary.config == "evtween_t10ms") & (summary.network == "firenet") & (summary.method == "v2e")].iloc[0]
    assert row.sequences == 1 and row.frames_scored == 4
    comp = pd.read_csv(report / "comparison.csv")
    assert {"mse@between_frames", "mse@10ms", "mse@50ms"} <= set(comp.columns)
    offsets = pd.read_csv(report / "first_event_offsets.csv")
    assert set(offsets.source) == {r.REAL, "v2e", "evtween_v5.2_teacher"} and (offsets.first_event_s >= 0).all()
    # a second call reuses the export
    ev.main([DATASET, "--events-root", a.events_root, "--data-root", a.data_root, "--evreal-root", str(root),
             "--export-root", str(toy.tmp / "export"), "--report-dir", str(toy.tmp / "report2"), "--bin-ms", "2",
             "--no-auto-setup", "--no-std"])
    assert set(pd.read_csv(toy.tmp / "report2" / "evreal_summary.csv").config) == {"evtween_t10ms"}


def test_checkpoint_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "evreal"
    fake_evreal(root, monkeypatch)
    (root / r.CHECKPOINTS["e2vid"][0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        ev.verify_evreal_checkpoints(root, ["e2vid"])
