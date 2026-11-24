"""Tests for recon_eval_tms.py. Torch tests are skipped when PyTorch is missing; run them on the GPU machine:
    python -m pytest benchmarks/test_recon_eval_tms.py -q
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recon_eval as r  # noqa: E402
import recon_eval_tms as tms  # noqa: E402
from test_recon_eval import FAKE_EVREAL, H, W, make_events, toy  # noqa: E402,F401  (toy is a fixture)

torch = r.torch
needs_torch = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


# ---------------------------------------------------------------- no torch needed

def test_window_edges_and_config_names():
    e = tms.window_edges(0.123, 0.025)
    assert np.allclose(e, [0, 0.025, 0.05, 0.075, 0.1]) and len(e) == 5   # tail < one window dropped
    assert len(tms.window_edges(0.1, 0.025)) == 5                          # exact multiple kept
    assert tms.config_name(5.0, "ms") == "bin5ms_ms" and tms.config_name(2.5, "events") == "bin2.5ms_events"


def test_match_frames_tolerance():
    frame_ts = np.array([0.0, 0.05, 0.1, 0.15])
    ends = np.array([0.025, 0.0505, 0.075, 0.1, 0.149])
    matched, offset, nearest = tms.match_frames(ends, frame_ts, 1.0)
    assert list(matched) == [-1, 1, -1, 2, 3]
    assert np.isclose(offset[1], -0.5) and np.isclose(offset[4], 1.0)


def test_voxel_ms_bins_are_fixed_in_time():
    # D = 2 ms, 5 bins, window [0, 10) ms: centres at 1, 3, 5, 7, 9 ms
    t = np.array([0.001, 0.003, 0.004, 0.0095, 0.0001])
    x = np.array([0, 1, 2, 3, 4]); y = np.zeros(5, int); p = np.ones(5)
    g = tms.voxel_ms_numpy(t, x, y, p, 0.0, 0.002, 5, 1, 5)
    assert g[0, 0, 0] == 1 and g[1, 0, 1] == 1                      # exactly on a centre
    assert np.isclose(g[1, 0, 2], 0.5) and np.isclose(g[2, 0, 2], 0.5)  # halfway between centres
    assert g[4, 0, 3] == 1 and g[0, 0, 4] == 1                      # outer half-bins clamp to edge bins
    assert np.isclose(g.sum(), 5)
    # unlike EVREAL scaling, the same events shifted in time land in different bins
    g2 = tms.voxel_ms_numpy(t + 0.004, x, y, p, 0.0, 0.002, 5, 1, 5)
    assert not np.allclose(g, g2)
    e1 = r.voxel_grid_numpy(np.sort(t), x, y, p, 5, 1, 5)
    e2 = r.voxel_grid_numpy(np.sort(t) + 0.004, x, y, p, 5, 1, 5)
    assert np.allclose(e1, e2)


def test_select_windows():
    import pandas as pd
    rows = []
    for s in range(100):
        for net in r.NETWORKS:
            for m in (r.REAL, "v2e", "iebcs"):
                rows.append(dict(network=net, method=m, window=s,
                                 agreement_lpips=np.nan if m == r.REAL else s / 100,
                                 agreement_ssim=np.nan if m == r.REAL else 1 - s / 100))
    windows = pd.DataFrame(rows)
    activity = {s: (1000 if s in (0, 50, 51, 52) else s) for s in range(100)}
    counts = dict(best_agreement=10, worst_agreement=10, high_activity=10, temporal=10)
    sel, score = tms.select_windows(windows, activity, counts, 40)
    assert len(sel) == 40
    assert [s for s, c in sel.items() if c == "best_agreement"] == list(range(10))     # lowest LPIPS
    assert [s for s, c in sel.items() if c == "worst_agreement"] == list(range(90, 100))
    hi = [s for s, c in sel.items() if c == "high_activity"]
    assert {50, 51, 52} <= set(hi) and 0 not in hi and len(hi) == 10                  # 0 already 'best'
    assert sum(c == "temporal" for c in sel.values()) == 10
    sel_ssim, _ = tms.select_windows(windows, activity, counts, 40, metric="agreement_ssim")
    assert [s for s, c in sel_ssim.items() if c == "best_agreement"] == list(range(10))  # highest SSIM


def test_summarize_across_bin_lengths(tmp_path):
    import csv
    import pandas as pd
    rng = np.random.default_rng(0)
    for bin_ms in (2.0, 10.0):
        cfg = tms.config_name(bin_ms, "ms")
        (tmp_path / cfg / "per_window").mkdir(parents=True)
        for s in range(3):
            rows = []
            for m, shift in ((r.REAL, 0), ("v2e", 0.1), ("evtween_v7_student", 0.02 * bin_ms)):
                for j in range(5, 15):
                    row = {c: "" for c in tms.WINDOW_COLUMNS}
                    row.update(dataset="eds", split="test", sequence=f"s{s}", config=cfg, bin_ms=bin_ms, bin_time="ms",
                               network="e2vid", method=m, window=j, t_start_s=j * 0.01, t_end_s=(j + 1) * 0.01)
                    if m != r.REAL:
                        for k in r.METRICS:
                            row[f"agreement_{k}"] = shift + rng.normal(0, 0.001)
                    if j == 10:
                        row.update(gt_frame=3, gt_offset_ms=0.2, gt_mse=0.01, gt_ssim=0.5, gt_lpips=0.3)
                    rows.append(row)
            with open(tmp_path / cfg / "per_window" / f"s{s}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=tms.WINDOW_COLUMNS)
                w.writeheader()
                w.writerows(rows)
    summary = tms.summarize(tmp_path)
    assert set(summary["config"]) == {"bin2ms_ms", "bin10ms_ms"}
    s2 = summary[(summary.config == "bin2ms_ms") & (summary.method == "evtween_v7_student")].iloc[0]
    assert s2["agreement_lpips_rank"] == 1 and s2["gt_windows"] == 3
    comp = pd.read_csv(tmp_path / "comparison.csv")
    row = comp[comp.method == "evtween_v7_student"].iloc[0]
    assert row["agreement_lpips@2ms"] < row["agreement_lpips@10ms"]
    assert r.REAL not in set(comp.method)
    for name in ("summary_all.csv", "wilcoxon_all.csv", "bin2ms_ms/summary.csv", "bin10ms_ms/per_sequence.csv"):
        assert (tmp_path / name).is_file()


# ---------------------------------------------------------------- torch

@needs_torch
def test_voxelize_ms_matches_numpy():
    rng = np.random.default_rng(5)
    edges = tms.window_edges(0.1, 0.01)
    ev = make_events(rng, 3000, 0.0999)
    ev = r._canonical(ev["t"], ev["x"], ev["y"], ev["p"])
    ev["cuts"] = np.searchsorted(ev["t"], edges)
    vox = tms.voxelize_ms(ev, edges, 2, 8, 5, 0.002, H, W, "cpu").numpy()
    for j in range(2, 8):
        a, b = ev["cuts"][j], ev["cuts"][j + 1]
        ref = r.normalize_numpy(tms.voxel_ms_numpy(ev["t"][a:b], ev["x"][a:b], ev["y"][a:b],
                                                   ev["p"][a:b].astype(float), edges[j], 0.002, 5, H, W))
        assert np.allclose(vox[j - 2], ref, atol=1e-4), j


@needs_torch
def test_end_to_end_cpu(toy, monkeypatch):  # noqa: F811
    import pandas as pd
    root = toy.tmp / "evreal"
    (root / "model").mkdir(parents=True)
    (root / "model" / "__init__.py").write_text(FAKE_EVREAL)
    with r.evreal_imports(root):
        import model as fake
        e2 = fake.E2VIDRecurrent(dict(num_bins=5, num_encoders=3, final_activation="sigmoid"))
        fn = fake.FireNet_legacy(dict(num_bins=5, num_encoders=2, final_activation=""))
    torch.save(dict(model=dict(num_bins=5, num_encoders=3), state_dict=e2.state_dict()), toy.tmp / "e2vid.pth")
    torch.save(dict(config=dict(model=dict(num_bins=5, num_encoders=2)), state_dict=fn.state_dict()),
               toy.tmp / "firenet.pth")
    monkeypatch.setattr(r, "PACKAGES", {})
    a = toy.args
    args = tms.parser().parse_args([
        a.dataset, "--events-root", a.events_root, "--data-root", a.data_root,
        "--recon-root", str(toy.tmp / "recon_tms"), "--run-dir", str(toy.tmp / "run_tms"),
        "--bin-ms", "4", "10", "--warmup-s", "0.1", "--devices", "cpu", "--skip-lpips", "--io-workers", "2",
        "--chunk", "7", "--evreal-root", str(root), "--no-auto-setup",
        "--e2vid-ckpt", str(toy.tmp / "e2vid.pth"), "--firenet-ckpt", str(toy.tmp / "firenet.pth"),
        "--select-best", "2", "--select-worst", "2", "--select-activity", "1", "--select-temporal", "1",
        "--select-total", "6"])
    run_dir = tms.run(args)
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["configs"] == ["bin4ms_ms", "bin4ms_events", "bin10ms_ms", "bin10ms_events"]
    # toy: 14 frames every 50 ms -> 0.65 s; 10 ms bins = 50 ms windows -> 13 windows, every end on a frame
    w10 = pd.read_csv(run_dir / "bin10ms_ms" / "per_window.csv")
    seq0 = w10[(w10.sequence == "seq0") & (w10.network == "e2vid") & (w10.method == "v2e")]
    assert list(seq0.window) == list(range(1, 13))                 # warm-up 0.1 s: windows ending >= 0.1 s
    assert seq0.gt_frame.notna().all() and (seq0.gt_offset_ms.abs() < 1e-6).all()
    assert w10.loc[w10.method != r.REAL, "agreement_mse"].notna().all()
    assert w10.loc[w10.method == r.REAL, "agreement_mse"].isna().all()
    # 4 ms bins = 20 ms windows -> 32 windows; ends hit a frame every 100 ms only
    w4 = pd.read_csv(run_dir / "bin4ms_ms" / "per_window.csv")
    s4 = w4[(w4.sequence == "seq0") & (w4.network == "e2vid") & (w4.method == "v2e")]
    assert len(s4) == 32 - 4 and set(s4.dropna(subset=["gt_frame"]).gt_frame.astype(int)) == {2, 4, 6, 8, 10, 12}
    # fixed-ms and EVREAL scaling give different reconstructions for the same windows
    we = pd.read_csv(run_dir / "bin4ms_events" / "per_window.csv")
    assert not np.allclose(w4.sort_values(["sequence", "network", "method", "window"]).agreement_mse.fillna(0),
                           we.sort_values(["sequence", "network", "method", "window"]).agreement_mse.fillna(0))
    sel = pd.read_csv(run_dir / "selection.csv")
    assert (sel.groupby(["config", "sequence"]).size() == 6).all()
    for _, row in sel[sel.config == "bin4ms_ms"].iterrows():
        for net in ("e2vid", "firenet"):
            assert (toy.tmp / "recon_tms" / "bin4ms_ms" / net / "v2e" / row.sequence / f"w{row.window:06d}.png").is_file() \
                or row.sequence == "seq1"                                # v2e missing in seq1
            assert (toy.tmp / "recon_tms" / "bin4ms_ms" / net / r.REAL / row.sequence / f"w{row.window:06d}.png").is_file()
    assert len(list((run_dir / "bin4ms_ms" / "contact_sheets" / "seq0").glob("*.png"))) == 6
    comp = pd.read_csv(run_dir / "comparison.csv")
    assert {"agreement_ssim@4ms", "agreement_ssim@10ms"} <= set(comp.columns)
    # resume: nothing recomputed, same numbers
    again = tms.run(args)
    assert pd.read_csv(again / "bin10ms_ms" / "per_window.csv").equals(w10)
