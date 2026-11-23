"""Tests for recon_eval.py. Torch tests are skipped when PyTorch is missing; run them on the GPU machine:
    python -m pytest benchmarks/test_recon_eval.py -q
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recon_eval as r  # noqa: E402

torch = r.torch
needs_torch = pytest.mark.skipif(torch is None, reason="PyTorch not installed")

H, W, N = 24, 32, 14
DATASET, SPLIT = "TOY", "test"


def make_events(rng, n, duration):
    t = np.sort(rng.uniform(0, duration, n))
    return dict(t=t, x=rng.integers(0, W, n).astype(np.int32), y=rng.integers(0, H, n).astype(np.int32),
                p=rng.choice(np.array([-1, 1], np.int8), n))


@pytest.fixture
def toy(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    evt = tmp_path / "evt"
    seqs = []
    for s in range(2):
        src = data / DATASET / SPLIT / f"seq{s}"
        (src / "frames").mkdir(parents=True)
        ts = 100.0 + np.arange(N) * 0.05
        for k in range(N):
            img = (rng.random((H, W, 3)) * 255).astype(np.uint8)
            Image.fromarray(img).save(src / "frames" / f"{k:06d}.png")
        np.save(src / "frame_ts.npy", ts)
        real = make_events(rng, 3000, ts[-1] - ts[0] + 0.2)
        real["t"] = real["t"] + ts[0] - 0.1  # some events outside the frame span
        for key in "txyp":
            np.save(src / f"ev_{key}.npy", real[key])
        _, _, times, meta, _ = r.v2.sequence_info(src)
        for m in ("v2e", "evtween_v5.2_teacher", "v2v"):
            out = evt / m / src.name
            out.mkdir(parents=True)
            (out / "input.json").write_text(json.dumps(meta))
            if m == "v2v":
                np.savez(out / "voxels.npz", voxels=np.zeros((N - 1, H, W), np.float32), edges_s=times)
                continue
            if m == "v2e" and s == 1:
                (out / "input.json").unlink()
                out.rmdir()
                continue  # v2e missing for seq1
            ev = make_events(rng, 2000, meta["duration"] * 0.999)
            np.savez(out / "events.npz", t=ev["t"], x=ev["x"], y=ev["y"], p=ev["p"])
        seqs.append(src)
    args = r.parser().parse_args([DATASET, "--events-root", str(evt), "--data-root", str(data),
                                  "--recon-root", str(tmp_path / "recon"), "--run-dir", str(tmp_path / "run"),
                                  "--warmup", "3", "--chunk", "5", "--devices", "cpu", "--skip-lpips",
                                  "--samples", "2", "--io-workers", "2"])
    return SimpleNamespace(args=args, tmp=tmp_path, seqs=seqs)


# ---------------------------------------------------------------- no torch needed

def test_discover_excludes_v2v_and_orders_methods(toy):
    methods, names = r.discover(toy.args)
    assert methods == ["v2e", "evtween_v5.2_teacher"]
    assert names == ["seq0", "seq1"]


def test_discover_dataset_is_a_parameter(toy):
    bad = r.parser().parse_args(["NOPE", "--events-root", str(toy.tmp / "missing")])
    with pytest.raises(FileNotFoundError):
        r.discover(bad)


def test_load_sequence_aligns_with_frames(toy):
    from concurrent.futures import ThreadPoolExecutor
    methods, _ = r.discover(toy.args)
    with ThreadPoolExecutor(2) as pool:
        d0 = r.load_sequence("seq0", toy.args, methods, pool)
        d1 = r.load_sequence("seq1", toy.args, methods, pool)
    assert set(d0.events) == {r.REAL, "v2e", "evtween_v5.2_teacher"}
    assert d1.missing == ["v2e"]
    assert len(d0.frames) == len(d0.edges) == N
    real = d0.events[r.REAL]
    assert real["t"].min() >= 0 and real["t"].max() < d0.edges[-1]
    assert set(np.unique(real["p"])) <= {-1, 1}
    st = d0.stats[r.REAL]
    assert st["windows"] == N - 1 and st["events"] == real["t"].size
    assert np.diff(real["cuts"]).sum() == real["t"].size


def test_sample_frames():
    assert r.sample_frames(14, 3, 2) == {3, 13}
    assert r.sample_frames(14, 3, 0) == set()
    assert r.sample_frames(3, 10, 5) == set()
    assert min(r.sample_frames(100, 0, 5)) == 1


def test_numpy_voxel_preserves_polarity_sum():
    rng = np.random.default_rng(1)
    ev = make_events(rng, 500, 1.0)
    grid = r.voxel_grid_numpy(ev["t"], ev["x"], ev["y"], ev["p"].astype(float), 0.0, 1.0, 5, H, W)
    assert np.isclose(grid.sum(), ev["p"].sum())
    norm = r.normalize_numpy(grid)
    nz = grid != 0
    assert np.isclose(norm[nz].mean(), 0, atol=1e-9) and np.isclose(norm[nz].std(ddof=1), 1)


def test_summarize_and_wilcoxon(tmp_path):
    rng = np.random.default_rng(2)
    run = tmp_path / "run"
    (run / "per_frame").mkdir(parents=True)
    import csv
    for s in range(7):
        rows = []
        for method, shift in ((r.REAL, 0), ("v2e", 0.1), ("evtween_v5.3_student", 0.05)):
            for k in range(3, 10):
                row = dict(dataset="X", split="test", sequence=f"s{s}", network="e2vid", method=method,
                           frame=k, t_s=k * 0.05)
                for kind in ("agreement", "gt"):
                    for m in r.METRICS:
                        row[f"{kind}_{m}"] = "" if (kind == "agreement" and method == r.REAL) else \
                            shift + rng.normal(0, 0.01)
                rows.append(row)
        with open(run / "per_frame" / f"s{s}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=r.FRAME_COLUMNS)
            w.writeheader()
            w.writerows(rows)
    summary = r.summarize(run)
    assert list(summary["method"]) == [r.REAL, "v2e", "evtween_v5.3_student"]
    row = summary.set_index("method").loc["evtween_v5.3_student"]
    assert row["agreement_mse_rank"] == 1 and row["agreement_ssim_rank"] == 2
    import pandas as pd
    wx = pd.read_csv(run / "wilcoxon.csv").set_index("metric")
    assert wx.loc["agreement_mse", "evtween_better"] and not wx.loc["agreement_ssim", "evtween_better"]
    assert wx.loc["agreement_mse", "p_value"] < 0.05 and wx.loc["agreement_mse", "sequences"] == 7
    for name in ("per_frame.csv", "per_sequence.csv", "summary.csv", "wilcoxon.csv"):
        assert (run / name).is_file()


# ---------------------------------------------------------------- torch

@needs_torch
def test_voxelize_matches_numpy():
    rng = np.random.default_rng(3)
    edges = np.array([0.0, 0.03, 0.05, 0.11, 0.2])
    ev = make_events(rng, 4000, 0.2 - 1e-9)
    ev = r._canonical(ev["t"], ev["x"], ev["y"], ev["p"])
    _, ev["cuts"] = r._stats(ev, edges)
    vox = r.voxelize(ev, torch.from_numpy(edges), 1, 4, 5, H, W, "cpu").numpy()
    for j in range(1, 4):
        a, b = ev["cuts"][j], ev["cuts"][j + 1]
        ref = r.normalize_numpy(r.voxel_grid_numpy(ev["t"][a:b], ev["x"][a:b], ev["y"][a:b],
                                                   ev["p"][a:b].astype(float), edges[j], edges[j + 1], 5, H, W))
        assert np.allclose(vox[j - 1], ref, atol=1e-4)


@needs_torch
def test_ssim_matches_skimage():
    from skimage.metrics import structural_similarity
    rng = np.random.default_rng(4)
    a = rng.random((40, 50)).astype(np.float32)
    b = np.clip(a + rng.normal(0, 0.1, a.shape), 0, 1).astype(np.float32)
    sc = r.Scorer("cpu", "none", use_lpips=False)
    ours = sc.ssim(torch.from_numpy(a)[None, None], torch.from_numpy(b)[None, None]).item()
    ref = structural_similarity(a, b, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                use_sample_covariance=False)
    assert abs(ours - ref) < 1e-4


DUMMY = '''
import torch
class Dummy(torch.nn.Module):
    """Recurrent toy network with the rpg_e2vid interface: forward(x, prev_states) -> (img, states)."""
    def __init__(self, config):
        super().__init__()
        self.conv = torch.nn.Conv2d(config.get("num_bins", 5), 1, 3, padding=1)
    def forward(self, x, prev_states=None):
        y = self.conv(x)
        if prev_states is not None:
            y = y + 0.5 * prev_states
        return torch.sigmoid(y), y.detach()
'''


@needs_torch
def test_end_to_end_cpu(toy):
    code = toy.tmp / "code"
    (code / "model").mkdir(parents=True)
    (code / "model" / "__init__.py").write_text("")
    (code / "model" / "model.py").write_text(DUMMY)
    sys.path.insert(0, str(code))
    from model.model import Dummy
    ckpt = toy.tmp / "dummy.pth"
    torch.save(dict(arch="Dummy", model={"num_bins": 5}, state_dict=Dummy({"num_bins": 5}).state_dict()), ckpt)
    sys.path.remove(str(code))
    for key in [k for k in sys.modules if k == "model" or k.startswith("model.")]:
        sys.modules.pop(key)
    args = toy.args
    args.e2vid_ckpt = args.firenet_ckpt = str(ckpt)
    args.e2vid_code = args.firenet_code = str(code)
    run_dir = r.run(args)
    import pandas as pd
    frames = pd.read_csv(run_dir / "per_frame.csv")
    # 2 networks x frames 3..13 x (seq0: 3 sources, seq1: 2 sources)
    assert len(frames) == 2 * 11 * (3 + 2)
    assert frames.loc[frames.method == r.REAL, "agreement_mse"].isna().all()
    assert frames.loc[frames.method != r.REAL, "agreement_mse"].notna().all()
    assert frames["gt_ssim"].between(-1, 1).all()
    summary = pd.read_csv(run_dir / "summary.csv")
    assert set(summary.method) == {r.REAL, "v2e", "evtween_v5.2_teacher"}
    sanity = pd.read_csv(run_dir / "sanity.csv")
    assert (sanity.status == "missing").sum() == 1
    saved = sorted((toy.tmp / "recon" / "e2vid" / "v2e" / "seq0").glob("*.png"))
    assert [p.stem for p in saved] == ["000003", "000013"]
    assert json.loads((toy.tmp / "recon" / "e2vid" / r.REAL / "seq0" / "recon.json").read_text())["frames"] == [3, 13]
    fig = r.qualitative(run_dir, "seq0", network="firenet")
    assert fig.is_file()
    # resuming skips finished sequences and reproduces the same numbers
    again = r.run(args)
    assert pd.read_csv(again / "per_frame.csv").equals(frames)
