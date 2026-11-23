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


def test_numpy_voxel_follows_evreal():
    rng = np.random.default_rng(1)
    ev = make_events(rng, 500, 1.0)
    grid = r.voxel_grid_numpy(ev["t"], ev["x"], ev["y"], ev["p"].astype(float), 5, H, W)
    assert np.isclose(grid.sum(), ev["p"].sum())  # bilinear weights of each event sum to 1
    # first event lands fully in bin 0, last event fully in the last bin
    one = r.voxel_grid_numpy(np.array([2.0, 3.0]), np.array([1, 2]), np.array([0, 0]), np.array([1.0, -1.0]), 5, 2, 4)
    assert one[0, 0, 1] == 1 and one[4, 0, 2] == -1 and np.count_nonzero(one) == 2
    # identical timestamps: spread by rank (EVREAL linspace)
    same = r.voxel_grid_numpy(np.zeros(3), np.array([0, 1, 2]), np.zeros(3, int), np.ones(3), 5, 1, 3)
    assert same[0, 0, 0] == 1 and same[2, 0, 1] == 1 and same[4, 0, 2] == 1
    norm = r.normalize_numpy(grid)
    nz = grid != 0
    assert np.isclose(norm[nz].mean(), 0, atol=1e-9) and np.isclose(norm[nz].std(ddof=0), 1)


def test_crop_params_match_evreal():
    pad, crop = r.crop_params(618, 773, 3)
    assert pad == (2, 1, 3, 3) and crop == (3, 621, 2, 775)
    assert crop[1] - crop[0] == 618 and crop[3] - crop[2] == 773
    assert r.crop_params(480, 640, 3) == ((0, 0, 0, 0), (0, 480, 0, 640))


def test_setup_checks(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        r.ensure_evreal(tmp_path / "none", auto=False)
    root = tmp_path / "evreal"
    rel = r.CHECKPOINTS["firenet"][0]
    (root / rel).parent.mkdir(parents=True)
    (root / rel).write_bytes(b"not the real weights")
    args = SimpleNamespace(evreal_root=str(root), firenet_ckpt=None, e2vid_ckpt=None)
    monkeypatch.delenv("RECON_FIRENET_CKPT", raising=False)
    with pytest.raises(ValueError, match="does not match"):
        r.checkpoint_path("firenet", args)
    args.firenet_ckpt = str(root / rel)
    path, sha, verified = r.checkpoint_path("firenet", args)
    assert path == root / rel and not verified and len(sha) == 64
    monkeypatch.setattr(r, "PACKAGES", {"json": "json"})
    assert r.ensure_packages(auto=False) == []
    monkeypatch.setattr(r, "PACKAGES", {"surely_missing_pkg_xyz": "surely-missing-pkg-xyz"})
    with pytest.raises(ImportError):
        r.ensure_packages(auto=False)


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
    edges = np.array([0.0, 0.03, 0.05, 0.11, 0.2, 0.3, 0.4])
    ev = make_events(rng, 4000, 0.2 - 1e-9)
    # window 4 [0.2, 0.3): one event; window 5 [0.3, 0.4): three events at the same time; window 3 empty-ish
    extra_t = np.array([0.25, 0.35, 0.35, 0.35])
    ev = {k: np.concatenate([ev[k], v]) for k, v in dict(
        t=extra_t, x=np.array([1, 2, 3, 4], np.int32), y=np.array([1, 1, 1, 1], np.int32),
        p=np.array([1, -1, 1, 1], np.int8)).items()}
    ev = r._canonical(ev["t"], ev["x"], ev["y"], ev["p"])
    _, ev["cuts"] = r._stats(ev, edges)
    vox = r.voxelize(ev, 1, 6, 5, H, W, "cpu").numpy()
    for j in range(1, 6):
        a, b = ev["cuts"][j], ev["cuts"][j + 1]
        ref = r.normalize_numpy(r.voxel_grid_numpy(ev["t"][a:b], ev["x"][a:b], ev["y"][a:b],
                                                   ev["p"][a:b].astype(float), 5, H, W))
        assert np.allclose(vox[j - 1], ref, atol=1e-4), j


@needs_torch
def test_ssim_matches_skimage():
    from skimage.metrics import structural_similarity
    rng = np.random.default_rng(4)
    a = rng.random((40, 50)).astype(np.float32)
    b = np.clip(a + rng.normal(0, 0.1, a.shape), 0, 1).astype(np.float32)
    sc = r.Scorer("cpu", use_lpips=False)
    ours = sc.ssim(torch.from_numpy(a)[None, None], torch.from_numpy(b)[None, None]).item()
    ref = structural_similarity(a, b, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                use_sample_covariance=False)
    assert abs(ours - ref) < 1e-4


FAKE_EVREAL = '''
import torch


class _Toy(torch.nn.Module):
    """Recurrent toy network with EVREAL's interface: reset_states(), forward(x) -> {'image': ...}."""
    def __init__(self, kwargs):
        super().__init__()
        self.num_encoders = int(kwargs.get("num_encoders", 3))
        self.final_activation = kwargs["final_activation"]
        self.conv = torch.nn.Conv2d(int(kwargs["num_bins"]), 1, 3, padding=1)
        self.reset_states()

    def reset_states(self):
        self.state = None

    def forward(self, x):
        assert x.shape[-1] % 2 ** self.num_encoders == 0 and x.shape[-2] % 2 ** self.num_encoders == 0
        y = self.conv(x)
        if self.state is not None:
            y = y + 0.5 * self.state
        self.state = y
        return {"image": torch.sigmoid(y) if self.final_activation == "sigmoid" else y}


class E2VIDRecurrent(_Toy):
    pass


class FireNet_legacy(_Toy):
    pass
'''


@needs_torch
def test_end_to_end_cpu(toy, monkeypatch):
    root = toy.tmp / "evreal"
    (root / "model").mkdir(parents=True)
    (root / "model" / "__init__.py").write_text(FAKE_EVREAL)
    with r.evreal_imports(root):
        import model as fake
        e2 = fake.E2VIDRecurrent(dict(num_bins=5, num_encoders=3, final_activation="sigmoid"))
        fn = fake.FireNet_legacy(dict(num_bins=5, num_encoders=2, final_activation=""))
    torch.save(dict(arch="E2VIDRecurrent", model=dict(num_bins=5, num_encoders=3), state_dict=e2.state_dict()),
               toy.tmp / "e2vid.pth")
    torch.save(dict(arch="FireNet", config=dict(model=dict(num_bins=5, num_encoders=2)), state_dict=fn.state_dict()),
               toy.tmp / "firenet.pth")
    monkeypatch.setattr(r, "PACKAGES", {})
    args = toy.args
    args.evreal_root, args.auto_setup = str(root), False
    args.e2vid_ckpt, args.firenet_ckpt = str(toy.tmp / "e2vid.pth"), str(toy.tmp / "firenet.pth")
    run_dir = r.run(args)
    import pandas as pd
    frames = pd.read_csv(run_dir / "per_frame.csv")
    # 2 networks x frames 3..13 x (seq0: 3 sources, seq1: 2 sources)
    assert len(frames) == 2 * 11 * (3 + 2)
    assert frames.loc[frames.method == r.REAL, "agreement_mse"].isna().all()
    assert frames.loc[frames.method != r.REAL, "agreement_mse"].notna().all()
    assert frames["gt_ssim"].between(-1, 1).all()
    assert frames["gt_mse"].between(0, 1).all()
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["norm"] == "evreal" and not cfg["checkpoints"]["e2vid"]["verified_against_pin"]
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
