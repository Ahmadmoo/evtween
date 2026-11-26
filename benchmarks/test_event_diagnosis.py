"""Tests for event_diagnosis.py (no torch needed):
    python -m pytest benchmarks/test_event_diagnosis.py -q

The toy sequence has a textured square moving over a gradient. "real" events are ideal threshold crossings;
v2e: the same rule with another threshold and time jitter; evtween_v5.2_teacher: the real events with random
polarity (same pixels, same times, same counts per pixel). The diagnosis must tell the last two apart.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import event_diagnosis as d  # noqa: E402

pytest.importorskip("matplotlib")

H, W, N, DT = 48, 64, 40, 0.01
DATASET, SPLIT, SEQ = "TOY", "test", "moving_square"
SHUFFLED = "evtween_v5.2_teacher"


def scene(k):
    yy, xx = np.mgrid[0:H, 0:W]
    img = 0.25 + 0.3 * xx / W
    x0 = 8 + 1.0 * k
    inside = (xx >= x0) & (xx < x0 + 16) & (yy >= 14) & (yy < 34)
    checker = ((np.floor((xx - x0) / 4) + np.floor((yy - 14) / 4)) % 2).astype(float)
    return np.where(inside, 0.1 + 0.8 * checker, img)


def crossings(logs, ts, c, rng, jitter=0.0):
    """Ideal integrate-and-fire on linearly interpolated log brightness: each event moves the reference by c."""
    ref = logs[0].copy()
    t, x, y, p = [], [], [], []
    for j in range(len(logs) - 1):
        a, b = logs[j], logs[j + 1]
        for yy in range(H):
            for xx in range(W):
                L0, L1, r = a[yy, xx], b[yy, xx], ref[yy, xx]
                while abs(L1 - r) >= c:
                    s = 1 if L1 > r else -1
                    r += s * c
                    f = (r - L0) / (L1 - L0) if L1 != L0 else 0.5
                    t.append(ts[j] + np.clip(f, 0, 1) * (ts[j + 1] - ts[j]))
                    x.append(xx), y.append(yy), p.append(s)
                ref[yy, xx] = r
    t = np.asarray(t) + rng.normal(0, jitter, len(t))
    o = np.argsort(t, kind="stable")
    return dict(t=t[o], x=np.asarray(x, np.int32)[o], y=np.asarray(y, np.int32)[o], p=np.asarray(p, np.int8)[o])


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("diag")
    rng = np.random.default_rng(0)
    data, evt = tmp / "data", tmp / "evt"
    src = data / DATASET / SPLIT / SEQ
    (src / "frames").mkdir(parents=True)
    ts = 50.0 + np.arange(N) * DT
    logs = []
    for k in range(N):
        img = scene(k)
        Image.fromarray((np.repeat(img[..., None], 3, 2) * 255).round().astype(np.uint8)).save(
            src / "frames" / f"{k:06d}.png")
        logs.append(np.log(np.round(img * 255) / 255 + 0.01))
    np.save(src / "frame_ts.npy", ts)
    real = crossings(logs, ts, 0.15, rng)
    for key in "txyp":
        np.save(src / f"ev_{key}.npy", real[key])
    _, _, times, meta, _ = d.r.v2.sequence_info(src)
    t0 = ts[0]
    sims = {"v2e": crossings(logs, ts, 0.17, rng, jitter=0.0005),
            SHUFFLED: dict(real, p=rng.choice(np.array([-1, 1], np.int8), len(real["t"])))}
    for m, ev in sims.items():
        out = evt / m / SEQ
        out.mkdir(parents=True)
        (out / "input.json").write_text(json.dumps(meta))
        keep = (ev["t"] - t0 >= 0) & (ev["t"] - t0 < meta["duration"])
        np.savez(out / "events.npz", t=ev["t"][keep] - t0, x=ev["x"][keep], y=ev["y"][keep], p=ev["p"][keep])
    for s, value in (("real", 0.3), ("v2e", 0.5), (SHUFFLED, 0.7)):  # what recon_eval.py would have saved
        folder = tmp / "recon" / "e2vid" / s / SEQ
        folder.mkdir(parents=True)
        for k in (12, 30):
            Image.fromarray(np.full((H, W), int(255 * value), np.uint8)).save(folder / f"{k:06d}.png")
    args = d.parser().parse_args([DATASET, SEQ, "--events-root", str(evt), "--data-root", str(data),
                                  "--recon-root", str(tmp / "recon"), "--methods", "v2e", SHUFFLED,
                                  "--frame", "30", "--gaps", "1", "5", "10", "--pools", "1", "2", "4", "8",
                                  "--windows", "8", "--pixels", "4", "--out", str(tmp / "out")])
    out = d.run(args)
    rows = list(__import__("csv").DictReader(open(out / "agreement_summary.csv")))
    summary = {(int(r["gaps"]), r["stream"], int(r["pool_px"])): r for r in rows}
    return out, summary


def value(summary, stream, key, pool=1, gaps=5):
    return float(summary[(gaps, stream, pool)][key])


def test_pool_and_corr():
    m = np.arange(16.0).reshape(4, 4)
    assert d.pool(m, 2).tolist() == [[10, 18], [42, 50]]
    assert d.pool(np.ones((5, 7)), 2).shape == (2, 3)
    assert d.corr([1, 2, 3], [2, 4, 6]) == pytest.approx(1)
    assert np.isnan(d.corr([1, 1, 1], [1, 2, 3]))
    assert d.end_frames(10, 3, 4) == [3, 5, 7, 9]
    assert d.end_frames(3, 5, 4) == []


def test_same_counts_random_polarity_is_caught(toy):
    _, s = toy
    # same pixels and times: the counts agree perfectly...
    assert value(s, SHUFFLED, "total_r") > 0.999
    # ...but ON - OFF no longer follows the real events or the camera
    assert value(s, SHUFFLED, "net_r") < 0.3
    assert value(s, SHUFFLED, "camera_r") < 0.3
    # an honest simulator with another threshold and jitter keeps both
    assert value(s, "v2e", "net_r") > 0.8
    assert value(s, "v2e", "camera_r") > 0.8
    assert value(s, "real", "camera_r") > 0.8


def test_outputs_and_threshold_fit(toy):
    out, _ = toy
    for name in ("counts_vs_net.png", "net_over_windows.png", "agreement_vs_pooling.png", "pixel_traces.png",
                 "agreement.csv", "agreement_summary.csv", "thresholds.csv", "summary.txt", "config.json"):
        assert (out / name).stat().st_size > 0, name
    fits = {r["stream"]: r for r in __import__("csv").DictReader(open(out / "thresholds.csv"))}
    assert float(fits["real"]["fit_c_on"]) == pytest.approx(0.15, rel=0.15)
    assert float(fits["v2e"]["fit_c_off"]) == pytest.approx(0.17, rel=0.15)
    assert float(fits[SHUFFLED]["fit_r2"]) < 0.2
    # step sizes ignore polarity: same counts as real, so about the same steps
    assert float(fits[SHUFFLED]["step_c_on"]) == pytest.approx(float(fits["real"]["step_c_on"]), rel=0.3)
    assert float(fits["real"]["step_c_on"]) == pytest.approx(0.15, rel=0.3)
    config = json.loads((out / "config.json").read_text())
    assert config["frame"] == 30 and config["streams"] == ["real", "v2e", SHUFFLED]
    assert len(config["pixels"]) == 4
    assert config["recon_shown"] is True


def test_frame_too_early_is_rejected(toy, tmp_path):
    out, _ = toy
    cfg = json.loads((out / "config.json").read_text())
    args = d.parser().parse_args([DATASET, SEQ, "--events-root", cfg["events_root"], "--data-root", cfg["data_root"],
                                  "--methods", "v2e", "--frame", "3", "--out", str(tmp_path / "x")])
    with pytest.raises(ValueError):
        d.run(args)
