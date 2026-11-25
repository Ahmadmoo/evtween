"""Tests for prepare_gopro.py and check_training_events.py (no torch, no GPU):
    python -m pytest benchmarks/recon_train/test_phase1a.py -q
"""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import prepare_gopro as pg  # noqa: E402
import check_training_events as ce  # noqa: E402
import recon_eval as r  # noqa: E402


@pytest.fixture
def gopro(tmp_path):
    """Fake GoPro download: 2 sequences of 128x72 RGB frames (16:9), a gradient that moves right."""
    src = tmp_path / "dl" / "train"
    for name, n in (("GOPR0001_00_00", 12), ("GOPR0002_00_00", 9)):
        d = src / name
        d.mkdir(parents=True)
        for k in range(n):
            x = np.arange(128)[None, :, None]
            img = np.broadcast_to((((x + 3 * k) % 128) * 2).astype(np.uint8), (72, 128, 3)).copy()
            img[..., 1] //= 2
            Image.fromarray(img).save(d / f"{k + 1:06d}.png")
    return tmp_path


def test_crop_box():
    assert pg.crop_box(1280, 720, "center", 4 / 3) == (160, 0, 1120, 720)
    assert pg.crop_box(1280, 720, "none", 4 / 3) == (0, 0, 1280, 720)
    assert pg.crop_box(640, 640, "center", 4 / 3) == (0, 80, 640, 560)


def test_prepare_layout_and_resume(gopro, capsys):
    out = gopro / "data" / "gopro" / "train"
    pg.main(["--src", str(gopro / "dl"), "--out", str(out), "--size", "24", "18", "--workers", "2"])
    seq = out / "GOPR0001_00_00"
    frames = sorted((seq / "frames").glob("*.png"))
    assert len(frames) == 12 and frames[0].name == "000000.png"
    ts = np.load(seq / "frame_ts.npy")
    assert np.allclose(ts, np.arange(12) / 240.0)
    with Image.open(frames[3]) as im:
        a = np.asarray(im)
    assert im.size == (24, 18) and im.mode == "RGB"
    assert (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all()      # gray in all channels
    meta = json.loads((seq / "prepare.json").read_text())
    assert meta["crop_box"] == [16, 0, 112, 72]                                    # 4:3 centre crop of 128x72
    assert not any(p.name.startswith(".") for p in out.iterdir())                  # no temp folders in <out>
    # the generator accepts it as a dataset sequence
    sys.path.insert(0, str(HERE.parent / "event_generation"))
    import generate_event_streams_v2 as v2
    files, ts2, times, meta2, size = v2.sequence_info(seq)
    assert len(files) == 12 and size == (24, 18) and meta2["uniform_timestamps"]
    capsys.readouterr()
    pg.main(["--src", str(gopro / "dl"), "--out", str(out), "--size", "24", "18", "--workers", "2"])
    assert capsys.readouterr().out.count("already prepared") == 2                 # resume: nothing redone
    out60 = gopro / "data" / "gopro60" / "train"
    pg.main(["--src", str(gopro / "dl"), "--out", str(out60), "--size", "24", "18", "--stride", "4", "--workers", "1"])
    assert np.allclose(np.load(out60 / "GOPR0001_00_00" / "frame_ts.npy"), np.arange(3) * 4 / 240.0)


def test_check_training_events(gopro, tmp_path):
    out = gopro / "data" / "gopro" / "train"
    pg.main(["--src", str(gopro / "dl"), "--out", str(out), "--size", "24", "18", "--workers", "1"])
    root = tmp_path / "evt"
    rng = np.random.default_rng(0)
    for m, n in (("v2e", 3000), ("evtween_v8.2_student", 1500)):
        for seq in ("GOPR0001_00_00", "GOPR0002_00_00"):
            d = root / m / seq
            d.mkdir(parents=True)
            frames = sorted((out / seq / "frames").glob("*.png"))
            dur = (len(frames) - 1) / 240.0
            t = np.sort(rng.uniform(0, dur * 0.999, n))
            np.savez(d / "events.npz", t=t, x=rng.integers(0, 24, n).astype(np.int32),
                     y=rng.integers(0, 18, n).astype(np.int32), p=rng.choice(np.array([-1, 1], np.int8), n))
            (d / "input.json").write_text(json.dumps(dict(source=str(out / seq), width=24, height=18)))
    report = ce.main(["gopro", "--events-root", str(root), "--out", str(tmp_path / "rep"), "--spacings", "1", "6",
                      "--samples", "3", "--sample-spacing", "6", "--tile-width", "48"])
    import pandas as pd
    per = pd.read_csv(report / "per_method.csv", index_col=0)
    assert set(per.index) == {"v2e", "evtween_v8.2_student"}
    assert per.loc["v2e", "events"] == 6000 and per.loc["v2e", "rate_mev_s"] > per.loc["evtween_v8.2_student", "rate_mev_s"]
    sanity = pd.read_csv(report / "sanity.csv")
    assert {"win1_median", "win6_median", "win6_empty_pct"} <= set(sanity.columns)
    assert len(list((report / "samples").glob("*.png"))) == 3
