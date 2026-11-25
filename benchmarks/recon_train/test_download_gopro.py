"""Tests for download_gopro.py with a local HTTP server (no internet):
    python -m pytest benchmarks/recon_train/test_download_gopro.py -q
"""
import hashlib
import http.server
import io
from pathlib import Path
import sys
import threading
import zipfile

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_gopro as dg  # noqa: E402


def make_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for split, seqs in (("train", ("GOPR0372_07_00", "GOPR0374_11_00")), ("test", ("GOPR0384_11_00",))):
            for seq in seqs:
                for k in range(3):
                    img = io.BytesIO()
                    Image.fromarray(np.full((18, 32, 3), 40 * k, np.uint8)).save(img, format="PNG")
                    z.writestr(f"GOPRO_Large_all/{split}/{seq}/{k:06d}.png", img.getvalue())
    return buf.getvalue()


class RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""
    cut = None  # serve only this many bytes once, to simulate a dropped connection

    def do_GET(self):  # noqa: N802
        data, start = self.payload, 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0])
        body = data[start:]
        if RangeHandler.cut is not None:
            body, RangeHandler.cut = body[:RangeHandler.cut], None
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.connection.close()
            return
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    payload = make_zip()
    RangeHandler.payload = payload
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/GOPRO_Large_all.zip", payload
    httpd.shutdown()


def test_resume_verify_extract(server, tmp_path, monkeypatch, capsys):
    url, payload = server
    monkeypatch.setitem(dg.FILES, "GOPRO_Large_all", dict(url=url, size=len(payload),
                                                          sha256=hashlib.sha256(payload).hexdigest()))
    monkeypatch.setattr(dg.time, "sleep", lambda s: None)
    RangeHandler.cut = len(payload) // 3              # first attempt drops after a third of the file
    dg.main(["--dest", str(tmp_path)])
    out = capsys.readouterr().out
    assert "resuming at" in out and "verified" in out
    train = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.png"))
    assert len(train) == 6 and all("/train/" in p for p in train)       # test split not extracted
    assert "train: 2 folders, 6 frames" in out and "frame size: 32x18" in out
    assert not (tmp_path / "GOPRO_Large_all.zip").exists()                # zip deleted by default
    dg.main(["--dest", str(tmp_path), "--splits", "train", "test", "--keep-zip"])
    assert len(list(tmp_path.rglob("*.png"))) == 9 and (tmp_path / "GOPRO_Large_all.zip").is_file()


def test_bad_checksum_is_rejected(server, tmp_path, monkeypatch):
    url, payload = server
    monkeypatch.setitem(dg.FILES, "GOPRO_Large_all", dict(url=url, size=len(payload), sha256="0" * 64))
    with pytest.raises(ValueError, match="SHA-256"):
        dg.main(["--dest", str(tmp_path), "--no-extract"])


def test_split_of():
    assert dg.split_of("GOPRO_Large_all/train/GOPR0372_07_00/000001.png") == "train"
    assert dg.split_of("GOPRO_Large_all/test/x/1.png") == "test"
    assert dg.split_of("GOPRO_Large_all/readme.txt") is None
