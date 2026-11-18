import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "evis-core": ("spikelab-jhu/isaac-sim-event-camera-plugin", "d6fe47923fad09c4bfa2fd2cb13f82520114cfae", ["dvs_gen/dvs", "dvs_gen/warp"]),
    "fracevent": ("bonaparte233/FracEvent", "1fe6ee7e8e7b4cbc3d2526aa80c033570bb2637f", ["core", "configs"]),
    "iebcs": ("neuromorphicsystems/IEBCS", "8fdab5c3d078435d091fa8a94ad657e1fedf0c59", ["src", "data"]),
    "v2e": ("SensorsINI/v2e", "23f6d5b18aedee5763a1cae7adf43112cf686e20", ["v2ecore"]),
    "v2ce": ("ucsd-hdsi-dvs/V2CE-Toolbox", "67f1624f6539eec7ee7df2a73f95198b3d546969", ["scripts"]),
    "dvs-voltmeter": ("Lynn0306/DVS-Voltmeter", "9d86353ecfaaa1af82b9eb66b162f0bf80cd4861", ["src"]),
}
WEIGHTS = {
    "v2e": ("SuperSloMo39.ckpt", "1ETID_4xqLpRBrRo1aOT7Yphs3QqWR_fx", "d4d3070431eff774a1c27038930121ffeae59ede3a7d184118c539b443ea27cb"),
    "v2ce": ("v2ce_3d.pt", "1-aC6CTGZgAZk3snANZ46FAGNkPzu_Scw", "97920e0c3f3c1df55f12800c166bf1de11466f7546bdb7b40261c5a14cddef26"),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name):
    repo, revision, paths = SOURCES[name]
    dest = ROOT / "upstream" / name
    dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            checkout = Path(tmp) / "repo"
            def git(*args):
                subprocess.run(["git", "-C", str(checkout), *args], check=True)
            checkout.mkdir()
            git("init", "-q")
            git("remote", "add", "origin", f"https://github.com/{repo}.git")
            git("sparse-checkout", "set", "--cone", *paths)
            git("fetch", "-q", "--depth=1", "--filter=blob:none", "origin", revision)
            git("checkout", "-q", "--detach", "FETCH_HEAD")
            checkout.rename(dest)
    actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise ValueError(f"{dest} is at {actual}, expected {revision}")
    if name in WEIGHTS:
        import gdown
        filename, file_id, sha = WEIGHTS[name]
        path = ROOT / "weights" / filename
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
                new = Path(tmp) / filename
                gdown.download(id=file_id, output=str(new))
                if not new.is_file() or digest(new) != sha:
                    raise ValueError(f"Invalid download for {filename}. Get the authors' checkpoint from https://drive.google.com/file/d/{file_id}/view")
                new.rename(path)
        elif digest(path) != sha:
            raise ValueError(f"Unexpected SHA-256 for {path}")
    print(f"{name} ready ({revision[:8]})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch the pinned authors' implementations and weights.")
    ap.add_argument("models", nargs="*", help="default: all six")
    args = ap.parse_args()
    for name in args.models or SOURCES:
        if name not in SOURCES:
            ap.error(f"unknown model {name}; choose from {', '.join(SOURCES)}")
        fetch(name)
