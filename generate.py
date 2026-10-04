import argparse
import numpy as np
import torch
from data import Sequence
from model import build_model
from physics import Simulator

ap = argparse.ArgumentParser(description="RGB frames -> continuous-time events")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--seq", required=True, help="folder with frames/*.png and frame_ts.npy (events not needed)")
ap.add_argument("--out", default="events.npz")
ap.add_argument("--set", nargs="*", default=[], help="sensor overrides, e.g. noise_rate=0 mismatch=0.05")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
model = build_model(ck["cfg"]).to(dev).eval()
model.load_state_dict(ck["model"])
sensor = dict(ck["cfg"]["sensor"], **{k: float(v) for k, v in (s.split("=") for s in args.set)})
sensor = {k: int(v) if k in ("min_steps", "max_steps", "seed") else v for k, v in sensor.items()}
sim = Simulator(model, **sensor)

seq, parts = Sequence(args.seq), []
with torch.no_grad():
    for i in range(len(seq) - 1):
        s = model(seq.frame(i)[None].to(dev), seq.frame(i + 1)[None].to(dev))
        parts.append(sim.run(s, seq.ts[i], seq.ts[i + 1]))
        print(f"\r{i + 1}/{len(seq) - 1} frames, {sum(len(p['t']) for p in parts)} events", end="", flush=True)

np.savez(args.out, **{k: np.concatenate([p[k] for p in parts]) for k in "txyp"})
print(f"\nsaved {args.out}  (c_on {model.c.item():.3f}, c_off {model.c.item() * model.r.item():.3f})")
