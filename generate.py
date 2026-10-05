import argparse
import numpy as np
import torch
from data import Sequence
from model import build_model
from physics import Simulator

ap = argparse.ArgumentParser(description="RGB video -> continuous-time events (student path, no events needed)")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--seq", required=True, help="folder with frames/*.png and frame_ts.npy")
ap.add_argument("--out", default="events.npz")
ap.add_argument("--set", nargs="*", default=[], help="sensor overrides, e.g. noise_rate=0 mismatch=0.05")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
model = build_model(ck["cfg"]).to(dev).eval()
model.load_state_dict(ck["model"], strict=False)
sensor = dict(ck["cfg"]["sensor"], **{k: float(v) for k, v in (s.split("=") for s in args.set)})
sensor = {k: int(v) if k in ("min_steps", "max_steps", "seed") else v for k, v in sensor.items()}
sim = Simulator(model, **sensor)

seq, c, parts = Sequence(args.seq), ck["cfg"]["data"]["context"], []
N, ts = len(seq), seq.ts
with torch.no_grad():
    for i in range(N - 1):
        idx = [min(max(q, 0), N - 1) for q in [*range(i - c + 1, i + 1), *range(i + 1, i + 1 + c)]]
        ctx = torch.stack([seq.frame(q) for q in idx])[None].to(dev)
        tau = torch.tensor([(ts[q] - ts[i]) / (ts[i + 1] - ts[i]) for q in idx], device=dev)[None].float()
        p = model.prepare(ctx[:, c - 1], ctx[:, c])
        z = model.predict(ctx, tau, torch.tensor([ts[i + 1] - ts[i]], device=dev).float())
        parts.append(sim.run(model.decode(p, z), ts[i], ts[i + 1]))
        print(f"\r{i + 1}/{N - 1} frames, {sum(len(q['t']) for q in parts)} events", end="", flush=True)

np.savez(args.out, **{k: np.concatenate([q[k] for q in parts]) for k in "txyp"})
print(f"\nsaved {args.out}  (c_on {model.c.item():.3f}, c_off {model.c.item() * model.r.item():.3f})")
