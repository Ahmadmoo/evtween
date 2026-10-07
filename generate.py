import argparse
import numpy as np
import torch
import yaml
from data import Sequence
from model import build_model
from physics import Simulator, resolve

ap = argparse.ArgumentParser(description="RGB video -> continuous-time events (student path, no events needed)")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--seq", required=True, help="folder with frames/*.png and frame_ts.npy")
ap.add_argument("--out", default="events.npz")
ap.add_argument("--sensor", default=None, help="learned | clean | profile.yaml (default: the checkpoint's config)")
ap.add_argument("--set", nargs="*", default=[], help="sensor overrides, e.g. pos_thres=0.3 shot_noise_rate_hz=x0.5")
ap.add_argument("--save-profile", default=None, help="write the final sensor profile to this YAML file")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
model = build_model(ck["cfg"]).to(dev).eval()
model.load_state_dict(ck["model"], strict=False)
S = ck["cfg"]["sensor"]
prof = resolve(model, args.sensor or S.get("profile", "learned"), dict(S.get("set") or {}, **dict(a.split("=") for a in args.set)))
print("sensor:", {k: float(f"{v:.4g}") for k, v in prof.items()})
if args.save_profile:
    yaml.safe_dump(prof, open(args.save_profile, "w"), sort_keys=False)
sim = Simulator(model, prof, int(S["min_steps"]), int(S["max_steps"]), int(S["seed"]))

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
print(f"\nsaved {args.out}")
