import argparse
import glob
import os
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser(description="per-interval npz events + frames -> canonical layout")
ap.add_argument("--images", required=True, help="folder of frames (png/jpg)")
ap.add_argument("--timestamps", required=True, help="txt with one frame timestamp per line")
ap.add_argument("--events", required=True, help="folder of *.npz event chunks")
ap.add_argument("--out", required=True)
ap.add_argument("--keys", default="x,y,t,p", help="npz keys for x,y,t,p")
ap.add_argument("--xy_scale", type=float, default=1.0, help="divide x,y by this (HS-ERGB: 32)")
ap.add_argument("--t_scale", type=float, default=1.0, help="event time * t_scale = seconds")
ap.add_argument("--ts_scale", type=float, default=1.0, help="frame time * ts_scale = seconds")
a = ap.parse_args()

kx, ky, kt, kp = a.keys.split(",")
chunks = [np.load(f) for f in sorted(glob.glob(os.path.join(a.events, "*.npz")))]
cat = lambda k: np.concatenate([c[k] for c in chunks])
t = cat(kt).astype(np.float64) * a.t_scale
o = np.argsort(t, kind="stable")
os.makedirs(os.path.join(a.out, "frames"), exist_ok=True)
np.save(os.path.join(a.out, "ev_t.npy"), t[o])
np.save(os.path.join(a.out, "ev_x.npy"), np.round(cat(kx)[o] / a.xy_scale).astype(np.int16))
np.save(os.path.join(a.out, "ev_y.npy"), np.round(cat(ky)[o] / a.xy_scale).astype(np.int16))
np.save(os.path.join(a.out, "ev_p.npy"), np.where(cat(kp)[o] > 0, 1, -1).astype(np.int8))

frames = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith((".png", ".jpg", ".jpeg")))
ts = np.loadtxt(a.timestamps).reshape(-1) * a.ts_scale
n = min(len(frames), len(ts))
for i, f in enumerate(frames[:n]):
    Image.open(f).convert("RGB").save(os.path.join(a.out, "frames", f"{i:06d}.png"))
np.save(os.path.join(a.out, "frame_ts.npy"), ts[:n])
print(f"{n} frames, {len(t)} events, frames {ts[0]:.3f}-{ts[n - 1]:.3f}s, events {t[o[0]]:.3f}-{t[o[-1]]:.3f}s")
