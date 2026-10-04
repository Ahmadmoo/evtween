import os
import sys
import numpy as np
from PIL import Image


def make(root, seed, H=96, W=128, n_frames=40, sub=20, fps=100.0, c=0.2, eps=0.01, gamma=2.2):
    # drifting Gaussian blobs + ideal event camera sampled densely in time (smoke-test data only)
    rng = np.random.default_rng(seed)
    n = 40
    cen, vel = rng.uniform([0, 0], [W, H], (n, 2)), rng.normal(0, 0.1, (n, 2))
    amp, sig = rng.uniform(-0.3, 0.5, n), rng.uniform(3, 12, n)
    yy, xx = np.mgrid[0:H, 0:W]
    dt = 1 / (fps * sub)

    def image(k):
        p = cen + vel * k
        d2 = (xx - p[:, 0, None, None]) ** 2 + (yy - p[:, 1, None, None]) ** 2
        return np.clip(0.35 + (amp[:, None, None] * np.exp(-d2 / (2 * sig[:, None, None] ** 2))).sum(0), 0.02, 1)

    os.makedirs(os.path.join(root, "frames"), exist_ok=True)
    ref, ev, ts = np.log(image(0) + eps), [], []
    for k in range(n_frames * sub - sub + 1):
        img = image(k)
        if k % sub == 0:
            Image.fromarray((img ** (1 / gamma) * 255).round().astype(np.uint8)).convert("RGB").save(
                os.path.join(root, "frames", f"{k // sub:06d}.png"))
            ts.append(k * dt)
        L = np.log(img + eps)
        for sign in (1, -1):
            cnt = np.floor(sign * (L - ref) / c).clip(min=0).astype(int)
            y, x = np.nonzero(cnt)
            m = cnt[y, x]
            ref[y, x] += sign * m * c
            t = k * dt - rng.uniform(0, dt, m.sum())
            ev.append(np.stack([t, np.repeat(x, m), np.repeat(y, m), np.full(m.sum(), sign)], 1))
    ev = np.concatenate(ev)
    ev = ev[np.argsort(ev[:, 0], kind="stable")]
    np.save(os.path.join(root, "frame_ts.npy"), np.array(ts))
    for i, (k, dtype) in enumerate(zip("txyp", (np.float64, np.int16, np.int16, np.int8))):
        np.save(os.path.join(root, f"ev_{k}.npy"), ev[:, i].astype(dtype))


out = sys.argv[1] if len(sys.argv) > 1 else "data/toy"
for split, seeds in (("train", range(4)), ("val", range(100, 101))):
    for s in seeds:
        make(os.path.join(out, split, f"seq{s:03d}"), s)
print(f"wrote {out}/train and {out}/val")
