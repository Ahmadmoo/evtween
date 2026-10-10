import glob, os, sys
import numpy as np
from PIL import Image

# python check_align.py data/eds [pairs] [scales, e.g. 1.2,1.4,1.6]: where do the events sit relative to the frames? For each flip and scale of the event
# coordinates (about the image center), the cross-correlation of |events| with |log change| between neighbouring frames over
# all shifts (both maps blurred). Maps are cut to one central size (at most 480x640) so sequences of any size can be averaged
root, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20
rng = np.random.default_rng(0)
seqs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(root, "*", "*", "frame_ts.npy")))
lum = lambda f: np.log(np.asarray(Image.open(f).convert("L"), dtype=np.float64) / 255 + 0.01)
SCALES = [float(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.8, 0.9, 1.0, 1.1, 1.25]
TRANSFORMS = [(fl, s) for fl in ("none", "flip x", "flip y", "flip x+y") for s in SCALES]
sizes = [Image.open(sorted(glob.glob(os.path.join(q, "frames", "*.png")))[0]).size for q in seqs]
H, W = min(480, min(h for _, h in sizes)), min(640, min(w for w, _ in sizes))  # one map size for all pairs
acc = {}
for _ in range(n):
    seq = seqs[rng.integers(len(seqs))]
    files, ts = sorted(glob.glob(os.path.join(seq, "frames", "*.png"))), np.load(os.path.join(seq, "frame_ts.npy"))
    k = int(rng.integers(0, len(files) - 1))
    A = np.abs(lum(files[k + 1]) - lum(files[k]))
    H0, W0 = A.shape
    y0, x0 = (H0 - H) // 2, (W0 - W) // 2
    A = A[y0:y0 + H, x0:x0 + W]
    t = np.load(os.path.join(seq, "ev_t.npy"), mmap_mode="r")
    i, j = np.searchsorted(t, [ts[k], ts[k + 1]])
    x, y = (np.asarray(np.load(os.path.join(seq, f"ev_{q}.npy"), mmap_mode="r")[i:j]).astype(np.float64) for q in "xy")
    fy, fx = np.fft.fftfreq(H)[:, None], np.fft.rfftfreq(W)[None]
    blur = np.exp(-2 * (np.pi * 2.0) ** 2 * (fy ** 2 + fx ** 2))  # gaussian, sigma 2 px
    Fa = np.fft.rfft2(A - A.mean()) * blur
    na = np.sqrt((np.fft.irfft2(Fa, (H, W)) ** 2).sum())
    for fl, s in TRANSFORMS:
        u = (W0 - 1 - x if "x" in fl[4:] else x)
        v = (H0 - 1 - y if "y" in fl[4:] else y)
        u, v = np.round((u - W0 / 2) * s + W0 / 2).astype(np.int64) - x0, np.round((v - H0 / 2) * s + H0 / 2).astype(np.int64) - y0
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        C = np.bincount(v[ok] * W + u[ok], minlength=H * W).reshape(H, W).astype(np.float64)
        C = np.minimum(C, np.percentile(C, 99.9))
        Fc = np.fft.rfft2(C - C.mean()) * blur
        r = np.fft.irfft2(Fa * np.conj(Fc), (H, W)) / (na * np.sqrt((np.fft.irfft2(Fc, (H, W)) ** 2).sum()) + 1e-12)
        acc[fl, s] = acc.get((fl, s), 0) + r / n
print(f"{root}: {n} frame pairs, correlation of |events| with |log change| (1 = perfect); best first")
H, W = next(iter(acc.values())).shape
for (fl, s), r in sorted(acc.items(), key=lambda kv: -kv[1].max())[:8]:
    dy, dx = np.unravel_index(r.argmax(), r.shape)
    dy, dx = (dy + H // 2) % H - H // 2, (dx + W // 2) % W - W // 2
    print(f"  {fl:9s} scale {s:4.2f}: best {r.max():.3f} with the events moved by (dy, dx) = ({dy:+d}, {dx:+d}); at (0, 0): {r[0, 0]:.3f}")
