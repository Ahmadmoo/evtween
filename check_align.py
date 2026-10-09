import glob, os, sys
import numpy as np
from PIL import Image

# python check_align.py data/eds [pairs]: where do the events sit relative to the frames? For each flip of the event map,
# the cross-correlation of |events| with |log change| between neighbouring frames over all shifts (both maps blurred)
root, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20
rng = np.random.default_rng(0)
seqs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(root, "*", "*", "frame_ts.npy")))
lum = lambda f: np.log(np.asarray(Image.open(f).convert("L"), dtype=np.float64) / 255 + 0.01)
FLIPS = {"none": lambda m: m, "flip x": lambda m: m[:, ::-1], "flip y": lambda m: m[::-1], "flip x+y": lambda m: m[::-1, ::-1]}
acc = {}
for _ in range(n):
    seq = seqs[rng.integers(len(seqs))]
    files, ts = sorted(glob.glob(os.path.join(seq, "frames", "*.png"))), np.load(os.path.join(seq, "frame_ts.npy"))
    k = int(rng.integers(0, len(files) - 1))
    A = np.abs(lum(files[k + 1]) - lum(files[k]))
    H, W = A.shape
    t = np.load(os.path.join(seq, "ev_t.npy"), mmap_mode="r")
    i, j = np.searchsorted(t, [ts[k], ts[k + 1]])
    x, y = (np.asarray(np.load(os.path.join(seq, f"ev_{q}.npy"), mmap_mode="r")[i:j]).astype(np.int64) for q in "xy")
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    C = np.bincount(y[ok] * W + x[ok], minlength=H * W).reshape(H, W).astype(np.float64)
    C = np.minimum(C, np.percentile(C, 99.9))
    fy, fx = np.fft.fftfreq(H)[:, None], np.fft.rfftfreq(W)[None]
    blur = np.exp(-2 * (np.pi * 2.0) ** 2 * (fy ** 2 + fx ** 2))  # gaussian, sigma 2 px
    Fa = np.fft.rfft2(A - A.mean()) * blur
    na = np.sqrt((np.fft.irfft2(Fa, (H, W)) ** 2).sum())
    for name, f in FLIPS.items():
        Fc = np.fft.rfft2(f(C) - C.mean()) * blur
        r = np.fft.irfft2(Fa * np.conj(Fc), (H, W)) / (na * np.sqrt((np.fft.irfft2(Fc, (H, W)) ** 2).sum()) + 1e-12)
        acc[name] = acc.get(name, 0) + r / n
H, W = next(iter(acc.values())).shape
print(f"{root}: {n} frame pairs, correlation of |events| with |log change| (1 = perfect)")
for name, r in acc.items():
    dy, dx = np.unravel_index(r.argmax(), r.shape)
    dy, dx = (dy + H // 2) % H - H // 2, (dx + W // 2) % W - W // 2
    print(f"  {name:9s} best {r.max():.3f} with the events moved by (dy, dx) = ({dy:+d}, {dx:+d}); at (0, 0): {r[0, 0]:.3f}")
