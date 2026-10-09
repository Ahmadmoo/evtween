import argparse
import glob
import os
import numpy as np
from PIL import Image

# python check_data.py data/hqevfi data/bsergb data/eds data/ced [--pairs 40] [--full]
# 1. every sequence: frames (count, size, links), frame times (order, fps, drops), events (lengths, types, order, pixel range,
#    polarity, events per frame interval, cover of the frames, rate). Problems are marked with "!".
# 2. per dataset, without the model: do the events explain the frames? For random neighbouring frames k, k+1 the net event
#    count per pixel in [t_k + d, t_k+1 + d] is correlated with the log brightness change L_k+1 - L_k. The best d is the time
#    offset of the frames (d > 0: a frame shows the scene d later than its stamp; it moves the pattern by speed * d, so it
#    needs motion), the best pixel shift whether frames and events are aligned, and gamma 1 vs 2.2 which brightness curve
#    the events follow (PNG value^gamma = linear light).
STEP = 20_000_000
D_MS = np.arange(-15, 16)
SHIFTS = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)]
ap = argparse.ArgumentParser(description="check converted datasets: integrity of every sequence + event / frame alignment")
ap.add_argument("roots", nargs="+", help="converted datasets (each holds train/ and val/ or test/)")
ap.add_argument("--pairs", type=int, default=60, help="frame pairs per dataset for the alignment check (0: skip)")
ap.add_argument("--full", action="store_true", help="read every event (slow for 1e9-event sequences); default: 12 windows of 1M")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
RGB = {"r": 0, "g": 1, "b": 2}


def check(seq):
    files, ts = sorted(glob.glob(os.path.join(seq, "frames", "*.png"))), np.load(os.path.join(seq, "frame_ts.npy"))
    ev = {k: np.load(os.path.join(seq, f"ev_{k}.npy"), mmap_mode="r") for k in "txyp"}
    n, dt, lost = len(ev["t"]), np.diff(ts), sum(not os.path.exists(f) for f in files)
    sizes = {Image.open(files[i]).size for i in {0, len(files) // 2, len(files) - 1} if os.path.exists(files[i])}
    W, H = max(sizes)
    bad = [f"{len(files)} frames, {len(ts)} times"] * (len(files) != len(ts)) + [f"{lost} frame files missing"] * bool(lost) \
        + [f"frame sizes {sorted(sizes)}"] * (len(sizes) > 1) + ["frame times not increasing"] * bool((dt <= 0).any()) \
        + [f"event lengths {[len(v) for v in ev.values()]}"] * (len({len(v) for v in ev.values()}) > 1) \
        + [f"types {[str(v.dtype) for v in ev.values()]}"] * ([str(v.dtype) for v in ev.values()] != ["float64", "int16", "int16", "int8"])
    span = [(i, min(i + STEP, n)) for i in range(0, n, STEP)] if a.full or n <= 12_000_000 else \
        [(int(i), int(i) + 1_000_000) for i in np.linspace(0, n - 1_000_000, 12)]
    order, outside, pol, on, seen = True, 0, 0, 0, 0
    for i, j in span:
        t, x, y, p = (np.asarray(ev[k][i:j]) for k in "txyp")
        order = order and bool(np.all(np.diff(t) >= 0)) and (i == 0 or ev["t"][i - 1] <= t[0])
        outside += int(((x < 0) | (x >= W) | (y < 0) | (y >= H)).sum())
        pol, on, seen = pol + int(((p != 1) & (p != -1)).sum()), on + int((p > 0).sum()), seen + len(p)
    per = np.diff(np.searchsorted(ev["t"], ts))  # events per frame interval
    bad += ["events not sorted by time"] * (not order) + [f"{outside} events outside {W}x{H}"] * bool(outside) \
        + [f"{pol} polarities not +-1"] * bool(pol) + [f"{(per == 0).sum()} frame intervals without events"] * bool((per == 0).any()) \
        + ["events start after the 2nd frame"] * bool(n and ev["t"][0] > ts[1]) + ["events end before the 2nd last frame"] * bool(n and ev["t"][-1] < ts[-2])
    row = dict(seq=os.path.relpath(seq, os.path.dirname(os.path.dirname(seq))), frames=len(files), size=f"{W}x{H}",
               fps=1 / np.median(dt), drops=int((dt > 1.5 * np.median(dt)).sum()), events=n,
               rate=n / max(ts[-1] - ts[0], 1e-9) / (W * H), on=on / max(seen, 1), per=np.median(per), bad=bad,
               done=os.path.exists(os.path.join(seq, "done")), cfa=open(os.path.join(seq, "bayer.txt")).read().strip()
               if os.path.exists(os.path.join(seq, "bayer.txt")) else "")
    return row


def brightness(png, cfa, gamma):
    # linear brightness each event pixel sees: luma, or under a color filter the channel of that pixel's filter
    v = (np.asarray(Image.open(png).convert("RGB"), dtype=np.float64) / 255) ** gamma
    if not cfa:
        return v @ np.array([0.299, 0.587, 0.114])
    out = np.empty(v.shape[:2])
    for q, c in enumerate(cfa):
        out[q // 2::2, q % 2::2] = v[q // 2::2, q % 2::2, RGB[c]]
    return out


def corr(u, v):
    u, v = u - u.mean(), v - v.mean()
    return float((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum() + 1e-12))


def pair(seq, k, cfa):
    # net event count over [t_k + d, t_k+1 + d] vs the log change between frames k and k+1, per offset d (ms):
    # - time offset: only pixels that change the same way over frames k-1 .. k+2 (a pixel that turns around needs 2 C before
    #   its next event, which delays events by C / 2 / speed; steady pixels have no such delay)
    # - gamma 1 vs 2.2 and the pixel shift (at the best d): all pixels
    ts, files = np.load(os.path.join(seq, "frame_ts.npy")), sorted(glob.glob(os.path.join(seq, "frames", "*.png")))
    ev = {q: np.load(os.path.join(seq, f"ev_{q}.npy"), mmap_mode="r") for q in "txyp"}
    lo, hi = np.searchsorted(ev["t"], [ts[k] + D_MS[0] * 1e-3, ts[k + 1] + D_MS[-1] * 1e-3])
    t, x, y, p = (np.asarray(ev[q][lo:hi]) for q in "txyp")
    L = {g: [np.log(brightness(files[q], cfa, g) + 0.01) for q in range(k - 1, k + 3)] for g in (1.0, 2.2)}
    H, W = L[1.0][0].shape
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    t, pix, p = t[ok], y[ok].astype(np.int64) * W + x[ok], p[ok].astype(np.float64)
    nets = []
    for d in D_MS:
        i, j = np.searchsorted(t, [ts[k] + d * 1e-3, ts[k + 1] + d * 1e-3])
        net = np.bincount(pix[i:j], p[i:j], W * H)
        c = max(np.percentile(np.abs(net), 99.9), 1.0)  # hot pixels must not dominate
        nets.append(np.clip(net, -c, c))
    if np.abs(nets[len(D_MS) // 2]).sum() < 0.02 * W * H:  # too few events to say anything
        return None
    out = {}
    for g in (1.0, 2.2):
        d1, d2, d3 = (np.diff(L[g], axis=0)[q].ravel() for q in range(3))
        steady = (np.sign(d1) == np.sign(d2)) & (np.sign(d2) == np.sign(d3)) & (np.abs(d1) > 0.4) & (np.abs(d2) > 0.05)
        out[g] = np.array([corr(net, d2) for net in nets])
        out["t", g] = np.array([corr(net[steady], d2[steady]) for net in nets]) if steady.sum() >= max(50, 1e-3 * W * H) else None
    best, dL = nets[int(out[1.0].argmax())].reshape(H, W), L[1.0][2] - L[1.0][1]
    inner = (slice(3, H - 3), slice(3, W - 3))
    out["shift"] = np.array([corr(np.roll(best, s, (0, 1))[inner], dL[inner]) for s in SHIFTS])
    return out


summary = []
for root in a.roots:
    seqs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(root, "*", "*", "frame_ts.npy")))
    print(f"\n== {root}: {len(seqs)} sequences " + ", ".join(f"{sum(f'/{s}/' in q for q in seqs)} {s}" for s in ("train", "val", "test")))
    print(f"  {'sequence':38s} {'frames':>7s} {'size':>9s} {'fps':>6s} {'drops':>5s} {'events':>9s} {'ev/px/s':>7s} "
          f"{'ON':>4s} {'ev/gap':>7s}  notes")
    rows = []
    for seq in seqs:
        try:
            r = check(seq)
        except Exception as e:
            r = dict(seq=os.path.relpath(seq, root), frames=0, size="?", fps=0, drops=0, events=0, rate=0, on=0, per=0,
                     bad=[f"unreadable ({type(e).__name__}: {e})"], done=False, cfa="")
        rows.append(r)
        notes = ["! " + b for b in r["bad"]] + ["no done mark"] * (not r["done"]) + [f"bayer {r['cfa']}"] * bool(r["cfa"])
        print(f"  {r['seq'][:38]:38s} {r['frames']:7d} {r['size']:>9s} {r['fps']:6.1f} {r['drops']:5d} {r['events'] / 1e6:8.1f}M "
              f"{r['rate']:7.2f} {100 * r['on']:3.0f}% {r['per']:7.0f}  " + "; ".join(notes), flush=True)
    res, w = [], np.array([max(r["frames"] - 1, 0) for r in rows], dtype=np.float64)
    for _ in range(a.pairs * 3 if w.sum() else 0):
        if len(res) == a.pairs:
            break
        seq = seqs[rng.choice(len(seqs), p=w / w.sum())]
        r = rows[seqs.index(seq)]
        if r["bad"] or r["frames"] < 2:
            continue
        try:
            q = pair(seq, int(rng.integers(1, r["frames"] - 2)), r["cfa"]) if r["frames"] >= 4 else None
        except Exception as e:
            print(f"  ! alignment pair failed in {r['seq']}: {type(e).__name__}: {e}")
            q = None
        res += [q] if q else []
    line = dict(root=root, seqs=len(seqs), frames=sum(r["frames"] for r in rows), events=sum(r["events"] for r in rows),
                hours=sum(r["frames"] / max(r["fps"], 1e-9) for r in rows) / 3600, bad=sum(bool(r["bad"]) for r in rows))
    if res:
        mean = {g: np.mean([q[g] for q in res], 0) for g in (1.0, 2.2)}
        g = 1.0 if mean[1.0].max() >= mean[2.2].max() else 2.2  # the brightness curve the events follow better
        tc = [q["t", g] for q in res if q["t", g] is not None]
        sh = np.mean([q["shift"] for q in res], 0)
        line.update(corr1=mean[1.0].max(), corr22=mean[2.2].max(), shift=SHIFTS[sh.argmax()])
        print(f"  alignment over {len(res)} frame pairs (events vs log change between neighbouring frames):")
        print(f"    gamma: corr {mean[1.0].max():.2f} with 1.0, {mean[2.2].max():.2f} with 2.2 -> {g} fits better")
        print(f"    pixel shift (dy, dx) of the events that fits best: {SHIFTS[sh.argmax()]} (corr {sh.max():.3f}; "
              f"at (0, 0): {sh[SHIFTS.index((0, 0))]:.3f})")
        if tc:
            tm, each = np.mean(tc, 0), np.array([D_MS[c.argmax()] for c in tc])
            line.update(offset=D_MS[tm.argmax()])
            print(f"    time offset (gamma {g}, {len(tc)} pairs with enough steady pixels): mean curve peaks at {D_MS[tm.argmax()]:+d} ms "
                  f"(corr {tm.max():.2f}); per pair median {np.median(each):+.0f} ms, middle half {np.percentile(each, 25):+.0f} "
                  f"to {np.percentile(each, 75):+.0f} ms")
            print("    corr vs offset (ms): " + " ".join(f"{d:+d}:{c:.2f}" for d, c in zip(D_MS[::3], tm[::3]))
                  + ("  (flat: too little motion to measure the offset)" if np.ptp(tm) < 0.02 else "")
                  + ("  (few pairs: rough, raise --pairs)" if len(tc) < 10 else ""))
    summary.append(line)

print(f"\n== summary\n  {'dataset':16s} {'seqs':>5s} {'frames':>8s} {'hours':>6s} {'events':>8s} {'problems':>8s} {'offset':>7s} "
      f"{'corr g1':>7s} {'corr g2.2':>9s} {'shift':>8s}")
for s in summary:
    print(f"  {os.path.basename(s['root'].rstrip('/')):16s} {s['seqs']:5d} {s['frames']:8d} {s['hours']:6.2f} {s['events'] / 1e9:7.2f}G "
          f"{s['bad']:8d} " + (f"{s['offset']:+5d}ms" if "offset" in s else "      -") +
          (f" {s['corr1']:7.2f} {s['corr22']:9.2f} {str(s['shift']):>8s}" if "corr1" in s else ""))
