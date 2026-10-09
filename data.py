import glob
import math
import os
import numpy as np
import torch
from PIL import Image


class Sequence:
    # one recording: frames/*.png, frame_ts.npy (s), and optional ev_t/x/y/p.npy sorted by time
    def __init__(self, root):
        self.root = root
        self.files = sorted(glob.glob(os.path.join(root, "frames", "*.png")))
        self.ts = np.load(os.path.join(root, "frame_ts.npy")).astype(np.float64)
        g = os.path.join(root, "gamma.txt")  # written by converters whose frames differ in encoding (CED: raw vs sRGB)
        self.gamma = float(open(g).read()) if os.path.exists(g) else None
        has_ev = os.path.exists(os.path.join(root, "ev_t.npy"))
        self.ev = {k: np.load(os.path.join(root, f"ev_{k}.npy"), mmap_mode="r") for k in "txyp"} if has_ev else None

    def __len__(self):
        return len(self.files)

    @property
    def fps(self):
        return 1 / np.median(np.diff(self.ts))

    @property
    def size(self):
        return Image.open(self.files[0]).size  # (W, H), header only

    def frame(self, i):
        img = np.asarray(Image.open(self.files[i]).convert("RGB"), dtype=np.float32) / 255
        return torch.from_numpy(img).permute(2, 0, 1)

    def events(self, t0, t1):
        a, b = np.searchsorted(self.ev["t"], [t0, t1])
        return {k: np.asarray(v[a:b]) for k, v in self.ev.items()}


class PairDataset(torch.utils.data.Dataset):
    # skip: frames hidden between I0 and I1. gap (seconds) instead picks the skip per sequence from its own frame rate,
    # skip = max(0, round(gap * fps) - 1), so datasets at 28, 75 and 142 fps all give about the same gap in seconds.
    # gamma: frame gamma sent with every sample: this value if given, else the sequence's gamma.txt, else default_gamma
    # (None everywhere: no gamma in the batch, the model uses its own). ds: dataset id in the batch.
    def __init__(self, root, crop=256, skip=7, context=4, bins=16, min_events=2000, train=True, gap=None, gamma=None, ds=0,
                 name=None, default_gamma=None):
        roots = root if isinstance(root, (list, tuple)) else [root]  # one folder or a list, searched recursively
        self.seqs = [Sequence(os.path.dirname(p)) for r in roots
                     for p in sorted(glob.glob(os.path.join(r, "**", "frame_ts.npy"), recursive=True))]
        self.seqs = [s for s in self.seqs if len(s) >= 2]
        self.name = name or ",".join(map(str, roots))
        assert self.seqs, f"{self.name}: no sequences (frame_ts.npy) under {roots}"
        small = [f"{os.path.basename(s.root)} {W}x{H}" for s in self.seqs for W, H in [s.size] if min(W, H) < crop]
        assert not small, (f"{self.name}: frames smaller than data.crop={crop} cannot be batched with the other datasets: "
                           f"{', '.join(small[:5])}{' ...' if len(small) > 5 else ''}; lower data.crop")
        self.skips = [max(0, round(gap * s.fps) - 1) if gap else skip for s in self.seqs]
        self.index = [(k, i) for k, (s, sk) in enumerate(zip(self.seqs, self.skips))
                      for i in range((context - 1) * (sk + 1), len(s) - context * (sk + 1))]
        self.crop, self.context, self.bins, self.min_events, self.train = crop, context, bins, min_events, train
        self.gammas = [gamma if gamma is not None else s.gamma if s.gamma is not None else default_gamma for s in self.seqs]
        assert len({g is None for g in self.gammas}) == 1, f"{self.name}: set data gamma (some sequences have gamma.txt)"
        self.ds = ds

    def summary(self):
        fps, sk = np.array([s.fps for s in self.seqs]), np.array(self.skips)
        gaps = (sk + 1) / fps * 1e3
        return (f"{self.name}: {len(self.seqs)} sequences, {len(self)} samples, {fps.min():.1f}-{fps.max():.1f} fps, "
                f"skip {sk.min()}-{sk.max()}, gap {gaps.min():.1f}-{gaps.max():.1f} ms"
                + ("" if self.gammas[0] is None else f", gamma {'/'.join(f'{g:g}' for g in sorted(set(self.gammas)))}"))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        # keyframes I0, I1, the hidden frames between them, the real events of the gap (time-binned for the teacher,
        # exact list for the likelihood), and context frames at the gap's own stride: the student sees exactly
        # the frames a video at this frame rate would give
        k, i = self.index[n]
        s, c, skip = self.seqs[k], self.context, self.skips[k]
        j, g = i + skip + 1, skip + 1
        t0, t1 = s.ts[i], s.ts[j]
        ev = s.events(t0, t1)
        ctx = [i - q * g for q in reversed(range(c))] + [j + q * g for q in range(c)]
        full = {q: s.frame(q) for q in {*ctx, *range(i, j + 1)}}
        H, W = full[i].shape[1:]
        h, w = min(self.crop, H), min(self.crop, W)

        for _ in range(10):
            if self.train:
                y0, x0 = int(torch.randint(H - h + 1, ())), int(torch.randint(W - w + 1, ()))
            else:
                y0, x0 = (H - h) // 2, (W - w) // 2
            m = (ev["x"] >= x0) & (ev["x"] < x0 + w) & (ev["y"] >= y0) & (ev["y"] < y0 + h)
            if m.sum() >= self.min_events or not self.train:
                break

        flip = self.train and bool(torch.rand(()) < 0.5)
        x = ev["x"][m].astype(np.int64) - x0
        x = w - 1 - x if flip else x
        pix = (ev["y"][m].astype(np.int64) - y0) * w + x
        tau = ((ev["t"][m] - t0) / (t1 - t0)).astype(np.float32)
        pol = ev["p"][m].astype(np.int8)
        o = np.lexsort((tau, pix))
        pix, tau, pol = pix[o], tau[o], pol[o]
        cell = ((pol < 0) * self.bins + np.minimum((tau * self.bins).astype(np.int64), self.bins - 1)) * h * w + pix
        voxel = np.bincount(cell, minlength=2 * self.bins * h * w).reshape(2 * self.bins, h, w).astype(np.float32)

        cut = lambda q: full[q][:, y0:y0 + h, x0:x0 + w].flip(-1) if flip else full[q][:, y0:y0 + h, x0:x0 + w]
        out = dict(i0=cut(i), i1=cut(j),
                   mid=torch.stack([cut(q) for q in range(i + 1, j)]) if skip else torch.zeros(0, 3, h, w),
                   mid_tau=torch.from_numpy((s.ts[i + 1:j] - t0) / (t1 - t0)).float(),
                   ctx=torch.stack([cut(q) for q in ctx]), ctx_tau=torch.from_numpy((s.ts[ctx] - t0) / (t1 - t0)).float(),
                   dt=torch.tensor(t1 - t0, dtype=torch.float32), voxel=torch.from_numpy(voxel),
                   ev_pix=torch.from_numpy(pix), ev_tau=torch.from_numpy(tau), ev_pol=torch.from_numpy(pol),
                   ds=torch.tensor(self.ds))
        if self.gammas[k] is not None:
            out["gamma"] = torch.tensor(float(self.gammas[k]))
        return out


def collate(items):
    # stack fixed-size fields; pad the hidden frames to the longest gap in the batch (mid_mask marks the real ones);
    # concatenate the variable-length event lists with a batch index
    ev, var = ("ev_pix", "ev_tau", "ev_pol"), ("mid", "mid_tau")
    out = {k: torch.stack([it[k] for it in items]) for k in items[0] if k not in ev + var}
    M = max(len(it["mid_tau"]) for it in items)
    if all(len(it["mid_tau"]) == M for it in items):
        out["mid"], out["mid_tau"] = torch.stack([it["mid"] for it in items]), torch.stack([it["mid_tau"] for it in items])
    else:
        pad = lambda v, fill: torch.cat([v, v.new_full((M - len(v), *v.shape[1:]), fill)])
        out["mid"] = torch.stack([pad(it["mid"], 0.0) for it in items])
        out["mid_tau"] = torch.stack([pad(it["mid_tau"], 0.5) for it in items])
        out["mid_mask"] = torch.stack([torch.arange(M) < len(it["mid_tau"]) for it in items])
    for k in ev:
        out[k] = torch.cat([it[k] for it in items])
    out["ev_b"] = torch.cat([torch.full((len(it["ev_tau"]),), b, dtype=torch.long) for b, it in enumerate(items)])
    return out


def data_sets(D, model_gamma=None):
    # the dataset list of the config: data.sets (one entry per dataset, any data.* key can be overridden per set), or
    # the single data.root / data.val_root of older configs
    # data.root (e.g. given on the command line) replaces the list with that one folder; data.use: names of the sets to
    # train on (null: all)
    sets = [dict(name="data", root=D["root"], val_root=D.get("val_root"))] if D.get("root") else D["sets"]
    if D.get("use"):
        use = [D["use"]] if isinstance(D["use"], str) else list(D["use"])
        missing = sorted(set(use) - {s.get("name") for s in sets})
        assert not missing, f"data.use: no set named {missing}; sets: {[s.get('name') for s in sets]}"
        sets = [s for s in sets if s.get("name") in use]
    keys = ("skip", "gap", "gamma", "min_events", "weight")
    base = dict(skip=D.get("skip", 4), gap=D.get("gap"), gamma=D.get("gamma"), min_events=D.get("min_events", 0), weight=1.0,
                default_gamma=model_gamma)
    out = []
    for n, s in enumerate(sets):
        o = dict(base, **{k: v for k, v in s.items() if k in keys or k in ("name", "root", "val_root")}, ds=n)
        if "skip" in s and "gap" not in s:  # a skip given for this set beats the global gap
            o["gap"] = None
        o.setdefault("name", f"set{n}")
        out.append(o)
    return out


def make_dataset(D, s, train):
    return PairDataset(s["root"] if train else s["val_root"], D["crop"], s["skip"], D["context"], D["bins"],
                       s["min_events"] if train else 0, train, gap=s["gap"], gamma=s["gamma"], ds=s["ds"],
                       name=s["name"] + ("" if train else " (val)"), default_gamma=s["default_gamma"])


class MixSampler(torch.utils.data.Sampler):
    # samples a ConcatDataset of several datasets: first a dataset by its weight, then a sample uniformly inside it, so a
    # large dataset does not drown a small one. Works with DDP (each rank takes its own share of one shared draw).
    def __init__(self, sizes, weights, num_samples=None, rank=0, world=1, seed=0):
        self.sizes, self.rank, self.world, self.seed, self.epoch = list(sizes), rank, world, seed, 0
        w = np.array([wt if n else 0.0 for wt, n in zip(weights, self.sizes)], dtype=np.float64)
        assert w.sum() > 0, "all datasets are empty or have weight 0"
        self.p = w / w.sum()
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.n = math.ceil((num_samples or sum(self.sizes)) / world)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        which = rng.choice(len(self.sizes), self.n * self.world, p=self.p)
        idx = self.offsets[which] + (rng.random(len(which)) * np.array(self.sizes)[which]).astype(np.int64)
        return iter(idx[self.rank::self.world].tolist())
