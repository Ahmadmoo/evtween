import glob
import hashlib
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


def seq_dirs(root):
    # converted sequence folders (holding frame_ts.npy) under one folder or a list of folders, searched recursively
    roots = root if isinstance(root, (list, tuple)) else [root]
    return [os.path.dirname(p) for r in roots for p in sorted(glob.glob(os.path.join(r, "**", "frame_ts.npy"), recursive=True))]


def recording(seq_dir):
    # the original recording behind a converted sequence: converters that split a recording into pieces (HQ-EVFI ranges,
    # BS-ERGB broken files) symlink the frames to the original images, so the pieces share the folder the links point to
    f = sorted(glob.glob(os.path.join(seq_dir, "frames", "*.png")))[:1]
    if f and os.path.islink(f[0]):
        return os.path.dirname(os.path.dirname(os.path.realpath(f[0])))
    return os.path.realpath(seq_dir)


def stable_hash(*parts):
    # deterministic across runs, machines and Python versions (unlike hash())
    return int.from_bytes(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest(), "big")


class PairDataset(torch.utils.data.Dataset):
    # skip: frames hidden between I0 and I1. gap (seconds) instead picks the skip per sequence from its own frame rate,
    # skip = max(0, round(gap * fps) - 1), so datasets at 28, 75 and 142 fps all give about the same gap in seconds.
    # gamma: frame gamma sent with every sample: this value if given, else the sequence's gamma.txt, else default_gamma
    # (None everywhere: no gamma in the batch, the model uses its own). ds: dataset id in the batch.
    # Sequences that cannot give a sample are kept out and listed in self.excluded with the reason.
    def __init__(self, root, crop=256, skip=7, context=4, bins=16, min_events=2000, train=True, gap=None, gamma=None, ds=0,
                 name=None, default_gamma=None):
        roots = root if isinstance(root, (list, tuple)) else [root]  # folders searched recursively, or sequence folders
        self.name = name or ",".join(map(str, roots))
        self.excluded = []
        seqs = []
        for d in seq_dirs(roots):
            s = Sequence(d)
            if len(s) < 2:
                self.excluded.append((os.path.basename(d), f"{len(s)} frame(s)"))
            elif s.ev is None:
                self.excluded.append((os.path.basename(d), "no event files (ev_t.npy ...)"))
            else:
                seqs.append(s)
        assert seqs, f"{self.name}: no usable sequences under {roots}" + (f"; excluded: {self.excluded[:5]}" if self.excluded else "")
        small = [f"{os.path.basename(s.root)} {W}x{H}" for s in seqs for W, H in [s.size] if min(W, H) < crop]
        assert not small, (f"{self.name}: frames smaller than data.crop={crop} cannot be batched with the other datasets: "
                           f"{', '.join(small[:5])}{' ...' if len(small) > 5 else ''}; lower data.crop")
        skips = [max(0, round(gap * s.fps) - 1) if gap else skip for s in seqs]
        self.seqs, self.skips = [], []
        for s, sk in zip(seqs, skips):  # a sample needs its context frames on both sides
            need = (2 * context - 1) * (sk + 1) + 1
            if len(s) < need:
                self.excluded.append((os.path.basename(s.root), f"too short: {len(s)} frames, a sample needs {need} "
                                                                 f"(skip {sk}, context {context})"))
            else:
                self.seqs.append(s)
                self.skips.append(sk)
        assert self.seqs, f"{self.name}: no sequence long enough for one sample; excluded: {self.excluded[:5]}"
        self.index = [(k, i) for k, (s, sk) in enumerate(zip(self.seqs, self.skips))
                      for i in range((context - 1) * (sk + 1), len(s) - context * (sk + 1))]
        self.eligible = len(self.index)
        self.selection = dict(level="all", fraction=1.0, seed=None, eligible=self.eligible, selected=self.eligible,
                              sequences=len(self.seqs))
        self.crop, self.context, self.bins, self.min_events, self.train = crop, context, bins, min_events, train
        self.gammas = [gamma if gamma is not None else s.gamma if s.gamma is not None else default_gamma for s in self.seqs]
        assert len({g is None for g in self.gammas}) == 1, f"{self.name}: set data gamma (some sequences have gamma.txt)"
        self.ds = ds

    def select(self, fraction, seed=0, level="sample"):
        # keep a deterministic fraction of the eligible samples. Every sample (or, at level "sequence", every sequence) gets
        # a hash of (seed, dataset, sequence, frame); the ceil(fraction * N) lowest are kept. The same seed gives the same
        # selection on any machine, and smaller fractions are subsets of larger ones (1% in 5% in 10% ...)
        assert 0 < fraction <= 1, f"{self.name}: fraction {fraction} not in (0, 1]"
        assert level in ("sample", "sequence"), level
        name = lambda k: os.path.basename(self.seqs[k].root)
        if fraction < 1:
            if level == "sample":
                h = np.array([stable_hash(seed, self.name, name(k), i) for k, i in self.index], dtype=np.uint64)
                keep = np.sort(np.argsort(h, kind="stable")[:math.ceil(fraction * len(self.index))])
                self.index = [self.index[j] for j in keep]
            else:
                ks = sorted(range(len(self.seqs)), key=lambda k: stable_hash(seed, self.name, name(k)))
                use = set(ks[:math.ceil(fraction * len(ks))])
                self.index = [(k, i) for k, i in self.index if k in use]
        self.selection = dict(level=level, fraction=fraction, seed=seed, eligible=self.eligible, selected=len(self.index),
                              sequences=len({k for k, _ in self.index}))
        return self.selection

    def sample_id(self, n):
        # (sequence name, I0 frame index) of sample n
        k, i = self.index[n]
        return os.path.basename(self.seqs[k].root), i

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

        for tries in range(1, 11):
            if self.train:
                y0, x0 = int(torch.randint(H - h + 1, ())), int(torch.randint(W - w + 1, ()))
            else:
                y0, x0 = (H - h) // 2, (W - w) // 2
            m = (ev["x"] >= x0) & (ev["x"] < x0 + w) & (ev["y"] >= y0) & (ev["y"] < y0 + h)
            if m.sum() >= self.min_events or not self.train:
                break
        reached = bool(m.sum() >= self.min_events)

        flip = self.train and bool(torch.rand(()) < 0.5)
        # where this sample came from (dataloader_visualization.ipynb traces it back to the original files)
        self.last = dict(seq=s.root, i=i, j=j, ctx=ctx, t0=t0, t1=t1, y0=y0, x0=x0, h=h, w=w, flip=flip, skip=skip)
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
                   ds=torch.tensor(self.ds),
                   # bookkeeping, not model input: sample index n in this dataset, and the crop that was used
                   # (y0, x0, flipped, tries, min_events reached)
                   sid=torch.tensor(n), crop_info=torch.tensor([y0, x0, int(flip), tries, int(reached)]))
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


DATA_KEYS = ("skip", "gap", "gamma", "min_events", "weight", "fraction", "select", "val", "val_frac")


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
    base = dict(skip=D.get("skip", 4), gap=D.get("gap"), gamma=D.get("gamma"), min_events=D.get("min_events", 0), weight=1.0,
                fraction=D.get("fraction", 1.0), select=D.get("select", "sample"), val=D.get("val"),
                val_frac=D.get("val_frac", 0.1), default_gamma=model_gamma)
    out = []
    for n, s in enumerate(sets):
        o = dict(base, **{k: v for k, v in s.items() if k in DATA_KEYS or k in ("name", "root", "val_root", "test_root", "raw")},
                 ds=n)
        if "skip" in s and "gap" not in s:  # a skip given for this set beats the global gap
            o["gap"] = None
        o.setdefault("name", f"set{n}")
        if o["val"] is None:  # official validation folder if the set has one, else held-out training recordings
            o["val"] = "official" if o.get("val_root") else "carve"
        out.append(o)
    return out


def resolve_split(s, split_seed=0, min_recordings=4):
    # sequence folders of train / val / test for one set. val = "official": the set's val_root; "carve": whole original
    # recordings held out of train (ceil(val_frac * recordings), picked by a hash of split_seed and the recording, so the
    # split does not move with the training seed); "none": no validation. The test folders are never changed.
    train = seq_dirs(s["root"])
    test = seq_dirs(s["test_root"]) if s.get("test_root") else []
    val, note = [], ""
    if s["val"] == "official":
        val = seq_dirs(s["val_root"]) if s.get("val_root") else []
        note = f"official validation folder {s.get('val_root')}"
    elif s["val"] == "carve":
        groups = {}
        for d in train:
            groups.setdefault(recording(d), []).append(d)
        if len(groups) >= min_recordings:
            order = sorted(groups, key=lambda g: stable_hash(split_seed, s["name"], os.path.basename(g)))
            held = set(order[:max(1, math.ceil(s["val_frac"] * len(groups)))])
            val = [d for g in held for d in groups[g]]
            train = [d for d in train if recording(d) not in held]
            note = (f"{len(held)} of {len(groups)} training recordings held out (val_frac {s['val_frac']}, "
                    f"split seed {split_seed})")
        else:
            note = f"no validation: only {len(groups)} training recordings (< {min_recordings})"
    elif s["val"] == "none":
        note = "no validation (val: none)"
    else:
        raise ValueError(f"{s['name']}: val must be official, carve or none, not {s['val']}")
    rec = lambda ds_: {recording(d) for d in ds_}
    shared = sorted(os.path.basename(r) for r in rec(train) & rec(test))
    return dict(train=train, val=sorted(val), test=test, val_note=note, train_test_shared_recordings=shared)


def make_dataset(D, s, train, dirs=None):
    # dirs: sequence folders (from resolve_split); default: the set's root (train) or val_root
    root = dirs if dirs is not None else (s["root"] if train else s["val_root"])
    return PairDataset(root, D["crop"], s["skip"], D["context"], D["bins"],
                       s["min_events"] if train else 0, train, gap=s["gap"], gamma=s["gamma"], ds=s["ds"],
                       name=s["name"] + ("" if train else " (val)"), default_gamma=s["default_gamma"])


class MixSampler(torch.utils.data.Sampler):
    # samples a ConcatDataset of several datasets: first a dataset by its weight, then a sample uniformly inside it, so a
    # large dataset does not drown a small one. Works with DDP (each rank takes its own share of one shared draw).
    # start: samples of this rank's share to skip in the current epoch (resume continues the same order)
    def __init__(self, sizes, weights, num_samples=None, rank=0, world=1, seed=0):
        self.sizes, self.rank, self.world, self.seed, self.epoch, self.start = list(sizes), rank, world, seed, 0, 0
        w = np.array([wt if n else 0.0 for wt, n in zip(weights, self.sizes)], dtype=np.float64)
        assert w.sum() > 0, "all datasets are empty or have weight 0"
        self.p = w / w.sum()
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.n = math.ceil((num_samples or sum(self.sizes)) / world)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n - self.start

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        which = rng.choice(len(self.sizes), self.n * self.world, p=self.p)
        idx = self.offsets[which] + (rng.random(len(which)) * np.array(self.sizes)[which]).astype(np.int64)
        return iter(idx[self.rank::self.world][self.start:].tolist())
