import glob
import os
import numpy as np
import torch
from PIL import Image


class Sequence:
    # one recording: frames/*.png, frame_ts.npy (s), and optional ev_t/x/y/p.npy sorted by time
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, "frames", "*.png")))
        self.ts = np.load(os.path.join(root, "frame_ts.npy")).astype(np.float64)
        has_ev = os.path.exists(os.path.join(root, "ev_t.npy"))
        self.ev = {k: np.load(os.path.join(root, f"ev_{k}.npy"), mmap_mode="r") for k in "txyp"} if has_ev else None

    def __len__(self):
        return len(self.files)

    def frame(self, i):
        img = np.asarray(Image.open(self.files[i]).convert("RGB"), dtype=np.float32) / 255
        return torch.from_numpy(img).permute(2, 0, 1)

    def events(self, t0, t1):
        a, b = np.searchsorted(self.ev["t"], [t0, t1])
        return {k: np.asarray(v[a:b]) for k, v in self.ev.items()}


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, root, crop=256, skip=7, n_tau=8, min_events=2000, train=True):
        roots = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "frame_ts.npy")))
        self.seqs = [Sequence(r) for r in (roots or [root])]
        self.index = [(k, i) for k, s in enumerate(self.seqs) for i in range(len(s) - skip - 1)]
        self.crop, self.skip, self.n_tau, self.min_events, self.train = crop, skip, n_tau, min_events, train

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        # two keyframes, the skipped frames between them, and real event counts on random sub-intervals
        k, i = self.index[n]
        s, j = self.seqs[k], i + self.skip + 1
        t0, t1 = s.ts[i], s.ts[j]
        ev = s.events(t0, t1)
        i0 = s.frame(i)
        H, W = i0.shape[1:]
        h, w = min(self.crop, H), min(self.crop, W)

        for _ in range(10):
            if self.train:
                y0, x0 = (int(torch.randint(H - h + 1, ())), int(torch.randint(W - w + 1, ())))
            else:
                y0, x0 = (H - h) // 2, (W - w) // 2
            m = (ev["x"] >= x0) & (ev["x"] < x0 + w) & (ev["y"] >= y0) & (ev["y"] < y0 + h)
            if m.sum() >= self.min_events or not self.train:
                break

        if self.train:
            taus = torch.cat([torch.zeros(1), torch.rand(self.n_tau - 1).sort()[0], torch.ones(1)])
        else:
            taus = torch.linspace(0, 1, self.n_tau + 1)
        te = (ev["t"][m] - t0) / (t1 - t0)
        b = np.searchsorted(taus[1:-1].numpy(), te, side="right")
        idx = b * h * w + (ev["y"][m].astype(np.int64) - y0) * w + (ev["x"][m].astype(np.int64) - x0)
        pos = ev["p"][m] > 0
        count = lambda sel: torch.from_numpy(np.bincount(idx[sel], minlength=self.n_tau * h * w)
                                             .reshape(self.n_tau, h, w).astype(np.float32))

        cut = lambda img: img[:, y0:y0 + h, x0:x0 + w]
        out = dict(i0=cut(i0), i1=cut(s.frame(j)),
                   mid=torch.stack([cut(s.frame(q)) for q in range(i + 1, j)]) if self.skip else torch.zeros(0, 3, h, w),
                   mid_tau=torch.from_numpy((s.ts[i + 1:j] - t0) / (t1 - t0)).float(),
                   taus=taus, on=count(pos), off=count(~pos))
        if self.train and torch.rand(()) < 0.5:
            for key in ("i0", "i1", "mid", "on", "off"):
                out[key] = out[key].flip(-1)
        return out
