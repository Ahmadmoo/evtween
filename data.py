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
    def __init__(self, root, crop=256, skip=7, context=4, bins=16, min_events=2000, train=True):
        roots = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "frame_ts.npy")))
        self.seqs = [Sequence(r) for r in (roots or [root])]
        self.index = [(k, i) for k, s in enumerate(self.seqs) for i in range(context - 1, len(s) - skip - context)]
        self.crop, self.skip, self.context, self.bins, self.min_events, self.train = crop, skip, context, bins, min_events, train

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        # keyframes I0, I1, the hidden frames between them, context frames for the student,
        # the real events of the gap (time-binned for the teacher, exact list for the likelihood)
        k, i = self.index[n]
        s, j, c = self.seqs[k], i + self.skip + 1, self.context
        t0, t1 = s.ts[i], s.ts[j]
        ev = s.events(t0, t1)
        full = {q: s.frame(q) for q in range(i - c + 1, j + c)}
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
        ctx = [*range(i - c + 1, i + 1), *range(j, j + c)]
        return dict(i0=cut(i), i1=cut(j),
                    mid=torch.stack([cut(q) for q in range(i + 1, j)]) if self.skip else torch.zeros(0, 3, h, w),
                    mid_tau=torch.from_numpy((s.ts[i + 1:j] - t0) / (t1 - t0)).float(),
                    ctx=torch.stack([cut(q) for q in ctx]), ctx_tau=torch.from_numpy((s.ts[ctx] - t0) / (t1 - t0)).float(),
                    dt=torch.tensor(t1 - t0, dtype=torch.float32), voxel=torch.from_numpy(voxel),
                    ev_pix=torch.from_numpy(pix), ev_tau=torch.from_numpy(tau), ev_pol=torch.from_numpy(pol))


def collate(items):
    # stack fixed-size fields; concatenate the variable-length event lists with a batch index
    ev = ("ev_pix", "ev_tau", "ev_pol")
    out = {k: torch.stack([it[k] for it in items]) for k in items[0] if k not in ev}
    for k in ev:
        out[k] = torch.cat([it[k] for it in items])
    out["ev_b"] = torch.cat([torch.full((len(it["ev_tau"]),), b, dtype=torch.long) for b, it in enumerate(items)])
    return out
