import sys
import torch
from torch.utils.data import DataLoader, Subset
from data import PairDataset, collate
from losses import event_nll
from model import build_model

# python check.py runs/teacher/last.pt [batches]  -> train vs test diagnostics, code / path size, time alignment
ck = torch.load(sys.argv[1], map_location="cuda", weights_only=False)
cfg, D, dev = ck["cfg"], ck["cfg"]["data"], "cuda"
model = build_model(cfg).to(dev).eval()
model.load_state_dict(ck["model"], strict=False)
root, nb = D["root"].rsplit("/", 1)[0], int(sys.argv[2]) if len(sys.argv) > 2 else 25

for split in ("train", "test"):
    ds = PairDataset(f"{root}/{split}", D["crop"], D["skip"], D["context"], D["bins"], 0, train=False)
    ds = Subset(ds, torch.linspace(0, len(ds) - 1, nb * 8).long().tolist())  # fixed samples over all sequences, as in train.py
    acc, shift_acc, n = {}, {}, 0
    for b in DataLoader(ds, 8, collate_fn=collate, num_workers=4):
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.no_grad():
            p = model.prepare(b["i0"], b["i1"])
            z = model.encode(p, b["voxel"])
            ev = (b["ev_b"], b["ev_pix"], b["ev_tau"], b["ev_pol"])
            nll = lambda code, e=ev: event_nll(model, model.decode(p, code), e, b["dt"], cfg["loss"]["grid"]).item()
            s = model.decode(p, z)
            out = dict(nll=nll(z), shuffled=nll(z.roll(1, 0)), zero=nll(torch.zeros_like(z)),
                       z_std=z.std().item(), z_absmax=z.abs().max().item(), flow_px=s["a"].abs().mean().item(),
                       events_per_px=len(b["ev_tau"]) / b["voxel"][:, 0].numel())
            m = 3e-3 / b["dt"][b["ev_b"]]
            keep = (b["ev_tau"] >= m) & (b["ev_tau"] < 1 - m)  # same events for every shift
            for ms in (-3, -2, -1, 0, 1, 2, 3):  # move all events in time, plain RAFT path
                t = b["ev_tau"] + ms * 1e-3 / b["dt"][b["ev_b"]]
                e = (b["ev_b"][keep], b["ev_pix"][keep], t[keep], b["ev_pol"][keep])
                shift_acc[ms] = shift_acc.get(ms, 0) + nll(torch.zeros_like(z), e)
        acc = {k: acc.get(k, 0) + v for k, v in out.items()}
        n += 1
    print(split, " ".join(f"{k} {v / n:.3f}" for k, v in acc.items()))
    print(split, "nll_zero vs event shift (ms):", " ".join(f"{k:+d}:{v / n:.3f}" for k, v in shift_acc.items()))
