import os
import sys
import torch
from torch.utils.data import DataLoader, Subset
from data import PairDataset, collate
from losses import event_nll
from model import build_model

# python check.py runs/teacher/last.pt [batches] [data_dir] [skip]  -> per dataset and split: diagnostics, code / path size,
# time alignment. Default: every dataset in the checkpoint's data.sets; data_dir (holds train/, test/, ...) and skip: one dataset
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(sys.argv[1], map_location=dev, weights_only=False)
cfg, D = ck["cfg"], ck["cfg"]["data"]
model = build_model(cfg).to(dev).eval()
model.load_state_dict(ck["model"], strict=False)
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 25
if len(sys.argv) > 3:
    sets = {os.path.basename(sys.argv[3].rstrip("/")): dict(root=sys.argv[3], skip=int(sys.argv[4]) if len(sys.argv) > 4 else 0)}
else:  # older checkpoints: data.root is the train folder of one dataset
    sets = {k: v for k, v in D["sets"].items() if v} if "sets" in D else {"data": dict(root=os.path.dirname(D["root"]), skip=D["skip"])}

for name, S in sets.items():
    for split in ("train", "val", "test"):
        if not os.path.isdir(f"{S['root']}/{split}"):
            continue
        ds = PairDataset(f"{S['root']}/{split}", D["crop"], S["skip"], D["context"], D["bins"], 0, train=False)
        print(f"{name} {split}: {len(ds.seqs)} sequences, {len(ds)} samples, skip {S['skip']}")
        if len(ds) == 0:
            continue
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
        print(f"{name} {split}", " ".join(f"{k} {v / n:.3f}" for k, v in acc.items()))
        print(f"{name} {split} nll_zero vs event shift (ms):", " ".join(f"{k:+d}:{v / n:.3f}" for k, v in shift_acc.items()))
