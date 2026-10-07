import math
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from data import PairDataset, collate
from losses import compute, diagnostics
from model import build_model

# which parameters each stage trains (pretrained RAFT / video backbone always stay frozen)
TRAIN = {"probe": ("student.", "to_counts."),
         "teacher": ("teacher.", "decoder.", "log_"),
         "student": ("student.", "to_z."),
         "joint": ("teacher.", "decoder.", "student.", "to_z.", "log_")}


def load_cfg(argv):
    # config path, then dotted overrides: python train.py config.yaml train.stage=student train.lr=1e-4
    path = next((a for a in argv if a.endswith(".yaml")), "config.yaml")
    cfg = yaml.safe_load(open(path))
    for kv in (a for a in argv if "=" in a):
        key, val = kv.split("=", 1)
        *keys, last = key.split(".")
        d = cfg
        for k in keys:
            d = d[k]
        val = yaml.safe_load(val)
        try:
            val = float(val) if isinstance(val, str) else val  # yaml reads "1e-4" as a string
        except ValueError:
            pass
        d[last] = val
    return cfg


class Step(nn.Module):
    # the whole loss runs inside forward so DDP can sync gradients
    def __init__(self, model, w, stage):
        super().__init__()
        self.model, self.w, self.stage = model, w, stage

    def forward(self, batch):
        return compute(self.model, batch, self.w, self.stage)


cfg = load_cfg(sys.argv[1:])
D, T, stage = cfg["data"], cfg["train"], cfg["train"]["stage"]
ddp, cuda = "WORLD_SIZE" in os.environ, torch.cuda.is_available()
if ddp:
    dist.init_process_group("nccl" if cuda else "gloo")
    if cuda:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
rank = dist.get_rank() if ddp else 0
dev = f"cuda:{torch.cuda.current_device()}" if cuda else "cpu"
torch.manual_seed(T["seed"] + rank)
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

model = build_model(cfg).to(dev)
if T.get("init"):  # parts whose size changed (e.g. a wider student) start fresh
    sd, own = torch.load(T["init"], map_location=dev, weights_only=False)["model"], model.state_dict()
    skip = sorted({k.split(".")[0] for k, v in sd.items() if k in own and own[k].shape != v.shape})
    model.load_state_dict({k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}, strict=False)
    if rank == 0 and skip:
        print("init: size changed, not loaded:", skip, flush=True)
for n, p in model.named_parameters():
    p.requires_grad = n.startswith(TRAIN[stage]) and not model.frozen(n)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW([{"params": [p for p in params if p.ndim > 1], "weight_decay": T["wd"]},
                         {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0}],
                        lr=T["lr"], betas=(0.9, 0.99))
ckpt, step = os.path.join(T["out"], "last.pt"), 0
if os.path.exists(ckpt):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    opt.load_state_dict(ck["opt"])
    step = ck["step"]
net = Step(model, cfg["loss"], stage)
net = DDP(net, device_ids=[torch.cuda.current_device()] if cuda else None) if ddp else net

make = lambda root, train: PairDataset(root, D["crop"], D["skip"], D["context"], D["bins"], D["min_events"] if train else 0, train)
data = make(D["root"], True)
sampler = DistributedSampler(data) if ddp else None
loader = DataLoader(data, T["batch"], shuffle=sampler is None, sampler=sampler, num_workers=D["workers"], collate_fn=collate,
                    pin_memory=True, drop_last=True, persistent_workers=D["workers"] > 0)
val = None
if D.get("val_root") and rank == 0:
    vd = make(D["val_root"], False)  # fixed samples spread over all val sequences
    vd = torch.utils.data.Subset(vd, torch.linspace(0, len(vd) - 1, T["val_batches"] * T["batch"]).long().tolist())
    val = DataLoader(vd, T["batch"], num_workers=D["workers"], collate_fn=collate)


def lr_at(step):
    if step < T["warmup"]:
        return T["lr"] * (step + 1) / T["warmup"]
    p = (step - T["warmup"]) / max(1, T["steps"] - T["warmup"])
    return T["lr"] * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def validate():
    model.eval()
    sums, n = {}, 0
    for b, batch in enumerate(val):
        if b == T["val_batches"]:
            break
        batch = {k: v.to(dev) for k, v in batch.items()}
        if stage == "probe":
            out = dict(probe=compute(model, batch, cfg["loss"], stage)[0].item())
        else:
            out = diagnostics(model, batch, cfg["loss"], student=stage != "teacher")
        sums = {k: sums.get(k, 0) + v for k, v in out.items()}
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


os.makedirs(T["out"], exist_ok=True)
model.train()
tic = time.time()
while step < T["steps"]:
    if sampler is not None:
        sampler.set_epoch(step)
    for batch in loader:
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss, terms = net(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, T["clip"])
        opt.step()
        step += 1

        if rank == 0 and step % T["log_every"] == 0:
            msg = " ".join(f"{k} {v.item():.4f}" for k, v in terms.items())
            print(f"[{stage}] step {step} loss {loss.item():.4f} {msg} c {model.c.item():.3f} r {model.r.item():.3f} "
                  f"k {model.log_k.exp().item():.2f} nu {model.log_nu.exp().item():.2f} R {model.R.item() * 1e6:.0f}us gn {gnorm.item():.2f} lr {lr_at(step):.1e} {time.time() - tic:.0f}s", flush=True)
        if rank == 0 and (step % T["save_every"] == 0 or step == T["steps"]):
            torch.save(dict(model=model.state(), opt=opt.state_dict(), step=step, cfg=cfg), ckpt)
            if val is not None:
                print("val " + " ".join(f"{k} {v:.4f}" for k, v in validate().items()), flush=True)
        if step >= T["steps"]:
            break

if ddp:
    dist.destroy_process_group()
