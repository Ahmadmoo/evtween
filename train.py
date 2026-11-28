import math
import os
import sys
import time
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from data import AtSkip, JointBatches, JointDataset, collate, datasets
from losses import compute, diagnostics
from model import build_model

# which parameters each stage trains (the pretrained video backbone always stays frozen, LoRA aside);
# interp: the teacher alone (decoder, sensor and student frozen), for frame interpolation
TRAIN = {"probe": ("student.", "to_counts."),
         "teacher": ("teacher.", "decoder.", "log_", "w_rgb"),
         # v8.5: the teacher stays frozen; its decoder keeps training on teacher codes, the student's own on student codes
         "student": ("student.", "to_z.", "decoder.", "student_decoder."),
         "joint": ("teacher.", "decoder.", "student.", "to_z.", "student_decoder.", "log_", "w_rgb"),
         "interp": ("teacher.",)}


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
    ck0 = torch.load(T["init"], map_location=dev, weights_only=False)
    sd, own = ck0["model"], model.state_dict()
    sd = {k: v.expand_as(own[k]).clone() if k in own and v.dim() == 0 and own[k].dim() == 1 else v for k, v in sd.items()}  # one sensor -> all
    skip = sorted({k.split(".")[0] for k, v in sd.items() if k in own and own[k].shape != v.shape})
    model.load_state_dict({k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}, strict=False)
    if rank == 0 and skip:
        print("init: size changed, not loaded:", skip, flush=True)
    if stage in ("student", "joint") and ck0.get("cfg", {}).get("train", {}).get("stage") not in ("student", "joint"):
        model.student_decoder.load_state_dict(model.decoder.state_dict())  # v8.5: the student's decoder starts as the teacher's
for n, p in model.named_parameters():
    p.requires_grad = n.startswith(TRAIN[stage]) and not model.frozen(n) and not model.fixed(n)
params = [p for n, p in model.named_parameters() if p.requires_grad and ".lora_" not in n]
lora = [p for n, p in model.named_parameters() if p.requires_grad and ".lora_" in n]
groups = [{"params": [p for p in params if p.ndim > 1], "weight_decay": T["wd"]},
          {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0}]
if lora:  # LoRA on the video backbone (model.lora_rank): its own learning rate, x train.lora_lr
    groups.append({"params": lora, "weight_decay": 0.0, "scale": float(T.get("lora_lr", 1.0))})
params = params + lora
opt = torch.optim.AdamW(groups, lr=T["lr"], betas=(0.9, 0.99))
ckpt, step = os.path.join(T["out"], "last.pt"), 0
if os.path.exists(ckpt):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    opt.load_state_dict(ck["opt"])
    step = ck["step"]
net = Step(model, cfg["loss"], stage)
net = DDP(net, device_ids=[torch.cuda.current_device()] if cuda else None) if ddp else net

sets = datasets(D, "train", True)  # one per entry of data.sets; every batch comes from one of them
data = JointDataset(list(sets.values()))
batches = JointBatches([len(d) for d in data.sets], [D["sets"][name].get("weight", 1.0) for name in sets], T["batch"],
                       T["seed"] + 1000 * rank + step, [d.skips for d in data.sets])  # each rank its own stream, a new one after a resume
loader = DataLoader(data, batch_sampler=batches, num_workers=D["workers"], collate_fn=collate, pin_memory=True,
                    persistent_workers=D["workers"] > 0)
vals = {}
if rank == 0:
    for (name, d), share in zip(sets.items(), batches.p):
        sk = d.skips[0] if d.skips[0] == d.skips[1] else f"{d.skips[0]}-{d.skips[1]} (one per batch)"
        print(f"data {name}: {len(d.seqs)} sequences, {len(d)} samples, skip {sk}, {100 * share:.0f}% of batches", flush=True)
    for name, vd in datasets(D, "val", False).items():  # fixed samples spread over all val (or test) sequences
        if len(vd):  # a skip range: validation at both ends, "val <name> skip<k>"
            idx = torch.linspace(0, len(vd) - 1, T["val_batches"] * T["batch"]).long().tolist()
            for k in sorted(set(vd.skips), reverse=True):
                key = name if vd.skips[0] == vd.skips[1] else f"{name} skip{k}"
                vals[key] = DataLoader(Subset(AtSkip(vd, k), idx), T["batch"], num_workers=D["workers"], collate_fn=collate)


def lr_at(step):
    if step < T["warmup"]:
        return T["lr"] * (step + 1) / T["warmup"]
    p = (step - T["warmup"]) / max(1, T["steps"] - T["warmup"])
    return T["lr"] * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def validate(val):
    model.eval()
    sums, n = {}, 0
    for batch in val:
        batch = {k: v.to(dev) for k, v in batch.items()}
        if stage == "probe":
            out = dict(probe=compute(model, batch, cfg["loss"], stage)[0].item())
        else:
            out = diagnostics(model, batch, cfg["loss"], student=stage not in ("teacher", "interp"))
        sums = {k: sums.get(k, 0) + v for k, v in out.items()}
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


os.makedirs(T["out"], exist_ok=True)
model.train()
tic, n_log = time.time(), 0
while step < T["steps"]:
    for batch in loader:  # endless
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        for g in opt.param_groups:
            g["lr"] = lr_at(step) * g.get("scale", 1.0)
        loss, terms = net(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, T["clip"])
        opt.step()
        step, n_log = step + 1, n_log + 1

        if rank == 0 and step % T["log_every"] == 0:
            msg = " ".join(f"{k} {v.item():.4f}" for k, v in terms.items())
            d = int(batch["ds"][0])  # the sensor of this batch's dataset
            C, K = (v[0] for v in model.sensor(batch["ds"][:1]))
            sec = (time.time() - tic) / n_log  # per step since the last log line (validation left out)
            print(f"[{stage}] step {step} {list(D['sets'])[d]} loss {loss.item():.4f} {msg} c {C[0].item():.3f} r {(C[1] / C[0]).item():.3f} "
                  f"k {K.item():.2f} "
                  f"gn {gnorm.item():.2f} lr {lr_at(step):.1e} | {T['batch'] * (dist.get_world_size() if ddp else 1) / sec:.1f} samples/s "
                  f"ETA {timedelta(seconds=int((T['steps'] - step) * sec))}", flush=True)
            tic, n_log = time.time(), 0
        if rank == 0 and (step % T["save_every"] == 0 or step == T["steps"]):
            torch.save(dict(model=model.state(), opt=opt.state_dict(), step=step, cfg=cfg), ckpt)
            for name, val in vals.items():
                print(f"val {name} " + " ".join(f"{k} {v:.4f}" for k, v in validate(val).items()), flush=True)
            tic, n_log = time.time(), 0
        if step >= T["steps"]:
            break

if ddp:
    dist.destroy_process_group()
