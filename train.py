import math
import os
import random
import signal
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from data import MixSampler, collate, data_sets, make_dataset, resolve_split
from losses import compute, diagnostics
from model import build_model
import experiment as X

# which parameters each stage trains (pretrained RAFT / video backbone always stay frozen)
TRAIN = {"probe": ("student.", "to_counts."),
         "teacher": ("teacher.", "decoder.", "log_"),
         "student": ("student.", "to_z."),
         "joint": ("teacher.", "decoder.", "student.", "to_z.", "log_")}
# validation metric that picks best.pt (mean over the datasets with a validation split)
MAIN = {"probe": "probe", "teacher": "nll", "student": "nll_student", "joint": "nll_student"}


def load_cfg(argv):
    # config path, then dotted overrides: python train.py config.yaml train.stage=student train.lr=1e-4
    path = next((a for a in argv if a.endswith(".yaml")), "config.yaml")
    cfg = yaml.safe_load(open(path))
    for kv in (a for a in argv if "=" in a and not a.startswith("-")):
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
        per = {}  # per-sample values of the terms (for per-dataset logs); the loss itself is unchanged
        loss, terms = compute(self.model, batch, self.w, self.stage, per)
        return loss, terms, per


def lr_at(T, step):
    if step < T["warmup"]:
        return T["lr"] * (step + 1) / T["warmup"]
    p = (step - T["warmup"]) / max(1, T["steps"] - T["warmup"])
    return T["lr"] * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p)))


OPTS = dict(exp=None, resume=False, force_resume=False, dry_run=False, allow_no_val=False, data_seed=0,
            min_val_recordings=4, inspect_samples=32)


def run(cfg, **opts):
    # one training run (or, with dry_run, only the dataset and DataLoader reports) in the experiment folder opts["exp"]
    # (default: train.out). Returns "completed", "dry_run", "nonfinite" or "signal".
    o = dict(OPTS, inspect_samples=cfg["train"].get("inspect_samples", 32), **opts)
    D, T, stage = cfg["data"], cfg["train"], cfg["train"]["stage"]
    T.setdefault("val_every", T["save_every"])
    T.setdefault("keep", 2)
    exp = o["exp"] or T["out"]
    T["out"] = exp
    ddp, cuda = "WORLD_SIZE" in os.environ, torch.cuda.is_available()
    if ddp:
        dist.init_process_group("nccl" if cuda else "gloo")
        if cuda:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    dev = f"cuda:{torch.cuda.current_device()}" if cuda else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    seed_all = lambda s: (random.seed(s), np.random.seed(s % 2 ** 32), torch.manual_seed(s))
    seed_all(T["seed"] + rank)

    log = X.Log()
    ck_dir = os.path.join(exp, "checkpoints")
    if rank == 0:
        X.prepare_dir(exp, o["resume"], o["dry_run"])
        log = X.Log(os.path.join(exp, "training.log"))
        log("=" * 100)
        log(f"{'resume' if o['resume'] else 'dry run' if o['dry_run'] else 'new run'} in {os.path.realpath(exp)}  "
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
        git, diff = X.git_info(os.path.dirname(os.path.abspath(__file__)))
        env = X.env_info()
        tag = time.strftime("%Y%m%d-%H%M%S")
        X.write_json(os.path.join(exp, f"env_{tag}.json"), dict(env=env, git=git, world_size=world))
        if diff:
            open(os.path.join(exp, f"git_diff_{tag}.patch"), "w").write(diff)
        log(f"code: {git.get('repo')} branch {git.get('branch')} commit {git.get('commit', '')[:10]}"
            + (f" WITH UNCOMMITTED CHANGES (saved to git_diff_{tag}.patch)" if git.get("dirty") else " (clean)"))
        log(f"env : {env['host']} python {env['python']} torch {env['torch']} cuda {env['cuda']} numpy {env['numpy']} "
            f"gpus {env['gpus']} slurm job {env['slurm'].get('SLURM_JOB_ID', '-')}")
    if ddp:
        dist.barrier()

    # ---- data: splits, selection, reports
    sets = data_sets(D, cfg["model"].get("gamma"))
    names = [s["name"] for s in sets]
    splits, parts, vals = [], [], []
    for s in sets:
        sp = resolve_split(s, o["data_seed"], o["min_val_recordings"])
        if not sp["train"]:
            raise SystemExit(f"{s['name']}: no converted training sequences under {s['root']} (convert it first, or "
                             f"leave it out with --datasets)")
        if not sp["val"] and s["val"] != "none" and not o["allow_no_val"] and not o["dry_run"]:
            raise SystemExit(f"{s['name']}: {sp['val_note']}. Train without validation for it with --allow-no-val")
        p = make_dataset(D, s, True, sp["train"])
        p.select(s["fraction"], o["data_seed"], s["select"])
        splits.append(sp)
        parts.append(p)
        vals.append(make_dataset(D, s, False, sp["val"]) if sp["val"] else None)
    if D.get("weights") == "size":
        for s, p in zip(sets, parts):
            s["weight"] = float(len(p))
    report = X.dataset_report(D, T, sets, splits, parts, vals, world, T["steps"])
    sampler = MixSampler([len(p) for p in parts], [s["weight"] for s in sets], rank=rank, world=world, seed=T["seed"])
    loader = DataLoader(torch.utils.data.ConcatDataset(parts), T["batch"], sampler=sampler, num_workers=D["workers"],
                        collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=D["workers"] > 0)
    if rank == 0:
        X.print_report(report, log)
        report["batches"] = X.inspect_batches(parts, loader, T["batch"], o["inspect_samples"], log)
        report["selection"] = dict(data_seed=o["data_seed"], selected={p.name: [p.sample_id(n) for n in range(min(len(p), 20))]
                                                                       for p in parts})
        X.write_json(os.path.join(exp, "dataset_report.json"), report)
        cfg_file = os.path.join(exp, "config.yaml")
        if not o["resume"]:
            open(cfg_file, "w").write(X.config_text(cfg))
    if o["dry_run"]:
        if rank == 0:
            log("dry run: datasets and DataLoader inspected, no model built, no optimizer step taken")
            X.write_json(os.path.join(exp, "summary.json"), dict(status="dry_run", report="dataset_report.json"))
        if ddp:
            dist.destroy_process_group()
        return "dry_run"

    # ---- model, optimizer, resume
    model = build_model(cfg).to(dev)
    prog = dict(step=0, epoch=0, in_epoch=0, samples=0, seen=[0] * len(sets), batches_with=[0] * len(sets), best=None,
                world=world)
    ck = None
    if o["resume"]:
        ck = torch.load(os.path.join(ck_dir, "last.pt"), map_location=dev, weights_only=False)
        diff_ = X.resume_diff(ck["cfg"], cfg)
        if diff_ and not o["force_resume"]:
            raise SystemExit("resume refused, settings differ from the checkpoint (use --force-resume to accept):\n"
                             + "\n".join(f"  {k}: {a} -> {b}" for k, a, b in diff_))
        model.load_state_dict(ck["model"], strict=False)
        prog.update(ck["progress"])
        if rank == 0:
            log(f"resumed from step {prog['step']} ({prog['samples']} samples seen)"
                + (f"; accepted setting changes: {diff_}" if diff_ else ""))
    elif T.get("init"):  # parts whose size changed (e.g. a wider student) start fresh
        sd, own = torch.load(T["init"], map_location=dev, weights_only=False)["model"], model.state_dict()
        skip = sorted({k.split(".")[0] for k, v in sd.items() if k in own and own[k].shape != v.shape})
        model.load_state_dict({k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}, strict=False)
        if rank == 0:
            log(f"init from {T['init']}" + (f"; size changed, not loaded: {skip}" if skip else ""))
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(TRAIN[stage]) and not model.frozen(n)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for p in params if p.ndim > 1], "weight_decay": T["wd"]},
                             {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0}],
                            lr=T["lr"], betas=(0.9, 0.99))
    if ck is not None:
        opt.load_state_dict(ck["opt"])
        if ck.get("rng") and len(ck["rng"]) == world:
            X.set_rng_state(ck["rng"][rank])
        else:
            seed_all(T["seed"] + rank + 1000003 * prog["step"])
        if prog.get("world") != world:  # another GPU count splits the epoch differently: start a fresh epoch
            prog.update(epoch=prog["epoch"] + 1, in_epoch=0, world=world)
        del ck
    else:
        seed_all(T["seed"] + rank)  # the batch inspection above used the generator
    net = Step(model, cfg["loss"], stage)
    net = DDP(net, device_ids=[torch.cuda.current_device()] if cuda else None) if ddp else net
    if rank == 0:
        log(f"model: stage {stage}, {sum(p.numel() for p in params) / 1e6:.2f} M trainable parameters "
            f"({', '.join(TRAIN[stage])}); no LR scheduler object (warmup + cosine from the step), no grad scaler (bf16)")

    # ---- validation (validation split only; the test split is never read during training)
    val_loaders = {}
    if rank == 0:
        for s, v in zip(sets, vals):
            if v is not None:
                sub = torch.utils.data.Subset(v, np.linspace(0, len(v) - 1, min(len(v), T["val_batches"] * T["batch"]))
                                              .round().astype(int).tolist())
                val_loaders[s["name"]] = DataLoader(sub, T["batch"], num_workers=D["workers"], collate_fn=collate)

    @torch.no_grad()
    def validate(loader_):
        model.eval()
        sums, n = {}, 0
        for batch in loader_:
            batch = {k: v.to(dev) for k, v in batch.items()}
            if stage == "probe":
                out = dict(probe=compute(model, batch, cfg["loss"], stage)[0].item())
            else:
                out = diagnostics(model, batch, cfg["loss"], student=stage != "teacher")
            sums = {k: sums.get(k, 0) + v for k, v in out.items()}
            n += 1
        model.train()
        return {k: v / max(n, 1) for k, v in sums.items()}

    def checkpoint_obj():
        return dict(model=model.state(), opt=opt.state_dict(), cfg=cfg, progress=dict(prog), names=names,
                    selection={p.name: p.selection for p in parts})

    # ---- training loop
    train_csv = X.CSVLog(os.path.join(exp, "train_metrics.csv")) if rank == 0 else None
    val_csv = X.CSVLog(os.path.join(exp, "val_metrics.csv")) if rank == 0 else None
    target = np.array([s["weight"] for s in sets], dtype=float)
    target = target / target.sum()
    stop = dict(signal=None)

    def on_signal(sig, frame):
        stop["signal"] = signal.Signals(sig).name

    for sg in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sg, on_signal)

    keys, acc, sums, nsum = None, None, {}, 0
    cnt = np.zeros(len(sets), np.int64)
    bwith = np.zeros(len(sets), np.int64)
    step, status = prog["step"], None
    tic = time.time()
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    model.train()
    if rank == 0:
        log("=" * 100)
        log(f"training: steps {step} -> {T['steps']}, log every {T['log_every']}, validate every {T['val_every']}, "
            f"save every {T['save_every']}")
        log("per-dataset losses: exact split of the batch terms that are means over samples (nll, nll_student, photo, "
            "smooth, cmax, jepa, probe); sigreg is a statistic of the whole batch and has no per-dataset value; photo is "
            "N/A (not 0) for samples without hidden frames")

    def reduce(t, op=dist.ReduceOp.SUM if ddp else None):
        if ddp:
            dist.all_reduce(t, op=op)
        return t

    while step < T["steps"] and status is None:
        sampler.set_epoch(prog["epoch"])
        sampler.start = prog["in_epoch"] * T["batch"]
        for batch in loader:
            ds_cpu = batch["ds"].numpy()
            meta = (batch["ds"].clone(), batch["sid"].clone(), batch["crop_info"].clone())
            present = np.bincount(ds_cpu, minlength=len(sets))
            cnt += present
            bwith += present > 0
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            for g in opt.param_groups:
                g["lr"] = lr_at(T, step)
            loss, terms, per = net(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, T["clip"])
            flags = torch.tensor([float(not torch.isfinite(loss)), float(not torch.isfinite(gnorm)),
                                  float(stop["signal"] is not None)], device=dev)
            flags = reduce(flags, dist.ReduceOp.MAX if ddp else None).tolist()
            if flags[0] or flags[1]:
                rep = X.nonfinite_report(os.path.join(exp, f"nonfinite_step{step + 1}_rank{rank}.json"), step + 1,
                                         loss.item(), {k: v.item() for k, v in terms.items()}, gnorm.item(), meta, parts,
                                         names, rank)
                if rank == 0:
                    log(f"STOP: non-finite {'loss' if flags[0] else 'gradient'} at step {step + 1} (on some GPU); "
                        f"this GPU: loss {rep['loss']}, grad norm {rep['grad_norm']}, non-finite terms "
                        f"{rep['nonfinite_terms']}; batch: {rep['batch'][:4]} ... -> nonfinite_step{step + 1}_rank*.json")
                status = "nonfinite"
                break
            opt.step()
            step += 1
            prog["in_epoch"] += 1
            prog["samples"] += T["batch"] * world

            # interval sums (no host sync)
            for k, v in [("loss", loss), ("grad_norm", gnorm)] + list(terms.items()):
                sums[k] = sums.get(k, 0) + v.detach().float()
            nsum += 1
            if keys is None:
                keys = sorted(k for k in per if not k.endswith("_cnt"))
                acc = torch.zeros(len(sets), len(keys), 2, device=dev)
            ds = batch["ds"]
            for j, k in enumerate(keys):
                v = per[k].float()
                c = per[k[:-4] + "_cnt"].float() if k.endswith("_sum") else torch.ones_like(v)
                acc[:, j, 0].index_add_(0, ds, v)
                acc[:, j, 1].index_add_(0, ds, c)

            if step % T["log_every"] == 0 or step == T["steps"]:
                tot = reduce(torch.stack([sums[k] for k in sorted(sums)])) / world
                a = reduce(acc.clone())
                c_all = reduce(torch.tensor(np.concatenate([cnt, bwith]), device=dev, dtype=torch.float64)).cpu().numpy()
                mem = reduce(torch.tensor([torch.cuda.max_memory_allocated(dev) / 2 ** 30 if cuda else 0.0], device=dev),
                             dist.ReduceOp.MAX if ddp else None).item()
                dt_step = (time.time() - tic) / nsum
                if rank == 0:
                    vals_ = dict(zip(sorted(sums), (tot / nsum).tolist()))
                    seen_int, bwith_int = c_all[:len(sets)], c_all[len(sets):]
                    prog["seen"] = [int(x + y) for x, y in zip(prog["seen"], seen_int)]
                    prog["batches_with"] = [int(x + y) for x, y in zip(prog["batches_with"], bwith_int)]
                    obs = np.array(prog["seen"], float) / max(1, sum(prog["seen"]))
                    photo_n = sum(a[d, keys.index("photo_sum"), 1].item() for d in range(len(sets))) if "photo_sum" in keys else None
                    row = dict(step=step, epoch=prog["epoch"], samples_seen=prog["samples"], lr=lr_at(T, step - 1),
                               loss=vals_["loss"], grad_norm=vals_["grad_norm"], step_time_s=dt_step,
                               samples_per_s=T["batch"] * world / dt_step, gpu_mem_gb=mem,
                               c=model.c.item(), r=model.r.item(), k=model.log_k.exp().item(), nu=model.log_nu.exp().item(),
                               R_us=model.R.item() * 1e6)
                    for k in sorted(terms):
                        row[k] = None if (k == "photo" and photo_n == 0) else vals_[k]
                    per_line = []
                    for d, nm in enumerate(names):
                        row[f"{nm}_samples"] = int(seen_int[d])
                        row[f"{nm}_share_observed"] = float(obs[d])
                        row[f"{nm}_share_target"] = float(target[d])
                        parts_txt = []
                        for j, k in enumerate(keys):
                            col = k[:-4] if k.endswith("_sum") else k
                            sm, ct = a[d, j, 0].item(), a[d, j, 1].item()
                            row[f"{nm}_{col}"] = sm / ct if ct else None
                            if col in ("nll", "nll_student", "photo", "jepa", "probe"):
                                parts_txt.append(f"{col} {sm / ct:.4f}" if ct else f"{col} N/A")
                        per_line.append(f"{nm} {100 * obs[d]:.1f}% (target {100 * target[d]:.1f}%) " + " ".join(parts_txt))
                    train_csv.write(row)
                    tv = " ".join(f"{k}{'(w=0)' if cfg['loss'].get(k, 1) == 0 else ''} "
                                  + ("N/A" if row[k] is None else f"{row[k]:.4f}") for k in sorted(terms))
                    log(f"[{stage}] step {step}/{T['steps']} | loss {row['loss']:.4f} | {tv} | lr {row['lr']:.1e} "
                        f"gn {row['grad_norm']:.2f} | {dt_step:.2f} s/step {row['samples_per_s']:.0f} samples/s | "
                        f"mem {mem:.1f} GB | seen {prog['samples']}")
                    log("    " + " | ".join(per_line))
                sums, nsum, cnt[:], bwith[:] = {}, 0, 0, 0
                acc.zero_()
                tic = time.time()
                if cuda:
                    torch.cuda.reset_peak_memory_stats(dev)

            if rank == 0 and val_loaders and (step % T["val_every"] == 0 or step == T["steps"]):
                scores = []
                for nm, vl in val_loaders.items():
                    res = validate(vl)
                    val_csv.write(dict(step=step, dataset=nm, samples=len(vl.dataset), **res))
                    log(f"val[{nm}] step {step} " + " ".join(f"{k} {v:.4f}" for k, v in res.items()))
                    scores.append(res.get(MAIN[stage], res.get("nll")))
                score = float(np.mean(scores))
                if prog["best"] is None or score < prog["best"]["score"]:
                    prog["best"] = dict(step=step, score=score, metric=f"mean over datasets of val {MAIN[stage]}")
                    X.save(os.path.join(ck_dir, "best.pt"), checkpoint_obj())
                    log(f"best.pt <- step {step} (mean val {MAIN[stage]} {score:.4f})")
                tic = time.time()

            save_now = step % T["save_every"] == 0 or step == T["steps"] or flags[2]
            if save_now:
                states = [None] * world
                if ddp:
                    dist.all_gather_object(states, X.rng_state())
                else:
                    states = [X.rng_state()]
                if rank == 0:
                    obj = dict(checkpoint_obj(), rng=states)
                    X.save(os.path.join(ck_dir, "last.pt"), obj)
                    if step % T["save_every"] == 0 and T["keep"] > 0:
                        X.save(os.path.join(ck_dir, f"step_{step:07d}.pt"), obj)
                        X.prune(ck_dir, T["keep"])
            if flags[2]:
                if rank == 0:
                    log(f"STOP: received {stop['signal'] or 'a signal on another GPU'} at step {step}; last.pt saved, "
                        f"resume with --resume")
                status = "signal"
                break
            if step >= T["steps"]:
                break
        else:
            prog["epoch"] += 1
            prog["in_epoch"] = 0
    status = status or "completed"

    if rank == 0:
        obs = np.array(prog["seen"], float) / max(1, sum(prog["seen"]))
        contrib = {nm: dict(samples_seen=prog["seen"][d], batches_containing=prog["batches_with"][d],
                            share_observed=float(obs[d]), share_target=float(target[d])) for d, nm in enumerate(names)}
        X.write_json(os.path.join(exp, "summary.json"),
                     dict(status=status, stage=stage, step=step, steps_planned=T["steps"], samples_seen=prog["samples"],
                          global_batch=T["batch"] * world, contribution=contrib, best=prog["best"],
                          batches="mixed: each sample picks its dataset independently (weights per sample)",
                          checkpoints=sorted(os.listdir(ck_dir)), finished=time.strftime("%Y-%m-%d %H:%M:%S")))
        log("=" * 100)
        log(f"{status} at step {step}: {prog['samples']} samples seen. Contribution per dataset:")
        for nm, c in contrib.items():
            log(f"  {nm:8s} {c['samples_seen']:10d} samples, in {c['batches_containing']:8d} batches, "
                f"{100 * c['share_observed']:.1f}% observed vs {100 * c['share_target']:.1f}% target")
        if prog["best"]:
            log(f"best.pt: step {prog['best']['step']}, {prog['best']['metric']} {prog['best']['score']:.4f}")
    if ddp:
        dist.destroy_process_group()
    return status


@torch.no_grad()
def final_test(ckpt, out, samples=1000, batch=8, workers=4, data_cfg=None):
    # held-out test split, evaluated once on an explicit request (never during training or for checkpoint selection)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    D, stage = data_cfg or cfg["data"], cfg["train"]["stage"]
    model = build_model(cfg).to(dev).eval()
    model.load_state_dict(ck["model"], strict=False)
    os.makedirs(out, exist_ok=True)
    log = X.Log(os.path.join(out, "test.log"))
    csv_ = X.CSVLog(os.path.join(out, "test_metrics.csv"))
    log(f"final test of {ckpt} (step {ck.get('progress', {}).get('step')}) -> {out}")
    for s in data_sets(D, cfg["model"].get("gamma")):
        dirs = resolve_split(s)["test"]
        if not dirs:
            log(f"{s['name']}: no test_root, skipped")
            continue
        v = make_dataset(D, s, False, dirs)
        sub = torch.utils.data.Subset(v, np.linspace(0, len(v) - 1, min(len(v), samples)).round().astype(int).tolist())
        sums, n = {}, 0
        for b in DataLoader(sub, batch, num_workers=workers, collate_fn=collate):
            b = {k: t.to(dev) for k, t in b.items()}
            res = dict(probe=compute(model, b, cfg["loss"], stage)[0].item()) if stage == "probe" else \
                diagnostics(model, b, cfg["loss"], student=stage != "teacher")
            sums = {k: sums.get(k, 0) + x for k, x in res.items()}
            n += 1
        res = {k: x / max(n, 1) for k, x in sums.items()}
        csv_.write(dict(dataset=s["name"], samples=len(sub), checkpoint=ckpt, **res))
        log(f"test[{s['name']}] {len(sub)} samples " + " ".join(f"{k} {x:.4f}" for k, x in res.items()))


if __name__ == "__main__":
    # python train.py [config.yaml] key=value ...   (experiment folder = train.out; train.resume=true to continue it)
    c = load_cfg(sys.argv[1:])
    st = run(c, resume=bool(c["train"].get("resume", False)))
    sys.exit(3 if st == "signal" else 1 if st == "nonfinite" else 0)
