"""Command line for training: flags -> resolved config -> train.run() (the one training loop, in train.py).

  python run_training.py --stage teacher --fraction 0.1 --steps 5000 --exp runs/pilot
  torchrun --nproc_per_node 4 run_training.py --stage teacher --exp runs/teacher_full
  python run_training.py --exp runs/teacher_full --resume            (settings from runs/teacher_full/config.yaml)
  python run_training.py --final-test runs/teacher_full/checkpoints/best.pt

Anything without a flag: --set section.key=value (same as train.py's dotted overrides)."""
import argparse
import copy
import os
import sys
import yaml

ALIASES = {"hqevfi": "hqevfi", "bsergb": "bsergb", "ced": "ced", "eds": "eds"}
norm = lambda n: ALIASES.get(n.lower().replace("-", "").replace("_", ""), n)


def fraction(v):
    f = float(v[:-1]) / 100 if str(v).endswith("%") else float(v)
    if not 0 < f <= 1:
        raise argparse.ArgumentTypeError(f"fraction {v} not in (0, 1] (or 1%..100%)")
    return f


def pairs(items, conv, what):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"{what}: expected name=value, got {it}")
        k, v = it.split("=", 1)
        out[norm(k)] = conv(v)
    return out


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--config", default=None, help="default: config.yaml; with --resume: the experiment's own config.yaml")
ap.add_argument("--exp", help="experiment folder (must not exist, unless --resume)")
ap.add_argument("--resume", action="store_true", help="continue --exp from its checkpoints/last.pt")
ap.add_argument("--force-resume", action="store_true", help="resume even if settings differ from the checkpoint")
ap.add_argument("--dry-run", action="store_true", help="dataset report + DataLoader inspection only, no model, no step")
ap.add_argument("--final-test", metavar="CKPT", help="evaluate CKPT once on the held-out test split and exit")
ap.add_argument("--test-samples", type=int, default=1000, help="test samples per dataset for --final-test (evenly spaced)")
g = ap.add_argument_group("what to train")
g.add_argument("--stage", choices=["probe", "teacher", "student", "joint"])
g.add_argument("--init", help="checkpoint to start from (e.g. the teacher's best.pt for the student stage)")
g.add_argument("--datasets", nargs="+", help="hqevfi bsergb ced eds (HQ-EVFI etc. also accepted); default: all in the config")
g = ap.add_argument_group("data selection")
g.add_argument("--fraction", type=fraction, help="fraction of the eligible training samples of every dataset (0.1 or 10%%)")
g.add_argument("--dataset-fraction", nargs="+", metavar="NAME=F", help="per dataset, e.g. eds=0.25 ced=50%%")
g.add_argument("--select", choices=["sample", "sequence"], help="select single samples (default) or whole sequences")
g.add_argument("--weights", nargs="+", metavar="NAME=W", help="sampling weights, e.g. hqevfi=1 bsergb=2 (default: equal); "
               "or the single word 'size' for weights proportional to the selected samples")
g.add_argument("--data-seed", type=int, default=0, help="seed of the validation split and the data selection")
g.add_argument("--val-frac", type=float, help="share of training recordings held out for validation (default 0.1)")
g.add_argument("--allow-no-val", action="store_true", help="train even if a dataset gets no validation split")
g.add_argument("--raw", nargs="+", metavar="NAME=DIR", help="original downloads, only for the report's raw counts")
g = ap.add_argument_group("optimization and bookkeeping")
for k, t, h in (("seed", int, "training seed (augmentation, sampling order, init)"), ("batch", int, "samples per GPU"),
                ("steps", int, "optimizer steps"), ("lr", float, "peak learning rate"), ("warmup", int, "warmup steps"),
                ("log-every", int, "steps between log lines / CSV rows"), ("val-every", int, "steps between validations"),
                ("save-every", int, "steps between checkpoints"), ("keep", int, "step_*.pt snapshots kept"),
                ("val-batches", int, "validation batches per dataset"), ("workers", int, "DataLoader workers per GPU"),
                ("inspect-samples", int, "samples per dataset for the startup crop statistics")):
    g.add_argument(f"--{k}", type=t, help=h)
ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="other config overrides, e.g. data.crop=192")
a = ap.parse_args()

import train  # noqa: E402  (after argparse: --help works without torch)

if a.final_test:
    cfg = yaml.safe_load(open(a.config or "config.yaml"))
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.final_test))),
                       "test_" + os.path.splitext(os.path.basename(a.final_test))[0])
    train.final_test(a.final_test, out, a.test_samples, workers=a.workers or 4)
    sys.exit(0)

saved = os.path.join(a.exp or "", "config.yaml")
if a.config is None and a.resume and a.exp and os.path.exists(saved):
    a.config = saved  # resume: the run's own settings; flags given now are applied on top (and checked)
    print(f"resume: settings from {saved}")
cfg = train.load_cfg([a.config or "config.yaml"] + a.set)
D, T = cfg["data"], cfg["train"]
if a.stage:
    T["stage"] = a.stage
if a.init:
    T["init"] = a.init
if a.exp:
    T["out"] = a.exp
for k in ("seed", "batch", "steps", "lr", "warmup", "log_every", "val_every", "save_every", "keep", "val_batches",
          "inspect_samples"):
    v = getattr(a, k)
    if v is not None:
        T[k] = v
if a.workers is not None:
    D["workers"] = a.workers
if a.select:
    D["select"] = a.select
if a.val_frac is not None:
    D["val_frac"] = a.val_frac
sets = {s["name"]: s for s in D["sets"]}
if a.datasets:
    want = [norm(n) for n in a.datasets]
    missing = [n for n in want if n not in sets]
    if missing:
        raise SystemExit(f"--datasets: unknown {missing}; config has {list(sets)}")
    D["use"] = want
if a.fraction is not None:
    D["fraction"] = a.fraction
for n, f in pairs(a.dataset_fraction, fraction, "--dataset-fraction").items():
    if n not in sets:
        raise SystemExit(f"--dataset-fraction: unknown dataset {n}")
    sets[n]["fraction"] = f
if a.weights == ["size"]:
    D["weights"] = "size"
elif a.weights:
    w = pairs(a.weights, float, "--weights")
    for n in w:
        if n not in sets:
            raise SystemExit(f"--weights: unknown dataset {n}")
    for n, s in sets.items():  # datasets not named keep weight 1
        s["weight"] = w.get(n, 1.0)
for n, p in pairs(a.raw, str, "--raw").items():
    if n not in sets:
        raise SystemExit(f"--raw: unknown dataset {n}")
    sets[n]["raw"] = p

status = train.run(copy.deepcopy(cfg), exp=T["out"], resume=a.resume, force_resume=a.force_resume, dry_run=a.dry_run,
                   allow_no_val=a.allow_no_val, data_seed=a.data_seed, inspect_samples=T.get("inspect_samples", 32))
sys.exit(3 if status == "signal" else 1 if status == "nonfinite" else 0)
