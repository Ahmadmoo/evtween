"""Check an experiment folder written by run_training.py and print PASS / WARN / FAIL per item.

  python scripts/check_run.py runs/smoke_1pct_b [--full-steps 200000]

Reads only the files of the run (and loads its checkpoints with torch if available); trains nothing.
--full-steps: also estimate the wall time of a run of that many steps from the measured step time."""
import argparse
import csv
import json
import math
import os
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("exp")
ap.add_argument("--full-steps", type=int, default=200000)
a = ap.parse_args()
E = a.exp
res = []


def check(ok, what, detail="", warn=False):
    res.append(("PASS" if ok else "WARN" if warn else "FAIL", what, detail))


def num(v):
    try:
        x = float(v)
        return x if v != "" else None
    except (TypeError, ValueError):
        return None


def rows(name):
    p = os.path.join(E, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None


# ---- files
need = ["config.yaml", "dataset_report.json", "train_metrics.csv", "training.log", "summary.json", "checkpoints/last.pt"]
missing = [f for f in need if not os.path.exists(os.path.join(E, f))]
check(not missing, "experiment files present", f"missing {missing}" if missing else "")
if not os.path.exists(os.path.join(E, "dataset_report.json")):
    print("\n".join(f"{s:4s}  {w}  {d}" for s, w, d in res))
    sys.exit(1)
rep = json.load(open(os.path.join(E, "dataset_report.json")))
summ = json.load(open(os.path.join(E, "summary.json"))) if os.path.exists(os.path.join(E, "summary.json")) else {}
names = [d["name"] for d in rep["datasets"]]
hidden = {d["name"]: d["hidden_frames"] for d in rep["datasets"]}

# ---- data
for d in rep["datasets"]:
    check(d["selected_samples"] > 0, f"{d['name']}: training samples selected", f"{d['selected_samples']} of {d['eligible_samples']}")
    check(d["val_samples"] > 0, f"{d['name']}: validation samples", f"{d['val_samples']} ({d['validation']['note']})",
          warn=True)
    ex = len(d["excluded_train"]) + len(d["excluded_val"])
    check(ex == 0, f"{d['name']}: no excluded sequences", f"{ex} excluded, see dataset_report.json", warn=True)
    if d["gap_ms"][1] > 2 * d["gap_ms"][0] + 1:
        check(False, f"{d['name']}: gap range", f"{d['gap_ms'][0]:.0f}-{d['gap_ms'][1]:.0f} ms (slow sequences)", warn=True)

# ---- status
st = summ.get("status")
check(st == "completed", "run finished", f"status {st}, step {summ.get('step')} of {summ.get('steps_planned')}",
      warn=st == "signal")

# ---- training metrics
tr = rows("train_metrics.csv") or []
check(len(tr) > 0, "train_metrics.csv has rows", f"{len(tr)} rows")
if tr:
    steps = [int(r["step"]) for r in tr]
    dup = len(steps) - len(set(steps))
    check(steps == sorted(steps), "steps increase", "")
    check(dup == 0, "no repeated steps", f"{dup} repeated (expected only after a crash between checkpoints)", warn=True)
    base = ["loss", "grad_norm", "lr", "step_time_s", "samples_per_s", "gpu_mem_gb", "c", "r", "k", "nu", "R_us"]
    bad = [(r["step"], k) for r in tr for k in base if k in r and (num(r[k]) is None or not math.isfinite(num(r[k])))]
    check(not bad, "loss, grad norm, lr, timing, sensor parameters finite in every row", f"bad: {bad[:5]}")
    terms = [k for k in tr[0] if k not in base and k not in ("step", "epoch", "samples_seen")
             and not any(k.startswith(n + "_") for n in names)]
    for t in terms:
        vals = [num(r[t]) for r in tr]
        na = sum(v is None for v in vals)
        badv = [v for v in vals if v is not None and not math.isfinite(v)]
        check(not badv and na < len(vals), f"term {t}: finite", f"{len(vals) - na} values, {na} N/A")
    lr = [num(r["lr"]) for r in tr]
    check(all(0 < x for x in lr), "learning rate positive", f"{lr[0]:.2e} -> {lr[-1]:.2e}")
    # per dataset
    for n in names:
        seen = sum(int(num(r.get(f"{n}_samples")) or 0) for r in tr)
        check(seen > 0, f"{n}: samples seen in training", f"{seen}")
        nll_col = f"{n}_nll" if f"{n}_nll" in tr[0] else f"{n}_nll_student" if f"{n}_nll_student" in tr[0] else None
        if nll_col:
            v = [num(r[nll_col]) for r in tr if num(r.get(f"{n}_samples")) and num(r[f"{n}_samples"]) > 0]
            check(v and all(x is not None and math.isfinite(x) for x in v), f"{n}: per-dataset {nll_col[len(n) + 1:]} finite",
                  f"{len(v)} values")
        if f"{n}_photo" in tr[0]:
            v = [num(r[f"{n}_photo"]) for r in tr]
            if hidden[n][1] == 0:
                check(all(x is None for x in v), f"{n}: photo N/A (no hidden frames)", "")
            else:
                check(any(x is not None for x in v), f"{n}: photo measured ({hidden[n][0]}-{hidden[n][1]} hidden frames)",
                      f"{sum(x is not None for x in v)} values")
    last = tr[-1]
    tot = sum(int(num(r.get(f"{n}_samples")) or 0) for r in tr for n in names)
    for n in names:
        p, obs = num(last[f"{n}_share_target"]), num(last[f"{n}_share_observed"])
        tol = 4 * math.sqrt(p * (1 - p) / max(tot, 1)) + 0.01
        check(abs(obs - p) <= tol, f"{n}: sampling share", f"{100 * obs:.1f}% observed vs {100 * p:.1f}% target "
              f"(tolerance +-{100 * tol:.1f}% for {tot} samples)", warn=True)
    if len(tr) >= 4:
        h = len(tr) // 2
        first = sum(num(r["loss"]) for r in tr[:h]) / h
        second = sum(num(r["loss"]) for r in tr[h:]) / (len(tr) - h)
        check(second < first, "loss goes down (second half vs first half)", f"{first:.4f} -> {second:.4f}",
              warn=True)
    times = sorted(num(r["step_time_s"]) for r in tr[1:] or tr)
    t = times[len(times) // 2]
    mem = max(num(r["gpu_mem_gb"]) for r in tr)
    check(mem < 75, "peak GPU memory", f"{mem:.1f} GB (H100 has 80 GB)", warn=mem >= 75)
    check(True, "speed", f"{t:.2f} s/step median, {num(last['samples_per_s']):.1f} samples/s; {a.full_steps} steps "
                         f"~ {t * a.full_steps / 3600:.1f} h of GPU time")

# ---- validation
va = rows("val_metrics.csv") or []
with_val = [d["name"] for d in rep["datasets"] if d["val_samples"] > 0]
check(bool(va) or not with_val, "val_metrics.csv has rows", f"{len(va)} rows")
for n in with_val:
    v = [r for r in va if r["dataset"] == n]
    finite = all(num(x) is not None and math.isfinite(num(x)) for r in v for k, x in r.items() if k not in ("dataset",))
    check(bool(v) and finite, f"{n}: validation rows finite", f"{len(v)} rows")

# ---- checkpoints
ck_dir = os.path.join(E, "checkpoints")
files = sorted(os.listdir(ck_dir)) if os.path.isdir(ck_dir) else []
check("last.pt" in files, "checkpoints/last.pt", f"{files}")
check("best.pt" in files or not with_val, "checkpoints/best.pt", "")
try:
    import torch
    ck = torch.load(os.path.join(ck_dir, "last.pt"), map_location="cpu", weights_only=False)
    keys = {"model", "opt", "cfg", "progress", "rng"}
    check(keys <= set(ck), "last.pt holds model, optimizer, config, progress, random states", f"{sorted(ck)}")
    check(ck["progress"]["step"] == summ.get("step"), "last.pt step matches the run", f"{ck['progress']['step']}")
    check(len(ck.get("rng") or []) == ck["progress"].get("world", 1), "random states for every GPU",
          f"{len(ck.get('rng') or [])}")
    n_params = sum(v.numel() for v in ck["model"].values())
    check(n_params > 0, "trainable weights saved", f"{n_params / 1e6:.2f} M values")
except ImportError:
    check(False, "checkpoint contents", "not checked (no torch here)", warn=True)

# ---- log
log = open(os.path.join(E, "training.log")).read() if os.path.exists(os.path.join(E, "training.log")) else ""
check("Traceback" not in log and "STOP: non-finite" not in log, "no errors or non-finite stops in training.log", "")
if "resumed from step" in log:
    check(True, "resume happened", log[log.index("resumed from step"):].split("\n")[0])

print(f"\ncheck of {os.path.realpath(E)}\n")
for s, w, d in res:
    print(f"{s:4s}  {w}" + (f"  -- {d}" if d else ""))
nf = sum(s == "FAIL" for s, _, _ in res)
nw = sum(s == "WARN" for s, _, _ in res)
print(f"\n{len(res) - nf - nw} passed, {nw} warnings, {nf} failed")
sys.exit(1 if nf else 0)
