"""Experiment bookkeeping for train.run(): the experiment folder, startup reports, metric files, checkpoints and
provenance. Nothing here trains, changes a loss or touches the model."""
import copy
import csv
import glob
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
import numpy as np
import torch
import yaml
from data import collate, recording

IMG = (".png", ".jpg", ".jpeg")
TITLES = {"hqevfi": "HQ-EVFI", "bsergb": "BS-ERGB", "ced": "CED", "eds": "EDS"}
# config keys that may change when resuming (bookkeeping, not what is learned or from which data)
RESUME_FREE = {("train", k) for k in ("steps", "log_every", "val_every", "save_every", "keep", "val_batches", "out", "init",
                                       "inspect_samples", "resume")} | {("data", "workers")}
AUGMENT = [
    "random crop {crop}x{crop} (training): uniform position, re-drawn up to 10 times until at least min_events events of the "
    "gap fall inside; if no draw reaches it the 10th is used (a soft preference, no sample is ever dropped)",
    "horizontal flip with p = 0.5 (training): frames, hidden frames, context frames and events flipped together",
    "validation / test: centre crop, no flip, no event threshold",
    "applied online, per sample, in the DataLoader workers; nothing is stored, the number of unique samples and the "
    "dataset length do not change",
]


class Log:
    # rank-0 log: prints and appends to training.log
    def __init__(self, path=None):
        self.f = open(path, "a", buffering=1) if path else None

    def __call__(self, *parts):
        msg = " ".join(str(p) for p in parts)
        print(msg, flush=True)
        if self.f:
            self.f.write(msg + "\n")


def prepare_dir(exp, resume, dry_run):
    last = os.path.join(exp, "checkpoints", "last.pt")
    if resume:
        if not os.path.exists(last):
            raise SystemExit(f"--resume: {last} not found, nothing to resume")
    elif os.path.isdir(exp) and os.listdir(exp):
        raise SystemExit(f"{exp} already exists and is not empty: nothing was changed. Continue it with --resume, or pick "
                         f"a new --exp")
    os.makedirs(os.path.join(exp, "checkpoints"), exist_ok=True)


# ---------------------------------------------------------------- provenance
def git_info(repo):
    run = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()
    try:
        info = dict(repo=os.path.realpath(repo), commit=run("rev-parse", "HEAD"), branch=run("branch", "--show-current"),
                    status=run("status", "--short").splitlines())
        info["dirty"] = any(not l.startswith("??") for l in info["status"])
        return info, run("diff", "HEAD")
    except FileNotFoundError:
        return dict(repo=os.path.realpath(repo), error="git not found"), ""


def env_info():
    cuda = torch.cuda.is_available()
    return dict(time=time.strftime("%Y-%m-%d %H:%M:%S %Z"), host=socket.gethostname(), cwd=os.getcwd(),
                command=" ".join(sys.argv), python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version() if cuda else None,
                gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
                slurm={k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
                visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------- dataset report
def count_images(d):
    return sum(1 for e in os.scandir(d) if e.name.lower().endswith(IMG)) if os.path.isdir(d) else 0


def raw_inventory(name, path):
    # recordings and frames of the original download (best effort, by layout); None if no raw path was given
    if not path:
        return None
    if not os.path.exists(path):
        return dict(path=path, error="raw path not found")
    kind = name.lower().replace("-", "").replace("_", "")
    recs = {}
    if kind == "hqevfi":
        for d in glob.glob(os.path.join(path, "**", "visual_RGB"), recursive=True):
            recs[os.path.basename(os.path.dirname(d))] = count_images(d)
    elif kind == "bsergb":
        for split in ("1_TEST", "2_VALIDATION", "3_TRAINING"):
            for d in glob.glob(os.path.join(path, "**", split, "*", "images"), recursive=True):
                recs[os.path.basename(os.path.dirname(d))] = count_images(d)
    elif kind == "ced":
        for b in glob.glob(os.path.join(path, "**", "*.bag"), recursive=True):
            recs[os.path.splitext(os.path.basename(b))[0]] = None  # frames are inside the bag (CED/ced.py --dry lists them)
    elif kind == "eds":
        for f in glob.glob(os.path.join(path, "**", "events*"), recursive=True):
            d = os.path.dirname(f)
            subs = [e.path for e in os.scandir(d) if e.is_dir() and any(k in e.name.lower() for k in ("image", "rgb", "frame"))]
            recs[os.path.basename(d)] = max((count_images(s) for s in subs), default=0)
    else:
        return dict(path=path, error=f"unknown layout for {name}")
    known = [v for v in recs.values() if v is not None]
    return dict(path=path, recordings=len(recs), frames=sum(known) if len(known) == len(recs) else None,
                frames_note=None if len(known) == len(recs) else "frames are inside the bags, not counted", names=recs)


def converted_counts(dirs):
    frames = sum(int(np.load(os.path.join(d, "frame_ts.npy"), mmap_mode="r").shape[0]) for d in dirs)
    return dict(sequences=len(dirs), recordings=len({recording(d) for d in dirs}), frames=frames)


def dataset_report(D, T, sets, splits, parts, vals, world, steps):
    gb = T["batch"] * world
    wsum = sum(s["weight"] for s in sets)
    rows = []
    for s, sp, p, v in zip(sets, splits, parts, vals):
        fps = np.array([q.fps for q in p.seqs])
        sk = np.array(p.skips)
        prob = s["weight"] / wsum
        conv_names = {os.path.basename(d) for k in ("train", "val", "test") for d in sp[k]}
        raw = raw_inventory(s["name"], s.get("raw"))
        if raw and "names" in raw:  # recordings of the download without any converted sequence
            raw["not_converted"] = sorted(r for r in raw["names"]
                                          if not any(c == r or c.startswith(r + "_") for c in conv_names))
            raw["not_converted_reason"] = "absent from the converted data: skipped by the converter or not converted yet " \
                                          "(its printed output gives the reason)"
            raw.pop("names")
        rows.append(dict(
            name=s["name"], title=TITLES.get(s["name"], s["name"]), raw=raw,
            converted=dict(train=converted_counts(sp["train"]), val=converted_counts(sp["val"]),
                           test=converted_counts(sp["test"])),
            validation=dict(mode=s["val"], note=sp["val_note"]),
            recordings_in_train_and_test=sp["train_test_shared_recordings"],
            excluded_train=[dict(sequence=n, reason=r) for n, r in p.excluded],
            excluded_val=[dict(sequence=n, reason=r) for n, r in v.excluded] if v is not None else [],
            train_sequences_used=p.selection["sequences"], eligible_samples=p.eligible,
            fraction_requested=s["fraction"], selection_level=p.selection["level"], selection_seed=p.selection["seed"],
            selected_samples=len(p), val_samples=len(v) if v is not None else 0,
            weight=s["weight"], dataset_probability=prob, per_sample_probability=prob / len(p),
            expected_draws_per_selected_sample=steps * gb * prob / len(p),
            fps=[float(fps.min()), float(fps.max())], hidden_frames=[int(sk.min()), int(sk.max())],
            gap_ms=[float(((sk + 1) / fps).min() * 1e3), float(((sk + 1) / fps).max() * 1e3)],
            resolutions=sorted({"%dx%d" % q.size for q in p.seqs}), crop=D["crop"],
            gamma=sorted({float(g) for g in p.gammas if g is not None}), min_events=s["min_events"]))
    return dict(datasets=rows, batch_per_gpu=T["batch"], gpus=world, global_batch=gb, steps=steps,
                samples_drawn=steps * gb,
                events=dict(voxel=f"{2 * D['bins']} channels: 0..{D['bins'] - 1} ON counts, {D['bins']}..{2 * D['bins'] - 1} "
                                  f"OFF counts, {D['bins']} equal time bins of the gap (teacher input)",
                            event_list="ev_pix (y*w+x in the crop), ev_tau (time in the gap, 0..1), ev_pol (+1/-1), "
                                       "ev_b (sample in the batch): exact events for the likelihood"),
                context_frames=2 * D["context"], augmentation=[a.format(crop=D["crop"]) for a in AUGMENT],
                sampling="mixed batches: every sample independently picks a dataset with probability weight / sum(weights), "
                         "then a selected sample of it uniformly, with replacement; weights act per sample")


def print_report(rep, log):
    log("=" * 100)
    log("DATASETS  (sequence = one converted recording or piece of it; sample = one (I0, I1) training pair; "
        "batch = %d samples per GPU x %d GPU)" % (rep["batch_per_gpu"], rep["gpus"]))
    for r in rep["datasets"]:
        c, raw = r["converted"], r["raw"]
        log("-" * 100)
        log(f"{r['title']}")
        if raw is None:
            log("  raw download      : not checked (give its path with --raw %s=<folder>)" % r["name"])
        elif "error" in raw:
            log(f"  raw download      : {raw['error']} ({raw['path']})")
        else:
            log(f"  raw download      : {raw['recordings']} recordings, "
                f"{raw['frames'] if raw['frames'] is not None else 'n/a'} frames"
                + (f" ({raw['frames_note']})" if raw.get("frames_note") else "")
                + (f"; not converted: {len(raw['not_converted'])} {raw['not_converted'][:6]}" if raw["not_converted"] else ""))
        for k in ("train", "val", "test"):
            log(f"  converted {k:5s}   : {c[k]['sequences']:4d} sequences from {c[k]['recordings']:4d} recordings, "
                f"{c[k]['frames']:8d} frames")
        log(f"  validation        : {r['validation']['mode']} - {r['validation']['note']}")
        if r["recordings_in_train_and_test"]:
            log(f"  note              : {len(r['recordings_in_train_and_test'])} recording(s) have pieces in both train and "
                f"test (the dataset's official split): {r['recordings_in_train_and_test'][:4]}")
        ex = r["excluded_train"] + r["excluded_val"]
        log(f"  excluded          : {len(ex)} sequence(s)" + "".join(f"\n      {e['sequence']}: {e['reason']}" for e in ex[:10])
            + (f"\n      ... {len(ex) - 10} more (dataset_report.json)" if len(ex) > 10 else ""))
        log(f"  training samples  : {r['eligible_samples']} eligible -> {r['selected_samples']} selected "
            f"({100 * r['fraction_requested']:g}% requested, {r['selection_level']} level, "
            f"{r['train_sequences_used']} sequences used); validation samples {r['val_samples']}")
        log(f"  sampling          : weight {r['weight']:g} -> {100 * r['dataset_probability']:.1f}% of drawn samples, "
            f"per selected sample p = {r['per_sample_probability']:.2e}, each drawn ~{r['expected_draws_per_selected_sample']:.1f}"
            f" times over the run")
        log(f"  timing            : {r['fps'][0]:.1f}-{r['fps'][1]:.1f} fps, {r['hidden_frames'][0]}-{r['hidden_frames'][1]} "
            f"hidden frames, gap {r['gap_ms'][0]:.1f}-{r['gap_ms'][1]:.1f} ms")
        log(f"  images            : {', '.join(r['resolutions'])} -> crop {r['crop']}x{r['crop']}; gamma "
            f"{'/'.join(f'{g:g}' for g in r['gamma']) or 'model default'}; min_events {r['min_events']} (crop retry, not a filter)")
    log("-" * 100)
    log(f"events     : {rep['events']['voxel']}")
    log(f"             {rep['events']['event_list']}")
    log(f"sampling   : {rep['sampling']}")
    log("augmentation:" + "".join(f"\n  - {a}" for a in rep["augmentation"]))
    log(f"run        : {rep['steps']} optimizer steps x {rep['global_batch']} samples = {rep['samples_drawn']} samples drawn")


# ---------------------------------------------------------------- batch inspection
def describe(batch):
    out = {}
    for k, v in batch.items():
        d = dict(shape=list(v.shape), dtype=str(v.dtype).replace("torch.", ""))
        if v.numel():
            x = v.float()
            d.update(min=float(x.min()), max=float(x.max()))
        out[k] = d
    return out


def inspect_batches(parts, loader, batch_size, n_samples, log):
    # one real batch per dataset (from the dataset itself, training augmentation on) and the first batch of the actual
    # training DataLoader (mixed); crop statistics over n_samples samples per dataset
    rep = dict(per_dataset={}, crop_stats={})
    log("=" * 100)
    log("BATCHES (real DataLoader output)")
    for p in parts:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(p), min(len(p), max(n_samples, batch_size)), replace=False)
        items = [p[int(i)] for i in idx]
        b = collate(items[:batch_size])
        info = describe(b)
        rep["per_dataset"][p.name] = info
        crop = np.stack([np.asarray(it["crop_info"]) for it in items])
        nev = np.array([len(it["ev_tau"]) for it in items])
        hidden = sorted({len(it["mid_tau"]) for it in items})
        rep["crop_stats"][p.name] = dict(samples=len(items), reached_min_events=float(crop[:, 4].mean()),
                                         mean_tries=float(crop[:, 3].mean()), flipped=float(crop[:, 2].mean()),
                                         events_per_crop=[int(np.percentile(nev, q)) for q in (5, 50, 95)],
                                         hidden_frames=hidden)
        cs = rep["crop_stats"][p.name]
        log(f"- {p.name}: batch of {b['i0'].shape[0]} | i0/i1 {info['i0']['shape']} {info['i0']['dtype']} "
            f"[{info['i0']['min']:.2f}, {info['i0']['max']:.2f}] | mid {info['mid']['shape']} | ctx {info['ctx']['shape']} | "
            f"voxel {info['voxel']['shape']} max {info['voxel'].get('max', 0):.0f} | events {info['ev_tau']['shape'][0]}")
        log(f"    over {cs['samples']} samples: min_events reached in {100 * cs['reached_min_events']:.0f}% of crops "
            f"({cs['mean_tries']:.1f} tries on average), events per crop 5/50/95%: {cs['events_per_crop']}, "
            f"hidden frames {cs['hidden_frames']}, flipped {100 * cs['flipped']:.0f}%")
    b = next(iter(loader))
    info = describe(b)
    rep["mixed"] = dict(tensors=info, datasets=np.bincount(b["ds"].numpy(), minlength=len(parts)).tolist())
    log(f"- mixed (first training batch): samples per dataset {dict(zip([p.name for p in parts], rep['mixed']['datasets']))}")
    for k, v in info.items():
        log(f"    {k:10s} {str(v['shape']):22s} {v['dtype']:8s}" + (f" [{v['min']:.3g}, {v['max']:.3g}]" if "min" in v else ""))
    return rep


# ---------------------------------------------------------------- metrics files
class CSVLog:
    # columns fixed by the first row (or the existing header when resuming); later keys outside it are reported once
    def __init__(self, path):
        self.path, self.cols, self.warned = path, None, set()
        if os.path.exists(path) and os.path.getsize(path):
            with open(path) as f:
                self.cols = next(csv.reader(f))

    def write(self, row):
        new = self.cols is None
        if new:
            self.cols = list(row)
        extra = set(row) - set(self.cols) - self.warned
        if extra:
            print(f"warning: {os.path.basename(self.path)} has no column for {sorted(extra)}", flush=True)
            self.warned |= extra
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, self.cols, restval="", extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})


# ---------------------------------------------------------------- checkpoints and resume
def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def set_rng_state(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(st["cuda"])


def save(path, obj):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # a crash while writing never leaves a broken checkpoint


def prune(folder, keep):
    snaps = sorted(glob.glob(os.path.join(folder, "step_*.pt")))
    for f in snaps[:max(0, len(snaps) - keep)]:
        os.remove(f)


def flatten(d, prefix=()):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


def resume_diff(old, new):
    # settings that differ between the checkpoint's config and this run, apart from bookkeeping ones
    a, b = flatten(old), flatten(new)
    return sorted((".".join(k), a.get(k), b.get(k)) for k in set(a) | set(b)
                  if a.get(k) != b.get(k) and k[:2] not in RESUME_FREE)


def nonfinite_report(path, step, loss, terms, gnorm, meta, parts, names, rank):
    ds, sid, crop = meta
    samples = []
    for d, n, c in zip(ds.tolist(), sid.tolist(), crop.tolist()):
        seq, i = parts[d].sample_id(n)
        samples.append(dict(dataset=names[d], sequence=seq, frame_i0=i, crop_y0=c[0], crop_x0=c[1], flipped=bool(c[2])))
    rep = dict(step=step, rank=rank, loss=float(loss), grad_norm=float(gnorm),
               terms={k: float(v) for k, v in terms.items()}, nonfinite_terms=[k for k, v in terms.items() if not np.isfinite(float(v))],
               batch=samples, note="the optimizer step was not taken; checkpoints were not overwritten")
    write_json(path, rep)
    return rep


def config_text(cfg):
    return yaml.safe_dump(copy.deepcopy(cfg), sort_keys=False)
