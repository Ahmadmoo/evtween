"""Per-frame-pair inference profile of one Evtween checkpoint.

Unlike the end-to-end timing in generate_event_streams, the model is loaded once and
only the work for one pair of frames is timed, stage by stage, with CUDA synchronised
before every clock reading:

  frames    read the context frames from disk and move them to the device
  prepare   RAFT flow in both directions + brightness mapping (model.prepare)
  student   V-JEPA backbone + gap transformer (model.predict)         [student mode]
  teacher   event voxel / event list + teacher encoder (model.encode)  [teacher mode]
  decode    path decoder (model.decode)
  simulate  event sampling from the decoded path (physics.Simulator.run)

The setup is the checkout's own generate.py up to its frame loop, executed unchanged
(apart from the fixed sensor seed the generation runners use), so the model, sensor
profile and simulator are exactly those used for generation. The per-pair code mirrors
that loop. FLOPs are counted on one pair with torch.utils.flop_counter, which counts
matrix multiplications, convolutions and attention only (2 FLOPs per multiply-add).

Usage: profile_pair.py REPO CHECKPOINT SEQUENCE DATASET MODE PAIRS WARMUP OUT_JSON
SEQUENCE holds frames/*.png and frame_ts.npy (and teacher_gt.npz for MODE=teacher)
and needs at least WARMUP + PAIRS + 1 frames.
"""
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 123  # generate_event_streams passes 123 to every learned method


def count_params(model):
    # top-level parts; the frozen pretrained backbone is split from the rest of the student
    groups = {}
    for name, param in model.named_parameters():
        if name.startswith("student.backbone."):
            key = "student_backbone"
        elif name.startswith("student."):
            key = "student_head"
        else:
            key = name.split(".")[0] if "." in name else "other"
        groups[key] = groups.get(key, 0) + param.numel()
    return groups


def main():
    repo, checkpoint, seq_dir, dataset, mode = sys.argv[1:6]
    pairs, warmup, out = int(sys.argv[6]), int(sys.argv[7]), Path(sys.argv[8]).resolve()
    if mode not in ("student", "teacher") or pairs < 1 or warmup < 0:
        raise SystemExit("MODE must be student or teacher, PAIRS >= 1, WARMUP >= 0")
    import torch
    sys.path.insert(0, str(HERE))
    from evtween_worker import teacher_inputs

    repo, seq_dir = Path(repo).resolve(), Path(seq_dir).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    local_hub = os.environ.get("ALLTEST_VJEPA2_LOCAL")
    if local_hub:  # same redirect as evtween_worker: V-JEPA 2 code from a local clone
        original_hub_load = torch.hub.load
        def load_backbone(repo_or_dir, model, *args, **kwargs):
            if str(repo_or_dir).split(":", 1)[0] == "facebookresearch/vjepa2":
                repo_or_dir = local_hub
                kwargs["source"] = "local"
            return original_hub_load(repo_or_dir, model, *args, **kwargs)
        torch.hub.load = load_backbone

    source = (repo / "generate.py").read_text()
    build = 'model = build_model(ck["cfg"]).to(dev).eval()'
    loop = "seq, c, parts = Sequence(args.seq)"
    if source.count(build) != 1 or source.count(loop) != 1:
        raise RuntimeError("generate.py changed; inspect it before profiling")
    setup = source.split(loop)[0].replace(build, f'ck["cfg"]["sensor"]["seed"] = {SEED}\n' + build, 1)
    scope = dict(__name__="__main__", __file__=str(repo / "generate.py"))
    argv = sys.argv
    sys.argv = [scope["__file__"], "--ckpt", str(checkpoint), "--seq", str(seq_dir),
                "--out", str(out.with_suffix(".unused.npz")), "--dataset", dataset]
    cuda = torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    start = time.perf_counter()
    try:
        exec(compile(setup, scope["__file__"], "exec"), scope)
    finally:
        sys.argv = argv
    sync()
    load_s = time.perf_counter() - start
    model, sim, dev, ck, ds = (scope[k] for k in ("model", "sim", "dev", "ck", "ds"))
    stage = ck["cfg"]["train"]["stage"]
    allowed = ("teacher", "joint") if mode == "teacher" else ("student", "joint")
    if stage not in allowed:
        raise ValueError(f"{mode} inference requires a {allowed} checkpoint, got {stage!r}")

    seq = scope["Sequence"](str(seq_dir))
    c, bins = ck["cfg"]["data"]["context"], ck["cfg"]["data"]["bins"]
    N, ts = len(seq), seq.ts
    if N < warmup + pairs + 1:
        raise ValueError(f"Need {warmup + pairs + 1} frames, sequence has {N}")
    gt = None
    if mode == "teacher":
        with np.load(seq_dir / "teacher_gt.npz", allow_pickle=False) as archive:
            gt = {k: archive[k] for k in "txyp"}
        gt["t"] = gt["t"].astype(np.float64)
    with_events = "ev" in inspect.signature(model.encode).parameters

    def clock():
        sync()
        return time.perf_counter()

    def one_pair(i, timed):
        marks = [clock()] if timed else None
        mark = (lambda: marks.append(clock())) if timed else (lambda: None)
        idx = [min(max(q, 0), N - 1) for q in [*range(i - c + 1, i + 1), *range(i + 1, i + 1 + c)]]
        ctx = torch.stack([seq.frame(q) for q in idx])[None].to(dev)
        tau = torch.tensor([(ts[q] - ts[i]) / (ts[i + 1] - ts[i]) for q in idx], device=dev)[None].float()
        mark()
        p = model.prepare(ctx[:, c - 1], ctx[:, c], torch.tensor([ds], device=dev))
        mark()
        if mode == "student":
            z = model.predict(ctx, tau, torch.tensor([ts[i + 1] - ts[i]], device=dev).float())
            n = None
        else:  # as evtween_worker.teacher_state, with the encoder timed apart from the decoder
            voxel, events = teacher_inputs(gt, ts[i], ts[i + 1], ctx.shape[-2], ctx.shape[-1], bins, dev)
            z = model.encode(p, voxel, events) if with_events else model.encode(p, voxel)
            n = model.detail_code(p, voxel, sample=False)[0] if getattr(model, "detail", False) else None
        mark()
        state = model.decode(p, z) if n is None else model.decode(p, z, n)
        mark()
        result = sim.run(state, ts[i], ts[i + 1])
        mark()
        return marks, len(result["t"])

    names = ["frames", "prepare", mode, "decode", "simulate"]
    times = {k: [] for k in names}
    events = []
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(warmup):  # also brings the simulator's pixel state to a running start
            one_pair(i, False)
        for i in range(warmup, warmup + pairs):
            marks, count = one_pair(i, True)
            for k, a, b in zip(names, marks, marks[1:]):
                times[k].append(1e3 * (b - a))
            events.append(count)

    flops, flop_note = None, ""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        flop_note = f"torch {torch.__version__} has no torch.utils.flop_counter (needs >= 2.1)"
    else:
        # one extra pair, counted stage by stage: same calls as one_pair
        i = warmup + pairs - 1
        flops = {}
        with torch.no_grad():
            idx = [min(max(q, 0), N - 1) for q in [*range(i - c + 1, i + 1), *range(i + 1, i + 1 + c)]]
            ctx = torch.stack([seq.frame(q) for q in idx])[None].to(dev)
            tau = torch.tensor([(ts[q] - ts[i]) / (ts[i + 1] - ts[i]) for q in idx], device=dev)[None].float()
            def counted(name, fn):
                with FlopCounterMode(display=False) as counter:
                    value = fn()
                flops[name] = int(counter.get_total_flops())
                return value
            p = counted("prepare", lambda: model.prepare(ctx[:, c - 1], ctx[:, c], torch.tensor([ds], device=dev)))
            if mode == "student":
                z = counted("student", lambda: model.predict(ctx, tau, torch.tensor([ts[i + 1] - ts[i]], device=dev).float()))
                n = None
            else:
                voxel, ev = teacher_inputs(gt, ts[i], ts[i + 1], ctx.shape[-2], ctx.shape[-1], bins, dev)
                def teach():
                    zz = model.encode(p, voxel, ev) if with_events else model.encode(p, voxel)
                    nn_ = model.detail_code(p, voxel, sample=False)[0] if getattr(model, "detail", False) else None
                    return zz, nn_
                z, n = counted("teacher", teach)
            state = counted("decode", lambda: model.decode(p, z) if n is None else model.decode(p, z, n))
            counted("simulate", lambda: sim.run(state, ts[i], ts[i + 1]))
        flop_note = "matmul/conv/attention only; elementwise work (most of the simulator) is not counted"

    record = dict(
        mode=mode, checkpoint=str(Path(checkpoint).resolve()), repo=str(repo), dataset=dataset,
        checkpoint_stage=stage, device=str(dev),
        gpu=torch.cuda.get_device_name() if cuda else None, cpu=platform.processor() or platform.machine(),
        torch=torch.__version__, amp=bool(getattr(model, "amp", False)),
        height=int(seq.frame(0).shape[-2]), width=int(seq.frame(0).shape[-1]), context_frames_per_side=int(c),
        pairs=pairs, warmup=warmup, load_s=load_s, times_ms=times, events=events,
        peak_memory_gb=torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
        params=count_params(model), flops=flops, flop_note=flop_note,
        loaded_but_unused=["student"] if mode == "teacher" else ["teacher"] + (
            ["detail_enc"] if getattr(model, "detail", False) else []))
    out.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
