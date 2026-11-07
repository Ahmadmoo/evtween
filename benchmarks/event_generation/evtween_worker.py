"""Run the pinned Evtween generator with isolated per-sequence simulation state."""
import inspect
import json
import os
from pathlib import Path
import random
import sys

import numpy as np


def teacher_inputs(ev, t0, t1, H, W, bins, device):
    import torch
    m = (ev["t"] >= t0) & (ev["t"] < t1)
    pix = ev["y"][m].astype(np.int64)*W + ev["x"][m].astype(np.int64)
    tau = ((ev["t"][m]-t0)/(t1-t0)).astype(np.float32)
    pol = np.where(ev["p"][m] > 0, 1, -1).astype(np.int8)
    order = np.lexsort((tau, pix))
    pix, tau, pol = pix[order], tau[order], pol[order]
    cell = ((pol < 0)*bins + np.minimum((tau*bins).astype(np.int64), bins-1))*H*W + pix
    voxel = np.bincount(cell, minlength=2*bins*H*W).reshape(2*bins,H,W).astype(np.float32)
    events = (torch.zeros(len(pix), dtype=torch.long, device=device),
              torch.as_tensor(pix, device=device), torch.as_tensor(tau, device=device),
              torch.as_tensor(pol, device=device))
    return torch.from_numpy(voxel)[None].to(device), events

def teacher_state(model, prepared, voxel, events):
    # v5.3 trajectory teacher consumes exact events; conv versions consume voxels.
    z = model.encode(prepared, voxel, events) if "ev" in inspect.signature(model.encode).parameters else model.encode(prepared, voxel)
    if getattr(model, "detail", False):
        n, _ = model.detail_code(prepared, voxel, sample=False)
        return model.decode(prepared, z, n)
    return model.decode(prepared, z)


def load_generator(repo, checkpoint, dataset, seed, mode, first_output):
    import torch
    from rng_snapshot import rng_reset

    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    local_hub = os.environ.get("ALLTEST_VJEPA2_LOCAL")
    if local_hub:
        original_hub_load = torch.hub.load
        def load_backbone(repo_or_dir, model, *args, **kwargs):
            if str(repo_or_dir).split(":", 1)[0] == "facebookresearch/vjepa2":
                repo_or_dir = local_hub
                kwargs["source"] = "local"
            return original_hub_load(repo_or_dir, model, *args, **kwargs)
    try:
        torch.from_numpy(np.zeros(1, dtype=np.float32)).numpy()
    except Exception as exc:
        raise RuntimeError("NumPy/PyTorch ABI mismatch. Set MODEL_PYTHON to a compatible training environment; for this older Torch build use numpy<2 in that environment.") from exc
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    from model import EvTween
    original = EvTween.load_state_dict
    def guarded(self, state, strict=False):
        result = original(self, state, strict=False)
        missing = [k for k in result.missing_keys if not self.frozen(k)]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={result.unexpected_keys}")
        Path(first_output).with_suffix(".load.json").write_text(json.dumps(dict(
            allowed_frozen_missing=result.missing_keys, unexpected=result.unexpected_keys), indent=2))
        return result

    source = (repo / "generate.py").read_text()
    needle = 'model = build_model(ck["cfg"]).to(dev).eval()'
    injected = """
stage = ck["cfg"]["train"]["stage"]
allowed_stages = ("teacher", "joint") if mode == "teacher" else ("student", "joint")
if stage not in allowed_stages:
    raise ValueError(f"{mode} inference requires {allowed_stages}, but checkpoint stage is {stage!r}")
sets = ck["cfg"]["data"].get("sets") or {"data": 1}
if args.dataset not in sets or not sets[args.dataset]:
    raise ValueError(f"Dataset {args.dataset!r} not enabled in checkpoint: {list(sets)}")
ck["cfg"]["sensor"]["seed"] = SEED_VALUE
__import__("pathlib").Path(args.out).with_suffix(".config.json").write_text(
    __import__("json").dumps(ck["cfg"], indent=2, default=str))
"""
    if source.count(needle) != 1:
        raise RuntimeError("Generation API changed; inspect source")
    source = source.replace(needle, injected.replace("SEED_VALUE", str(seed)) + "\n" + needle, 1)

    loop = "    for i in range(N - 1):"
    predict = '        z = model.predict(ctx, tau, torch.tensor([ts[i + 1] - ts[i]], device=dev).float())'
    simulate = '        parts.append(sim.run(model.decode(p, z), ts[i], ts[i + 1]))'
    if any(source.count(s) != 1 for s in (loop, predict, simulate)):
        raise RuntimeError("Teacher adapter API changed; inspect pinned generate.py")
    source = source.replace(loop, loop + """
        if ts[i] < eval_a or ts[i+1] > eval_b:
            continue
""", 1)
    if mode == "teacher":
        source = source.replace(predict, """
        voxel, events = teacher_inputs(teacher_gt, ts[i], ts[i+1], ctx.shape[-2], ctx.shape[-1], ck["cfg"]["data"]["bins"], dev)
        state = teacher_state(model, p, voxel, events)
    """, 1)
        source = source.replace(simulate, '        parts.append(sim.run(state, ts[i], ts[i + 1]))', 1)

    # Everything after this boundary (including Simulator) is per sequence.
    boundary = 'S, names = ck["cfg"]["sensor"],'
    if source.count(boundary) != 1:
        raise RuntimeError("Generation setup changed; inspect pinned generate.py")
    setup, tail = source.split(boundary)
    filename = str(repo / "generate.py")
    scope = dict(globals(), __name__="__main__", __file__=filename, mode=mode)
    previous_argv = sys.argv
    sys.argv = [filename, "--ckpt", str(checkpoint), "--seq", ".", "--out",
                str(first_output), "--dataset", dataset]
    if local_hub:
        torch.hub.load = load_backbone
    EvTween.load_state_dict = guarded
    try:
        exec(compile(setup, filename, "exec"), scope)
    finally:
        EvTween.load_state_dict = original
        sys.argv = previous_argv
        if local_hub:
            torch.hub.load = original_hub_load
    # Upstream only reads the config after the weights have been installed.
    scope["ck"] = {"cfg": scope["ck"]["cfg"]}
    run_sequence = compile(boundary + tail, filename, "exec")
    reset = rng_reset(torch)

    def generate(seq, out):
        os.chdir(repo)
        reset()
        seq, out = Path(seq), Path(out)
        args = scope["args"]
        args.seq, args.out = str(seq), str(out)
        meta = json.loads((seq / "evaluation_meta.json").read_text())
        start, end = meta["evaluation_window_in_run_s"]
        job = dict(scope, eval_a=start, eval_b=end)
        if mode == "teacher":
            with np.load(seq / "teacher_gt.npz", allow_pickle=False) as archive:
                gt = {k: archive[k] for k in "txyp"}
            gt["t"] = gt["t"].astype(np.float64) + start
            job["teacher_gt"] = gt
        exec(run_sequence, job)

    return generate


if __name__ == "__main__":
    repo, checkpoint, seq, out, dataset, seed, mode = sys.argv[1:]
    load_generator(repo, checkpoint, dataset, int(seed), mode, out)(seq, out)
