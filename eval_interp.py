import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from data import collate, datasets
from model import build_model, warp

# event-guided frame interpolation with the teacher's path: RGB frames at the hidden frame times vs the real hidden frames.
# blend: (1 - tau) I0 + tau I1; frames only: z = 0 (the plain SloMo path); student: z from frames (world model);
# teacher: z from the real events. All warp and blend the RGB keyframes exactly as render() does (in linear light)
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=200, help="samples per dataset, spread over the split")
ap.add_argument("--crop", type=int, default=None, help="center crop size (default: the checkpoint's data.crop; large = full frame)")
ap.add_argument("--skip", nargs="*", default=[], help="hidden frames per gap, e.g. bsergb=3 (default: as trained)")
ap.add_argument("--student", action="store_true", help="also the student (needs a trained student in the checkpoint)")
ap.add_argument("--offsets", nargs="*", type=float, default=[], help="also the teacher at tau + offset (ms): finds an event-frame time offset")
ap.add_argument("--datasets", nargs="*", default=None, help="only these, e.g. eds (the figure shows the first)")
ap.add_argument("--out", default="interp.png")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
cfg = ck["cfg"]
if not args.student:  # the frozen video backbone is only needed by the student: don't load it
    cfg["model"]["backbone"] = "none"
model = build_model(cfg).to(dev).eval()
own = model.state_dict()
model.load_state_dict({k: v for k, v in ck["model"].items() if k in own and own[k].shape == v.shape}, strict=False)
D = cfg["data"]
D["crop"] = args.crop or D["crop"]
for kv in args.skip:
    name, v = kv.split("=")
    D["sets"][name]["skip"] = int(v)
try:
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev)
except Exception:
    lp = None


def render_rgb(s, i0, i1, tau, g):
    tau = tau.view(-1, 1, 1, 1)
    F0, _, F1, _ = model.flows(s, tau)
    j = torch.arange(1, model.K + 1, device=dev, dtype=tau.dtype).view(1, -1, 1, 1)
    V = torch.sigmoid(s["d"][:, :1] + (s["d"][:, 1:] * tau ** j).sum(1, keepdim=True))
    w0, w1 = (1 - tau) * V, tau * (1 - V)
    out = (w0 * warp(i0 ** g, F0) + w1 * warp(i1 ** g, F1)) / (w0 + w1 + 1e-6)
    return out.clamp(0, 1) ** (1 / g)


def ssim(a, b):
    x = torch.arange(11, device=dev, dtype=a.dtype) - 5
    k = torch.exp(-x ** 2 / 4.5)
    k = (k[:, None] * k[None]) / k.sum() ** 2
    f = lambda t: F.conv2d(t, k.expand(3, 1, 11, 11), groups=3)
    ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean((1, 2, 3))


def scores(pred, gt):
    psnr = -10 * torch.log10(((pred - gt) ** 2).mean((1, 2, 3)).clamp(min=1e-10))
    out = [psnr, ssim(pred, gt)]
    if lp is not None:
        out.append(lp(pred * 2 - 1, gt * 2 - 1).flatten())
    return torch.stack(out, 1)


names = ["blend", "frames only (z=0)"] + (["student (frames)"] if args.student else []) + ["teacher (frames + events)"]
cols = ["PSNR", "SSIM"] + (["LPIPS"] if lp is not None else [])
shown = None
for name, ds in datasets(D, args.split, False).items():
    if args.datasets and name not in args.datasets:
        continue
    if not len(ds) or ds.skip == 0:
        print(f"{name}: no hidden frames (skip 0); pass --skip {name}=N")
        continue
    idx = torch.linspace(0, len(ds) - 1, min(args.n, len(ds))).long().tolist()
    tot, cnt, off = {m: 0 for m in names}, 0, {o: 0 for o in args.offsets}
    for b in DataLoader(Subset(ds, idx), batch_size=1, collate_fn=collate, num_workers=2):
        b = {k: v.to(dev) for k, v in b.items()}
        g = model.gammas[b["ds"]].view(-1, 1, 1, 1)
        with torch.no_grad():
            p = model.prepare(b["i0"], b["i1"], b["ds"], b.get("cfa"))
            zt = model.encode(p, b["voxel"])
            z = {"frames only (z=0)": torch.zeros_like(zt), "teacher (frames + events)": zt}
            if args.student:
                z["student (frames)"] = model.predict(b["ctx"], b["ctx_tau"], b["dt"])
            n = model.detail_code(p, b["voxel"], sample=False)[0] if model.detail else None
            s = {m: model.decode(p, z[m], n if m.startswith("teacher") else None) for m in z}
            for j in range(b["mid"].shape[1]):
                tau, gt = b["mid_tau"][:, j], b["mid"][:, j]
                pred = {"blend": (((1 - tau).view(-1, 1, 1, 1) * b["i0"] ** g + tau.view(-1, 1, 1, 1) * b["i1"] ** g) ** (1 / g))}
                pred.update({m: render_rgb(s[m], b["i0"], b["i1"], tau, g) for m in s})
                for m in names:
                    tot[m] = tot[m] + scores(pred[m], gt).sum(0)
                for o in args.offsets:
                    t = (tau + o * 1e-3 / b["dt"]).clamp(0, 1)
                    off[o] = off[o] + scores(render_rgb(s["teacher (frames + events)"], b["i0"], b["i1"], t, g), gt).sum(0)
                cnt += gt.shape[0]
                if shown is None and j == b["mid"].shape[1] // 2:
                    shown = (name, gt[0], {m: pred[m][0] for m in names})
    print(f"\n{name} ({len(idx)} samples, {cnt} hidden frames, skip {ds.skip}, crop {D['crop']})")
    print(f"  {'':28}" + "".join(f"{c:>9}" for c in cols))
    for m in names:
        print(f"  {m:28}" + "".join(f"{v:9.4f}" for v in (tot[m] / cnt).tolist()))
    for o in args.offsets:
        print(f"  {'teacher at tau %+.1f ms' % o:28}" + "".join(f"{v:9.4f}" for v in (off[o] / cnt).tolist()))

if shown is not None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    name, gt, pred = shown
    fig, ax = plt.subplots(2, len(names) + 1, figsize=(4 * (len(names) + 1), 8))
    img = lambda t: t.permute(1, 2, 0).cpu().numpy()
    ax[0, 0].imshow(img(gt)); ax[0, 0].set_title(f"GT hidden frame ({name})")
    ax[1, 0].axis("off")
    for c, m in enumerate(names, 1):
        ax[0, c].imshow(img(pred[m])); ax[0, c].set_title(m)
        ax[1, c].imshow((pred[m] - gt).abs().mean(0).cpu().numpy(), cmap="magma", vmin=0, vmax=0.2)
        ax[1, c].set_title(f"|error|, PSNR {-10 * math.log10(((pred[m] - gt) ** 2).mean().item() + 1e-10):.2f}")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    plt.savefig(args.out, dpi=80)
    print("\nsaved", args.out)
