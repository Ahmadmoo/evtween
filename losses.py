import torch
import torch.nn.functional as F
from model import luminance


def blur(x, sigma):
    # separable Gaussian blur over the last two dims
    if sigma <= 0:
        return x
    r = int(3 * sigma)
    g = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    shape = x.shape
    x = x.reshape(-1, 1, *shape[-2:])
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))
    return x.reshape(shape)


def anchored(L, on, off):
    # pair every time with a known frame: changes from tau=0 and to tau=1, with their event counts
    # (quantization error stays below ~2c per pixel while the signal grows with the interval)
    con, coff = on.cumsum(1), off.cumsum(1)
    dl = torch.cat([L[:, 1:] - L[:, :1], L[:, -1:] - L[:, 1:-1]], 1)
    n_on = torch.cat([con, con[:, -1:] - con[:, :-1]], 1)
    n_off = torch.cat([coff, coff[:, -1:] - coff[:, :-1]], 1)
    return dl, n_on, n_off


def event_loss(dl, on, off, c, r, patch=16, margin=1.0):
    # dl: predicted log change (B,T,H,W); on/off: real event counts over the same intervals
    # cos: per-patch direction match, independent of the threshold c
    # fit: |dl - c*(on - r*off)| is free below margin*c (quantization), Huber above it
    e = on - r * off
    pool = lambda x: F.avg_pool2d(x, patch)
    cos = pool(dl * e) / (pool(dl ** 2) * pool(e ** 2) + 1e-8).sqrt()
    w = (pool(on + off) > 0).float()
    l_cos = ((1 - cos) * w).sum() / w.sum().clamp(min=1)

    cd = c.detach()
    x = F.relu((dl - c * e).abs() - margin * cd)
    l_fit = torch.where(x < cd, 0.5 * x ** 2 / cd, x - 0.5 * cd).mean()
    return l_cos, l_fit


def smooth_loss(x, img):
    # edge-aware first-order smoothness
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    wx = torch.exp(-10 * dx(img).abs().mean(1, keepdim=True))
    wy = torch.exp(-10 * dy(img).abs().mean(1, keepdim=True))
    return (dx(x).abs() * wx).mean() + (dy(x).abs() * wy).mean()


def total_loss(model, s, batch, w):
    taus = batch["taus"]
    L = torch.stack([model.render(s, taus[:, i])[0][:, 0] for i in range(taus.shape[1])], 1)
    dl, on, off = (blur(x, w["blur"]) for x in anchored(L, batch["on"], batch["off"]))
    l_cos, l_fit = event_loss(dl, on, off, model.c, model.r, w["patch"], w["margin"])

    mid, mt = batch["mid"], batch["mid_tau"]
    l_photo = L.new_zeros(())
    for j in range(mid.shape[1]):
        target = torch.log(luminance(mid[:, j], model.gamma) + model.eps)
        l_photo = l_photo + (model.render(s, mt[:, j])[0] - target).abs().mean() / mid.shape[1]

    coef = torch.cat([s["a"].flatten(1, 2), s["b"].flatten(1, 2), s["d"]], 1)
    terms = dict(cos=l_cos, fit=l_fit, photo=l_photo, smooth=smooth_loss(coef, batch["i0"]))
    return sum(w[k] * v for k, v in terms.items()), terms
