import math
import torch
import torch.nn.functional as F
import lejepa
from model import luminance

SIGREG = lejepa.multivariate.SlicingUnivariateTest(univariate_test=lejepa.univariate.EppsPulley(n_points=17), num_slices=256)


def ig_logs(x, k):
    # inverse Gaussian with mean 1 and shape k (float64 for stability):
    # log pdf, log survival, and log survival of the stationary first wait (pixel state unknown at the window start)
    x, k = x.double().clamp(min=1e-6), k.double()
    s = (k / x).sqrt()
    la, lb = torch.special.log_ndtr(-s * (x - 1)), torch.special.log_ndtr(-s * (x + 1))
    q = -torch.expm1(2 * k + lb - la)
    logpdf = 0.5 * (k.log() - math.log(2 * math.pi) - 3 * x.log()) - k * (x - 1) ** 2 / (2 * x)
    logsf = la + q.clamp(min=1e-300).log()
    logsf_eq = la + (2 - (1 + x) * q).clamp(min=1e-300).log()
    return logpdf.float(), logsf.float(), logsf_eq.float()


def event_nll(model, s, ev, steps=32):
    # point-process NLL of the real event times, per pixel
    # clocks: L_on = int relu(dL/dtau)/C_on, L_off = int relu(-dL/dtau)/C_off; waits in clock units ~ IG(1, k)
    # ON and OFF compete and both reset after any event; the no-event stretches enter through the survival terms;
    # events the trajectory cannot explain fall on a background noise floor nu
    b, pix, tau, pol = ev
    B, _, H, W = s["y0"].shape
    taus = torch.linspace(0, 1, steps + 1, device=tau.device)
    D = torch.stack([model.render(s, t.expand(B))[1][:, 0] for t in taus], 1).flatten(2)
    rate = torch.stack([F.relu(D) / model.c, F.relu(-D) / (model.c * model.r)], 1)
    clock = F.pad(torch.cumsum((rate[:, :, 1:] + rate[:, :, :-1]) / (2 * steps), 2), (0, 0, 1, 0))
    j = (tau * steps).clamp(0, steps - 1e-4)
    j0, w = j.long(), (j - j.floor())[:, None]
    at = lambda v: v[b, :, j0, pix] * (1 - w) + v[b, :, j0 + 1, pix] * w
    lam, clk = at(rate), at(clock)

    key = b * H * W + pix
    first = torch.ones_like(key, dtype=torch.bool)
    first[1:] = key[1:] != key[:-1]
    last = torch.ones_like(first)
    last[:-1] = key[1:] != key[:-1]
    x = clk - torch.where(first[:, None], torch.zeros_like(clk), clk.roll(1, 0))
    own = (pol < 0).long()[:, None]
    k = model.log_k.exp()
    logpdf, logsf, logsf_eq = ig_logs(x, k)
    f_own = torch.where(first[:, None], logsf, logpdf).gather(1, own)[:, 0]
    s_other = torch.where(first[:, None], logsf_eq, logsf).gather(1, 1 - own)[:, 0]
    log_ev = torch.logaddexp(lam.gather(1, own)[:, 0].clamp(min=1e-12).log() + f_own + s_other, model.log_nu)

    end = clock[:, :, -1]
    tail = ig_logs(end[b, :, pix] - clk, k)[1].sum(1)
    seen = torch.zeros(B * H * W, dtype=torch.bool, device=key.device)
    seen[key] = True
    silent = ig_logs(end.transpose(1, 2).reshape(-1, 2)[~seen], k)[2].sum(1)
    ll = log_ev.sum() + tail[last].sum() + silent.sum() - model.log_nu.exp() * B * H * W
    return -ll / (B * H * W)


def photo_loss(model, s, mid, mid_tau):
    if mid.shape[1] == 0:
        return mid.new_zeros(())
    err = [(model.render(s, mid_tau[:, j])[0] - torch.log(luminance(mid[:, j], model.gamma) + model.eps)).abs().mean()
           for j in range(mid.shape[1])]
    return sum(err) / len(err)


def cmax_loss(model, s, ev):
    # contrast maximization: real events moved along the model's flow to tau=0 and tau=1 should stack into sharp edges
    b, pix, tau, pol = ev
    B, _, H, W = s["y0"].shape
    N = tau.shape[0]
    g = lambda m: m.flatten(-2)[b, ..., pix]
    se = dict(f01=g(s["f01"]).view(N, 2, 1, 1), f10=g(s["f10"]).view(N, 2, 1, 1),
              a=g(s["a"]).view(N, model.K, 2, 1, 1), b=g(s["b"]).view(N, model.K, 2, 1, 1))
    F0, _, F1, _ = model.flows(se, tau.view(N, 1, 1, 1))
    y, x = (pix // W).float(), (pix % W).float()
    img = b * 2 + (pol < 0).long()

    def contrast(px, py):
        iwe = torch.zeros(B * 2 * H * W, device=tau.device)
        for dx in (0, 1):
            for dy in (0, 1):
                xi, yi = px.floor() + dx, py.floor() + dy
                wgt = (1 - (px - xi).abs()) * (1 - (py - yi).abs())
                ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                iwe = iwe.index_add(0, ((img * H + yi.long()) * W + xi.long())[ok], wgt[ok])
        iwe = iwe.view(B * 2, H * W)
        return iwe.var(1) / (iwe.mean(1) ** 2 + 1e-6)

    base = contrast(x, y).detach() + 1e-6  # contrast of the unwarped events, sets the scale to ~1
    return -sum((contrast(x + f[:, 0, 0, 0], y + f[:, 1, 0, 0]) / base).mean() for f in (F0, F1)) / 2


def smooth_loss(s, img):
    # edge-aware first-order smoothness of the path coefficients
    x = torch.cat([s["a"].flatten(1, 2), s["b"].flatten(1, 2), s["d"]], 1)
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    wx = torch.exp(-10 * dx(img).abs().mean(1, keepdim=True))
    wy = torch.exp(-10 * dy(img).abs().mean(1, keepdim=True))
    return (dx(x).abs() * wx).mean() + (dy(x).abs() * wy).mean()


def compute(model, batch, w, stage):
    # stage: probe | teacher | student | joint
    ev = (batch["ev_b"], batch["ev_pix"], batch["ev_tau"], batch["ev_pol"])
    if stage == "probe":
        target = torch.log1p(F.avg_pool2d(batch["voxel"], 16) * 256)
        loss = F.mse_loss(model.predict(batch["ctx"], batch["ctx_tau"], batch["dt"], head="counts"), target)
        return loss, dict(probe=loss)

    p = model.prepare(batch["i0"], batch["i1"])
    terms = {}
    if stage in ("teacher", "joint"):
        z = model.encode(p, batch["voxel"])
        s = model.decode(p, z)
        terms.update(nll=event_nll(model, s, ev, w["grid"]), photo=photo_loss(model, s, batch["mid"], batch["mid_tau"]),
                     cmax=cmax_loss(model, s, ev), smooth=smooth_loss(s, batch["i0"]),
                     sigreg=SIGREG.to(z.device)(z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])))
    if stage in ("student", "joint"):
        zt = model.encode(p, batch["voxel"]).detach() if stage == "student" else z
        zh = model.predict(batch["ctx"], batch["ctx_tau"], batch["dt"])
        s = model.decode(p, zh)
        terms.update(jepa=F.mse_loss(zh, zt), nll_student=event_nll(model, s, ev, w["grid"]))
        if stage == "student":
            terms["photo"] = photo_loss(model, s, batch["mid"], batch["mid_tau"])
    weight = dict(w, nll_student=w["nll"])
    return sum(weight[k] * v for k, v in terms.items()), terms


@torch.no_grad()
def diagnostics(model, batch, w, student=True):
    # does the decoder use z (shuffled / zero codes must hurt), and how much can the student predict (gap)?
    ev = (batch["ev_b"], batch["ev_pix"], batch["ev_tau"], batch["ev_pol"])
    p = model.prepare(batch["i0"], batch["i1"])
    z = model.encode(p, batch["voxel"])
    nll = lambda code: event_nll(model, model.decode(p, code), ev, w["grid"]).item()
    out = dict(nll=nll(z), nll_shuffled=nll(z.roll(1, 0)), nll_zero=nll(torch.zeros_like(z)))
    if student:
        out["nll_student"] = nll(model.predict(batch["ctx"], batch["ctx_tau"], batch["dt"]))
        out["gap"] = out["nll_student"] - out["nll"]
    return out
