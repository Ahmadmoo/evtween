import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import lejepa

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


SHIFTS = (-8, -4, -2, -1, 0, 1, 2, 4, 8)  # path time shifts (grid steps) averaged per pixel with the predicted uncertainty


def event_nll(model, s, ev, dt, steps=128):
    # point-process NLL of the real event times per pixel, same sensor model as physics.Simulator:
    # after every event the pixel draws thresholds X_on*C_on, X_off*C_off with X ~ IG(1, k) and takes a reference ref = L;
    # ON fires when the running max of (L - ref)/C_on reaches X_on (OFF: running max of (ref - L)/C_off);
    # background events (nu per second, half per polarity) compete with the signal and also reset the pixel;
    # after any event the pixel is blind for the refractory time R and takes ref = L at its end (not at the event);
    # the first wait of each pixel starts from the stationary state (its ref and thresholds are unknown).
    # With an uncertainty map s["sig"] (gap units) the whole path of a pixel may be shifted in time by d ~ N(0, sig^2)
    # and the pixel's likelihood is averaged over shifts: a shift moves all its events together, threshold noise does not
    b, pix, tau, pol = ev
    B, _, H, W = s["y0"].shape
    taus = torch.linspace(0, 1, steps + 1, device=tau.device)
    frame = lambda t: model.render(s, t.expand(B))[0][:, 0]
    grad = torch.is_grad_enabled()  # recompute renders in backward: memory stays flat as the grid gets finer
    Lg = torch.stack([checkpoint(frame, t, use_reentrant=False) if grad else frame(t) for t in taus], 1).flatten(2)
    C, K, nu, R = model.sensor(s["ds"])  # per sample (each dataset has its own sensor)
    nu = nu * dt
    c, k = C[b], K[b, None]  # per event

    key = b * H * W + pix
    first = torch.ones_like(key, dtype=torch.bool)
    first[1:] = key[1:] != key[:-1]
    last = torch.ones_like(first)
    last[:-1] = key[1:] != key[:-1]
    seen = torch.zeros(B * H * W, dtype=torch.bool, device=key.device)
    seen[key] = True
    tprev = torch.where(first, torch.zeros_like(tau), tau.roll(1))
    Rt, wR = R[b] / dt[b], 1e-5 / dt[b]  # refractory time and the width of its soft edge, in gap units
    wake = torch.where(first, tprev, torch.minimum(tprev + Rt, tau))  # end of the blind time: ref is taken here
    awake = torch.where(first, torch.ones_like(tau), torch.sigmoid((tau - tprev - Rt) / wR))
    own = (pol < 0).long()[:, None]

    def pixel_ll(Lg, d):
        # log-likelihood per pixel with the path delayed by d (gap units): events see L(t - d), L held constant outside [0, 1]
        def at(t):
            j = (t * steps).clamp(0, steps - 1e-4)
            j0, w = j.long(), j - j.floor()
            return Lg[b, j0, pix] * (1 - w) + Lg[b, j0 + 1, pix] * w

        def slope(t):
            j = (t * steps).floor().clamp(0, steps - 1).long()
            return (Lg[b, j + 1, pix] - Lg[b, j, pix]) * steps * ((t >= 0) & (t < 1))

        def reach(lo, hi, ref):  # running max / min of +-(L - ref)/C over the grid points strictly inside (lo, hi)
            with torch.no_grad():  # only to find where the max / min sits; the value is gathered with gradient
                row, inner = Lg[b, :, pix], (taus > lo[:, None]) & (taus < hi[:, None])
                imax = torch.where(inner, row, -math.inf).argmax(1)
                imin = torch.where(inner, row, math.inf).argmin(1)
                has = inner.any(1)
            top = torch.where(has, Lg[b, imax, pix], ref).maximum(ref)
            bot = torch.where(has, Lg[b, imin, pix], ref).minimum(ref)
            return torch.stack([top - ref, ref - bot], 1) / c

        t, tw = tau - d, wake - d
        Li, Lp, Di = at(t), at(tw), slope(t)
        before = reach(tw, t, Lp)
        now = torch.stack([Li - Lp, Lp - Li], 1) / c
        M = torch.maximum(before, now)
        # signal only at a new max (a wider gate biases c low when k is small) and never while blind
        rate = torch.stack([F.relu(Di), F.relu(-Di)], 1) / c * torch.sigmoid((now - before) / 1e-3) * awake[:, None]
        logpdf, logsf, logsf_eq = ig_logs(M, k)
        log_h = torch.where(first[:, None], logsf - logsf_eq, logpdf - logsf)
        log_s = torch.where(first[:, None], logsf_eq, logsf)
        signal = (rate.clamp(min=1e-12).log() + log_h).gather(1, own)[:, 0]
        log_ev = torch.logaddexp(signal, (nu[b] / 2).log()) + log_s.sum(1)

        end = (tau + Rt).clamp(max=1) - d
        tail = ig_logs(reach(end, torch.full_like(end, 1 - d + 0.5 / steps), at(end)), k)[1].sum(1)
        q = round(d * steps)
        win = Lg[:, max(0, -q):steps + 1 - max(0, q)]  # the path seen inside the window
        L0 = win[:, 0]
        whole = (torch.stack([win.amax(1) - L0, L0 - win.amin(1)], 1) / C[:, :, None]).transpose(1, 2).reshape(-1, 2)
        ll = torch.zeros(B * H * W, device=tau.device).index_add(0, key, log_ev).index_add(0, key[last], tail[last])
        return torch.where(seen, ll, ig_logs(whole, K.repeat_interleave(H * W)[:, None])[2].sum(1))

    shifts = SHIFTS if "sig" in s else (0,)
    lls = torch.stack([checkpoint(pixel_ll, Lg, q / steps, use_reentrant=False) if grad else pixel_ll(Lg, q / steps) for q in shifts])
    if len(shifts) > 1:
        d = torch.tensor(shifts, device=tau.device, dtype=torch.float32) / steps
        logw = (-0.5 * (d[:, None] / s["sig"].flatten()[None]) ** 2 + torch.gradient(d)[0].log()[:, None]).log_softmax(0)
        lls = torch.logsumexp(logw + lls, 0, keepdim=True)
    return -(lls[0].sum() - (nu * H * W).sum()) / (B * H * W)


def photo_loss(model, s, mid, mid_tau):
    if mid.shape[1] == 0:
        return mid.new_zeros(())
    err = [(model.render(s, mid_tau[:, j])[0] - torch.log(model.bright(mid[:, j], s["ds"], s["cfa"]) + model.eps)).abs().mean()
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
        target = torch.log1p(F.avg_pool2d(batch["voxel"], model.mult) * model.mult ** 2)
        loss = F.mse_loss(model.predict(batch["ctx"], batch["ctx_tau"], batch["dt"], head="counts"), target)
        return loss, dict(probe=loss)

    p = model.prepare(batch["i0"], batch["i1"], batch.get("ds"), batch.get("cfa"))
    terms = {}
    if stage in ("teacher", "joint"):
        z = model.encode(p, batch["voxel"], ev)
        s = model.decode(p, z)
        terms.update(nll=event_nll(model, s, ev, batch["dt"], w["grid"]), photo=photo_loss(model, s, batch["mid"], batch["mid_tau"]),
                     cmax=cmax_loss(model, s, ev), smooth=smooth_loss(s, batch["i0"]),
                     sigreg=SIGREG.to(z.device)(z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])))
    if stage in ("student", "joint"):
        zt = model.encode(p, batch["voxel"], ev).detach() if stage == "student" else z
        zh = model.predict(batch["ctx"], batch["ctx_tau"], batch["dt"])
        s = model.decode(p, zh)
        terms.update(jepa=F.mse_loss(zh, zt), nll_student=event_nll(model, s, ev, batch["dt"], w["grid"]))
        if stage == "student":
            terms["photo"] = photo_loss(model, s, batch["mid"], batch["mid_tau"])
    weight = dict(w, nll_student=w["nll"])
    return sum(weight[k] * v for k, v in terms.items()), terms


@torch.no_grad()
def diagnostics(model, batch, w, student=True):
    # does the decoder use z (shuffled / zero codes must hurt), and how much can the student predict (gap)?
    ev = (batch["ev_b"], batch["ev_pix"], batch["ev_tau"], batch["ev_pol"])
    p = model.prepare(batch["i0"], batch["i1"], batch.get("ds"), batch.get("cfa"))
    z = model.encode(p, batch["voxel"], ev)
    nll = lambda code: event_nll(model, model.decode(p, code), ev, batch["dt"], w["grid"]).item()
    out = dict(nll=nll(z), nll_shuffled=nll(z.roll(1, 0)), nll_zero=nll(torch.zeros_like(z)))
    if student:
        out["nll_student"] = nll(model.predict(batch["ctx"], batch["ctx_tau"], batch["dt"]))
        out["gap"] = out["nll_student"] - out["nll"]
    return out
