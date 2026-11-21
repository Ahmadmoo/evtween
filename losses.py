import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import lejepa

SIGREG = lejepa.multivariate.SlicingUnivariateTest(univariate_test=lejepa.univariate.EppsPulley(n_points=17), num_slices=256)


def sigreg_loss(z):
    # lejepa's statistic grows with the number of samples (x N x GPUs): rescale to 256 samples so the weight
    # does not depend on batch, crop or GPU count
    x = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    n = len(x) * (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
    return SIGREG.to(z.device)(x) * 256 / n


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


MIX = 16  # how far back (in events) the likelihood looks for a pixel's last crossing (loss.mix)
XT_BINS = 64  # time bins of the x-t / y-t loss (the tau grid summed in groups)


def path_grid(model, s, steps=128):
    # the raw path L on the tau grid and what the pixels see (the taut path in the tube): (B, steps + 1, H*W) each
    B = s["y0"].shape[0]
    taus = torch.linspace(0, 1, steps + 1, device=s["y0"].device)
    frame = lambda t: model.render(s, t.expand(B))[0][:, 0]
    grad = torch.is_grad_enabled()  # recompute renders in backward: memory stays flat as the grid gets finer
    Lr = torch.stack([checkpoint(frame, t, use_reentrant=False) if grad else frame(t) for t in taus], 1).flatten(2)
    return Lr, model.paths(Lr, s)


def residual_grid(model, s, Lr, dt):
    # residual events (ON, OFF) the model expects on each grid step, (B, 2, steps, H*W), from the raw path
    steps = Lr.shape[1] - 1
    return model.residual_rate(s, (Lr[:, 1:] - Lr[:, :-1]) * steps, dt) / steps


def _blur(v, ax, sig):
    # Gaussian blur of v (B, C, T, H, W) along axis ax (2: t, 3: y, 4: x), one sigma per sample (B,), zeros outside
    B, C = v.shape[:2]
    r = int(math.ceil(3 * float(sig.max())))
    if r == 0:
        return v
    x = torch.arange(-r, r + 1, device=v.device, dtype=v.dtype)
    k = torch.exp(-0.5 * (x / sig[:, None].clamp(min=1e-3)) ** 2)
    k = (k / k.sum(1, keepdim=True)).repeat_interleave(C, 0)
    shape, pad = [B * C, 1, 1, 1, 1], [0, 0, 0]
    shape[ax], pad[ax - 2] = 2 * r + 1, r
    return F.conv3d(v.reshape(1, B * C, *v.shape[2:]), k.view(shape), padding=tuple(pad), groups=B * C).view_as(v)


def xt_loss(model, s, ev, dt, Lg, Rg, blur=(2.0, 2.0)):
    # x-t / y-t loss: the ON / OFF events the model expects (crossings of the taut path: rise / C_on, fall / C_off; residual
    # events Rg) against the real events, both cut into x-t slices (one per row, blurred in x and t) and y-t slices (one per
    # column, blurred in y and t), compared slice by slice with the generalized KL of counts (0 when equal). The blur
    # (px, ms) compares the curves, not single events: which pixel fires when inside a curve is the sensor's randomness
    b, pix, tau, pol = ev
    B, _, H, W = s["y0"].shape
    steps = Lg.shape[1] - 1
    T = XT_BINS if steps % XT_BINS == 0 else steps
    C = model.sensor(s["ds"])[0]

    def run(Lg, Rg):
        dL = (Lg[:, 1:] - Lg[:, :-1]).view(B, T, steps // T, H * W)
        m = torch.stack([F.relu(dL).sum(2) / C[:, :1, None], F.relu(-dL).sum(2) / C[:, 1:, None]], 1)
        m = (m + Rg.view(B, 2, T, steps // T, H * W).sum(3)).view(B, 2, T, H, W)
        key = ((b * 2 + (pol < 0).long()) * T + (tau * T).long().clamp(0, T - 1)) * H * W + pix
        g = torch.zeros(B * 2 * T * H * W, device=m.device, dtype=m.dtype).index_add_(0, key, torch.ones_like(tau, dtype=m.dtype)).view_as(m)
        st = blur[1] * 1e-3 / dt * T  # time blur in bins, per sample
        px = torch.full_like(st, float(blur[0]))
        mt, gt = _blur(m, 2, st), _blur(g, 2, st)
        out = 0
        for ax in (4, 3):  # x-t slices: blur along x; y-t slices: blur along y
            mb, gb = _blur(mt, ax, px), _blur(gt, ax, px)
            out = out + (mb - gb + torch.xlogy(gb, gb) - gb * mb.clamp(min=1e-12).log()).sum() / 2
        return out / (B * H * W)

    return checkpoint(run, Lg, Rg, use_reentrant=False) if torch.is_grad_enabled() else run(Lg, Rg)


def event_nll(model, s, ev, dt, steps=128, mix=MIX, grids=None):
    # point-process NLL of the real event times per pixel, same sensor model as physics.Simulator. Two kinds of events:
    # Crossings of the taut path in the frame-noise tube (model.paths): after every crossing the pixel draws thresholds
    # X_on*C_on, X_off*C_off with X ~ IG(1, k) and takes a reference ref = L; ON fires when the running max of (L - ref)/C_on
    # reaches X_on (OFF: running max of (ref - L)/C_off); the first wait of each pixel starts from the stationary state.
    # Residual events (model.residual_rate: kappa x the raw path's slope, plus the background nu) come on top at their own
    # rate and leave the pixel's state alone. Which events were crossings is unknown: a forward pass over each pixel's
    # events sums over it (the last crossing is among the `mix` events before, or there was none since the gap started).
    # Every pixel pays for the residual events it expects.
    # grids: (Lr, Lg, Rg) from path_grid and residual_grid (computed here if not given)
    b, pix, tau, pol = ev
    B, _, H, W = s["y0"].shape
    taus = torch.linspace(0, 1, steps + 1, device=tau.device)
    if grids is None:
        Lr, Lg = path_grid(model, s, steps)
        grids = Lr, Lg, residual_grid(model, s, Lr, dt)
    _, Lg, Rg = grids
    C, K = model.sensor(s["ds"])  # per sample (each dataset has its own sensor)
    Kp = K[:, None].expand(B, H * W)  # threshold spread per pixel
    c, k = C[b], Kp[b, pix][:, None]  # per event

    key = b * H * W + pix
    first = torch.ones_like(key, dtype=torch.bool)
    first[1:] = key[1:] != key[:-1]
    last = torch.ones_like(first)
    last[:-1] = key[1:] != key[:-1]
    seen = torch.zeros(B * H * W, dtype=torch.bool, device=key.device)
    seen[key] = True
    own = (pol < 0).long()[:, None]
    L0 = Lg[:, 0]  # pixels without events: no crossing over the whole path
    whole = (torch.stack([Lg.amax(1) - L0, L0 - Lg.amin(1)], 1) / C[:, :, None]).transpose(1, 2).reshape(-1, 2)
    still = ig_logs(whole, Kp.reshape(-1)[:, None])[2].sum(1)
    # candidates for the last crossing before each event: the q-th event before it (q <= rank), or none since the gap
    # started (always allowed, so every event sequence has an explanation)
    gs = torch.arange(len(tau), device=tau.device)
    gq = lambda x, q: x[(gs - q).clamp(min=0)]  # value at the q-th event before (same pixel when q <= rank)
    gstart = torch.where(first, gs, 0).cummax(0).values
    rank = gs - gstart
    mix = min(mix, int(rank.max()) if len(rank) else 0)  # no pixel has more events before: no further candidates
    cand = [gq(tau, q) for q in range(1, mix + 1)]
    order = torch.argsort(rank, stable=True)
    nrank = torch.bincount(rank)
    pos = torch.empty_like(order)
    pos[order] = gs - torch.repeat_interleave(nrank.cumsum(0) - nrank, nrank)
    groups, inv = torch.split(order, nrank.tolist()), torch.argsort(order)
    lidx = last.nonzero()[:, 0]

    def pixel_ll(Lg, Rg):
        # log-likelihood per pixel
        def at(t):
            j = (t * steps).clamp(0, steps - 1e-4)
            j0, w = j.long(), j - j.floor()
            return Lg[b, j0, pix] * (1 - w) + Lg[b, j0 + 1, pix] * w

        def reach(lo, hi, ref):  # running max / min of +-(L - ref)/C over the grid points strictly inside (lo, hi)
            with torch.no_grad():  # only to find where the max / min sits; the value is gathered with gradient
                row, inner = Lg[b, :, pix], (taus > lo[:, None]) & (taus < hi[:, None])
                imax = torch.where(inner, row, -math.inf).argmax(1)
                imin = torch.where(inner, row, math.inf).argmin(1)
                has = inner.any(1)
            top = torch.where(has, Lg[b, imax, pix], ref).maximum(ref)
            bot = torch.where(has, Lg[b, imin, pix], ref).minimum(ref)
            return torch.stack([top - ref, ref - bot], 1) / c

        j = (tau * steps).floor().clamp(0, steps - 1).long()  # grid step of each event
        Li, Di = at(tau), (Lg[b, j + 1, pix] - Lg[b, j, pix]) * steps * (tau < 1)

        def crossing(wake, first, before=None):
            # log density of a crossing at tau (own polarity) since the last crossing at `wake` (first: stationary start),
            # and the log survival of both thresholds up to tau (before: (top, bottom) of L since the wake, if known)
            Lp = at(wake)
            if before is None:
                before = reach(wake, tau, Lp)
            else:
                before = torch.stack([before[0].maximum(Lp) - Lp, Lp - before[1].minimum(Lp)], 1) / c
            now = torch.stack([Li - Lp, Lp - Li], 1) / c
            M = torch.maximum(before, now)
            # signal only at a new max (a wider gate biases c low when k is small)
            rate = torch.stack([F.relu(Di), F.relu(-Di)], 1) / c * torch.sigmoid((now - before) / 1e-3)
            logpdf, logsf, logsf_eq = ig_logs(M, k)
            log_h = torch.where(first[:, None], logsf - logsf_eq, logpdf - logsf)
            log_s = torch.where(first[:, None], logsf_eq, logsf)
            return (rate.clamp(min=1e-12).log() + log_h).gather(1, own)[:, 0], log_s.sum(1)

        tail = ig_logs(reach(tau, torch.full_like(tau, 1 + 0.5 / steps), Li), k)[1].sum(1)  # to the end of the path
        log_o = (Rg[b, own[:, 0], j, pix] * steps).log()  # residual events of this polarity per unit tau (no reset)
        O = log_o.double().cumsum(0)  # sums of residual events between two of a pixel's events: differences of this (float64)
        Ob = O - log_o
        signal, log_s = crossing(torch.zeros_like(tau), torch.ones_like(first))
        terms = [signal + log_s + (Ob - Ob[gstart]).float()]  # crossing here, none before it in the gap (residuals between)
        # top / bottom of L since candidate q: the stretches between consecutive events from it up to this one, found once
        # per event and combined (where they sit: no gradient)
        with torch.no_grad():
            row, tp = Lg[b, :, pix], gq(tau, 1)

            def stretch(lo):  # value and grid index of the top and bottom of L strictly inside (lo, tau)
                inside = (taus > lo[:, None]) & (taus < tau[:, None])
                hi_i = torch.where(inside, row, -math.inf).argmax(1)
                lo_i = torch.where(inside, row, math.inf).argmin(1)
                has = inside.any(1)
                return (torch.where(has, row.gather(1, hi_i[:, None])[:, 0], -math.inf), hi_i,
                        torch.where(has, row.gather(1, lo_i[:, None])[:, 0], math.inf), lo_i)

            full = stretch(tp)
            hv, hi_, lv, li_ = torch.full_like(tau, -math.inf), torch.zeros_like(gs), torch.full_like(tau, math.inf), torch.zeros_like(gs)
        no = torch.zeros_like(first)
        for q, wk in enumerate(cand, 1):
            with torch.no_grad():  # add the stretch ending at the (q-1)-th event before
                u = gq(full[0], q - 1) > hv
                hv, hi_ = torch.where(u, gq(full[0], q - 1), hv), torch.where(u, gq(full[1], q - 1), hi_)
                u = gq(full[2], q - 1) < lv
                lv, li_ = torch.where(u, gq(full[2], q - 1), lv), torch.where(u, gq(full[3], q - 1), li_)
            top = torch.where(torch.isfinite(hv), Lg[b, hi_, pix], -math.inf)
            bot = torch.where(torch.isfinite(lv), Lg[b, li_, pix], math.inf)
            signal, log_s = crossing(wk, no, (top, bot))
            terms.append(signal + log_s + (Ob - gq(O, q)).float())
        beta = []  # log p(events of the pixel up to this one, this one a crossing), by rank
        for r, idx in enumerate(groups):
            beta.append(torch.logsumexp(torch.stack([terms[0][idx]] + [terms[q][idx] + beta[r - q][pos[idx - q]]
                                                                         for q in range(1, min(mix, r) + 1)]), 0))
        beta = torch.cat(beta)[inv]
        fin = [torch.where(rank[lidx] >= q, beta[(lidx - q).clamp(min=0)] + (O[lidx] - O[(lidx - q).clamp(min=0)]).float()
                           + tail[(lidx - q).clamp(min=0)], -math.inf) for q in range(mix)]
        fin.append((O[lidx] - Ob[gstart[lidx]]).float() + still[key[lidx]])  # no crossing at all in the gap
        ll = torch.zeros(B * H * W, device=tau.device).index_add(0, key[lidx], torch.logsumexp(torch.stack(fin), 0))
        return torch.where(seen, ll, still)

    ll = checkpoint(pixel_ll, Lg, Rg, use_reentrant=False) if torch.is_grad_enabled() else pixel_ll(Lg, Rg)
    return -(ll.sum() - Rg.sum()) / (B * H * W)


def photo_loss(model, s, mid, mid_tau):
    if mid.shape[1] == 0:
        return mid.new_zeros(())
    err = [(model.render(s, mid_tau[:, j])[0] - torch.log(model.bright(mid[:, j], s["ds"], s["cfa"]) + model.eps)).abs().mean()
           for j in range(mid.shape[1])]
    return sum(err) / len(err)


def compute(model, batch, w, stage):
    # stage: probe | teacher | student | joint | interp
    ev = (batch["ev_b"], batch["ev_pix"], batch["ev_tau"], batch["ev_pol"])
    if stage == "probe":
        target = torch.log1p(F.avg_pool2d(batch["voxel"], model.patch) * model.patch ** 2)
        loss = F.mse_loss(model.predict(batch["ctx"], batch["ctx_tau"], batch["dt"], head="counts"), target)
        return loss, dict(probe=loss)

    p = model.prepare(batch["i0"], batch["i1"], batch.get("ds"), batch.get("cfa"))
    if model.training and stage != "interp":  # interp: the noise table and the thresholds stay as trained
        if model.tube:
            model.measure_noise(p, batch["ev_b"], batch["ev_pix"])
        if model.measure_c:
            model.measure_thresholds(p, batch["ev_b"], batch["ev_pix"], batch["ev_pol"])
    terms = {}
    if stage in ("teacher", "joint", "interp"):
        z = model.encode(p, batch["voxel"])
        s = model.decode(p, z)
        Lr, Lg = path_grid(model, s, w["grid"])
        Rg = residual_grid(model, s, Lr, batch["dt"])
        terms["nll"] = event_nll(model, s, ev, batch["dt"], w["grid"], w.get("mix", MIX), (Lr, Lg, Rg))
        if stage == "interp":  # teacher only, decoder frozen: the path also matches the hidden real frames
            terms["photo"] = photo_loss(model, s, batch["mid"], batch["mid_tau"])
        else:
            terms["sigreg"] = sigreg_loss(z)
            if w.get("xt", 0) > 0:
                terms["xt"] = xt_loss(model, s, ev, batch["dt"], Lg, Rg, w.get("xt_blur", (2.0, 2.0)))
    if stage in ("student", "joint"):
        zt = model.encode(p, batch["voxel"]).detach() if stage == "student" else z
        zh = model.predict(batch["ctx"], batch["ctx_tau"], batch["dt"])
        terms.update(jepa=F.mse_loss(zh, zt), nll_student=event_nll(model, model.decode(p, zh), ev, batch["dt"], w["grid"],
                                                                   w.get("mix", MIX)))
    weight = dict(w, nll_student=w["nll"])
    return sum(weight[k] * v for k, v in terms.items()), terms


@torch.no_grad()
def diagnostics(model, batch, w, student=True):
    # does the decoder use z (shuffled / zero codes must hurt), and how much can the student predict (gap)?
    ev = (batch["ev_b"], batch["ev_pix"], batch["ev_tau"], batch["ev_pol"])
    p = model.prepare(batch["i0"], batch["i1"], batch.get("ds"), batch.get("cfa"))
    z = model.encode(p, batch["voxel"])
    nll = lambda code: event_nll(model, model.decode(p, code), ev, batch["dt"], w["grid"], w.get("mix", MIX)).item()
    out = dict(nll=nll(z), nll_shuffled=nll(z.roll(1, 0)), nll_zero=nll(torch.zeros_like(z)))
    s = model.decode(p, z)
    Lr, Lg = path_grid(model, s, w["grid"])
    Rg = residual_grid(model, s, Lr, batch["dt"])
    if w.get("xt", 0) > 0:
        out["xt"] = xt_loss(model, s, ev, batch["dt"], Lg, Rg, w.get("xt_blur", (2.0, 2.0))).item()
    dL, C = Lg[:, 1:] - Lg[:, :-1], model.sensor(s["ds"])[0]
    cross = (F.relu(dL).sum((1, 2)) / C[:, 0] + F.relu(-dL).sum((1, 2)) / C[:, 1]).sum()
    out["res_share"] = (Rg.sum() / (cross + Rg.sum())).item()  # share of the expected events that are residual events
    if batch["mid"].shape[1]:
        out["photo"] = photo_loss(model, s, batch["mid"], batch["mid_tau"]).item()
    if student:
        out["nll_student"] = nll(model.predict(batch["ctx"], batch["ctx_tau"], batch["dt"]))
        out["gap"] = out["nll_student"] - out["nll"]
    return out
