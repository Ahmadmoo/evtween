import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import lejepa
from model import taut, wiggle_k, wiggle_rate

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


SHIFTS = (-8, -4, -2, -1, 0, 1, 2, 4, 8)  # mid-gap timing errors (grid steps) averaged per pixel with the predicted uncertainty
MIX = 16  # with texture: how far back (in events) the last real crossing before an event is looked for (loss.mix)
XT_BINS = 64  # time bins of the x-t / y-t loss (the tau grid summed in groups)


def path_grid(model, s, steps=128):
    # the path the pixels see on the tau grid, (B, steps + 1, H*W): L, or the taut path in the tube
    B = s["y0"].shape[0]
    taus = torch.linspace(0, 1, steps + 1, device=s["y0"].device)
    frame = lambda t: model.render(s, t.expand(B))[0][:, 0]
    grad = torch.is_grad_enabled()  # recompute renders in backward: memory stays flat as the grid gets finer
    Lg = torch.stack([checkpoint(frame, t, use_reentrant=False) if grad else frame(t) for t in taus], 1).flatten(2)
    return taut(Lg, model.tube_eps(s)) if "tube" in s else Lg


def wiggle_grid(model, s, Lg):
    # texture events per unit tau and per polarity on each grid step (B, steps, H*W), from the wiggle size s["wiggle"]:
    # model.wiggle_rate with the path's own slope and the speed of the surface point the pixel sees (|dF0/dtau|, px per gap).
    # Slope and speed are taken as they are (no gradient): the wiggle size is learned, the path and the flow are not bent
    # to make texture
    B, steps = Lg.shape[0], Lg.shape[1] - 1
    with torch.no_grad():
        mid = (torch.arange(steps, device=Lg.device, dtype=Lg.dtype) + 0.5) / steps
        speed = torch.stack([model.flows(s, t.expand(B).view(B, 1, 1, 1))[1].norm(dim=1).flatten(1) for t in mid], 1)
        slope = (Lg[:, 1:] - Lg[:, :-1]).abs() * steps
    C = model.sensor(s["ds"])[0].sum(1)
    return wiggle_rate(s["wiggle"].flatten(1)[:, None], slope, speed, C[:, None, None])


def other_grid(model, s, Lg):
    # texture events (no reset) the path is expected to make on each grid step, ON and OFF: (B, 2, steps, H*W), or None
    steps = Lg.shape[1] - 1
    if "texture" in s:  # v5: a |dL| / (2 C_pol)
        C = model.sensor(s["ds"])[0]
        if "tex_q" in s:  # a share q with the slope's polarity (model.texture_pol)
            a, q, dL = s["texture"].flatten(1)[:, None], s["tex_q"].view(-1, 1, 1), Lg[:, 1:] - Lg[:, :-1]
            up, down = F.relu(dL) * a, F.relu(-dL) * a
            return torch.stack([(q * up + (1 - q) * down) / C[:, :1, None], (q * down + (1 - q) * up) / C[:, 1:, None]], 1)
        dL = (Lg[:, 1:] - Lg[:, :-1]).abs() * s["texture"].flatten(1)[:, None]
        return torch.stack([dL / (2 * C[:, :1, None]), dL / (2 * C[:, 1:, None])], 1)
    if "wiggle" in s:  # pairs: as many ON as OFF
        return (wiggle_grid(model, s, Lg) / steps)[:, None].expand(-1, 2, -1, -1)
    return None


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


def xt_loss(model, s, ev, dt, Lg, blur=(2.0, 2.0)):
    # x-t / y-t loss: the ON / OFF events the path is expected to make (rise / C_on, fall / C_off, texture, background) against
    # the real events, both cut into x-t slices (one per row, blurred in x and t) and y-t slices (one per column, blurred
    # in y and t), compared slice by slice with the generalized KL of counts (0 when equal). The blur (px, ms) compares the
    # curves, not single events: which pixel fires when inside a curve is the sensor's randomness
    b, pix, tau, pol = ev
    B, _, H, W = s["y0"].shape
    steps = Lg.shape[1] - 1
    T = XT_BINS if steps % XT_BINS == 0 else steps
    C, _, nu, _ = model.sensor(s["ds"])

    def run(Lg):
        dL = (Lg[:, 1:] - Lg[:, :-1]).view(B, T, steps // T, H * W)
        m = torch.stack([F.relu(dL).sum(2) / C[:, :1, None], F.relu(-dL).sum(2) / C[:, 1:, None]], 1)
        oth = other_grid(model, s, Lg)  # texture events count too: a 50/50 texture inside a one-polarity curve costs
        if oth is not None:
            m = m + oth.reshape(B, 2, T, steps // T, H * W).sum(3)
        m = (m + (nu * dt / (2 * T))[:, None, None, None]).view(B, 2, T, H, W)
        key = ((b * 2 + (pol < 0).long()) * T + (tau * T).long().clamp(0, T - 1)) * H * W + pix
        g = torch.zeros(B * 2 * T * H * W, device=m.device, dtype=m.dtype).index_add_(0, key, torch.ones_like(tau, dtype=m.dtype)).view_as(m)
        st = blur[1] * 1e-3 / dt * T  # time blur in bins, per sample
        return _xt_kl(m, g, st, torch.full_like(st, float(blur[0]))) / (B * H * W)

    return checkpoint(run, Lg, use_reentrant=False) if torch.is_grad_enabled() else run(Lg)


def _xt_kl(m, g, st, px):
    # expected counts m vs real counts g (B, 2, T, H, W): blurred in t (st bins), then x-t slices (blurred along x) and y-t
    # slices (blurred along y), each compared with the generalized KL of counts (0 when equal); summed, mean of the two
    mt, gt = _blur(m, 2, st), _blur(g, 2, st)
    out = 0
    for ax in (4, 3):
        mb, gb = _blur(mt, ax, px), _blur(gt, ax, px)
        out = out + (mb - gb + torch.xlogy(gb, gb) - gb * mb.clamp(min=1e-12).log()).sum() / 2
    return out


def mask_loss(model, p, ev, dt, share=0.25, hidden=(2, 3), blur=(2.0, 2.0)):
    # v5.3 (B), traj teacher: a second teacher pass on the first share of the batch with 2-3 random event slices hidden;
    # the teacher guesses their ON / OFF events per pixel (from the other slices along the motion, and the frames), scored
    # like the x-t loss (blurred x-t / y-t slices, generalized KL) where the visible slices are the real counts, so only
    # the hidden slices count (blur reaches across their edges)
    b, pix, tau, pol = ev
    n, S, H, W = max(1, round(len(dt) * share)), model.teacher.S, p["H"], p["W"]
    hide = torch.rand(n, S, device=dt.device).argsort(1).argsort(1) < torch.randint(hidden[0], hidden[1] + 1, (n, 1), device=dt.device)
    keep = b < n
    e = (b[keep], pix[keep], tau[keep], pol[keep])
    q = {k: v[:n] if torch.is_tensor(v) and v.dim() else v for k, v in p.items()}
    rate = model.encode(q, None, e, hide)[..., :H, :W].exp()
    T = rate.shape[2]
    g = _counts(e, n, T, H, W, rate.dtype)
    m = torch.where(hide.repeat_interleave(T // S, 1).view(n, 1, T, 1, 1), rate, g)
    st = blur[1] * 1e-3 / dt[:n] * T
    px = torch.full_like(st, float(blur[0]))
    return (checkpoint(_xt_kl, m, g, st, px, use_reentrant=False) if torch.is_grad_enabled() else _xt_kl(m, g, st, px)) / (n * H * W)


def _counts(ev, B, T, H, W, dtype):
    # real events binned into (B, 2 (ON, OFF), T, H, W)
    b, pix, tau, pol = ev
    key = ((b * 2 + (pol < 0).long()) * T + (tau * T).long().clamp(0, T - 1)) * H * W + pix
    return torch.zeros(B * 2 * T * H * W, device=tau.device, dtype=dtype).index_add_(0, key, torch.ones_like(tau, dtype=dtype)).view(B, 2, T, H, W)


def painted_grid(model, s, Lg, T):
    # wiggle texture events (ON, OFF) on each of T time steps, (B, 2, T, H*W), from one texture drawn by model.paint, as
    # the sampler makes them: the level A * texture each pixel sees, in rungs of (C_on + C_off) / 2 from a random height,
    # every rung crossed an event (ON up, OFF down), kept with probability exp(-|dL/dtau| / (A speed)). The counts are a
    # draw; the gradient is the expected count's (straight-through). Path and flow carry no gradient; the painter and A do
    B, steps = Lg.shape[0], Lg.shape[1] - 1
    H, W = s["y0"].shape[-2:]
    A = s["wiggle"]
    with torch.no_grad():
        gy, gx = torch.meshgrid(torch.arange(H, device=A.device, dtype=A.dtype), torch.arange(W, device=A.device, dtype=A.dtype),
                                indexing="ij")
        taus = torch.linspace(0, 1, T + 1, device=A.device)
        seen = [torch.stack([gx, gy])[None] + model.flows(s, t.expand(B).view(B, 1, 1, 1))[0] for t in taus]
        speed = torch.stack([(seen[i + 1] - seen[i]).norm(dim=1).flatten(1) * T for i in range(T)], 1)  # px per gap
        Lt = Lg[:, torch.linspace(0, steps, T + 1, device=A.device).round().long()]
        slope = (Lt[:, 1:] - Lt[:, :-1]).abs() * T
    tex = model.paint(s, seen, lambda shape: torch.randn(shape, device=A.device))
    rung = model.sensor(s["ds"])[0].sum(1).view(B, 1, 1) / 2
    u = torch.stack([(A * tex(q)).flatten(1) for q in seen], 1) / rung + torch.rand(B, 1, H * W, device=A.device)
    keep = torch.exp(-slope / (A.flatten(1)[:, None] * speed + 1e-6))
    d = u[:, 1:] - u[:, :-1]
    mean = torch.stack([F.relu(d), F.relu(-d)], 1) * keep[:, None]
    with torch.no_grad():
        n = u.floor()
        n = torch.stack([F.relu(n[:, 1:] - n[:, :-1]), F.relu(n[:, :-1] - n[:, 1:])], 1)
        draw = torch.distributions.Binomial(n, keep[:, None].expand_as(n).clamp(0, 1)).sample()
    return draw + mean - mean.detach()


def path_events(Lg, C, T):
    # the path's own ON / OFF events in each of T time steps, (B, 2, T, H*W), drawn with fixed thresholds C_on / C_off from
    # a random start (no threshold noise, no blind time): regular like a real pixel, not a Poisson count. No gradient
    B, steps = Lg.shape[0], Lg.shape[1] - 1
    con, coff = C[:, :1], C[:, 1:]
    ref = Lg[:, 0] - torch.rand_like(Lg[:, 0]) * (con + coff) + coff
    out = []
    for i in range(1, steps + 1):
        n_on = ((Lg[:, i] - ref) / con).floor().clamp(min=0)
        ref = ref + n_on * con
        n_off = ((ref - Lg[:, i]) / coff).floor().clamp(min=0)
        ref = ref - n_off * coff
        out.append(torch.stack([n_on, n_off], 1))
    return torch.stack(out, 2).view(B, 2, T, steps // T, -1).sum(3)


_FILTERS = {}


def _filters(device, n=64):
    # fixed random 3D filters over (ON, OFF) x (t, y, x) 5 x 5 x 5, zero mean, unit norm, the same in every run: their rectified
    # responses, averaged over a block, describe which small x-y-t patterns occur and how often (an OFF line with an ON line
    # one pixel behind it, a slanted streak, a lone dot), beyond what a spectrum can tell
    if device not in _FILTERS:
        w = torch.randn(n, 2, 5, 5, 5, generator=torch.Generator().manual_seed(0))
        w = w - w.mean((1, 2, 3, 4), keepdim=True)
        _FILTERS[device] = (w / w.flatten(1).norm(dim=1).view(-1, 1, 1, 1, 1)).to(device)
    return _FILTERS[device]


def stats_loss(model, s, ev, dt, Lg, block=(16, 32), power=2, blocks=8, kind="features"):
    # x-t / y-t pattern loss: the model's events (its path's, plus those of one texture drawn by the painter) against the
    # real events, compared by statistics of small x-y-t blocks (block: time bins of the gap's 64, px) at the same place,
    # not pixel by pixel. A block holds every x-t and y-t slice through it. kind "features": the mean rectified response
    # of fixed random 3D filters at two scales (which small patterns occur, and how often); "spectrum": band powers of
    # ON + OFF and ON - OFF (line width, slope, spacing). Compared as the squared difference of log. Both sides are draws
    # of events (the real sensor is far more regular than a Poisson count, so an expected count would not compare); the
    # gradient comes through the texture's expected count (straight-through). `blocks` random blocks per sample (features).
    # Trains the painter and the wiggle size, not the path
    B, steps = Lg.shape[0], Lg.shape[1] - 1
    H, W = s["y0"].shape[-2:]
    T = XT_BINS if steps % XT_BINS == 0 else steps
    bt, bp = block
    with torch.no_grad():
        path = path_events(Lg.detach(), model.sensor(s["ds"])[0], T)
    m = (path + painted_grid(model, s, Lg.detach(), T)).view(B, 2, T, H, W)
    g = _counts(ev, B, T, H, W, m.dtype)
    nt, ny, nx = T // bt, H // bp, W // bp
    if kind == "features":
        w, h = _filters(m.device), 4  # halo: the filters' reach at the larger scale
        pick = torch.randint(nt * ny * nx, (B, blocks), device=m.device)
        it, iy, ix = pick // (ny * nx), pick // nx % ny, pick % nx
        ar = lambda n: torch.arange(-h, n + h, device=m.device)
        ti = (it[..., None] * bt + ar(bt)).clamp(0, T - 1)
        yi = (iy[..., None] * bp + ar(bp)).clamp(0, H - 1)
        xi = (ix[..., None] * bp + ar(bp)).clamp(0, W - 1)
        bi = torch.arange(B, device=m.device).view(B, 1, 1, 1, 1)

        def feats(v):  # random blocks with a halo -> mean rectified responses (B * blocks, 2 x 64)
            c = v.permute(0, 2, 3, 4, 1)[bi, ti[..., :, None, None], yi[..., None, :, None], xi[..., None, None, :]]
            c = c.flatten(0, 1).permute(0, 4, 1, 2, 3)
            f = [F.relu(F.conv3d(c, w, dilation=d, padding=2 * d))[..., h:-h, h:-h, h:-h].mean((2, 3, 4)) for d in (1, 2)]
            return torch.cat(f + [c[..., h:-h, h:-h, h:-h].mean((2, 3, 4))], 1)

        return ((feats(m) + 0.01).log() - (feats(g) + 0.01).log()).abs().pow(power).mean()
    win = torch.hann_window(bt + 2, periodic=False, device=m.device)[1:-1].view(-1, 1, 1) * \
          torch.outer(*[torch.hann_window(bp + 2, periodic=False, device=m.device)[1:-1]] * 2)

    def bands(v):  # (B, 2, T, H, W) -> band powers of ON + OFF and ON - OFF per block
        v = v[:, :, :nt * bt, :ny * bp, :nx * bp].reshape(B, 2, nt, bt, ny, bp, nx, bp).permute(0, 2, 4, 6, 1, 3, 5, 7)
        x = torch.stack([v[..., 0, :, :, :] + v[..., 1, :, :, :], v[..., 0, :, :, :] - v[..., 1, :, :, :]], -4) * win
        P = torch.fft.fftshift(torch.fft.rfftn(x, dim=(-3, -2, -1)).abs() ** 2, dim=(-3, -2))
        return F.adaptive_avg_pool3d(P.flatten(0, 4), (bt // 2, bp // 4, bp // 8)).view(*x.shape[:5], -1)

    return ((bands(m) + 1).log() - (bands(g) + 1).log()).abs().pow(power).mean()


def polarity_loss(model, s, ev, dt, Lg, win=1.0):
    # the polarity part, per real event: of the events the model expects at that pixel within +-win ms (crossings: the
    # path's rise / C_on and fall / C_off; texture with its polarity share; background), the share with the event's
    # polarity; -log of it, averaged over events. It asks only "which polarity", not "how many" or "exactly when".
    # (Every event's score splits into "an event here, now" and "this polarity"; this weighs the second part again)
    b, pix, tau, pol = ev
    steps = Lg.shape[1] - 1
    C, _, nu, _ = model.sensor(s["ds"])
    dL = Lg[:, 1:] - Lg[:, :-1]
    zero = torch.zeros_like(Lg[:, :1])
    up, down = torch.cat([zero, F.relu(dL).cumsum(1)], 1), torch.cat([zero, F.relu(-dL).cumsum(1)], 1)  # rise / fall so far

    def at(cum, t):
        j = t * steps
        j0 = j.floor().clamp(0, steps - 1).long()
        f = j - j0
        return cum[b, j0, pix] * (1 - f) + cum[b, j0 + 1, pix] * f

    lo, hi = (tau - win * 1e-3 / dt[b]).clamp(0, 1), (tau + win * 1e-3 / dt[b]).clamp(0, 1)
    r, f, c = at(up, hi) - at(up, lo), at(down, hi) - at(down, lo), C[b]
    on, off = r / c[:, 0], f / c[:, 1]
    if "texture" in s:
        a, q = s["texture"].flatten(1)[b, pix], s["tex_q"][b] if "tex_q" in s else 0.5
        on, off = on + a * (q * r + (1 - q) * f) / c[:, 0], off + a * (q * f + (1 - q) * r) / c[:, 1]
    n = nu[b] * dt[b] * (hi - lo) / 2
    own = torch.where(pol > 0, on, off) + n
    return -(own / (on + off + 2 * n)).clamp(min=1e-12).log().mean()


def event_nll(model, s, ev, dt, steps=128, mix=MIX, Lg=None, patch=1):
    # point-process NLL of the real event times per pixel, same sensor model as physics.Simulator:
    # after every event the pixel draws thresholds X_on*C_on, X_off*C_off with X ~ IG(1, k) and takes a reference ref = L;
    # ON fires when the running max of (L - ref)/C_on reaches X_on (OFF: running max of (ref - L)/C_off);
    # background events (nu per second, half per polarity) compete with the signal and also reset the pixel;
    # after any event the pixel is blind for the refractory time R and takes ref = L at its end (not at the event);
    # the first wait of each pixel starts from the stationary state (its ref and thresholds are unknown).
    # With an uncertainty map s["sig"] (gap units) the timing of a pixel may be off by d ~ N(0, sig^2) in the middle of
    # the gap and the pixel's likelihood is averaged over d: a time warp moves all its events together, threshold noise
    # does not. The warp is pinned at both frames (their times are exact), so every d still scores the whole path.
    # With s["tube"] the events come from the least-event path inside the frame-noise tube (model.taut), not from L itself.
    # With s["texture"] (a per pixel) the path also hides up-and-down change a |dL/dtau|, split evenly into up and down:
    # random events at rate a |dL/dtau| / (2 C_pol) (texture the frames do not show). An up-and-back swing leaves the
    # pixel's reference where it was, so texture and background events do not reset the pixel: only real crossings do,
    # and the events of a pixel add up to its brightness change. Which events were crossings is unknown: a forward pass
    # over each pixel's events sums over it (the last crossing is among the `mix` events before, or there was none since the
    # gap started). Every pixel pays for
    # the texture events it expects, a * total variation of its path * (1/C_on + 1/C_off) / 2
    # With s["wiggle"] (A per pixel) the texture is a hidden brightness wiggle instead: on a curve it only moves the curve's
    # events in time (wider threshold spread per pixel, model.wiggle_k, same polarity as the curve); where the path is flat
    # and the surface moves it makes up-and-down pairs at model.wiggle_rate per polarity (summed over like v5's texture)
    # With s["tex_q"] (model.texture_pol) a share q of the texture events has the polarity of the path's slope, 1 - q the
    # other (v5: q = 0.5). patch > 1: the timing shift d is shared by each patch x patch block of pixels (wrong motion moves
    # a whole region's events together), not drawn per pixel.
    # Lg: the path on the grid from path_grid (computed here if not given)
    b, pix, tau, pol = ev
    B, _, H, W = s["y0"].shape
    taus = torch.linspace(0, 1, steps + 1, device=tau.device)
    grad = torch.is_grad_enabled()
    Lg = path_grid(model, s, steps) if Lg is None else Lg
    C, K, nu, R = model.sensor(s["ds"])  # per sample (each dataset has its own sensor)
    nu = nu * dt
    Kp = K[:, None].expand(B, H * W)  # threshold spread per pixel
    if "wiggle" in s:
        Kp = wiggle_k(K[:, None], s["wiggle"].flatten(1), C.sum(1, keepdim=True))
    c, k = C[b], Kp[b, pix][:, None]  # per event

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
    L0 = Lg[:, 0]  # pixels without events: no crossing over the whole path (the same for every warp)
    whole = (torch.stack([Lg.amax(1) - L0, L0 - Lg.amin(1)], 1) / C[:, :, None]).transpose(1, 2).reshape(-1, 2)
    still = ig_logs(whole, Kp.reshape(-1)[:, None])[2].sum(1)
    tex, cost, Rw = s.get("texture"), 0, None
    qe = s["tex_q"][b] if "tex_q" in s and tex is not None else None  # per event: texture share with the slope's polarity
    if tex is not None:
        tex = tex.flatten(1)
        if qe is None:
            cost = (tex * (Lg[:, 1:] - Lg[:, :-1]).abs().sum(1) * (1 / C).sum(1, keepdim=True) / 2).sum()
            tex = tex[b, pix] / (2 * c.gather(1, own)[:, 0])  # per event: texture rate per unit |dL/dtau|, its own polarity
        else:
            q, dLg = s["tex_q"][:, None], Lg[:, 1:] - Lg[:, :-1]
            up, down = F.relu(dLg).sum(1), F.relu(-dLg).sum(1)
            cost = (tex * (up * (q / C[:, :1] + (1 - q) / C[:, 1:]) + down * (q / C[:, 1:] + (1 - q) / C[:, :1]))).sum()
            tex = tex[b, pix] / c.gather(1, own)[:, 0]  # per event: per unit |dL/dtau|, before the polarity share
    elif "wiggle" in s:
        Rw = wiggle_grid(model, s, Lg)  # texture events per unit tau and polarity on each grid step
        cost = 2 * Rw.sum() / steps  # every pixel pays for the pairs it expects
    other = tex is not None or Rw is not None
    if other:
        # candidates for the last crossing before each event: the q-th event before it (q <= rank), or none since the gap
        # started (always allowed, so every event sequence has an explanation)
        gs = torch.arange(len(tau), device=tau.device)
        gq = lambda x, q: x[(gs - q).clamp(min=0)]  # value at the q-th event before (same pixel when q <= rank)
        gstart = torch.where(first, gs, 0).cummax(0).values
        rank = gs - gstart
        mix = min(mix, int(rank.max()) if len(rank) else 0)  # no pixel has more events before: no further candidates
        cand = [(torch.minimum(gq(tau, q) + Rt, tau), torch.sigmoid((tau - gq(tau, q) - Rt) / wR)) for q in range(1, mix + 1)]
        order = torch.argsort(rank, stable=True)
        nrank = torch.bincount(rank)
        pos = torch.empty_like(order)
        pos[order] = gs - torch.repeat_interleave(nrank.cumsum(0) - nrank, nrank)
        groups, inv = torch.split(order, nrank.tolist()), torch.argsort(order)
        lidx = last.nonzero()[:, 0]

    def pixel_ll(Lg, d):
        # log-likelihood per pixel with timing off by d (gap units) mid-gap: events at t see L(phi(t)), phi(t) = t - d sin(pi t)
        # (phi(0) = 0, phi(1) = 1, monotone for |d| < 1 / pi)
        phi = lambda t: t - d * torch.sin(math.pi * t)
        dphi = lambda t: 1 - d * math.pi * torch.cos(math.pi * t)

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

        t = phi(tau)
        Li, Di = at(t), slope(t) * dphi(tau)

        def crossing(wake, awake, first, before=None):
            # log density of a crossing at t (own polarity) since the pixel woke at `wake` (first: stationary start),
            # and the log survival of both thresholds up to t (before: (top, bottom) of L since the wake, if known)
            tw = phi(wake)
            Lp = at(tw)
            if before is None:
                before = reach(tw, t, Lp)
            else:
                before = torch.stack([before[0].maximum(Lp) - Lp, Lp - before[1].minimum(Lp)], 1) / c
            now = torch.stack([Li - Lp, Lp - Li], 1) / c
            M = torch.maximum(before, now)
            # signal only at a new max (a wider gate biases c low when k is small) and never while blind
            rate = torch.stack([F.relu(Di), F.relu(-Di)], 1) / c * torch.sigmoid((now - before) / 1e-3) * awake[:, None]
            logpdf, logsf, logsf_eq = ig_logs(M, k)
            log_h = torch.where(first[:, None], logsf - logsf_eq, logpdf - logsf)
            log_s = torch.where(first[:, None], logsf_eq, logsf)
            return (rate.clamp(min=1e-12).log() + log_h).gather(1, own)[:, 0], log_s.sum(1)

        end = phi((tau + Rt).clamp(max=1))
        tail = ig_logs(reach(end, torch.full_like(end, 1 + 0.5 / steps), at(end)), k)[1].sum(1)  # to the end of the path
        if not other:  # every event resets the pixel: crossing or background
            signal, log_s = crossing(wake, awake, first)
            log_ev = torch.logaddexp(signal, (nu[b] / 2).log()) + log_s
            ll = torch.zeros(B * H * W, device=tau.device).index_add(0, key, log_ev).index_add(0, key[last], tail[last])
            return torch.where(seen, ll, still)

        if Rw is None:
            share = 1 if qe is None else torch.where((Di >= 0) == (pol > 0), qe, 1 - qe)
            log_o = (nu[b] / 2 + tex * share * Di.abs()).log()  # background + texture events of this polarity (no reset)
        else:
            j = (t * steps).floor().clamp(0, steps - 1).long()
            log_o = (nu[b] / 2 + Rw[b, j, pix] * dphi(tau).abs()).log()  # (abs: as v5's |Di|, for coarse grids)
        O = log_o.double().cumsum(0)  # sums of "other" events between two of a pixel's events: differences of this (float64)
        Ob = O - log_o
        signal, log_s = crossing(torch.zeros_like(tau), torch.ones_like(tau), torch.ones_like(first))
        terms = [signal + log_s + (Ob - Ob[gstart]).float()]  # crossing here, none before it in the gap (others between)
        # top / bottom of L since candidate q's wake: the stretch from its wake to the next event, then whole stretches
        # between consecutive events up to this one, found once per event and combined (where they sit: no gradient)
        with torch.no_grad():
            row, tp = Lg[b, :, pix], gq(tau, 1)

            def stretch(lo):  # value and grid index of the top and bottom of L strictly inside (lo, t)
                inside = (taus > lo[:, None]) & (taus < t[:, None])
                hi_i = torch.where(inside, row, -math.inf).argmax(1)
                lo_i = torch.where(inside, row, math.inf).argmin(1)
                has = inside.any(1)
                return (torch.where(has, row.gather(1, hi_i[:, None])[:, 0], -math.inf), hi_i,
                        torch.where(has, row.gather(1, lo_i[:, None])[:, 0], math.inf), lo_i)

            part, full = stretch(phi(torch.minimum(tp + Rt, tau))), stretch(phi(tp))
            hv, hi_, lv, li_ = torch.full_like(tau, -math.inf), torch.zeros_like(gs), torch.full_like(tau, math.inf), torch.zeros_like(gs)
        no = torch.zeros_like(first)
        for q, (wk, aw) in enumerate(cand, 1):
            with torch.no_grad():
                if q >= 2:  # whole stretch ending at the (q-2)-th event before
                    u = gq(full[0], q - 2) > hv
                    hv, hi_ = torch.where(u, gq(full[0], q - 2), hv), torch.where(u, gq(full[1], q - 2), hi_)
                    u = gq(full[2], q - 2) < lv
                    lv, li_ = torch.where(u, gq(full[2], q - 2), lv), torch.where(u, gq(full[3], q - 2), li_)
                u = gq(part[0], q - 1) > hv
                tv, ti = torch.where(u, gq(part[0], q - 1), hv), torch.where(u, gq(part[1], q - 1), hi_)
                u = gq(part[2], q - 1) < lv
                bv, bi = torch.where(u, gq(part[2], q - 1), lv), torch.where(u, gq(part[3], q - 1), li_)
            top = torch.where(torch.isfinite(tv), Lg[b, ti, pix], -math.inf)
            bot = torch.where(torch.isfinite(bv), Lg[b, bi, pix], math.inf)
            signal, log_s = crossing(wk, aw, no, (top, bot))
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

    shifts = SHIFTS if "sig" in s else (0,)
    lls = torch.stack([checkpoint(pixel_ll, Lg, q / steps, use_reentrant=False) if grad else pixel_ll(Lg, q / steps) for q in shifts])
    if len(shifts) > 1:
        d = torch.tensor(shifts, device=tau.device, dtype=torch.float32) / steps
        sig = s["sig"]
        if patch > 1:  # one shift per patch: sum the patch's pixels for each shift, then average over shifts
            ph, pw = -H % patch, -W % patch
            lls = F.pad(lls.view(len(shifts), B, H, W), (0, pw, 0, ph)).view(
                len(shifts), B, (H + ph) // patch, patch, (W + pw) // patch, patch).sum((3, 5)).flatten(1)
            sig = F.avg_pool2d(F.pad(sig, (0, pw, 0, ph), mode="replicate"), patch)
        logw = (-0.5 * (d[:, None] / sig.flatten()[None]) ** 2 + torch.gradient(d)[0].log()[:, None]).log_softmax(0)
        lls = torch.logsumexp(logw + lls, 0, keepdim=True)
    return -(lls[0].sum() - (nu * H * W).sum() - cost) / (B * H * W)


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
    # edge-aware first-order smoothness of the path coefficients (and of the tube factor and log texture when present):
    # neighbouring pixels get the same path over the whole gap, so an edge reaches them at consistent times
    x = torch.cat([s["a"].flatten(1, 2), s["b"].flatten(1, 2), s["d"]] + ([s["tube"]] if "tube" in s else [])
                  + ([s["texture"].log()] if "texture" in s else []) + ([s["wiggle"].log()] if "wiggle" in s else []), 1)
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    wx = torch.exp(-10 * dx(img).abs().mean(1, keepdim=True))
    wy = torch.exp(-10 * dy(img).abs().mean(1, keepdim=True))
    return (dx(x).abs() * wx).mean() + (dy(x).abs() * wy).mean()


def compute(model, batch, w, stage):
    # stage: probe | teacher | student | joint
    ev = (batch["ev_b"], batch["ev_pix"], batch["ev_tau"], batch["ev_pol"])
    if stage == "probe":
        target = torch.log1p(F.avg_pool2d(batch["voxel"], model.patch) * model.patch ** 2)
        loss = F.mse_loss(model.predict(batch["ctx"], batch["ctx_tau"], batch["dt"], head="counts"), target)
        return loss, dict(probe=loss)

    p = model.prepare(batch["i0"], batch["i1"], batch.get("ds"), batch.get("cfa"))
    if model.tube and model.training:
        model.measure_noise(p, batch["ev_b"], batch["ev_pix"])
    if model.measure_c and model.training:
        model.measure_thresholds(p, batch["ev_b"], batch["ev_pix"], batch["ev_pol"])
    terms = {}
    if stage in ("teacher", "joint"):
        z = model.encode(p, batch["voxel"], ev)
        n = None
        if model.detail:  # the random part of the detail image, read from the events (a sample: the KL prices it)
            n, terms["detail_kl"] = model.detail_code(p, batch["voxel"])
        s = model.decode(p, z, n)
        Lg = path_grid(model, s, w["grid"])
        terms.update(nll=event_nll(model, s, ev, batch["dt"], w["grid"], w.get("mix", MIX), Lg, w.get("sig_patch", 1)),
                     photo=photo_loss(model, s, batch["mid"], batch["mid_tau"]),
                     cmax=cmax_loss(model, s, ev), smooth=smooth_loss(s, batch["i0"]),
                     sigreg=sigreg_loss(z))
        if w.get("xt", 0) > 0:
            terms["xt"] = xt_loss(model, s, ev, batch["dt"], Lg, w.get("xt_blur", (2.0, 2.0)))
        if w.get("nll_pol", 0) > 0:
            terms["nll_pol"] = polarity_loss(model, s, ev, batch["dt"], Lg, w.get("pol_win", 1.0))
        if w.get("stats", 0) > 0 and model.painter is not None:
            terms["stats"] = stats_loss(model, s, ev, batch["dt"], Lg, w.get("stats_block", (16, 32)), kind=w.get("stats_kind", "features"))
        if w.get("mask", 0) > 0 and hasattr(model.teacher, "guess"):
            terms["mask"] = mask_loss(model, p, ev, batch["dt"], w.get("mask_share", 0.25), w.get("mask_slices", (2, 3)),
                                      w.get("xt_blur", (2.0, 2.0)))
    if stage in ("student", "joint"):
        zt = model.encode(p, batch["voxel"], ev).detach() if stage == "student" else z
        zh = model.predict(batch["ctx"], batch["ctx_tau"], batch["dt"])
        s = model.decode(p, zh)  # detail image: its random part drawn from the prior, as in generation
        terms.update(jepa=F.mse_loss(zh, zt), nll_student=event_nll(model, s, ev, batch["dt"], w["grid"], w.get("mix", MIX),
                                                                   patch=w.get("sig_patch", 1)))
        if w.get("nll_pol", 0) > 0:
            terms["nll_pol_student"] = polarity_loss(model, s, ev, batch["dt"], path_grid(model, s, w["grid"]), w.get("pol_win", 1.0))
        if stage == "student":
            terms["photo"] = photo_loss(model, s, batch["mid"], batch["mid_tau"])
    weight = dict(w, nll_student=w["nll"], detail_kl=w.get("detail_kl", 1.0), nll_pol_student=w.get("nll_pol", 0))
    return sum(weight[k] * v for k, v in terms.items()), terms


@torch.no_grad()
def diagnostics(model, batch, w, student=True):
    # does the decoder use z (shuffled / zero codes must hurt), and how much can the student predict (gap)?
    ev = (batch["ev_b"], batch["ev_pix"], batch["ev_tau"], batch["ev_pol"])
    # detail image: the teacher's codes use its posterior mean; nll_no_detail drops the random part (how much it helps);
    # the student draws it from the prior, as in generation
    p = model.prepare(batch["i0"], batch["i1"], batch.get("ds"), batch.get("cfa"))
    z = model.encode(p, batch["voxel"], ev)
    n, kl = model.detail_code(p, batch["voxel"], sample=False) if model.detail else (None, None)
    nll = lambda code, n=n: event_nll(model, model.decode(p, code, n), ev, batch["dt"], w["grid"], w.get("mix", MIX),
                                      patch=w.get("sig_patch", 1)).item()
    out = dict(nll=nll(z), nll_shuffled=nll(z.roll(1, 0)), nll_zero=nll(torch.zeros_like(z)))
    if model.detail:
        out.update(nll_no_detail=nll(z, torch.zeros_like(n)), detail_kl=kl.item())
    s = model.decode(p, z, n)
    if w.get("xt", 0) > 0:
        out["xt"] = xt_loss(model, s, ev, batch["dt"], path_grid(model, s, w["grid"]), w.get("xt_blur", (2.0, 2.0))).item()
    if w.get("stats", 0) > 0 and model.painter is not None:
        out["stats"] = stats_loss(model, s, ev, batch["dt"], path_grid(model, s, w["grid"]), w.get("stats_block", (16, 32)),
                                  kind=w.get("stats_kind", "features")).item()
    if w.get("nll_pol", 0) > 0:
        out["nll_pol"] = polarity_loss(model, s, ev, batch["dt"], path_grid(model, s, w["grid"]), w.get("pol_win", 1.0)).item()
    if w.get("mask", 0) > 0 and hasattr(model.teacher, "guess"):  # masked-slice guess, all samples
        out["mask"] = mask_loss(model, p, ev, batch["dt"], 1.0, w.get("mask_slices", (2, 3)), w.get("xt_blur", (2.0, 2.0))).item()
    if "tex_q" in s:
        out["q"] = s["tex_q"].mean().item()  # texture share with the slope's polarity
    if "wiggle" in s:
        out["wiggle"] = s["wiggle"].mean().item()  # mean wiggle size (log units)
    if student:
        out["nll_student"] = nll(model.predict(batch["ctx"], batch["ctx_tau"], batch["dt"]), None)
        out["gap"] = out["nll_student"] - out["nll"]
    return out
