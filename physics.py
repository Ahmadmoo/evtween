import math
import numpy as np
import torch
import yaml

TUBE_STEPS = 128  # with the tube the path is fixed on a grid first, at least as fine as the likelihood's (loss.grid)


def first_root(a, b, c, lo, hi):
    # smallest root of a*h^2 + b*h + c = 0 inside (lo, hi], inf if none (stable quadratic formula)
    disc = b * b - 4 * a * c
    sq = disc.clamp(min=0).sqrt()
    q = -0.5 * (b + torch.where(b >= 0, sq, -sq))
    inf = torch.full_like(b, math.inf)
    r1 = torch.where(a.abs() > 1e-12, q / a, inf)
    r2 = torch.where(q.abs() > 1e-12, c / q, inf)
    ok = lambda r: (disc >= 0) & (r > lo) & (r <= hi)
    return torch.minimum(torch.where(ok(r1), r1, inf), torch.where(ok(r2), r2, inf))


def profile(model, ds=0):
    # the learned sensor of dataset ds in physical units (v2e names and meanings where v2e has the parameter);
    # sigma_event: per-event threshold noise as a fraction of the threshold;
    # sigma_thres: fixed per-pixel threshold spread in log units, as in v2e (not learned, v2e's default 0.03)
    # shot_noise_rate_hz: background events / px / s at z = 0 (model.nu); the decoder's per-pixel background is scaled by
    # this over model.nu (so 0 removes it)
    C, k, R = (v[0] for v in model.sensor(torch.tensor([ds], device=model.log_c.device)))
    return dict(pos_thres=C[0].item(), neg_thres=C[1].item(), sigma_event=k.item() ** -0.5,
                sigma_thres=0.03, shot_noise_rate_hz=model.nu0, refractory_period_s=R.item())


def resolve(model, name="learned", overrides=None, ds=0):
    # learned | clean | profile YAML (any subset of keys), then overrides: a number, or "x1.5" = 1.5 x the profile value
    p = profile(model, ds)
    if name == "clean":  # like v2e --dvs_params clean: no noise, small threshold spread
        p.update(sigma_event=0.03, sigma_thres=0.0, shot_noise_rate_hz=0.0)
    elif name != "learned":
        p.update(yaml.safe_load(open(name)))
    for k, v in (overrides or {}).items():
        assert k in p, f"unknown sensor parameter {k}; use one of {list(p)}"
        p[k] = p[k] * float(v[1:]) if isinstance(v, str) and v.startswith("x") else float(v)
    return p


class Simulator:
    # samples the sensor model the likelihood is trained on (losses.event_nll) with a sensor profile (default: the learned one).
    # Crossings: thresholds X*C with X ~ IG(1, sigma_event^-2) on the taut path in the frame-noise tube (on a grid of at least
    # TUBE_STEPS steps, exact crossing of its linear pieces; without a tube: exact crossing of the local quadratic of L);
    # after a crossing the pixel is blind for the refractory time and takes ref = L at the end of it; sigma_thres adds a
    # fixed per-pixel threshold offset (sampler only). Residual events (model.residual_rate: kappa x the raw path's slope,
    # plus the background nu, scaled by the profile's shot_noise_rate_hz / model.nu) come on top as a Poisson process and
    # leave the pixel's state alone. Output "s": where each event comes from, 0 crossing, 1 residual (slope), 2 background.
    # residual_events=False leaves out residual and background events: the crossings only
    def __init__(self, model, profile=None, min_steps=4, max_steps=256, seed=0, max_events=64, residual_events=True):
        self.m, self.p = model, profile or resolve(model)
        self.min_steps, self.max_steps, self.max_events = min_steps, max_steps, max_events
        self.residual_events = residual_events
        self.gen = torch.Generator().manual_seed(seed)
        self.ref = self.last = None

    def _u(self, shape):
        return torch.rand(shape, generator=self.gen, dtype=torch.float64).to(self.dev).clamp(min=1e-12)

    def _ig(self, shape):
        # inverse Gaussian, mean 1, shape k (Michael, Schucany & Haas)
        k = self.k
        y = torch.randn(shape, generator=self.gen, dtype=torch.float64).to(self.dev) ** 2
        x = 1 + y / (2 * k) - torch.sqrt(4 * k * y + y * y) / (2 * k)
        return torch.where(self._u(shape) <= 1 / (1 + x), x, 1 / x).float()

    @torch.no_grad()
    def run(self, s, t0, t1):
        # all events of one frame interval [t0, t1] (seconds); pixel state carries over between calls
        m, dt, p = self.m, t1 - t0, self.p
        self.dev, self.k, self.R = s["d"].device, max(p["sigma_event"], 1e-3) ** -2, p["refractory_period_s"]
        tau = lambda v: torch.full((1,), v, device=self.dev)
        La, Da, _ = m.render(s, tau(0.0))
        fresh = self.ref is None
        if fresh:
            shape = (2, *La.shape)
            spread = p["sigma_thres"] * torch.randn(shape, generator=self.gen).to(self.dev)
            self.c = (torch.tensor([p["pos_thres"], p["neg_thres"]], device=self.dev).view(2, 1, 1, 1, 1) + spread).clamp(min=0.01)
            self.ref = La.clone()
            self.x = (self._u(shape) / self._ig(shape)).float()  # stationary start: uniform part of a size-biased wait
            self.wake = torch.full_like(La, -math.inf, dtype=torch.float64)  # end of the blind time
            self.pending = torch.zeros_like(La, dtype=torch.bool)             # blind past the current step: ref not set yet

        speed = max(m.render(s, tau(v))[1].abs().flatten().quantile(0.999).item() for v in (0.0, 0.5, 1.0))
        S = int(min(max(math.ceil(2 * speed / self.c.min().item()), self.min_steps), self.max_steps))
        L = g = None
        if m.tube:  # each gap's path starts free inside the tube: the jump from the last gap's end fires nothing
            S = max(S, TUBE_STEPS)
            L = torch.stack([m.render(s, tau(i / S))[0] for i in range(S + 1)], 1)
            g = m.paths(L.flatten(2), s).view_as(L)
            self.ref = g[:, 0].clone() if fresh else self.ref + g[:, 0] - self.last
            self.last = g[:, -1]
        B, H, out = La.shape[0], 1.0 / S, []
        scale = p["shot_noise_rate_hz"] / m.nu0  # background as set in the profile (learned: 1)
        C = self.c.transpose(0, 1).flatten(3)  # (B, 2, 1, H*W) thresholds per pixel
        bg = (s["nu"] * scale * dt).transpose(0, 1)[:, :, None]  # background per unit tau, (2, B, 1, H, W)
        for i in range(S):
            if g is not None:  # crossings: the taut path, linear between grid points
                La, Lb = g[:, i], g[:, i + 1]
                Da, alpha = (Lb - La) / H, torch.zeros_like(La)
                ra, rb = L[:, i], L[:, i + 1]  # residual events: the raw path
            else:
                Lb, Db, _ = m.render(s, tau((i + 1) * H))
                alpha = (Lb - La - Da * H) / H ** 2
                ra, rb = La, Lb
            res = None
            if self.residual_events:  # residual events per unit tau, ON / OFF: (2, B, 1, H, W)
                res = m.residual_rate(s, ((rb - ra) / H).flatten(1)[:, None], dt, scale, C).view(B, 2, *La.shape[1:]).transpose(0, 1)
            self._cross(La, Da, alpha, H, t0 + dt * i * H, dt, out, res, bg)
            if g is None:
                La, Da = Lb, Db
        t, x, y, p, src = (torch.cat(v).cpu().numpy() for v in zip(*out)) if out else ([np.zeros(0)] * 5)
        o = np.argsort(t, kind="stable")
        return dict(t=t[o], x=x[o].astype(np.int16), y=y[o].astype(np.int16), p=p[o].astype(np.int8), s=src[o].astype(np.int8))

    def warm(self, s, t0, t1):
        # run a gap and drop its events (e.g. the gap before the one of interest): the pixels then start in a real state
        # (reference, thresholds, blind time) instead of the stationary guess, so the first events of the gap are no burst
        self.run(s, t0, t1)

    def _cross(self, La, Da, alpha, H, ta, dt, out, res=None, bg=None):
        q = lambda h: La + Da * h + alpha * h * h
        hw = ((self.wake - ta) / dt).float().clamp(min=0)  # where in this step each pixel wakes up
        woke = self.pending & (hw <= H)
        self.ref = torch.where(woke, q(hw), self.ref)
        self.pending = self.pending & ~woke
        h_last, inf = torch.zeros_like(La), torch.full_like(La, math.inf)
        if res is not None:  # residual events: a Poisson process in this step (ON + OFF), its next time drawn ahead (memoryless)
            lam = res[0] + res[1]
            next_res = lambda h: h - self._u(La.shape).log().float() / lam
            h_res = next_res(h_last)
        for _ in range(self.max_events):
            lo = torch.where(self.pending, inf, torch.maximum(h_last, hw))  # no signal while blind
            r_on = first_root(alpha, Da, La - self.ref - self.x[0] * self.c[0], lo, H)
            r_off = first_root(alpha, Da, La - self.ref + self.x[1] * self.c[1], lo, H)
            r_res = inf if res is None else torch.where(h_res <= H, h_res, inf)
            r_sig = torch.minimum(r_on, r_off)
            h = torch.minimum(r_sig, r_res)
            fire = torch.isfinite(h)
            if not fire.any():
                break
            resid = fire & (r_res < r_sig)
            pol = torch.where(r_on <= r_off, 1, -1)
            src = torch.zeros_like(La, dtype=torch.int8)
            if res is not None:  # polarity by its share of the rate, then background or slope by their shares
                on = self._u(La.shape) < res[0] / lam
                pol = torch.where(resid, torch.where(on, 1, -1), pol)
                share = torch.where(on, bg[0] / res[0], bg[1] / res[1])
                src = torch.where(resid, torch.where(self._u(La.shape) < share, 2, 1), 0).to(torch.int8)
            h_last = torch.where(fire, h, h_last)
            t = ta + dt * h_last.double()
            reset = fire & ~resid  # only crossings reset the pixel
            self.wake = torch.where(reset, t + self.R, self.wake)
            hw = torch.where(reset, ((self.wake - ta) / dt).float(), hw)
            self.pending = torch.where(reset, hw > H, self.pending)
            self.ref = torch.where(reset & ~self.pending, q(hw), self.ref)
            self.x = torch.where(reset, self._ig(self.x.shape), self.x)
            if res is not None:
                h_res = torch.where(resid, next_res(h_last), h_res)
            i = fire.nonzero(as_tuple=True)
            out.append((t[i], i[3], i[2], pol[i], src[i]))
