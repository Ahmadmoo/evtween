import math
import numpy as np
import torch
import yaml

GRID = 128  # the sampler's path grid has at least these steps per gap, as fine as the likelihood's (loss.grid)


def first_root(d, c, lo, hi):
    # root of c + d*h = 0 inside (lo, hi], inf if none
    r = -c / torch.where(d.abs() > 1e-12, d, torch.ones_like(d))
    return torch.where((d.abs() > 1e-12) & (r > lo) & (r <= hi), r, torch.full_like(r, math.inf))


def profile(model, ds=0):
    # the learned sensor of dataset ds in physical units (v2e names and meanings where v2e has the parameter);
    # sigma_event: per-event threshold noise as a fraction of the threshold;
    # sigma_thres: fixed per-pixel threshold spread in log units, as in v2e (not learned, v2e's default 0.03)
    # shot_noise_rate_hz: background events / px / s at z = 0 (model.nu); the decoder's per-pixel background is scaled by
    # this over model.nu (so 0 removes it)
    C, k = (v[0] for v in model.sensor(torch.tensor([ds], device=model.log_c.device)))
    return dict(pos_thres=C[0].item(), neg_thres=C[1].item(), sigma_event=k.item() ** -0.5,
                sigma_thres=0.03, shot_noise_rate_hz=model.nu0)


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
    # GRID steps, exact crossing of its linear pieces); after a crossing the pixel takes ref = L; sigma_thres adds a fixed
    # per-pixel threshold offset (sampler only). On top, leaving the pixel's crossing state alone (model.residual_rate):
    # slope events (kappa x the raw path's rise and fall) fire like a pixel does: each polarity adds up its expected events
    # and fires each time the sum passes a whole number (uniform start phase, kept between gaps), so every pixel an edge
    # passes fires (no holes along edges); the background (nu, scaled by the profile's shot_noise_rate_hz / model.nu) is a
    # Poisson process. The likelihood scores slope events as Poisson with the same rate (same expected count).
    # Output "s": 0 crossing, 1 slope event, 2 background.
    # residual_events=False leaves out slope and background events: the crossings only
    def __init__(self, model, profile=None, max_steps=256, seed=0, max_events=64, residual_events=True):
        self.m, self.p = model, profile or resolve(model)
        self.max_steps, self.max_events = max_steps, max_events
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
        self.dev, self.k = s["d"].device, max(p["sigma_event"], 1e-3) ** -2
        path = lambda n: torch.stack([m.render(s, torch.full((1,), i / n, device=self.dev))[0] for i in range(n + 1)], 1)
        L = path(GRID)  # raw path (B, steps + 1, 1, H, W)
        fresh = self.ref is None
        if fresh:
            shape = (2, *L[:, 0].shape)
            spread = p["sigma_thres"] * torch.randn(shape, generator=self.gen).to(self.dev)
            self.c = (torch.tensor([p["pos_thres"], p["neg_thres"]], device=self.dev).view(2, 1, 1, 1, 1) + spread).clamp(min=0.01)
            self.ref = L[:, 0].clone()
            self.x = (self._u(shape) / self._ig(shape)).float()  # stationary start: uniform part of a size-biased wait
            self.w = self._u(shape).float()  # slope events: expected count left to each polarity's next one (uniform phase)
        # finer grid where the path is fast: per step it moves at most half the smallest threshold (99.9% of pixels)
        speed = (L[:, 1:] - L[:, :-1]).abs().amax(1).flatten().quantile(0.999).item() * GRID
        S = int(min(max(math.ceil(2 * speed / self.c.min().item()), GRID), self.max_steps))
        if S > GRID:
            L = path(S)
        g = m.paths(L.flatten(2), s).view_as(L)  # crossings: the taut path in the tube (L without a tube)
        if m.tube:  # each gap's path starts free inside the tube: the jump from the last gap's end fires nothing
            self.ref = g[:, 0].clone() if fresh else self.ref + g[:, 0] - self.last
            self.last = g[:, -1]
        B, H, out = L.shape[0], 1.0 / S, []
        scale = p["shot_noise_rate_hz"] / m.nu0  # background as set in the profile (learned: 1)
        C = self.c.transpose(0, 1).flatten(3)  # (B, 2, 1, H*W) thresholds per pixel
        # background per unit tau, (2, B, 1, H, W); none (and no slope events) with residual_events=False
        bg = (s["nu"] * scale * dt * self.residual_events).transpose(0, 1)[:, :, None]
        rate = torch.zeros_like(bg)
        for i in range(S):
            La, Lb = g[:, i], g[:, i + 1]
            if self.residual_events:  # slope events per unit tau from the raw path, ON / OFF: (2, B, 1, H, W)
                rate = m.residual_rate(s, ((L[:, i + 1] - L[:, i]) / H).flatten(1)[:, None], dt, 0.0, C)
                rate = rate.view(B, 2, *La.shape[1:]).transpose(0, 1)
            self._cross(La, (Lb - La) / H, H, t0 + dt * i * H, dt, out, rate, bg)
        t, x, y, p, src = (torch.cat(v).cpu().numpy() for v in zip(*out)) if out else ([np.zeros(0)] * 5)
        o = np.argsort(t, kind="stable")
        return dict(t=t[o], x=x[o].astype(np.int16), y=y[o].astype(np.int16), p=p[o].astype(np.int8), s=src[o].astype(np.int8))

    def warm(self, s, t0, t1):
        # run a gap and drop its events (e.g. the gap before the one of interest): the pixels then start in a real state
        # (reference, thresholds) instead of the stationary guess, so the first events of the gap are no burst
        self.run(s, t0, t1)

    def _cross(self, La, Da, H, ta, dt, out, rate, bg):
        # events on the linear piece La + Da h of the path, h in (0, H]: crossings; slope events at rate (2, B, 1, H, W) per
        # unit tau, each polarity when its count passes the next whole number; background at rate bg, Poisson (ON + OFF,
        # its next time drawn ahead: memoryless)
        h_last, inf = torch.zeros_like(La), torch.full_like(La, math.inf)
        lam = bg[0] + bg[1]
        next_bg = lambda h: h - self._u(La.shape).log().float() / lam
        h_ev, h_bg = torch.where(rate > 0, self.w / rate, inf), next_bg(h_last)
        for _ in range(self.max_events):
            r_on = first_root(Da, La - self.ref - self.x[0] * self.c[0], h_last, H)
            r_off = first_root(Da, La - self.ref + self.x[1] * self.c[1], h_last, H)
            r_sig, q = torch.minimum(r_on, r_off), h_ev[0] <= h_ev[1]
            r_ev = torch.minimum(h_ev[0], h_ev[1])
            r_ev, r_bg = torch.where(r_ev <= H, r_ev, inf), torch.where(h_bg <= H, h_bg, inf)
            h = torch.minimum(r_sig, torch.minimum(r_ev, r_bg))
            fire = torch.isfinite(h)
            if not fire.any():
                break
            cross = fire & (r_sig <= torch.minimum(r_ev, r_bg))
            slope = fire & ~cross & (r_ev <= r_bg)
            back = fire & ~cross & ~slope
            on = torch.where(cross, r_on <= r_off, torch.where(slope, q, self._u(La.shape) < bg[0] / lam))
            h_last = torch.where(fire, h, h_last)
            self.ref = torch.where(cross, La + Da * h_last, self.ref)  # only crossings reset the pixel
            self.x = torch.where(cross, self._ig(self.x.shape), self.x)
            h_ev = torch.where(torch.stack([slope & q, slope & ~q]), h_ev + 1 / rate, h_ev)
            h_bg = torch.where(back, next_bg(h_last), h_bg)
            t = ta + dt * h_last.double()
            i = fire.nonzero(as_tuple=True)
            out.append((t[i], i[3], i[2], torch.where(on, 1, -1)[i], (slope * 1 + back * 2).to(torch.int8)[i]))
        self.w = torch.where(rate > 0, ((h_ev - H) * rate).clamp(min=0), self.w)
