import math
import numpy as np
import torch
import yaml


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
    C, k, nu, R = (v[0] for v in model.sensor(torch.tensor([ds], device=model.log_c.device)))
    return dict(pos_thres=C[0].item(), neg_thres=C[1].item(), sigma_event=k.item() ** -0.5,
                sigma_thres=0.03, shot_noise_rate_hz=nu.item(), refractory_period_s=R.item())


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
    # samples the sensor model the likelihood is trained on (losses.event_nll) with a sensor profile (default: the learned one):
    # thresholds X*C with X ~ IG(1, sigma_event^-2), exact crossing of the local quadratic of L, background events;
    # after any event the pixel is blind for the refractory time and takes ref = L at the end of it;
    # sigma_thres adds a fixed per-pixel threshold offset (sampler only)
    def __init__(self, model, profile=None, min_steps=4, max_steps=256, seed=0, max_events=64):
        self.m, self.p = model, profile or resolve(model)
        self.min_steps, self.max_steps, self.max_events = min_steps, max_steps, max_events
        self.gen = torch.Generator().manual_seed(seed)
        self.ref = None

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
        m, dt = self.m, t1 - t0
        p = self.p
        self.dev, self.k, self.nu, self.R = s["d"].device, max(p["sigma_event"], 1e-3) ** -2, p["shot_noise_rate_hz"], p["refractory_period_s"]
        tau = lambda v: torch.full((1,), v, device=self.dev)
        La, Da, _ = m.render(s, tau(0.0))
        if self.ref is None:
            shape = (2, *La.shape)
            spread = p["sigma_thres"] * torch.randn(shape, generator=self.gen).to(self.dev)
            self.c = (torch.tensor([p["pos_thres"], p["neg_thres"]], device=self.dev).view(2, 1, 1, 1, 1) + spread).clamp(min=0.01)
            self.ref = La.clone()
            self.x = (self._u(shape) / self._ig(shape)).float()  # stationary start: uniform part of a size-biased wait
            self.wake = torch.full_like(La, -math.inf, dtype=torch.float64)  # end of the blind time
            self.pending = torch.zeros_like(La, dtype=torch.bool)             # blind past the current step: ref not set yet
            self.next_noise = t0 - self._u(La.shape).log() / max(self.nu, 1e-12)

        speed = max(m.render(s, tau(v))[1].abs().flatten().quantile(0.999).item() for v in (0.0, 0.5, 1.0))
        S = int(min(max(math.ceil(2 * speed / self.c.min().item()), self.min_steps), self.max_steps))
        H, out = 1.0 / S, []
        for i in range(S):
            Lb, Db, _ = m.render(s, tau((i + 1) * H))
            self._cross(La, Da, (Lb - La - Da * H) / H ** 2, H, t0 + dt * i * H, dt, out)
            La, Da = Lb, Db
        t, x, y, p = (torch.cat(v).cpu().numpy() for v in zip(*out)) if out else ([np.zeros(0)] * 4)
        o = np.argsort(t, kind="stable")
        return dict(t=t[o], x=x[o].astype(np.int16), y=y[o].astype(np.int16), p=p[o].astype(np.int8))

    def _cross(self, La, Da, alpha, H, ta, dt, out):
        q = lambda h: La + Da * h + alpha * h * h
        hw = ((self.wake - ta) / dt).float().clamp(min=0)  # where in this step each pixel wakes up
        woke = self.pending & (hw <= H)
        self.ref = torch.where(woke, q(hw), self.ref)
        self.pending = self.pending & ~woke
        h_last, inf = torch.zeros_like(La), torch.full_like(La, math.inf)
        for _ in range(self.max_events):
            lo = torch.where(self.pending, inf, torch.maximum(h_last, hw))  # no signal while blind
            r_on = first_root(alpha, Da, La - self.ref - self.x[0] * self.c[0], lo, H)
            r_off = first_root(alpha, Da, La - self.ref + self.x[1] * self.c[1], lo, H)
            h_noise = ((self.next_noise - ta) / dt).float()
            r_noise = torch.where((h_noise > h_last) & (h_noise <= H), h_noise, torch.full_like(h_noise, math.inf))
            h = torch.minimum(torch.minimum(r_on, r_off), r_noise)
            fire = torch.isfinite(h)
            if not fire.any():
                break
            noise = fire & (r_noise < torch.minimum(r_on, r_off))
            coin = torch.where(self._u(La.shape) < 0.5, 1, -1)
            pol = torch.where(noise, coin, torch.where(r_on <= r_off, 1, -1))
            h_last = torch.where(fire, h, h_last)
            t = ta + dt * h_last.double()
            self.wake = torch.where(fire, t + self.R, self.wake)
            hw = torch.where(fire, ((self.wake - ta) / dt).float(), hw)
            self.pending = torch.where(fire, hw > H, self.pending)
            self.ref = torch.where(fire & ~self.pending, q(hw), self.ref)
            self.x = torch.where(fire, self._ig(self.x.shape), self.x)
            self.next_noise = torch.where(noise, self.next_noise - self._u(La.shape).log() / max(self.nu, 1e-12), self.next_noise)
            i = fire.nonzero(as_tuple=True)
            out.append((t[i], i[3], i[2], pol[i]))
