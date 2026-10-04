import math
import numpy as np
import torch


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


class Simulator:
    def __init__(self, model, mismatch=0.03, refractory=1e-4, noise_rate=0.1,
                 min_steps=4, max_steps=256, seed=0, max_events=64):
        self.m, self.mismatch, self.refractory, self.noise_rate = model, mismatch, refractory, noise_rate
        self.min_steps, self.max_steps, self.max_events = min_steps, max_steps, max_events
        self.gen = torch.Generator().manual_seed(seed)
        self.ref = None

    def _rand(self, like):
        return torch.randn(like.shape, generator=self.gen).to(like.device)

    @torch.no_grad()
    def run(self, s, t0, t1):
        # all events of one frame interval [t0, t1] in seconds; pixel state carries over between calls
        # per sub-step: L follows the quadratic through L_a, dL_a, L_b; crossings are solved exactly
        dev, c, dt = s["d"].device, self.m.c, t1 - t0
        tau = lambda v: torch.full((1,), v, device=dev)
        La, Da, _ = self.m.render(s, tau(0.0))
        if self.ref is None:
            self.c_on = (c * (1 + self.mismatch * self._rand(La))).clamp(min=0.01)
            self.c_off = (c * self.m.r * (1 + self.mismatch * self._rand(La))).clamp(min=0.01)
            self.ref = La + (torch.rand(La.shape, generator=self.gen).to(dev) - 0.5) * self.c_on
            self.last = torch.full_like(La, -math.inf, dtype=torch.float64)

        speed = max(self.m.render(s, tau(v))[1].abs().flatten().quantile(0.999).item() for v in (0.0, 0.5, 1.0))
        S = int(min(max(math.ceil(2 * speed / c.item()), self.min_steps), self.max_steps))
        H, out = 1.0 / S, []
        for i in range(S):
            Lb, Db, _ = self.m.render(s, tau((i + 1) * H))
            alpha = (Lb - La - Da * H) / H ** 2
            self._cross(La, Da, alpha, H, t0 + dt * i * H, dt, out)
            La, Da = Lb, Db
        out.append(self._noise(La.shape[-2:], t0, dt, dev))
        t, x, y, p = (torch.cat(v).cpu().numpy() for v in zip(*out))
        o = np.argsort(t, kind="stable")
        return dict(t=t[o], x=x[o].astype(np.int16), y=y[o].astype(np.int16), p=p[o].astype(np.int8))

    def _cross(self, La, Da, alpha, H, ta, dt, out):
        h_last = torch.zeros_like(La)
        for _ in range(self.max_events):
            r_on = first_root(alpha, Da, La - self.ref - self.c_on, h_last, H)
            r_off = first_root(alpha, Da, La - self.ref + self.c_off, h_last, H)
            h = torch.minimum(r_on, r_off)
            fire = torch.isfinite(h)
            if not fire.any():
                break
            pol = torch.where(r_on <= r_off, 1, -1)
            self.ref = torch.where(fire, self.ref + torch.where(pol > 0, self.c_on, -self.c_off), self.ref)
            h_last = torch.where(fire, h, h_last)
            t = ta + dt * h.double()
            keep = fire & (t - self.last >= self.refractory)
            self.last = torch.where(keep, t, self.last)
            i = keep.nonzero(as_tuple=True)
            out.append((t[i], i[3], i[2], pol[i]))

    def _noise(self, hw, t0, dt, dev):
        n = int(torch.poisson(torch.tensor(float(self.noise_rate * dt * hw[0] * hw[1])), generator=self.gen))
        r = lambda hi: torch.randint(hi, (n,), generator=self.gen).to(dev)
        t = t0 + dt * torch.rand(n, generator=self.gen, dtype=torch.float64).to(dev)
        return t, r(hw[1]), r(hw[0]), r(2) * 2 - 1
