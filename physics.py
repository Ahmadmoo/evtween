import math
import numpy as np
import torch
import yaml
from model import taut, wiggle_k

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
    # sigma_thres adds a fixed per-pixel threshold offset (sampler only).
    # With the tube (s["tube"]) the pixel sees the least-event path in the frame-noise tube, linear between grid points.
    # With s["texture"] random texture events come on top, at rate a |dL/dtau| / (2 C_pol) (as in losses.event_nll);
    # then texture and background events leave the pixel's state alone (only real crossings reset it). With s["tex_q"]
    # (model.texture_pol) a share q of the texture events takes the polarity of the path's slope, 1 - q the other one.
    # Output "s": where each event comes from, 0 threshold crossing, 1 background, 2 texture.
    # With the detail image (model.detail) nothing changes here: decode() puts it on the keyframes and render() moves it.
    # With s["wiggle"] (texture as a hidden brightness wiggle of size A, as in losses.event_nll): each pixel's thresholds
    # get the wider spread model.wiggle_k (a curve's events move in time, same polarity), and the up-and-down events come
    # from a texture on the frame-0 image (model.paint: the learned painter, or white noise blurred over `grain` px; a new
    # one every gap), scaled per pixel so that one pixel of motion changes it by A on average, and moved by the path's own
    # flow (each pixel sees the point p + F0(tau)). A pixel fires each time the texture it sees crosses a level of a ladder with
    # rungs (C_on + C_off) / 2 apart, at its own random height: ON going up, OFF going down, kept with probability
    # exp(-|dL/dtau| / (A speed)) (none on steep curves). The expected count is the likelihood's (model.wiggle_rate per
    # polarity); a bright blob gives ON on its front and OFF on its back, and nearby pixels see the same blob, so texture
    # events form patches along the motion. They are texture events: they leave the pixel's state alone.
    # texture_events=False leaves out all texture events (v5's and the wiggle's): the path's own events only
    def __init__(self, model, profile=None, min_steps=4, max_steps=256, seed=0, max_events=64, grain=2.0, texture_events=True):
        self.m, self.p = model, profile or resolve(model)
        self.min_steps, self.max_steps, self.max_events, self.grain = min_steps, max_steps, max_events, grain
        self.texture_events = texture_events
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
        m, dt = self.m, t1 - t0
        p = self.p
        self.dev, self.k, self.nu, self.R = s["d"].device, max(p["sigma_event"], 1e-3) ** -2, p["shot_noise_rate_hz"], p["refractory_period_s"]
        if "wiggle" in s:  # per pixel: the wiggle spreads a curve's events in time
            self.k = wiggle_k(self.k, s["wiggle"].double(), p["pos_thres"] + p["neg_thres"])
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
            self.next_noise = t0 - self._u(La.shape).log() / max(self.nu, 1e-12)

        speed = max(m.render(s, tau(v))[1].abs().flatten().quantile(0.999).item() for v in (0.0, 0.5, 1.0))
        S = int(min(max(math.ceil(2 * speed / self.c.min().item()), self.min_steps), self.max_steps))
        g, A = None, s.get("wiggle") if self.texture_events else None
        if "tube" in s or A is not None:
            S = max(S, TUBE_STEPS)
            L = torch.stack([m.render(s, tau(i / S))[0] for i in range(S + 1)], 1)
            g = taut(L.flatten(2), m.tube_eps(s)).view_as(L) if "tube" in s else L
        if "tube" in s:  # each gap's path starts free inside the tube: the jump from the last gap's end fires nothing
            self.ref = g[:, 0].clone() if fresh else self.ref + g[:, 0] - self.last
            self.last = g[:, -1]
        if A is not None:
            Hh, Ww = A.shape[-2:]
            gy, gx = torch.meshgrid(torch.arange(Hh, device=self.dev, dtype=torch.float32),
                                    torch.arange(Ww, device=self.dev, dtype=torch.float32), indexing="ij")
            seen = [torch.stack([gx, gy])[None] + m.flows(s, tau(i / S).view(1, 1, 1, 1))[0] for i in range(S + 1)]
            tex_at = m.paint(s, seen, lambda shape: torch.randn(shape, generator=self.gen).to(self.dev), self.grain)
            phase = self._u(A.shape).float()
            level = [A * tex_at(q) / ((p["pos_thres"] + p["neg_thres"]) / 2) + phase for q in seen]  # in rungs
        H, out, a = 1.0 / S, [], s.get("texture") if self.texture_events else None
        for i in range(S):
            if g is not None:  # linear between grid points
                La, Lb = g[:, i], g[:, i + 1]
                Da, alpha = (Lb - La) / H, torch.zeros_like(La)
            else:
                Lb, Db, _ = m.render(s, tau((i + 1) * H))
                alpha = (Lb - La - Da * H) / H ** 2
            tex = None if a is None else a * (Lb - La).abs() / H / (2 * self.c)  # texture events per unit time, ON / OFF
            if a is not None and "tex_q" in s:  # a share q of them with the slope's polarity (q = 0.5: the line above)
                qv, rise, fall = s["tex_q"].view(-1, 1, 1, 1), torch.relu(Lb - La) / H, torch.relu(La - Lb) / H
                tex = a * torch.stack([qv * rise + (1 - qv) * fall, qv * fall + (1 - qv) * rise]) / self.c
            self._cross(La, Da, alpha, H, t0 + dt * i * H, dt, out, tex)
            if A is not None:
                speed = (seen[i + 1] - seen[i]).norm(dim=1, keepdim=True) / H  # px per gap
                keep = torch.exp(-(Lb - La).abs() / H / (A * speed + 1e-6))
                self._rungs(level[i], level[i + 1], keep, H, t0 + dt * i * H, dt, out)
            if g is None:
                La, Da = Lb, Db
        t, x, y, p, src = (torch.cat(v).cpu().numpy() for v in zip(*out)) if out else ([np.zeros(0)] * 5)
        o = np.argsort(t, kind="stable")
        return dict(t=t[o], x=x[o].astype(np.int16), y=y[o].astype(np.int16), p=p[o].astype(np.int8), s=src[o].astype(np.int8))

    def warm(self, s, t0, t1):
        # run a gap and drop its events (e.g. the gap before the one of interest): the pixels then start in a real state
        # (reference, thresholds, blind time) instead of the stationary guess, so the first events of the gap are no burst
        self.run(s, t0, t1)

    def _rungs(self, ua, ub, keep, H, ta, dt, out):
        # texture events in this step: the texture level a pixel sees (in rungs, with its random height) goes from ua to ub,
        # linear in between; each whole number crossed is an event (ON up, OFF down), kept with probability keep
        fa, fb, up = ua.floor(), ub.floor(), ub > ua
        n = (fb - fa).abs()
        for j in range(int(n.max().item())):
            hit = (n > j) & (self._u(ua.shape) < keep)
            t = ta + dt * H * ((torch.where(up, fa + 1 + j, fa - j) - ua) / (ub - ua)).double()
            i = hit.nonzero(as_tuple=True)
            out.append((t[i], i[3], i[2], torch.where(up, 1, -1)[i], torch.full((len(i[0]),), 2, dtype=torch.int8, device=t.device)))

    def _cross(self, La, Da, alpha, H, ta, dt, out, tex=None):
        q = lambda h: La + Da * h + alpha * h * h
        hw = ((self.wake - ta) / dt).float().clamp(min=0)  # where in this step each pixel wakes up
        woke = self.pending & (hw <= H)
        self.ref = torch.where(woke, q(hw), self.ref)
        self.pending = self.pending & ~woke
        h_last, inf = torch.zeros_like(La), torch.full_like(La, math.inf)
        if tex is not None:  # texture: a Poisson process in this step (ON + OFF), its next time drawn ahead (memoryless)
            lam = tex[0] + tex[1]
            p_on = torch.where(lam > 0, tex[0] / lam, 0.5)
            next_tex = lambda h: h - self._u(La.shape).log().float() / lam
            h_tex = next_tex(h_last)
        for _ in range(self.max_events):
            lo = torch.where(self.pending, inf, torch.maximum(h_last, hw))  # no signal while blind
            r_on = first_root(alpha, Da, La - self.ref - self.x[0] * self.c[0], lo, H)
            r_off = first_root(alpha, Da, La - self.ref + self.x[1] * self.c[1], lo, H)
            h_noise = ((self.next_noise - ta) / dt).float()
            r_noise = torch.where((h_noise > h_last) & (h_noise <= H), h_noise, torch.full_like(h_noise, math.inf))
            r_tex = inf if tex is None else torch.where(h_tex <= H, h_tex, inf)
            r_sig = torch.minimum(r_on, r_off)
            h = torch.minimum(torch.minimum(r_sig, r_noise), r_tex)
            fire = torch.isfinite(h)
            if not fire.any():
                break
            noise = fire & (r_noise < r_sig) & (r_noise <= r_tex)
            textured = fire & (r_tex < r_sig) & (r_tex < r_noise)
            coin = torch.where(self._u(La.shape) < 0.5, 1, -1)
            pol = torch.where(noise, coin, torch.where(r_on <= r_off, 1, -1))
            if tex is not None:
                pol = torch.where(textured, torch.where(self._u(La.shape) < p_on, 1, -1), pol)
            h_last = torch.where(fire, h, h_last)
            t = ta + dt * h_last.double()
            # with texture only real crossings reset the pixel: an up-and-back swing leaves its reference where it was
            reset = fire if tex is None else fire & ~noise & ~textured
            self.wake = torch.where(reset, t + self.R, self.wake)
            hw = torch.where(reset, ((self.wake - ta) / dt).float(), hw)
            self.pending = torch.where(reset, hw > H, self.pending)
            self.ref = torch.where(reset & ~self.pending, q(hw), self.ref)
            self.x = torch.where(reset, self._ig(self.x.shape), self.x)
            self.next_noise = torch.where(noise, self.next_noise - self._u(La.shape).log() / max(self.nu, 1e-12), self.next_noise)
            if tex is not None:
                h_tex = torch.where(textured, next_tex(h_last), h_tex)
            i = fire.nonzero(as_tuple=True)
            out.append((t[i], i[3], i[2], pol[i], (noise.to(torch.int8) + 2 * textured.to(torch.int8))[i]))
