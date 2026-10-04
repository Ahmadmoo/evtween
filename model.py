import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


def luminance(img, gamma=2.2):
    # RGB in [0,1] -> linear luminance (event pixels respond to linear light)
    y = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    return y.clamp(0, 1) ** gamma


def _coords(flow):
    _, _, H, W = flow.shape
    gy, gx = torch.meshgrid(torch.arange(H, device=flow.device, dtype=flow.dtype),
                            torch.arange(W, device=flow.device, dtype=flow.dtype), indexing="ij")
    return (gx + flow[:, 0]).clamp(0, W - 1), (gy + flow[:, 1]).clamp(0, H - 1)


def _sample(x, px, py):
    H, W = x.shape[-2:]
    grid = torch.stack([2 * px / (W - 1) - 1, 2 * py / (H - 1) - 1], -1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def warp(x, flow):
    # backward warp: out(p) = x(p + flow(p))
    return _sample(x, *_coords(flow))


def warp_with_grad(y, flow, vel):
    # y(p + flow) and its exact spatial gradient under bilinear interpolation:
    # d/dx = forward difference in x sampled at the left integer column (linear in y), same for d/dy;
    # on integer coordinates (kinks) take the one-sided difference in the direction of motion vel
    px, py = _coords(flow)
    corner = lambda p, v: torch.where(v >= 0, p.floor(), p.ceil() - 1).clamp(min=0)
    dx = F.pad(y[..., 1:] - y[..., :-1], (0, 1))
    dy = F.pad(y[..., 1:, :] - y[..., :-1, :], (0, 0, 0, 1))
    out = _sample(torch.cat([y, dx, dy]), torch.cat([px, corner(px, vel[:, 0]), px]),
                  torch.cat([py, py, corner(py, vel[:, 1])]))
    return out.chunk(3)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.norm = LayerNorm2d(c)
        self.mlp = nn.Sequential(nn.Conv2d(c, 4 * c, 1), nn.GELU(), nn.Conv2d(4 * c, c, 1))
        self.scale = nn.Parameter(torch.full((1, c, 1, 1), 1e-2))

    def forward(self, x):
        return x + self.scale * self.mlp(self.norm(self.dw(x)))


class UNet(nn.Module):
    def __init__(self, cin, cout, width, depth):
        super().__init__()
        ch = [width * 2 ** i for i in range(len(depth))]
        self.stem = nn.Conv2d(cin, ch[0], 3, padding=1)
        self.enc = nn.ModuleList(nn.Sequential(*[Block(c) for _ in range(n)]) for c, n in zip(ch, depth))
        self.down = nn.ModuleList(nn.Sequential(LayerNorm2d(a), nn.Conv2d(a, b, 2, stride=2)) for a, b in zip(ch, ch[1:]))
        self.up = nn.ModuleList(nn.Conv2d(b, a, 1) for a, b in zip(ch, ch[1:]))
        self.dec = nn.ModuleList(nn.Sequential(nn.Conv2d(2 * a, a, 1), Block(a)) for a in ch[:-1])
        self.head = nn.Sequential(LayerNorm2d(ch[0]), nn.Conv2d(ch[0], cout, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight)  # start from the plain flow-prior interpolation
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        x, skips = self.stem(x), []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.down):
                skips.append(x)
                x = self.down[i](x)
        for i in reversed(range(len(skips))):
            x = F.interpolate(self.up[i](x), scale_factor=2, mode="bilinear", align_corners=False)
            x = self.dec[i](torch.cat([x, skips[i]], 1))
        return self.head(x)


class FlowPrior(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.net = raft_large(weights=Raft_Large_Weights.DEFAULT) if kind == "raft" else None
        if self.net is not None:
            self.net.requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)  # frozen, always eval

    @torch.no_grad()
    def forward(self, i0, i1):
        if self.net is None:
            z = i0.new_zeros(i0.shape[0], 2, *i0.shape[-2:])
            return z, z
        H, W = i0.shape[-2:]
        k = max(1.0, 128 / min(H, W))  # RAFT needs >= 128 px per side
        size = (round(H * k / 8) * 8, round(W * k / 8) * 8)
        a, b = (F.interpolate(torch.cat(x), size, mode="bilinear", align_corners=False) * 2 - 1
                for x in ([i0, i1], [i1, i0]))
        f = self.net(a, b, num_flow_updates=12)[-1]
        f = F.interpolate(f, (H, W), mode="bilinear", align_corners=False)
        f = f * torch.tensor([W / size[1], H / size[0]], device=f.device).view(1, 2, 1, 1)
        return f.chunk(2)


class EvTween(nn.Module):
    def __init__(self, width=64, depth=(2, 2, 4, 2), K=3, flow_prior="raft", flow_scale=8.0,
                 gamma=2.2, eps=0.01, c_init=0.25, r_init=1.0, amp=True):
        super().__init__()
        self.K, self.flow_scale, self.gamma, self.eps, self.amp = K, flow_scale, gamma, eps, amp
        self.mult = max(8, 2 ** (len(depth) - 1))
        self.flow = FlowPrior(flow_prior)
        self.net = UNet(12, 5 * K + 1, width, depth)
        self.log_c = nn.Parameter(torch.tensor(math.log(c_init)))
        self.log_r = nn.Parameter(torch.tensor(math.log(r_init)))

    @property
    def c(self):
        return self.log_c.exp()

    @property
    def r(self):
        return self.log_r.exp()

    def forward(self, i0, i1):
        # predict the trajectory state of one frame pair (run once, render at any tau)
        B, _, H, W = i0.shape
        pad = (0, -W % self.mult, 0, -H % self.mult)
        i0, i1 = F.pad(i0, pad, mode="replicate"), F.pad(i1, pad, mode="replicate")
        f01, f10 = self.flow(i0, i1)
        y0, y1 = luminance(i0, self.gamma), luminance(i1, self.gamma)
        e0, e1 = (warp(y1, f01) - y0).abs(), (warp(y0, f10) - y1).abs()
        x = torch.cat([i0, i1, f01 / 32, f10 / 32, e0, e1], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            out = self.net(x)
        out, crop = out.float()[..., :H, :W], (lambda t: t[..., :H, :W])
        a, b, d = out.split([2 * self.K, 2 * self.K, self.K + 1], 1)
        return dict(y0=crop(y0), y1=crop(y1), f01=crop(f01), f10=crop(f10),
                    a=self.flow_scale * a.reshape(B, self.K, 2, H, W),
                    b=self.flow_scale * b.reshape(B, self.K, 2, H, W), d=d)

    def render(self, s, tau):
        # log intensity L(tau) and its exact derivative dL/dtau, tau: (B,) or (B,1,H,W)
        # flows:  F0 = SloMo(tau) + sum_k a_k tau^(k+1)        (zero at tau=0)
        #         F1 = SloMo(tau) + sum_k b_k (1-tau)^(k+1)    (zero at tau=1)
        # blend:  Y = (w0*Y0(p+F0) + w1*Y1(p+F1)) / (w0+w1),  w0=(1-tau)V, w1=tau(1-V)
        # dY/dtau by chain rule: d/dtau Y0(p+F0) = grad Y0(p+F0) . dF0/dtau
        tau = tau.view(-1, 1, 1, 1) if tau.dim() == 1 else tau
        f01, f10 = s["f01"], s["f10"]
        k = torch.arange(self.K, device=tau.device, dtype=tau.dtype).view(1, -1, 1, 1, 1)
        t, u = tau.unsqueeze(1), (1 - tau).unsqueeze(1)
        F0 = -tau * (1 - tau) * f01 + tau ** 2 * f10 + (s["a"] * t ** (k + 1)).sum(1)
        dF0 = -(1 - 2 * tau) * f01 + 2 * tau * f10 + (s["a"] * (k + 1) * t ** k).sum(1)
        F1 = (1 - tau) ** 2 * f01 - tau * (1 - tau) * f10 + (s["b"] * u ** (k + 1)).sum(1)
        dF1 = -2 * (1 - tau) * f01 - (1 - 2 * tau) * f10 - (s["b"] * (k + 1) * u ** k).sum(1)

        j = torch.arange(1, self.K + 1, device=tau.device, dtype=tau.dtype).view(1, -1, 1, 1)
        z = s["d"][:, :1] + (s["d"][:, 1:] * tau ** j).sum(1, keepdim=True)
        dz = (s["d"][:, 1:] * j * tau ** (j - 1)).sum(1, keepdim=True)
        V = torch.sigmoid(z)
        dV = V * (1 - V) * dz

        y0, gx0, gy0 = warp_with_grad(s["y0"], F0, dF0)
        y1, gx1, gy1 = warp_with_grad(s["y1"], F1, -dF1)
        dy0 = gx0 * dF0[:, :1] + gy0 * dF0[:, 1:]
        dy1 = gx1 * dF1[:, :1] + gy1 * dF1[:, 1:]
        w0, w1 = (1 - tau) * V, tau * (1 - V)
        dw0, dw1 = -V + (1 - tau) * dV, (1 - V) - tau * dV
        den = w0 + w1 + 1e-6
        Y = (w0 * y0 + w1 * y1) / den
        dY = (dw0 * y0 + w0 * dy0 + dw1 * y1 + w1 * dy1 - Y * (dw0 + dw1)) / den
        return torch.log(Y + self.eps), dY / (Y + self.eps), Y


def build_model(cfg):
    return EvTween(**cfg["model"])
