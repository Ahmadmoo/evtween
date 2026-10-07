import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

# official V-JEPA 2.1 ViT-L checkpoint (the hub file in facebookresearch/vjepa2 points its downloads to localhost)
VJEPA21_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
LEVJEPA_ID = "galilai-group/LeVJEPA-VideoMix-Large"


def luminance(img, gamma=2.2):
    # RGB in [0,1] -> linear luminance (event pixels respond to linear light)
    y = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    return y.clamp(0, 1) ** gamma


def _coords(flow):
    _, _, H, W = flow.shape
    gy, gx = torch.meshgrid(torch.arange(H, device=flow.device, dtype=flow.dtype),
                            torch.arange(W, device=flow.device, dtype=flow.dtype), indexing="ij")
    return gx + flow[:, 0], gy + flow[:, 1]


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
    # on integer coordinates (kinks) take the one-sided difference in the direction of motion vel;
    # outside the image (or on the border moving out) the border value is constant, so the gradient is 0
    H, W = y.shape[-2:]
    rx, ry = _coords(flow)
    px, py = rx.clamp(0, W - 1), ry.clamp(0, H - 1)
    corner = lambda p, v: torch.where(v >= 0, p.floor(), p.ceil() - 1).clamp(min=0)
    inside = lambda p, v, n: (((p > 0) | ((p == 0) & (v >= 0))) & ((p < n - 1) | ((p == n - 1) & (v < 0))))[:, None]
    dx = F.pad(y[..., 1:] - y[..., :-1], (0, 1))
    dy = F.pad(y[..., 1:, :] - y[..., :-1, :], (0, 0, 0, 1))
    v, gx, gy = _sample(torch.cat([y, dx, dy]), torch.cat([px, corner(px, vel[:, 0]), px]),
                        torch.cat([py, py, corner(py, vel[:, 1])])).chunk(3)
    return v, gx * inside(rx, vel[:, 0], W), gy * inside(ry, vel[:, 1], H)


def fourier(v, n=16):
    f = torch.exp(torch.linspace(0, math.log(100), n // 2, device=v.device))
    a = v.float()[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


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


class Backbone(nn.Module):
    # frozen pretrained video encoder, built and loaded with the authors' own code;
    # "none" is a trainable per-frame patch embedding (baseline without world knowledge)
    def __init__(self, kind, dim):
        super().__init__()
        self.kind = kind
        if kind == "vjepa2_1":
            self.net, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_large_384", pretrained=False)
            sd = torch.hub.load_state_dict_from_url(VJEPA21_URL, map_location="cpu")["ema_encoder"]
            self.net.load_state_dict({k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()})
            self.tubelet, self.dim = 2, self.net.embed_dim
        elif kind == "levjepa":
            from transformers import AutoModel
            self.net = AutoModel.from_pretrained(LEVJEPA_ID, trust_remote_code=True)
            self.tubelet, self.dim = self.net.config.tubelet_size, self.net.config.embed_dim
        else:
            self.net, self.tubelet, self.dim = nn.Conv2d(3, dim, 16, stride=16), 1, dim
        if kind != "none":
            self.net.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        if self.kind != "none":
            self.net.eval()
        return self

    def forward(self, clip):
        # (B,T,3,H,W) in [0,1] -> tokens (B, T/tubelet, H/16, W/16, D)
        B, T, _, H, W = clip.shape
        if self.kind == "none":
            return self.net(clip.flatten(0, 1)).view(B, T, self.dim, H // 16, W // 16).permute(0, 1, 3, 4, 2)
        x = (clip.transpose(1, 2) - self.mean) / self.std
        with torch.no_grad():
            tok = self.net(x) if self.kind == "vjepa2_1" else self.net(pixel_values=x).last_hidden_state[:, 1:]
        return tok.reshape(B, T // self.tubelet, H // 16, W // 16, self.dim)


class Student(nn.Module):
    # frozen video backbone on the context frames + small transformer that fills the gap
    def __init__(self, backbone, dim, depth):
        super().__init__()
        self.backbone = Backbone(backbone, dim)
        self.inp = nn.Linear(self.backbone.dim, dim)
        self.pos = nn.Linear(64, dim)
        self.query = nn.Parameter(torch.zeros(dim))
        layer = nn.TransformerDecoderLayer(dim, 8, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, depth)

    def embed(self, gy, gx, t, dt):
        return self.pos(torch.cat([fourier(gy / 16), fourier(gx / 16), fourier(t), fourier(dt.log())], -1))

    def forward(self, ctx, ctx_tau, dt):
        # ctx: frames up to I0 then from I1 on (gap frames never included); ctx_tau: their times in gap units
        B, T = ctx.shape[:2]
        tok, tt = [], []
        for side in (slice(0, T // 2), slice(T // 2, T)):  # encode each side alone so the backbone never sees a fake jump
            tok.append(self.backbone(ctx[:, side]))
            tt.append(ctx_tau[:, side].reshape(B, -1, self.backbone.tubelet).mean(-1))
        tok, tt = torch.cat(tok, 1), torch.cat(tt, 1)
        _, n, h, w, _ = tok.shape
        gy, gx = torch.meshgrid(torch.arange(h, device=tok.device), torch.arange(w, device=tok.device), indexing="ij")
        mem = self.inp(tok) + self.embed(gy.expand(B, n, h, w), gx.expand(B, n, h, w),
                                         tt.view(B, n, 1, 1).expand(B, n, h, w), dt.view(B, 1, 1, 1).expand(B, n, h, w))
        q = self.query + self.embed(gy.expand(B, h, w), gx.expand(B, h, w),
                                    torch.full((B, h, w), 0.5, device=tok.device), dt.view(B, 1, 1).expand(B, h, w))
        out = self.dec(q.flatten(1, 2), mem.flatten(1, 3))
        return out.transpose(1, 2).reshape(B, -1, h, w)


class EventEncoder(nn.Module):
    # teacher: real events in the gap (time-binned) + I0, I1 -> path code on the 16x16 patch grid
    def __init__(self, cin, z_dim, width):
        super().__init__()
        ch = [width, 2 * width, 4 * width, 4 * width, 4 * width]
        layers = [nn.Conv2d(cin, ch[0], 3, padding=1)]
        for a, b in zip(ch, ch[1:]):
            layers += [Block(a), LayerNorm2d(a), nn.Conv2d(a, b, 2, stride=2)]
        self.net = nn.Sequential(*layers, Block(ch[-1]), Block(ch[-1]), LayerNorm2d(ch[-1]), nn.Conv2d(ch[-1], z_dim, 1))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    # frames + path code -> per-pixel path coefficients; every coefficient is gated by z,
    # so z = 0 gives exactly the plain SloMo path and the decoder cannot ignore the code
    def __init__(self, z_dim, cout, width, depth):
        super().__init__()
        self.feat = UNet(12, width, width, depth)
        self.gate = nn.Conv2d(z_dim, width, 1, bias=False)
        self.head = nn.Conv2d(width, cout, 1, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, x, z):
        g = self.gate(F.interpolate(z, size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.head(self.feat(x) * g)


class EvTween(nn.Module):
    def __init__(self, width=64, depth=(2, 2, 4, 2), K=3, z_dim=32, flow_prior="raft", flow_scale=8.0, gamma=2.2,
                 eps=0.01, c_init=0.25, r_init=1.0, backbone="vjepa2_1", pred_dim=384, pred_depth=4, bins=16, amp=True,
                 uncertainty=True):
        super().__init__()
        self.K, self.flow_scale, self.gamma, self.eps, self.amp = K, flow_scale, gamma, eps, amp
        self.mult = max(16, 2 ** (len(depth) - 1))
        self.flow = FlowPrior(flow_prior)
        self.teacher = EventEncoder(2 * bins + 6, z_dim, width)
        self.uncertainty = uncertainty  # one more output: how unsure the path is about arrival times (losses.event_nll)
        self.decoder = Decoder(z_dim, 5 * K + 1 + int(uncertainty), width, depth)
        self.student = Student(backbone, pred_dim, pred_depth)
        self.to_z = nn.Conv2d(pred_dim, z_dim, 1)
        self.to_counts = nn.Conv2d(pred_dim, 2 * bins, 1)
        self.log_c = nn.Parameter(torch.tensor(math.log(c_init)))   # ON threshold
        self.log_r = nn.Parameter(torch.tensor(math.log(r_init)))   # C_off / C_on
        self.log_k = nn.Parameter(torch.tensor(math.log(4.0)))      # inverse-Gaussian shape (timing regularity)
        self.log_nu = nn.Parameter(torch.tensor(math.log(0.2)))     # background events per pixel per second
        self.log_R = nn.Parameter(torch.tensor(math.log(5e-5)))     # refractory time (s); grows from below to the real floor
        self.R_max = 2e-3                                            # soft cap: with few events per pixel R is weakly pinned

    @property
    def c(self):
        return self.log_c.exp()

    @property
    def r(self):
        return self.log_r.exp()

    @property
    def R(self):
        return self.R_max * torch.tanh(self.log_R.exp() / self.R_max)

    def frozen(self, name):
        return name.startswith("flow.net.") or (name.startswith("student.backbone.net.") and self.student.backbone.kind != "none")

    def state(self):
        # checkpoint without the frozen pretrained networks (reloaded from their official sources)
        return {k: v for k, v in self.state_dict().items() if not self.frozen(k)}

    def _ac(self, x):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda)

    def _pad(self, x):
        H, W = x.shape[-2:]
        y = F.pad(x.reshape(-1, *x.shape[-3:]), (0, -W % self.mult, 0, -H % self.mult), mode="replicate")
        return y.view(*x.shape[:-2], *y.shape[-2:])

    def prepare(self, i0, i1):
        # frame-only inputs shared by teacher and decoder, padded to the patch grid
        H, W = i0.shape[-2:]
        i0, i1 = self._pad(i0), self._pad(i1)
        f01, f10 = self.flow(i0, i1)
        y0, y1 = luminance(i0, self.gamma), luminance(i1, self.gamma)
        e0, e1 = (warp(y1, f01) - y0).abs(), (warp(y0, f10) - y1).abs()
        x = torch.cat([i0, i1, f01 / 32, f10 / 32, e0, e1], 1)
        return dict(x=x, i0=i0, i1=i1, y0=y0, y1=y1, f01=f01, f10=f10, H=H, W=W)

    def encode(self, p, voxel):
        x = torch.cat([torch.log1p(self._pad(voxel)), p["i0"], p["i1"]], 1)
        with self._ac(x):
            return self.teacher(x).float()

    def predict(self, ctx, ctx_tau, dt, head="z"):
        ctx = self._pad(ctx)
        with self._ac(ctx):
            f = self.student(ctx, ctx_tau, dt).float()
        return (self.to_z if head == "z" else self.to_counts)(f)

    def decode(self, p, z):
        # path code -> trajectory state used by render()
        with self._ac(z):
            out = self.decoder(p["x"], z)
        H, W, B = p["H"], p["W"], z.shape[0]
        crop = lambda t: t[..., :H, :W]
        a, b, d, u = crop(out.float()).split([2 * self.K, 2 * self.K, self.K + 1, int(self.uncertainty)], 1)
        bound = lambda x, m: m * torch.tanh(x / m)  # corrections stay within +-flow_scale px per term, visibility logits within +-6
        s = dict(y0=crop(p["y0"]), y1=crop(p["y1"]), f01=crop(p["f01"]), f10=crop(p["f10"]), d=bound(d, 6.0),
                 a=bound(a, self.flow_scale).reshape(B, self.K, 2, H, W), b=bound(b, self.flow_scale).reshape(B, self.K, 2, H, W))
        if self.uncertainty:
            s["sig"] = 0.01 * bound(u, 3.0).exp()  # arrival-time uncertainty (gap units), 0.01 at z = 0, range 0.0005 - 0.2
        return s

    def flows(self, s, tau):
        # backward flows to frame 0 / frame 1 at time tau and their tau-derivatives:
        # F0 = SloMo(tau) + sum_k a_k tau^(k+1)  (zero at tau=0),  F1 = SloMo(tau) + sum_k b_k (1-tau)^(k+1)  (zero at tau=1)
        f01, f10 = s["f01"], s["f10"]
        k = torch.arange(self.K, device=tau.device, dtype=tau.dtype).view(1, -1, 1, 1, 1)
        t, u = tau.unsqueeze(1), (1 - tau).unsqueeze(1)
        F0 = -tau * (1 - tau) * f01 + tau ** 2 * f10 + (s["a"] * t ** (k + 1)).sum(1)
        dF0 = -(1 - 2 * tau) * f01 + 2 * tau * f10 + (s["a"] * (k + 1) * t ** k).sum(1)
        F1 = (1 - tau) ** 2 * f01 - tau * (1 - tau) * f10 + (s["b"] * u ** (k + 1)).sum(1)
        dF1 = -2 * (1 - tau) * f01 - (1 - 2 * tau) * f10 - (s["b"] * (k + 1) * u ** k).sum(1)
        return F0, dF0, F1, dF1

    def render(self, s, tau):
        # log intensity L(tau) and its analytic derivative dL/dtau, tau: (B,) or (B,1,H,W)
        # Y = (w0*Y0(p+F0) + w1*Y1(p+F1)) / (w0+w1),  w0=(1-tau)V, w1=tau(1-V),  V = sigmoid(poly(tau))
        tau = tau.view(-1, 1, 1, 1) if tau.dim() == 1 else tau
        F0, dF0, F1, dF1 = self.flows(s, tau)
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
    return EvTween(**cfg["model"], bins=cfg["data"]["bins"])
