import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from third_party.dpt import DPTHead

# official V-JEPA 2.1 ViT-L checkpoint (the hub file in facebookresearch/vjepa2 points its downloads to localhost)
VJEPA21_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
LEVJEPA_ID = "galilai-group/LeVJEPA-VideoMix-Large"
VIT = {"s": dict(embed_dim=384, num_heads=6), "b": dict(embed_dim=768, num_heads=12)}  # depth 12, 16 px patches
DPT = {"s": (64, [48, 96, 192, 384]), "b": (128, [96, 192, 384, 768])}  # Depth-Anything-V2 head sizes for vits / vitb
TAKE = (2, 5, 8, 11)  # ViT blocks the DPT head reads (Depth-Anything-V2, vits / vitb)


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


def vit(size, cin):
    # timm's ViT (authors' code) trained from scratch, any image size (the position embedding is resampled)
    return VisionTransformer(img_size=256, patch_size=16, in_chans=cin, depth=12, num_classes=0, global_pool="",
                             class_token=False, dynamic_img_size=True, **VIT[size])


def run_vit(v, x):
    # patch tokens (B, h, w, D) -> output tokens (B, h*w, D) and the normed outputs of the TAKE blocks
    x, outs = v.norm_pre(v.patch_drop(v._pos_embed(x))), []
    for i, blk in enumerate(v.blocks):
        x = blk(x)
        if i in TAKE:
            outs.append(v.norm(x))
    return v.norm(x), outs


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
        qy, qx = (g[::2, ::2] + 0.5 for g in (gy, gx))  # one query per 32 px cell (centre of its 2x2 tokens)
        hq, wq = qy.shape
        q = self.query + self.embed(qy.expand(B, hq, wq), qx.expand(B, hq, wq),
                                    torch.full((B, hq, wq), 0.5, device=tok.device), dt.view(B, 1, 1).expand(B, hq, wq))
        out = self.dec(q.flatten(1, 2), mem.flatten(1, 3))
        return out.transpose(1, 2).reshape(B, -1, hq, wq)


class EventEncoder(nn.Module):
    # teacher, hybrid: exact-time event tokens (each event: place in its 16 px patch, time, polarity -> MLP; 4 queries pool the
    # events of a patch by attention) added to a patch embedding of the time-binned voxel and I0, I1 -> ViT -> 2x2 tokens
    # merged -> path code on the 32 px grid
    def __init__(self, cin, z_dim, size, ev_dim=128, queries=4, cap=400_000):
        super().__init__()
        self.vit, self.q, self.cap = vit(size, cin), queries, cap
        D = self.vit.embed_dim
        self.ev = nn.Sequential(nn.Linear(36, ev_dim), nn.GELU(), nn.Linear(ev_dim, ev_dim), nn.GELU())
        self.key, self.val = nn.Linear(ev_dim, queries), nn.Linear(ev_dim, queries * ev_dim)
        self.out = nn.Linear(queries * ev_dim, D)
        nn.init.zeros_(self.out.weight)  # starts as the plain voxel ViT
        nn.init.zeros_(self.out.bias)
        self.merge = nn.Linear(4 * D, z_dim)

    def pool(self, e, seg, n):
        # attention pooling per patch without padding: query j weights the patch's events by softmax over the patch of key_j
        a, v = self.key(e).float(), self.val(e).float().view(len(e), self.q, -1)
        m = torch.full((n, self.q), -1e30, device=e.device).scatter_reduce(0, seg[:, None].expand_as(a), a.detach(), "amax")
        w = (a - m[seg]).exp()
        s = torch.zeros(n, self.q, device=e.device).index_add(0, seg, w)
        out = torch.zeros(n, self.q, v.shape[-1], device=e.device).index_add(0, seg, w[..., None] * v)
        return (out / s.clamp(min=1e-12)[..., None]).flatten(1)

    def forward(self, x, ev, W):
        # x: (B, cin, H, W) padded to the 32 px grid; ev: batch index, pixel index in the W-wide crop, tau, polarity
        tok = self.vit.patch_embed(x)  # (B, h, w, D)
        B, h, w, D = tok.shape
        b, pix, tau, pol = ev
        if self.training and len(tau) > self.cap:  # memory stays bounded: a random subset of the events
            k = torch.randperm(len(tau), device=tau.device)[:self.cap]
            b, pix, tau, pol = b[k], pix[k], tau[k], pol[k]
        if len(tau):
            px, py = pix % W, pix // W
            u, v = (px % 16 + 0.5) / 16, (py % 16 + 0.5) / 16
            f = torch.cat([torch.stack([u, v, tau, pol.float()], -1), fourier(tau, 16), fourier(u, 8), fourier(v, 8)], -1)
            seg = (b * h + py // 16) * w + px // 16
            tok = tok + self.out(self.pool(self.ev(f), seg, B * h * w)).view(B, h, w, D)
        z, _ = run_vit(self.vit, tok)
        z = z.view(B, h // 2, 2, w // 2, 2, D).permute(0, 1, 3, 2, 4, 5).reshape(B, h // 2, w // 2, 4 * D)
        return self.merge(z).permute(0, 3, 1, 2)


class Decoder(nn.Module):
    # frames (12 ch) -> ViT -> DPT head (Depth-Anything-V2) at full resolution, plus a full-resolution conv branch for fine
    # edges; every output is gated by z, so z = 0 gives exactly the plain SloMo path and the decoder cannot ignore the code
    def __init__(self, z_dim, cout, width, size):
        super().__init__()
        self.vit = vit(size, 12)
        feat, chans = DPT[size]
        self.dpt = DPTHead(self.vit.embed_dim, feat, False, chans, out_dim=width)
        self.dpt.scratch.refinenet4.resConfUnit1 = None  # never used (the deepest fusion block has one input); DDP needs every param used
        self.skip = nn.Sequential(nn.Conv2d(12, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, width, 3, padding=1))
        self.gate = nn.Conv2d(z_dim, width, 1, bias=False)
        self.head = nn.Conv2d(width, cout, 1, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, x, z):
        tok = self.vit.patch_embed(x)
        _, h, w, _ = tok.shape
        _, outs = run_vit(self.vit, tok)
        f = self.dpt([(o,) for o in outs], h, w) + self.skip(x)
        g = self.gate(F.interpolate(z, size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.head(f * g)


class EvTween(nn.Module):
    def __init__(self, vit="s", width=64, K=3, z_dim=256, ev_dim=128, ev_cap=400_000, flow_prior="raft", flow_scale=8.0,
                 gamma=2.2, eps=0.01, c_init=0.25, r_init=1.0, backbone="vjepa2_1", pred_dim=384, pred_depth=4, bins=16,
                 amp=True, uncertainty=True, gammas=None):
        super().__init__()
        self.K, self.flow_scale, self.gamma, self.eps, self.amp = K, flow_scale, gamma, eps, amp
        self.mult = 32  # z lives on the 32 px grid
        self.flow = FlowPrior(flow_prior)
        self.teacher = EventEncoder(2 * bins + 6, z_dim, vit, ev_dim, cap=ev_cap)
        self.uncertainty = uncertainty  # one more output: how unsure the path is about arrival times (losses.event_nll)
        self.decoder = Decoder(z_dim, 5 * K + 1 + int(uncertainty), width, vit)
        self.student = Student(backbone, pred_dim, pred_depth)
        self.to_z = nn.Conv2d(pred_dim, z_dim, 1)
        self.to_counts = nn.Conv2d(pred_dim, 2 * bins, 1)
        # one sensor and one brightness mapping per dataset (cameras differ), picked by the dataset id of each sample
        n = len(gammas or [gamma])
        full = lambda v: nn.Parameter(torch.full((n,), math.log(v)))
        self.log_c, self.log_r = full(c_init), full(r_init)          # ON threshold, C_off / C_on
        self.log_k = full(4.0)                                        # inverse-Gaussian shape (timing regularity)
        self.log_nu = full(0.2)                                       # background events per pixel per second
        self.log_R = full(5e-5)                                       # refractory time (s); grows from below to the real floor
        self.R_max = 2e-3                                             # soft cap: with few events per pixel R is weakly pinned
        self.w_rgb = nn.Parameter(torch.tensor([[0.299, 0.587, 0.114]]).log().repeat(n, 1))  # RGB weights (softmax), learned
        self.register_buffer("gammas", torch.tensor(gammas or [gamma], dtype=torch.float32), persistent=False)  # from config

    @property
    def c(self):
        return self.log_c.exp()

    @property
    def r(self):
        return self.log_r.exp()

    @property
    def R(self):
        return self.R_max * torch.tanh(self.log_R.exp() / self.R_max)

    def bright(self, img, ds=None, cfa=None):
        # linear brightness the event pixels see: per dataset, learned RGB weights then PNG value^gamma;
        # under a Bayer filter (cfa: 2x2 channel ids of the crop, -1 = none) each pixel sees only its own channel
        B, _, H, W = img.shape
        ds = torch.zeros(B, dtype=torch.long, device=img.device) if ds is None else ds
        w = self.w_rgb[ds].softmax(-1)[:, :, None, None].expand(B, 3, H, W)
        if cfa is not None and bool((cfa >= 0).any()):
            tile = cfa.repeat(1, (H + 1) // 2, (W + 1) // 2)[:, :H, :W]
            w = torch.where((tile >= 0)[:, None], F.one_hot(tile.clamp(min=0), 3).permute(0, 3, 1, 2).to(img.dtype), w)
        return (img * w).sum(1, keepdim=True).clamp(1e-6, 1) ** self.gammas[ds].view(B, 1, 1, 1)

    def sensor(self, ds):
        # per sample: thresholds (C_on, C_off), IG shape k, background rate nu (1/s), refractory R (s)
        c = self.log_c.exp()[ds]
        return torch.stack([c, c * self.log_r.exp()[ds]], 1), self.log_k.exp()[ds], self.log_nu.exp()[ds], self.R[ds]

    def load_state_dict(self, sd, strict=False):
        # checkpoints from before per-dataset sensors hold one value per parameter: every dataset starts from it
        own = self.state_dict()
        sd = {k: v.expand_as(own[k]).clone() if k in own and v.dim() == 0 and own[k].dim() == 1 else v for k, v in sd.items()}
        return super().load_state_dict(sd, strict=strict)

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

    def prepare(self, i0, i1, ds=None, cfa=None):
        # frame-only inputs shared by teacher and decoder, padded to the patch grid; ds: dataset id per sample
        H, W = i0.shape[-2:]
        i0, i1 = self._pad(i0), self._pad(i1)
        ds = torch.zeros(len(i0), dtype=torch.long, device=i0.device) if ds is None else ds
        f01, f10 = self.flow(i0, i1)
        y0, y1 = self.bright(i0, ds, cfa), self.bright(i1, ds, cfa)
        e0, e1 = (warp(y1, f01) - y0).abs(), (warp(y0, f10) - y1).abs()
        x = torch.cat([i0, i1, f01 / 32, f10 / 32, e0, e1], 1)
        return dict(x=x, i0=i0, i1=i1, y0=y0, y1=y1, f01=f01, f10=f10, H=H, W=W, ds=ds, cfa=cfa)

    def encode(self, p, voxel, ev):
        # ev: (batch index, pixel index in the crop, tau, polarity) of the real events in the gap
        x = torch.cat([torch.log1p(self._pad(voxel)), p["i0"], p["i1"]], 1)
        with self._ac(x):
            return self.teacher(x, ev, p["W"]).float()

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
        s = dict(y0=crop(p["y0"]), y1=crop(p["y1"]), f01=crop(p["f01"]), f10=crop(p["f10"]), d=bound(d, 6.0), ds=p["ds"], cfa=p["cfa"],
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
    # configs from before the uncertainty output have no "uncertainty" key: build those checkpoints without it.
    # one gamma per entry of data.sets (position = dataset id); older configs: one dataset
    g = cfg["model"].get("gamma", 2.2)
    gammas = [(v or {}).get("gamma", g) for v in (cfg["data"].get("sets") or {}).values()] or [g]
    return EvTween(**{"uncertainty": False, **cfg["model"]}, bins=cfg["data"]["bins"], gammas=gammas)
