"""Gaze-LLE 主干 + 点空间输出头（B3 MDN / B4 MCL / 我方法 flow matching），共用同一个冻结 backbone 与同一套场景 token。

主干 = 官方 GazeLLE 去掉热图头（1x1 linear + 2D 位置编码 + 头框 token + 3 层 Block，参数与官方逐项相同）；
头 = 交叉注意力解码器（query -> 场景 token），三种头的解码器深度/宽度相同，只有 query 与输出层不同：
  MDN  : 1 个可学习 query -> M 个二维高斯分量（π, μ, σx, σy, ρ），NLL 训练
  MCL  : K 个可学习 query -> K 个假设点 + 得分 logit；relaxed WTA 训练 + 赢家分类
  Flow : 速度场 v(x_t, t | token)，x_t 经 Fourier 特征 + 时间 FiLM；条件 flow matching（x0~N(0,I) -> x1 = 标注点）
所有头统一接口：loss(tokens, y)  /  sample(tokens, n, ...) -> [B, n, 2]（坐标在 [0,1]^2，x 在前 y 在后）。
"""
from pathlib import Path
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_cached')
from common import LocalDinoV2Backbone  # noqa: E402,F401  (顺带把官方 gazelle 加进 sys.path)
import gazelle.utils as gutils  # noqa: E402
from gazelle.model import GazeLLE  # noqa: E402

GRID = 64


def cell_centers(device=None):
    """[4096,2]，下标 = iy*64+ix，列为 (x, y)；与 setmetrics/metrics.py 的 CELLS 一致（test_gaze_models.py 校验）。"""
    xs = (torch.arange(GRID, device=device) + 0.5) / GRID
    yy, xx = torch.meshgrid(xs, xs, indexing='ij')
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)


# ------------------------------------------------------------------ 主干
class Trunk(GazeLLE):
    """官方 GazeLLE 去掉热图头：tokens() 返回 [sum(N_p), 1024, dim] 的场景 token。"""

    def __init__(self, backbone, **kw):
        super().__init__(backbone, inout=False, **kw)
        del self.heatmap_head

    def tokens(self, images, bboxes):
        num_ppl = [len(b) for b in bboxes]
        x = self.backbone.forward(images)
        x = self.linear(x) + self.pos_embed
        x = gutils.repeat_tensors(x, num_ppl)
        head_maps = torch.cat(self.get_input_head_maps(bboxes), dim=0).to(x.device)
        x = x + head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x.flatten(2).permute(0, 2, 1)
        return self.transformer(x)


class HeatmapGaze(GazeLLE):
    """B1/B2：官方热图模型，加一个 sample()（按 S1 口径 raw γ=1 的 pmf 抽样，格内均匀抖动）。"""

    @torch.no_grad()
    def sample(self, images, bboxes, n, gen=None, lam=1e-3):
        h = torch.cat(self.forward({'images': images, 'bboxes': bboxes})['heatmap'], 0).flatten(1).double().clamp_min(1e-12)
        P = (1 - lam) * h / h.sum(1, keepdim=True) + lam / h.shape[1]
        cells = torch.multinomial(P.float(), n, replacement=True, generator=gen)
        jit = torch.rand(cells.shape + (2,), device=cells.device, generator=gen)
        return torch.stack([(cells % GRID + jit[..., 0]) / GRID, (cells // GRID + jit[..., 1]) / GRID], -1)


# ------------------------------------------------------------------ 交叉注意力解码器
class XBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4, film=False):
        super().__init__()
        self.h = heads
        self.nq, self.nkv, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv, self.o = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))
        self.film = None
        if film:  # AdaLN-lite：(scale1, shift1, scale2, shift2)，零初始化 -> 初始为恒等调制
            self.film = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim))
            nn.init.zeros_(self.film[1].weight)
            nn.init.zeros_(self.film[1].bias)

    def make_kv(self, tokens):
        B, N, D = tokens.shape
        k, v = self.kv(self.nkv(tokens)).view(B, N, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        return k, v

    def forward(self, x, k, v, c=None):
        B, Q, D = x.shape
        h = self.nq(x)
        if self.film is not None:
            s1, b1, s2, b2 = self.film(c).chunk(4, -1)
            h = h * (1 + s1) + b1
        q = self.q(h).view(B, Q, self.h, D // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Q, D)
        x = x + self.o(a)
        h = self.n2(x)
        if self.film is not None:
            h = h * (1 + s2) + b2
        return x + self.mlp(h)


class XDecoder(nn.Module):
    def __init__(self, dim=256, depth=2, heads=8, film=False):
        super().__init__()
        self.blocks = nn.ModuleList(XBlock(dim, heads, film=film) for _ in range(depth))

    def make_kv(self, tokens):
        return [b.make_kv(tokens) for b in self.blocks]

    def forward(self, q, kvs, c=None):
        for b, (k, v) in zip(self.blocks, kvs):
            q = b(q, k, v, c)
        return q


def _mlp_out(dim, n_out):
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_out))


# ------------------------------------------------------------------ B3 MDN
class MDNHead(nn.Module):
    """M 个二维高斯（σ 取 exp 且夹在 [e^-6, 1]，ρ = 0.95 tanh）。loss = 未截断混合密度的 NLL（标准 MDN）；
    pmf/抽样用截断到 [0,1]^2 的版本（pmf 在 64x64 格上归一化，抽样用拒绝法，二者一致）。"""

    def __init__(self, dim=256, M=8, depth=3):
        super().__init__()
        self.M = M
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.dec = XDecoder(dim, depth)
        self.out = _mlp_out(dim, M * 6)

    def params(self, tokens):
        B = tokens.shape[0]
        z = self.dec(self.query.expand(B, -1, -1), self.dec.make_kv(tokens))[:, 0]
        o = self.out(z).float().view(B, self.M, 6)
        return dict(logpi=F.log_softmax(o[..., 0], -1), mu=torch.sigmoid(o[..., 1:3]), sig=torch.exp(o[..., 3:5].clamp(-6, 0)),
                    rho=0.95 * torch.tanh(o[..., 5]))

    @staticmethod
    def comp_logprob(p, pts):
        """pts [B,P,2] -> [B,P,M] 各分量的对数密度。"""
        d = (pts[:, :, None, :] - p['mu'][:, None]) / p['sig'][:, None]  # [B,P,M,2]
        rho = p['rho'][:, None]
        zq = (d[..., 0] ** 2 - 2 * rho * d[..., 0] * d[..., 1] + d[..., 1] ** 2) / (1 - rho ** 2)
        return -math.log(2 * math.pi) - p['sig'][:, None].log().sum(-1) - 0.5 * (1 - rho ** 2).log() - 0.5 * zq

    def loss(self, tokens, y):
        p = self.params(tokens)
        lp = torch.logsumexp(p['logpi'][:, None] + self.comp_logprob(p, y[:, None]), -1)[:, 0]
        return -lp.mean(), {'nll_density': -lp.mean().item()}

    @torch.no_grad()
    def pmf(self, tokens):
        p = self.params(tokens)
        pts = cell_centers(tokens.device)[None].expand(tokens.shape[0], -1, -1)
        lp = torch.logsumexp(p['logpi'][:, None] + self.comp_logprob(p, pts), -1)
        return F.softmax(lp, -1)

    @torch.no_grad()
    def sample(self, tokens, n, gen=None, max_tries=30):
        p = self.params(tokens)
        B = tokens.shape[0]
        out = torch.empty(B, n, 2, device=tokens.device)
        todo = torch.ones(B, n, dtype=torch.bool, device=tokens.device)
        for _ in range(max_tries):
            comp = torch.multinomial(p['logpi'].exp(), n, replacement=True, generator=gen)
            g = lambda t: t.gather(1, comp[..., None].expand(-1, -1, 2)) if t.dim() == 3 else t.gather(1, comp)
            mu, sig, rho = g(p['mu']), g(p['sig']), g(p['rho'])
            z = torch.randn(B, n, 2, device=tokens.device, generator=gen)
            x = mu[..., 0] + sig[..., 0] * z[..., 0]
            y = mu[..., 1] + sig[..., 1] * (rho * z[..., 0] + (1 - rho ** 2).sqrt() * z[..., 1])
            pt = torch.stack([x, y], -1)
            ok = ((pt >= 0) & (pt <= 1)).all(-1) & todo
            out[ok] = pt[ok]
            todo &= ~ok
            if not todo.any():
                break
        out[todo] = torch.stack([x, y], -1)[todo].clamp(0, 1)  # 残余（极少）夹到边界
        return out


# ------------------------------------------------------------------ B4 MCL / WTA
class MCLHead(nn.Module):
    """K 个假设；relaxed WTA（赢家 1-ε，其余 ε/(K-1)），距离 = 欧氏距离；得分头用赢家下标做交叉熵（权重 score_w）。"""

    def __init__(self, dim=256, K=5, depth=3, eps=0.05, score_w=0.1):
        super().__init__()
        self.K, self.eps, self.score_w = K, eps, score_w
        self.queries = nn.Parameter(torch.randn(1, K, dim) * 0.02)
        self.dec = XDecoder(dim, depth)
        self.out = _mlp_out(dim, 3)

    def forward_hyps(self, tokens):
        B = tokens.shape[0]
        o = self.out(self.dec(self.queries.expand(B, -1, -1), self.dec.make_kv(tokens))).float()
        return torch.sigmoid(o[..., :2]), o[..., 2]

    def loss(self, tokens, y):
        hyp, logit = self.forward_hyps(tokens)
        d = ((hyp - y[:, None]) ** 2).sum(-1).add(1e-12).sqrt()  # [B,K]
        best = d.argmin(1)
        db = d.gather(1, best[:, None])[:, 0]
        wta = ((1 - self.eps) * db + self.eps / (self.K - 1) * (d.sum(1) - db)).mean()
        ce = F.cross_entropy(logit, best)
        return wta + self.score_w * ce, {'wta': wta.item(), 'ce': ce.item(), 'best_l2': db.mean().item(),
                                         'used_k': torch.bincount(best, minlength=self.K).float().div(len(best)).max().item()}

    @torch.no_grad()
    def predict(self, tokens):
        """原生输出：假设 [B,K,2] 与权重 [B,K]（按权重降序）。"""
        hyp, logit = self.forward_hyps(tokens)
        w, idx = F.softmax(logit, -1).sort(-1, descending=True)
        return hyp.gather(1, idx[..., None].expand(-1, -1, 2)), w

    @torch.no_grad()
    def sample(self, tokens, n, gen=None):
        hyp, w = self.predict(tokens)
        idx = torch.multinomial(w, n, replacement=True, generator=gen)
        return hyp.gather(1, idx[..., None].expand(-1, -1, 2))  # 离散点；平滑交给评测管线的 σ_s


# ------------------------------------------------------------------ 我方法：条件 flow matching
class FlowHead(nn.Module):
    """x ∈ [0,1]^2 先映到 [-1,1]^2；x_t = (1-t) x0 + t x1，x0~N(0,I)；目标速度 x1-x0。
    速度网络：query = Linear(Fourier(x_t)) ，每层交叉注意力到场景 token，时间通过 FiLM 注入。
    采样默认 Heun + γ=2 时间网格 + 16 步（合成双峰上 TV 0.017，到采样噪声底；均匀 Euler 16 步为 0.15，窄目标在 t→1 处欠分辨）。"""

    def __init__(self, dim=256, depth=3, n_freq=8, n_draw=16, steps=16, gamma=2.0, solver='heun'):
        super().__init__()
        self.dim, self.n_freq, self.n_draw, self.steps, self.gamma, self.solver = dim, n_freq, n_draw, steps, gamma, solver
        self.inp = nn.Linear(2 + 4 * n_freq, dim)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.dec = XDecoder(dim, depth, film=True)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def _fourier(self, x):
        f = (2.0 ** torch.arange(self.n_freq, device=x.device)) * math.pi
        a = x[..., None] * f  # [...,2,F]
        return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)

    def _temb(self, t):
        half = self.dim // 2
        fr = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        a = (t[..., None] * 1000.0) * fr
        return self.t_mlp(torch.cat([a.sin(), a.cos()], -1))

    def velocity(self, x, t, kvs):
        """x [B,Q,2]，t [B,Q] 或 [B,1]，kvs = dec.make_kv(tokens)。"""
        c = self._temb(t)
        return self.out(self.dec(self.inp(self._fourier(x)) + c, kvs, c)).float()

    def loss(self, tokens, y):
        B = tokens.shape[0]
        x1 = (y * 2 - 1)[:, None].expand(-1, self.n_draw, -1)
        x0 = torch.randn_like(x1)
        t = torch.rand(B, self.n_draw, device=tokens.device)
        xt = (1 - t[..., None]) * x0 + t[..., None] * x1
        v = self.velocity(xt, t, self.dec.make_kv(tokens))
        l = ((v - (x1 - x0)) ** 2).mean()
        return l, {'fm_mse': l.item()}

    @torch.no_grad()
    def sample(self, tokens, n, gen=None, steps=None, chunk=2500, gamma=None, solver=None):
        """时间网格 t_i = 1-(1-i/steps)^gamma（gamma>1 在 t→1 处加密，窄目标需要）；solver = euler | heun（heun 每步 2 次函数求值）。"""
        steps, gamma, solver = steps or self.steps, gamma or self.gamma, solver or self.solver
        kvs = self.dec.make_kv(tokens)
        B = tokens.shape[0]
        ts = (1 - (1 - torch.linspace(0, 1, steps + 1, device=tokens.device)) ** gamma).tolist()
        outs = []
        for s in range(0, n, chunk):
            m = min(chunk, n - s)
            x = torch.randn(B, m, 2, device=tokens.device, generator=gen)
            for i in range(steps):
                t0, t1 = ts[i], ts[i + 1]
                v = self.velocity(x, torch.full((B, 1), t0, device=tokens.device), kvs)
                if solver == 'euler':
                    x = x + (t1 - t0) * v
                else:  # heun（梯形）
                    xe = x + (t1 - t0) * v
                    v2 = self.velocity(xe, torch.full((B, 1), t1, device=tokens.device), kvs)
                    x = x + (t1 - t0) * 0.5 * (v + v2)
            outs.append(((x + 1) / 2).clamp(0, 1))
        return torch.cat(outs, 1)


HEADS = {'mdn': MDNHead, 'mcl': MCLHead, 'flow': FlowHead}


class GazeModel(nn.Module):
    def __init__(self, backbone, head, **head_kw):
        super().__init__()
        self.trunk = Trunk(backbone)
        self.head = HEADS[head](**head_kw)
        self.head_name = head

    def forward(self, images, bboxes, gaze):  # 训练：返回 (loss, logs)
        return self.head.loss(self.trunk.tokens(images, bboxes), gaze)

    @torch.no_grad()
    def sample(self, images, bboxes, n, **kw):
        return self.head.sample(self.trunk.tokens(images, bboxes), n, **kw)

    def decoder_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('trunk.backbone.')}

    def load_decoder_state(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith('trunk.backbone.') for k in missing), (missing[:3], unexpected[:3])
