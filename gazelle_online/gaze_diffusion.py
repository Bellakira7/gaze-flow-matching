"""B5：条件热图扩散（Heatmap diffusion）头。与 gaze_models.py 里的 MDN/MCL/Flow 共用同一个 Trunk（同一套场景 token），只换头。

生成对象 = 官方训练目标热图 gutils.get_heatmap(gx, gy, 64, 64)（σ=3 格的高斯、峰值 1），映射到 [-1,1]；
去噪网络 = 把 64x64 噪声热图 2x2 patchify 成 32x32 个 token（与场景 token 空间对齐）+ 场景 token（LN+Linear）+ 位置编码，
          3 个带时间 FiLM（adaLN-zero）的自注意力 Block，Linear 回 2x2 patch，输出 v；参数化 v-prediction，余弦 α/σ 调度，连续 t∈[0,1]。
采样 = DDIM（η=0，clip x0 到 [-1,1]），每实例画 M_draw 张热图；原生 pmf = M_draw 张热图各自归一化后取平均（负值截 0）；
        点样本 = 从该 pmf 抽（格内均匀抖动），假设 = 该 pmf 上 NMS 前 5 个峰（与 B1 同定义：半径 0.1、权重 = 峰周围半径 0.1 圆盘质量）。
        因此 B5 与 B1 的差别只在"热图是确定回归的"还是"扩散后验样本的混合"，其余后处理相同。
接口与其它头一致：loss(tokens, y) / sample(tokens, n, gen) -> [B,n,2]；另有 predict(tokens)（用最近一次 sample 缓存的 pmf 给原生假设）
和 last_pmf（[B,4096]，供 predict_test 写精确 pmf）。
"""
from pathlib import Path
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_online')
sys.path.insert(0, f'{G}/setmetrics')
import gaze_models as GM  # noqa: E402
import metrics as M  # noqa: E402
from gazelle.model import positionalencoding2d  # noqa: E402

SIGMA = 3.0  # 与官方 get_heatmap 默认一致


def target_heatmap(y, size=64, sigma=SIGMA):
    """与官方 gutils.get_heatmap 逐元素一致的向量化版本（test_gaze_diffusion.py 对拍）。y [B,2] 归一化坐标 -> [B,size,size]。"""
    cx = (y[:, 0] * size).long()  # int() 向零截断；坐标已夹在 [0,1]，非负故等价于 floor
    cy = (y[:, 1] * size).long()
    ax = torch.arange(size, device=y.device)
    dx = (ax[None] - cx[:, None]).float()  # [B,size]
    dy = (ax[None] - cy[:, None]).float()
    w = int(3 * sigma)
    gx = torch.exp(-dx ** 2 / (2 * sigma ** 2)) * (dx.abs() <= w)
    gy = torch.exp(-dy ** 2 / (2 * sigma ** 2)) * (dy.abs() <= w)
    img = gy[:, :, None] * gx[:, None, :]  # [B,y,x]
    return img / img.amax((1, 2), keepdim=True).clamp_min(1e-12)


class SBlock(nn.Module):
    """自注意力 Block，时间 FiLM 为 adaLN-zero（scale/shift/gate 全零初始化 -> 初始恒等）。"""

    def __init__(self, dim, heads=8, mlp_ratio=4):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(dim, elementwise_affine=False), nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv, self.o = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        B, N, D = x.shape
        s1, b1, g1, s2, b2, g2 = self.ada(c)[:, None].chunk(6, -1)
        q, k, v = self.qkv(self.n1(x) * (1 + s1) + b1).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        x = x + g1 * self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))
        return x + g2 * self.mlp(self.n2(x) * (1 + s2) + b2)


class DiffHead(nn.Module):
    def __init__(self, dim=256, depth=3, M_draw=16, steps=16, grid=32):
        super().__init__()
        self.dim, self.M_draw, self.steps, self.grid = dim, M_draw, steps, grid
        self.patch = nn.Conv2d(1, dim, 2, 2)
        self.cond = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.register_buffer('pos', positionalencoding2d(dim, grid, grid).flatten(1).t().contiguous(), persistent=False)  # [N,dim]
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(SBlock(dim) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)
        self.last_pmf = None

    def _temb(self, t):
        half = self.dim // 2
        fr = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        a = (t[:, None] * 1000.0) * fr
        return self.t_mlp(torch.cat([a.sin(), a.cos()], -1))

    def velocity(self, x, t, ctx):
        """x [B,1,64,64]，t [B]，ctx [B,N,dim]（已过 cond）-> v [B,1,64,64]。"""
        B = x.shape[0]
        h = self.patch(x).flatten(2).transpose(1, 2) + ctx + self.pos
        c = self._temb(t)
        for b in self.blocks:
            h = b(h, c)
        p = self.out(h).float()  # [B,N,4]
        return p.view(B, self.grid, self.grid, 2, 2).permute(0, 1, 3, 2, 4).reshape(B, 1, 2 * self.grid, 2 * self.grid)

    @staticmethod
    def _ab(t):  # 余弦调度 α=cos(πt/2)，σ=sin(πt/2)
        return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

    def loss(self, tokens, y):
        B = tokens.shape[0]
        x0 = target_heatmap(y)[:, None] * 2 - 1
        t = torch.rand(B, device=tokens.device)
        a, s = (u[:, None, None, None] for u in self._ab(t))
        eps = torch.randn_like(x0)
        v = self.velocity(a * x0 + s * eps, t, self.cond(tokens))
        l = F.mse_loss(v, a * eps - s * x0)
        return l, {'v_mse': l.item()}

    @torch.no_grad()
    def draw_heatmaps(self, tokens, M_draw=None, steps=None, gen=None, chunk=512, amp=True):
        """DDIM(η=0) 画 M_draw 张热图 -> [B,M,64,64]，值域 [0,1]。按 chunk 张分块（显存）；去噪网络默认 bf16 autocast（实测快 3.1x），
        状态 x 与 DDIM 更新始终是 fp32。"""
        M_draw, steps = M_draw or self.M_draw, steps or self.steps
        B, dev = tokens.shape[0], tokens.device
        ctx_all = self.cond(tokens)
        ts = torch.linspace(1, 0, steps + 1, device=dev)
        outs = []
        for s0 in range(0, B * M_draw, chunk):
            idx = torch.arange(s0, min(s0 + chunk, B * M_draw), device=dev) // M_draw
            ctx = ctx_all[idx]
            x = torch.randn(len(idx), 1, 2 * self.grid, 2 * self.grid, device=dev, generator=gen)
            for i in range(steps):
                t, s_ = ts[i].expand(len(idx)), ts[i + 1]
                a_t, sg_t = self._ab(t)
                a_t, sg_t = a_t[:, None, None, None], sg_t[:, None, None, None]
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    v = self.velocity(x, t, ctx)
                x0h = (a_t * x - sg_t * v).clamp(-1, 1)
                if i == steps - 1:
                    x = x0h
                else:
                    a_s, sg_s = self._ab(s_)
                    eps_h = (x - a_t * x0h) / sg_t.clamp_min(1e-6)  # 用 clip 后的 x0 重算 eps，保持 x_t = α x0 + σ eps 一致
                    x = a_s * x0h + sg_s * eps_h
            outs.append(x)
        return ((torch.cat(outs) + 1) / 2).view(B, M_draw, 2 * self.grid, 2 * self.grid)

    @torch.no_grad()
    def pmf(self, tokens, M_draw=None, steps=None, gen=None, lam=1e-3):
        """原生 pmf [B,4096]：各张热图（截 0）单独归一化后平均；全零的热图退化为均匀。"""
        H = self.draw_heatmaps(tokens, M_draw, steps, gen).clamp_min(0).flatten(2).double()  # [B,M,4096]
        s = H.sum(-1, keepdim=True)
        P = torch.where(s > 1e-6, H / s.clamp_min(1e-12), torch.full_like(H, 1.0 / H.shape[-1])).mean(1)
        return ((1 - lam) * P + lam / P.shape[-1]).float()

    @torch.no_grad()
    def sample(self, tokens, n, gen=None, **kw):
        P = self.pmf(tokens, gen=gen, **kw)
        self.last_pmf = P
        cells = torch.multinomial(P, n, replacement=True, generator=gen)
        jit = torch.rand(cells.shape + (2,), device=cells.device, generator=gen)
        G_ = 2 * self.grid
        return torch.stack([(cells % G_ + jit[..., 0]) / G_, (cells // G_ + jit[..., 1]) / G_], -1)

    @torch.no_grad()
    def predict(self, tokens=None):
        """原生假设：对最近一次 sample() 缓存的 pmf 做 NMS（B1 同定义）。返回 (hyps [B,5,2], w [B,5])，w 降序、和为 1。tokens 参数仅为与 MCL 同接口。"""
        assert self.last_pmf is not None, 'predict() 要在 sample() 之后调用'
        P = self.last_pmf.double().cpu().numpy()
        hy, ws = [], []
        for p in P:
            pk = M.nms_peaks(p.reshape(64, 64), 5)
            w = np.array([M.disc_mass(p, i) for i in pk])
            o = np.argsort(-w, kind='stable')
            h, w = M.CELLS[pk][o], w[o]
            if len(pk) < 5:
                h = np.concatenate([h, np.repeat(h[:1], 5 - len(pk), 0)])
                w = np.concatenate([w, np.zeros(5 - len(pk))])
            hy.append(h)
            ws.append(w / max(w.sum(), 1e-12))
        return torch.tensor(np.stack(hy), dtype=torch.float32), torch.tensor(np.stack(ws), dtype=torch.float32)


GM.HEADS['diff'] = DiffHead
NATIVE_HYPS = ('mcl', 'diff')  # 这些头用原生假设（其它头用样本聚类）
