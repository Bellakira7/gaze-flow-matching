"""【集合原生混合 flow 头】（方案一，prereg 澄清 C21）：模型直接输出「K 个假设 + 权重 + 每个假设自己的 flow 密度」。

    p(y | x) = Σ_k π_k(x) · p_k(y | x, e_k, a_k)

  * K 个可学习 query 经交叉注意力解码器（不带时间 FiLM）→ 每个 query 得到
        锚点 a_k ∈ [0,1]^2（sigmoid）、权重 logit、成分 embedding e_k（解码器输出，供 flow 条件化）；
  * flow 速度网络与 FlowHead 同构（Fourier(x_t) 作 query、3×FiLM 之外的层数可调、末层零初始化），
    只多一路条件：c = temb(t) + comp(e_k, Fourier(a_k))，同时进 query 与 FiLM；锚点作条件时 detach（锚点只由 WTA 损失训练）；
  * 训练 = hard-EM 的 relaxed 版：赢家 k* = argmin_k ‖a_k − y‖；成分权重 w = (1-ε) 给赢家、ε/(K-1) 给其余；
        loss = Σ_k w_k · FM_k(y)  +  λ_a · WTA(a, y)  +  λ_π · CE(softmax(logit), w)
    ε 同时起到两个作用：非赢家成分也学到尾部（密度的"地板"，防单标注下的过度自信），π 的软目标上限为 1-ε；
  * 采样：k ~ π，再按该成分的 conditioning 走 ODE（Heun / γ 时间网格，与 FlowHead 同一采样器，读 sampler_selection.json 的配置）；
  * 原生假设：predict() 返回按权重降序的 (假设, 权重)，不需要 NMS、不需要 σ_h；K > 峰数时重复锚点按固定半径 0.10
    确定性合并（权重求和、位置取加权质心），点估计 = 权重最大的合并假设。

接口与其它头一致：loss(tokens, y) / sample(tokens, n, gen=None, **kw) / predict(tokens)。
K=1 退化为"单成分 + 锚点条件"的 flow，作为同结构的容量对照。

注册：`import gaze_mixflow` 会把 'mixflow' 加进 gaze_models.HEADS，并把它加进 gaze_diffusion.NATIVE_HYPS（训练脚本据此在 hold-out 日志里用原生假设）。
"""
from pathlib import Path
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_online')
import gaze_models as GM  # noqa: E402
import gaze_diffusion as GD  # noqa: E402

XDecoder = GM.XDecoder


class MixFlowHead(nn.Module):
    def __init__(self, dim=256, K=5, gate_depth=2, depth=4, n_freq=8, n_draw=16, steps=16, gamma=2.0, solver='heun',
                 eps=0.05, w_anchor=1.0, w_pi=0.1):
        super().__init__()
        self.dim, self.K, self.n_freq, self.n_draw = dim, K, n_freq, n_draw
        self.steps, self.gamma, self.solver = steps, gamma, solver
        self.eps, self.w_anchor, self.w_pi = eps, w_anchor, w_pi
        # 假设分支：K 个 query -> (锚点, 权重 logit, embedding)
        self.queries = nn.Parameter(torch.randn(1, K, dim) * 0.02)
        self.gate = XDecoder(dim, gate_depth)
        self.gate_norm = nn.LayerNorm(dim)
        self.gate_out = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, 3))
        # flow 分支：与 FlowHead 同构，多一路成分条件
        self.inp = nn.Linear(2 + 4 * n_freq, dim)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.comp = nn.Linear(dim + 2 + 4 * n_freq, dim)
        self.dec = XDecoder(dim, depth, film=True)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    # ---------------------------------------------------------------- 假设分支
    def hypotheses(self, tokens):
        """-> anchors [B,K,2]（[0,1]）、logit [B,K]（float32）、emb [B,K,dim]。"""
        B = tokens.shape[0]
        z = self.gate_norm(self.gate(self.queries.expand(B, -1, -1), self.gate.make_kv(tokens)))
        o = self.gate_out(z).float()
        return torch.sigmoid(o[..., :2]), o[..., 2], z

    def _fourier(self, x):
        f = (2.0 ** torch.arange(self.n_freq, device=x.device)) * math.pi
        a = x[..., None] * f
        return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)

    def _temb(self, t):
        half = self.dim // 2
        fr = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        a = (t[..., None] * 1000.0) * fr
        return self.t_mlp(torch.cat([a.sin(), a.cos()], -1))

    def comp_cond(self, emb, anchors):
        """成分条件 [.., dim]：embedding 与（detach 后的）锚点的 Fourier 特征。锚点用 [-1,1] 坐标，与 flow 的坐标系一致。"""
        return self.comp(torch.cat([emb, self._fourier(anchors.detach() * 2 - 1)], -1))

    def velocity(self, x, t, kvs, cc):
        """x [B,Q,2]，t [B,Q] 或 [B,1]，cc [B,Q,dim]（每个 query 所属成分的条件）。"""
        c = self._temb(t) + cc
        return self.out(self.dec(self.inp(self._fourier(x)) + c, kvs, c)).float()

    # ---------------------------------------------------------------- 训练
    def loss(self, tokens, y):
        B, K, nd = tokens.shape[0], self.K, self.n_draw
        anchors, logit, emb = self.hypotheses(tokens)
        d = ((anchors - y[:, None]) ** 2).sum(-1).add(1e-12).sqrt()  # [B,K]
        best = d.argmin(1)
        db = d.gather(1, best[:, None])[:, 0]
        if K > 1:
            w = torch.full_like(d, self.eps / (K - 1))
            w.scatter_(1, best[:, None], 1 - self.eps)
            wta = ((1 - self.eps) * db + self.eps / (K - 1) * (d.sum(1) - db)).mean()
        else:
            w = torch.ones_like(d)
            wta = db.mean()
        ce = -(w * F.log_softmax(logit, -1)).sum(-1).mean()  # 软目标交叉熵（目标 = 松弛后的赢家分布）

        cc = self.comp_cond(emb, anchors)  # [B,K,dim]
        cc = cc[:, :, None].expand(-1, -1, nd, -1).reshape(B, K * nd, -1)
        x1 = (y * 2 - 1)[:, None].expand(-1, K * nd, -1)
        x0 = torch.randn_like(x1)
        t = torch.rand(B, K * nd, device=tokens.device)
        xt = (1 - t[..., None]) * x0 + t[..., None] * x1
        v = self.velocity(xt, t, self.dec.make_kv(tokens), cc)
        mse = ((v - (x1 - x0)) ** 2).mean(-1).view(B, K, nd).mean(-1)  # [B,K] 每个成分自己的 FM 误差
        fm = (w * mse).sum(1).mean()
        total = fm + self.w_anchor * wta + self.w_pi * ce
        pi = F.softmax(logit.detach(), -1)
        logs = {'fm': fm.item(), 'fm_win': mse.gather(1, best[:, None]).mean().item(), 'wta': wta.item(), 'ce': ce.item(),
                'best_l2': db.mean().item(), 'used_k': torch.bincount(best, minlength=K).float().div(B).max().item(),
                'pi_max': pi.max(-1)[0].mean().item(), 'pi_ent': (-(pi * pi.clamp_min(1e-12).log()).sum(-1)).mean().item()}
        return total, logs

    # ---------------------------------------------------------------- 原生假设与采样
    @torch.no_grad()
    def predict(self, tokens, merge=0.10):
        """原生输出：假设 [B,K,2] 与权重 [B,K]，按权重降序、和为 1。
        K > 真实峰数时锚点会重复覆盖同一峰、π 在重复锚点间分摊（合成自检实测：3+2 个锚点的 π 按峰求和恰为 0.6/0.4），
        所以 merge>0 时做确定性合并：按 π 降序贪心，领头锚点半径 merge 内的全部并入（权重求和、位置取 π 加权质心），再按合并后权重降序。
        半径固定 0.10 = 评测协议的假设匹配半径，不是调出来的超参。不足 K 个用 top1 补齐、权重 0（与 gaze_eval 同约定）。"""
        anchors, logit, _ = self.hypotheses(tokens)
        w = F.softmax(logit, -1)
        if not merge:
            w, idx = w.sort(-1, descending=True)
            return anchors.gather(1, idx[..., None].expand(-1, -1, 2)), w
        B, K = w.shape
        out_h, out_w = anchors.new_zeros(B, K, 2), w.new_zeros(B, K)
        for b in range(B):
            a, ww = anchors[b], w[b]
            avail = torch.ones(K, dtype=torch.bool, device=a.device)
            hs, ws = [], []
            while avail.any():
                i = torch.where(avail, ww, ww.new_full((K,), -1.0)).argmax()
                grp = avail & ((a - a[i]).norm(dim=-1) <= merge)
                wg = ww[grp]
                ws.append(wg.sum())
                hs.append((a[grp] * wg[:, None]).sum(0) / wg.sum().clamp_min(1e-12))
                avail &= ~grp
            hs, ws = torch.stack(hs), torch.stack(ws)
            o = ws.argsort(descending=True)
            hs, ws = hs[o], ws[o]
            m = len(ws)
            out_h[b, :m], out_w[b, :m] = hs, ws
            if m < K:
                out_h[b, m:] = hs[0]
        return out_h, out_w

    @torch.no_grad()
    def sample(self, tokens, n, gen=None, steps=None, chunk=2500, gamma=None, solver=None, return_comp=False):
        """k ~ π，再按该成分的条件走 ODE。时间网格 / 求解器与 FlowHead.sample 逐行相同。"""
        steps, gamma, solver = steps or self.steps, gamma or self.gamma, solver or self.solver
        anchors, logit, emb = self.hypotheses(tokens)
        cc_all = self.comp_cond(emb, anchors)  # [B,K,dim]
        pi = F.softmax(logit, -1)
        kvs = self.dec.make_kv(tokens)
        B = tokens.shape[0]
        ts = (1 - (1 - torch.linspace(0, 1, steps + 1, device=tokens.device)) ** gamma).tolist()
        outs, comps = [], []
        for s in range(0, n, chunk):
            m = min(chunk, n - s)
            comp = torch.multinomial(pi, m, replacement=True, generator=gen)  # [B,m]
            cc = cc_all.gather(1, comp[..., None].expand(-1, -1, cc_all.shape[-1]))
            x = torch.randn(B, m, 2, device=tokens.device, generator=gen)
            for i in range(steps):
                t0, t1 = ts[i], ts[i + 1]
                v = self.velocity(x, torch.full((B, 1), t0, device=tokens.device), kvs, cc)
                if solver == 'euler':
                    x = x + (t1 - t0) * v
                else:
                    xe = x + (t1 - t0) * v
                    v2 = self.velocity(xe, torch.full((B, 1), t1, device=tokens.device), kvs, cc)
                    x = x + (t1 - t0) * 0.5 * (v + v2)
            outs.append(((x + 1) / 2).clamp(0, 1))
            comps.append(comp)
        S = torch.cat(outs, 1)
        return (S, torch.cat(comps, 1)) if return_comp else S


GM.HEADS['mixflow'] = MixFlowHead
if 'mixflow' not in GD.NATIVE_HYPS:
    GD.NATIVE_HYPS = GD.NATIVE_HYPS + ('mixflow',)
