"""射线分解头（ray head）：把注视目标写成"从头部出发的一条射线 + 射线上的位置"，p(y | 图, 头) = p(θ | ·) · p(τ | θ, ·)。

动机（事后分析，`tools/ray_decomp_analysis.py`，`results/ray_decomp/`）：GazeFollow 测试集上标注者分歧沿射线方向更大（径向/切向标准差中位 1.35，
M_sep 上 1.62）；M_sep 两个最大标注簇的位移 64% 主要沿射线；模型点误差也以径向为主，而眼部 token 只降低了方向误差（两个 seed 都是 −7.5% 左右，
径向误差不变）。所以把"看哪个方向"（人物线索）和"沿射线看多远"（场景线索）分开建模。

参数化（全部在归一化图像坐标里，与模型看到的 448×448 输入一致；仿射缩放把直线映成直线，所以不需要宽高比）：
  c = 头框中心（夹到 [0,1]²），θ = atan2(y−c)，s = |y−c|，s_max(θ) = 从 c 沿 θ 到图像边界的距离，τ = s / s_max ∈ [0,1)。
  1. 方向 p(θ | ·)：把方向写成单位圆上的点 u = (cos θ, sin θ)，直接用 gaze_models.FlowHead 在 R² 里做 flow matching（目标 (u+1)/2 ∈ [0,1]²，
     高斯基分布、直线插值、交叉注意力到全部 token 即场景 + 眼部），采样后取 θ = atan2(u)。
     10-28 初版用圆周 flow（基分布 U(−π,π)、测地插值）：速度场在每个峰的对径点有 ±π 跳变，合成自检里两方向双峰的方向边缘 TV 0.072，
     同数据 2D flow 为 0.021；换成单位圆嵌入后速度场光滑，且方向部分与 flow / flowET 是同一份代码、同一采样器。
  2. 沿射线 p(τ | θ, ·)：沿 θ 在 32×32 场景 token 网格上等距取 T 个点（双线性 grid_sample，"ray marching"），加上位置 / τ / 方向嵌入，
     再加一个"射出画面"token；每层 = 射线内自注意力 + 对全部 token 的交叉注意力；输出 T+1 类（T 个位置档 + 射出画面）。
  3. 画面外：P(out) = E_{θ~p(θ)}[p(exit | θ)]。GazeFollow 训练集全是画面内（exit 类从不当目标）；VAT 的画面外样本训练 exit 类：方向未知，
     射线从**模型当前的方向分布**采样（no_grad，8 步 Heun），标签都是 exit；不训方向。这样 p(exit|θ) 在 p(θ) 有质量的地方被训练，
     E_{θ~p(θ)}[p(exit|θ)] 才是 P(out) 的一致估计（10-28 合成自检：用均匀随机方向时 P(out) 真值 0.30 只估出 0.045）。
训练：L = L_FM(θ) + w_ray · CE(τ 档)。射线头只用**真值方向**的射线（n_rays=1、jitter_deg=0）：(θ_gt, τ_gt) 取自真实联合分布，CE 就是 p(τ|θ) 的
     最大似然。10-28 初版用了"真值 + 3 条 N(0,3°) 抖动射线、标签取投影"，等于把 p(τ|θ) 在方向上抹宽 3°，与方向边缘分布组合后联合分布有偏
     （合成自检：同一射线上近 / 远两峰 0.6 / 0.4 采出 0.534 / 0.441），已弃用；参数保留只为可复现。
采样：先采 n 个 θ；把方向量化到 A 个等宽档（默认 360，即 1°），在每个档中心各算一次 p(τ | θ)（与 n 无关的固定开销）；
     画面内条件采样时按 1 − p(exit | θ) 对 θ 样本重采样，再在画面内 T 档里抽档、档内均匀抖动 τ，θ 保持连续值。
输出与其它头相同：[B, n, 2] 的样本（x 在前），评测管线不变。

消融开关：raymarch=False → 射线 token 不取场景特征（只有位置 / τ / 方向嵌入，仍有交叉注意力）；RayGazeModel(eye=False) → 不要眼部 token。
不改 gaze_models.py / gaze_eye.py（已出数字的 flow / flowET 依赖它们）。
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
from gaze_eye import EyeTrunk  # noqa: E402

SCENE_HW = 32  # 448 / 14
N_SCENE = SCENE_HW * SCENE_HW


def wrap(a):
    """角度折回 [−π, π)。"""
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def ray_len(c, d, eps=1e-9):
    """c [...,2] ∈ [0,1]² 起点，d [...,2] 单位方向 -> 沿 d 到 [0,1]² 边界的距离 [...]（下限 1e-4）。"""
    big = torch.full_like(d[..., 0], 1e6)

    def axis(ci, di):
        return torch.where(di > eps, (1 - ci) / di.clamp_min(eps), torch.where(di < -eps, -ci / di.clamp_max(-eps), big))

    return torch.minimum(axis(c[..., 0], d[..., 0]), axis(c[..., 1], d[..., 1])).clamp_min(1e-4)


def unit(th):
    return torch.stack([th.cos(), th.sin()], -1)


def march(tokens, pts):
    """在 32×32 场景 token 网格上双线性取特征。tokens [B,N,D]（前 1024 个行优先，下标 iy*32+ix，token 中心 ((ix+0.5)/32, (iy+0.5)/32)），
    pts [B,R,T,2]（x 在前）-> [B,R,T,D]。align_corners=False 正好对上 token 中心的位置（test_gaze_ray.py 校验）。"""
    B, D = tokens.shape[0], tokens.shape[2]
    grid = tokens[:, :N_SCENE].transpose(1, 2).reshape(B, D, SCENE_HW, SCENE_HW)
    return F.grid_sample(grid, pts * 2 - 1, mode='bilinear', padding_mode='border', align_corners=False).permute(0, 2, 3, 1)


def to_polar(y, c):
    """y, c [...,2] -> θ [...], τ [...]（τ 夹到 [0, 1)）。"""
    v = y - c
    th = torch.atan2(v[..., 1], v[..., 0])
    tau = v.norm(dim=-1) / ray_len(c, unit(th))
    return th, tau.clamp(0, 1 - 1e-6)


class RayHead(nn.Module):
    def __init__(self, dim=256, depth=3, n_freq=8, n_draw=16, steps=16, gamma=3.0, solver='heun',
                 T=64, ray_depth=2, n_rays=1, jitter_deg=0.0, w_ray=1.0, A=360, raymarch=True, heads=8, out_dir_steps=8):
        super().__init__()
        self.dim, self.n_freq, self.n_draw, self.steps, self.gamma, self.solver = dim, n_freq, n_draw, steps, gamma, solver
        self.T, self.n_rays, self.jitter, self.w_ray, self.A, self.raymarch = T, n_rays, math.radians(jitter_deg), w_ray, A, raymarch
        self.out_dir_steps = out_dir_steps
        self.register_buffer('afreq', 2.0 ** torch.arange(n_freq), persistent=False)  # 整数频率 1,2,4,…,128：周期函数，最细约 1.4°
        # 方向：单位圆上的点，2D flow（与 FlowHead 同一份代码）
        self.dirflow = GM.FlowHead(dim, depth, n_freq, n_draw, steps, gamma, solver)
        # 沿射线
        self.pos_emb = nn.Linear(2 + 4 * n_freq, dim)
        self.tau_emb = nn.Linear(1 + 2 * n_freq, dim)
        self.dir_emb = nn.Linear(2 * n_freq, dim)
        if raymarch:
            self.feat_proj = nn.Linear(dim, dim)
        self.exit_tok = nn.Parameter(torch.randn(dim) * 0.02)
        self.ray_self = nn.ModuleList(GM.XBlock(dim, heads) for _ in range(ray_depth))
        self.ray_cross = nn.ModuleList(GM.XBlock(dim, heads) for _ in range(ray_depth))
        self.ray_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.register_buffer('tau_c', (torch.arange(T) + 0.5) / T, persistent=False)

    # ---------------- 嵌入
    def _afour(self, th):
        a = th[..., None] * self.afreq
        return torch.cat([a.cos(), a.sin()], -1)

    def _f1(self, x):  # 标量 x ∈ [0,1]
        a = x[..., None] * self.afreq * math.pi
        return torch.cat([x[..., None], a.sin(), a.cos()], -1)

    def _f2(self, p):  # 二维点 p ∈ [0,1]²，与 FlowHead._fourier 同式（先映到 [−1,1]）
        x = p * 2 - 1
        a = x[..., None] * self.afreq * math.pi
        return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)

    # ---------------- 方向
    def dir_loss(self, tokens, th1):
        return self.dirflow.loss(tokens, (unit(th1) + 1) / 2)[0]

    @torch.no_grad()
    def sample_dir(self, tokens, n, gen=None, steps=None, gamma=None, solver=None, chunk=2500):
        """-> θ [B,n]。"""
        u = self.dirflow.sample(tokens, n, gen=gen, steps=steps, chunk=chunk, gamma=gamma, solver=solver) * 2 - 1
        return torch.atan2(u[..., 1], u[..., 0])

    # ---------------- 沿射线
    def ray_logits(self, tokens, kv_cross, c, th):
        """tokens [B,N,D]（前 1024 个是 32×32 场景 token，行优先），kv_cross = [blk.make_kv(tokens) for blk in ray_cross]，
        c [B,2]，th [B,R] -> logits [B,R,T+1]（最后一类 = 射出画面）。"""
        B, R = th.shape
        T, D = self.T, self.dim
        d = unit(th)
        smax = ray_len(c[:, None].expand(-1, R, -1), d)
        pts = c[:, None, None] + self.tau_c[None, None, :, None] * smax[..., None, None] * d[:, :, None]  # [B,R,T,2]
        de = self.dir_emb(self._afour(th))  # [B,R,D]
        x = self.pos_emb(self._f2(pts)) + self.tau_emb(self._f1(self.tau_c))[None, None] + de[:, :, None]
        if self.raymarch:
            x = x + self.feat_proj(march(tokens, pts))
        x = torch.cat([x, (self.exit_tok + de)[:, :, None]], 2)  # [B,R,T+1,D]
        for sb, cb, (k, v) in zip(self.ray_self, self.ray_cross, kv_cross):
            xs = x.reshape(B * R, T + 1, D)
            xs = sb(xs, *sb.make_kv(xs))
            x = cb(xs.reshape(B, R * (T + 1), D), k, v).reshape(B, R, T + 1, D)
        return self.ray_out(x).float()[..., 0]

    # ---------------- 训练
    def loss(self, tokens, y, c, inout=None):
        """tokens [B,N,D]，y [B,2]（画面外的行内容不用），c [B,2] 头框中心，inout [B] bool（None = 全部画面内）。"""
        B, dev = tokens.shape[0], tokens.device
        c = c.clamp(0, 1)
        inn = torch.ones(B, dtype=torch.bool, device=dev) if inout is None else inout.bool()
        th1, _ = to_polar(y, c)
        logs = {}
        loss = tokens.new_zeros(())
        if inn.any():
            l_dir = self.dir_loss(tokens[inn], th1[inn])
            loss = loss + l_dir
            logs['fm_mse'] = l_dir.item()
        R = self.n_rays
        jit = torch.randn(B, R, device=dev) * self.jitter
        jit[:, 0] = 0
        thr = wrap(th1[:, None] + jit)
        if not inn.all():  # 画面外：射线方向从模型当前的方向分布采样（不回传）
            with torch.no_grad():
                th_out = self.sample_dir(tokens[~inn].detach(), R, steps=self.out_dir_steps)
            thr = thr.masked_scatter(~inn[:, None].expand(-1, R), th_out)
        d = unit(thr)
        proj = ((y - c)[:, None] * d).sum(-1)
        tau = (proj / ray_len(c[:, None].expand(-1, R, -1), d)).clamp(0, 1 - 1e-6)
        tgt = torch.where(inn[:, None], (tau * self.T).long(), torch.full_like(tau, self.T, dtype=torch.long))
        logits = self.ray_logits(tokens, [b.make_kv(tokens) for b in self.ray_cross], c, thr)
        ce = F.cross_entropy(logits.reshape(-1, self.T + 1), tgt.reshape(-1))
        loss = loss + self.w_ray * ce
        with torch.no_grad():
            logs.update(ray_ce=ce.item(), ray_acc0=(logits[:, 0].argmax(-1) == tgt[:, 0]).float().mean().item())
        return loss, logs

    # ---------------- 采样
    @torch.no_grad()
    def sample(self, tokens, n, c, gen=None, steps=None, gamma=None, solver=None, chunk=2500, A=None, in_frame=True,
               ray_chunk=90, return_pout=False):
        """-> 样本 [B,n,2]（in_frame=True：以画面内为条件）；return_pout 时另返回 P(out) [B]。"""
        B, dev = tokens.shape[0], tokens.device
        c = c.clamp(0, 1)
        A, T = A or self.A, self.T
        th = self.sample_dir(tokens, n, gen, steps, gamma, solver, chunk)
        grid_th = (-math.pi + (torch.arange(A, device=dev) + 0.5) * (2 * math.pi / A))[None].expand(B, -1)
        kvc = [b.make_kv(tokens) for b in self.ray_cross]
        P = torch.cat([self.ray_logits(tokens, kvc, c, grid_th[:, s:s + ray_chunk]) for s in range(0, A, ray_chunk)], 1).softmax(-1)  # [B,A,T+1]
        a = ((th + math.pi) * (A / (2 * math.pi))).floor().long().clamp(0, A - 1)
        pexit = P[..., T].gather(1, a)
        p_out = pexit.mean(1)
        if in_frame:
            idx = torch.multinomial((1 - pexit).clamp_min(1e-12), n, replacement=True, generator=gen)
            th, a = th.gather(1, idx), a.gather(1, idx)
            Pr = P[..., :T] / P[..., :T].sum(-1, keepdim=True).clamp_min(1e-12)
        else:
            raise NotImplementedError('画面外样本没有位置；要 P(out) 用 return_pout')
        rows = Pr.gather(1, a[..., None].expand(-1, -1, T))  # [B,n,T]
        j = torch.multinomial(rows.reshape(-1, T), 1, generator=gen).view(B, n)
        tau = (j + torch.rand(B, n, device=dev, generator=gen)) / T
        d = unit(th)
        pts = (c[:, None] + (tau * ray_len(c[:, None].expand(-1, n, -1), d))[..., None] * d).clamp(0, 1)
        return (pts, p_out) if return_pout else pts


def head_centers(bboxes, device):
    """bboxes：每张图一个头框列表（归一化 xyxy）-> [sum(N_p), 2] 头框中心，顺序与 trunk.tokens 的实例顺序一致。"""
    return torch.tensor([[(float(b[0]) + float(b[2])) / 2, (float(b[1]) + float(b[3])) / 2] for bl in bboxes for b in bl],
                        dtype=torch.float32, device=device)


class RayGazeModel(GM.GazeModel):
    """与 GazeModel 同接口（trunk.tokens / decoder_state / load_decoder_state）；head 需要头框中心，所以 forward / 采样另走 sample_tokens。"""

    def __init__(self, backbone, eye=True, eye_kw=None, **head_kw):
        nn.Module.__init__(self)
        self.trunk = EyeTrunk(backbone, **(eye_kw or {})) if eye else GM.Trunk(backbone)
        self.head = RayHead(**head_kw)
        self.head_name = 'ray'

    def forward(self, images, bboxes, gaze, inout=None):
        return self.head.loss(self.trunk.tokens(images, bboxes), gaze, head_centers(bboxes, gaze.device), inout)

    @torch.no_grad()
    def sample_tokens(self, tok, bboxes, n, **kw):
        return self.head.sample(tok, n, head_centers(bboxes, tok.device), **kw)

    @torch.no_grad()
    def sample(self, images, bboxes, n, **kw):
        return self.sample_tokens(self.trunk.tokens(images, bboxes), bboxes, n, **kw)


# ====================================================================== v2：射线 token（输出仍是 2D 笛卡尔 flow）
"""v2（10-29，prereg 澄清 C16）。v1 射线分解头的诊断：极坐标输出在近处（|y−c| 前 20%）NLL 比 flowET 好 0.13，远处（后 20%）差 0.21——
方向上的小误差在远处被放大成弧、沿射线分档在远处变粗。所以 v2 **不改输出**（仍是 FlowHead 的 2D flow），只把"以头为中心的极坐标重采样"
作为额外的条件 token：
  从头框中心 c 发 K 条等角射线（默认 64，5.6°），每条在 32×32 场景 token 上等距取 T 个点（默认 16，从 c 到图像边界），
  token = 双线性场景特征 + 绝对位置 / 方向 / τ 嵌入；再过 depth 层**极坐标轴向自注意力**（偶数层射线内：同一方向由近到远；
  奇数层射线间：同一 τ 的各方向），加类型向量后追加到 trunk token（场景 1024 + 眼部 256）后面，FlowHead 的交叉注意力一并读取。
容量：射线编码器 4 层 ≈ 324 万，FlowHead 3 层 330 万，合计 ≈ 654 万，与容量对照 flowET_d6（6 层 FlowHead，646 万）对齐。
"""


class RayTokens(nn.Module):
    def __init__(self, dim=256, K=64, T=16, depth=4, heads=8, n_freq=8):
        super().__init__()
        self.dim, self.K, self.T = dim, K, T
        self.register_buffer('afreq', 2.0 ** torch.arange(n_freq), persistent=False)
        self.register_buffer('th', torch.arange(K) * (2 * math.pi / K), persistent=False)
        self.register_buffer('tau_c', (torch.arange(T) + 0.5) / T, persistent=False)
        self.feat_proj = nn.Linear(dim, dim)
        self.pos_emb = nn.Linear(2 + 4 * n_freq, dim)
        self.dir_emb = nn.Linear(2 * n_freq, dim)
        self.tau_emb = nn.Linear(1 + 2 * n_freq, dim)
        self.blocks = nn.ModuleList(GM.XBlock(dim, heads) for _ in range(depth))
        self.ray_type = nn.Parameter(torch.zeros(dim))

    _afour, _f1, _f2 = RayHead._afour, RayHead._f1, RayHead._f2

    def points(self, c):
        """c [B,2] -> 射线采样点 [B,K,T,2]。"""
        B = c.shape[0]
        d = unit(self.th)[None].expand(B, -1, -1)  # [B,K,2]
        smax = ray_len(c[:, None].expand(-1, self.K, -1), d)  # [B,K]
        return c[:, None, None] + self.tau_c[None, None, :, None] * smax[..., None, None] * d[:, :, None]

    def forward(self, tokens, c):
        B, K, T, D = tokens.shape[0], self.K, self.T, self.dim
        pts = self.points(c.clamp(0, 1))
        x = (self.feat_proj(march(tokens, pts)) + self.pos_emb(self._f2(pts))
             + self.dir_emb(self._afour(self.th))[None, :, None] + self.tau_emb(self._f1(self.tau_c))[None, None])  # [B,K,T,D]
        for i in range(len(self.blocks)):
            x = self.layer(i, x)
        return x.reshape(B, K * T, D) + self.ray_type

    def layer(self, i, x):
        """x [B,K,T,D]；偶数层射线内自注意力，奇数层射线间（同一 τ）自注意力。"""
        B, K, T, D = x.shape
        blk = self.blocks[i]
        if i % 2 == 0:
            xs = x.reshape(B * K, T, D)
            return blk(xs, *blk.make_kv(xs)).reshape(B, K, T, D)
        xs = x.transpose(1, 2).reshape(B * T, K, D)
        return blk(xs, *blk.make_kv(xs)).reshape(B, T, K, D).transpose(1, 2)


class RayTokEyeTrunk(EyeTrunk):
    """EyeTrunk + 射线 token：tokens() = [场景 1024, 眼部 256, 射线 K*T]。接口与 EyeTrunk 相同（含测试路径的 raw_images），所以
    train_heads_eye / eval / predict 里"trunk.tokens → head.sample"的写法不用改。"""

    def __init__(self, backbone, ray_kw=None, **kw):
        super().__init__(backbone, **kw)
        self.rays = RayTokens(self.dim, **(ray_kw or {}))

    def tokens(self, images, bboxes, raw_images=None):
        t = super().tokens(images, bboxes, raw_images=raw_images)
        return torch.cat([t, self.rays(t, head_centers(bboxes, t.device))], 1)


class RayTokGazeModel(GM.GazeModel):
    """v2：trunk = RayTokEyeTrunk，head = FlowHead（默认 3 层）。sample_tokens 与 v1 的 RayGazeModel 同接口，方便共用脚本。"""

    def __init__(self, backbone, eye_kw=None, ray_kw=None, **head_kw):
        nn.Module.__init__(self)
        self.trunk = RayTokEyeTrunk(backbone, ray_kw=ray_kw, **(eye_kw or {}))
        self.head = GM.FlowHead(**head_kw)
        self.head_name = 'raytok'

    @torch.no_grad()
    def sample_tokens(self, tok, bboxes, n, return_pout=False, **kw):
        S = self.head.sample(tok, n, **kw)
        return (S, torch.full((S.shape[0],), float('nan'), device=S.device)) if return_pout else S
