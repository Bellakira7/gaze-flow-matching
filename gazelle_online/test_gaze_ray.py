"""gaze_ray 自检（单卡，不依赖图像 / backbone）。
CUDA_VISIBLE_DEVICES=4 python -I -u test_gaze_ray.py

1. 几何：ray_len 的终点落在 [0,1]² 边界上；to_polar 往返误差；wrap 落在 [−π, π)。
2. march（射线取特征）与 token 网格对齐：在 token 中心取样恰好取回该 token 的值。
3. 合成条件分布（3 个"实例"，固定随机 token 作为 trunk 输出，只训头）：
   实例 0 = **同一条射线上的近 / 远两个峰**（0.6 / 0.4）——射线头要分出两档；
   实例 1 = 两个不同方向的峰（0.5 / 0.5）——方向 flow 要双峰；
   实例 2 = 单峰 + 30% 画面外标签——检验 exit 类与 P(out)。
   训练 3000 步、Adam 1e-3 + 余弦退火到 0（同 test_gaze_models.py 的做法）；另在同一数据上训一个 2D FlowHead 作参照。
   判据：画面内采样直方图对真实分布的 TV（16×16 分箱，真实分布用 40 万样本）< 0.05，并报 2D flow 参照的 TV 与方向边缘分布（5° 分箱）的 TV；
   实例 0 近峰质量 0.6±0.05；实例 2 的 P(out) 0.30±0.05，实例 0/1 的 P(out) < 0.03；同种子两次采样逐位相同；A=360 与 A=720 的 TV < 0.03。
4. 计时：B=8、N=1280 token、10000 个样本的单实例采样时间（只作参考）。
"""
from pathlib import Path
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gazelle_online'))
import gaze_ray as GR  # noqa: E402
import gaze_models as GM  # noqa: E402

dev = 'cuda'
torch.manual_seed(0)
fails = []


def check(name, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not ok:
        fails.append(name)


# 1 几何
c = torch.rand(10000, 2, device=dev)
th = (torch.rand(10000, device=dev) * 2 - 1) * math.pi
L = GR.ray_len(c, GR.unit(th))
p = c + L[:, None] * GR.unit(th)
edge = torch.minimum(p, 1 - p).min(1).values
free = L > 1e-4  # ray_len 有 1e-4 的下限（起点贴边、朝外时），这些射线终点最多越界 1e-4
check('ray_len 终点在边界上', edge[free].abs().max().item() < 1e-5 and edge[~free].abs().max().item() <= 1.01e-4 if (~free).any() else edge.abs().max().item() < 1e-5,
      f'max |到边界距离| 未触发下限 {edge[free].abs().max().item():.1e}（{int(free.sum())} 条），触发下限 {int((~free).sum())} 条')
y = torch.rand(10000, 2, device=dev)
th2, tau = GR.to_polar(y, c)
y2 = c + (tau * GR.ray_len(c, GR.unit(th2)))[:, None] * GR.unit(th2)
check('to_polar 往返', (y2 - y).abs().max().item() < 1e-5, f'max err {(y2 - y).abs().max().item():.1e}')
w = GR.wrap(torch.linspace(-20, 20, 10001, device=dev))
check('wrap ∈ [−π, π)', bool((w >= -math.pi).all() & (w < math.pi).all()))

# 2 march 对齐
tok = torch.zeros(1, GR.N_SCENE + 7, 4, device=dev)
iy, ix = torch.meshgrid(torch.arange(32), torch.arange(32), indexing='ij')
tok[0, :GR.N_SCENE, 0] = ix.reshape(-1).float().to(dev)
tok[0, :GR.N_SCENE, 1] = iy.reshape(-1).float().to(dev)
cent = torch.stack([(ix.reshape(-1) + 0.5) / 32, (iy.reshape(-1) + 0.5) / 32], -1).float().to(dev)
f = GR.march(tok, cent[None, None])[0, 0]
check('march 在 token 中心取回 (ix, iy)', torch.allclose(f[:, 0], tok[0, :GR.N_SCENE, 0], atol=1e-4) and torch.allclose(f[:, 1], tok[0, :GR.N_SCENE, 1], atol=1e-4),
      f'max err {(f[:, :2] - tok[0, :GR.N_SCENE, :2]).abs().max().item():.1e}')
mid = GR.march(tok, torch.tensor([[[[(3 + 0.5 + 0.5) / 32, (5 + 0.5) / 32]]]], device=dev))[0, 0, 0]
check('march 两 token 中点取平均', abs(mid[0].item() - 3.5) < 1e-4 and abs(mid[1].item() - 5) < 1e-4, f'{mid[:2].tolist()}')

# 3 合成分布
CENTERS = torch.tensor([[0.3, 0.5], [0.5, 0.3], [0.7, 0.7]], device=dev)
MODES = [  # (权重, 中心, σ)
    [(0.6, (0.45, 0.5), 0.015), (0.4, (0.85, 0.5), 0.015)],   # 同一条射线（θ=0）上的近 / 远
    [(0.5, (0.2, 0.7), 0.02), (0.5, (0.8, 0.7), 0.02)],       # 两个方向
    [(1.0, (0.4, 0.85), 0.03)],
]
P_OUT = [0.0, 0.0, 0.3]


def draw(k, n, rng):
    w = np.array([m[0] for m in MODES[k]])
    comp = rng.choice(len(w), n, p=w / w.sum())
    mu = np.array([m[1] for m in MODES[k]])[comp]
    sg = np.array([m[2] for m in MODES[k]])[comp]
    return np.clip(mu + rng.standard_normal((n, 2)) * sg[:, None], 0, 1 - 1e-6)


def hist(S, b=16):
    idx = np.clip((S * b).astype(int), 0, b - 1)
    h = np.bincount(idx[:, 1] * b + idx[:, 0], minlength=b * b).astype(float)
    return h / h.sum()


toks = torch.randn(3, GR.N_SCENE, 256, device=dev)
rng = np.random.default_rng(0)
BS, STEPS = 48, 3000


def batch():
    ys, ks, ins = [], [], []
    for k in range(3):
        ys.append(draw(k, BS, rng))
        ks += [k] * BS
        ins.append(rng.random(BS) >= P_OUT[k])
    return (torch.tensor(np.concatenate(ys), dtype=torch.float32, device=dev), torch.tensor(ks, device=dev),
            torch.tensor(np.concatenate(ins), device=dev))


def fit(head, lossfn, tag):
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    t0 = time.time()
    for it in range(STEPS):
        y, kk, inout = batch()
        loss, logs = lossfn(head, y, kk, inout)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0 or it == STEPS - 1:
            print(f'  [{tag}] it {it} loss {loss.item():.4f} {logs}  {time.time() - t0:.0f}s', flush=True)
    return head.eval()


head = fit(GR.RayHead().to(dev), lambda h, y, kk, io: h.loss(toks[kk], y, CENTERS[kk], io), 'ray')
flow = fit(GM.FlowHead().to(dev), lambda h, y, kk, io: h.loss(toks[kk][io], y[io]), 'flow2d')
N = 20000
gen = torch.Generator(device=dev).manual_seed(7)
S, pout = head.sample(toks, N, CENTERS, gen=gen, return_pout=True)
S = S.cpu().numpy()
gen = torch.Generator(device=dev).manual_seed(7)
S_again = head.sample(toks, N, CENTERS, gen=gen).cpu().numpy()
check('同种子两次采样逐位相同', np.array_equal(S, S_again))
gen = torch.Generator(device=dev).manual_seed(8)
S720 = head.sample(toks, N, CENTERS, gen=gen, A=720).cpu().numpy()
S2 = flow.sample(toks, N, gen=gen, steps=16, gamma=3.0, solver='heun').cpu().numpy()


def ahist(P, k, b=72):
    v = P - CENTERS[k].cpu().numpy()
    a = np.arctan2(v[:, 1], v[:, 0])
    h = np.bincount(np.clip(((a + np.pi) / (2 * np.pi) * b).astype(int), 0, b - 1), minlength=b).astype(float)
    return h / h.sum()


for k in range(3):
    T_big, T_true2 = draw(k, 400000, rng), draw(k, N, rng)
    tv = 0.5 * np.abs(hist(S[k]) - hist(T_big)).sum()
    floor = 0.5 * np.abs(hist(T_true2) - hist(T_big)).sum()
    tv2 = 0.5 * np.abs(hist(S2[k]) - hist(T_big)).sum()
    tva = 0.5 * np.abs(ahist(S[k], k) - ahist(T_big, k)).sum()
    tva2 = 0.5 * np.abs(ahist(S2[k], k) - ahist(T_big, k)).sum()
    tva0 = 0.5 * np.abs(ahist(T_true2, k) - ahist(T_big, k)).sum()
    check(f'实例 {k} TV < 0.05', tv < 0.05, f'射线头 TV {tv:.4f}｜2D flow 参照 {tv2:.4f}｜噪声底 {floor:.4f}｜方向边缘 TV 射线头 {tva:.4f} / 2D flow {tva2:.4f} / 噪声底 {tva0:.4f}')
    tvA = 0.5 * np.abs(hist(S[k]) - hist(S720[k])).sum()
    check(f'实例 {k} A=360 vs 720 TV < 0.03', tvA < 0.03, f'{tvA:.4f}')
near = (np.hypot(*(S[0] - np.array([0.45, 0.5])).T) < 0.06).mean()
far = (np.hypot(*(S[0] - np.array([0.85, 0.5])).T) < 0.06).mean()
check('实例 0 近峰质量 0.6±0.05（同一射线上的两档）', abs(near - 0.6) < 0.05, f'近 {near:.3f} 远 {far:.3f}')
po = pout.cpu().numpy()
check('实例 2 P(out) 0.30±0.05', abs(po[2] - 0.3) < 0.05, f'{po.round(4).tolist()}')
check('实例 0/1 P(out) < 0.03', po[0] < 0.03 and po[1] < 0.03)

# 4 计时
big = torch.randn(8, GR.N_SCENE + 256, 256, device=dev)  # 计时用真实 token 数（场景 + 眼部）
cb = torch.rand(8, 2, device=dev)
torch.cuda.synchronize()
t1 = time.time()
head.sample(big, 10000, cb, gen=gen)
torch.cuda.synchronize()
print(f'计时：B=8、10000 样本，{(time.time() - t1) / 8 * 1000:.0f} ms/实例；头参数 {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M', flush=True)

print('\n全部通过' if not fails else f'\n失败 {len(fails)} 项：{fails}')
sys.exit(1 if fails else 0)
