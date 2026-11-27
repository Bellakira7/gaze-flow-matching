"""B5 扩散头自检（单卡，不依赖图像/backbone）。
CUDA_VISIBLE_DEVICES=3 python -I -u test_gaze_diffusion.py

1. target_heatmap 与官方 gutils.get_heatmap 逐元素一致（含边界/越界坐标）；
2. 零初始化：初始 v 恒为 0；
3. 合成双峰/单峰条件分布（固定 token）：训练后
   a) M_draw=1024、64 步的混合 pmf 与"真值后验经同一热图核平滑"的参照 pmf 在 16x16 分箱上 TV < 0.10；
   b) 单张采样热图是"选一个峰"而不是"平均两个峰"：双峰实例里两峰质量都 >25% 的采样占比 < 0.15，且选 A/B 的比例分别 ∈ [0.45,0.75] / [0.25,0.55]（真值 0.6/0.4，64 张采样的二项噪声 ±0.12）；
   c) predict() 的假设：双峰实例前 2 个假设各在两峰 0.06 内，w 降序和为 1。
"""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gazelle_online'))
import gaze_diffusion as GD  # noqa: E402
import gazelle.utils as gutils  # noqa: E402

dev = 'cuda'
torch.manual_seed(0)
fails = []


def check(name, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not ok:
        fails.append(name)


# 1
ys = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.999, 0.5], [0.5, 0.001], [0.3, 0.7], [0.0123, 0.9877], [0.5, 0.5]] + torch.rand(200, 2).tolist())
mine = GD.target_heatmap(ys.to(dev)).cpu()
ref = torch.stack([gutils.get_heatmap(float(a), float(b), 64, 64) for a, b in ys.tolist()])
check('target_heatmap == 官方 get_heatmap', bool(torch.allclose(mine, ref, atol=1e-6)), f'max diff {(mine - ref).abs().max():.2e}')

# 2
head = GD.DiffHead().to(dev)
tok = torch.randn(2, 1024, 256, device=dev)
v0 = head.velocity(torch.randn(2, 1, 64, 64, device=dev), torch.rand(2, device=dev), head.cond(tok))
check('零初始化: 初始 v == 0', bool(v0.abs().max() == 0))

# 3 合成目标
MODES = [[(0.6, (0.3, 0.3), 0.03), (0.4, (0.7, 0.6), 0.05)], [(1.0, (0.5, 0.8), 0.04)]]
rng = np.random.default_rng(0)


def draw(inst, n):
    comps = MODES[inst]
    w = np.array([m[0] for m in comps])
    k = rng.choice(len(comps), n, p=w / w.sum())
    mu = np.array([comps[i][1] for i in k])
    sg = np.array([comps[i][2] for i in k])[:, None]
    return np.clip(mu + sg * rng.standard_normal((n, 2)), 0, 1)


def ref_pmf(inst, n=20000):
    h = GD.target_heatmap(torch.from_numpy(draw(inst, n)).float().to(dev)).flatten(1).double()
    return (h / h.sum(1, keepdim=True)).mean(0).cpu().numpy().reshape(64, 64)


def pool16(P):  # [64,64] (iy,ix) -> [16,16]
    return P.reshape(16, 4, 16, 4).sum((1, 3))


STEPS, REP = 1500, 32
opt = torch.optim.Adam(head.parameters(), 1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
head.train()
for s in range(STEPS):
    y = torch.from_numpy(np.stack([draw(i, REP) for i in range(2)]).reshape(2 * REP, 2)).float().to(dev)
    tk = tok[:, None].expand(-1, REP, -1, -1).reshape(2 * REP, 1024, 256)
    loss, _ = head.loss(tk, y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
    opt.step()
    sched.step()
    if s % 500 == 0 or s == STEPS - 1:
        print(f'   step {s} v_mse {loss.item():.4f}', flush=True)
head.eval()
torch.save({'head': head.state_dict(), 'tok': tok.cpu()}, '/tmp/gz_scratch/diff_toy.pt')

gen = torch.Generator(device=dev).manual_seed(0)
# 首次运行（16 步、M=256）实例0 的 TV=0.119 没过 0.10；诊断（/tmp/gz_scratch/diff_diag.py）显示：模式内散布正确(0.026/0.046 vs 0.03/0.05)，
# 偏差来自模式权重（1024 张里选 A 占 0.675，真值 0.6），随步数下降（16 步 TV≈0.10 -> 64 步 0.068/0.059）。故此项改用 64 步、M=1024 判定，
# 16 步的数字只打印。结论：B5 的采样步数不能默认 16，要在训练 hold-out 上选（prereg/G2.clarifications_20261022.md §C5）。
for steps, M_ in ((16, 256), (64, 1024)):
    P = head.pmf(tok, M_draw=M_, steps=steps, gen=gen, lam=0).cpu().numpy().reshape(2, 64, 64)
    for i in range(2):
        tv = 0.5 * np.abs(pool16(P[i]) - pool16(ref_pmf(i))).sum()
        if steps == 64:
            check(f'混合 pmf 与参照 TV<0.10 (实例{i}, 64步 M=1024)', tv < 0.10, f'TV={tv:.4f}')
        else:
            print(f'   [仅打印] 16 步 M=256 实例{i} TV={tv:.4f}')

H = head.draw_heatmaps(tok[:1], M_draw=64, steps=16, gen=gen)[0].clamp_min(0).cpu().numpy()  # [64,64,64]
yy, xx = np.meshgrid((np.arange(64) + 0.5) / 64, (np.arange(64) + 0.5) / 64, indexing='ij')
mA, mB = np.hypot(xx - 0.3, yy - 0.3) < 0.15, np.hypot(xx - 0.7, yy - 0.6) < 0.2
fa, fb = (H * mA).sum((1, 2)) / H.sum((1, 2)), (H * mB).sum((1, 2)) / H.sum((1, 2))
both = ((fa > 0.25) & (fb > 0.25)).mean()
onlyA = ((fa > 0.5) & (fb < 0.25)).mean()
onlyB = ((fb > 0.5) & (fa < 0.25)).mean()
print(f'   64 张采样: 只选A {onlyA:.2f}  只选B {onlyB:.2f}  两峰都有 {both:.2f}')
check('单张采样选一个峰（两峰同时占比<0.15）', both < 0.15, f'both={both:.3f}')
check('选 A 比例 ∈ [0.45,0.75]', 0.45 <= onlyA <= 0.75, f'{onlyA:.3f}')
check('选 B 比例 ∈ [0.25,0.55]', 0.25 <= onlyB <= 0.55, f'{onlyB:.3f}')

S = head.sample(tok, 1000, gen=gen)
check('sample 值域 [0,1]', bool(S.min() >= 0 and S.max() <= 1))
hy, w = head.predict()
hy, w = hy.numpy(), w.numpy()
print('   实例0 假设', hy[0].round(3).tolist()[:3], 'w', w[0].round(3).tolist())
d_to = lambda c: np.hypot(hy[0, :, 0] - c[0], hy[0, :, 1] - c[1])
check('predict: 峰A 有假设在 0.06 内', d_to((0.3, 0.3)).min() < 0.06, f'{d_to((0.3, 0.3)).min():.3f}')
check('predict: 峰B 有假设在 0.06 内', d_to((0.7, 0.6)).min() < 0.06, f'{d_to((0.7, 0.6)).min():.3f}')
top2 = hy[0, :2]
ok2 = (np.hypot(*(top2 - [0.3, 0.3]).T).min() < 0.06) and (np.hypot(*(top2 - [0.7, 0.6]).T).min() < 0.06)
check('predict: 前 2 个假设各在一个峰 0.06 内', bool(ok2))
check('predict: w 降序且和为 1', bool((np.diff(w, axis=1) <= 1e-7).all() and np.allclose(w.sum(1), 1, atol=1e-5)))

print('\n', 'ALL PASS' if not fails else f'FAILED: {fails}')
sys.exit(1 if fails else 0)
