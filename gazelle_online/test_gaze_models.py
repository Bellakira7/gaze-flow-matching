"""gaze_models 自检（单卡，不依赖图像/backbone）。
CUDA_VISIBLE_DEVICES=3 python -I -u test_gaze_models.py

1. cell_centers 与 setmetrics/metrics.py 的 CELLS 逐元素一致；
2. 合成条件分布：2 个"实例"（两组固定随机 token），实例 0 = 双峰混合，实例 1 = 单峰；
   各头只训头（token 固定），训完比较采样直方图与真实分布的总变差距离 TV（16x16 分箱），期望 < 0.04（噪声底约 0.01）；
   MCL 不是分布模型：检查两个峰各有一个假设落在 0.05 内、单峰实例的 5 个假设都在峰附近；
3. MDN 的 pmf 与它自己的抽样直方图一致（TV < 0.05）。
"""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gazelle_online'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'setmetrics'))
import gaze_models as GM  # noqa: E402
import metrics as M  # noqa: E402

dev = 'cuda'
torch.manual_seed(0)
fails = []


def check(name, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not ok:
        fails.append(name)


# 1
c = GM.cell_centers().numpy()
check('cell_centers == metrics.CELLS', np.array_equal(c.astype(np.float64).round(9), np.asarray(M.CELLS).round(9)), f'max diff {np.abs(c - M.CELLS).max():.2e}')

# 2 合成目标
MODES = [  # (权重, 中心, σ)
    [(0.6, (0.3, 0.3), 0.03), (0.4, (0.7, 0.6), 0.05)],
    [(1.0, (0.5, 0.8), 0.04)],
]


def draw(inst, n, rng):
    comps = MODES[inst]
    w = np.array([m[0] for m in comps])
    k = rng.choice(len(comps), n, p=w / w.sum())
    mu = np.array([comps[i][1] for i in k])
    sg = np.array([comps[i][2] for i in k])[:, None]
    return np.clip(mu + sg * rng.standard_normal((n, 2)), 0, 1)


def hist16(p):
    h, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=16, range=[[0, 1], [0, 1]])
    return h / h.sum()


rng = np.random.default_rng(0)
tokens = torch.randn(2, 1024, 256, device=dev)  # 固定的"场景 token"，两个实例互不相同
tok_hold = tokens.clone()
true_h = [hist16(draw(i, 400000, rng)) for i in range(2)]
REP = 64  # 每步每个实例重复的样本数


def train_head(head, steps, lr=1e-3):
    head = head.to(dev).train()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        ys = np.stack([draw(i, REP, rng) for i in range(2)])  # [2,REP,2]
        tk = tokens[:, None].expand(-1, REP, -1, -1).reshape(2 * REP, 1024, 256)
        y = torch.from_numpy(ys.reshape(2 * REP, 2)).float().to(dev)
        loss, logs = head.loss(tk, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()
        sched.step()
        if s % 500 == 0 or s == steps - 1:
            print(f'   [{head.__class__.__name__}] step {s} loss {loss.item():.4f} {logs}', flush=True)
    return head.eval()


def tv_of(samples):  # samples [2,n,2] torch
    s = samples.cpu().numpy()
    return [0.5 * np.abs(hist16(s[i]) - true_h[i]).sum() for i in range(2)]


for name, mk, steps in [('mdn', lambda: GM.MDNHead(M=8), 2500), ('flow', lambda: GM.FlowHead(), 4000)]:
    head = train_head(mk(), steps)
    with torch.no_grad():
        S = head.sample(tok_hold, 40000)
    check(f'{name}: 采样值域 [0,1]', bool(S.min() >= 0 and S.max() <= 1))
    tv = tv_of(S)
    check(f'{name}: 双峰实例 TV<0.04', tv[0] < 0.04, f'TV={tv[0]:.4f}')
    check(f'{name}: 单峰实例 TV<0.04', tv[1] < 0.04, f'TV={tv[1]:.4f}')
    if name == 'mdn':
        with torch.no_grad():
            P = head.pmf(tok_hold).cpu().numpy().reshape(2, 64, 64)
        P16 = P.reshape(2, 16, 4, 16, 4).sum((2, 4))  # [inst, iy, ix]
        s = S.cpu().numpy()
        tvs = [0.5 * np.abs(hist16(s[i]).T - P16[i]).sum() for i in range(2)]  # hist16 的轴序是 (x,y)
        check('mdn: pmf 与自身抽样一致 TV<0.05', max(tvs) < 0.05, f'TV={tvs[0]:.4f}/{tvs[1]:.4f}')
    if name == 'flow':
        for st, sv, gm in ((16, 'euler', 1.0), (8, 'heun', 2.0), (32, 'heun', 2.0)):
            with torch.no_grad():
                Sx = head.sample(tok_hold, 40000, steps=st, solver=sv, gamma=gm)
            print(f'   flow {sv} γ={gm} steps={st}: TV {tv_of(Sx)[0]:.4f} / {tv_of(Sx)[1]:.4f}', flush=True)

head = train_head(GM.MCLHead(K=5), 2500)
with torch.no_grad():
    hyp, w = head.predict(tok_hold)
hyp, w = hyp.cpu().numpy(), w.cpu().numpy()
print('   MCL 实例0 假设', hyp[0].round(3).tolist(), 'w', w[0].round(3).tolist())
print('   MCL 实例1 假设', hyp[1].round(3).tolist(), 'w', w[1].round(3).tolist())
for ci, (cx, cy) in enumerate([(0.3, 0.3), (0.7, 0.6)]):
    d = np.hypot(hyp[0, :, 0] - cx, hyp[0, :, 1] - cy).min()
    check(f'mcl: 实例0 峰{ci} 有假设在 0.05 内', d < 0.05, f'最近 {d:.3f}')
d1 = np.hypot(hyp[1, :, 0] - 0.5, hyp[1, :, 1] - 0.8).max()
check('mcl: 实例1(单峰) 5 个假设都在峰 0.1 内', d1 < 0.1, f'最远 {d1:.3f}')
check('mcl: w 降序且和为 1', bool((np.diff(w, axis=1) <= 1e-7).all() and np.allclose(w.sum(1), 1, atol=1e-5)))

print('\n', 'ALL PASS' if not fails else f'FAILED: {fails}')
sys.exit(1 if fails else 0)
