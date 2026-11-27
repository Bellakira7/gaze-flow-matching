"""gaze_mixflow 自检（单卡，不依赖图像/backbone）。
CUDA_VISIBLE_DEVICES=3 python -I -u test_gaze_mixflow.py

1. 容量：默认配置的头参数量与 FlowHead(depth=6)（flowET_d6 的头）相差 < 5%；
2. 梯度：锚点作条件时 detach（FM 损失不回传到锚点）；FM 损失能回传到场景 token 与 query（经 embedding）；
3. 合成条件分布（同 test_gaze_models.py：实例 0 = 双峰 0.6/0.4，实例 1 = 单峰），只训头：
   a. 采样直方图与真实分布的 TV（16×16 分箱）< 0.04（flow 基线在同一合成上为噪声底 ~0.01）；
   b. 原生假设（merge=0.10 合并后）：双峰实例的两个峰各有一个假设在 0.05 内，权重与 (0.6, 0.4) 相差 < 0.12、两峰权重和 > 0.85；
      单峰实例权重最大的假设在峰 0.03 内；原始（merge=0）锚点的 π 按最近峰求和同样 ≈ (0.6, 0.4)；
   c. 成分一致性：采样返回的成分编号频率 ≈ π（差 < 0.02）；双峰实例里 ≥ 90% 的样本落在它所属成分的锚点 0.15 内；
   d. 同种子下 return_comp=True 与 False 的样本逐位相同；
4. K=1 退化：能训、采样 TV < 0.05、π ≡ 1；
5. 采样值域 [0,1]，predict 的权重降序且和为 1。
"""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gazelle_online'))
import gaze_models as GM  # noqa: E402
import gaze_mixflow as MX  # noqa: E402
import gaze_diffusion as GD  # noqa: E402

dev = 'cuda'
torch.manual_seed(0)
fails = []


def check(name, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not ok:
        fails.append(name)


# 1 容量
n_mix = sum(p.numel() for p in MX.MixFlowHead().parameters())
n_d6 = sum(p.numel() for p in GM.FlowHead(depth=6).parameters())
check('容量对齐 FlowHead(depth=6)', abs(n_mix / n_d6 - 1) < 0.05, f'mixflow {n_mix / 1e6:.3f}M vs d6 {n_d6 / 1e6:.3f}M')
check('已注册 HEADS / NATIVE_HYPS', 'mixflow' in GM.HEADS and 'mixflow' in GD.NATIVE_HYPS)

# 2 梯度
h = MX.MixFlowHead(K=3).to(dev).train()
tk = torch.randn(4, 1024, 256, device=dev, requires_grad=True)
anc = torch.rand(4, 3, 2, device=dev, requires_grad=True)
emb = torch.randn(4, 3, 256, device=dev, requires_grad=True)
h.comp_cond(emb, anc).sum().backward()
check('锚点作条件时 detach', anc.grad is None and emb.grad is not None and emb.grad.abs().sum() > 0)
h.zero_grad()
y = torch.rand(4, 2, device=dev)
# 让 flow 输出层非零，否则 FM 梯度因零初始化恒为 0
with torch.no_grad():
    h.out[1].weight.normal_(0, 0.02)
loss, logs = h.loss(tk, y)
loss.backward()
check('损失能回传到场景 token', tk.grad is not None and tk.grad.abs().sum() > 0)
check('损失能回传到 query', h.queries.grad is not None and h.queries.grad.abs().sum() > 0)
print('   logs', {k: round(v, 4) for k, v in logs.items()})

# 3 合成目标
MODES = [
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
    h_, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=16, range=[[0, 1], [0, 1]])
    return h_ / h_.sum()


rng = np.random.default_rng(0)
tokens = torch.randn(2, 1024, 256, device=dev)
true_h = [hist16(draw(i, 400000, rng)) for i in range(2)]
REP = 64


def train_head(head, steps, lr=1e-3):
    head = head.to(dev).train()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        ys = np.stack([draw(i, REP, rng) for i in range(2)])
        tk_ = tokens[:, None].expand(-1, REP, -1, -1).reshape(2 * REP, 1024, 256)
        y_ = torch.from_numpy(ys.reshape(2 * REP, 2)).float().to(dev)
        loss_, logs_ = head.loss(tk_, y_)
        opt.zero_grad()
        loss_.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()
        sched.step()
        if s % 500 == 0 or s == steps - 1:
            print(f'   [K={head.K}] step {s} loss {loss_.item():.4f} {({k: round(v, 3) for k, v in logs_.items()})}', flush=True)
    return head.eval()


def tv_of(S):
    s = S.cpu().numpy()
    return [0.5 * np.abs(hist16(s[i]) - true_h[i]).sum() for i in range(2)]


head = train_head(MX.MixFlowHead(K=5), 4000)
with torch.no_grad():
    S, comp = head.sample(tokens, 40000, return_comp=True)
    hyp, w = head.predict(tokens)                 # merge=0.10 合并后
    raw_h, raw_w = head.predict(tokens, merge=0)  # 原始锚点与 π（降序）
    anchors, logit, _ = head.hypotheses(tokens)
check('采样值域 [0,1]', bool(S.min() >= 0 and S.max() <= 1))
tv = tv_of(S)
check('K=5: 双峰实例 TV<0.04', tv[0] < 0.04, f'TV={tv[0]:.4f}')
check('K=5: 单峰实例 TV<0.04', tv[1] < 0.04, f'TV={tv[1]:.4f}')
hyp_np, w_np = hyp.cpu().numpy(), w.cpu().numpy()
print('   实例0 合并后假设', hyp_np[0].round(3).tolist(), 'w', w_np[0].round(3).tolist())
print('   实例1 合并后假设', hyp_np[1].round(3).tolist(), 'w', w_np[1].round(3).tolist())
dA = np.hypot(hyp_np[0, :, 0] - 0.3, hyp_np[0, :, 1] - 0.3)
dB = np.hypot(hyp_np[0, :, 0] - 0.7, hyp_np[0, :, 1] - 0.6)
check('K=5: 双峰实例两个峰各有合并假设在 0.05 内', dA.min() < 0.05 and dB.min() < 0.05, f'最近 {dA.min():.3f}/{dB.min():.3f}')
wA, wB = w_np[0][dA.argmin()], w_np[0][dB.argmin()]
check('K=5: 合并后双峰权重 ≈ (0.6,0.4) 且两峰权重和 > 0.85', abs(wA - 0.6) < 0.12 and abs(wB - 0.4) < 0.12 and wA + wB > 0.85, f'wA={wA:.3f} wB={wB:.3f}')
check('K=5: 单峰实例 top 合并假设在峰 0.03 内', np.hypot(hyp_np[1, 0, 0] - 0.5, hyp_np[1, 0, 1] - 0.8) < 0.03,
      f'{np.hypot(hyp_np[1, 0, 0] - 0.5, hyp_np[1, 0, 1] - 0.8):.3f}')
check('K=5: predict 权重降序且和为 1', bool((np.diff(w_np, axis=1) <= 1e-7).all() and np.allclose(w_np.sum(1), 1, atol=1e-5)))
rh, rw = raw_h.cpu().numpy()[0], raw_w.cpu().numpy()[0]
to_A = np.hypot(rh[:, 0] - 0.3, rh[:, 1] - 0.3) < np.hypot(rh[:, 0] - 0.7, rh[:, 1] - 0.6)
mA, mB = rw[to_A].sum(), rw[~to_A].sum()
check('K=5: 原始锚点 π 按峰求和 ≈ (0.6,0.4)', abs(mA - 0.6) < 0.1 and abs(mB - 0.4) < 0.1, f'{mA:.3f}/{mB:.3f}')
pi = torch.softmax(logit, -1)
freq = torch.stack([torch.bincount(comp[i], minlength=5).float() / comp.shape[1] for i in range(2)])
check('K=5: 采样的成分频率 ≈ π', float((freq - pi).abs().max()) < 0.02, f'max|freq-π|={float((freq - pi).abs().max()):.4f}')
near = (S[0] - anchors[0][comp[0]]).norm(dim=-1) < 0.15
check('K=5: 双峰实例样本落在所属成分的锚点 0.15 内 ≥ 90%', float(near.float().mean()) >= 0.90, f'{float(near.float().mean()):.3f}')
g1, g2 = torch.Generator(device=dev).manual_seed(7), torch.Generator(device=dev).manual_seed(7)
with torch.no_grad():
    Sa = head.sample(tokens, 3000, gen=g1)
    Sb, _ = head.sample(tokens, 3000, gen=g2, return_comp=True)
check('同种子 return_comp 不改变样本', torch.equal(Sa, Sb))

# 4 K=1
head1 = train_head(MX.MixFlowHead(K=1), 3000)
with torch.no_grad():
    S1 = head1.sample(tokens, 40000)
    _, w1 = head1.predict(tokens)
tv1 = tv_of(S1)
check('K=1: 采样 TV<0.05（双峰/单峰）', max(tv1) < 0.05, f'TV={tv1[0]:.4f}/{tv1[1]:.4f}')
check('K=1: π ≡ 1', bool(torch.allclose(w1, torch.ones_like(w1))))

print('\n', 'ALL PASS' if not fails else f'FAILED: {fails}')
sys.exit(1 if fails else 0)
