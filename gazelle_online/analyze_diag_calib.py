"""汇总 diag_calib_holdout.py 的输出（只用 hold-out）。python -I analyze_diag_calib.py
1. NLL（σ_s 取各配置最优）与对 d6_t1 的逐实例配对差（bootstrap B=2000）；
2. HPD 校准：q 网格上 P(hpd_m ≤ q) 与 q 的差（负 = 标注落在高密度区外的比例超过预期 = 过度自信）；
3. split-conformal（实例随机对半，种子 0）：标定半区按 hpd_m 的 ⌈(n+1)(1−α)⌉/n 分位定质量阈 q̂，评估半区报覆盖率与平均区域面积；
   并按"模型自报的分散度"（q=0.5 的 HPD 面积）在标定半区分 4 档，报全局阈值下各档覆盖率，以及各档单独标定（Mondrian）后的覆盖率 / 面积。
"""
from pathlib import Path
import glob

import numpy as np

D = str(Path(__file__).resolve().parents[1] / 'experiments/diag_calib')
fs = sorted(f for f in glob.glob(f'{D}/shard*of4.npz') if 'smoke' not in f)
P = [np.load(f) for f in fs]
idx = np.concatenate([p['idx'] for p in P])
o = np.argsort(idx)
assert len(idx) == 3762, len(idx)
Q = P[0]['qgrid']
cfgs = sorted({k.split('__')[0] for k in P[0].files if '__' in k})
R = {c: {k: np.concatenate([p[f'{c}__{k}'] for p in P])[o] for k in ('nll', 'hpd_m', 'area')} for c in cfgs}
rng = np.random.default_rng(0)
N = len(idx)
BI = rng.integers(0, N, (2000, N))

print('== 1. NLL（σ_s 取最优）与对 d6_t1 的配对差')
base = R['d6_t1']['nll'][:, int(np.argmin(R['d6_t1']['nll'].mean(0)))]
for c in cfgs:
    m = R[c]['nll'].mean(0)
    j = int(np.argmin(m))
    d = R[c]['nll'][:, j] - base
    bs = d[BI].mean(1)
    print(f'  {c:16s} NLL {m[j]:.4f}（σ_s={(1, 2, 3)[j]}）  Δ {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]')

print('\n== 2. HPD 校准：P(hpd_m ≤ q) − q（负 = 过度自信）')
show = [0.2, 0.5, 0.8, 0.9, 0.95]
print('  ' + ' ' * 16 + ''.join(f'  q={q:<5g}' for q in show) + '   平均|差|  平均差')
for c in cfgs:
    h = R[c]['hpd_m']
    dev = np.array([(h <= q).mean() - q for q in Q])
    print(f'  {c:16s}' + ''.join(f'  {(h <= q).mean() - q:+.3f} ' for q in show) + f'   {np.abs(dev).mean():.3f}   {dev.mean():+.3f}')

print('\n== 3. split-conformal（评估半区）')
perm = rng.permutation(N)
cal, ev = perm[: N // 2], perm[N // 2:]


def area_at(area_row, q):
    return np.interp(q, Q, area_row)


for c in ('d6_t1', 'd6_t1.2', 'flowET_t1', 'mix_d6_flowET'):
    if c not in R:
        continue
    h, A = R[c]['hpd_m'], R[c]['area']
    spread = A[:, list(np.round(Q, 2)).index(0.5)]
    edges = np.quantile(spread[cal], [0.25, 0.5, 0.75])
    bin_of = np.digitize(spread, edges)
    print(f'  [{c}]')
    for alpha in (0.5, 0.2, 0.1):
        n = len(cal)
        qhat = np.sort(h[cal])[min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)]
        qhat_c = min(qhat, Q[-1])
        cov = (h[ev] <= qhat).mean()
        area = np.mean([area_at(A[i], qhat_c) for i in ev])
        naive_cov = (h[ev] <= 1 - alpha).mean()
        naive_area = np.mean([area_at(A[i], 1 - alpha) for i in ev])
        per_bin = [(h[ev][bin_of[ev] == b] <= qhat).mean() for b in range(4)]
        mond_cov, mond_area = [], []
        for b in range(4):
            cb, eb = cal[bin_of[cal] == b], ev[bin_of[ev] == b]
            nb = len(cb)
            qb = np.sort(h[cb])[min(nb - 1, int(np.ceil((nb + 1) * (1 - alpha))) - 1)]
            mond_cov.append((h[eb] <= qb).mean())
            mond_area.append(np.mean([area_at(A[i], min(qb, Q[-1])) for i in eb]))
        print(f'    α={alpha:<4g} 未校准(q=1−α) 覆盖 {naive_cov:.3f} 面积 {naive_area:.4f} | conformal q̂={qhat:.3f} 覆盖 {cov:.3f} 面积 {area:.4f}'
              f' | 全局 q̂ 下按分散度 4 档覆盖 {" ".join(f"{v:.3f}" for v in per_bin)}'
              f' | Mondrian 各档覆盖 {" ".join(f"{v:.3f}" for v in mond_cov)} 平均面积 {np.mean(mond_area):.4f}')
    print('    （q̂>0.95 时面积按 q=0.95 截断，偏小）')
