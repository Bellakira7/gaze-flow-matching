"""协议 S1 §6 的自检：全部通过才允许读数。python -I test_metrics.py"""
from pathlib import Path
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402

G_ = str(Path(__file__).resolve().parents[1])
ok = []


def check(name, cond, info=''):
    ok.append(bool(cond))
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{info}]' if info else ''), flush=True)


# 1. 均匀 pmf 的 NLL = ln 4096
P_u = M.heat_to_pmf(np.ones((64, 64)))
A = np.random.default_rng(0).random((10, 2))
check('uniform NLL == ln 4096', abs(M.nll(P_u, A) - np.log(4096)) < 1e-9, f'{M.nll(P_u, A):.6f}')

# 2. W1：delta-delta、手算两点
c0 = M.CELLS[100]
P = np.zeros(4096); P[100] = 1.0
d = np.hypot(*(c0 - np.array([0.9, 0.9])))
check('W1 delta-delta == distance', abs(M.w1_grid(P, np.tile([[0.9, 0.9]], (10, 1))) - d) < 1e-6)
A2 = np.array([c0, c0 + [0.2, 0.0]])
check('W1 hand case == 0.1', abs(M.w1_grid(P, A2) - 0.1) < 1e-6, f'{M.w1_grid(P, A2):.6f}')
check('W1 discrete identical == 0', M.w1_discrete(A, A) < 1e-9)
check('split-half W1 of coincident points == 0', M.split_half_w1(np.tile([[0.3, 0.4]], (10, 1)), np.random.default_rng(0)) < 1e-9)

# 3. 手算假设指标
hm = M.hyp_metrics(np.array([[0.5, 0.5]]), np.array([[0.5, 0.5], [0.7, 0.5]]))
check('hyp metrics hand case', abs(hm['minL2@1']) < 1e-12 and abs(hm['avgL2@1'] - 0.1) < 1e-12 and abs(hm['cham@1'] - 0.1) < 1e-12
      and abs(hm['cov0.05@1'] - 0.5) < 1e-12 and abs(hm['cov0.1@1'] - 0.5) < 1e-12, str({k: round(v, 3) for k, v in hm.items() if k.endswith('@1')}))

# 4. 单调性：minL2@K 不增、cov@K 不减
rng = np.random.default_rng(1)
mono = True
for _ in range(200):
    Hh = rng.random((10, 2)); Aa = rng.random((rng.integers(6, 15), 2))
    r = M.hyp_metrics(Hh, Aa)
    for k0, k1 in zip(M.KS[:-1], M.KS[1:]):
        mono &= r[f'minL2@{k1}'] <= r[f'minL2@{k0}'] + 1e-12 and r[f'cov0.1@{k1}'] >= r[f'cov0.1@{k0}'] - 1e-12 and r[f'cham@{k1}'] <= r[f'cham@{k0}'] + 1e-12
check('minL2/cham non-increasing, cov non-decreasing in K', mono)

# 5. mode 精度/召回
two = np.concatenate([np.array([0.2, 0.2]) + rng.normal(0, 0.01, (5, 2)), np.array([0.8, 0.7]) + rng.normal(0, 0.01, (5, 2))])
modes = M.gt_modes(two)
check('gt_modes finds 2 modes', len(modes) == 2, f'{len(modes)}')
check('gt_modes fallback: scattered -> 1 median mode', len(M.gt_modes(np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9], [0.5, 0.5], [0.3, 0.7]]))) == 1)
p, r = M.mode_pr(modes, np.array([0.5, 0.5]), modes, 3)
check('hyps = GT modes -> P=R=1', p == 1.0 and r == 1.0, f'{p},{r}')
p, r = M.mode_pr(modes[:1], np.array([1.0]), modes, 1)
check('one of two modes -> P=1, R=0.5', p == 1.0 and r == 0.5, f'{p},{r}')
p, r = M.mode_pr(np.array([[0.5, 0.9]]), np.array([1.0]), modes, 1)
check('far hypothesis -> P=R=0', p == 0.0 and r == 0.0)
p, r = M.mode_pr(np.array([modes[0], modes[0] + 0.01]), np.array([0.5, 0.5]), modes[:1], 2)
check('duplicate hypotheses penalised: P=0.5, R=1', abs(p - 0.5) < 1e-12 and r == 1.0, f'{p},{r}')

# 6. NMS 取峰
CX, CY = M.CELLS[:, 0].reshape(64, 64), M.CELLS[:, 1].reshape(64, 64)
hh = np.exp(-((CX - 0.25) ** 2 + (CY - 0.3) ** 2) / (2 * 0.03 ** 2)) + 0.6 * np.exp(-((CX - 0.7) ** 2 + (CY - 0.6) ** 2) / (2 * 0.03 ** 2))
pk = M.nms_peaks(hh, 5)
pts = M.CELLS[pk]
pair_ok = all(np.hypot(*(pts[i] - pts[j])) >= 0.10 for i in range(5) for j in range(i))
check('nms: top-2 peaks at bumps, pairwise >= 0.10',
      np.hypot(*(pts[0] - [0.25, 0.3])) < 0.02 and np.hypot(*(pts[1] - [0.7, 0.6])) < 0.02 and pair_ok, f'{pts[:2].round(3).tolist()}')

# 7. disc-ECE：真实分布 ≈ 0；过锐的 pmf 过度自信
S = 3000
pred_t, obs_t, pred_s, obs_s = [], [], [], []
rng = np.random.default_rng(2)
for _ in range(S):
    k = rng.integers(1, 3)
    cen = rng.random((k, 2)) * 0.6 + 0.2
    sg = rng.uniform(0.03, 0.08, k)
    w = rng.dirichlet(np.ones(k))
    dens = sum(wi * np.exp(-((CX - c[0]) ** 2 + (CY - c[1]) ** 2) / (2 * s ** 2)) / s ** 2 for wi, c, s in zip(w, cen, sg))
    Pt = (dens / dens.sum()).ravel()
    An = M.sample_hyps(Pt, rng, 10)
    pk = M.nms_peaks(Pt.reshape(64, 64), 3)
    a, b = M.disc_pairs(Pt, An, pk); pred_t += list(a); obs_t += list(b)
    Ps = Pt ** 8; Ps /= Ps.sum()
    a, b = M.disc_pairs(Ps, An, pk); pred_s += list(a); obs_s += list(b)
e_t, g_t = M.ece(pred_t, obs_t)
e_s, g_s = M.ece(pred_s, obs_s)
check('disc-ECE ~0 when pmf is the truth', e_t < 0.03, f'ECE {e_t:.4f}, signed gap {g_t:+.4f}')
check('disc-ECE flags over-confident pmf (gap>0.05)', g_s > 0.05 and e_s > 3 * e_t, f'ECE {e_s:.4f}, signed gap {g_s:+.4f}')

# 8. LOO-KDE 比均匀好
cl = np.array([0.4, 0.5]) + np.random.default_rng(3).normal(0, 0.03, (10, 2))
check('LOO-KDE NLL < ln 4096 on a tight cluster', M.loo_kde_nll(cl, 0.03) < np.log(4096) - 1.0, f'{M.loo_kde_nll(cl, 0.03):.3f}')

# 9. 真实数据：K=1 的 argmax 与 G1 一致
frames = json.load(open(f'{G_}/data/gazefollow_test_hf/test_preprocessed.json'))
inst = [np.stack([h['gazex_norm'], h['gazey_norm']], 1) for fr in frames for h in fr['heads']]
Hm = np.load(f'{G_}/results/repro_test/full.npz')['heatmaps'].astype(np.float32)
mn, av = [], []
for i, Ai in enumerate(inst):
    pk = M.nms_peaks(Hm[i], 1)
    r = M.hyp_metrics(M.CELLS[pk], Ai, ks=(1,))
    mn.append(r['minL2@1']); av.append(r['avgL2@1'])
check('real data K=1: Avg L2 == 0.0969 (G1)', abs(np.mean(av) - 0.0969) < 1e-3, f'{np.mean(av):.5f}')
check('real data K=1: Min L2 == 0.0400 (G1)', abs(np.mean(mn) - 0.0400) < 1e-3, f'{np.mean(mn):.5f}')

print(f'\n{sum(ok)}/{len(ok)} passed')
sys.exit(0 if all(ok) else 1)
