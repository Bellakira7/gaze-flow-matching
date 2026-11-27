"""由热图模型（B1/B2）生成 G2 bundle，口径与 g2_eval.build_official_bundle 逐项一致（那个函数写死了官方热图，这里参数化；g2_eval.py 冻结不改）。

python -I -u g2_bundles.py --dump results/eval_ckpt/b1_dinov2l_sd_s0_ep14_gt.npz --tag b1_s0
输出 results/g2/bundle_B1_<tag>.npz、bundle_B2_<tag>.npz，并与官方 ckpt 的参照 bundle 并排打印。

B1：假设 = 热图 NMS 前 5 峰；pmf 两种变体 raw(γ=1) 与 cf(γ 在 S1 的 10 个取值里 2 折交叉拟合)；两种 pmf 各抽 10000 个样本走预注册 §3.2 管线取 NLL。
B2：同一网络，假设 = 从 pmf 独立抽 5 个点（S1 采样臂，10 次重复取均值，种子 [s,i]）；raw/cf 里取全体 cov 更高的那个（对 baseline 有利）；NLL 与 B1 相同（同一 pmf）。
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402
import g2_eval as E  # noqa: E402

G = E.G_
GAMMAS = [0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16]  # 与 run_gazelle_baseline.py 相同
REPS = 10


def gamma_crossfit(H, inst, fidx, F):
    mat = np.array([[M.nll(M.heat_to_pmf(H[i], g), inst[i]) for g in GAMMAS] for i in range(len(inst))])
    fold = E.fold_of(fidx, F)
    gam = np.zeros(len(inst))
    chosen = {}
    for f in (0, 1):
        k = int(np.argmin(mat[fold != f].mean(0)))
        gam[fold == f] = GAMMAS[k]
        chosen[f] = GAMMAS[k]
    return gam, chosen


def build(H, per_instance, out_dir, tag):
    E._check_prereg()
    inst, fidx, hw, subsets, F = E.load_context()
    N = len(inst)
    assert H.shape[0] == N and per_instance.shape[0] == N
    gam, chosen = gamma_crossfit(H, inst, fidx, F)
    print(f'[{tag}] γ 交叉拟合: {chosen}', flush=True)
    nll_v, exact_v, pm = [], [], {}
    for name, g, seed in (('raw', np.ones(N), 31), ('cf', gam, 32)):
        S = np.zeros((N, E.N_SAMP, 2), np.float32)
        ex = np.zeros(N)
        pm[name] = [M.heat_to_pmf(H[i], g[i]) for i in range(N)]
        for i in range(N):
            S[i] = M.sample_hyps(pm[name][i], np.random.default_rng([seed, i]), E.N_SAMP)
            ex[i] = M.nll(pm[name][i], inst[i])
        nl, _, ch = E.crossfit_nll(S.astype(float), inst, fidx, F)
        print(f'[{tag}/{name}] 样本管线 NLL {nl.mean():.4f}（精确 {ex.mean():.4f}），σ_s={ch}', flush=True)
        nll_v.append(nl)
        exact_v.append(ex)
    # B1：NMS 前 5 峰
    cov, mP, mR = np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(N):
        pk = M.nms_peaks(H[i], 5)
        Hn = M.CELLS[pk]
        cov[i] = M.hyp_metrics(Hn, inst[i], ks=(5,))['cov0.1@5']
        w = np.array([M.disc_mass(pm['raw'][i], idx) for idx in pk])
        mm = M.mode_metrics(Hn, w, M.gt_modes(inst[i]), ks=(3,))
        mP[i], mR[i] = mm['modeP@3'], mm['modeR@3']
    b1 = {'cov0.1@5': cov, 'nll_variants': np.stack(nll_v, 1), 'nll_variant_names': np.array(['raw', 'cf']), 'nll_exact_variants': np.stack(exact_v, 1),
          'auc': per_instance[:, 0], 'avg_l2_fix': per_instance[:, 3], 'min_l2_fix': per_instance[:, 4], 'modeP@3': mP, 'modeR@3': mR}
    # B2：独立采样 5 点，10 次重复取均值
    covs, avgs, mins = {}, {}, {}
    for name in ('raw', 'cf'):
        c, a, m_ = np.zeros(N), np.zeros(N), np.zeros(N)
        for i in range(N):
            cs, as_, ms = [], [], []
            for s in range(REPS):
                hy = M.sample_hyps(pm[name][i], np.random.default_rng([s, i]), 10)[:5]
                cs.append(M.hyp_metrics(hy, inst[i], ks=(5,))['cov0.1@5'])
                as_.append(np.hypot(*(hy[0] - inst[i].mean(0))))
                ms.append(np.hypot(*(inst[i] - hy[0]).T).min())
            c[i], a[i], m_[i] = np.mean(cs), np.mean(as_), np.mean(ms)
        covs[name], avgs[name], mins[name] = c, a, m_
    pick = max(covs, key=lambda k: covs[k].mean())
    b2 = {'cov0.1@5': covs[pick], 'cov_variant': np.array(pick), 'nll_variants': b1['nll_variants'], 'nll_variant_names': b1['nll_variant_names'],
          'auc': per_instance[:, 0], 'avg_l2_fix': avgs[pick], 'min_l2_fix': mins[pick]}
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f'{out_dir}/bundle_B1_{tag}.npz', **b1)
    np.savez(f'{out_dir}/bundle_B2_{tag}.npz', **b2)
    return b1, b2, subsets


def report(b, subsets, label):
    V = np.asarray(b['nll_variants'], float)
    names = [str(x) for x in b['nll_variant_names']]
    for sn, sel in subsets.items():
        s = f'{label:18s} [{sn:5s}] cov0.10@5 {np.asarray(b["cov0.1@5"], float)[sel].mean():.4f}  NLL ' + ' '.join(f'{n}={V[sel, k].mean():.4f}' for k, n in enumerate(names))
        if sn == 'all':
            s += f'  AUC {np.mean(b["auc"]):.4f} AvgL2 {np.mean(b["avg_l2_fix"]):.4f} MinL2 {np.mean(b["min_l2_fix"]):.4f}'
        print(s, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True, help='common.evaluate(dump=...) 存的 npz：heatmaps / meta / per_instance')
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    d = np.load(a.dump if os.path.isabs(a.dump) else f'{G}/{a.dump}')
    H = d['heatmaps'].astype(np.float32)
    b1, b2, subsets = build(H, d['per_instance'], f'{G}/results/g2', a.tag)
    print('---- 对照 ----')
    off = E.official_bundle()
    report(off, subsets, 'official B1')
    report(b1, subsets, f'{a.tag} B1')
    report(b2, subsets, f'{a.tag} B2({b2["cov_variant"]})')
