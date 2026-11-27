"""S1 读数：Gaze-LLE 热图 baseline 的 set-aware 指标。严格按 gaze/prereg/S1.prereg.md。python -I -u run_gazelle_baseline.py"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402

G_ = str(Path(__file__).resolve().parents[1])
OUT = f'{G_}/results/setmetrics'
os.makedirs(OUT, exist_ok=True)
PREREG_SHA = '02bc05a90d79f0ac90c287f7b5d1e396f05f76299fbdd1225c0cbc37101f48f4'
assert hashlib.sha256(open(f'{G_}/prereg/S1.prereg.md', 'rb').read()).hexdigest() == PREREG_SHA, '预注册文件已被改动'

GAMMAS = [0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16]
SIGMAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
REPS = 10
B = 2000

frames = json.load(open(f'{G_}/data/gazefollow_test_hf/test_preprocessed.json'))
inst, frame_idx = [], []
for fi, fr in enumerate(frames):
    for h in fr['heads']:
        inst.append(np.stack([h['gazex_norm'], h['gazey_norm']], 1))
        frame_idx.append(fi)
frame_idx = np.array(frame_idx)
N, F = len(inst), len(frames)
H = np.load(f'{G_}/results/repro_test/full.npz')['heatmaps'].astype(np.float32)
R1 = np.load(f'{G_}/results/g1/per_instance.npz')
assert (R1['n_annot'] == np.array([len(a) for a in inst])).all() and H.shape[0] == N
MULTI, SEP = R1['multi'], R1['sep']
SUBSETS = {'all': np.ones(N, bool), 'unimodal': ~MULTI, 'M_sep': SEP}

# ---------- 阶段 A：γ 的 2 折交叉拟合（折按帧随机划分）----------
t0 = time.time()
nll_mat = np.array([[M.nll(M.heat_to_pmf(H[i], g), inst[i]) for g in GAMMAS] for i in range(N)])
perm = np.random.default_rng(0).permutation(F)
fold_of_frame = np.zeros(F, int)
fold_of_frame[perm[F // 2:]] = 1
fold = fold_of_frame[frame_idx]
best_g = {}
gamma_cf = np.zeros(N)
for f in (0, 1):
    other = fold != f
    gi = int(np.argmin(nll_mat[other].mean(0)))
    best_g[f] = GAMMAS[gi]
    gamma_cf[fold == f] = GAMMAS[gi]
print(f'[A] γ 交叉拟合（用另一折选）: 折0实例用 γ={best_g[0]}（在折1上选），折1实例用 γ={best_g[1]}（在折0上选）；全体 raw(γ=1) NLL {nll_mat[:, 2].mean():.4f}  耗时 {time.time() - t0:.0f}s', flush=True)
print('    各 γ 的全体平均 NLL: ' + '  '.join(f'γ{g}:{nll_mat[:, k].mean():.3f}' for k, g in enumerate(GAMMAS)), flush=True)
P_UNI = np.full(M.NCELL, 1.0 / M.NCELL)


def pre(prefix, d):
    return {f'{prefix}/{k}': v for k, v in d.items()}


def work(i):
    h, A = H[i], inst[i]
    out = {}
    P = {'raw': M.heat_to_pmf(h, 1.0), 'cf': M.heat_to_pmf(h, gamma_cf[i]), 'uni': P_UNI}
    pk = M.nms_peaks(h, 10)
    Hn = M.CELLS[pk]
    modes = M.gt_modes(A)
    out.update(pre('nms', M.hyp_metrics(Hn, A)))
    for tag in ('raw', 'cf'):
        w = np.array([M.disc_mass(P[tag], idx) for idx in pk])
        out.update(pre(f'nms_{tag}', M.mode_metrics(Hn, w, modes)))
        pr, ob = M.disc_pairs(P[tag], A, pk)
        out[f'ecepred_{tag}'], out[f'eceobs_{tag}'] = pr, ob
    for tag in ('raw', 'cf', 'uni'):
        out[f'nll_{tag}'] = M.nll(P[tag], A)
        out[f'w1_{tag}'] = M.w1_grid(P[tag], A)
        acc = {}
        for s in range(REPS):
            hy = M.sample_hyps(P[tag], np.random.default_rng([s, i]), 10)
            r = M.hyp_metrics(hy, A)
            r.update(M.mode_metrics(hy, np.ones(10), modes))
            for k, v in r.items():
                acc.setdefault(k, []).append(v)
        for k, v in acc.items():
            out[f'sample_{tag}/{k}'] = np.array(v)
    # K=1 点估计（G1 定义）
    xs = M.CELLS
    w = h.ravel() * (h.ravel() >= 0.3 * h.max())
    cen = np.array([(w * xs[:, 0]).sum() / w.sum(), (w * xs[:, 1]).sum() / w.sum()])[None]
    hf = h.ravel()
    mid = ((xs[pk[0]] + xs[pk[1]]) / 2)[None] if hf[pk[1]] >= 0.3 * hf[pk[0]] else xs[pk[:1]]
    for name, hy in (('centroid', cen), ('mid2', mid)):
        out.update(pre(name, M.hyp_metrics(hy, A, ks=(1,))))
        out.update(pre(name, M.mode_metrics(hy, np.ones(1), modes, ks=(1,))))
    # 参照
    out['loo'] = np.array([M.loo_kde_nll(A, s) for s in SIGMAS])
    out['splithalf_w1'] = M.split_half_w1(A, np.random.default_rng([12345, i]))
    return out


t0 = time.time()
with Pool(88) as pool:
    res = pool.map(work, range(N), chunksize=8)
print(f'[B] 逐实例指标完成 {time.time() - t0:.0f}s', flush=True)
X = {k: np.array([r[k] for r in res]) for k in res[0]}

sig_i = int(np.argmin(X['loo'].mean(0)))
SIG = SIGMAS[sig_i]
X['loo_nll'] = X['loo'][:, sig_i]
print(f'    LOO-KDE 全局带宽 σ*={SIG}（各 σ 平均 LOO-NLL: ' + ', '.join(f'{s}:{X["loo"][:, k].mean():.3f}' for k, s in enumerate(SIGMAS)) + ')', flush=True)

# ---------- 汇总 ----------
BI = np.random.default_rng(0).integers(0, F, (B, F))
_fc_cache = {}


def rep_mean(x):
    return x.mean(1) if x.ndim == 2 else x


def boot(x, sel, key=None):
    x = rep_mean(x)
    fs = np.bincount(frame_idx, weights=np.where(sel, x, 0.0), minlength=F)
    if key not in _fc_cache:
        fc = np.bincount(frame_idx, weights=sel.astype(float), minlength=F)
        _fc_cache[key] = np.maximum(fc[BI].sum(1), 1)
    ms = fs[BI].sum(1) / _fc_cache[key]
    return float(x[sel].mean()), [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]


def ece_w(pred, obs, wt, bins=10):
    b = np.minimum((pred * bins).astype(int), bins - 1)
    tot, W = 0.0, wt.sum()
    for k in range(bins):
        s = b == k
        if wt[s].sum() > 0:
            tot += wt[s].sum() / W * abs((obs[s] * wt[s]).sum() / wt[s].sum() - (pred[s] * wt[s]).sum() / wt[s].sum())
    return tot, float(((pred - obs) * wt).sum() / W)


results = {'prereg_sha256': PREREG_SHA, 'gamma_cf': {str(k): v for k, v in best_g.items()}, 'sigma_kde': SIG, 'subsets': {}}
for sn, sel in SUBSETS.items():
    R = {'n': int(sel.sum()), 'metrics': {}, 'ece': {}}
    for k, x in X.items():
        if k.startswith(('ecepred', 'eceobs', 'loo')) and k != 'loo_nll':
            continue
        m, ci = boot(x, sel, sn)
        r = {'mean': m, 'ci95': ci}
        if x.ndim == 2:  # 采样重复间标准差（对每次重复先取子集均值）
            r['repeat_std'] = float(x[sel].mean(0).std())
        R['metrics'][k] = r
    for tag in ('raw', 'cf'):
        pr, ob = X[f'ecepred_{tag}'], X[f'eceobs_{tag}']  # [N,3]
        fi = np.repeat(frame_idx[:, None], 3, 1)[sel].ravel()
        p_, o_ = pr[sel].ravel(), ob[sel].ravel()
        e0, g0 = ece_w(p_, o_, np.ones(len(p_)))
        es, gs = [], []
        for b_ in range(B):
            cnt = np.bincount(BI[b_], minlength=F)
            e, g = ece_w(p_, o_, cnt[fi].astype(float))
            es.append(e); gs.append(g)
        R['ece'][tag] = {'ece': e0, 'ece_ci95': [float(np.percentile(es, 2.5)), float(np.percentile(es, 97.5))], 'signed_gap': g0,
                         'gap_ci95': [float(np.percentile(gs, 2.5)), float(np.percentile(gs, 97.5))], 'n_pairs': int(len(p_))}
    # 配对差
    R['paired'] = {}
    for tag in ('raw', 'cf'):
        for K in (3, 5, 10):
            for met in ('minL2', 'cov0.1'):
                d = rep_mean(X[f'nms/{met}@{K}']) - rep_mean(X[f'sample_{tag}/{met}@{K}'])
                R['paired'][f'nms-sample_{tag}/{met}@{K}'] = dict(zip(('mean', 'ci95'), boot(d, sel, sn)))
        for K in (3, 5):
            d = rep_mean(X[f'nms_{tag}/modeR@{K}']) - rep_mean(X[f'sample_{tag}/modeR@{K}'])
            R['paired'][f'nms-sample_{tag}/modeR@{K}'] = dict(zip(('mean', 'ci95'), boot(d, sel, sn)))
    R['paired']['nll_raw-nll_cf'] = dict(zip(('mean', 'ci95'), boot(X['nll_raw'] - X['nll_cf'], sel, sn)))
    results['subsets'][sn] = R

json.dump(results, open(f'{OUT}/s1_results.json', 'w'), indent=1, ensure_ascii=False)
np.savez(f'{OUT}/s1_per_instance.npz', **{k: v for k, v in X.items()}, gamma_cf=gamma_cf, frame_idx=frame_idx)


# ---------- 打印 ----------
def g(sn, k, f='{:.4f}'):
    return f.format(results['subsets'][sn]['metrics'][k]['mean'])


for sn in SUBSETS:
    R = results['subsets'][sn]
    print(f'\n================ 子集 {sn}（n={R["n"]}）================')
    print('[假设类] minL2@K / cov0.10@K（标注者被 K 个假设覆盖的占比）/ cham@K')
    print(f'{"方法":14s} ' + ' '.join(f'{"minL2@" + str(K):>9s}' for K in M.KS) + ' | ' + ' '.join(f'{"cov.1@" + str(K):>9s}' for K in M.KS) + ' | ' + ' '.join(f'{"cham@" + str(K):>9s}' for K in M.KS))
    for name in ('nms', 'sample_raw', 'sample_cf', 'sample_uni'):
        print(f'{name:14s} ' + ' '.join(f'{g(sn, f"{name}/minL2@{K}"):>9s}' for K in M.KS) + ' | ' + ' '.join(f'{g(sn, f"{name}/cov0.1@{K}", "{:.3f}"):>9s}' for K in M.KS) + ' | ' + ' '.join(f'{g(sn, f"{name}/cham@{K}"):>9s}' for K in M.KS))
    for name in ('centroid', 'mid2'):
        print(f'{name:14s} {g(sn, f"{name}/minL2@1"):>9s} (K=1 only)   cov.1@1 {g(sn, f"{name}/cov0.1@1", "{:.3f}")}  cham@1 {g(sn, f"{name}/cham@1")}')
    print('[mode 精度/召回] P@K / R@K (K=1,3,5)')
    for name in ('nms_raw', 'nms_cf', 'sample_raw', 'sample_cf'):
        print(f'{name:14s} ' + ' '.join(f'P@{K}={g(sn, f"{name}/modeP@{K}", "{:.3f}")} R@{K}={g(sn, f"{name}/modeR@{K}", "{:.3f}")}' for K in M.MODE_KS))
    for name in ('centroid', 'mid2'):
        print(f'{name:14s} P@1={g(sn, f"{name}/modeP@1", "{:.3f}")} R@1={g(sn, f"{name}/modeR@1", "{:.3f}")}')
    print('[分布类] NLL（nats，均匀=8.318）/ W1')
    print(f'  NLL: raw {g(sn, "nll_raw")}  cf {g(sn, "nll_cf")}  uniform {g(sn, "nll_uni")}  LOO-KDE(σ={SIG}) {g(sn, "loo_nll")}')
    print(f'  W1 : raw {g(sn, "w1_raw")}  cf {g(sn, "w1_cf")}  uniform {g(sn, "w1_uni")}  split-half(参照) {g(sn, "splithalf_w1")}')
    for tag in ('raw', 'cf'):
        e = R['ece'][tag]
        print(f'  disc-ECE[{tag}]: {e["ece"]:.4f} [{e["ece_ci95"][0]:.4f},{e["ece_ci95"][1]:.4f}]  有符号差(pred−obs) {e["signed_gap"]:+.4f} [{e["gap_ci95"][0]:+.4f},{e["gap_ci95"][1]:+.4f}]  pairs={e["n_pairs"]}')
    print('[配对差 nms − sample(cf)]（正 = nms 更好；minL2 为负才更好，见符号）')
    for k, v in R['paired'].items():
        if '_cf' in k or k.startswith('nll'):
            print(f'  {k:28s} {v["mean"]:+.4f} [{v["ci95"][0]:+.4f},{v["ci95"][1]:+.4f}]')
print(f'\nsaved {OUT}/s1_results.json, s1_per_instance.npz')
