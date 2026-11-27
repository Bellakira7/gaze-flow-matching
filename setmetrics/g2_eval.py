"""G2 评测适配器 + 判定，逐字落地 gaze/prereg/G2.prereg.md（v1，sha256 见 PRE_SHA）。

两个入口：
  python -I g2_eval.py eval --pred pred.npz --out bundle.npz
      pred.npz: hyps[N,5,2], w[N,5], samples[N,1000,2]，实例顺序同 results/setmetrics/s1_per_instance.npz
  python -I g2_eval.py gate --method bundle.npz --baseline B1=b1.npz --baseline B2=b2.npz ...
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402

G_ = str(Path(__file__).resolve().parents[1])
GAZELLE_ROOT = str(Path(__file__).resolve().parents[1] / 'third_party/gazelle')
PRE_SHA = '6f7602d11ae0d46f14a1d72e1d4ecf3ed1032ef01cd2b24c5d7f040442abaf96'  # G2.prereg.md v2


def _check_prereg():
    assert hashlib.sha256(open(f'{G_}/prereg/G2.prereg.md', 'rb').read()).hexdigest() == PRE_SHA, 'G2 预注册文件与代码记录的 sha256 不一致'

K_HYP, N_SAMP = 5, 10000
SIGMAS_S = [0.5, 1, 1.5, 2, 3, 4]
NLL_REF = {'all': 5.437658, 'M_sep': 5.872344}
REQUIRED = ('B1', 'B2', 'B3', 'B4', 'B5')
STD_TOL = 0.005
B_BOOT = 2000


def load_context():
    frames = json.load(open(f'{G_}/data/gazefollow_test_hf/test_preprocessed.json'))
    inst, fidx, hw = [], [], []
    for fi, fr in enumerate(frames):
        for h in fr['heads']:
            inst.append(np.stack([h['gazex_norm'], h['gazey_norm']], 1))
            fidx.append(fi)
            hw.append((fr['height'], fr['width']))
    sep = np.load(f'{G_}/results/g1/per_instance.npz')['sep']
    return inst, np.array(fidx), hw, {'all': np.ones(len(inst), bool), 'M_sep': sep}, len(frames)


# ---------------- 方法输出 -> pmf / 假设 ----------------
def validate(pred, N):
    hyps, w, S = (np.asarray(pred[k], float) for k in ('hyps', 'w', 'samples'))
    assert hyps.shape == (N, K_HYP, 2), f'hyps 形状 {hyps.shape}，应为 ({N},{K_HYP},2)'
    assert w.shape == (N, K_HYP), f'w 形状 {w.shape}'
    assert S.shape == (N, N_SAMP, 2), f'samples 形状 {S.shape}，应为 ({N},{N_SAMP},2)'
    assert (w >= 0).all() and np.allclose(w.sum(1), 1.0, atol=1e-6), 'w 必须非负且和为 1'
    assert (np.diff(w, axis=1) <= 1e-12).all(), 'w 必须按降序排列（hyps[0] 是权重最大的假设）'
    assert S.min() >= 0 and S.max() <= 1, 'samples 必须在 [0,1]^2'
    assert np.isfinite(hyps).all() and np.isfinite(S).all()
    return hyps, w, S


def pmf_from_samples(S, sigma, lam=M.LAMBDA):
    idx = M.cell_index(np.clip(S, 0, 1 - 1e-9))
    cnt = np.bincount(idx, minlength=M.NCELL).reshape(M.G, M.G).astype(float)
    sm = gaussian_filter(cnt, sigma, mode='constant')
    P = sm.ravel() / sm.sum()
    return (1 - lam) * P + lam / M.NCELL


def fold_of(fidx, F):
    perm = np.random.default_rng(0).permutation(F)  # 与 S1 相同：后半为折 1
    fof = np.zeros(F, int)
    fof[perm[F // 2:]] = 1
    return fof[fidx]


def crossfit_nll(S, inst, fidx, F):
    N = len(inst)
    mat = np.array([[M.nll(pmf_from_samples(S[i], s), inst[i]) for s in SIGMAS_S] for i in range(N)])
    fold = fold_of(fidx, F)
    sig_idx = np.zeros(N, int)
    chosen = {}
    for f in (0, 1):
        k = int(np.argmin(mat[fold != f].mean(0)))
        sig_idx[fold == f] = k
        chosen[f] = SIGMAS_S[k]
    return mat[np.arange(N), sig_idx], np.array(SIGMAS_S)[sig_idx], chosen


def _auc_worker(a):
    import torch
    torch.set_num_threads(1)
    if GAZELLE_ROOT not in sys.path:
        sys.path.insert(0, GAZELLE_ROOT)
    import gazelle.utils as gutils
    heat, gx, gy, h, w = a
    return float(gutils.gazefollow_auc(torch.from_numpy(heat), gx, gy, h, w))


def evaluate_prediction(pred, n_proc=88):
    _check_prereg()
    inst, fidx, hw, subsets, F = load_context()
    N = len(inst)
    hyps, w, S = validate(pred, N)
    modes = [M.gt_modes(A) for A in inst]
    b = {k: np.zeros(N) for k in ('cov0.1@5', 'cov0.05@5', 'minL2@5', 'cham@5', 'modeP@3', 'modeR@3', 'avg_l2_fix', 'min_l2_fix')}
    for i, A in enumerate(inst):
        r = M.hyp_metrics(hyps[i], A, ks=(5,))
        for k in ('cov0.1@5', 'cov0.05@5', 'minL2@5', 'cham@5'):
            b[k][i] = r[k]
        mm = M.mode_metrics(hyps[i], w[i], modes[i], ks=(3,))
        b['modeP@3'][i], b['modeR@3'][i] = mm['modeP@3'], mm['modeR@3']
        pt = hyps[i, 0]  # 点估计固定为 hyps[0]
        b['avg_l2_fix'][i] = np.hypot(*(pt - A.mean(0)))
        b['min_l2_fix'][i] = np.hypot(*(A - pt).T).min()
    nll_cf, sig, chosen = crossfit_nll(S, inst, fidx, F)
    jobs = []
    for i, A in enumerate(inst):
        P = pmf_from_samples(S[i], sig[i]).reshape(M.G, M.G)
        jobs.append(((P / P.max()).astype(np.float32), list(A[:, 0]), list(A[:, 1]), hw[i][0], hw[i][1]))
    with Pool(n_proc) as pool:
        auc = np.array(pool.map(_auc_worker, jobs, chunksize=16))
    if 'pmf' in pred:  # 可选：精确 pmf 的 NLL，照报但不参与判定
        Pm = np.asarray(pred['pmf'], float).reshape(N, -1)
        Pm = Pm / Pm.sum(1, keepdims=True)
        b['nll_exact'] = np.array([M.nll((1 - M.LAMBDA) * Pm[i] + M.LAMBDA / M.NCELL, inst[i]) for i in range(N)])
    b.update({'auc': auc, 'nll_variants': nll_cf[:, None], 'nll_variant_names': np.array(['cf']), 'sigma_s': sig,
              'sigma_s_chosen': np.array([chosen[0], chosen[1]])})
    return b


OFFICIAL_BUNDLE = f'{G_}/results/setmetrics/official_gazelle_vitl_bundle_v2.npz'


def build_official_bundle(path=OFFICIAL_BUNDLE):
    """官方 Gaze-LLE ViT-L（S1 热图）在 G2 v2 管线下的参照 bundle：raw(γ=1) 与 cf(γ 取 S1 交叉拟合值)两个 pmf 变体，
    各自抽 N_SAMP 个样本后走与方法相同的“样本 -> pmf”管线（σ_s 交叉拟合）。精确 pmf 的 NLL 另存为 nll_exact_variants（不参与判定）。"""
    inst, fidx, hw, subsets, F = load_context()
    N = len(inst)
    H = np.load(f'{G_}/results/repro_test/full.npz')['heatmaps'].astype(np.float32)
    X = np.load(f'{G_}/results/setmetrics/s1_per_instance.npz')
    full = np.load(f'{G_}/results/repro_test/full.npz')['per_instance']
    gam = X['gamma_cf']
    nll_v, exact_v = [], []
    for tag, gam_i in (('raw', np.ones(N)), ('cf', gam)):
        S = np.zeros((N, N_SAMP, 2), np.float32)
        ex = np.zeros(N)
        for i in range(N):
            P = M.heat_to_pmf(H[i], gam_i[i])
            S[i] = M.sample_hyps(P, np.random.default_rng([31 if tag == 'raw' else 32, i]), N_SAMP)
            ex[i] = M.nll(P, inst[i])
        nl, _, chosen = crossfit_nll(S.astype(float), inst, fidx, F)
        print(f'[official/{tag}] 样本管线 NLL {nl.mean():.4f}（精确 {ex.mean():.4f}），σ_s={chosen}', flush=True)
        nll_v.append(nl); exact_v.append(ex)
    b = {'cov0.1@5': X['nms/cov0.1@5'], 'nll_variants': np.stack(nll_v, 1), 'nll_variant_names': np.array(['raw', 'cf']),
         'nll_exact_variants': np.stack(exact_v, 1), 'auc': full[:, 0], 'avg_l2_fix': full[:, 3], 'min_l2_fix': full[:, 4],
         'modeP@3': X['nms_raw/modeP@3'], 'modeR@3': X['nms_raw/modeR@3']}
    np.savez(path, **b)
    return b


def official_bundle():
    return _load(OFFICIAL_BUNDLE)


# ---------------- 判定 ----------------
def _resample_means(x, sel, fidx, F, BI):
    fs = np.bincount(fidx, weights=np.where(sel, x, 0.0), minlength=F)
    fc = np.bincount(fidx, weights=sel.astype(float), minlength=F)
    return fs[BI].sum(1) / np.maximum(fc[BI].sum(1), 1)


def _best_nll(bundle, sel):
    V = np.asarray(bundle['nll_variants'], float)
    means = V[sel].mean(0)
    k = int(np.argmin(means))
    return k, float(means[k])


def gate(method, baselines, required=REQUIRED, std_baseline='B1', ctx=None):
    _check_prereg()
    inst, fidx, hw, subsets, F = ctx or load_context()
    missing = [r for r in required if r not in baselines]
    if missing:
        return {'verdict': 'INCOMPLETE', 'missing_baselines': missing}
    BI = np.random.default_rng(0).integers(0, F, (B_BOOT, F))
    out = {'prereg_sha256': PRE_SHA, 'subsets': {}}
    for sn in ('all', 'M_sep'):
        sel = subsets[sn]
        R = {'n': int(sel.sum())}
        # 最强 baseline（点估计层面选定身份，之后固定身份做 bootstrap）
        cov_means = {n: float(np.asarray(b['cov0.1@5'], float)[sel].mean()) for n, b in baselines.items()}
        bn = max(cov_means, key=cov_means.get)
        cov_b, cov_m = cov_means[bn], float(np.asarray(method['cov0.1@5'], float)[sel].mean())
        nl = {n: _best_nll(b, sel) for n, b in baselines.items()}
        nn = min(nl, key=lambda n: nl[n][1])
        nk, nll_b = nl[nn]
        _, nll_m = _best_nll(method, sel)
        d1 = (cov_m - cov_b) / (1 - cov_b)
        thr2 = max(0.10 * (nll_b - NLL_REF[sn]), 0.01)
        d2 = nll_b - nll_m
        # bootstrap（配对，重采样帧）
        cm = _resample_means(np.asarray(method['cov0.1@5'], float), sel, fidx, F, BI)
        cb = _resample_means(np.asarray(baselines[bn]['cov0.1@5'], float), sel, fidx, F, BI)
        mv = np.asarray(method['nll_variants'], float)[:, int(np.argmin(np.asarray(method['nll_variants'], float)[sel].mean(0)))]
        bv = np.asarray(baselines[nn]['nll_variants'], float)[:, nk]
        d1b = (cm - cb) / (1 - cb)
        d2b = _resample_means(bv, sel, fidx, F, BI) - _resample_means(mv, sel, fidx, F, BI)
        R.update({'P1': {'baseline': bn, 'cov_base': cov_b, 'cov_method': cov_m, 'delta1_miss_reduction': d1, 'threshold': 0.10,
                         'ci95': [float(np.percentile(d1b, 2.5)), float(np.percentile(d1b, 97.5))], 'point_pass': bool(d1 >= 0.10)},
                  'P2': {'baseline': f'{nn}/{baselines[nn]["nll_variant_names"][nk]}', 'nll_base': nll_b, 'nll_method': nll_m, 'improvement': d2,
                         'threshold': thr2, 'nll_ref': NLL_REF[sn],
                         'ci95': [float(np.percentile(d2b, 2.5)), float(np.percentile(d2b, 97.5))], 'point_pass': bool(d2 >= thr2)}})
        R['ci_lower_positive'] = bool(R['P1']['ci95'][0] > 0 and R['P2']['ci95'][0] > 0)
        out['subsets'][sn] = R
    # 约束 C：标准指标（全体，对 std_baseline）
    sb = baselines[std_baseline]
    C = {}
    for k, sign in (('auc', -1), ('avg_l2_fix', +1), ('min_l2_fix', +1)):
        mm_, bb_ = float(np.asarray(method[k], float).mean()), float(np.asarray(sb[k], float).mean())
        delta = mm_ - bb_
        C[k] = {'method': mm_, 'baseline': bb_, 'delta': delta, 'ok': bool(delta >= -STD_TOL if k == 'auc' else delta <= STD_TOL)}
    out['C'] = C
    a, bsub = out['subsets']['all'], out['subsets']['M_sep']
    A_ok = a['P1']['point_pass'] and a['P2']['point_pass'] and a['ci_lower_positive']
    B_ok = bsub['P1']['point_pass'] and bsub['P2']['point_pass']
    C_ok = all(v['ok'] for v in C.values())
    out['A'], out['B'], out['C_ok'] = bool(A_ok), bool(B_ok), bool(C_ok)
    out['verdict'] = 'PASS' if (A_ok and B_ok and C_ok) else 'FAIL'
    return out


def _load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('eval'); e.add_argument('--pred', required=True); e.add_argument('--out', required=True)
    g = sub.add_parser('gate'); g.add_argument('--method', required=True); g.add_argument('--baseline', action='append', default=[]); g.add_argument('--out')
    a = ap.parse_args()
    if a.cmd == 'eval':
        np.savez(a.out, **evaluate_prediction(_load(a.pred)))
        print('saved', a.out)
    else:
        bl = {kv.split('=')[0]: _load(kv.split('=')[1]) for kv in a.baseline}
        res = gate(_load(a.method), bl)
        print(json.dumps(res, indent=1, ensure_ascii=False))
        if a.out:
            json.dump(res, open(a.out, 'w'), indent=1, ensure_ascii=False)
