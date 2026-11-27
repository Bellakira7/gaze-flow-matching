"""样本 -> pmf / 假设 的共用后处理，以及训练 hold-out 的逐实例指标。

pmf 一律调 g2_eval.pmf_from_samples（预注册 §3.2 同一条管线，不另写一份）；
假设 = 样本平滑直方图上的 NMS 取 K 个峰（半径 0.10，与 S1 的 nms_peaks 同定义），权重 = 峰周围半径 0.10 圆盘内的质量，
归一化后按权重降序（hyps[0] 即点估计）。B3/B5 与我方法共用这一条，B4(MCL) 与 B1 各用原生假设。
"""
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/setmetrics')
import metrics as M  # noqa: E402
from g2_eval import pmf_from_samples  # noqa: E402

K_HYP = 5
HOLD_SIGMAS = (1.0, 2.0, 3.0)


def samples_to_hyps(S, sigma_h=2.0, K=K_HYP, radius=0.10):
    """S [n,2] -> (hyps [K,2], w [K])。"""
    idx = M.cell_index(np.clip(S, 0, 1 - 1e-9))
    cnt = np.bincount(idx, minlength=M.NCELL).reshape(M.G, M.G).astype(float)
    sm = gaussian_filter(cnt, sigma_h, mode='constant')
    pk = M.nms_peaks(sm, K, radius)
    P = sm.ravel() / sm.sum()
    w = np.array([M.disc_mass(P, i, radius) for i in pk])
    hyps = M.CELLS[pk]
    if len(pk) < K:  # 极端情形（样本几乎全落一处）：用最高峰补齐，权重 0
        pad = K - len(pk)
        hyps = np.concatenate([hyps, np.repeat(hyps[:1], pad, 0)])
        w = np.concatenate([w, np.zeros(pad)])
    o = np.argsort(-w, kind='stable')
    hyps, w = hyps[o], w[o]
    return hyps, w / max(w.sum(), 1e-12)


def holdout_instance(S, y, hyps=None, w=None, sigma_h=2.0):
    """单标注 hold-out 的逐实例指标。S [n,2] 样本，y [2] 标注。hyps/w 缺省时由样本聚类得到。"""
    if hyps is None:
        hyps, w = samples_to_hyps(S, sigma_h)
    nll = [M.nll(pmf_from_samples(S, s), y[None]) for s in HOLD_SIGMAS]
    d = np.hypot(*(hyps - y).T)
    return dict(nll=nll, cov=float(d.min() <= 0.10), cov05=float(d.min() <= 0.05), l2=float(d[0]))


def summarize(rows):
    """rows: list of holdout_instance 输出 -> dict（nll_best = 3 个 σ 里均值最低者）。"""
    nll = np.array([r['nll'] for r in rows]).mean(0)
    k = int(np.argmin(nll))
    return dict(n=len(rows), nll_best=float(nll[k]), sigma_best=HOLD_SIGMAS[k], **{f'nll_s{s:g}': float(v) for s, v in zip(HOLD_SIGMAS, nll)},
                cov10=float(np.mean([r['cov'] for r in rows])), cov05=float(np.mean([r['cov05'] for r in rows])),
                l2_pt=float(np.mean([r['l2'] for r in rows])))
