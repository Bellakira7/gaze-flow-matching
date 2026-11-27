"""forecast 汇总与 C19 判定（只用 VAT 训练集切出的 val）。python -I compare_forecast.py [--H 30] [--names fc_h0,fc_h30_s,fc_h30_t]
联合 NLL：画面内 −log(1−P_out) − log p(y)，出画 −log P_out（σ_s 取该臂最优）。配对差按片段 cluster bootstrap（B=2000）。
copy_gt（oracle 参照，用当前帧真值注视）：在 fit 上拟合
  P(出画 | 当前在内 / 在外)；当前在内 → 画面内密度 = w·N(g_t, s²)（在 64×64 格上归一化）+ (1−w)·边缘直方图，s∈{.01,.02,.03,.05,.08}、w∈{.3,.5,.7,.85}；当前在外 → 边缘直方图。
"""
from pathlib import Path
import argparse
import glob
import sys

import numpy as np

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_online')
sys.path.insert(0, f'{G}/setmetrics')
import metrics as M  # noqa: E402
from gaze_forecast import load_tracks  # noqa: E402

D = f'{G}/experiments/forecast/eval'
E = 6  # sample_every


def pairs(tracks, H):
    for ti, tr in enumerate(tracks):
        for k in range(0, len(tr['fid']), E):
            j = tr['pos'].get(int(tr['fid'][k]) + H)
            if j is not None:
                yield ti, k, j, tr


def fit_copy(H):
    tr_f = load_tracks('fit')
    hist = np.ones(M.NCELL)
    trans = np.zeros((2, 2))  # [当前在内?][未来在内?]
    P = []
    for _, k, j, tr in pairs(tr_f, H):
        ci, fi = int(tr['inn'][k]), int(tr['inn'][j])
        trans[ci, fi] += 1
        if fi:
            yy = tr['gaze'][j] / [tr['W'], tr['H']]
            hist[M.cell_index(np.clip(yy[None], 0, 1 - 1e-9))[0]] += 1
            if ci:
                P.append((tr['gaze'][k] / [tr['W'], tr['H']], yy))
    hist /= hist.sum()
    A = np.array(P)
    idx = np.random.default_rng(0).permutation(len(A))[:4000]
    best = (1e9, None, None)
    for s in (.01, .02, .03, .05, .08):
        for w in (.3, .5, .7, .85):
            v = copy_nll(A[idx, 0], A[idx, 1], hist, s, w).mean()
            if v < best[0]:
                best = (v, s, w)
    p_out = trans[:, 0] / trans.sum(1)  # p_out[当前在内?]
    return hist, p_out, best[1], best[2]


def copy_nll(cur, fut, hist, s, w):
    d2 = ((M.CELLS[None] - cur[:, None]) ** 2).sum(-1)
    g = np.exp(-d2 / (2 * s * s))
    g /= g.sum(1, keepdims=True)
    P = (1 - M.LAMBDA) * (w * g + (1 - w) * hist) + M.LAMBDA / M.NCELL
    return -np.log(P[np.arange(len(cur)), M.cell_index(np.clip(fut, 0, 1 - 1e-9))])


def load(name):
    fs = sorted(f for f in glob.glob(f'{D}/{name}_shard*of*.npz') if 'smoke' not in f)
    assert fs, name
    P = [np.load(f) for f in fs]
    d = {k: np.concatenate([p[k] for p in P]) for k in P[0].files}
    o = np.lexsort((d['k'], d['t']))
    return {k: v[o] for k, v in d.items()}


def joint(d):
    m = np.nanmean(d['nll_in'], 0)
    j = int(np.argmin(m))
    x = np.where(d['inn'], d['nll_in'][:, j] - np.log(1 - d['p_out']), -np.log(d['p_out']))
    return x, (1, 2, 3)[j], d['nll_in'][:, j]


def ap_out(p, out):
    o = np.argsort(-p)
    t = out[o].astype(float)
    return float((np.cumsum(t) / np.arange(1, len(t) + 1) * t).sum() / t.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--H', type=int, default=30)
    ap.add_argument('--names', default='fc_h0,fc_h30_s,fc_h30_t')
    a = ap.parse_args()
    names = [n for n in a.names.split(',') if glob.glob(f'{D}/{n}_shard*of*.npz')]
    arms = {n: load(n) for n in names}
    ref = arms[names[0]]
    for n in names[1:]:
        assert np.array_equal(arms[n]['t'], ref['t']) and np.array_equal(arms[n]['k'], ref['k']), n
    val = load_tracks('val')
    clip_of = {t: val[t]['clip'] for t in range(len(val))}
    cl = np.array([clip_of[t] for t in ref['t']])
    uc, cinv = np.unique(cl, return_inverse=True)
    inn = ref['inn']
    print(f'val 实例 {len(inn)}，片段 {len(uc)}，t+{a.H} 画面内 {inn.mean():.3f}')

    hist, p_out, s, w = fit_copy(a.H)
    print(f'copy_gt 拟合：s={s} w={w} P(出画|当前在外)={p_out[0]:.3f} P(出画|当前在内)={p_out[1]:.3f}')
    cj = np.zeros(len(inn))
    ci = np.full(len(inn), np.nan)
    for q, (t, k) in enumerate(zip(ref['t'], ref['k'])):
        tr = val[t]
        j = tr['pos'][int(tr['fid'][k]) + a.H]
        cur = int(tr['inn'][k])
        if tr['inn'][j]:
            yy = tr['gaze'][j] / [tr['W'], tr['H']]
            if cur:
                n = copy_nll((tr['gaze'][k] / [tr['W'], tr['H']])[None], yy[None], hist, s, w)[0]
            else:
                n = -np.log((1 - M.LAMBDA) * hist[M.cell_index(np.clip(yy[None], 0, 1 - 1e-9))[0]] + M.LAMBDA / M.NCELL)
            ci[q] = n
            cj[q] = n - np.log(1 - p_out[cur])
        else:
            cj[q] = -np.log(p_out[cur])

    J = {}
    print(f'\n{"臂":12s} {"联合NLL↓":>9s} {"σ_s":>3s} {"画面内NLL↓":>10s} {"hit@5.10↑":>9s} {"hit@5.05↑":>9s} {"点L2↓":>7s} {"出画AP↑":>8s}')
    print(f'{"copy_gt":12s} {cj.mean():9.4f} {"-":>3s} {np.nanmean(ci):10.4f} {"-":>9s} {"-":>9s} {"-":>7s} {"-":>8s}')
    for n, d in arms.items():
        J[n.replace('_best', '')], sg, nin = joint(d)  # _best 臂（C19 补充：按 val 选 epoch）与默认臂共用配对名
        print(f'{n:12s} {J[n.replace("_best", "")].mean():9.4f} {sg:3d} {np.nanmean(nin):10.4f} {np.nanmean(d["hit10"]):9.4f} {np.nanmean(d["hit05"]):9.4f} {np.nanmean(d["l2"]):7.4f} {ap_out(d["p_out"], ~d["inn"]):8.4f}')
    J['copy_gt'] = cj

    rng = np.random.default_rng(0)

    def cb(x, B=2000):
        s_, c = np.bincount(cinv, x), np.bincount(cinv)
        W = np.stack([np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc)) for _ in range(B)])
        m = (W * s_).sum(1) / (W * c).sum(1)
        return x.mean(), *np.percentile(m, [2.5, 97.5])

    print('\n配对差（负 = 前者更好；按片段 cluster bootstrap）：')
    res = {}
    for a1, a2 in (('fc_h30_s', 'fc_h0'), ('fc_h30_t', 'fc_h30_s'), ('fc_h30_t', 'fc_h0'), ('fc_h30_s', 'copy_gt'), ('fc_h30_t', 'copy_gt')):
        if a1 in J and a2 in J:
            res[(a1, a2)] = cb(J[a1] - J[a2])
            print(f'  {a1} − {a2}: {res[(a1, a2)][0]:+.4f} [{res[(a1, a2)][1]:+.4f}, {res[(a1, a2)][2]:+.4f}]')
    print('\n== C19 判定 ==')
    if ('fc_h30_s', 'fc_h0') in res:
        m, lo, hi = res[('fc_h30_s', 'fc_h0')]
        print(f'G1 直接预测未来优于照搬现时（差 ≤ −0.05 且上界 < 0）：{m:+.4f} [{lo:+.4f}, {hi:+.4f}] → {"过" if m <= -0.05 and hi < 0 else "不过"}')
    if ('fc_h30_t', 'fc_h30_s') in res:
        m, lo, hi = res[('fc_h30_t', 'fc_h30_s')]
        print(f'G2 时序上下文有用（差 ≤ −0.02 且上界 < 0）：{m:+.4f} [{lo:+.4f}, {hi:+.4f}] → {"过" if m <= -0.02 and hi < 0 else "不过"}')


if __name__ == '__main__':
    main()
