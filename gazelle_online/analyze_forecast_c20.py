"""C20 汇总与判定：python -I analyze_forecast_c20.py [--split test]（默认 test；--split val 用于读测试集之前的健全性检查）
- σ_s：每个臂在 val 上取联合 NLL 最低的 σ_s（1/2/3），其余 split 沿用。
- 联合 NLL：画面内 −log(1−p_out) − log P_in(y)，出画 −log p_out。
- conformal：在 cal 上取 q̂ = 第 ⌈(n+1)(1−α)⌉ 小的分数，向上取到 0.01 网格；被评 split 上报覆盖率（分数 ≤ q̂）、
  区域画面内面积（格数/4096）均值、区域含"出画"的比例。α ∈ {0.1, 0.2}。
- copy_gt（oracle，用当前帧真值注视）：同 compare_forecast.py，转移概率 / 高斯宽度 / 混合权重在 fit 上拟合；只报 NLL。
- 配对差（主臂 − 各对照）：按片段 cluster bootstrap（B=2000）。C20 判据见 prereg 澄清 C20。
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
from eval_forecast_c20 import QG  # noqa: E402

D = f'{G}/experiments/forecast/c20_eval'
ARMS = ['c20_ours_s0', 'c20_ours_s1', 'c20_flow', 'c20_mdn', 'c20_mcl', 'c20_heatmap', 'c20_ours_h0']
LABEL = {'c20_ours_s0': 'Ours（flowET_d6）s0', 'c20_ours_s1': 'Ours s1', 'c20_flow': 'flow（无眼部）', 'c20_mdn': 'B3 MDN', 'c20_mcl': 'B4 MCL',
         'c20_heatmap': 'B1 热图', 'c20_ours_h0': 'Ours@H=0（当前帧模型）', 'copy_gt': 'copy_gt（oracle）'}
H, E = 30, 6


def load(name, split):
    fs = sorted(f for f in glob.glob(f'{D}/{name}_{split}_shard*of*.npz') if 'smoke' not in f)
    if not fs:
        return None
    P = [np.load(f) for f in fs]
    d = {k: np.concatenate([p[k] for p in P]) for k in P[0].files}
    o = np.lexsort((d['k'], d['t']))
    return {k: v[o] for k, v in d.items()}


def joint_all(d):
    """[N,3]：三个 σ_s 下的联合 NLL。"""
    return np.where(d['inn'][:, None], d['nll_in'] - np.log(1 - d['p_out'])[:, None], -np.log(d['p_out'])[:, None])


def copy_gt_nll(split):
    fit = load_tracks('fit', 'c20')
    hist, trans, P = np.ones(M.NCELL), np.zeros((2, 2)), []
    for tr in fit:
        for k in range(0, len(tr['fid']), E):
            j = tr['pos'].get(int(tr['fid'][k]) + H)
            if j is None:
                continue
            ci, fi = int(tr['inn'][k]), int(tr['inn'][j])
            trans[ci, fi] += 1
            if fi:
                yy = tr['gaze'][j] / [tr['W'], tr['H']]
                hist[M.cell_index(np.clip(yy[None], 0, 1 - 1e-9))[0]] += 1
                if ci:
                    P.append((tr['gaze'][k] / [tr['W'], tr['H']], yy))
    hist /= hist.sum()
    A = np.array(P)[np.random.default_rng(0).permutation(len(P))[:4000]]

    def cn(cur, fut, s, w):
        g = np.exp(-((M.CELLS[None] - cur[:, None]) ** 2).sum(-1) / (2 * s * s))
        g /= g.sum(1, keepdims=True)
        Pm = (1 - M.LAMBDA) * (w * g + (1 - w) * hist) + M.LAMBDA / M.NCELL
        return -np.log(Pm[np.arange(len(cur)), M.cell_index(np.clip(fut, 0, 1 - 1e-9))])

    s, w = min(((s, w) for s in (.01, .02, .03, .05, .08) for w in (.3, .5, .7, .85)), key=lambda sw: cn(A[:, 0], A[:, 1], *sw).mean())
    p_out = trans[:, 0] / trans.sum(1)
    tracks = load_tracks(split, 'c20', allow_test=split == 'test')
    out = {}
    for t, tr in enumerate(tracks):
        for k in range(0, len(tr['fid']), E):
            j = tr['pos'].get(int(tr['fid'][k]) + H)
            if j is None:
                continue
            cur = int(tr['inn'][k])
            if tr['inn'][j]:
                yy = tr['gaze'][j] / [tr['W'], tr['H']]
                n_in = cn((tr['gaze'][k] / [tr['W'], tr['H']])[None], yy[None], s, w)[0] if cur else \
                    -np.log((1 - M.LAMBDA) * hist[M.cell_index(np.clip(yy[None], 0, 1 - 1e-9))[0]] + M.LAMBDA / M.NCELL)
                out[(t, k)] = n_in - np.log(1 - p_out[cur])
            else:
                out[(t, k)] = -np.log(p_out[cur])
    return out, (s, w, p_out)


def conformal(cal, ev, j, alpha):
    sc = np.sort(cal['score'][:, j])
    n = len(sc)
    q = sc[min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)]
    gi = min(len(QG) - 1, int(np.searchsorted(QG, q - 1e-12)))
    qg = QG[gi]
    return dict(q=float(q), qg=float(qg), cov=float((ev['score'][:, j] <= qg).mean()), area=ev['area'][:, j, gi], out=float(ev['out_in'][:, j, gi].mean()),
                covered=(ev['score'][:, j] <= qg).astype(float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--main', default='c20_ours_s0')
    a = ap.parse_args()
    V = {n: load(n, 'val') for n in ARMS}
    C = {n: load(n, 'cal') for n in ARMS}
    T = {n: load(n, a.split) for n in ARMS}
    arms = [n for n in ARMS if V[n] is not None and C[n] is not None and T[n] is not None]
    ref = T[arms[0]]
    for n in arms[1:]:
        assert np.array_equal(T[n]['t'], ref['t']) and np.array_equal(T[n]['k'], ref['k']), n
    cl = ref['clip']
    uc, cinv = np.unique(cl, return_inverse=True)
    print(f'split={a.split}：实例 {len(cl)}，片段 {len(uc)}，t+{H} 画面内 {ref["inn"].mean():.3f}；cal 实例 {len(C[arms[0]]["t"])}')
    sig = {n: int(np.argmin(joint_all(V[n]).mean(0))) for n in arms}
    J = {n: joint_all(T[n])[:, sig[n]] for n in arms}
    cg, (s, w, po) = copy_gt_nll(a.split)
    J['copy_gt'] = np.array([cg[(t, k)] for t, k in zip(ref['t'], ref['k'])])
    print(f'copy_gt 拟合：s={s} w={w} P(出画|当前在外)={po[0]:.3f} P(出画|当前在内)={po[1]:.3f}')

    def ap_out(p, out):
        o = np.argsort(-p)
        t = out[o].astype(float)
        return float((np.cumsum(t) / np.arange(1, len(t) + 1) * t).sum() / t.sum())

    CF = {}
    print(f'\n{"臂":24s} {"σ_s":>3s} {"联合NLL↓":>8s} {"画面内NLL↓":>9s} {"hit@5↑":>7s} {"hit.05↑":>7s} {"点L2↓":>6s} {"出画AP↑":>7s} | '
          f'{"α=.1 覆盖":>8s} {"面积↓":>6s} {"含出画":>6s} | {"α=.2 覆盖":>8s} {"面积↓":>6s}')
    for n in arms + ['copy_gt']:
        if n == 'copy_gt':
            print(f'{LABEL[n]:24s} {"-":>3s} {J[n].mean():8.4f}')
            continue
        d, j = T[n], sig[n]
        c1, c2 = conformal(C[n], d, j, 0.1), conformal(C[n], d, j, 0.2)
        CF[n] = c1
        print(f'{LABEL[n]:24s} {(1, 2, 3)[j]:3d} {J[n].mean():8.4f} {np.nanmean(d["nll_in"][:, j]):9.4f} {np.nanmean(d["hit10"]):7.4f} {np.nanmean(d["hit05"]):7.4f} '
              f'{np.nanmean(d["l2"]):6.4f} {ap_out(d["p_out"], ~d["inn"]):7.4f} | {c1["cov"]:8.3f} {c1["area"].mean():6.4f} {c1["out"]:6.3f} | {c2["cov"]:8.3f} {c2["area"].mean():6.4f}')

    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc)) for _ in range(2000)])

    def cb(x):
        s_, c = np.bincount(cinv, x), np.bincount(cinv)
        m = (W * s_).sum(1) / (W * c).sum(1)
        return x.mean(), *np.percentile(m, [2.5, 97.5])

    mn = a.main
    print(f'\n配对差（{LABEL[mn]} − 对照；负 = 主臂更好；按片段 cluster bootstrap）')
    h1, h2 = [], []
    for n in arms + ['copy_gt']:
        if n in (mn, 'c20_ours_s1'):
            continue
        r = cb(J[mn] - J[n])
        line = f'  vs {LABEL[n]:24s} 联合NLL {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]'
        h1.append((n, r[2] < 0))
        if n in CF:
            ra = cb(CF[mn]['area'] - CF[n]['area'])
            line += f'   α=0.1 区域面积 {ra[0]:+.4f} [{ra[1]:+.4f}, {ra[2]:+.4f}]'
            h2.append((n, ra[2] < 0))
        print(line)
    if 'c20_ours_s1' in J:
        r = cb(J['c20_ours_s1'] - J[mn])
        print(f'  seed 1 − seed 0：联合NLL {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]')
    print(f'\n== C20 判定（split={a.split}）==')
    print('H1 联合 NLL 显著低于每个对照（含 copy_gt）：' + ('全部成立' if all(v for _, v in h1) else '不成立：' + ', '.join(LABEL[n] for n, v in h1 if not v)))
    print('H2 α=0.1 conformal 区域面积显著小于每个学习型对照：' + ('全部成立' if all(v for _, v in h2) else '不成立：' + ', '.join(LABEL[n] for n, v in h2 if not v)))
    print('覆盖率（α=0.1，目标 ≥ 0.90；cal 与被评 split 节目不同，只是经验覆盖）：' + '  '.join(f'{LABEL[n]} {CF[n]["cov"]:.3f}' for n in CF))


if __name__ == '__main__':
    main()
