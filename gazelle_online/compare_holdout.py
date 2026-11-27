"""汇总 hold-out 逐实例评测（eval_variant_holdout.py 的输出），给出：各 ckpt 的指标、flow 各 seed 的波动、眼部 token 变体对 flow_s0 的配对差与 CI。
python -I compare_holdout.py [--names flow_s0,flow_s1,flow_s2,flow_eye_s0] [--base flow_s0]
hold-out 同时被用来选过 lr 与采样超参，只能用来比较与选超参，不是无偏估计。

读法（预先写死）：
  * seed 波动 = flow_s0/s1/s2 在各指标上的标准差（样本标准差，n=3，只是粗略量级）。
  * 变体对 flow_s0 的差若落在 ±2×seed 标准差之内，则判为"与基线无法区分"，**不能**说眼部 token 有效。
  * 配对 CI = 逐实例差的 bootstrap（B=2000，重采样实例）；它只含"评测样本"噪声，不含"训练 seed"噪声，所以必须与 seed 波动一起看。
"""
from pathlib import Path
import argparse
import glob

import numpy as np

G = str(Path(__file__).resolve().parents[1])
D = f'{G}/experiments/holdout_eval'


def load(name):
    fs = sorted(f for f in glob.glob(f'{D}/{name}_shard*of*.npz') if 'smoke' not in f)
    assert fs, f'没有 {name} 的 shard'
    P = [np.load(f) for f in fs]
    idx = np.concatenate([p['idx'] for p in P])
    o = np.argsort(idx)
    assert len(idx) == 3762 and len(set(idx.tolist())) == 3762, (name, len(idx))
    return {k: np.concatenate([p[k] for p in P])[o] for k in ('nll', 'cov', 'cov05', 'l2')}


def metrics(d):
    m = d['nll'].mean(0)
    j = int(np.argmin(m))
    return dict(nll_best=float(m[j]), sigma_s=(1, 2, 3)[j], cov10=float(d['cov'].mean()), cov05=float(d['cov05'].mean()), l2_pt=float(d['l2'].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', default='flow_s0,flow_s1,flow_s2,flow_eye_s0')
    ap.add_argument('--base', default='flow_s0')
    a = ap.parse_args()
    names = [n for n in a.names.split(',') if glob.glob(f'{D}/{n}_shard*of*.npz')]
    R, DD = {}, {}
    for n in names:
        DD[n] = load(n)
        R[n] = metrics(DD[n])
    print(f'{"ckpt":14s} {"NLL↓":>8s} {"σ_s":>4s} {"cov10@5↑":>9s} {"cov05@5↑":>9s} {"点估计L2↓":>9s}')
    for n in names:
        r = R[n]
        print(f'{n:14s} {r["nll_best"]:8.4f} {r["sigma_s"]:4d} {r["cov10"]:9.4f} {r["cov05"]:9.4f} {r["l2_pt"]:9.4f}')
    seeds = [n for n in ('flow_s0', 'flow_s1', 'flow_s2') if n in R]
    sd = {}
    if len(seeds) >= 2:
        print(f'\nflow seed 波动（{len(seeds)} 个 seed，样本标准差）：')
        for k in ('nll_best', 'cov10', 'cov05', 'l2_pt'):
            v = np.array([R[n][k] for n in seeds])
            sd[k] = float(v.std(ddof=1))
            print(f'  {k:9s} mean {v.mean():.4f}  sd {sd[k]:.4f}  range [{v.min():.4f}, {v.max():.4f}]')
    rng = np.random.default_rng(0)
    for n in names:
        if n == a.base or a.base not in R or 'eye' not in n:
            continue
        d0, d1 = DD[a.base], DD[n]
        N = len(d0['cov'])
        BI = rng.integers(0, N, (2000, N))
        print(f'\n{n} − {a.base}（配对；NLL/L2 负 = 变体更好，cov 正 = 变体更好）：')
        j0, j1 = int(np.argmin(d0['nll'].mean(0))), int(np.argmin(d1['nll'].mean(0)))
        for k, x in (('nll_best', d1['nll'][:, j1] - d0['nll'][:, j0]), ('cov10', d1['cov'] - d0['cov']), ('cov05', d1['cov05'] - d0['cov05']), ('l2_pt', d1['l2'] - d0['l2'])):
            bs = x[BI].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            verdict = ''
            if k in sd:
                verdict = f'  |Δ|/seed_sd = {abs(x.mean()) / sd[k]:.1f}' + ('（在 2×seed_sd 内，无法与基线区分）' if abs(x.mean()) <= 2 * sd[k] else '（超出 2×seed_sd）')
            print(f'  {k:9s} Δ {x.mean():+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]{"  跨0" if lo < 0 < hi else ""}{verdict}')


if __name__ == '__main__':
    main()
