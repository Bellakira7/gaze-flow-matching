"""C21 机制判据（hold-out）：mixflow 的原生假设口径（predict() 合并后）对同臂样本聚类口径的逐实例配对差。
python -I compare_native_holdout.py [--name mixflow_eye_s0]
读 experiments/holdout_eval/{name}_shard*of*.npz（聚类口径）与 {name}_native_shard*of*.npz（原生口径，eval_ray_holdout --variant mixflow 一并产出）。
判据（写于 prereg C21）：原生 − 聚类 的 点 L2 差 CI95 上界 < +0.005 且 cov10@5 差 CI95 下界 > −0.01 → "原生假设可替代聚类"成立。
nll 两份文件相同（同一批样本），只作加载一致性断言。
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
    return {k: np.concatenate([p[k] for p in P])[o] for k in ('nll', 'cov', 'cov05', 'l2', 'hyp0')}


def boot(x, rng, B=2000):
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='mixflow_eye_s0')
    a = ap.parse_args()
    clu, nat = load(a.name), load(f'{a.name}_native')
    assert np.allclose(clu['nll'], nat['nll']), 'nll 应相同（同一批样本）——加载对不上'
    rng = np.random.default_rng(0)
    print(f'{a.name}：原生假设口径 vs 样本聚类口径（σ_h 来自 sampler_selection）')
    print(f'{"口径":8s} {"cov10↑":>8s} {"cov05↑":>8s} {"点L2↓":>8s}')
    for lbl, d in (('聚类', clu), ('原生', nat)):
        print(f'{lbl:8s} {d["cov"].mean():8.4f} {d["cov05"].mean():8.4f} {d["l2"].mean():8.4f}')
    res = {}
    for k, col in (('cov10', 'cov'), ('cov05', 'cov05'), ('点L2', 'l2')):
        mu, lo, hi = boot(nat[col] - clu[col], rng)
        res[k] = (mu, lo, hi)
        print(f'原生−聚类 {k:6s} Δ {mu:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]{"  跨0" if lo < 0 < hi else ""}')
    ok_l2, ok_cov = res['点L2'][2] < 0.005, res['cov10'][1] > -0.01
    print(f'\n== C21 机制判据：点 L2 差上界 {res["点L2"][2]:+.4f} < +0.005 {"✓" if ok_l2 else "✗"}；'
          f'cov10 差下界 {res["cov10"][1]:+.4f} > −0.01 {"✓" if ok_cov else "✗"} → '
          f'{"成立（原生假设可替代聚类）" if ok_l2 and ok_cov else "不成立（论文只报数字，不写替代）"}')


if __name__ == '__main__':
    main()
