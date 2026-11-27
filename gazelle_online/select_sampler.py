"""按 prereg/G2.clarifications_20261022.md §C8 的预声明规则聚合 tune_sampler_holdout.py 的各 shard 输出，写 experiments/sampler_selection.json。
python -I select_sampler.py [--dir experiments/sampler_tuning] [--out experiments/sampler_selection.json] [--expect 3762]
规则：
  flow 采样器 = 13 个配置里 nll_best 最低者（恰好相等取函数求值次数少者）；
  B5 步数     = diff|M64|{16,32,64} 里 nll_best 最低者（M16 照报不参与）；
  σ_h         = 在选出的 flow 配置上，使 {mdn, flow} 两头 cov10@5 平均最高者（恰好相等取 2.0）。
"""
from pathlib import Path
import argparse
import glob
import json

import numpy as np

G = str(Path(__file__).resolve().parents[1])
SIGMA_S = (1.0, 2.0, 3.0)
SIGMA_H = (1.0, 2.0, 3.0, 4.0)


def nfe(key):
    p = key.split('|')
    return int(p[2]) * (2 if p[1] == 'heun' else 1) if p[0] == 'flow' else int(p[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=f'{G}/experiments/sampler_tuning')
    ap.add_argument('--out', default=f'{G}/experiments/sampler_selection.json')
    ap.add_argument('--expect', type=int, default=3762)
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(f'{a.dir}/shard*of*.npz') if 'smoke' not in f)
    parts = [np.load(f) for f in files]
    idx = np.concatenate([p['idx'] for p in parts])
    assert len(idx) == a.expect and len(set(idx.tolist())) == a.expect, f'实例数 {len(idx)}（期望 {a.expect}，要 4 个 shard 都跑完）'
    keys = sorted({k.split('/')[0] for p in parts for k in p.files if '/' in k})
    D = {k: {m: np.concatenate([p[f'{k}/{m}'] for p in parts]) for m in ('nll', 'cov', 'cov05', 'l2')} for k in keys}
    T = {k: float(sum(p[f'{k}/sample_time_s'] for p in parts)) / a.expect for k in keys}  # 秒/实例（含与训练抢卡，仅作量级参考）
    st = {}
    for k in keys:
        m = D[k]['nll'].mean(0)
        j = int(np.argmin(m))
        st[k] = dict(nll_best=float(m[j]), sigma_s=SIGMA_S[j], col=D[k]['nll'][:, j])

    def pick(cands):
        best = min(st[k]['nll_best'] for k in cands)
        tied = [k for k in cands if st[k]['nll_best'] == best]
        return min(tied, key=nfe)

    def table(cands, title):
        kb = pick(cands)
        print(f'\n== {title}  ({len(D[cands[0]]["nll"])} 实例)')
        print(f'{"config":22s} {"nll_best":>9s} {"σ_s":>4s} {"Δvs最优":>8s} {"±配对SE":>8s} {"cov10(σh=2)":>11s} {"NFE":>4s} {"s/实例":>7s}')
        for k in sorted(cands, key=lambda k: st[k]['nll_best']):
            d = st[k]['col'] - st[kb]['col']
            se = d.std(ddof=1) / np.sqrt(len(d))
            j = 1 if D[k]['cov'].shape[1] > 1 else 0
            print(f'{k:22s} {st[k]["nll_best"]:9.4f} {st[k]["sigma_s"]:4.0f} {d.mean():+8.4f} {se:8.4f} {D[k]["cov"][:, j].mean():11.4f} {nfe(k):4d} {T[k]:7.3f}'
                  + ('  <- 选' if k == kb else ''))
        return kb

    flow_keys = [k for k in keys if k.startswith('flow|')]
    diff64 = [k for k in keys if k.startswith('diff|M64|')]
    diff16 = [k for k in keys if k.startswith('diff|M16|')]
    kf = table(flow_keys, 'Flow 采样器（nll_best 最低者）')
    kd = table(diff64, 'B5 步数（M_draw=64，nll_best 最低者）')
    print('\n-- B5 M_draw=16 同步数（不参与选择）')
    for k in diff16:
        print(f'  {k:16s} nll_best {st[k]["nll_best"]:.4f}  cov10 {D[k]["cov"][:, 0].mean():.4f}')

    print(f'\n== σ_h（flow={kf} 与 mdn 的 cov10@5 平均）')
    print(f'{"σ_h":>4s} {"flow cov10":>10s} {"mdn cov10":>10s} {"平均":>8s} | {"flow cov05":>10s} {"mdn cov05":>10s} | {"flow l2_pt":>10s} {"mdn l2_pt":>10s}')
    score = []
    for j, sh in enumerate(SIGMA_H):
        fc, mc = D[kf]['cov'][:, j].mean(), D['mdn']['cov'][:, j].mean()
        score.append((fc + mc) / 2)
        print(f'{sh:4.0f} {fc:10.4f} {mc:10.4f} {score[-1]:8.4f} | {D[kf]["cov05"][:, j].mean():10.4f} {D["mdn"]["cov05"][:, j].mean():10.4f} | '
              f'{D[kf]["l2"][:, j].mean():10.4f} {D["mdn"]["l2"][:, j].mean():10.4f}')
    top = max(score)
    tied = [SIGMA_H[j] for j, s in enumerate(score) if s == top]
    sigma_h = 2.0 if 2.0 in tied else min(tied)
    print(f'  -> σ_h = {sigma_h:g}')

    _, solver, steps, gamma = kf.split('|')
    sel = dict(flow=dict(solver=solver, steps=int(steps), gamma=float(gamma)), diff=dict(steps=int(kd.split('|')[2]), M_draw=64), sigma_h=sigma_h,
               flow_nll_best=st[kf]['nll_best'], diff_nll_best=st[kd]['nll_best'], n_holdout=a.expect, n_samples=10000)
    json.dump(sel, open(a.out, 'w'), indent=1, ensure_ascii=False)
    print('\nwrote', a.out, json.dumps(sel, ensure_ascii=False))


if __name__ == '__main__':
    main()
