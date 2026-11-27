"""训练 hold-out（3762 实例，单标注）上的采样超参调参，规则见 prereg/G2.clarifications_20261022.md §C8（读数前预声明）。
一次遍历 hold-out，对每个实例在同一份场景 token 上跑完所有预声明的采样配置，逐实例指标落盘；选择由 select_sampler.py 聚合后按规则做。
  flow：solver × steps × γ 网格；mdn：单配置（解析 pmf，仅用于 σ_h）；diff：M_draw=64 × steps∈{16,32,64}，另报 M_draw=16。
每配置每实例存 nll[σ_s∈{1,2,3}] 与 cov/cov05/l2（聚类型头对 σ_h∈{1,2,3,4} 各一列；原生假设只有一列）。
只用 hold-out，不碰测试集。
python -I -u tune_sampler_holdout.py --shard 0 --nshards 4 [--limit N]   # CUDA_VISIBLE_DEVICES 选卡；4 个 shard 并行
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_online')
from train_online import AmpBackbone, SDGazeDataset  # noqa: E402
from common import IN_SIZE  # noqa: E402
from gazelle.dataloader import collate_fn  # noqa: E402
import gaze_eval as GE  # noqa: E402
import metrics as M  # noqa: E402
from g2_eval import pmf_from_samples  # noqa: E402
from gaze_models import GazeModel  # noqa: E402
import gaze_diffusion as GD  # noqa: E402

SIGMA_H = (1.0, 2.0, 3.0, 4.0)
CKPT = {h: f'{G}/experiments/{h}_s0/epoch_14.pt' for h in ('flow', 'mdn', 'diff')}

# (head, key, sample_kw, nfe)；key 是落盘名，select_sampler.py 按它读
CONFIGS = []
for solver in ('heun', 'euler'):
    for steps in (8, 16, 32, 64):
        for gamma in (1.0, 2.0, 3.0):
            if solver == 'euler' and gamma != 2.0:  # euler 只扫步数（γ 固定为默认 2），控制算力
                continue
            if solver == 'heun' and steps == 64:  # heun-64 步算力过大，且 heun-32 已是 64 次函数求值
                continue
            CONFIGS.append(('flow', f'flow|{solver}|{steps}|{gamma:g}', dict(solver=solver, steps=steps, gamma=gamma), steps * (2 if solver == 'heun' else 1)))
CONFIGS.append(('mdn', 'mdn', {}, 0))
for M_draw in (64, 16):
    for steps in (16, 32, 64):
        CONFIGS.append(('diff', f'diff|M{M_draw}|{steps}', dict(M_draw=M_draw, steps=steps), steps))


def instance_row(head, S, y, native=None):
    """-> nll[3], cov[Sh], cov05[Sh], l2[Sh]。"""
    nll = [M.nll(pmf_from_samples(S, s), y[None]) for s in GE.HOLD_SIGMAS]
    cov, cov05, l2 = [], [], []
    hyp_sets = [native] if native is not None else [GE.samples_to_hyps(S, sh) for sh in SIGMA_H]
    for hyps, _ in hyp_sets:
        d = np.hypot(*(hyps - y).T)
        cov.append(float(d.min() <= 0.10))
        cov05.append(float(d.min() <= 0.05))
        l2.append(float(d[0]))
    return nll, cov, cov05, l2


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--n_samples', type=int, default=10000)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='每个 shard 只跑前 N 个实例（冒烟）')
    ap.add_argument('--out', default=f'{G}/experiments/sampler_tuning')
    ap.add_argument('--only', default='', help='逗号分隔的 head 子集（冒烟用）')
    a = ap.parse_args()
    dev = 'cuda'
    heads = [h for h in ('flow', 'mdn', 'diff') if not a.only or h in a.only.split(',')]
    bb = AmpBackbone('dinov2_vitl14')
    models = {}
    for h in heads:
        m = GazeModel(bb, h).to(dev).eval()
        m.load_decoder_state(torch.load(CKPT[h], map_location='cpu', weights_only=True))
        models[h] = m
    ds = SDGazeDataset(f'{G}/data/gazefollow_train_sd', 'holdout', bb.get_transform(IN_SIZE))
    sel = list(range(len(ds.data_idxs)))[a.shard::a.nshards]
    if a.limit:
        sel = sel[:a.limit]
    ds.data_idxs = [ds.data_idxs[i] for i in sel]
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, collate_fn=collate_fn, num_workers=a.workers)
    gen = torch.Generator(device=dev).manual_seed(1234 + a.shard)
    cfgs = [c for c in CONFIGS if c[0] in heads]
    acc = {c[1]: dict(nll=[], cov=[], cov05=[], l2=[], t=0.0) for c in cfgs}
    t0, n_done = time.time(), 0
    for imgs, bboxes, gazex, gazey, _, _, _ in dl:
        imgs, bb_ = imgs.to(dev), [[b] for b in bboxes]
        ys = [np.array([gazex[i][0], gazey[i][0]]) for i in range(len(bboxes))]
        toks = {h: models[h].trunk.tokens(imgs, bb_) for h in heads}
        for head, key, kw, _nfe in cfgs:
            ts = time.time()
            S = models[head].head.sample(toks[head], a.n_samples, gen=gen, **kw).float().cpu().numpy().astype(np.float64)
            hy_n = w_n = None
            if head == 'diff':  # 原生假设：对刚采样用的 pmf 做 NMS（predict 读 last_pmf）
                hy, w = models[head].head.predict(toks[head])
                hy_n, w_n = hy.cpu().numpy().astype(np.float64), w.cpu().numpy().astype(np.float64)
            acc[key]['t'] += time.time() - ts
            for i in range(S.shape[0]):
                nll, cov, cov05, l2 = instance_row(head, S[i], ys[i], None if hy_n is None else (hy_n[i], w_n[i]))
                acc[key]['nll'].append(nll)
                acc[key]['cov'].append(cov)
                acc[key]['cov05'].append(cov05)
                acc[key]['l2'].append(l2)
        n_done += len(bboxes)
        print(f'[shard {a.shard}] {n_done}/{len(ds.data_idxs)}  {time.time() - t0:.0f}s', flush=True)
    import os
    os.makedirs(a.out, exist_ok=True)
    out = {'idx': np.array(sel)}
    for key, d in acc.items():
        out[f'{key}/nll'] = np.array(d['nll'])
        out[f'{key}/cov'] = np.array(d['cov'])
        out[f'{key}/cov05'] = np.array(d['cov05'])
        out[f'{key}/l2'] = np.array(d['l2'])
        out[f'{key}/sample_time_s'] = np.array(d['t'])
    path = f'{a.out}/shard{a.shard}of{a.nshards}{"_smoke" if a.limit else ""}.npz'
    np.savez(path, **out)
    print('saved', path, ' 采样用时(s)：', {k: round(v['t'], 1) for k, v in acc.items()}, flush=True)


if __name__ == '__main__':
    main()
