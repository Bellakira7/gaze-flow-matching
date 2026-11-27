"""hold-out 诊断（11-02，只用训练 hold-out，不碰测试集）：主模型 flowET_d6 还差在哪——校准 / 温度 / 集成。
python -I -u diag_calib_holdout.py --shard 0 --nshards 4   （4–7 卡各一个 shard）→ experiments/diag_calib/shard<i>of4.npz
配置（每实例 10000 样本，采样器读 sampler_selection.json 的 flow 配置）：
  d6_t{0.85,1.0,1.2}：flowET_d6，初始噪声乘以温度 τ（τ>1 更散）；flowET_t1.0、eyefuse_t1.0；
  mix_d6_flowET、mix_d6_eyefuse：两模型各取前 5000 个样本拼成 10000（等权混合 = 两模型集成）。
每实例每配置存：nll[3]（σ_s=1,2,3）；hpd_m = 包含标注所在格的最高密度区域的质量（σ_s=1；校准良好时 ~U(0,1)）；
  area[Q] = 质量 q 的最高密度区域面积（格数/4096，q 取 QGRID，σ_s=1），供离线做 split-conformal。
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_online')
sys.path.insert(0, f'{G}/setmetrics')
from train_online import AmpBackbone, SDGazeDataset  # noqa: E402
from common import IN_SIZE  # noqa: E402
from gazelle.dataloader import collate_fn  # noqa: E402
import gaze_eval as GE  # noqa: E402
import metrics as M  # noqa: E402
from g2_eval import pmf_from_samples  # noqa: E402
from gaze_eye import EyeGazeModel  # noqa: E402
from gaze_eye2 import EyeGazeModel2  # noqa: E402

QGRID = np.round(np.arange(0.05, 1.0, 0.05), 2)  # 0.05..0.95
TEMPS = (0.85, 1.0, 1.2)


@torch.no_grad()
def sample_temp(head, tokens, n, gen, steps, gamma, solver, tau, chunk=2500):
    """与 gaze_models.FlowHead.sample 逐行相同，唯一差别：初始噪声乘以 tau。"""
    kvs = head.dec.make_kv(tokens)
    B = tokens.shape[0]
    ts = (1 - (1 - torch.linspace(0, 1, steps + 1, device=tokens.device)) ** gamma).tolist()
    outs = []
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        x = torch.randn(B, m, 2, device=tokens.device, generator=gen) * tau
        for i in range(steps):
            t0, t1 = ts[i], ts[i + 1]
            v = head.velocity(x, torch.full((B, 1), t0, device=tokens.device), kvs)
            if solver == 'euler':
                x = x + (t1 - t0) * v
            else:
                xe = x + (t1 - t0) * v
                v2 = head.velocity(xe, torch.full((B, 1), t1, device=tokens.device), kvs)
                x = x + (t1 - t0) * 0.5 * (v + v2)
        outs.append(((x + 1) / 2).clamp(0, 1))
    return torch.cat(outs, 1)


def inst_stats(S, y):
    nll = [M.nll(pmf_from_samples(S, s), y[None]) for s in GE.HOLD_SIGMAS]
    P = pmf_from_samples(S, 1.0)
    py = P[M.cell_index(np.clip(y[None], 0, 1 - 1e-9))[0]]
    hpd_m = float(P[P >= py].sum())
    cs = np.cumsum(np.sort(P)[::-1])
    area = (np.searchsorted(cs, QGRID) + 1) / P.size
    return nll, hpd_m, area


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=4)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=f'{G}/experiments/diag_calib')
    a = ap.parse_args()
    kw = json.load(open(f'{G}/experiments/sampler_selection.json'))['flow']
    bb = AmpBackbone('dinov2_vitl14')
    E = f'{G}/experiments'
    models = {'d6': EyeGazeModel(bb, 'flow', depth=6), 'flowET': EyeGazeModel(bb, 'flow'), 'eyefuse': EyeGazeModel2(bb, 'fuse')}
    ck = {'d6': f'{E}/flow_eye_d6_s0/epoch_14.pt', 'flowET': f'{E}/flow_eye_s0/epoch_14.pt', 'eyefuse': f'{E}/eyefuse_s0/epoch_14.pt'}
    for k, m in models.items():
        m.cuda().eval().load_decoder_state(torch.load(ck[k], map_location='cpu', weights_only=True))
    ds = SDGazeDataset(f'{G}/data/gazefollow_train_sd', 'holdout', bb.get_transform(IN_SIZE))
    sel = list(range(len(ds.data_idxs)))[a.shard::a.nshards]
    if a.limit:
        sel = sel[:a.limit]
    ds.data_idxs = [ds.data_idxs[i] for i in sel]
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, collate_fn=collate_fn, num_workers=6)
    gen = torch.Generator(device='cuda').manual_seed(4321 + a.shard)
    cfgs = [f'd6_t{t:g}' for t in TEMPS] + ['flowET_t1', 'eyefuse_t1', 'mix_d6_flowET', 'mix_d6_eyefuse']
    R = {c: dict(nll=[], hpd_m=[], area=[]) for c in cfgs}
    t0 = time.time()
    for imgs, bboxes, gazex, gazey, _, _, _ in dl:
        boxes = [[b] for b in bboxes]
        x = imgs.cuda()
        S = {}
        tok = models['d6'].trunk.tokens(x, boxes)
        for t in TEMPS:
            S[f'd6_t{t:g}'] = sample_temp(models['d6'].head, tok, a.n, gen, kw['steps'], kw['gamma'], kw['solver'], t).float().cpu().numpy().astype(np.float64)
        S['flowET_t1'] = sample_temp(models['flowET'].head, models['flowET'].trunk.tokens(x, boxes), a.n, gen, kw['steps'], kw['gamma'], kw['solver'], 1.0).float().cpu().numpy().astype(np.float64)
        S['eyefuse_t1'] = sample_temp(models['eyefuse'].head, models['eyefuse'].trunk.tokens(x, boxes), a.n, gen, kw['steps'], kw['gamma'], kw['solver'], 1.0).float().cpu().numpy().astype(np.float64)
        h = a.n // 2
        S['mix_d6_flowET'] = np.concatenate([S['d6_t1'][:, :h], S['flowET_t1'][:, :h]], 1)
        S['mix_d6_eyefuse'] = np.concatenate([S['d6_t1'][:, :h], S['eyefuse_t1'][:, :h]], 1)
        for i in range(x.shape[0]):
            y = np.array([gazex[i][0], gazey[i][0]])
            for c in cfgs:
                nll, hm, ar = inst_stats(S[c][i], y)
                R[c]['nll'].append(nll)
                R[c]['hpd_m'].append(hm)
                R[c]['area'].append(ar)
        n_done = len(R[cfgs[0]]['nll'])
        if n_done % 160 < a.bs:
            print(f'[shard {a.shard}] {n_done}/{len(sel)}  {time.time() - t0:.0f}s', flush=True)
    os.makedirs(a.out, exist_ok=True)
    out = {'idx': np.array(sel), 'qgrid': QGRID}
    for c in cfgs:
        for k, v in R[c].items():
            out[f'{c}__{k}'] = np.array(v)
    tag = f'shard{a.shard}of{a.nshards}' + ('_smoke' if a.limit else '')
    np.savez(f'{a.out}/{tag}.npz', **out)
    print('saved', f'{a.out}/{tag}.npz', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
