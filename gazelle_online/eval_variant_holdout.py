"""训练 hold-out（3762 实例，单标注）上对一个 flow ckpt 做逐实例评测：10000 样本、采样器与 σ_h 读 experiments/sampler_selection.json（C8 选定的，不重选）。
用途：给 flow 的 seed 间波动、以及眼部 token 变体与基线的**配对**比较提供同一口径的逐实例数字。只用 hold-out，不碰测试集。
python -I -u eval_variant_holdout.py --variant base|eye --ckpt <epoch_14.pt> --name flow_s0 [--shard 0 --nshards 4]
输出 experiments/holdout_eval/<name>_shard<i>of<n>.npz：idx[N]、nll[N,3]（σ_s=1,2,3）、cov[N]、cov05[N]、l2[N]（σ_h 取选定值）。
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
from train_online import AmpBackbone, SDGazeDataset  # noqa: E402
from common import IN_SIZE  # noqa: E402
from gazelle.dataloader import collate_fn  # noqa: E402
from gaze_models import GazeModel  # noqa: E402
from gaze_eye import EyeGazeModel  # noqa: E402
from tune_sampler_holdout import instance_row, SIGMA_H  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['base', 'eye'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--eye_kw', default='{}')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--n_samples', type=int, default=10000)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=f'{G}/experiments/holdout_eval')
    a = ap.parse_args()
    sel_cfg = json.load(open(f'{G}/experiments/sampler_selection.json'))
    kw, sigma_h = sel_cfg['flow'], sel_cfg['sigma_h']
    col = SIGMA_H.index(sigma_h)
    bb = AmpBackbone('dinov2_vitl14')
    model = (EyeGazeModel(bb, 'flow', eye_kw=json.loads(a.eye_kw)) if a.variant == 'eye' else GazeModel(bb, 'flow')).cuda().eval()
    model.load_decoder_state(torch.load(a.ckpt, map_location='cpu', weights_only=True))
    ds = SDGazeDataset(f'{G}/data/gazefollow_train_sd', 'holdout', bb.get_transform(IN_SIZE))
    sel = list(range(len(ds.data_idxs)))[a.shard::a.nshards]
    if a.limit:
        sel = sel[:a.limit]
    ds.data_idxs = [ds.data_idxs[i] for i in sel]
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, collate_fn=collate_fn, num_workers=a.workers)
    gen = torch.Generator(device='cuda').manual_seed(1234 + a.shard)
    rows, t0 = [], time.time()
    for imgs, bboxes, gazex, gazey, _, _, _ in dl:
        tok = model.trunk.tokens(imgs.cuda(), [[b] for b in bboxes])
        S = model.head.sample(tok, a.n_samples, gen=gen, **kw).float().cpu().numpy().astype(np.float64)
        for i in range(S.shape[0]):
            y = np.array([gazex[i][0], gazey[i][0]])
            nll, cov, cov05, l2 = instance_row('flow', S[i], y)
            rows.append((nll, cov[col], cov05[col], l2[col]))
        if len(rows) % 320 < a.bs:
            print(f'[{a.name} shard {a.shard}] {len(rows)}/{len(sel)}  {time.time() - t0:.0f}s', flush=True)
    os.makedirs(a.out, exist_ok=True)
    tag = f'{a.name}_shard{a.shard}of{a.nshards}' + ('_smoke' if a.limit else '')
    np.savez(f'{a.out}/{tag}.npz', idx=np.array(sel), nll=np.array([r[0] for r in rows]), cov=np.array([r[1] for r in rows]),
             cov05=np.array([r[2] for r in rows]), l2=np.array([r[3] for r in rows]))
    print('saved', f'{a.out}/{tag}.npz', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
