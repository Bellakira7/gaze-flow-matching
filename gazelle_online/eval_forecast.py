"""forecast 在 val 上的逐实例评测（C19）。python -I -u eval_forecast.py --name fc_h30_s --ckpt <epoch_7.pt> --ctx 0 [--H 30] [--shard 0 --nshards 4]
10000 样本、采样器同 sampler_selection.json；输出 experiments/forecast/eval/<name>_shard<i>of<n>.npz：
t,k（轨迹、帧下标）、inn（t+H 画面内）、p_out、nll_in[N,3]（σ_s=1,2,3，出画为 NaN）、hit10、hit05、l2（hyps[0]，σ_h=1）。
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
from train_online import AmpBackbone  # noqa: E402
from common import IN_SIZE  # noqa: E402
import gaze_eval as GE  # noqa: E402
import metrics as M  # noqa: E402
from g2_eval import pmf_from_samples  # noqa: E402
from gaze_forecast import VATForecast, ForecastModel, collate  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ctx', default='0')
    ap.add_argument('--H', type=int, default=30)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=f'{G}/experiments/forecast/eval')
    a = ap.parse_args()
    sel = json.load(open(f'{G}/experiments/sampler_selection.json'))
    kw, sigma_h = sel['flow'], sel['sigma_h']
    ctx = [int(v) for v in a.ctx.split(',')]
    bb = AmpBackbone('dinov2_vitl14')
    model = ForecastModel(bb, C=len(ctx), depth=a.depth).cuda().eval()
    model.load_decoder_state(torch.load(a.ckpt, map_location='cpu', weights_only=True))
    ds = VATForecast('val', a.H, ctx, bb.get_transform(IN_SIZE))
    ds.items = ds.items[a.shard::a.nshards]
    if a.limit:
        ds.items = ds.items[:a.limit]
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, collate_fn=collate, num_workers=6)
    gen = torch.Generator(device='cuda').manual_seed(1234 + a.shard)
    R = {k: [] for k in ('t', 'k', 'inn', 'p_out', 'nll_in', 'hit10', 'hit05', 'l2')}
    t0 = time.time()
    for imgs, bn, y, inn, crops, meta in dl:
        tok = model.trunk.tokens(imgs.cuda(), [[b] for b in bn], crops.cuda())
        po = torch.sigmoid(model.out_logit(tok)).clamp(1e-4, 1 - 1e-4).cpu().numpy()
        mask = inn > 0.5
        S = model.head.sample(tok[mask.cuda()], a.n, gen=gen, **kw).float().cpu().numpy().astype(np.float64) if mask.any() else None
        si = 0
        for i in range(len(bn)):
            R['t'].append(meta[i][0])
            R['k'].append(meta[i][1])
            R['inn'].append(bool(mask[i]))
            R['p_out'].append(float(po[i]))
            if mask[i]:
                yy = y[i].numpy().astype(np.float64)
                hy, _ = GE.samples_to_hyps(S[si], sigma_h)
                d = np.hypot(*(hy - yy).T)
                R['nll_in'].append([M.nll(pmf_from_samples(S[si], s), yy[None]) for s in GE.HOLD_SIGMAS])
                R['hit10'].append(float(d.min() <= 0.10))
                R['hit05'].append(float(d.min() <= 0.05))
                R['l2'].append(float(d[0]))
                si += 1
            else:
                R['nll_in'].append([np.nan] * 3)
                R['hit10'].append(np.nan)
                R['hit05'].append(np.nan)
                R['l2'].append(np.nan)
        if len(R['t']) % 160 < a.bs:
            print(f'[{a.name} shard {a.shard}] {len(R["t"])}/{len(ds)}  {time.time() - t0:.0f}s', flush=True)
    os.makedirs(a.out, exist_ok=True)
    tag = f'{a.name}_shard{a.shard}of{a.nshards}' + ('_smoke' if a.limit else '')
    np.savez(f'{a.out}/{tag}.npz', **{k: np.array(v) for k, v in R.items()})
    print('saved', f'{a.out}/{tag}.npz', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
