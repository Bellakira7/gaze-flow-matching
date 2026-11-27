"""C20 forecast 逐实例评测（val / cal / test）：联合 NLL 与 conformal 所需的分数 / 区域大小。
python -I -u eval_forecast_c20.py --name c20_ours_s0 --ckpt <epoch_k.pt> --head flow --eye 1 --depth 6 --split val|cal|test [--read_test] [--shard i --nshards 4]
**--split test 必须同时给 --read_test**（C20：每个臂只读一次测试集，由 run_forecast_c20.sh 调用）。
每实例（含画面外实例）都抽 10000 个样本（采样器读 sampler_selection.json，只对 flow 头生效），存 experiments/forecast/c20_eval/<name>_<split>_shard<i>of<n>.npz：
  t, k, clip, inn, p_out；nll_in[N,3]（σ_s=1,2,3，画面外为 NaN）；hit10 / hit05 / l2（hyps[0]；MCL 用原生 5 假设，热图用 NMS 前 5 峰，其余样本聚类 σ_h）；
  conformal（联合分布 = (1−p_out)·P_in(格) ∪ {出画: p_out}，4097 个原子，σ_s=1,2,3 各一份）：
    score[N,3] = 包含式 HPD 分数 Σ_{P(a) ≥ P(y)} P(a)（y 为标注所在原子，出画时为出画原子）；
    area[N,3,99] = 区域 {a : 包含式累计质量 ≤ q} 里画面内格数 / 4096，q = 0.01..0.99；out_in[N,3,99] = 出画原子是否在区域里。
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
from gaze_forecast import VATForecast, ForecastAny, collate  # noqa: E402

QG = np.round(np.arange(0.01, 1.0, 0.01), 2)


def conformal_row(P_in, p_out, y_cell):
    """P_in [4096]，y_cell = 格下标或 -1（出画）→ score, area[99], out_in[99]。"""
    J = np.concatenate([(1 - p_out) * P_in, [p_out]])
    o = np.argsort(-J, kind='stable')
    cum = np.cumsum(J[o])
    cum_of = np.empty_like(cum)
    cum_of[o] = cum
    ya = 4096 if y_cell < 0 else y_cell
    score = float(cum_of[ya])
    in_cum = np.sort(cum_of[:4096])
    area = np.searchsorted(in_cum, QG, side='right') / 4096
    out_in = cum_of[4096] <= QG
    return score, area, out_in


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--head', required=True, choices=['flow', 'mdn', 'mcl', 'heatmap'])
    ap.add_argument('--eye', type=int, required=True)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--H', type=int, default=30)
    ap.add_argument('--split', required=True, choices=['val', 'cal', 'test'])
    ap.add_argument('--read_test', action='store_true')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=f'{G}/experiments/forecast/c20_eval')
    a = ap.parse_args()
    assert a.split != 'test' or a.read_test, '读测试集须显式 --read_test（C20）'
    sel = json.load(open(f'{G}/experiments/sampler_selection.json'))
    fkw, sigma_h = sel['flow'], sel['sigma_h']
    bb = AmpBackbone('dinov2_vitl14')
    model = ForecastAny(bb, a.head, eye=bool(a.eye), **({'depth': a.depth} if a.head == 'flow' else {})).cuda().eval()
    model.load_decoder_state(torch.load(a.ckpt, map_location='cpu', weights_only=True))
    ds = VATForecast(a.split, a.H, (0,), bb.get_transform(IN_SIZE), scheme='c20', allow_test=a.split == 'test')
    ds.items = ds.items[a.shard::a.nshards]
    if a.limit:
        ds.items = ds.items[:a.limit]
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, collate_fn=collate, num_workers=6)
    gen = torch.Generator(device='cuda').manual_seed(1234 + a.shard)
    R = {k: [] for k in ('t', 'k', 'clip', 'inn', 'p_out', 'nll_in', 'hit10', 'hit05', 'l2', 'score', 'area', 'out_in')}
    t0 = time.time()
    for imgs, bn, y, inn, crops, meta in dl:
        tok = model.tokens(imgs.cuda(), [[b] for b in bn], crops.cuda())
        po = torch.sigmoid(model.out_logit(tok)).clamp(1e-4, 1 - 1e-4).cpu().numpy()
        S = model.sample(tok, a.n, gen=gen, **fkw).float().cpu().numpy().astype(np.float64)
        if a.head == 'mcl':
            nh, _ = model.head.predict(tok)
            nh = nh.cpu().numpy()
        elif a.head == 'heatmap':
            hm = model.head.heat(tok).float().cpu().numpy()
        for i in range(len(bn)):
            R['t'].append(meta[i][0])
            R['k'].append(meta[i][1])
            R['clip'].append(ds.tracks[meta[i][0]]['clip'])
            R['inn'].append(bool(inn[i] > 0.5))
            R['p_out'].append(float(po[i]))
            Ps = [pmf_from_samples(S[i], s) for s in GE.HOLD_SIGMAS]
            if inn[i] > 0.5:
                yy = y[i].numpy().astype(np.float64)
                yc = int(M.cell_index(np.clip(yy[None], 0, 1 - 1e-9))[0])
                if a.head == 'mcl':
                    hy = nh[i]
                elif a.head == 'heatmap':
                    hy = M.CELLS[M.nms_peaks(hm[i], 5)]
                else:
                    hy, _ = GE.samples_to_hyps(S[i], sigma_h)
                d = np.hypot(*(hy - yy).T)
                R['nll_in'].append([float(-np.log(P[yc])) for P in Ps])
                R['hit10'].append(float(d.min() <= 0.10))
                R['hit05'].append(float(d.min() <= 0.05))
                R['l2'].append(float(d[0]))
            else:
                yc = -1
                R['nll_in'].append([np.nan] * 3)
                R['hit10'].append(np.nan)
                R['hit05'].append(np.nan)
                R['l2'].append(np.nan)
            cr = [conformal_row(P, float(po[i]), yc) for P in Ps]
            R['score'].append([c[0] for c in cr])
            R['area'].append([c[1] for c in cr])
            R['out_in'].append([c[2] for c in cr])
        if len(R['t']) % 160 < a.bs:
            print(f'[{a.name}/{a.split} shard {a.shard}] {len(R["t"])}/{len(ds)}  {time.time() - t0:.0f}s', flush=True)
    os.makedirs(a.out, exist_ok=True)
    tag = f'{a.name}_{a.split}_shard{a.shard}of{a.nshards}' + ('_smoke' if a.limit else '')
    np.savez(f'{a.out}/{tag}.npz', **{k: np.array(v) for k, v in R.items()})
    print('saved', f'{a.out}/{tag}.npz', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
