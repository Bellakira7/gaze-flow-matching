"""训练 hold-out（3762 实例，单标注）逐实例评测，覆盖射线头与 flow / flowET（与 eval_variant_holdout.py 同口径：10000 样本、采样器与 σ_h 读
experiments/sampler_selection.json，种子 1234+shard、batch 16，输出格式相同，compare_holdout.py 直接可读）。
比 eval_variant_holdout.py 多存：hyp0[N,2]（点估计）、y[N,2]（标注）、center[N,2]（头框中心）、p_out[N]（射线头的 P(画面外)，其它头为 NaN）、wh[N,2]（原图宽高），
用于"方向误差 / 径向误差"的机制检查（tools/ray_decomp_analysis.py 的口径）。只用 hold-out，不碰测试集。

python -I -u eval_ray_holdout.py --variant ray|raytok|base|eye|mixflow --ckpt <epoch_14.pt> --name ray_eye_s0 [--no_eye] [--ray_kw JSON] [--head_kw JSON] [--shard 0 --nshards 4]
  --variant raytok：v2 射线 token + FlowHead（C16）。
  --variant eyefuse / eyehr：眼部 token 变体 2（gaze_eye2，C17）；eyehr 的数据集多返回原图 crop。
  --variant mixflow：集合原生混合 flow 头（C21）；除常规（样本聚类口径）npz 外另存 <name>_native_shard*.npz——
    cov/cov05/l2/hyp0 换成原生假设口径（predict() 合并后），nll 与常规文件相同（同一批样本），供 compare_native_holdout.py 做机制判据。
  --variant base / eye：与 eval_variant_holdout.py 的 flow / flowET 逐位一致（同种子同 batch），用来补存 hyp0 并对拍。
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
import gaze_eval as GE  # noqa: E402
from gaze_models import GazeModel  # noqa: E402
from gaze_eye import EyeGazeModel  # noqa: E402
from gaze_ray import RayGazeModel, RayTokGazeModel, head_centers  # noqa: E402
from gaze_eye2 import EyeGazeModel2, HiResCropDataset  # noqa: E402
import gaze_mixflow  # noqa: E402,F401  (注册 'mixflow' 头，C21)
from tune_sampler_holdout import instance_row, SIGMA_H  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['ray', 'raytok', 'base', 'eye', 'eyefuse', 'eyehr', 'mixflow'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--eye_kw', default='{}')
    ap.add_argument('--ray_kw', default='{}')
    ap.add_argument('--head_kw', default='{}', help='base / eye / raytok 时传给 FlowHead（如 {"depth": 6}）')
    ap.add_argument('--no_eye', action='store_true')
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
    if a.variant == 'ray':
        model = RayGazeModel(bb, eye=not a.no_eye, eye_kw=json.loads(a.eye_kw), **json.loads(a.ray_kw))
    elif a.variant == 'raytok':
        model = RayTokGazeModel(bb, eye_kw=json.loads(a.eye_kw), ray_kw=json.loads(a.ray_kw), **json.loads(a.head_kw))
    elif a.variant in ('eyefuse', 'eyehr'):
        model = EyeGazeModel2(bb, {'eyefuse': 'fuse', 'eyehr': 'hires'}[a.variant], eye_kw=json.loads(a.eye_kw), **json.loads(a.head_kw))
    elif a.variant == 'mixflow':
        model = EyeGazeModel(bb, 'mixflow', eye_kw=json.loads(a.eye_kw), **json.loads(a.head_kw))
    elif a.variant == 'eye':
        model = EyeGazeModel(bb, 'flow', eye_kw=json.loads(a.eye_kw), **json.loads(a.head_kw))
    else:
        model = GazeModel(bb, 'flow', **json.loads(a.head_kw))
    model = model.cuda().eval()
    model.load_decoder_state(torch.load(a.ckpt, map_location='cpu', weights_only=True))
    hires = a.variant == 'eyehr'
    if hires:
        ek = json.loads(a.eye_kw)
        ds = HiResCropDataset(f'{G}/data/gazefollow_train_sd', 'holdout', bb.get_transform(IN_SIZE), margin=ek.get('margin', 1.25), crop_size=ek.get('crop_size', 224))
    else:
        ds = SDGazeDataset(f'{G}/data/gazefollow_train_sd', 'holdout', bb.get_transform(IN_SIZE))
    sel = list(range(len(ds.data_idxs)))[a.shard::a.nshards]
    if a.limit:
        sel = sel[:a.limit]
    ds.data_idxs = [ds.data_idxs[i] for i in sel]
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, collate_fn=collate_fn, num_workers=a.workers)
    gen = torch.Generator(device='cuda').manual_seed(1234 + a.shard)
    rows, extra, rows_nat, extra_nat, t0 = [], [], [], [], time.time()
    for batch in dl:
        imgs, bboxes, gazex, gazey, _, hs, ws = batch[:7]
        boxes = [[b] for b in bboxes]
        tok = model.trunk.tokens(imgs.cuda(), boxes, crops=batch[7].cuda()) if hires else model.trunk.tokens(imgs.cuda(), boxes)
        if a.variant in ('ray', 'raytok'):
            S, pout = model.sample_tokens(tok, boxes, a.n_samples, gen=gen, return_pout=True, **kw)
            pout = pout.cpu().numpy()
        else:
            S, pout = model.head.sample(tok, a.n_samples, gen=gen, **kw), np.full(len(boxes), np.nan)
        if a.variant == 'mixflow':
            hy_n, w_n = model.head.predict(tok)
            hy_n = hy_n.float().cpu().numpy().astype(np.float64)
        S = S.float().cpu().numpy().astype(np.float64)
        cen = head_centers(boxes, 'cpu').numpy()
        for i in range(S.shape[0]):
            y = np.array([gazex[i][0], gazey[i][0]])
            nll, cov, cov05, l2 = instance_row('flow', S[i], y)
            rows.append((nll, cov[col], cov05[col], l2[col]))
            extra.append((GE.samples_to_hyps(S[i], sigma_h)[0][0], y, cen[i], pout[i], (float(ws[i]), float(hs[i]))))
            if a.variant == 'mixflow':  # 原生假设口径：同一批样本的 nll 不变，只换假设
                d = np.hypot(*(hy_n[i] - y).T)
                rows_nat.append((nll, float(d.min() <= 0.10), float(d.min() <= 0.05), float(d[0])))
                extra_nat.append((hy_n[i][0], y, cen[i], pout[i], (float(ws[i]), float(hs[i]))))
        if len(rows) % 320 < a.bs:
            print(f'[{a.name} shard {a.shard}] {len(rows)}/{len(sel)}  {time.time() - t0:.0f}s', flush=True)
    os.makedirs(a.out, exist_ok=True)
    tag = f'{a.name}_shard{a.shard}of{a.nshards}' + ('_smoke' if a.limit else '')

    def save(fn, rr, ee):
        np.savez(fn, idx=np.array(sel), nll=np.array([r[0] for r in rr]), cov=np.array([r[1] for r in rr]),
                 cov05=np.array([r[2] for r in rr]), l2=np.array([r[3] for r in rr]),
                 hyp0=np.array([e[0] for e in ee]), y=np.array([e[1] for e in ee]), center=np.array([e[2] for e in ee]),
                 p_out=np.array([e[3] for e in ee], dtype=float), wh=np.array([e[4] for e in ee]))
        print('saved', fn, f'{time.time() - t0:.0f}s', flush=True)

    save(f'{a.out}/{tag}.npz', rows, extra)
    if a.variant == 'mixflow':
        save(f'{a.out}/{a.name}_native_shard{a.shard}of{a.nshards}' + ('_smoke' if a.limit else '') + '.npz', rows_nat, extra_nat)


if __name__ == '__main__':
    main()
