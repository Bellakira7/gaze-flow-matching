"""对一个 ckpt 在训练 hold-out（单标注，3762 实例）上评测，所有头走同一条口径：
每实例 n_samples 个样本 -> pmf(σ_s∈{1,2,3} 取均值最低) -> NLL；假设 = 原生（heatmap: NMS 取峰；mcl: 原生）或样本聚类（mdn/flow）。
python -I -u eval_holdout_ckpt.py --head heatmap --ckpt <epoch_14.pt>   [--head flow --ckpt ...] [--limit N]
只用 hold-out，不碰测试集。
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_online')
from train_online import AmpBackbone, SDGazeDataset  # noqa: E402
from common import IN_SIZE  # noqa: E402
from gazelle.dataloader import collate_fn  # noqa: E402
import gaze_eval as GE  # noqa: E402
import metrics as M  # noqa: E402
from gaze_models import GazeModel, HeatmapGaze  # noqa: E402
import gaze_diffusion as GD  # noqa: E402


def heat_native_hyps(h, K=5):
    """B1 原生假设：热图上 NMS 取峰，权重 = 峰周围半径 0.1 圆盘内的 pmf 质量（S1 定义），归一化后降序。"""
    P = M.heat_to_pmf(h, 1.0)
    pk = M.nms_peaks(h, K)
    w = np.array([M.disc_mass(P, i) for i in pk])
    o = np.argsort(-w, kind='stable')
    return M.CELLS[pk][o], w[o] / w.sum()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head', required=True, choices=['heatmap', 'mdn', 'mcl', 'flow', 'diff'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--head_kw', default='{}')
    ap.add_argument('--n_samples', type=int, default=1000)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--sample_kw', default='{}', help='传给 head.sample 的参数（flow: steps/gamma/solver）')
    a = ap.parse_args()
    dev = 'cuda'
    bb = AmpBackbone('dinov2_vitl14')
    model = (HeatmapGaze(bb) if a.head == 'heatmap' else GazeModel(bb, a.head, **json.loads(a.head_kw))).to(dev).eval()
    sd = torch.load(a.ckpt, map_location='cpu', weights_only=True)
    if a.head == 'heatmap':
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith('backbone') for k in missing), (missing[:3], unexpected[:3])
    else:
        model.load_decoder_state(sd)
    ds = SDGazeDataset(f'{G}/data/gazefollow_train_sd', 'holdout', model.backbone.get_transform(IN_SIZE) if a.head == 'heatmap' else model.trunk.backbone.get_transform(IN_SIZE))
    if a.limit:
        ds.data_idxs = ds.data_idxs[:a.limit]
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, collate_fn=collate_fn, num_workers=12)
    gen = torch.Generator(device=dev).manual_seed(1234)
    skw = json.loads(a.sample_kw)
    rows = []
    for imgs, bboxes, gazex, gazey, _, _, _ in dl:
        imgs, bb_ = imgs.to(dev), [[b] for b in bboxes]
        if a.head == 'heatmap':
            hm = torch.cat(model.forward({'images': imgs, 'bboxes': bb_})['heatmap'], 0).float().cpu().numpy()
            S = model.sample(imgs, bb_, a.n_samples, gen=gen).float().cpu().numpy()
        else:
            tok = model.trunk.tokens(imgs, bb_)
            S = model.head.sample(tok, a.n_samples, gen=gen, **skw).float().cpu().numpy()
            if a.head in GD.NATIVE_HYPS:
                hy, w = model.head.predict(tok)
                hy, w = hy.cpu().numpy(), w.cpu().numpy()
        for i in range(S.shape[0]):
            y = np.array([gazex[i][0], gazey[i][0]])
            if a.head == 'heatmap':
                hyps, w_ = heat_native_hyps(hm[i])
                rows.append(GE.holdout_instance(S[i], y, hyps, w_))
            elif a.head in GD.NATIVE_HYPS:
                rows.append(GE.holdout_instance(S[i], y, hy[i], w[i]))
            else:
                rows.append(GE.holdout_instance(S[i], y))
    res = GE.summarize(rows)
    print(a.head, a.ckpt.split('/')[-2:], json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
