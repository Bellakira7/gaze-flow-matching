"""测试集预测：对一个训练好的头，在 GazeFollow 测试集（4782 实例，真值头框）上生成 G2 要求的 pred.npz，并可直接走 g2_eval 得到 bundle。

python -I -u predict_test.py --head flow --ckpt <epoch_14.pt> --out results/g2/pred_flow.npz [--bundle results/g2/bundle_flow.npz]
        [--hyps cluster|native] [--sigma_h 2.0] [--n 10000] [--sample_kw '{"steps":16}'] [--limit N] [--tag_check]
        [--box gt|eye] [--frac 0.12]
        --head heatmap：B1 热图模型走通用路径（样本=从 raw pmf 抽样，假设=NMS 峰），用来核对本脚本 + g2_eval 与 g2_bundles 的 B1 数字一致

用缓存的测试特征（feats/gf_test_vitl14，与官方整网前向 argmax 100% 一致）。实例顺序 = 帧顺序 × 帧内 head 顺序，与 g2_eval.load_context 一致。
**测试集读数是一次性的（预注册 §6.4）：对方法只在调参全部完成后跑；本脚本的 --limit 只用于冒烟，且不保存。**
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
sys.path.insert(0, f'{G}/gazelle_cached')
sys.path.insert(0, f'{G}/setmetrics')
from common import CachedFeatureBackbone  # noqa: E402
import gaze_eval as GE  # noqa: E402
import metrics as M  # noqa: E402
from gaze_models import GazeModel, HeatmapGaze  # noqa: E402
import gaze_diffusion as GD  # noqa: E402
from eval_holdout_ckpt import heat_native_hyps  # noqa: E402


def head_box(fr, h, kind, frac):
    if kind == 'gt':
        return h['bbox_norm']
    W, H = fr['width'], fr['height']
    s = frac * np.sqrt(W * H)  # 与 tools/eval_ckpt_test.py 的 EyeBoxTest、训练补框 build_gazefollow_train_from_sd.py 同一公式
    cx, cy = h['eye_norm'][0] * W, h['eye_norm'][1] * H
    return (max(cx - s / 2, 0) / W, max(cy - s / 2, 0) / H, min(cx + s / 2, W) / W, min(cy + s / 2, H) / H)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head', required=True, choices=['heatmap', 'mdn', 'mcl', 'flow', 'diff'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--head_kw', default='{}')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--hyps', default=None, choices=['cluster', 'native'], help='缺省：mcl/diff/heatmap 用原生，mdn/flow 用样本聚类')
    ap.add_argument('--sigma_h', type=float, default=2.0)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--sample_kw', default='{}')
    ap.add_argument('--bs', type=int, default=8, help='每批帧数')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 帧（冒烟，不保存）')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--box', default='gt', choices=['gt', 'eye'], help='测试时的头框来源：gt = 真值头框（判定口径）；eye = 眼点为中心的正方形（训练口径），见 prereg/G2.clarifications_20261022.md §C6')
    ap.add_argument('--frac', type=float, default=0.12, help='--box eye 的边长 = frac*sqrt(W*H)，与训练补框一致')
    a = ap.parse_args()
    dev = 'cuda'
    mode = a.hyps or ('native' if a.head in GD.NATIVE_HYPS + ('heatmap',) else 'cluster')
    bb = CachedFeatureBackbone(1024)
    model = (HeatmapGaze(bb) if a.head == 'heatmap' else GazeModel(bb, a.head, **json.loads(a.head_kw))).to(dev).eval()
    sd = torch.load(a.ckpt, map_location='cpu', weights_only=True)
    if a.head == 'heatmap':
        model.load_state_dict(sd, strict=True)
    else:
        model.load_decoder_state(sd)
    frames = json.load(open(f'{G}/data/gazefollow_test_hf/test_preprocessed.json'))
    feats = np.load(f'{G}/feats/gf_test_vitl14/test_orig.npy', mmap_mode='r')
    nf = a.limit or len(frames)
    skw = json.loads(a.sample_kw)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    hyps_l, w_l, S_l, pm_l = [], [], [], []
    t0 = time.time()
    for f0 in range(0, nf, a.bs):
        idx = list(range(f0, min(f0 + a.bs, nf)))
        x = torch.from_numpy(np.stack([np.array(feats[i]) for i in idx])).to(dev).float()
        boxes = [[head_box(frames[i], h, a.box, a.frac) for h in frames[i]['heads']] for i in idx]
        if a.head == 'heatmap':
            hm = torch.cat(model.forward({'images': x, 'bboxes': boxes})['heatmap'], 0).float().cpu().numpy()
            S = model.sample(x, boxes, a.n, gen=gen)
        else:
            tok = model.trunk.tokens(x, boxes)
            S = model.head.sample(tok, a.n, gen=gen, **skw)
            if a.head == 'diff':
                pm_l.append(model.head.last_pmf.cpu().numpy())  # 精确 pmf（只用于报告精确 NLL，不参与判定）
            if mode == 'native':
                hy, w = model.head.predict(tok)
                hy, w = hy.cpu().numpy(), w.cpu().numpy()
        S = S.float().cpu().numpy()
        for j in range(S.shape[0]):
            if a.head == 'heatmap':
                hh, ww = heat_native_hyps(hm[j])
            elif mode == 'native':
                hh, ww = hy[j], w[j]
            else:
                hh, ww = GE.samples_to_hyps(S[j].astype(np.float64), a.sigma_h)
            hyps_l.append(hh)
            w_l.append(ww)
            S_l.append(S[j])
        if (f0 // a.bs) % 40 == 0:
            print(f'  frames {f0 + len(idx)}/{nf}  instances {len(S_l)}  {time.time() - t0:.0f}s', flush=True)
    pred = dict(hyps=np.stack(hyps_l), w=np.stack(w_l), samples=np.stack(S_l).astype(np.float32))
    if pm_l:
        pred['pmf'] = np.concatenate(pm_l).reshape(-1, 64, 64)
    print(f'预测完成 {len(S_l)} 实例，{time.time() - t0:.0f}s；hyps[0] 点估计均值 {pred["hyps"][:, 0].mean(0)}；假设模式 {mode}', flush=True)
    if a.limit:
        print('--limit 冒烟，不保存')
        return
    np.savez(a.out if a.out.startswith('/') else f'{G}/{a.out}', **pred)
    if a.bundle:
        sys.path.insert(0, f'{G}/setmetrics')
        import g2_eval as E
        b = E.evaluate_prediction(pred)
        np.savez(a.bundle if a.bundle.startswith('/') else f'{G}/{a.bundle}', **b)
        _, _, _, subsets, _ = E.load_context()
        for sn, sel in subsets.items():
            print(f'[{a.head}/{sn}] cov0.10@5 {b["cov0.1@5"][sel].mean():.4f}  NLL(cf σ_s) {b["nll_variants"][sel, 0].mean():.4f}'
                  + (f'  AUC {b["auc"].mean():.4f} AvgL2 {b["avg_l2_fix"].mean():.4f} MinL2 {b["min_l2_fix"].mean():.4f}' if sn == 'all' else ''), flush=True)


if __name__ == '__main__':
    main()
