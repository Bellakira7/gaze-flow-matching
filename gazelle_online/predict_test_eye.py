"""眼部 token 变体的测试集预测（与 predict_test.py 逐段对应，差别只有：模型 = EyeGazeModel；每批多读原图并在线算头部 crop 的 token）。
python -I -u predict_test_eye.py --variant eye --ckpt <flow_eye_s0/epoch_14.pt> --box gt|eye --out results/g2/pred_flowET_gt.npz --bundle results/g2/bundle_flowET_gt.npz
        [--sample_kw JSON] [--sigma_h 1.0] [--n 10000] [--limit N] [--parity_ref results/g2/pred_flow_gt.npz]
  --variant base  ：不带眼部 token 的 flow（GazeModel + 缓存特征），用来对拍 predict_test.py 的存档输出，证明本脚本没有引入别的差异。
  --parity_ref    ：与 --limit 连用，把本次前若干个实例的 hyps/samples 与存档对比并打印最大差异（不保存）。
场景 token 用缓存特征（feats/gf_test_vitl14，与 flow 基线测试路径同源）；crop 从原图经同一 get_transform(448) 后在线过冻结 DINOv2（fp16 autocast，与训练一致）。
**测试集读数是一次性的（G2 预注册 §6.4 / 澄清 C9）：本脚本的 --limit 只用于冒烟与对拍，且不保存。**
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch
from PIL import Image

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_online')
sys.path.insert(0, f'{G}/gazelle_cached')
sys.path.insert(0, f'{G}/setmetrics')
from common import CachedFeatureBackbone, IN_SIZE  # noqa: E402
import gaze_eval as GE  # noqa: E402
from gaze_models import GazeModel  # noqa: E402
from gaze_eye import EyeGazeModel  # noqa: E402
from predict_test import head_box  # noqa: E402  (同一个头框函数，gt / eye 口径与基线逐位一致)
from train_online import AmpBackbone  # noqa: E402


class TestFrames(torch.utils.data.Dataset):
    def __init__(self, frames, feats, transform, n):
        self.frames, self.feats, self.tf, self.n = frames, feats, transform, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = Image.open(f'{G}/data/gazefollow_test_hf/{self.frames[i]["path"]}').convert('RGB')
        return i, torch.from_numpy(np.array(self.feats[i])), self.tf(img)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['base', 'eye'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--sigma_h', type=float, default=2.0)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--sample_kw', default='{}')
    ap.add_argument('--eye_kw', default='{}')
    ap.add_argument('--bs', type=int, default=8, help='每批帧数（与 predict_test.py 默认一致，随机数序列才对得上）')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--box', default='gt', choices=['gt', 'eye'])
    ap.add_argument('--frac', type=float, default=0.12)
    ap.add_argument('--parity_ref', default=None)
    a = ap.parse_args()
    dev = 'cuda'
    cb = CachedFeatureBackbone(1024)
    model = (EyeGazeModel(cb, 'flow', eye_kw=json.loads(a.eye_kw)) if a.variant == 'eye' else GazeModel(cb, 'flow')).to(dev).eval()
    model.load_decoder_state(torch.load(a.ckpt, map_location='cpu', weights_only=True))
    tf = None
    if a.variant == 'eye':
        real = AmpBackbone('dinov2_vitl14').to(dev).eval()
        model.trunk.set_crop_backbone(real)
        tf = real.get_transform(IN_SIZE)
    frames = json.load(open(f'{G}/data/gazefollow_test_hf/test_preprocessed.json'))
    feats = np.load(f'{G}/feats/gf_test_vitl14/test_orig.npy', mmap_mode='r')
    nf = a.limit or len(frames)
    skw = json.loads(a.sample_kw)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    if a.variant == 'eye':
        dl = torch.utils.data.DataLoader(TestFrames(frames, feats, tf, nf), batch_size=a.bs, shuffle=False, num_workers=8)
    else:
        dl = torch.utils.data.DataLoader(TestFrames(frames, feats, lambda im: torch.zeros(1), nf), batch_size=a.bs, shuffle=False, num_workers=4)
    hyps_l, w_l, S_l = [], [], []
    t0 = time.time()
    for bi, (ids, fe, im) in enumerate(dl):
        idx = ids.tolist()
        x = fe.to(dev).float()
        boxes = [[head_box(frames[i], h, a.box, a.frac) for h in frames[i]['heads']] for i in idx]
        tok = model.trunk.tokens(x, boxes, raw_images=im.to(dev)) if a.variant == 'eye' else model.trunk.tokens(x, boxes)
        S = model.head.sample(tok, a.n, gen=gen, **skw).float().cpu().numpy()
        for j in range(S.shape[0]):
            hh, ww = GE.samples_to_hyps(S[j].astype(np.float64), a.sigma_h)
            hyps_l.append(hh)
            w_l.append(ww)
            S_l.append(S[j])
        if bi % 40 == 0:
            print(f'  frames {idx[-1] + 1}/{nf}  instances {len(S_l)}  {time.time() - t0:.0f}s', flush=True)
    pred = dict(hyps=np.stack(hyps_l), w=np.stack(w_l), samples=np.stack(S_l).astype(np.float32))
    print(f'预测完成 {len(S_l)} 实例，{time.time() - t0:.0f}s；hyps[0] 点估计均值 {pred["hyps"][:, 0].mean(0)}；假设模式 cluster(σ_h={a.sigma_h})', flush=True)
    if a.limit:
        if a.parity_ref:
            ref = np.load(a.parity_ref if a.parity_ref.startswith('/') else f'{G}/{a.parity_ref}')
            n = len(S_l)
            ds = np.abs(pred['samples'] - ref['samples'][:n]).max()
            dh = np.abs(pred['hyps'] - ref['hyps'][:n]).max()
            print(f'[对拍] 前 {n} 个实例 与存档 {a.parity_ref.split("/")[-1]}：samples 最大绝对差 {ds:.3e}，hyps 最大绝对差 {dh:.3e}，hyps[0] 完全相同比例 {(np.abs(pred["hyps"][:, 0] - ref["hyps"][:n, 0]).max(1) == 0).mean():.3f}')
        print('--limit 冒烟，不保存')
        return
    np.savez(a.out if a.out.startswith('/') else f'{G}/{a.out}', **pred)
    if a.bundle:
        import g2_eval as E
        b = E.evaluate_prediction(pred)
        np.savez(a.bundle if a.bundle.startswith('/') else f'{G}/{a.bundle}', **b)
        _, _, _, subsets, _ = E.load_context()
        for sn, sel in subsets.items():
            print(f'[flow{"ET" if a.variant == "eye" else ""}/{sn}] cov0.10@5 {b["cov0.1@5"][sel].mean():.4f}  NLL(cf σ_s) {b["nll_variants"][sel, 0].mean():.4f}'
                  + (f'  AUC {b["auc"].mean():.4f} AvgL2 {b["avg_l2_fix"].mean():.4f} MinL2 {b["min_l2_fix"].mean():.4f}' if sn == 'all' else ''), flush=True)


if __name__ == '__main__':
    main()
