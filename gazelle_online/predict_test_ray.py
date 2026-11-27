"""射线分解头的测试集预测（与 predict_test_eye.py 逐段对应，差别只有：模型 = gaze_ray.RayGazeModel，采样走 model.sample_tokens（要头框中心））。
python -I -u predict_test_ray.py --variant ray --ckpt <ray_eye_s0/epoch_14.pt> --box gt|eye --out results/g2/pred_rayET_gt.npz --bundle results/g2/bundle_rayET_gt.npz
        [--no_eye] [--ray_kw JSON] [--sample_kw JSON] [--sigma_h 1.0] [--n 10000] [--limit N]
  --variant eye / base：flowET / flow，与 predict_test_eye.py 相同，用来对拍（--limit + --parity_ref），证明本脚本没有引入别的差异。
  --parity_ref    ：与 --limit 连用，把本次前若干个实例的 hyps/samples 与存档对比并打印最大差异（不保存）。
以下为 predict_test_eye.py 原说明：
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
from gaze_ray import RayGazeModel, RayTokGazeModel  # noqa: E402
from gaze_eye2 import EyeGazeModel2, hires_crop  # noqa: E402
import gaze_mixflow  # noqa: E402,F401  (注册 'mixflow' 头，C21)
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


class TestFramesHR(TestFrames):
    """eyehr（C17）：另外从**原图**（不 Resize）按与训练相同的 hires_crop 取每个头的 crop，返回 [n_heads,3,S,S]。"""

    def __init__(self, frames, feats, transform, n, box, frac, margin, crop_size):
        super().__init__(frames, feats, transform, n)
        self.box, self.frac, self.margin, self.crop_size = box, frac, margin, crop_size

    def __getitem__(self, i):
        img = Image.open(f'{G}/data/gazefollow_test_hf/{self.frames[i]["path"]}').convert('RGB')
        crops = torch.stack([hires_crop(img, head_box(self.frames[i], h, self.box, self.frac), self.margin, self.crop_size)
                             for h in self.frames[i]['heads']])
        return i, torch.from_numpy(np.array(self.feats[i])), self.tf(img), crops


def collate_hr(batch):
    ids, fe, im, cr = zip(*batch)
    return torch.tensor(ids), torch.stack(fe), torch.stack(im), torch.cat(cr)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['base', 'eye', 'ray', 'raytok', 'eyefuse', 'eyehr', 'mixflow'])
    ap.add_argument('--no_eye', action='store_true', help='ray 的消融：不要眼部 token')
    ap.add_argument('--ray_kw', default='{}')
    ap.add_argument('--head_kw', default='{}', help='base / eye / raytok 时传给 FlowHead（如 {"depth": 6}）')
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
    if a.variant == 'ray':
        model = RayGazeModel(cb, eye=not a.no_eye, eye_kw=json.loads(a.eye_kw), **json.loads(a.ray_kw))
    elif a.variant == 'raytok':  # v2（C16）：射线 token 在 trunk 里（测试路径同样经 trunk.tokens(..., raw_images)），采样走 head.sample
        model = RayTokGazeModel(cb, eye_kw=json.loads(a.eye_kw), ray_kw=json.loads(a.ray_kw), **json.loads(a.head_kw))
    elif a.variant in ('eyefuse', 'eyehr'):  # C17：眼部 token 变体 2（gaze_eye2）
        model = EyeGazeModel2(cb, {'eyefuse': 'fuse', 'eyehr': 'hires'}[a.variant], eye_kw=json.loads(a.eye_kw), **json.loads(a.head_kw))
    elif a.variant == 'mixflow':  # C21：集合原生混合 flow 头（假设 = predict() 原生输出，不走样本聚类）
        model = EyeGazeModel(cb, 'mixflow', eye_kw=json.loads(a.eye_kw), **json.loads(a.head_kw))
    else:
        hk = json.loads(a.head_kw)
        model = EyeGazeModel(cb, 'flow', eye_kw=json.loads(a.eye_kw), **hk) if a.variant == 'eye' else GazeModel(cb, 'flow', **hk)
    model = model.to(dev).eval()
    use_eye = a.variant in ('eye', 'raytok', 'eyefuse', 'eyehr', 'mixflow') or (a.variant == 'ray' and not a.no_eye)
    hires = a.variant == 'eyehr'
    model.load_decoder_state(torch.load(a.ckpt, map_location='cpu', weights_only=True))
    tf = None
    if use_eye:
        real = AmpBackbone('dinov2_vitl14').to(dev).eval()
        model.trunk.set_crop_backbone(real)
        tf = real.get_transform(IN_SIZE)
    frames = json.load(open(f'{G}/data/gazefollow_test_hf/test_preprocessed.json'))
    feats = np.load(f'{G}/feats/gf_test_vitl14/test_orig.npy', mmap_mode='r')
    nf = a.limit or len(frames)
    skw = json.loads(a.sample_kw)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    if hires:
        ek = json.loads(a.eye_kw)
        dl = torch.utils.data.DataLoader(TestFramesHR(frames, feats, tf, nf, a.box, a.frac, ek.get('margin', 1.25), ek.get('crop_size', 224)),
                                         batch_size=a.bs, shuffle=False, num_workers=8, collate_fn=collate_hr)
    elif use_eye:
        dl = torch.utils.data.DataLoader(TestFrames(frames, feats, tf, nf), batch_size=a.bs, shuffle=False, num_workers=8)
    else:
        dl = torch.utils.data.DataLoader(TestFrames(frames, feats, lambda im: torch.zeros(1), nf), batch_size=a.bs, shuffle=False, num_workers=4)
    hyps_l, w_l, S_l = [], [], []
    t0 = time.time()
    for bi, bt in enumerate(dl):
        ids, fe, im = bt[:3]
        idx = ids.tolist()
        x = fe.to(dev).float()
        boxes = [[head_box(frames[i], h, a.box, a.frac) for h in frames[i]['heads']] for i in idx]
        if hires:
            tok = model.trunk.tokens(x, boxes, crops=bt[3].to(dev))
        else:
            tok = model.trunk.tokens(x, boxes, raw_images=im.to(dev)) if use_eye else model.trunk.tokens(x, boxes)
        if a.variant == 'ray':
            S = model.sample_tokens(tok, boxes, a.n, gen=gen, **skw).float().cpu().numpy()
        else:
            S = model.head.sample(tok, a.n, gen=gen, **skw).float().cpu().numpy()
        if a.variant == 'mixflow':  # 原生假设（merge=0.10 合并后，按权重降序），不聚类
            hy_n, w_n = model.head.predict(tok)
            hy_n, w_n = hy_n.float().cpu().numpy().astype(np.float64), w_n.float().cpu().numpy().astype(np.float64)
        for j in range(S.shape[0]):
            if a.variant == 'mixflow':
                hh, ww = hy_n[j], w_n[j]
            else:
                hh, ww = GE.samples_to_hyps(S[j].astype(np.float64), a.sigma_h)
            hyps_l.append(hh)
            w_l.append(ww)
            S_l.append(S[j])
        if bi % 40 == 0:
            print(f'  frames {idx[-1] + 1}/{nf}  instances {len(S_l)}  {time.time() - t0:.0f}s', flush=True)
    pred = dict(hyps=np.stack(hyps_l), w=np.stack(w_l), samples=np.stack(S_l).astype(np.float32))
    hyp_mode = 'native(merge=0.10)' if a.variant == 'mixflow' else f'cluster(σ_h={a.sigma_h})'
    print(f'预测完成 {len(S_l)} 实例，{time.time() - t0:.0f}s；hyps[0] 点估计均值 {pred["hyps"][:, 0].mean(0)}；假设模式 {hyp_mode}', flush=True)
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
            print(f'[{a.variant}{"" if a.variant == "base" else "/no_eye" if a.no_eye else "/eye"}/{sn}] cov0.10@5 {b["cov0.1@5"][sel].mean():.4f}  NLL(cf σ_s) {b["nll_variants"][sel, 0].mean():.4f}'
                  + (f'  AUC {b["auc"].mean():.4f} AvgL2 {b["avg_l2_fix"].mean():.4f} MinL2 {b["min_l2_fix"].mean():.4f}' if sn == 'all' else ''), flush=True)


if __name__ == '__main__':
    main()
