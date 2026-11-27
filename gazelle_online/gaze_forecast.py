"""VAT 上的未来注视预测（forecast）可行性实验（prereg 澄清 C19）。只用 VAT 训练集，按节目切 fit / val；**val 与测试集都不是 GazeFollow**，VAT 测试集不读。

任务：给定 t 帧与头框，预测 t+H 帧的注视：P(出画) + 画面内 2D flow 密度（p(y|画面内)）。
模型 = flowET_d6（EyeTrunk + 6 层 FlowHead）的扩展，只加两样东西：
  1. 头部 crop 可取多帧（ctx 偏移，如 (−15,−10,−5,0)），每帧一组 256 个 token，加可学习时间嵌入（零初始化）；场景 token 只取 t 帧。
     C=1 时 tokens 与 flowET_d6 相同（test_gaze_forecast.py 校验）。crop 在 dataloader 里从原图取（gaze_eye2.hires_crop）。
  2. 出画头：1 个 query 经 2 层交叉注意力读全部 token → logit，BCE。flow 损失只在 t+H 画面内的样本上算。
轨迹按人读 annotations/<split>/<show>/<clip>/sNN.txt（官方 preprocess_vat.py 把人的身份丢了，所以不用它的 json 取轨迹，只取宽高）；
头框沿用官方 11 帧滑动平均并夹到画面内。
增强沿用官方 GazeDataset 的三项（各 p=0.5）：random_crop（保留 t 帧头框与 t+H 目标）、horiz_flip、bbox 抖动；非 t 的上下文帧只跟随翻转。
"""
from pathlib import Path
import glob
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_online')
import gaze_models as GM  # noqa: E402
from gaze_eye import EyeTrunk  # noqa: E402
from gaze_eye2 import hires_crop  # noqa: E402
import gazelle.utils as gutils  # noqa: E402

VAT = f'{G}/data/vat'
N_VAL_SHOWS = 4


def _smooth(v, w=11):
    """与官方 preprocess_vat.smooth_by_conv 同式。"""
    pad = np.concatenate([v[0].repeat(w // 2), v, v[-1].repeat(w // 2)])
    return np.convolve(pad, np.ones(w) / w, mode='valid')


def val_shows():
    """节目名 sha1 排序前 N_VAL_SHOWS 个（C19）。"""
    shows = sorted(os.path.basename(p) for p in glob.glob(f'{VAT}/annotations/train/*'))
    return sorted(shows, key=lambda s: hashlib.sha1(s.encode()).hexdigest())[:N_VAL_SHOWS]


def c20_split():
    """C20：官方 train 的 40 个节目按 sha1 排序，前 4 = val（与 C19 相同），第 5–8 = cal（conformal 标定），其余 32 = fit。"""
    shows = sorted(os.path.basename(p) for p in glob.glob(f'{VAT}/annotations/train/*'))
    o = sorted(shows, key=lambda s: hashlib.sha1(s.encode()).hexdigest())
    return {'val': set(o[:4]), 'cal': set(o[4:8]), 'fit': set(o[8:])}


def load_tracks(mode, scheme='c19', allow_test=False):
    """scheme='c19'：mode = fit（36 个节目）| val（4 个）。scheme='c20'：mode = fit（32）| val（4）| cal（4）| test（官方测试集，须 allow_test=True）。"""
    if scheme == 'c19':
        assert mode in ('fit', 'val'), mode
        vs = set(val_shows())
        keep = (lambda show: show in vs) if mode == 'val' else (lambda show: show not in vs)
        split = 'train'
    else:
        assert scheme == 'c20' and mode in ('fit', 'val', 'cal', 'test'), (scheme, mode)
        if mode == 'test':
            assert allow_test, 'VAT 测试集只能由 C20 规定的读数脚本显式打开'
            split, keep = 'test', (lambda show: True)
        else:
            sp = c20_split()[mode]
            split, keep = 'train', (lambda show: show in sp)
    size = {s['path'].split('images/')[-1]: (s['width'], s['height']) for s in json.load(open(f'{VAT}/{split}_preprocessed.json'))}
    tracks = []
    for f in sorted(glob.glob(f'{VAT}/annotations/{split}/*/*/*.txt')):
        show, clip = f.split('/')[-3:-1]
        if not keep(show):
            continue
        W, H = size[f'{show}/{clip}']
        rows = [l.strip().split(',') for l in open(f) if l.strip()]
        fid = np.array([int(os.path.splitext(r[0])[0]) for r in rows])
        box = np.array([[float(v) for v in r[1:5]] for r in rows])
        box = np.stack([_smooth(box[:, i]) for i in range(4)], 1)
        box = np.clip(box, 0, [W, H, W, H])
        g = np.array([[float(r[5]), float(r[6])] for r in rows])
        inn = (g[:, 0] >= 0) & (g[:, 1] >= 0)
        g = np.where(inn[:, None], np.clip(g, 0, [W, H]), -1.0)
        tracks.append(dict(dir=f'{VAT}/images/{show}/{clip}', clip=f'{show}/{clip}', names=[r[0] for r in rows], fid=fid, box=box, gaze=g, inn=inn, W=W, H=H,
                           pos={int(v): i for i, v in enumerate(fid)}))
    return tracks


class VATForecast(torch.utils.data.Dataset):
    def __init__(self, mode, H, ctx, transform, sample_every=6, aug=None, margin=1.25, crop_size=224, scheme='c19', allow_test=False):
        self.tracks = load_tracks(mode, scheme, allow_test)
        self.H, self.ctx, self.tf, self.margin, self.crop_size = H, tuple(ctx), transform, margin, crop_size
        self.aug = (mode == 'fit') if aug is None else aug
        assert self.ctx[-1] == 0, 'ctx 最后一项必须是 0（当前帧）'
        self.items = []
        for t, tr in enumerate(self.tracks):
            for k in range(0, len(tr['fid']), sample_every):
                if int(tr['fid'][k]) + H in tr['pos']:
                    self.items.append((t, k))

    def __len__(self):
        return len(self.items)

    def _ctx_index(self, tr, k, o):
        """偏移 o（≤0）的上下文帧在轨迹里的下标；该帧没有标注时取不早于它的最近一帧（不超过当前帧）。"""
        return min(k, int(np.searchsorted(tr['fid'], tr['fid'][k] + o)))

    def __getitem__(self, i):
        t, k = self.items[i]
        tr = self.tracks[t]
        j = tr['pos'][int(tr['fid'][k]) + self.H]
        img = Image.open(os.path.join(tr['dir'], tr['names'][k])).convert('RGB')
        bbox = [float(v) for v in tr['box'][k]]
        in_f = bool(tr['inn'][j])
        gazex, gazey = [float(tr['gaze'][j, 0])], [float(tr['gaze'][j, 1])]
        flip = False
        if self.aug:
            if np.random.sample() <= 0.5:
                img, bbox, gazex, gazey = gutils.random_crop(img, bbox, gazex, gazey, in_f)
            if np.random.sample() <= 0.5:
                flip = True
                img, bbox, gazex, gazey = gutils.horiz_flip(img, bbox, gazex, gazey, in_f)
            if np.random.sample() <= 0.5:
                bbox = gutils.random_bbox_jitter(img, bbox)
        w, h = img.size
        bn = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
        y = [gazex[0] / w, gazey[0] / h] if in_f else [0.5, 0.5]
        crops = []
        for o in self.ctx:
            if o == 0:
                crops.append(hires_crop(img, bn, self.margin, self.crop_size))
                continue
            c = self._ctx_index(tr, k, o)
            im = Image.open(os.path.join(tr['dir'], tr['names'][c])).convert('RGB')
            b = tr['box'][c] / np.array([tr['W'], tr['H'], tr['W'], tr['H']])
            b = [float(v) for v in b]
            if flip:
                im = TF.hflip(im)
                b = [1 - b[2], b[1], 1 - b[0], b[3]]
            crops.append(hires_crop(im, b, self.margin, self.crop_size))
        return self.tf(img), bn, torch.tensor(y, dtype=torch.float32), torch.tensor(float(in_f)), torch.stack(crops), (t, k)


def collate(batch):
    imgs, bn, y, inn, crops, meta = zip(*batch)
    return torch.stack(imgs), list(bn), torch.stack(y), torch.stack(inn), torch.stack(crops), list(meta)


class ForecastTrunk(EyeTrunk):
    def __init__(self, backbone, C=1, **kw):
        super().__init__(backbone, **kw)
        self.time_emb = nn.Parameter(torch.zeros(C, self.dim))

    def tokens(self, images, bboxes, crops):
        """crops [B,C,3,S,S] -> [B, 1024 + C*256, D]（场景只取 t 帧）。"""
        scene = GM.Trunk.tokens(self, images, bboxes)
        B, C = crops.shape[:2]
        f = self.backbone.forward(crops.flatten(0, 1))
        e = self.eye_proj(f.flatten(2).transpose(1, 2)) + self.eye_pos + self.eye_type
        e = e.view(B, C, -1, self.dim) + self.time_emb[None, :, None]
        return torch.cat([scene, e.flatten(1, 2)], 1)


class ForecastModel(GM.GazeModel):
    def __init__(self, backbone, C=1, depth=6, out_depth=2, eye_kw=None):
        nn.Module.__init__(self)
        self.trunk = ForecastTrunk(backbone, C, **(eye_kw or {}))
        self.head = GM.FlowHead(depth=depth)
        self.head_name = 'forecast'
        dim = self.trunk.dim
        self.out_q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.out_dec = GM.XDecoder(dim, out_depth)
        self.out_fc = GM._mlp_out(dim, 1)

    def out_logit(self, tok):
        z = self.out_dec(self.out_q.expand(tok.shape[0], -1, -1), self.out_dec.make_kv(tok))[:, 0]
        return self.out_fc(z).float()[:, 0]

    def forward(self, images, bboxes, y, inn, crops):
        tok = self.trunk.tokens(images, bboxes, crops)
        bce = F.binary_cross_entropy_with_logits(self.out_logit(tok), 1 - inn)
        mask = inn > 0.5
        fm, logs = self.head.loss(tok[mask], y[mask])
        logs.update(bce=bce.item(), frac_in=mask.float().mean().item())
        return fm + bce, logs

    def load_init(self, path):
        """从 flowET_d6 这类 ckpt 初始化：只允许缺 backbone / time_emb / 出画头。"""
        missing, unexpected = self.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=False)
        assert not unexpected and all(k.startswith(('trunk.backbone.', 'trunk.time_emb', 'out_')) for k in missing), (missing[:5], unexpected[:5])


# ====================================================================== C20：各输出头的 forecast 版本（baseline 与主方法同一套出画头 / 训练配方）
class HeatmapHead(nn.Module):
    """B1 的热图头（官方 GazeLLE.heatmap_head：ConvTranspose 2× + 1×1 conv + sigmoid，32×32 token → 64×64），BCE 对官方高斯目标（σ=3）。
    抽样同 gaze_models.HeatmapGaze.sample（raw γ=1 的 pmf + λ 均匀混合，格内均匀抖动）；hyps 用热图 NMS 前 5 峰（同 g2_bundles 的 B1）。"""

    def __init__(self, dim=256):
        super().__init__()
        self.heatmap_head = nn.Sequential(nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2), nn.Conv2d(dim, 1, kernel_size=1, bias=False), nn.Sigmoid())

    def heat(self, tokens):
        B, _, D = tokens.shape
        x = tokens[:, :1024].reshape(B, 32, 32, D).permute(0, 3, 1, 2)  # 只取 32×32 场景 token
        return self.heatmap_head(x).squeeze(1)  # [B,64,64]，与 GazeLLE.forward 同（out_size 64 时 resize 不起作用）

    def loss(self, tokens, y):
        tgt = torch.stack([gutils.get_heatmap(float(a), float(b), 64, 64) for a, b in y.tolist()]).to(tokens.device)
        l = F.binary_cross_entropy(self.heat(tokens).float(), tgt)
        return l, {'bce_hm': l.item()}

    @torch.no_grad()
    def sample(self, tokens, n, gen=None, lam=1e-3, **_):
        h = self.heat(tokens).flatten(1).double().clamp_min(1e-12)
        P = (1 - lam) * h / h.sum(1, keepdim=True) + lam / h.shape[1]
        cells = torch.multinomial(P.float(), n, replacement=True, generator=gen)
        jit = torch.rand(cells.shape + (2,), device=cells.device, generator=gen)
        return torch.stack([(cells % GM.GRID + jit[..., 0]) / GM.GRID, (cells // GM.GRID + jit[..., 1]) / GM.GRID], -1)


class ForecastAny(GM.GazeModel):
    """head ∈ {flow, mdn, mcl, heatmap}；eye=True 时 trunk = ForecastTrunk（眼部 token），否则普通 Trunk（crops 忽略）。
    出画头与 ForecastModel 相同（1 个 query、2 层交叉注意力、BCE）。ForecastAny('flow', eye=True, depth=6) 与 ForecastModel 参数逐键相同。"""

    def __init__(self, backbone, head, eye=False, C=1, out_depth=2, eye_kw=None, **head_kw):
        nn.Module.__init__(self)
        self.eye = eye
        self.trunk = ForecastTrunk(backbone, C, **(eye_kw or {})) if eye else GM.Trunk(backbone)
        self.head = HeatmapHead(self.trunk.dim) if head == 'heatmap' else GM.HEADS[head](**head_kw)
        self.head_name = head
        dim = self.trunk.dim
        self.out_q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.out_dec = GM.XDecoder(dim, out_depth)
        self.out_fc = GM._mlp_out(dim, 1)

    out_logit = ForecastModel.out_logit

    def tokens(self, images, bboxes, crops):
        return self.trunk.tokens(images, bboxes, crops) if self.eye else self.trunk.tokens(images, bboxes)

    def forward(self, images, bboxes, y, inn, crops):
        tok = self.tokens(images, bboxes, crops)
        bce = F.binary_cross_entropy_with_logits(self.out_logit(tok), 1 - inn)
        mask = inn > 0.5
        hl, logs = self.head.loss(tok[mask], y[mask])
        logs.update(bce=bce.item(), frac_in=mask.float().mean().item())
        return hl + bce, logs

    @torch.no_grad()
    def sample(self, tok, n, gen=None, **flow_kw):
        return self.head.sample(tok, n, gen=gen, **(flow_kw if self.head_name == 'flow' else {}))

    def load_init(self, path):
        """GazeFollow 上训好的 s0 ckpt：GazeModel / EyeGazeModel 的键直接对上；B1（train_online.py 的 GazeLLE）键无前缀，
        heatmap_head.* → head.heatmap_head.*，其余 → trunk.*。只允许缺 backbone / time_emb / 出画头。"""
        sd = torch.load(path, map_location='cpu', weights_only=True)
        if self.head_name == 'heatmap' and not any(k.startswith('trunk.') for k in sd):
            sd = {('head.' + k if k.startswith('heatmap_head.') else 'trunk.' + k): v for k, v in sd.items()}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith(('trunk.backbone.', 'trunk.time_emb', 'out_')) for k in missing), (missing[:5], unexpected[:5])
