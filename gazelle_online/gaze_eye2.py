"""眼部 token 的两个变体（prereg 澄清 C17）。都在 flowET（gaze_eye.EyeTrunk + 3 层 FlowHead）上只改一处，**不加参数**，与 flowET 同容量比较。

1. fuse（FuseEyeTrunk）：眼部 token 在 trunk 的 3 层自注意力**之前**拼到场景 token 后面，一起过自注意力；
   flowET 里眼部 token 只经过 Linear(1024→256)，在自注意力**之后**才拼上（gaze_eye.py 设计第 4 条），场景 token 看不到人脸。
   依据：眼部 token 唯一有显著证据的作用是降低方向误差（两个 seed 都约 −7.5%），而这条通路只有一层线性投影；
   Gaze-LLE 的头部提示也是在 trunk 自注意力之前注入的。输出布局不变：[场景 1024, 眼部 256]。
2. hires（HiResEyeTrunk + HiResCropDataset）：crop 区域与 flowET 完全相同（头框中心、边长 = margin × max(框宽, 框高)，归一化坐标），
   但 roi_align 的取样源从"已缩放到 448×448 的图"换成"增强后的原图"（同样 ToTensor + Normalize，不 Resize），隔离"分辨率"这一个变量。
   增益有限（10-30 量过：原图对 448 图的线性像素比，训练镜像图横 1.52× / 纵 1.14×（中位），镜像本身又是原图放大 1.07–1.42× 的；
   测试原图横 1.43× / 纵 0.95×），所以作为第二臂。crop 在 dataloader 的 worker 里算（CPU），作为额外张量传给 trunk。
不改 gaze_models.py / gaze_eye.py（已出数字的臂依赖它们）。
"""
from pathlib import Path
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_online')
import gaze_models as GM  # noqa: E402
from gaze_eye import EyeTrunk  # noqa: E402
import gazelle.utils as gutils  # noqa: E402
from train_online import SDGazeDataset  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # 与 DinoV2Backbone.get_transform 相同


def eye_tokens(trunk, src, bboxes, crops=None):
    """与 EyeTrunk.tokens 里的眼部部分逐项相同；crops（[sum N_p,3,S,S]，已归一化）给定时不再从 src 里 roi_align。"""
    cb = trunk._crop_bb[0] if getattr(trunk, '_crop_bb', None) else trunk.backbone
    f = cb.forward(trunk.crops(src, bboxes) if crops is None else crops)
    return trunk.eye_proj(f.flatten(2).transpose(1, 2)) + trunk.eye_pos + trunk.eye_type


class FuseEyeTrunk(EyeTrunk):
    def tokens(self, images, bboxes, raw_images=None, crops=None):
        num_ppl = [len(b) for b in bboxes]
        x = self.backbone.forward(images)
        x = self.linear(x) + self.pos_embed
        x = gutils.repeat_tensors(x, num_ppl)
        head_maps = torch.cat(self.get_input_head_maps(bboxes), dim=0).to(x.device)
        x = x + head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x.flatten(2).permute(0, 2, 1)
        e = eye_tokens(self, images if raw_images is None else raw_images, bboxes, crops)
        return self.transformer(torch.cat([x, e], 1))


class HiResEyeTrunk(EyeTrunk):
    def tokens(self, images, bboxes, raw_images=None, crops=None):
        assert crops is not None, 'hires 变体需要 dataloader 给出原图 crop'
        return torch.cat([GM.Trunk.tokens(self, images, bboxes), eye_tokens(self, None, bboxes, crops)], 1)


TRUNKS = {'fuse': FuseEyeTrunk, 'hires': HiResEyeTrunk}


class EyeGazeModel2(GM.GazeModel):
    def __init__(self, backbone, variant, eye_kw=None, **head_kw):
        nn.Module.__init__(self)
        self.variant = variant
        self.trunk = TRUNKS[variant](backbone, **(eye_kw or {}))
        self.head = GM.FlowHead(**head_kw)
        self.head_name = 'flow'

    def forward(self, images, bboxes, gaze, crops=None):
        return self.head.loss(self.trunk.tokens(images, bboxes, crops=crops), gaze)

    @torch.no_grad()
    def sample_tokens(self, tok, bboxes, n, return_pout=False, **kw):
        S = self.head.sample(tok, n, **kw)
        return (S, torch.full((S.shape[0],), float('nan'), device=S.device)) if return_pout else S


# ------------------------------------------------------------------ hires：原图 crop
def hires_crop(img, bbox_norm, margin=1.25, size=224):
    """img：PIL（增强后的原图）；bbox_norm：归一化 xyxy。返回 [3,size,size]，与 EyeTrunk.crops 同一区域、同一 roi_align 设置。"""
    t = TF.normalize(TF.to_tensor(img), MEAN, STD)
    H, W = t.shape[1:]
    b = [float(v) for v in bbox_norm]
    cx, cy = (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H
    s = max(b[2] - b[0], b[3] - b[1]) * margin
    roi = torch.tensor([[0.0, cx - s * W / 2, cy - s * H / 2, cx + s * W / 2, cy + s * H / 2]])
    return torchvision.ops.roi_align(t[None], roi, (size, size), spatial_scale=1.0, sampling_ratio=2, aligned=True)[0]


class HiResCropDataset(SDGazeDataset):
    """SDGazeDataset，__getitem__ 与官方 GazeDataset.__getitem__ 逐行相同（同一套增强与随机数调用顺序），
    只在 transform 之前多算一个原图 crop，追加为最后一个返回值。"""

    def __init__(self, path, mode, transform, margin=1.25, crop_size=224):
        super().__init__(path, mode, transform)
        self.margin, self.crop_size = margin, crop_size

    def __getitem__(self, idx):
        img_idx, head_idx = self.data_idxs[idx]
        img_data = self.data[img_idx]
        head_data = copy.deepcopy(img_data['heads'][head_idx])
        bbox_norm = head_data['bbox_norm']
        gazex_norm = head_data['gazex_norm']
        gazey_norm = head_data['gazey_norm']
        inout = head_data['inout']

        img = Image.open(os.path.join(self.path, img_data['path'])).convert('RGB')
        width, height = img.size

        if self.aug:
            bbox = head_data['bbox']
            gazex = head_data['gazex']
            gazey = head_data['gazey']
            if np.random.sample() <= 0.5:
                img, bbox, gazex, gazey = gutils.random_crop(img, bbox, gazex, gazey, inout)
            if np.random.sample() <= 0.5:
                img, bbox, gazex, gazey = gutils.horiz_flip(img, bbox, gazex, gazey, inout)
            if np.random.sample() <= 0.5:
                bbox = gutils.random_bbox_jitter(img, bbox)
            width, height = img.size
            bbox_norm = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            gazex_norm = [x / float(width) for x in gazex]
            gazey_norm = [y / float(height) for y in gazey]

        crop = hires_crop(img, bbox_norm, self.margin, self.crop_size)
        img = self.transform(img)
        if self.split == 'train':
            heatmap = gutils.get_heatmap(gazex_norm[0], gazey_norm[0], 64, 64)
            return img, bbox_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width, heatmap, crop
        return img, bbox_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width, crop
