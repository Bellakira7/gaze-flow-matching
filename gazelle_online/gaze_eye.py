"""眼部 token（计划 §4）：从头框区域取高分辨率 crop，过同一个冻结 DINOv2，得到的 token 追加到场景 token 后面，作为 flow 解码器交叉注意力的额外 key/value。

设计选择（逐条写明，别默默扩散）：
  1. crop 只由**头框**决定（以头框中心为中心、边长 = margin × max(框宽, 框高) 的正方形），**不使用眼点标注**：
     测试时 gt 口径下眼点是标注，标准协议里不给；训练用的头框本身就是"眼点为中心的正方形"，所以训练时 crop 恰好以眼睛为中心，测试 gt 口径下是真实头框中心，
     两者分布略有不同（真实头框比眼点偏上/偏头心），这是已知的训练-测试差异，eye 口径下没有这个差异。
  2. crop 取自已经 resize 到 448×448 的（增强后的）图像，用 roi_align 双线性放大到 crop_size×crop_size 再喂 backbone。
     镜像训练图本身是缩放过的（短边多为 512），所以这里**没有额外的原始像素**，"高分辨率"指的是 token 密度：头部区域从场景里约 4×4 个 token 变成 crop 里 16×16 个 token。
  3. 与场景 token 共用同一个冻结 backbone（fp16 autocast，no_grad）；新增参数只有 Linear(1024→256) 与一个类型向量，约 0.26M。
  4. crop token 在 trunk 之后追加（不经过 trunk 的 3 层自注意力）：最小改动，让 flow 解码器的交叉注意力直接读它。
  5. 训练 / hold-out 评测：输入是真实图像。测试集（缓存场景特征）路径：scene token 用缓存特征（与 flow 基线的测试路径逐位同源），crop 从原图经同一 transform 在线过 backbone（fp16 autocast，与训练一致），见 predict_test_eye.py。
不改 gaze_models.py / train_heads.py（它们正被 seed 队列使用）。
"""
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torchvision

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_online')
import gaze_models as GM  # noqa: E402
import gaze_diffusion as GD  # noqa: E402,F401  (注册 'diff' 头)
import gazelle.utils as gutils  # noqa: E402
from gazelle.model import positionalencoding2d  # noqa: E402


class EyeTrunk(GM.Trunk):
    def __init__(self, backbone, crop_size=224, margin=1.25, **kw):
        super().__init__(backbone, **kw)
        assert crop_size % 14 == 0
        self.crop_size, self.margin = crop_size, margin
        g = crop_size // 14
        self.eye_proj = nn.Linear(backbone.get_dimension(), self.dim)
        self.eye_type = nn.Parameter(torch.zeros(self.dim))
        self.register_buffer('eye_pos', positionalencoding2d(self.dim, g, g).reshape(self.dim, g * g).t().contiguous())  # [g*g, dim]

    def crops(self, images, bboxes):
        assert images.dim() == 4 and images.shape[1] == 3, '眼部 token 需要真实图像输入（不支持缓存特征路径）'
        H, W = images.shape[-2:]
        rois = []
        for bi, bl in enumerate(bboxes):
            for b in bl:
                cx, cy = (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H
                s = max((b[2] - b[0]) * W, (b[3] - b[1]) * H) * self.margin
                rois.append([float(bi), cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2])
        rois = torch.tensor(rois, dtype=images.dtype, device=images.device)
        return torchvision.ops.roi_align(images, rois, (self.crop_size, self.crop_size), spatial_scale=1.0, sampling_ratio=2, aligned=True)

    def set_crop_backbone(self, bb):
        """测试路径用：场景 token 来自缓存特征（self.backbone 是恒等），crop 另走这个真 backbone。放进 list 里避免被注册成子模块（否则 state_dict 多出键）。"""
        self._crop_bb = [bb]

    def tokens(self, images, bboxes, raw_images=None):
        """训练 / hold-out：images 即真实图像，crop 与场景走同一个冻结 backbone。
        测试（缓存特征）：images 是缓存的场景特征，raw_images 是同一张图经同一 transform 后的 [B,3,448,448]，crop 走 set_crop_backbone 设的 backbone。"""
        scene = super().tokens(images, bboxes)  # [sum(N_p), 1024, dim]
        src = images if raw_images is None else raw_images
        cb = self._crop_bb[0] if getattr(self, '_crop_bb', None) else self.backbone
        f = cb.forward(self.crops(src, bboxes))  # [sum(N_p), C, g, g]，冻结 backbone
        e = self.eye_proj(f.flatten(2).transpose(1, 2)) + self.eye_pos + self.eye_type
        return torch.cat([scene, e], 1)


class EyeGazeModel(GM.GazeModel):
    """与 GazeModel 同接口（trunk.tokens / head.loss / head.sample / decoder_state），trunk 换成 EyeTrunk。"""

    def __init__(self, backbone, head, eye_kw=None, **head_kw):
        nn.Module.__init__(self)
        self.trunk = EyeTrunk(backbone, **(eye_kw or {}))
        self.head = GM.HEADS[head](**head_kw)
        self.head_name = head
