"""gaze_eye2 自检（CPU，不需要真实 backbone）：python -I test_gaze_eye2.py
1. hires_crop 与 EyeTrunk.crops 同一区域：
   a. 原图恰为 448×448 时（Resize 不起作用）两条路径逐位 / 近乎逐位相同；
   b. 1000×600 的线性坐标渐变图（R 编码 x、G 编码 y）：两条路径取到的 crop 在内部像素上一致（线性函数在双线性插值下精确）；
   c. 区域越出图像时，两条路径的越界部分同样为 0（归一化空间）。
2. FuseEyeTrunk：参数与 EyeTrunk 逐键相同；transformer 换成恒等时输出与 EyeTrunk 相同；
   正常 transformer 下，改眼部 crop 会改变场景 token 输出（EyeTrunk 不会）。
3. HiResEyeTrunk：给定 crops 时，眼部 token = 对这些 crops 过 backbone 的结果；场景部分与 EyeTrunk 相同。
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gazelle_online'))
import gaze_eye2 as E2  # noqa: E402
from gaze_eye import EyeTrunk  # noqa: E402
from gazelle.backbone import Backbone  # noqa: E402
import torchvision.transforms as T  # noqa: E402

torch.manual_seed(0)
fails = []


def check(name, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not ok:
        fails.append(name)


TF448 = T.Compose([T.ToTensor(), T.Normalize(E2.MEAN, E2.STD), T.Resize((448, 448))])  # 与 DinoV2Backbone.get_transform 同构


class FakeBB(Backbone):
    """确定性的假 backbone：14×14 平均池化 + 固定线性 3→1024。"""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1024)

    def forward(self, x):
        return self.lin(nn.functional.avg_pool2d(x, 14).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def get_dimension(self):
        return 1024

    def get_out_size(self, in_size):
        return in_size[0] // 14, in_size[1] // 14


# 1a
rng = np.random.default_rng(0)
img = Image.fromarray(rng.integers(0, 255, (448, 448, 3), dtype=np.uint8))
box = [0.40, 0.30, 0.52, 0.46]
tr = EyeTrunk(FakeBB())
c_base = tr.crops(TF448(img)[None], [[box]])[0]
c_hr = E2.hires_crop(img, box)
check('1a 448×448 原图：hires 与 448 路径一致', (c_base - c_hr).abs().max().item() < 1e-5, f'max diff {(c_base - c_hr).abs().max().item():.1e}')

# 1b
W, H = 1000, 600
xx, yy = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
grad = np.stack([xx / W * 255, yy / H * 255, np.full_like(xx, 128)], -1).round().astype(np.uint8)
img2 = Image.fromarray(grad)
box2 = [0.31, 0.22, 0.39, 0.37]
c_base = tr.crops(TF448(img2)[None], [[box2]])[0]
c_hr = E2.hires_crop(img2, box2)
inner = (slice(None), slice(8, -8), slice(8, -8))
d = (c_base[inner] - c_hr[inner]).abs()
scale = c_hr[:2].max() - c_hr[:2].min()
check('1b 渐变图：两条路径取到同一区域（内部像素差 < 量化误差）', d.max().item() < 0.03, f'max diff {d.max().item():.4f}（归一化单位；R/G 跨度 {scale:.2f}）')

# 1c
box3 = [0.0, 0.0, 0.06, 0.08]
c_base = tr.crops(TF448(img2)[None], [[box3]])[0]
c_hr = E2.hires_crop(img2, box3)
zb, zh = (c_base.abs().sum(0) == 0), (c_hr.abs().sum(0) == 0)
check('1c 越界区域两条路径都为 0', zb.float().mean().item() > 0.1 and (zb ^ zh).float().mean().item() < 0.02,
      f'零像素占比 448 路径 {zb.float().mean():.3f} / hires {zh.float().mean():.3f}，不一致 {(zb ^ zh).float().mean():.4f}')

# 2
torch.manual_seed(1)
base = EyeTrunk(FakeBB()).eval()
fuse = E2.FuseEyeTrunk(FakeBB()).eval()
fuse.load_state_dict(base.state_dict())
check('2 Fuse 与 EyeTrunk 参数逐键相同', list(base.state_dict()) == list(fuse.state_dict()))
imgs = torch.randn(2, 3, 448, 448)
boxes = [[[0.4, 0.3, 0.5, 0.45]], [[0.1, 0.1, 0.2, 0.25], [0.6, 0.5, 0.7, 0.6]]]
with torch.no_grad():
    tb, tf_ = base.transformer, fuse.transformer
    base.transformer, fuse.transformer = nn.Identity(), nn.Identity()
    same = (base.tokens(imgs, boxes) - fuse.tokens(imgs, boxes)).abs().max().item()
    base.transformer, fuse.transformer = tb, tf_
    check('2 transformer 恒等时 Fuse == EyeTrunk', same < 1e-5, f'max diff {same:.1e}')
    imgs2 = imgs.clone()
    imgs2[0, :, 120:220, 170:240] += 1.0  # 只动实例 0 的头部区域，且只喂给 crop（raw_images），场景输入不变
    cb = (base.tokens(imgs, boxes, raw_images=imgs) - base.tokens(imgs, boxes, raw_images=imgs2))[0, :1024]
    cf = (fuse.tokens(imgs, boxes, raw_images=imgs) - fuse.tokens(imgs, boxes, raw_images=imgs2))[0, :1024]
    check('2 只改人脸 crop：EyeTrunk 场景 token 不变、Fuse 会变', cb.abs().max().item() < 1e-6 and cf.abs().max().item() > 1e-3,
          f'EyeTrunk {cb.abs().max().item():.1e} / Fuse {cf.abs().max().item():.1e}')

# 3
hr = E2.HiResEyeTrunk(FakeBB()).eval()
hr.load_state_dict(base.state_dict())
crops = torch.randn(3, 3, 224, 224)
with torch.no_grad():
    out = hr.tokens(imgs, boxes, crops=crops)
    ref_scene = base.tokens(imgs, boxes)[:, :1024]
    ref_eye = E2.eye_tokens(base, None, boxes, crops)
check('3 HiRes：场景部分与 EyeTrunk 相同', (out[:, :1024] - ref_scene).abs().max().item() < 1e-5)
check('3 HiRes：眼部部分 = 给定 crops 的 token', (out[:, 1024:] - ref_eye).abs().max().item() < 1e-5)

print('\n全部通过' if not fails else f'\n失败 {len(fails)} 项：{fails}')
sys.exit(1 if fails else 0)
