import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LocalDinoV2Backbone, build_cached_gazelle, GazeLLE  # noqa: E402

NAME = 'dinov2_vitl14'
dev = 'cuda'
torch.manual_seed(0)
rng = np.random.RandomState(0)

backbone = LocalDinoV2Backbone(NAME)
transform = backbone.get_transform((448, 448))
official = GazeLLE(backbone).to(dev).eval()
cached = build_cached_gazelle(NAME).to(dev).eval()
cached.load_gazelle_state_dict(official.get_gazelle_state_dict())

imgs = []
for k in range(4):
    yy, xx = np.mgrid[0:480, 0:640]
    arr = np.stack([(xx * (k + 1) % 256), (yy * (k + 2) % 256), rng.randint(0, 256, (480, 640))], -1).astype(np.uint8)
    imgs.append(Image.fromarray(arr))
x = torch.stack([transform(im) for im in imgs]).to(dev)
bboxes = [[(0.1, 0.1, 0.3, 0.4)], [(0.5, 0.2, 0.7, 0.5), (0.1, 0.3, 0.2, 0.5)], [None], [(0.0, 0.0, 0.5, 0.5)]]

with torch.no_grad():
    ref = official({'images': x, 'bboxes': bboxes})['heatmap']
    feats = official.backbone(x)
    out32 = cached({'images': feats, 'bboxes': bboxes})['heatmap']
    out16 = cached({'images': feats.half().float(), 'bboxes': bboxes})['heatmap']

def cmp(a, b):
    d = max((u - v).abs().max().item() for u, v in zip(a, b))
    am = np.mean([(pu.flatten().argmax() == pv.flatten().argmax()).item()
                  for u, v in zip(a, b) for pu, pv in zip(u, v)])
    return d, am

print('feat', tuple(feats.shape), feats.dtype, 'abs max', feats.abs().max().item(), '(fp16 max 65504)')
print('heatmap range', min(h.min().item() for h in ref), max(h.max().item() for h in ref))
d32, a32 = cmp(ref, out32)
d16, a16 = cmp(ref, out16)
print(f'fp32 cache : max|diff| = {d32:.3e}  argmax agree = {a32:.3f}')
print(f'fp16 cache : max|diff| = {d16:.3e}  argmax agree = {a16:.3f}')
assert d32 < 1e-5, 'cached forward 与官方整网 forward 不等价'
print('PASS')
