"""C20 新增部分的自检（CPU，假 backbone）：python -I test_gaze_forecast_c20.py
1. c20 切分：val 4 / cal 4 / fit 32 个节目，两两不相交，并集 = 官方 train 的 40 个；val 与 C19 相同；不给 allow_test 时拒绝 test。
2. ForecastAny('flow', eye=True, depth=6) 与 ForecastModel 参数逐键、逐形状相同。
3. 五个 GazeFollow s0 ckpt 都能 load_init，且关键权重确实载入。
4. HeatmapHead 与官方 GazeLLE.forward 的热图逐位相同（同权重、同输入）。
5. 四种头 loss 可反传、sample 输出形状 [B,n,2] 且在 [0,1]。
"""
from pathlib import Path
import sys

import numpy as np
import torch

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_online')
import gaze_forecast as GF  # noqa: E402
from gazelle.model import GazeLLE  # noqa: E402
from gazelle.backbone import Backbone  # noqa: E402
import torch.nn as nn  # noqa: E402


class FakeBB(Backbone):
    """确定性的假 backbone：14×14 平均池化 + 固定线性 3→1024（同 test_gaze_eye2.py）。"""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1024)

    def forward(self, x):
        return self.lin(nn.functional.avg_pool2d(x, 14).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def get_dimension(self):
        return 1024

    def get_out_size(self, in_size):
        return in_size[0] // 14, in_size[1] // 14

torch.manual_seed(0)
fails = []


def check(name, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not ok:
        fails.append(name)


print('\n==== C20 自检 ====')
sp = GF.c20_split()
allv = sp['val'] | sp['cal'] | sp['fit']
check('1 切分 4/4/32、不相交、并集 40', (len(sp['val']), len(sp['cal']), len(sp['fit'])) == (4, 4, 32) and len(allv) == 40
      and not (sp['val'] & sp['cal']) and not (sp['val'] & sp['fit']) and not (sp['cal'] & sp['fit']), f'cal={sorted(sp["cal"])}')
check('1 val 与 C19 相同', sp['val'] == set(GF.val_shows()))
try:
    GF.load_tracks('test', 'c20')
    check('1 不给 allow_test 拒绝 test', False)
except AssertionError:
    check('1 不给 allow_test 拒绝 test', True)
n = {m: len(GF.load_tracks(m, 'c20')) for m in ('fit', 'val', 'cal')}
check('1 c20 三个子集都有轨迹', all(v > 0 for v in n.values()), str(n))

a = GF.ForecastAny(FakeBB(), 'flow', eye=True, depth=6).state_dict()
b = GF.ForecastModel(FakeBB(), C=1, depth=6).state_dict()
check('2 ForecastAny(flow,eye,6) == ForecastModel', list(a) == list(b) and all(a[k].shape == b[k].shape for k in a))

E = f'{G}/experiments'
CK = {('heatmap', False, 3): f'{E}/b1_dinov2l_sd_s0/epoch_14.pt', ('mdn', False, 3): f'{E}/mdn_s0/epoch_14.pt', ('mcl', False, 3): f'{E}/mcl_s0/epoch_14.pt',
      ('flow', False, 3): f'{E}/flow_s0/epoch_14.pt', ('flow', True, 6): f'{E}/flow_eye_d6_s0/epoch_14.pt'}
models = {}
for (h, eye, dp), p in CK.items():
    m = GF.ForecastAny(FakeBB(), h, eye=eye, **({'depth': dp} if h == 'flow' else {}))
    try:
        m.load_init(p)
        sd = torch.load(p, map_location='cpu', weights_only=True)
        key = 'transformer.0.attn.qkv.weight' if h == 'heatmap' else 'trunk.transformer.0.attn.qkv.weight'
        ok = torch.equal(m.trunk.transformer[0].attn.qkv.weight, sd[key])
        if h == 'heatmap':
            ok &= torch.equal(m.head.heatmap_head[1].weight, sd['heatmap_head.1.weight'])
        check(f'3 load_init {h}{"+eye" if eye else ""}', ok, p.split('/')[-2])
    except AssertionError as e:
        check(f'3 load_init {h}{"+eye" if eye else ""}', False, str(e)[:200])
    models[(h, eye)] = m.eval()

gl = GazeLLE(FakeBB()).eval()
sd = torch.load(CK[('heatmap', False, 3)], map_location='cpu', weights_only=True)
miss, unexp = gl.load_state_dict(sd, strict=False)
fm = models[('heatmap', False)]
fm.trunk.backbone.lin.load_state_dict(gl.backbone.lin.state_dict())
imgs = torch.randn(2, 3, 448, 448)
boxes = [[[0.4, 0.3, 0.5, 0.45]], [[0.1, 0.1, 0.2, 0.25]]]
with torch.no_grad():
    ref = torch.cat(gl({'images': imgs, 'bboxes': boxes})['heatmap'], 0)
    got = fm.head.heat(fm.tokens(imgs, boxes, None))
check('4 HeatmapHead == 官方 GazeLLE 热图', (ref - got).abs().max().item() < 1e-6, f'max diff {(ref - got).abs().max().item():.1e}；GazeLLE 缺键只有 backbone：{all(k.startswith("backbone") for k in miss)}')

y = torch.tensor([[0.3, 0.6], [0.7, 0.2]])
inn = torch.tensor([1.0, 0.0])
crops = torch.randn(2, 1, 3, 224, 224)
for (h, eye), m in models.items():
    m.train()
    loss, logs = m(imgs, boxes, torch.tensor([[0.3, 0.6], [0.5, 0.5]]), torch.tensor([1.0, 1.0]), crops)
    loss.backward()
    m.eval()
    with torch.no_grad():
        S = m.sample(m.tokens(imgs, boxes, crops), 64, steps=4) if h == 'flow' else m.sample(m.tokens(imgs, boxes, crops), 64)
    check(f'5 {h}{"+eye" if eye else ""} loss/sample', torch.isfinite(loss).item() and S.shape == (2, 64, 2) and bool((S >= 0).all() & (S <= 1).all()), f'loss {loss.item():.3f} {list(logs)}')

print('\n全部通过' if not fails else f'\n失败 {len(fails)} 项：{fails}')
sys.exit(1 if fails else 0)
