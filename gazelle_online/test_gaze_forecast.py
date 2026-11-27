"""gaze_forecast 自检（CPU，用假 backbone，不训练）：python -I test_gaze_forecast.py
1. 切分：val 恰 4 个节目，与 fit 节目不相交；两者都来自官方 train；load_tracks 拒绝 test。
2. 目标对齐：取若干样本，y 与轨迹 t+H 帧的标注一致（无增强时逐位）；帧号差恰为 H；画面外样本 inn=0。
3. 上下文：偏移帧号正确；缺帧时不越过当前帧；ctx 单帧时 crops 形状 [1,3,224,224]。
4. 翻转：同一样本翻转后 crop 与原 crop 左右镜像（非当前帧）。
5. C=1 且 time_emb=0 时，ForecastTrunk.tokens 与 EyeTrunk（同权重、同 crop）一致；C=4 时 token 数 = 1024+4×256。
6. load_init 能加载真实的 flowET_d6 ckpt，只缺 backbone / time_emb / 出画头。
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_online')
import gaze_forecast as GF  # noqa: E402
from gaze_eye import EyeTrunk  # noqa: E402
from gaze_eye2 import MEAN, STD, eye_tokens  # noqa: E402
import gaze_models as GM  # noqa: E402
from gazelle.backbone import Backbone  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
fails = []


def check(name, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not ok:
        fails.append(name)


TF448 = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD), T.Resize((448, 448))])


class FakeBB(Backbone):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1024)

    def forward(self, x):
        return self.lin(nn.functional.avg_pool2d(x, 14).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def get_dimension(self):
        return 1024

    def get_out_size(self, in_size):
        return in_size[0] // 14, in_size[1] // 14


# 1
vs = GF.val_shows()
fit, val = GF.load_tracks('fit'), GF.load_tracks('val')
fs, vss = {t['clip'].split('/')[0] for t in fit}, {t['clip'].split('/')[0] for t in val}
check('1 val 4 个节目、与 fit 不相交', len(vss) == 4 and not (fs & vss) and vss == set(vs), f'val={sorted(vss)}；fit {len(fs)} 个节目；轨迹 {len(fit)} / {len(val)}；帧 {sum(len(t["fid"]) for t in fit)} / {sum(len(t["fid"]) for t in val)}')
try:
    GF.load_tracks('test')
    check('1 拒绝 test', False)
except AssertionError:
    check('1 拒绝 test', True)

# 2
ds = GF.VATForecast('val', 30, (0,), TF448)
ok, n_out = True, 0
rng = np.random.default_rng(0)
for i in rng.integers(0, len(ds), 40):
    t, k = ds.items[i]
    tr = ds.tracks[t]
    _, bn, y, inn, crops, meta = ds[int(i)]
    j = tr['pos'][int(tr['fid'][k]) + 30]
    ok &= (tr['fid'][j] - tr['fid'][k] == 30) and meta == (t, k) and crops.shape == (1, 3, 224, 224)
    ok &= bool(inn) == bool(tr['inn'][j])
    if inn:
        ok &= abs(y[0].item() - tr['gaze'][j, 0] / tr['W']) < 1e-6 and abs(y[1].item() - tr['gaze'][j, 1] / tr['H']) < 1e-6
    else:
        n_out += 1
    ok &= abs(bn[0] - tr['box'][k, 0] / tr['W']) < 1e-9
check('2 目标与轨迹 t+H 帧一致', bool(ok), f'40 个样本里画面外 {n_out}；数据集 {len(ds)} 实例')

# 3
ds4 = GF.VATForecast('val', 30, (-15, -10, -5, 0), TF448)
tr = next(t for t in ds4.tracks if len(t['fid']) > 60)
k = 40
check('3 上下文下标', all(tr['fid'][ds4._ctx_index(tr, k, o)] == tr['fid'][k] + o for o in (-15, -10, -5)) if tr['fid'][k] - tr['fid'][k - 15] == 15 else True)
check('3 缺帧不越过当前帧', ds4._ctx_index(tr, 2, -15) == 0 and ds4._ctx_index(tr, 0, -5) == 0)

# 4
s = ds4[len(ds4) // 2]
t, k = s[5]
tr = ds4.tracks[t]
c_noflip = s[4][0]  # o=-15 的 crop
c = ds4._ctx_index(tr, k, -15)
from PIL import Image  # noqa: E402
import torchvision.transforms.functional as TFF  # noqa: E402
from gaze_eye2 import hires_crop  # noqa: E402
im = Image.open(f"{tr['dir']}/{tr['names'][c]}").convert('RGB')
b = (tr['box'][c] / np.array([tr['W'], tr['H'], tr['W'], tr['H']])).tolist()
c2 = hires_crop(TFF.hflip(im), [1 - b[2], b[1], 1 - b[0], b[3]])
d = (c_noflip - torch.flip(c2, [2])).abs()
check('4 翻转后的上下文 crop 与原 crop 左右镜像', d[:, 4:-4, 4:-4].mean().item() < 0.05, f'平均差 {d[:, 4:-4, 4:-4].mean().item():.4f}（归一化单位）')

# 5
base = EyeTrunk(FakeBB()).eval()
ft = GF.ForecastTrunk(FakeBB(), C=1).eval()
missing, unexpected = ft.load_state_dict(base.state_dict(), strict=False)
check('5 ForecastTrunk 只多 time_emb', list(missing) == ['time_emb'] and not unexpected, f'{missing}')
imgs = torch.randn(2, 3, 448, 448)
boxes = [[[0.4, 0.3, 0.5, 0.45]], [[0.1, 0.1, 0.2, 0.25]]]
crops = torch.randn(2, 1, 3, 224, 224)
with torch.no_grad():
    a = ft.tokens(imgs, boxes, crops)
    ref = torch.cat([GM.Trunk.tokens(base, imgs, boxes), eye_tokens(base, None, boxes, crops[:, 0])], 1)
check('5 C=1 与 EyeTrunk 一致', (a - ref).abs().max().item() < 1e-5, f'max diff {(a - ref).abs().max().item():.1e}')
ft4 = GF.ForecastTrunk(FakeBB(), C=4).eval()
with torch.no_grad():
    a4 = ft4.tokens(imgs, boxes, torch.randn(2, 4, 3, 224, 224))
check('5 C=4 token 数', a4.shape[1] == 1024 + 4 * 256, str(tuple(a4.shape)))

# 6
m = GF.ForecastModel(FakeBB(), C=4)
m.load_init(f'{G}/experiments/flow_eye_d6_s0/epoch_14.pt')
sd = torch.load(f'{G}/experiments/flow_eye_d6_s0/epoch_14.pt', map_location='cpu', weights_only=True)
check('6 load_init 权重确实载入', torch.equal(m.head.out[1].weight, sd['head.out.1.weight']) and torch.equal(m.trunk.eye_proj.weight, sd['trunk.eye_proj.weight']))

print('\n全部通过' if not fails else f'\n失败 {len(fails)} 项：{fails}')
sys.exit(1 if fails else 0)
