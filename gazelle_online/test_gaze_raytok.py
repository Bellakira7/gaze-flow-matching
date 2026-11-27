"""gaze_ray v2（射线 token）自检，CPU 即可：python -I test_gaze_raytok.py
1. 射线采样点：第一个点离头中心 ≤ smax/(2T)，最后一个点在 [0,1]² 内且离边界 ≤ smax/(2T)，相邻点共线等距；
2. 轴向注意力的局部性：射线内层只在同一条射线内传播；射线间层只在同一 τ 档内传播；
3. 输出形状 [B, K*T, D]，梯度能回到 trunk token。
"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gazelle_online'))
import gaze_ray as GR  # noqa: E402

torch.manual_seed(0)
fails = []


def check(name, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  {info}' if info else ''), flush=True)
    if not ok:
        fails.append(name)


R = GR.RayTokens()
K, T = R.K, R.T
c = torch.rand(256, 2)
pts = R.points(c)
smax = GR.ray_len(c[:, None].expand(-1, K, -1), GR.unit(R.th)[None].expand(256, -1, -1))
step = smax / T
check('第一个点在 c 外半步', torch.allclose((pts[:, :, 0] - c[:, None]).norm(dim=-1), step / 2, atol=1e-5))
last_gap = torch.minimum(pts[:, :, -1], 1 - pts[:, :, -1]).min(-1).values
check('最后一个点在图内（ray_len 有 1e-4 下限，容差 1e-4）、离边界不超过半步', bool((pts[:, :, -1] >= -1e-4).all() & (pts[:, :, -1] <= 1 + 1e-4).all()) and bool((last_gap <= step / 2 + 1e-5).all()))
dd = pts[:, :, 1:] - pts[:, :, :-1]
check('相邻点等距', torch.allclose(dd.norm(dim=-1), step[..., None].expand_as(dd[..., 0]), atol=1e-5))

x = torch.randn(2, K, T, R.dim)
x2 = x.clone()
x2[:, 5] += torch.randn(2, T, R.dim)
with torch.no_grad():
    y, y2 = R.layer(0, x), R.layer(0, x2)
diff = (y - y2).abs().amax(dim=(0, 2, 3))
check('射线内层只改变被改的那条射线', diff[5] > 1e-3 and torch.cat([diff[:5], diff[6:]]).max() == 0, f'被改射线 {diff[5]:.3f}，其它最大 {torch.cat([diff[:5], diff[6:]]).max():.1e}')
x3 = x.clone()
x3[:, 5, 3] += torch.randn(2, R.dim)
with torch.no_grad():
    y, y3 = R.layer(1, x), R.layer(1, x3)
dt = (y - y3).abs().amax(dim=(0, 1, 3))
dk = (y - y3).abs()[:, :, 3].amax(dim=(0, 2))
check('射线间层只改变同一 τ 档', dt[3] > 1e-3 and torch.cat([dt[:3], dt[4:]]).max() == 0 and bool((dk > 0).all()),
      f'τ=3 {dt[3]:.3f}，其它 τ 最大 {torch.cat([dt[:3], dt[4:]]).max():.1e}；τ=3 上受影响的射线 {int((dk > 0).sum())}/{K}')

tok = torch.randn(3, GR.N_SCENE + 256, R.dim, requires_grad=True)
out = R(tok, torch.rand(3, 2))
check('输出形状', tuple(out.shape) == (3, K * T, R.dim), str(tuple(out.shape)))
out.sum().backward()
g = tok.grad[:, :GR.N_SCENE].abs().sum().item()
check('梯度回到场景 token、不碰眼部 token', g > 0 and tok.grad[:, GR.N_SCENE:].abs().sum().item() == 0, f'场景 {g:.2f}')
print('\n全部通过' if not fails else f'\n失败 {len(fails)} 项：{fails}')
sys.exit(1 if fails else 0)
