"""采样步数 / 求解器 / γ / σ_h 消融表：只是把 logs/sampler_selection.log（select_sampler.py 对 hold-out 调参的聚合输出）整理成 markdown，不重算任何东西。
python -I paper_ablation_sampler.py  ->  results/paper/ablation_sampler_holdout.md
数据来源与口径：训练 hold-out 3762 实例（单标注）、每实例 10000 样本、flow_s0 / mdn_s0 / diff_s0 的 epoch_14；这些配置在同一批 hold-out 上选出，所以数字只能横向比较，不是无偏估计。
"""
from pathlib import Path
import re

G = str(Path(__file__).resolve().parents[1])
txt = open(f'{G}/logs/sampler_selection.log', encoding='utf-8').read()


def block(start, stop):
    a = txt.index(start)
    b = txt.index(stop, a + 1) if stop else len(txt)
    return txt[a:b]


def rows(blk, prefix):
    out = []
    for ln in blk.splitlines():
        m = re.match(rf'\s*({prefix}\S*)\s+([\d.]+)\s+(\d)\s+([+\-][\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)', ln)
        if m:
            out.append((m.group(1), float(m.group(2)), float(m.group(4)), float(m.group(5)), float(m.group(6)), int(m.group(7)), float(m.group(8)), '<- 选' in ln))
    return out


L = ['# 采样器消融（hold-out，自动整理自 `logs/sampler_selection.log`，脚本 `gazelle_online/paper_ablation_sampler.py`）', '',
     '> 口径：训练 hold-out（3762 实例，**单标注**），每实例 10000 样本；NLL 为 σ_s∈{1,2,3} 格里最低者（实际全部选 σ_s=1）。**只能横向比较**，不能与测试集数字比；'
     '这批配置就是在这份 hold-out 上选的（C8 预声明规则），所以对"所选配置"是乐观的。选择在 `flow_s0` 上做，`flow_eye_s0` 直接沿用同一采样器。'
     'γ=3 是网格上沿、σ_h=1 是网格下沿（按预声明规则没有扩网格）。', '',
     '> "s/实例"是调参时与别的训练作业**共用 GPU** 时测的，含竞争，仅供相对比较；干净延迟见 `results/paper/latency.md`（若已测）。', '']
fl = rows(block('== Flow 采样器', '== B5'), 'flow')
L += ['## Flow：求解器 × 步数 × 时间网格 γ（13 个配置）', '', '| 配置（求解器/步数/γ） | NLL↓ | Δ vs 最优 | 配对 SE | cov.10@5(σ_h=2)↑ | 函数求值次数 | s/实例（含竞争） |', '|---|---:|---:|---:|---:|---:|---:|']
for k, nll, d, se, cov, nfe, t, sel in fl:
    _, sol, st, g = k.split('|')
    L.append(f'| {sol} / {st} / γ={g}{" **（选）**" if sel else ""} | {nll:.4f} | {d:+.4f} | {se:.4f} | {cov:.4f} | {nfe} | {t:.3f} |')
L += ['', '> 读法：Heun 16 步 γ=3（32 次函数求值）与 Euler 64 步 γ=2（64 次）、Heun 32 步 γ=3（64 次）的差都在 1 个配对 SE 之内，即 Heun 16→32 步没有可检出的收益；步数降到 8 步（Heun，16 次）NLL 变差 0.022（约 5.7 SE），Euler 8 步（γ=2）变差 0.091。'
      'Heun 16 步下 γ=1/2/3 的 NLL 单调变好（+0.034 / +0.007 / 0），趋势在 γ=3 处到达网格边缘，**没有验证 γ>3**。', '']
df = rows(block('== B5 步数', '-- B5 M_draw=16'), 'diff')
L += ['## B5（热图扩散）：DDIM 步数（M_draw=64）', '', '| 配置 | NLL↓ | Δ vs 最优 | 配对 SE | cov.10@5(σ_h=2)↑ | 函数求值次数 | s/实例（含竞争） |', '|---|---:|---:|---:|---:|---:|---:|']
for k, nll, d, se, cov, nfe, t, sel in df:
    L.append(f'| {k.replace("diff|", "").replace("|", " / 步数 ")}{" **（选）**" if sel else ""} | {nll:.4f} | {d:+.4f} | {se:.4f} | {cov:.4f} | {nfe} | {t:.3f} |')
for ln in block('-- B5 M_draw=16', '== σ_h').strip().splitlines()[1:]:
    m = re.match(r'\s*(diff\|M16\|\d+)\s+nll_best\s+([\d.]+)\s+cov10\s+([\d.]+)', ln)
    if m:
        L.append(f'| {m.group(1).replace("diff|", "").replace("|", " / 步数 ")}（不参与选择） | {float(m.group(2)):.4f} | | | {float(m.group(3)):.4f} | | |')
L += ['', '> B5 的 NLL 对 DDIM 步数没有随步数改善的趋势（非单调：16 步 +0.007、32 步 +0.024，约 2 个 SE，64 步最优是规则选出的），M_draw 影响大得多：每个 pmf 平均 16 张 vs 64 张热图，NLL 差约 0.2。', '']
sg = block('== σ_h', 'wrote ')
L += ['## 假设聚类带宽 σ_h（flow = Heun/16/γ3，与 MDN 一起选）', '', '| σ_h | flow cov.10@5 | MDN cov.10@5 | 两者平均 | flow cov.05@5 | MDN cov.05@5 | flow 点估计 L2 | MDN 点估计 L2 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for ln in sg.splitlines():
    m = re.match(r'\s*(\d)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\|\s*([\d.]+)\s+([\d.]+)\s*\|\s*([\d.]+)\s+([\d.]+)', ln)
    if m:
        L.append('| ' + ' | '.join([m.group(1)] + [f'{float(x):.4f}' for x in m.groups()[1:]]) + ' |')
L += ['', '> σ_h 越小 cov.10@5 越高（σ_h=1 最高，网格下沿，**没有验证 σ_h<1**）；点估计 L2 随 σ_h 略降（0.1220→0.1195，hold-out 单标注口径）。'
      'σ_h 只影响"从样本提取 5 个假设"这一步，**不影响 NLL**（NLL 直接用样本 pmf）。', '']
open(f'{G}/results/paper/ablation_sampler_holdout.md', 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
