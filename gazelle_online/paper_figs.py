"""M_sep 可视化：标注点 / B1 热图采样 / B4 MCL / flow / flowET（flow+眼部 token，C12–C13 事后追加变体）的 10000 样本密度 + 5 个假设。
python -I -u paper_figs.py   # 依赖 paper_tables.py 先跑完（要 results/paper/per_instance_*.npz）
输出 results/paper/fig_msep_random.png（M_sep 随机 8 个，seed 0，**无挑选**）与 fig_msep_selected.png（按预先写好的规则挑的 6 个，**是挑选的**），
以及 figs.json（实例号、挑选规则、两个方向的计数）。
挑选规则（看图之前写死）：在 M_sep 中按 (flow 的 modeR@3 − B1 的 modeR@3) 降序，其次 flow 的 minL2@5 升序，取前 6；
同时报告 flow 比 B1 好 / 差 / 持平的实例数，以及 flow 对 MCL 的同类计数，防止只展示好看的一面。
挑选规则仍按 **flow** 定义（与加 flowET 列之前的图保持一致，不因多了一列而重挑）；flowET 只是多一列，另外给 flowET 对 flow / B1 / MCL 的同类计数。
C22（11-11）再加主方法 flowET_d6 一列和它的同类计数，规则与随机 8 个的种子都不变。
"""
from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/setmetrics')
import g2_eval as E  # noqa: E402

OUT = f'{G}/results/paper'
R2 = f'{G}/results/g2'
COLS = [('B1 heatmap', 'pred_check_b1_s0'), ('B4 MCL', 'pred_mcl_gt'), ('Ours (flow)', 'pred_flow_gt'), ('Ours+eye token (flowET)', 'pred_flowET_gt'),
        ('Ours main (flowET_d6)', 'pred_flowET_d6_gt')]  # C22：主方法 d6（C18）只多一列，挑选规则不变


def density(S, g=64, sigma=1.5):
    c = np.bincount((np.clip(S[:, 1], 0, 1 - 1e-9) * g).astype(int) * g + (np.clip(S[:, 0], 0, 1 - 1e-9) * g).astype(int), minlength=g * g).reshape(g, g).astype(float)
    c = gaussian_filter(c, sigma)
    return c / c.max()


def panel(ax, img, A, box, S=None, hy=None, w=None, title=None):
    ax.imshow(img, extent=(0, 1, 1, 0), interpolation='bilinear', aspect=img.size[1] / img.size[0])  # 归一化坐标 + 原图宽高比
    if S is not None:
        d = density(S)
        ax.imshow(np.ma.masked_less(d, 0.04), extent=(0, 1, 1, 0), cmap='magma', alpha=0.62, vmin=0, vmax=1, interpolation='bilinear', aspect=img.size[1] / img.size[0])
    ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec='lime', lw=1.2))
    ax.scatter(A[:, 0], A[:, 1], s=16, c='cyan', ec='black', lw=0.6, zorder=4)
    if hy is not None:
        for k in range(len(hy)):
            ax.scatter(hy[k, 0], hy[k, 1], marker='*', s=60 + 520 * float(w[k]), c='white', ec='red', lw=0.9, zorder=5)
            if w[k] >= 0.08:
                ax.text(hy[k, 0] + 0.012, hy[k, 1] - 0.012, f'{100 * w[k]:.0f}%', color='white', fontsize=6.5, zorder=6,
                        bbox=dict(fc='black', alpha=0.5, pad=0.6, lw=0))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def draw(idx, path, frames, inst_fh, inst, preds, ttl):
    n = len(idx)
    fig, axs = plt.subplots(n, 1 + len(COLS), figsize=(3.1 * (1 + len(COLS)), 2.35 * n), squeeze=False)
    for r, i in enumerate(idx):
        fi, hi = inst_fh[i]
        fr = frames[fi]
        h = fr['heads'][hi]
        img = Image.open(f'{G}/data/gazefollow_test_hf/{fr["path"]}').convert('RGB')
        A, box = inst[i], h['bbox_norm']
        panel(axs[r, 0], img, A, box, title='Annotations (cyan) + head box' if r == 0 else None)
        axs[r, 0].set_ylabel(f'#{i}', fontsize=8)
        for c, (name, _) in enumerate(COLS):
            p = preds[name]
            panel(axs[r, c + 1], img, A, box, S=p['samples'][i].astype(float), hy=p['hyps'][i].astype(float), w=p['w'][i].astype(float),
                  title=name if r == 0 else None)
    fig.suptitle(ttl, fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=105, bbox_inches='tight')
    plt.close(fig)


def main():
    inst, fidx, hw, subsets, F = E.load_context()
    frames = json.load(open(f'{G}/data/gazefollow_test_hf/test_preprocessed.json'))
    inst_fh = [(fi, hi) for fi, fr in enumerate(frames) for hi in range(len(fr['heads']))]
    assert len(inst_fh) == len(inst)
    sep = np.where(subsets['M_sep'])[0]
    preds = {name: dict(np.load(f'{R2}/{pn}.npz')) for name, pn in COLS}
    pi = {k: np.load(f'{OUT}/per_instance_{k}.npz') for k in ('pred_check_b1_s0', 'pred_mcl_gt', 'pred_flow_gt', 'pred_flowET_gt', 'pred_flowET_d6_gt')}
    rng = np.random.default_rng(0)
    rand8 = sorted(rng.choice(sep, 8, replace=False).tolist())
    dR = pi['pred_flow_gt']['modeR@3'] - pi['pred_check_b1_s0']['modeR@3']
    order = sorted(sep.tolist(), key=lambda i: (-dR[i], pi['pred_flow_gt']['minL2@5'][i], i))
    sel6 = order[:6]
    cnt = {}
    for ours in ('pred_flow_gt', 'pred_flowET_gt', 'pred_flowET_d6_gt'):
        for rival in ('pred_check_b1_s0', 'pred_mcl_gt', 'pred_flow_gt'):
            if rival == ours:
                continue
            d = pi[ours]['modeR@3'][sep] - pi[rival]['modeR@3'][sep]
            cnt[f'{ours} vs {rival}'] = dict(ours_better=int((d > 1e-9).sum()), ours_worse=int((d < -1e-9).sum()), tie=int((np.abs(d) <= 1e-9).sum()), n=int(len(sep)))
    draw(rand8, f'{OUT}/fig_msep_random.png', frames, inst_fh, inst, preds, 'M_sep: 8 random instances (seed 0, not cherry-picked). Star = hypothesis (size ~ weight); heat = density of 10,000 samples')
    draw(sel6, f'{OUT}/fig_msep_selected.png', frames, inst_fh, inst, preds, '[SELECTED] M_sep instances where flow gains the most modeR@3 over B1 (rule in paper_figs.py); not representative of the average case')
    json.dump(dict(random8=rand8, selected6=sel6, modeR3_counts_in_Msep=cnt, rule='sort M_sep by (flow modeR@3 - B1 modeR@3) desc, then flow minL2@5 asc（按 flow 定义，flowET / flowET_d6 只是多列）'),
              open(f'{OUT}/figs.json', 'w'), indent=1)
    print('random8', rand8, '\nselected6', sel6, '\n', json.dumps(cnt, ensure_ascii=False))


if __name__ == '__main__':
    main()
