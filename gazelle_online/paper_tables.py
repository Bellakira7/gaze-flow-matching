"""论文表格：从已有的 pred/bundle 出全部表（不重跑模型）。非判定项照报（G2.prereg §7），口径与 S1 一致，复用 setmetrics/metrics.py。
python -I -u paper_tables.py [--procs 64]
输出 results/paper/：per_instance_<arm>_<box>.npz（逐实例指标，可复用）、tables.md、tables.json。
所有臂统一走“pred.npz(hyps,w,samples) + 自身 bundle 的 σ_s”：点估计 = hyps[0]（预注册 §2 规则，对所有臂一视同仁）；
NLL = 样本管线 σ_s 2 折交叉拟合（check/bundle 的 nll_variants[:,0]），**与 G2 判定里"baseline 取 raw/cf 更低者"略有不同**，G2 判定数字以 results/g2/gate_flow_*.json 为准。
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/setmetrics')
import metrics as M  # noqa: E402
import g2_eval as E  # noqa: E402

R2 = f'{G}/results/g2'
OUT = f'{G}/results/paper'
KS3 = (1, 3, 5)
# 名称 -> (gt pred, gt bundle, eye pred, eye bundle)，都在 results/g2
ARMS = {
    'Official Gaze-LLE': ('pred_check_official_gt', 'bundle_check_official_gt', 'pred_check_official_eye', 'bundle_check_official_eye'),
    'B1 heatmap (s0)': ('pred_check_b1_s0', 'bundle_check_b1_s0', 'pred_check_b1_s0_eye', 'bundle_check_b1_s0_eye'),
    'B1 heatmap (lr3e-4)': ('pred_check_b1_lr3e-4_gt', 'bundle_check_b1_lr3e-4_gt', 'pred_check_b1_lr3e-4_eye', 'bundle_check_b1_lr3e-4_eye'),
    'B3 MDN': ('pred_mdn_gt', 'bundle_mdn_gt', 'pred_mdn_eye', 'bundle_mdn_eye'),
    'B4 MCL': ('pred_mcl_gt', 'bundle_mcl_gt', 'pred_mcl_eye', 'bundle_mcl_eye'),
    'B5 heatmap diffusion': ('pred_diff_gt', 'bundle_diff_gt', 'pred_diff_eye', 'bundle_diff_eye'),
    'Ours (flow)': ('pred_flow_gt', 'bundle_flow_gt', 'pred_flow_eye', 'bundle_flow_eye'),
    'Ours+eye token (flowET)': ('pred_flowET_gt', 'bundle_flowET_gt', 'pred_flowET_eye', 'bundle_flowET_eye'),
}
# flowET 的 seed 1（C14 读数之后才有文件；没有就不进表）
if all(os.path.exists(f'{R2}/{n}.npz') for n in ('pred_flowET_s1_gt', 'bundle_flowET_s1_gt', 'pred_flowET_s1_eye', 'bundle_flowET_s1_eye')):
    ARMS['Ours+eye token (flowET, seed 1)'] = ('pred_flowET_s1_gt', 'bundle_flowET_s1_gt', 'pred_flowET_s1_eye', 'bundle_flowET_s1_eye')
FLOW, FLOWET, FLOWET1 = 'Ours (flow)', 'Ours+eye token (flowET)', 'Ours+eye token (flowET, seed 1)'
# C22（2026-11-11）：主方法 flowET_d6（C18）的三个 seed、预注册 flow 的 seed 1、两个最强 baseline 的 seed 1。文件不全就不进表。
D6 = 'Ours main (flowET_d6)'
D6_1, D6_2 = 'Ours main (flowET_d6, seed 1)', 'Ours main (flowET_d6, seed 2)'
for _name, _m in ((D6, 'flowET_d6'), (D6_1, 'flowET_d6_s1'), (D6_2, 'flowET_d6_s2'), ('Ours (flow, seed 1)', 'flow_s1'),
                  ('B3 MDN (seed 1)', 'mdn_s1'), ('B4 MCL (seed 1)', 'mcl_s1')):
    _f = (f'pred_{_m}_gt', f'bundle_{_m}_gt', f'pred_{_m}_eye', f'bundle_{_m}_eye')
    if all(os.path.exists(f'{R2}/{n}.npz') for n in _f):
        ARMS[_name] = _f
# seed 汇总（C22：各 seed 各报，再报均值 ± 标准差；不在 seed 间挑选）
FAMILIES = {'Ours main (flowET_d6)': [D6, D6_1, D6_2], 'Ours+eye token (flowET)': [FLOWET, FLOWET1],
            'Ours (flow，预注册)': [FLOW, 'Ours (flow, seed 1)'], 'B4 MCL': ['B4 MCL', 'B4 MCL (seed 1)'], 'B3 MDN': ['B3 MDN', 'B3 MDN (seed 1)']}
GATES = (('flowET_d6', 'd6 s0（主行）'), ('flowET_d6_s1', 'd6 s1'), ('flowET_d6_s2', 'd6 s2'), ('flowET', 'flowET s0'), ('flowET_s1', 'flowET s1'),
         ('flow', 'flow s0（预注册）'), ('flow_s1', 'flow s1'))
_CTX = {}


def _one(i):
    c = _CTX
    A, hy, w, S, sig = c['inst'][i], c['hyps'][i], c['w'][i], c['S'][i], c['sigma'][i]
    r = M.hyp_metrics(hy, A, ks=KS3)
    mm = M.mode_metrics(hy, w, c['modes'][i], ks=KS3)
    P = E.pmf_from_samples(S, sig)
    pk = M.cell_index(np.clip(hy[:3], 0, 1 - 1e-9))
    pr, ob = M.disc_pairs(P, A, pk, 0.10, 3)
    out = {f'minL2@{K}': r[f'minL2@{K}'] for K in KS3}
    out.update({f'cov0.05@{K}': r[f'cov0.05@{K}'] for K in KS3})
    out.update({f'cov0.1@{K}': r[f'cov0.1@{K}'] for K in KS3})
    out['cham@5'] = r['cham@5']
    out.update(mm)
    out['w1'] = M.w1_grid(P, A)
    return out, pr, ob


def compute_arm(name, box, procs, modes, ctx):
    pn, bn = ARMS[name][:2] if box == 'gt' else ARMS[name][2:]
    path = f'{OUT}/per_instance_{pn}.npz'
    if os.path.exists(path):
        return dict(np.load(path))
    p, b = np.load(f'{R2}/{pn}.npz'), np.load(f'{R2}/{bn}.npz')
    N = len(ctx[0])
    _CTX.update(inst=ctx[0], hyps=np.asarray(p['hyps'], float), w=np.asarray(p['w'], float), S=p['samples'],
                sigma=np.asarray(b['sigma_s'], float), modes=modes)
    t0 = time.time()
    with get_context('fork').Pool(procs) as pool:
        res = pool.map(_one, range(N), chunksize=8)
    d = {k: np.array([r[0][k] for r in res]) for k in res[0][0]}
    d['ece_pred'] = np.stack([r[1] for r in res])
    d['ece_obs'] = np.stack([r[2] for r in res])
    d['nll'] = np.asarray(b['nll_variants'], float)[:, 0]
    for k in ('auc', 'avg_l2_fix', 'min_l2_fix'):
        d[k] = np.asarray(b[k], float)
    np.savez(path, **d)
    print(f'  [{name}/{box}] {time.time() - t0:.0f}s', flush=True)
    return d


def sub_mean(d, k, sel):
    return float(np.asarray(d[k], float)[sel].mean())


def sub_ece(d, sel):
    return M.ece(d['ece_pred'][sel].ravel(), d['ece_obs'][sel].ravel())


def table_rows(per, subsets, cols, sn):
    sel = subsets[sn]
    return [(arm, [fmt(c, d, sel) for c in cols]) for arm, d in per.items()]


def fmt(c, d, sel):
    if c == 'ece':
        return f'{sub_ece(d, sel)[0]:.3f}'
    if c == 'ece_gap':
        return f'{sub_ece(d, sel)[1]:+.3f}'
    v = sub_mean(d, c, sel)
    return f'{v:.4f}' if c in ('auc', 'avg_l2_fix', 'min_l2_fix', 'nll', 'w1') or c.startswith(('cov', 'modeP', 'modeR', 'minL2', 'cham')) else f'{v:.3f}'


NAMES = {'auc': 'AUC↑', 'avg_l2_fix': 'AvgL2↓', 'min_l2_fix': 'MinL2↓', 'nll': 'NLL↓', 'w1': 'W1↓', 'ece': 'ECE↓', 'ece_gap': 'ECE偏差',
         'minL2@1': 'minL2@1↓', 'minL2@3': 'minL2@3↓', 'minL2@5': 'minL2@5↓', 'cov0.1@1': 'cov.10@1↑', 'cov0.1@3': 'cov.10@3↑', 'cov0.1@5': 'cov.10@5↑',
         'cov0.05@5': 'cov.05@5↑', 'cham@5': 'cham@5↓', 'modeP@1': 'modeP@1↑', 'modeP@3': 'modeP@3↑', 'modeP@5': 'modeP@5↑', 'modeR@1': 'modeR@1↑',
         'modeR@3': 'modeR@3↑', 'modeR@5': 'modeR@5↑'}


def md(title, per, subsets, cols, sn, note=''):
    best = {}
    for j, c in enumerate(cols):
        vals = [float(r[1][j].replace('+', '')) for r in table_rows(per, subsets, cols, sn)]
        hi = c in ('auc',) or c.startswith(('cov', 'modeP', 'modeR'))
        best[j] = max(vals) if hi else (min(vals, key=abs) if c == 'ece_gap' else min(vals))
    lines = [f'**{title}**（子集 `{sn}`，n={int(subsets[sn].sum())}）{note}', '', '| 方法 | ' + ' | '.join(NAMES[c] for c in cols) + ' |', '|---|' + '---:|' * len(cols)]
    for arm, vs in table_rows(per, subsets, cols, sn):
        lines.append(f'| {arm} | ' + ' | '.join((f'**{v}**' if float(v.replace('+', '')) == best[j] else v) for j, v in enumerate(vs)) + ' |')
    return '\n'.join(lines) + '\n'


def paired(per, a, b, subsets, fidx, F, BI, keys):
    """a 减 b 的配对差，frame-cluster bootstrap 95% CI（与 g2_eval 同一套 BI）。"""
    rows = {}
    for sn in ('all', 'M_sep'):
        sel = subsets[sn]
        for k in keys:
            x = np.asarray(per[a][k], float) - np.asarray(per[b][k], float)
            bs = E._resample_means(x, sel, fidx, F, BI)
            rows[(sn, k)] = (float(x[sel].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=64)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    inst, fidx, hw, subsets, F = E.load_context()
    pi = np.load(f'{G}/results/g1/per_instance.npz')
    subsets = {'all': subsets['all'], 'unimodal': ~pi['multi'], 'M_sep': subsets['M_sep']}
    t0 = time.time()
    modes = [M.gt_modes(A) for A in inst]
    print(f'gt_modes {time.time() - t0:.0f}s', flush=True)
    ctx = (inst, fidx, hw, subsets, F)
    per = {'gt': {}, 'eye': {}}
    for box in ('gt', 'eye'):
        for arm in ARMS:
            per[box][arm] = compute_arm(arm, box, a.procs, modes, ctx)
    BI = np.random.default_rng(0).integers(0, F, (E.B_BOOT, F))

    md_ = ['# 论文表格（自动生成 by `gazelle_online/paper_tables.py`）', '',
           '> 口径：所有臂 10000 样本、K=5 个假设；点估计 = `hyps[0]`；NLL 为样本管线 σ_s 交叉拟合；W1 为 pmf 与标注经验分布的精确 W1；'
           'ECE 为 top-3 假设的圆盘质量（半径 0.10）校准（S1 定义）。**非判定项**；G2 判定见 `results/g2/gate_flow_*.json`。', '',
           '> **没有 minL2@10**：假设数固定为 K=5（预注册 §2），@10 不可得。**没有按标注数分桶**：99.6% 的实例标注数在 6–10（4762/4782），分桶无意义。', '',
           '> **`Ours+eye token (flowET)` 是预注册方法 flow 在 G2 判定 FAIL 之后追加的变体（C11–C13），主方法是看过测试集后选定的；表中 flow 与 flowET 两行都要报。**'
           'flowET 只有 1 个训练 seed（若表中有 seed 1 一行，则那是 C14 追加的第二个 seed，**不用于挑选**）。', '',
           '> **主方法 = `Ours main (flowET_d6)`（C18，看过测试集后由作者选定，原是 C15 的容量对照）**；seed 1 / 2 是 C22 追加的（读数前写规则），只报 seed 方差、不在 seed 间挑选。'
           'seed 1 / 2 与 MCL / MDN seed 1 在 DLC 上训练（同一镜像），三者训练中途各因 NCCL internal error 断过一次、从 epoch 8 的 last.pt 续训（优化器 / 调度器状态恢复，Python 随机状态未恢复）；s0 都在 DSW 上一次跑完。', '',
           '> **B1/官方的 `samples` 取自 raw 热图 pmf 抽样；点估计取 NMS 峰里质量最大者（`hyps[0]`）**，所以 AvgL2 与"热图 argmax"口径有 ~0.003 差（C3）。', '']
    std = ['auc', 'avg_l2_fix', 'min_l2_fix']
    for box, tag in (('gt', '真值头框（判定口径）'), ('eye', '眼点正方形头框（并排）')):
        md_ += [f'## 口径：{tag}', '']
        md_ += [md('表 1 标准指标 + 全体主指标', per[box], subsets, std + ['cov0.1@5', 'nll'], 'all'), '']
        for sn in ('all', 'unimodal', 'M_sep'):
            md_ += [md('表 2 set-aware 指标', per[box], subsets, ['minL2@1', 'minL2@3', 'minL2@5', 'cov0.1@5', 'cov0.05@5', 'cham@5', 'modeP@3', 'modeR@3', 'nll', 'w1', 'ece', 'ece_gap'], sn), '']
        md_ += [md('表 3 mode 精度/召回 @1/@3/@5', per[box], subsets, ['modeP@1', 'modeR@1', 'modeP@3', 'modeR@3', 'modeP@5', 'modeR@5'], 'all'), '',
                md('表 3 mode 精度/召回 @1/@3/@5', per[box], subsets, ['modeP@1', 'modeR@1', 'modeP@3', 'modeR@3', 'modeP@5', 'modeR@5'], 'M_sep'), '']
    # 配对比较
    keys = ['cov0.1@5', 'nll', 'modeP@3', 'modeR@3', 'minL2@1', 'minL2@5', 'w1', 'auc', 'avg_l2_fix', 'min_l2_fix']
    good = {'cov0.1@5': +1, 'nll': -1, 'modeP@3': +1, 'modeR@3': +1, 'minL2@1': -1, 'minL2@5': -1, 'w1': -1, 'auc': +1, 'avg_l2_fix': -1, 'min_l2_fix': -1}
    jt = {}
    md_ += ['## 配对比较：我方变体减对手（frame-cluster bootstrap B=2000，95% CI；M_sep 仅 291 实例，功效低）', '',
            '> 符号约定：“有利”= 差值的方向对我方变体有利（↑指标差>0、↓指标差<0）。CI 不含 0 才算有统计证据；M_sep 的 CI 大多会跨 0，**不能当成"没差别"也不能当成"有差别"**。'
            '这里的 CI 只含**测试实例**的抽样噪声，不含训练 seed 噪声（flowET 对 flowET seed 1 那一组才是 seed 间差异的直接度量）。', '']
    rivals = ('B4 MCL', 'B3 MDN', 'Official Gaze-LLE', 'B1 heatmap (s0)')
    pairs = [(FLOW, r, '') for r in rivals] + [(FLOWET, r, 'flowET:') for r in (FLOW,) + rivals]
    if FLOWET1 in ARMS:
        pairs += [(FLOWET1, r, 'flowET1:') for r in (FLOWET, FLOW) + rivals]
    if D6 in ARMS:
        d6s = [x for x in (D6, D6_1, D6_2) if x in ARMS]
        for k_, x in enumerate(d6s):
            pairs += [(x, r, f'd6s{k_}:') for r in (FLOWET, FLOW) + rivals]
        pairs += [(D6, r, 'd6s0x:') for r in ('B4 MCL (seed 1)', 'B3 MDN (seed 1)') if r in ARMS]
    for ours, rival, pre in pairs:
        for box in ('gt', 'eye'):
            rows = paired(per[box], ours, rival, subsets, fidx, F, BI, keys)
            md_ += [f'**{ours} − {rival}**（{box} 口径）', '', '| 指标 | 全体 Δ | 全体 CI95 | 有利? | M_sep Δ | M_sep CI95 | 有利? |', '|---|---:|---|---|---:|---|---|']
            for k in keys:
                cells = []
                for sn in ('all', 'M_sep'):
                    m, lo, hi = rows[(sn, k)]
                    sig = (lo > 0 or hi < 0)
                    favour = (m * good[k] > 0)
                    cells += [f'{m:+.4f}', f'[{lo:+.4f}, {hi:+.4f}]', ('有利·显著' if sig and favour else '不利·显著' if sig else '有利·不显著' if favour else '不利·不显著')]
                    jt[f'{pre}{box}|{rival}|{sn}|{k}'] = dict(delta=m, lo=lo, hi=hi)
                md_.append(f'| {NAMES[k] if k in NAMES else k} | ' + ' | '.join(cells) + ' |')
            md_ += ['']
    # seed 汇总
    sk = ['cov0.1@5', 'nll']
    md_ += ['## seed 汇总（C22）', '', '> 每格 = 各 seed 的子集均值，后面是 均值 ± 标准差（ddof=1）。各 seed 的完整数字见上面各表；**不在 seed 间挑选**。', '']
    for box in ('gt', 'eye'):
        md_ += [f'**{box} 口径**', '', '| 方法 | seeds | cov.10@5 全体 | cov.10@5 M_sep | NLL 全体 | NLL M_sep | AUC | AvgL2 | MinL2 |', '|---|---|---|---|---|---|---|---|---|']
        for fam, arms in FAMILIES.items():
            arms = [x for x in arms if x in ARMS]
            if not arms:
                continue
            cells = []
            for k, sn in [(c, 'all') for c in sk[:1]] + [('cov0.1@5', 'M_sep'), ('nll', 'all'), ('nll', 'M_sep'), ('auc', 'all'), ('avg_l2_fix', 'all'), ('min_l2_fix', 'all')]:
                v = np.array([sub_mean(per[box][x], k, subsets[sn]) for x in arms])
                cells.append(' / '.join(f'{x:.4f}' for x in v) + (f'（{v.mean():.4f} ± {v.std(ddof=1):.4f}）' if len(v) > 1 else ''))
            md_.append(f'| {fam} | {len(arms)} | ' + ' | '.join(cells) + ' |')
        md_ += ['']
    md_ += ['**G2 门槛读数逐 seed**（`results/g2/gate_<method>_<box>.json`；P1Δ = 漏检率相对降（门槛 0.10），P2 = NLL 改进；对同 backbone 最强 baseline，s0 基线集合，C2/C7）', '',
            '| 方法 | 口径 | 判定 | P1Δ 全体 [CI] | P1Δ M_sep [CI] | P2 全体 [CI] | P2 M_sep [CI] |', '|---|---|---|---|---|---|---|']
    for m_, lab in GATES:
        for box in ('gt', 'eye'):
            fp = f'{R2}/gate_{m_}_{box}.json'
            if not os.path.exists(fp):
                continue
            g = json.load(open(fp))
            a_, s_ = g['subsets']['all'], g['subsets']['M_sep']
            f1 = lambda p: f"{p['delta1_miss_reduction']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"  # noqa: E731
            f2 = lambda p: f"{p['improvement']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"  # noqa: E731
            md_.append(f"| {lab} | {box} | {g['verdict']} | {f1(a_['P1'])} | {f1(s_['P1'])} | {f2(a_['P2'])} | {f2(s_['P2'])} |")
    md_ += ['', '> d6 / flowET 的 PASS 都是事后追加方法的读数，不能写成"G2 通过"（C10 预注册方法 flow 判定 FAIL）。', '']
    # 效率
    md_ += ['## 效率', '', '| 方法 | 可训练参数(含头) | 其中头 | 采样每实例函数求值 |', '|---|---:|---:|---|']
    import re
    for arm, fn, nfe in (('B3 MDN', 'mdn_s0', '1（解析）'), ('B4 MCL', 'mcl_s0', '1（单次前向）'), ('B5 heatmap diffusion', 'diff_s0', '64 步 DDIM × M_draw=64（去噪网络 bf16）'), ('Ours (flow)', 'flow_s0', 'Heun 16 步 = 32 次'),
                         ('Ours+eye token (flowET)', 'flow_eye_s0', 'Heun 16 步 = 32 次 + 头框 crop 的 1 次额外 backbone 前向（256 个眼部 token，头外 +0.26M 参数）'),
                         (D6, 'flow_eye_d6_s0', '同 flowET，FlowHead 3 → 6 层')):
        t = open(f'{G}/logs/train_{fn}.log', errors='ignore').read()
        m = re.search(r'可训练参数 (\d+)（头 (\d+)）', t)
        md_.append(f'| {arm} | {int(m.group(1)):,} | {int(m.group(2)):,} | {nfe} |')
    lat = f'{OUT}/latency_c22.md'
    if os.path.exists(lat):  # C22 第 6 项：DLC 独占卡、同一进程重测全部行（含 d6）
        md_ += ['', '**延迟（C22，DLC 独占卡同进程重测，`bench_latency.py`）**', '', open(lat).read()]
    else:
        md_ += ['', '> 延迟：见 `results/paper/latency.md`（10-26，DSW 独占卡）。', '']
    open(f'{OUT}/tables.md', 'w').write('\n'.join(md_))
    json.dump(jt, open(f'{OUT}/tables.json', 'w'), indent=1)
    print('wrote', f'{OUT}/tables.md')


if __name__ == '__main__':
    main()
