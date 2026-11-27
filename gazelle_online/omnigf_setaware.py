"""C22 第 4 项：OmniGF（官方权重）在 GF test 上的 set-aware 读数（prereg/G2.clarifications_20261022.md §C22）。只用 CPU。
不改 g2_eval.py / metrics.py / g2_bundles.py / g2_read.py：热图 -> bundle 复用 g2_bundles.build，per_instance 复用
gazelle_cached/common._instance_metrics（官方 Gaze-LLE 的 results/repro_test/full.npz 就是它算的），配对比较复用 g2_eval.gate。

  envs/omnigf/bin/python -I gazelle_online/omnigf_setaware.py h5check   # 需 pandas+tables：H5 行 <-> G2 实例逐条核对
  /usr/local/bin/python -I gazelle_online/omnigf_setaware.py build      # 对齐核对 + 官方热图复现检查 + bundle_B{1,2}_omnigf_gt（已存在则跳过）
  /usr/local/bin/python -I gazelle_online/omnigf_setaware.py compare    # flowET_d6 各 seed（存在才算）对 OmniGF、OmniGF 对官方
                                                                        #   -> results/omnigf_setaware/summary.{md,json}
输入：results/omnigf/gf_test_dump.npz（11-10 用官方权重 + 官方代码、真值头框推理；path = test/%05d.jpg，%05d = 我们建的 H5 行号）。
H5 / 符号链接 / index.json 由 tools/omnigf_build_gf_test_h5.py 按 test_preprocessed.json 的（帧, 头）顺序生成，与 G2 load_context 同序。
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
from multiprocessing import Pool

import numpy as np

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/setmetrics')
import g2_eval as E  # noqa: E402

DUMP = f'{G}/results/omnigf/gf_test_dump.npz'
H5DIR = f'{G}/data/omnigf_gf_test'
JSONL = f'{G}/data/vqa/gazefollow_test.jsonl'
R = f'{G}/results/g2'
OUT = f'{G}/results/omnigf_setaware'
TAG = 'omnigf_gt'
SEEDS = (('flowET_d6', 's0'), ('flowET_d6_s1', 's1'), ('flowET_d6_s2', 's2'))
TOL = 1e-6


def frame_heads():
    frames = json.load(open(f'{G}/data/gazefollow_test_hf/test_preprocessed.json'))
    return frames, [(fi, hi) for fi, fr in enumerate(frames) for hi in range(len(fr['heads']))]


# ---------------- 对齐 ----------------
def h5check():
    import pandas as pd
    inst, fidx, hw, subsets, F = E.load_context()
    frames, fh = frame_heads()
    df = pd.read_hdf(f'{H5DIR}/vsgaze/gazefollow_test.h5', 'data')
    assert len(df) == len(inst), (len(df), len(inst))
    bad = []
    for n, row in enumerate(df.itertuples()):
        fi, hi = fh[n]
        hd = frames[fi]['heads'][hi]
        ok = (row.path == f'test/{n:05d}.jpg'
              and np.allclose(row.head_bboxes[0], hd['bbox_norm'], atol=TOL)
              and row.gaze_points_p1.shape == inst[n].shape and np.allclose(row.gaze_points_p1, inst[n], atol=TOL))
        if not ok:
            bad.append(n)
    print(f'[h5check] H5 {len(df)} 行 vs G2 {len(inst)} 实例：路径 / 头框 / 全部标注点 不一致 {len(bad)} 行 {bad[:10]}')
    sys.exit(1 if bad else 0)


def align():
    """返回按 G2 实例顺序排好的 OmniGF 热图与官方点；任何一条对不上就退出。"""
    inst, fidx, hw, subsets, F = E.load_context()
    frames, fh = frame_heads()
    N = len(inst)
    d = np.load(DUMP)
    rows = np.array([int(p.split('/')[1].split('.')[0]) for p in d['path']])
    assert len(rows) == N and (np.sort(rows) == np.arange(N)).all(), 'dump 行号不是 0..N-1 的排列'
    order = np.argsort(rows)
    index = json.load(open(f'{H5DIR}/index.json'))
    jl = [json.loads(x) for x in open(JSONL)]
    bad = {'index': [], 'symlink': [], 'jsonl': []}
    for n in range(N):
        fi, hi = fh[n]
        assert fidx[n] == fi
        ix = index[n]
        if (ix['row'], ix['frame_idx'], ix['head_idx'], ix['src']) != (n, fi, hi, frames[fi]['path']):
            bad['index'].append(n)
        link = f'{H5DIR}/gazefollow/test/{n:05d}.jpg'
        if os.path.realpath(link) != os.path.realpath(f'{G}/data/gazefollow_test_hf/{frames[fi]["path"]}'):
            bad['symlink'].append(n)
        r = jl[n]
        hd = frames[fi]['heads'][hi]
        if not (np.allclose(r['bbox_norm'], hd['bbox_norm'], atol=TOL) and len(r['gt_xs_norm']) == len(inst[n])
                and np.allclose(np.stack([r['gt_xs_norm'], r['gt_ys_norm']], 1), inst[n], atol=TOL)
                and r['images'][0].endswith(frames[fi]['path'])):
            bad['jsonl'].append(n)
    rep = {k: len(v) for k, v in bad.items()}
    print(f'[align] dump {N} 行（文件内顺序 == 行号：{bool((rows == np.arange(N)).all())}）；不一致：{rep}', flush=True)
    if any(rep.values()):
        sys.exit(f'对齐失败：{ {k: v[:10] for k, v in bad.items() if v} }')
    return d['heatmap'][order].astype(np.float32), d['point'][order], rep


# ---------------- per_instance ----------------
def per_instance(H, n_proc=64):
    sys.path.insert(0, f'{G}/gazelle_cached')
    import common  # noqa: E402  官方口径：auc / avg_l2 / min_l2（argmax 无 +0.5）/ avg_l2_fix / min_l2_fix
    frames, fh = frame_heads()
    jobs = []
    for n, (fi, hi) in enumerate(fh):
        hd, fr = frames[fi]['heads'][hi], frames[fi]
        jobs.append((H[n], hd['gazex_norm'], hd['gazey_norm'], fr['height'], fr['width']))
    with Pool(n_proc) as pool:
        return np.array(pool.map(common._instance_metrics, jobs, chunksize=16))


def _save_verified(path, arr_dict):
    tmp = path + '.tmp.npz'
    np.savez(tmp, **arr_dict)
    chk = np.load(tmp, allow_pickle=True)
    assert all(np.array_equal(chk[k], v) for k, v in arr_dict.items()), f'{tmp} 回读不一致（cpfs 写满？）'
    os.replace(tmp, path)


def _bundle_ok(p):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return False
    try:
        b = np.load(p, allow_pickle=True)
        return len(b['cov0.1@5']) == 4782
    except Exception:
        return False


def _build_verified(H, pi, tag, final_dir):
    """g2_bundles.build 先写到临时目录，回读校验后再挪到 final_dir（cpfs 写满时 np.savez 会静默写出 0 字节）。"""
    import g2_bundles as GB
    tmp = f'{OUT}/tmp_build_{tag}'
    os.makedirs(tmp, exist_ok=True)
    GB.build(H, pi, tmp, tag)
    out = []
    for b in ('B1', 'B2'):
        src = f'{tmp}/bundle_{b}_{tag}.npz'
        assert _bundle_ok(src), f'{src} 校验失败'
        if final_dir != tmp:
            os.replace(src, f'{final_dir}/bundle_{b}_{tag}.npz')
        out.append(f'{final_dir}/bundle_{b}_{tag}.npz')
    if final_dir != tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def build():
    os.makedirs(OUT, exist_ok=True)
    report = {}
    H, pt, rep = align()
    report['align'] = rep
    # 1) 管线复现检查：官方 Gaze-LLE 热图（与 OmniGF 一样是 fp16 落盘）走同一条路径
    full = np.load(f'{G}/results/repro_test/full.npz')
    Ho, pio_ref = full['heatmaps'].astype(np.float32), full['per_instance']
    pio = per_instance(Ho)
    d_pi = np.abs(pio - pio_ref).max(0)
    print(f'[repro] 官方热图 per_instance 重算 vs full.npz（fp32 时算的）逐列最大差 auc/avgL2/minL2/avgL2fix/minL2fix = {d_pi}', flush=True)
    report['repro_per_instance_maxabs'] = d_pi.tolist()
    report['repro_per_instance_n_diff'] = [int((np.abs(pio[:, c] - pio_ref[:, c]) > TOL).sum()) for c in range(5)]
    rb = f'{OUT}/repro_official'
    os.makedirs(rb, exist_ok=True)
    if not _bundle_ok(f'{rb}/bundle_B1_official_repro.npz'):
        _build_verified(Ho, pio_ref, 'official_repro', rb)
    mine, off = E._load(f'{rb}/bundle_B1_official_repro.npz'), E.official_bundle()
    rdiff = {}
    for k in ('cov0.1@5', 'nll_variants', 'nll_exact_variants', 'auc', 'avg_l2_fix', 'min_l2_fix', 'modeP@3', 'modeR@3'):
        rdiff[k] = float(np.abs(np.asarray(mine[k], float) - np.asarray(off[k], float)).max())
    print(f'[repro] g2_bundles.build(官方热图) vs 冻结 official bundle v2 逐键最大差：{rdiff}', flush=True)
    report['repro_bundle_maxabs'] = rdiff
    # 2) OmniGF
    pi = per_instance(H)
    _save_verified(f'{OUT}/omnigf_per_instance.npz', {'per_instance': pi, 'point_official': pt})
    own_avg = np.array([np.hypot(*(pt[n] - a.mean(0))) for n, a in enumerate(E.load_context()[0])])
    own_min = np.array([np.hypot(*(a - pt[n]).T).min() for n, a in enumerate(E.load_context()[0])])
    report['omnigf_per_instance_mean'] = dict(zip(['auc', 'avg_l2', 'min_l2', 'avg_l2_fix', 'min_l2_fix'], pi.mean(0).tolist()))
    report['omnigf_own_point'] = {'avg_l2': float(own_avg.mean()), 'min_l2': float(own_min.mean()),
                                  'n_argmax_differs_from_official_point': int((np.abs(np.stack(np.unravel_index(H.reshape(len(H), -1).argmax(1), (64, 64)))[::-1].T / 64 - pt) > 1e-6).any(1).sum())}
    report['omnigf_heatmap'] = {'min': float(H.min()), 'max': float(H.max()), 'neg_frac': float((H < 0).mean())}
    print(f'[omnigf] per_instance 均值 {report["omnigf_per_instance_mean"]}；官方点口径 {report["omnigf_own_point"]}；热图 {report["omnigf_heatmap"]}', flush=True)
    finals = [f'{R}/bundle_{b}_{TAG}.npz' for b in ('B1', 'B2')]
    if all(os.path.exists(p) for p in finals):
        print(f'[omnigf] SKIP：{finals} 已存在，不覆盖', flush=True)
        report['bundles'] = 'skipped (exist)'
    else:
        assert not any(os.path.exists(p) for p in finals), f'只存在一半：{finals}，人工检查后再跑'
        report['bundles'] = _build_verified(H, pi, TAG, R)
    json.dump(report, open(f'{OUT}/build_report.json', 'w'), indent=1, ensure_ascii=False)
    print('[build] 完成', flush=True)


# ---------------- 比较 ----------------
def _ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def _paired(m, b, sel, fidx, F, BI):
    """均值差 m - b（逐实例配对、按帧 cluster bootstrap，与 gate 同一组重采样下标）。"""
    m, b = np.asarray(m, float), np.asarray(b, float)
    d = _resample_means(m, sel, fidx, F, BI) - _resample_means(b, sel, fidx, F, BI)
    return {'method': float(m[sel].mean()), 'base': float(b[sel].mean()), 'diff': float(m[sel].mean() - b[sel].mean()), 'ci95': _ci(d)}


_resample_means = E._resample_means


def row(b, subsets):
    V = np.asarray(b['nll_variants'], float)
    r = {}
    for sn, sel in subsets.items():
        r[f'cov_{sn}'] = float(np.asarray(b['cov0.1@5'], float)[sel].mean())
        k = int(np.argmin(V[sel].mean(0)))
        r[f'nll_{sn}'] = float(V[sel, k].mean())
        r[f'nll_{sn}_variant'] = str(b['nll_variant_names'][k])
    for k in ('auc', 'avg_l2_fix', 'min_l2_fix'):
        r[k] = float(np.asarray(b[k], float).mean())
    return r


def versus(method, bases, ctx, BI, std):
    """P1 / P2 直接用冻结的 g2_eval.gate（最强 baseline 的挑法与 g2_read 相同；只取数字，不报 verdict）；
    另补标准指标与原始 cov 差的配对区间。"""
    inst, fidx, hw, subsets, F = ctx
    g = E.gate(method, bases, required=(), std_baseline=std, ctx=ctx)
    out = {}
    for sn, sel in subsets.items():
        s = g['subsets'][sn]
        cb = bases[s['P1']['baseline']]['cov0.1@5']
        out[sn] = {'n': s['n'], 'P1': s['P1'], 'P2': s['P2'], 'cov_diff': _paired(method['cov0.1@5'], cb, sel, fidx, F, BI)}
    allsel = subsets['all']
    out['std'] = {k: _paired(method[k], bases[std][k], allsel, fidx, F, BI) for k in ('auc', 'avg_l2_fix', 'min_l2_fix')}
    return out


def compare():
    ctx = E.load_context()
    inst, fidx, hw, subsets, F = ctx
    BI = np.random.default_rng(0).integers(0, F, (E.B_BOOT, F))  # 与 gate 相同
    omni = {'OMNIGF_B1': E._load(f'{R}/bundle_B1_{TAG}.npz'), 'OMNIGF_B2': E._load(f'{R}/bundle_B2_{TAG}.npz')}
    off = E.official_bundle()
    rows = {'Gaze-LLE ViT-L 官方（B1 型 bundle）': row(off, subsets), 'OmniGF B1（NMS 前 5 峰）': row(omni['OMNIGF_B1'], subsets),
            'OmniGF B2（热图采样 5 点）': row(omni['OMNIGF_B2'], subsets)}
    res = {'rows': rows, 'vs_omnigf': {}, 'note': 'P1/P2 由 g2_eval.gate 计算，baseline = {OmniGF B1, OmniGF B2} 中各指标最强者；标准指标对 OmniGF B1（argmax 点）。不是 G2 判定。'}
    for name, seed in SEEDS:
        p = f'{R}/bundle_{name}_gt.npz'
        if not os.path.exists(p):
            continue
        m = E._load(p)
        rows[f'Ours flowET_d6 {seed}'] = row(m, subsets)
        res['vs_omnigf'][seed] = versus(m, omni, ctx, BI, 'OMNIGF_B1')
    # OmniGF 对官方 Gaze-LLE：OmniGF 两个 bundle 各作一次 "method"，baseline 只有官方
    res['omnigf_vs_official'] = {k: versus(v, {'OFFICIAL': off}, ctx, BI, 'OFFICIAL') for k, v in omni.items()}
    # C22 第 4 项措辞规则：OmniGF 在 cov 或 NLL 任一项（全体或 M_sep）上不差于我们 -> 主张收窄为「同一冻结 backbone 下」
    rule = {}
    for seed, v in res['vs_omnigf'].items():
        cells = {}
        for sn in ('all', 'M_sep'):
            cells[f'cov_{sn}'] = {'ours_better_point': v[sn]['P1']['delta1_miss_reduction'] > 0, 'ci_lower_gt0': v[sn]['P1']['ci95'][0] > 0}
            cells[f'nll_{sn}'] = {'ours_better_point': v[sn]['P2']['improvement'] > 0, 'ci_lower_gt0': v[sn]['P2']['ci95'][0] > 0}
        rule[seed] = {'cells': cells, 'omnigf_not_worse_point': not all(c['ours_better_point'] for c in cells.values()),
                      'omnigf_not_sig_worse': not all(c['ci_lower_gt0'] for c in cells.values())}
    trig_point = any(r['omnigf_not_worse_point'] for r in rule.values())
    trig_ci = any(r['omnigf_not_sig_worse'] for r in rule.values())
    res['c22_item4_rule'] = {'per_seed': rule, 'triggered_by_point_estimate': trig_point, 'triggered_by_ci': trig_ci,
                             'claim': '在同一冻结 backbone 下超过所有 baseline（表中如实列出 OmniGF）' if (trig_point or trig_ci)
                             else '超过所有 baseline（含 OmniGF；cov/NLL 全体与 M_sep 的 CI 下界均 > 0）'}
    os.makedirs(OUT, exist_ok=True)
    tmpj = f'{OUT}/summary.json.tmp'
    json.dump(res, open(tmpj, 'w'), indent=1, ensure_ascii=False)
    json.load(open(tmpj))
    os.replace(tmpj, f'{OUT}/summary.json')
    md = render(res)
    tmpm = f'{OUT}/summary.md.tmp'
    open(tmpm, 'w').write(md)
    assert open(tmpm).read() == md
    os.replace(tmpm, f'{OUT}/summary.md')
    print(md)


def render(res):
    br = json.load(open(f'{OUT}/build_report.json')) if os.path.exists(f'{OUT}/build_report.json') else {}
    L = ['# OmniGF set-aware 读数（C22 第 4 项，GF test，gt 口径）', '',
         '> 生成：`/usr/local/bin/python -I gazelle_online/omnigf_setaware.py compare`。区间 = 按帧 cluster bootstrap 2000 次、seed 0（与 g2_eval.gate 同一组下标），未校正多重比较。',
         '> OmniGF 是不同 backbone 的第三方模型，**不进** G2 同 backbone 基线集合、不重算 G2 判定（C22）。', '']
    if br:
        L += ['## 对齐与管线复现', '',
              f'- 对齐：dump 4782 行按 H5 行号对到 G2 实例顺序；index.json / 符号链接指向的原图 / 主表 jsonl 的头框与全部标注点 不一致数 = {br.get("align")}（h5check 另查 H5 本体）。',
              f'- 官方 Gaze-LLE 热图走同一路径：per_instance 重算（fp16 热图）对 full.npz（fp32 时算的）逐列最大差 = {np.round(br["repro_per_instance_maxabs"], 8).tolist()}，'
              f'超过 1e-6 的实例数 = {br["repro_per_instance_n_diff"]}；bundle 逐键最大差 = {br["repro_bundle_maxabs"]}。',
              f'- OmniGF 热图是解码器原始输出（min {br["omnigf_heatmap"]["min"]:.3f}、max {br["omnigf_heatmap"]["max"]:.3f}、负值格占 {br["omnigf_heatmap"]["neg_frac"]:.1%}）；'
              '冻结的 `heat_to_pmf` 把负值截到 1e-12 再与均匀分布按 λ=1e-3 混合，与官方 Gaze-LLE 同一处理，未做额外变换。',
              f'- OmniGF 自己的点（官方 `spatial_argmax2d`，idx/64，fp32 热图）：AvgL2 {br["omnigf_own_point"]["avg_l2"]:.4f} / MinL2 {br["omnigf_own_point"]["min_l2"]:.4f}'
              f'（自评 0.0929 / 0.0411）；本表 fix 口径 = fp16 热图 argmax 格子中心 (idx+0.5)/64，与官方 Gaze-LLE 行同口径。'
              f'fp16 热图的 argmax 与官方点不同的实例 {br["omnigf_own_point"]["n_argmax_differs_from_official_point"]} 个（含 +0.5 偏移之外的格子差异）。', '']
    L += ['## 各行', '', '| 行 | cov 全体 | cov M_sep | NLL 全体 | NLL M_sep | AUC | AvgL2 | MinL2 |', '|---|---|---|---|---|---|---|---|']
    for k, r in res['rows'].items():
        L.append(f'| {k} | {r["cov_all"]:.4f} | {r["cov_M_sep"]:.4f} | {r["nll_all"]:.4f} ({r["nll_all_variant"]}) | {r["nll_M_sep"]:.4f} ({r["nll_M_sep_variant"]}) | '
                 f'{r["auc"]:.4f} | {r["avg_l2_fix"]:.4f} | {r["min_l2_fix"]:.4f} |')

    def block(title, v):
        out = [f'### {title}', '', '| 子集 | P1 最强 baseline | 漏检率相对降 Δ1 [CI] | cov 差 [CI] | P2 最强 baseline | NLL 改进（正=方法好）[CI] |', '|---|---|---|---|---|---|']
        for sn in ('all', 'M_sep'):
            s = v[sn]
            out.append(f'| {sn} (n={s["n"]}) | {s["P1"]["baseline"]} | {s["P1"]["delta1_miss_reduction"]:+.3f} [{s["P1"]["ci95"][0]:+.3f}, {s["P1"]["ci95"][1]:+.3f}] | '
                       f'{s["cov_diff"]["diff"]:+.4f} [{s["cov_diff"]["ci95"][0]:+.4f}, {s["cov_diff"]["ci95"][1]:+.4f}] | {s["P2"]["baseline"]} | '
                       f'{s["P2"]["improvement"]:+.4f} [{s["P2"]["ci95"][0]:+.4f}, {s["P2"]["ci95"][1]:+.4f}] |')
        out += ['', '标准指标（全体，方法 − baseline）：' + '；'.join(
            f'{k} {s["diff"]:+.4f} [{s["ci95"][0]:+.4f}, {s["ci95"][1]:+.4f}]' for k, s in v['std'].items()), '']
        return out
    L += ['', '## flowET_d6 各 seed 对 OmniGF', '']
    for seed, v in res['vs_omnigf'].items():
        L += block(f'flowET_d6 {seed} 对 OmniGF', v)
    L += ['## OmniGF 对官方 Gaze-LLE（参照）', '']
    for k, v in res['omnigf_vs_official'].items():
        L += block(f'{k} 对官方 Gaze-LLE', v)
    r = res['c22_item4_rule']
    L += ['## C22 第 4 项措辞规则', '',
          f'- 按点估计（OmniGF 在 cov 或 NLL 任一格不差于我们）触发：{r["triggered_by_point_estimate"]}；按区间（任一格我方优势 CI 下界 ≤ 0）触发：{r["triggered_by_ci"]}。',
          f'- 论文总主张：**{r["claim"]}**', '']
    for seed, x in r['per_seed'].items():
        L.append(f'- {seed}：' + '，'.join(f'{c} 点估计我方更好={v["ours_better_point"]} / CI下界>0={v["ci_lower_gt0"]}' for c, v in x['cells'].items()))
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['h5check', 'build', 'compare'])
    a = ap.parse_args()
    {'h5check': h5check, 'build': build, 'compare': compare}[a.cmd]()
