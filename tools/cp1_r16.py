#!/usr/bin/env python
"""CP1 的 R16 与配套数字（prereg/CP1_CP2.prereg.md §2 / §5 / §7）。纯 CPU，读 `eval_gaze.py infer` 的 preds.jsonl，不跑模型。

  /usr/local/bin/python -I tools/cp1_r16.py select --preds ckpt2500=results/sft_v1/ckpt2500_val_greedy/preds.jsonl ckpt3000=... ckpt3434=...
  /usr/local/bin/python -I tools/cp1_r16.py r16 --preds results/sft_v1/<ckpt>_val_k16_T1.0/preds.jsonl --out results/sft_v1/<ckpt>_val_k16_T1.0/r16.json
  /usr/local/bin/python -I tools/cp1_r16.py r16 --preds <零样本 K=16 的 preds.jsonl> --label zeroshot --no-verdict --out ...
  /usr/local/bin/python -I tools/cp1_r16.py selftest

select  §2：在 **val_sel** 上按 greedy Avg L2 挑 checkpoint。只读 greedy，只打印 val_sel；**不打印 val_gate**，
        也不碰采样（"选择时不看任何采样指标"）。
r16     §2：
  主判定（val_gate）  R16 = (L2_greedy − L2_best16) / L2_greedy，两项都是均值；按**帧** cluster bootstrap（2000 次，seed 0），
                      每次重采样内先各自取均值再相除；区间 = 2.5 / 97.5 分位。判定 通过 / 边缘 / 不通过 见 verdict()。
  描述性（val_gate、val_sel、全 val 三套都报）：best-of-{2,4,8}、avg-of-{2,4,8,16}、R_k；敏感性（仅成功解析的采样）；
                      采样离散度（两两距离中位数、DBSCAN 多峰占比）；聚合器 medoid / mean-shift 的 Avg L2 与 AUC；采样与 greedy 的 AUC。
  读数前检查（§5.5 / §5.6，不过就非零退出、不输出数字）：meta 里 num_samples == 16、temperature 与 --temperature 一致、top_p == 1.0；
                      每个样本恰有 16 个采样；best-of-k 前缀单调；切分文件的 sha256 与 split.json 一致。

口径细节（prereg 没写死的地方，按下面做；若要改，改这里并在报告里写偏离）:
  * L2 口径：解析失败 / 回答 out → 记画面中心 (0.5,0.5)（§1），与 eval_gaze.py 同一套函数（make_instance / score_instances）。
  * best-of-k 取前 k 个采样的最小值（§1 的前缀嵌套）。
  * 敏感性「仅在成功解析的采样上取」：status ∈ {ok, clamped} 的采样才参与 min；某人 16 个采样全失败时，没有成功采样可取，
    退回与主分析相同的值（全是中心点，min 就是中心点 L2），并在 n_no_parsed_sample 里单独报数。
  * 离散度 / 聚合器只用成功解析的采样点（clamped 已夹进 [0,1]）。离散度 = 每人成功解析点的两两距离中位数（≥2 点才有），
    再对人取中位数；多峰 = DBSCAN(eps=0.075, min_samples=3) 得到 ≥2 个簇（同 tools/g1_analysis.py），点数 <3 的人算非多峰，分母是全部人。
  * medoid：成功解析点里到其余点距离和最小者（并列取下标小的）；无成功解析点取画面中心。
  * mean-shift：高斯核 exp(-d²/(2·0.05²))，从 medoid 出发，位移 < 1e-4 停，最多 200 次；无成功解析点取画面中心。
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

G = str(Path(__file__).resolve().parents[1])
HERE = os.path.dirname(os.path.abspath(__file__))
KS = (2, 4, 8, 16)
K_REQ = 16
CENTER = (0.5, 0.5)
THR = 0.30            # CP1 阈值（prereg §2，来自 TODO）
BOOT_B, BOOT_SEED = 2000, 0
DB_EPS, DB_MIN = 0.075, 3
MS_BW, MS_TOL, MS_ITERS = 0.05, 1e-4, 200
OK_STATUS = ('ok', 'clamped')


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m  # score_instances 里的 multiprocessing.Pool 要按模块名 pickle 函数
    spec.loader.exec_module(m)
    return m


E = _load('eval_gaze', 'eval_gaze.py')
SPLIT = _load('cp_split', 'cp_split.py')


def sha256_file(p):
    return SPLIT.sha256_file(p)


# ------------------------------------------------------------------------------------------------------------------
# 纯函数（selftest 逐个手算校验）
# ------------------------------------------------------------------------------------------------------------------
def boot_ratio(g, b, frame_ids, B=BOOT_B, seed=BOOT_SEED):
    """R = (mean g − mean b) / mean g 的点估计与 95% 区间；按帧 cluster bootstrap，重采样内先各自取均值再相除。"""
    g, b = np.asarray(g, float), np.asarray(b, float)
    _, inv = np.unique(np.asarray(frame_ids), return_inverse=True)
    F = int(inv.max()) + 1
    BI = np.random.default_rng(seed).integers(0, F, (B, F))
    cnt = np.bincount(inv, minlength=F).astype(float)
    mg = np.bincount(inv, weights=g, minlength=F)[BI].sum(1) / cnt[BI].sum(1)
    mb = np.bincount(inv, weights=b, minlength=F)[BI].sum(1) / cnt[BI].sum(1)
    R = (mg - mb) / mg
    pt = float((g.mean() - b.mean()) / g.mean())
    return {'point': pt, 'ci95': [float(np.percentile(R, 2.5)), float(np.percentile(R, 97.5))]}


def medoid(P):
    P = np.asarray(P, float)
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    return P[int(np.argmin(D.sum(1)))]


def mean_shift(P, x0, bw=MS_BW, tol=MS_TOL, max_iter=MS_ITERS):
    P, x = np.asarray(P, float), np.asarray(x0, float)
    for _ in range(max_iter):
        w = np.exp(-((P - x) ** 2).sum(1) / (2.0 * bw * bw))
        xn = (w[:, None] * P).sum(0) / w.sum()
        shift = float(np.hypot(*(xn - x)))
        x = xn
        if shift < tol:
            break
    return x


def parsed_points(samples):
    return np.array([pt for pt, st in samples if pt is not None], dtype=float).reshape(-1, 2)


def aggregate_points(samples):
    """-> (medoid_pt, mean_shift_pt)。无成功解析点时两者都是画面中心。"""
    P = parsed_points(samples)
    if len(P) == 0:
        return CENTER, CENTER
    m = medoid(P)
    return (float(m[0]), float(m[1])), tuple(float(v) for v in mean_shift(P, m))


def n_clusters(P, eps=DB_EPS, min_samples=DB_MIN):
    if len(P) < min_samples:
        return 0
    from sklearn.cluster import DBSCAN
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(P)
    return len(set(lab) - {-1})


def median_pairwise(P):
    if len(P) < 2:
        return None
    iu = np.triu_indices(len(P), 1)
    return float(np.median(np.hypot(P[iu[0], 0] - P[iu[1], 0], P[iu[0], 1] - P[iu[1], 1])))


def verdict(r16_gate, r16_all, r16_parsed_gate, thr=THR):
    """prereg §2 的判定。输入是三个 R16：val_gate 主分析（{'point','ci95'}）、全 val 点估计、val_gate 仅成功解析采样点估计。
    边缘条件先于「不通过」检查：边缘就是「交用户裁决」，与「点估计 < 30%」同时成立时也只会更保守地交给用户，不会自动判过。"""
    pt, lo = r16_gate['point'], r16_gate['ci95'][0]
    reasons = []
    if pt >= thr and lo < thr:
        reasons.append(f'点估计 {pt:.4f} ≥ {thr} 但区间下界 {lo:.4f} < {thr}')
    if (pt >= thr) != (r16_all >= thr):
        reasons.append(f'val_gate 点估计 {pt:.4f} 与全 val 点估计 {r16_all:.4f} 分居 {thr} 两侧')
    if (pt >= thr) != (r16_parsed_gate >= thr):
        reasons.append(f'主分析 {pt:.4f} 与「仅计成功解析的采样」{r16_parsed_gate:.4f} 分居 {thr} 两侧')
    if reasons:
        return {'verdict': '边缘（交用户裁决）', 'reasons': reasons, 'point_below_thr': pt < thr}
    if pt >= thr and lo >= thr:
        return {'verdict': '通过', 'reasons': [f'R16 {pt:.4f}，区间下界 {lo:.4f} ≥ {thr}']}
    return {'verdict': '不通过', 'reasons': [f'点估计 {pt:.4f} < {thr}']}


# ------------------------------------------------------------------------------------------------------------------
# 读入与检查
# ------------------------------------------------------------------------------------------------------------------
def load_preds(paths):
    recs = {}
    for p in paths:
        for r in E.load_jsonl(p):
            recs[r['id']] = r
    return recs


def check_meta(preds_paths, expect_T):
    """§5.5：meta 里的采样参数与声称值一致。多个 preds 文件时每个都要过。"""
    for p in preds_paths:
        mp = p + '.meta.json'
        if not os.path.exists(mp):
            sys.exit(f'{mp} 不存在：没有落盘的采样参数，无法核对（§5.5）')
        m = json.load(open(mp))
        bad = []
        if m.get('num_samples') != K_REQ:
            bad.append(f"num_samples={m.get('num_samples')}（要求 {K_REQ}）")
        if m.get('temperature') is None or abs(m['temperature'] - expect_T) > 1e-9:
            bad.append(f"temperature={m.get('temperature')}（声称 {expect_T}）")
        if m.get('top_p') != 1.0:
            bad.append(f"top_p={m.get('top_p')}（要求 1.0）")
        if bad:
            sys.exit(f'{mp} 与 prereg §1 / §5.5 不符: ' + '; '.join(bad))
        print(f"  meta OK  adapter={m.get('adapter')}  num_samples={m['num_samples']}  T={m['temperature']}  top_p={m['top_p']}  seed={m.get('seed')}")


def build(rows, recs):
    """-> insts, per（score_instances 的逐人结果，与 rows 对齐）。"""
    missing = [r['id'] for r in rows if r['id'] not in recs]
    if missing:
        sys.exit(f'{len(missing)}/{len(rows)} 条样本没有预测（例 {missing[:3]}），拒绝打分')
    insts = [E.make_instance(r, recs[r['id']]) for r in rows]
    bad = [i['id'] for i in insts if len(i['samples']) != K_REQ]
    if bad:
        sys.exit(f'{len(bad)} 条样本的采样数 != {K_REQ}（例 {bad[:3]}）；先把 infer 跑全')
    assert all(i['inout'] == 1 for i in insts), 'val 应当全是 inout==1'
    return insts


def subset_stats(name, idx, g, A, parsed, frames, extra, boot_b):
    """idx: 该子集的人下标。g: n 贪心 L2；A: n×K 采样 L2；parsed: n×K bool；frames: n 帧号；extra: 其它逐人数组的 dict。"""
    g, A, parsed, fr = g[idx], A[idx], parsed[idx], frames[idx]
    best = {k: A[:, :k].min(1) for k in KS}
    avg = {k: A[:, :k].mean(1) for k in KS}
    Ap = np.where(parsed, A, np.inf)
    best16_parsed = np.where(parsed.any(1), Ap.min(1), A.min(1))
    out = {'n_people': int(len(idx)), 'n_frames': int(len(np.unique(fr))), 'greedy_l2': float(g.mean()),
           'best_of_k': {str(k): float(best[k].mean()) for k in KS}, 'avg_of_k': {str(k): float(avg[k].mean()) for k in KS},
           'R_k_point': {str(k): float((g.mean() - best[k].mean()) / g.mean()) for k in KS},
           'R16': boot_ratio(g, best[16], fr, boot_b), 'R16_parsed_only': boot_ratio(g, best16_parsed, fr, boot_b),
           'n_no_parsed_sample': int((~parsed.any(1)).sum()),
           'sample_parse_fail_rate': float(extra['s_parse_fail'][idx].mean()), 'sample_pred_out_rate': float(extra['s_pred_out'][idx].mean()),
           'auc_greedy': float(extra['auc_greedy'][idx].mean()), 'auc_k16_kde': float(extra['auc_k'][idx].mean())}
    # 前缀嵌套单调（§5.6）：逐人检查 best-of-2 ≥ 4 ≥ 8 ≥ 16
    mono = all((best[a] >= best[b] - 1e-12).all() for a, b in zip(KS[:-1], KS[1:]))
    if not mono:
        sys.exit(f'[{name}] best-of-k 对 k 不单调（§5.6），前缀嵌套被破坏，先修实现，不读数')
    # 离散度
    med_pw = [v for v in extra['med_pairwise'][idx] if v is not None]
    ncl = extra['n_clusters'][idx]
    out['dispersion'] = {'median_of_person_median_pairwise_dist': float(np.median(med_pw)) if med_pw else None,
                         'n_people_with_pairwise': len(med_pw), 'dbscan_eps': DB_EPS, 'dbscan_min_samples': DB_MIN,
                         'multimodal_rate': float((ncl >= 2).mean()), 'n_clusters_hist': {str(int(k)): int((ncl == k).sum()) for k in np.unique(ncl)}}
    out['aggregators'] = {n: {'avg_l2': float(extra[n + '_l2'][idx].mean()), 'auc': float(extra[n + '_auc'][idx].mean())}
                          for n in ('medoid', 'mean_shift')}
    out['aggregators']['greedy'] = {'avg_l2': float(g.mean()), 'auc': out['auc_greedy']}
    return out


def cmd_r16(a):
    t0 = time.time()
    print('[r16] 读数前检查')
    check_meta(a.preds, a.temperature)
    rows = E.load_jsonl(a.data)
    split = SPLIT.load_split(a.split_dir, a.data)
    print(f"  split OK  val_sel {len(split['val_sel'])} 人 / val_gate {len(split['val_gate'])} 人（sha256 与 split.json 一致）")
    insts = build(rows, load_preds(a.preds))
    pos = {r['id']: n for n, r in enumerate(rows)}
    frames = np.array([r['frame_idx'] for r in rows])

    metrics, per = E.score_instances(insts, 3.0, list(KS), n_proc=a.n_proc)  # 逐人 L2 / AUC / 采样状态
    g = np.array([p['avg_l2'] for p in per])
    A = np.array([p['s_avg_l2'] for p in per])
    parsed = np.array([[s in OK_STATUS for s in p['s_status']] for p in per])
    extra = {'auc_greedy': np.array([p['auc_greedy'] for p in per]), 'auc_k': np.array([p['auc'] for p in per]),
             's_parse_fail': np.array([[s == 'parse_fail' for s in p['s_status']] for p in per]).mean(1),
             's_pred_out': np.array([[s == 'pred_out' for s in p['s_status']] for p in per]).mean(1)}
    # 离散度 + 聚合器（只用成功解析的采样点）
    extra['med_pairwise'] = np.empty(len(insts), dtype=object)
    extra['n_clusters'] = np.zeros(len(insts), int)
    agg_pts = {'medoid': [], 'mean_shift': []}
    for n, i in enumerate(insts):
        P = parsed_points(i['samples'])
        extra['med_pairwise'][n] = median_pairwise(P)
        extra['n_clusters'][n] = n_clusters(P)
        md, ms = aggregate_points(i['samples'])
        agg_pts['medoid'].append(md)
        agg_pts['mean_shift'].append(ms)
    for name, pts in agg_pts.items():  # 聚合点当作单点预测，走同一套 L2 / AUC（K=0 路径）
        fake = []
        for i, pt in zip(insts, pts):
            f = dict(i)
            f.update(samples=[], g_pt=pt, g_status='ok', heatmap=None)
            fake.append(f)
        _, pp = E.score_instances(fake, 3.0, [], n_proc=a.n_proc)
        extra[name + '_l2'] = np.array([p['avg_l2'] for p in pp])
        extra[name + '_auc'] = np.array([p['auc'] for p in pp])

    subsets = {'val_gate': np.array([pos[i] for i in split['val_gate']]), 'val_sel': np.array([pos[i] for i in split['val_sel']]),
               'all': np.arange(len(rows))}
    res = {}
    for name, idx in subsets.items():
        res[name] = subset_stats(name, idx, g, A, parsed, frames, extra, a.boot)
    out = {'meta': {'label': a.label, 'preds': [os.path.abspath(p) for p in a.preds], 'expect_temperature': a.temperature,
                    'infer_meta': [json.load(open(p + '.meta.json')) for p in a.preds],
                    'prereg': 'prereg/CP1_CP2.prereg.md', 'prereg_sha256': sha256_file(f'{G}/prereg/CP1_CP2.prereg.md'),
                    'eval_gaze_sha256': sha256_file(os.path.join(HERE, 'eval_gaze.py')), 'cp1_r16_sha256': sha256_file(os.path.abspath(__file__)),
                    'split_dir': os.path.abspath(a.split_dir), 'data_sha256': sha256_file(a.data),
                    'bootstrap': {'B': a.boot, 'seed': BOOT_SEED, 'cluster': 'frame_idx'}, 'K': K_REQ, 'seconds': round(time.time() - t0, 1)},
           'subsets': res}
    if not a.no_verdict:
        gate = res['val_gate']
        out['verdict'] = verdict(gate['R16'], res['all']['R16']['point'], gate['R16_parsed_only']['point'])
    print_report(out)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, 'w'), indent=1, ensure_ascii=False)
        print('->', a.out)


def print_report(out):
    print(f"\n== R16  label={out['meta']['label']}  K={out['meta']['K']}  bootstrap B={out['meta']['bootstrap']['B']} (按帧)")
    for name, s in out['subsets'].items():
        r, rp = s['R16'], s['R16_parsed_only']
        print(f"-- {name}: {s['n_people']} 人 / {s['n_frames']} 帧")
        print(f"   greedy L2 {s['greedy_l2']:.4f}   best-of-{{2,4,8,16}} " + ' '.join(f"{s['best_of_k'][str(k)]:.4f}" for k in KS)
              + "   avg-of " + ' '.join(f"{s['avg_of_k'][str(k)]:.4f}" for k in KS))
        print(f"   R16 {r['point']:.4f}  [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]   R_k(2,4,8) " + ' '.join(f"{s['R_k_point'][str(k)]:.4f}" for k in KS[:3])
              + f"   仅成功解析 {rp['point']:.4f}  [{rp['ci95'][0]:.4f}, {rp['ci95'][1]:.4f}]（{s['n_no_parsed_sample']} 人无成功采样）")
        d = s['dispersion']
        print(f"   采样解析失败 {s['sample_parse_fail_rate']:.4f}  pred_out {s['sample_pred_out_rate']:.4f}   AUC greedy {s['auc_greedy']:.4f}  K16-KDE {s['auc_k16_kde']:.4f}")
        print(f"   离散度 两两距离中位数 {d['median_of_person_median_pairwise_dist']}  多峰占比 {d['multimodal_rate']:.4f}   簇数分布 {d['n_clusters_hist']}")
        print('   聚合器 ' + '  '.join(f"{n}: L2 {v['avg_l2']:.4f} AUC {v['auc']:.4f}" for n, v in s['aggregators'].items()))
    if 'verdict' in out:
        v = out['verdict']
        print(f"\n== CP1 判定（val_gate）: {v['verdict']}")
        for x in v['reasons']:
            print('   - ' + x)
        if v.get('point_below_thr'):
            print('   - 注意：val_gate 点估计 < 阈值，同时满足 §2「不通过」条件；边缘与不通过同时成立，交用户裁决')


# ------------------------------------------------------------------------------------------------------------------
# select：只用 val_sel 的 greedy 挑 checkpoint
# ------------------------------------------------------------------------------------------------------------------
def cmd_select(a):
    rows = {r['id']: r for r in E.load_jsonl(a.data)}
    split = SPLIT.load_split(a.split_dir, a.data)
    ids = split['val_sel']
    table = []
    for item in a.preds:
        label, _, path = item.partition('=')
        if not path:
            sys.exit(f'--preds 要写成 label=path，收到 {item!r}')
        recs = load_preds([path])
        miss = [i for i in ids if i not in recs]
        if miss:
            sys.exit(f'{label}: val_sel 里 {len(miss)}/{len(ids)} 人没有预测（例 {miss[:2]}）')
        l2, fail = [], 0
        for i in ids:
            pt, st = E.completion_to_point(recs[i]['greedy']['text'])  # 只读 greedy，不碰 samples
            fail += st in ('parse_fail', 'pred_out')
            r = rows[i]
            l2.append(E.l2_stats(E._sp(pt), r['gt_xs_norm'], r['gt_ys_norm'])[0])
        table.append((label, float(np.mean(l2)), fail / len(ids)))
    print(f'== val_sel（{len(ids)} 人）greedy Avg L2；只用于挑 checkpoint，不含 val_gate、不含任何采样指标')
    for label, v, f in table:
        print(f'   {label:12s} {v:.4f}   中心回退率 {f:.4f}')
    best = min(table, key=lambda t: t[1])
    print(f'-> 选 {best[0]}（val_sel greedy Avg L2 最低 {best[1]:.4f}）')
    if a.out:
        json.dump({'split_dir': os.path.abspath(a.split_dir), 'n_val_sel': len(ids), 'table': [{'label': l, 'val_sel_greedy_l2': v, 'center_fallback_rate': f} for l, v, f in table],
                   'selected': best[0]}, open(a.out, 'w'), indent=1)
        print('->', a.out)


# ------------------------------------------------------------------------------------------------------------------
# selftest
# ------------------------------------------------------------------------------------------------------------------
class _Check:
    def __init__(self):
        self.fails = []

    def ok(self, cond, msg):
        print(('  PASS ' if cond else '  FAIL ') + msg)
        if not cond:
            self.fails.append(msg)


def cmd_selftest(a):
    C = _Check()
    print('[1] R16 与 bootstrap')
    g, b = np.array([0.4, 0.2]), np.array([0.1, 0.2])  # 均值 0.3 / 0.15 -> R = 0.5
    r = boot_ratio(g, b, [0, 1], B=200)
    C.ok(abs(r['point'] - 0.5) < 1e-12, '手算 R = (0.3−0.15)/0.3 = 0.5')
    r = boot_ratio(np.full(40, 0.4), np.full(40, 0.2), np.repeat(np.arange(10), 4), B=500)
    C.ok(r['ci95'] == [0.5, 0.5] and abs(r['point'] - 0.5) < 1e-12, '逐人相同 -> 区间塌缩到点估计')
    r1, r2 = boot_ratio(np.arange(1, 31) / 30, np.arange(1, 31) / 60, np.arange(30) // 3, B=300), boot_ratio(np.arange(1, 31) / 30, np.arange(1, 31) / 60, np.arange(30) // 3, B=300)
    C.ok(r1 == r2, '同 seed 两次区间逐位相同')
    # 帧内相关：同一帧 3 人同值，按帧重采样的区间必须比按人重采样宽
    rng = np.random.default_rng(1)
    fv = rng.uniform(0.1, 0.9, 200)
    gg, bb = np.repeat(fv, 3), np.repeat(fv * rng.uniform(0.2, 0.8, 200), 3)
    wf = np.diff(boot_ratio(gg, bb, np.repeat(np.arange(200), 3))['ci95'])[0]
    wp = np.diff(boot_ratio(gg, bb, np.arange(600))['ci95'])[0]
    C.ok(wf > wp, f'按帧重采样的区间 {wf:.4f} 宽于按人 {wp:.4f}（帧内相关被计入）')

    print('[2] medoid / mean-shift / DBSCAN')
    P = np.array([[0, 0], [0.1, 0], [0.9, 0]], float)
    C.ok(np.allclose(medoid(P), [0.1, 0]), 'medoid 手算 = (0.1, 0)')
    C.ok(aggregate_points([(None, 'parse_fail'), (None, 'pred_out')]) == (CENTER, CENTER), '无成功解析点 -> 画面中心')
    Q = np.vstack([np.full((6, 2), 0.30), [[0.95, 0.95]]])
    C.ok(np.allclose(mean_shift(Q, medoid(Q)), [0.30, 0.30], atol=1e-6), 'mean-shift：离群点不拉动密集簇')
    two = np.array([[0.3, 0.3]] * 3 + [[0.31, 0.3]] + [[0.8, 0.8]] * 3 + [[0.8, 0.81]], float)
    C.ok(n_clusters(two) == 2 and n_clusters(np.array([[0.3, 0.3]] * 8)) == 1 and n_clusters(two[:2]) == 0, 'DBSCAN：两簇=2、单簇=1、<3 点=0')
    C.ok(abs(median_pairwise(np.array([[0, 0], [0.3, 0], [0.3, 0.4]], float)) - 0.4) < 1e-12 and median_pairwise(P[:1]) is None, '两两距离中位数手算 (0.3, 0.4, 0.5 -> 0.4)')

    print('[3] 判定逻辑（§2）')
    mk = lambda pt, lo: {'point': pt, 'ci95': [lo, pt + 0.05]}
    C.ok(verdict(mk(0.35, 0.31), 0.36, 0.34)['verdict'] == '通过', '点估计与下界都 ≥ 30%，三处同侧 -> 通过')
    C.ok(verdict(mk(0.35, 0.28), 0.36, 0.34)['verdict'].startswith('边缘'), '点估计 ≥ 30% 但下界 < 30% -> 边缘')
    C.ok(verdict(mk(0.35, 0.31), 0.28, 0.34)['verdict'].startswith('边缘'), 'val_gate 与全 val 分居两侧 -> 边缘')
    C.ok(verdict(mk(0.35, 0.31), 0.36, 0.28)['verdict'].startswith('边缘'), '主分析与仅成功解析分居两侧 -> 边缘')
    C.ok(verdict(mk(0.20, 0.15), 0.22, 0.21)['verdict'] == '不通过', '三处都 < 30% -> 不通过')
    v = verdict(mk(0.28, 0.2), 0.33, 0.27)
    C.ok(v['verdict'].startswith('边缘') and v['point_below_thr'], 'gate < 30% 而全 val ≥ 30%：边缘与不通过同时成立 -> 标边缘并注明')

    print('[4] 端到端：真实 val 的行 + 合成采样（只在临时目录，不落进 results/）')
    rows = E.load_jsonl(f'{G}/data/vqa/gazefollow_val.jsonl')
    rng = np.random.default_rng(0)
    split_dir = f'{G}/data/vqa/cp2_split'
    pt = lambda x, y: f'<dir>0</dir><pt>{int(round(x * 1000))},{int(round(y * 1000))}</pt><io>in</io>'
    with tempfile.TemporaryDirectory() as td:
        for tag, mode in (('good', 'one_exact'), ('noise', 'noise')):
            p = os.path.join(td, tag + '.jsonl')
            with open(p, 'w') as f:
                for r in rows:
                    gx, gy = r['gt_x_norm'], r['gt_y_norm']
                    gr = {'text': pt(gx + 0.1, gy), 'p_in_io': 1.0, 'p_in_dir': 1.0}  # greedy 恒偏 0.1
                    smp = []
                    for k in range(K_REQ):
                        if mode == 'one_exact' and k == 5:
                            smp.append({'text': pt(gx, gy), 'p_in_io': 1.0, 'p_in_dir': 1.0})
                        elif k == 3:
                            smp.append({'text': 'garbage', 'p_in_io': None, 'p_in_dir': None})  # 解析失败 -> 中心回退
                        else:
                            q = np.clip(np.array([gx, gy]) + rng.normal(0, 0.15, 2), 0, 1)
                            smp.append({'text': pt(q[0], q[1]), 'p_in_io': 1.0, 'p_in_dir': 1.0})
                    f.write(json.dumps({'id': r['id'], 'greedy': gr, 'samples': smp}) + '\n')
            json.dump({'num_samples': 16, 'temperature': 1.0, 'top_p': 1.0, 'adapter': 'synthetic', 'seed': 0}, open(p + '.meta.json', 'w'))
            o = os.path.join(td, tag + '.json')
            ns = argparse.Namespace(data=f'{G}/data/vqa/gazefollow_val.jsonl', preds=[p], split_dir=split_dir, out=o, temperature=1.0,
                                    label='synthetic-' + tag, no_verdict=False, boot=200, n_proc=a.n_proc)
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                cmd_r16(ns)
            res = json.load(open(o))
            gate = res['subsets']['val_gate']
            if mode == 'one_exact':
                C.ok(gate['best_of_k']['16'] < 0.002 and gate['R16']['point'] > 0.97, f"含精确点的 best-of-16 ≈ 量化误差（{gate['best_of_k']['16']:.5f}），R16 {gate['R16']['point']:.3f}")
                C.ok(abs(gate['greedy_l2'] - 0.1) < 0.002, f"greedy 恒偏 0.1 -> L2 {gate['greedy_l2']:.4f}")
                C.ok(res['verdict']['verdict'] == '通过', '合成「必过」用例走到 通过')
            else:
                C.ok(0 < gate['R16']['point'] < 1 and gate['R16']['ci95'][0] < gate['R16']['point'] < gate['R16']['ci95'][1], f"噪声采样：R16 {gate['R16']['point']:.3f} 在区间内")
            C.ok(abs(gate['sample_parse_fail_rate'] - 1 / 16) < 1e-9, '每人恰 1/16 采样解析失败 -> 采样解析失败率 0.0625')
            C.ok(res['subsets']['val_gate']['n_people'] == 1675 and res['subsets']['val_sel']['n_people'] == 1705 and res['subsets']['all']['n_people'] == 3380, '三个子集人数 1675 / 1705 / 3380')
            C.ok(gate['R16_parsed_only']['point'] <= gate['R16']['point'] + 1e-12, '仅成功解析的 R16 ≤ 主分析（去掉了中心回退的机械增益）')
            C.ok(all(0 < v['auc'] < 1 for v in gate['aggregators'].values()), '聚合器 AUC 有效')
        # meta 拒绝路径
        json.dump({'num_samples': 8, 'temperature': 1.0, 'top_p': 1.0}, open(os.path.join(td, 'good.jsonl.meta.json'), 'w'))
        try:
            check_meta([os.path.join(td, 'good.jsonl')], 1.0)
            C.ok(False, 'num_samples=8 应当被拒绝')
        except SystemExit:
            C.ok(True, 'meta 的 num_samples != 16 被拒绝（§5.5）')
        json.dump({'num_samples': 16, 'temperature': 0.7, 'top_p': 1.0}, open(os.path.join(td, 'good.jsonl.meta.json'), 'w'))
        try:
            check_meta([os.path.join(td, 'good.jsonl')], 1.0)
            C.ok(False, 'temperature 不符应当被拒绝')
        except SystemExit:
            C.ok(True, 'meta 的 temperature 与声称值不符被拒绝（§5.5）')
    print('\nselftest ' + ('PASS' if not C.fails else 'FAIL: ' + '; '.join(C.fails)))
    sys.exit(1 if C.fails else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('r16', cmd_r16), ('select', cmd_select), ('selftest', cmd_selftest)):
        s = sp.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument('--data', default=f'{G}/data/vqa/gazefollow_val.jsonl')
        s.add_argument('--split-dir', default=f'{G}/data/vqa/cp2_split')
        s.add_argument('--out', default=None)
        s.add_argument('--n-proc', type=int, default=None, help='AUC 打分的进程数（缺省 min(32, cpu)）')
        if name == 'select':
            s.add_argument('--preds', nargs='+', required=True, help='label=path/to/preds.jsonl（greedy）')
        if name == 'r16':
            s.add_argument('--preds', nargs='+', required=True, help='K=16 采样的 preds.jsonl（多个会合并，按 id 后者覆盖前者）')
            s.add_argument('--temperature', type=float, default=1.0, help='声称的采样温度，与 preds.jsonl.meta.json 核对（§5.5）')
            s.add_argument('--label', default='sft')
            s.add_argument('--no-verdict', action='store_true', help='零样本对照等只作描述性报告的用法：不输出 CP1 判定')
            s.add_argument('--boot', type=int, default=BOOT_B)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
