#!/usr/bin/env python
"""把 OmniGF 的逐样本热图落盘（tools/omnigf_dump.py）喂进我们冻结的打分（eval_gaze.py），出与主表同口径的行。

  /usr/local/bin/python -I tools/omnigf_score.py gf  --dump results/omnigf/gf_test_dump.npz  --out results/omnigf/gf_test_scored.json
  /usr/local/bin/python -I tools/omnigf_score.py vat --dump results/omnigf/vat_test_dump_shard*.npz --out results/omnigf/vat_test_scored.json
  /usr/local/bin/python -I tools/omnigf_score.py gf  --dump results/omnigf/smoke_dump/dump.npz --limit 64      # 冒烟：只比前 64 条

点 = OmniGF 官方 `spatial_argmax2d(normalize=True)` 的输出（他们的口径，复现论文 0.091/0.040 用的就是它）；热图原样进 AUC
（GF：双线性插值到原图分辨率，与官方 / 我们的协议一致；VAT：64×64 方窗）。**不**套 Gaze-LLE 的 idx/64 口径，因为那是 Gaze-LLE 的官方实现，
OmniGF 有自己的。AUC 的核对：官方 `GFTestAUC` 与 eval_gaze 的 `auc_gazefollow` 数值是否一致，见输出里的 `official_*` 对照（需传 --official-log）。
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

G = Path(str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(G / 'tools'))
import eval_gaze as E  # noqa: E402

SIGMA = 3.0


def replay_gf(dump, rows):
    d = np.load(dump)
    idx = np.array([int(re.search(r'(\d+)\.jpg$', p).group(1)) for p in d['path']])
    n = len(rows)
    assert sorted(idx.tolist()) == list(range(n)), f'dump 的行号必须恰好是 0..{n - 1}（官方 dataloader 不能丢样本）：{len(idx)} 行'
    o = np.argsort(idx)
    hms = d['heatmap'][o].astype(np.float32)
    pts = [(float(x), float(y)) for x, y in d['point'][o]]
    insts = E._replay_instances(rows, hms, pts, d['p_in'][o].tolist())
    return insts


def vat_index(index_path=None):
    """(H5 path, 帧内序号 hi) → 主表 jsonl 风格的 id，以及 id → 片段内持久的人 id。
    hi 取自 id 末尾的 `#<hi>`：dump 里的 pid 是官方 collator 重标后的帧内序号（见 omnigf_dump.py），不是 H5 里的 person_ids。
    旧 index.json（test，pid 列本来就是 hi）与 omnigf_build_vat_h5.py 的 index_*.json（pid 列 = 持久人 id）两种格式都吃。"""
    index, track = {}, {}
    for p, pid, i in json.load(open(index_path or G / 'data/omnigf_vat_test/index.json')):
        index[(p, int(i.rsplit('#', 1)[1]))] = i
        track[i] = int(pid)
    return index, track


def replay_vat(dumps, rows, partial=False, index_path=None, subset=False, branch='heatmap'):
    """VAT：dump 的键是 (H5 path, 帧内序号)；index 把它映射回主表 jsonl 的 id。每个分片一个 npz，合并后必须恰好覆盖 rows 里的每一个人。
    index_path：默认旧 test 的 data/omnigf_vat_test/index.json；val / 带轨迹的 test 用 omnigf_build_vat_h5.py 的 index_<split>.json。
    subset：dump 比 rows 多（如 val 全帧 dump、只打 stride 3 的行）时忽略多出来的，缺的仍然报错。
    branch：heatmap = 热图分支（argmax 点 + 热图 AUC + in/out 头）；text = 文本分支（官方 `gaze_pts_text`，画面外 / 未解析 = (0.5,0.5)，与官方
            `metric/test/dist_text` 同口径；AUC 用点的 KDE，p_in = 文本里的 inside/outside 0/1，没解析到 = 0.5）。"""
    index, _ = vat_index(index_path)
    row_of = {r['id']: k for k, r in enumerate(rows)}
    n = len(rows)
    hms, pts, pin, got = [None] * n, [None] * n, [0.5] * n, np.zeros(n, bool)
    for dp in dumps:
        d = np.load(dp)
        path, pid = d['path'], d['pid']
        if branch == 'text':
            assert 'point_text' in d.files, f'{dp}: 没有文本分支（旧 dump，用新版 omnigf_dump.py 重跑）'
            point, pi_ = d['point_text'], d['p_in_text']
            heat = None
        else:
            heat, point, pi_ = d['heatmap'], d['point'], d['p_in']
        for k in range(len(path)):
            iid = index[(str(path[k]), int(pid[k]))]
            if iid not in row_of:
                assert subset, f'{dp}: dump 里的 {iid} 不在 rows 里（要忽略多出来的行请加 subset=True）'
                continue
            r = row_of[iid]
            assert not got[r], f'{dp}: 重复样本 {rows[r]["id"]}'
            if branch == 'text':
                x, y = (float(point[k][0]), float(point[k][1]))
                nan = np.isnan(x) or np.isnan(y)
                pts[r] = (0.5, 0.5) if nan else (x, y)     # 官方：NaN → 0.5；画面外预测本来就写成 (0.5,0.5)
                pin[r] = 0.5 if nan else float(pi_[k])
                hms[r] = None
            else:
                hms[r], pts[r], pin[r] = heat[k].astype(np.float32), (float(point[k][0]), float(point[k][1])), float(pi_[k])
            got[r] = True
            assert int(d['gt_inout'][k]) == rows[r]['inout'], f'{rows[r]["id"]}: dump 的 GT inout 与 rows 不一致'
    if partial:   # 只给单个分片的验证用：只打分被覆盖的那部分人，指标不是全量 test 的数
        keep = [k for k in range(n) if got[k]]
        rows, hms, pts, pin = [rows[k] for k in keep], [hms[k] for k in keep], [pts[k] for k in keep], [pin[k] for k in keep]
    else:
        assert got.all(), f'dump 只覆盖 {int(got.sum())}/{n} 人，缺 {n - int(got.sum())}'
    insts = E._replay_instances(rows, hms, pts, pin)
    for i in insts:   # OmniGF 只有一个 in/out 头，两种 AP 取同一个数
        i['p_in']['dir'] = i['p_in']['io']
    return insts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ds', choices=('gf', 'vat'))
    ap.add_argument('--dump', required=True, nargs='+')
    ap.add_argument('--out')
    ap.add_argument('--partial', action='store_true', help='VAT：只打分 dump 覆盖到的人（单分片验证用，不是全量读数）')
    ap.add_argument('--limit', type=int, default=0, help='只取 jsonl 前 N 行（冒烟）')
    ap.add_argument('--official-log', help='官方 trainer.test 的日志（grep avg_dist/min_dist/auc 做对照）')
    a = ap.parse_args()
    rows = E.load_jsonl(str(G / f'data/vqa/{"gazefollow" if a.ds == "gf" else "vat"}_test.jsonl'))
    if a.limit:
        rows = rows[:a.limit]
    insts = replay_gf(a.dump[0], rows) if a.ds == 'gf' else replay_vat(a.dump, rows, a.partial)
    m, per = E.score_instances(insts, SIGMA, ())
    (metrics,) = m.values()
    out = {'dump': [str(Path(x).resolve()) for x in a.dump], 'n': len(insts), 'kde_sigma_cells': SIGMA, 'metrics': metrics}
    print({k: metrics.get(k) for k in ('auc', 'avg_l2', 'min_l2', 'l2', 'ap_inout', 'n_in_frame')}, f'n={len(insts)}')
    if a.official_log:
        txt = open(a.official_log, errors='ignore').read()
        mm = re.findall(r'avg_dist=([\d.]+)\s+min_dist=([\d.]+)\s+n_obs=(\d+)', txt)
        au = re.findall(r'metric/test/auc=([\d.]+)', txt)
        if mm and au:
            off = {'avg_dist': float(mm[-1][0]), 'min_dist': float(mm[-1][1]), 'n_obs': int(mm[-1][2]), 'auc': float(au[-1])}
            out['official'] = off
            out['diff'] = {'avg_l2': metrics['avg_l2'] - off['avg_dist'], 'min_l2': metrics['min_l2'] - off['min_dist'], 'auc': metrics['auc'] - off['auc']}
            print('official', off, '\ndiff', out['diff'])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
