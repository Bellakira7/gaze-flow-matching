"""C22 第 5 项的配对比较：flow 线标准 VAT（c20_ours_h0，epoch 3，σ=6）对官方 Gaze-LLE 与 OmniGF（自评）。

三行都走冻结的 eval_gaze 打分（Gaze-LLE 用 maintable.score_gazelle，OmniGF 用 omnigf_score.replay_vat，我们用 results/vat_standard/test_per_instance.npz），
按片段（seq，131 个）cluster bootstrap 2000 次、seed 0（与 maintable / omnigf_table 相同）：
  L2   ：画面内，相对改善 (base − ours) / base（cp1_r16.boot_ratio，正 = 我们更好）
  AUC  ：画面内，差 ours − base
  AP   ：全部样本（含画面外），差 ours − base；每次重采样整段片段后重算 average_precision
python -I gazelle_online/vat_standard_compare.py   → results/vat_standard/compare.{json,md}
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

G = Path(str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(G / 'tools'))
import grpo_val_report as R  # noqa: E402
CP = R.load_module()           # 先加载 cp1_r16（它注册 sys.modules['eval_gaze']），之后的 import 必须拿到同一个对象
import eval_gaze as E  # noqa: E402
import maintable as MT  # noqa: E402
import omnigf_score as OS  # noqa: E402
assert E is CP.E and OS.E is E
OUT = G / 'results/vat_standard'


def per_rows(rows, insts=None, per=None):
    return dict(l2=np.array([p.get('avg_l2', np.nan) for p in per]), auc=np.array([p.get('auc', np.nan) for p in per]),
                p_in=np.array([float(i['p_in']['io']) for i in insts]) if insts is not None else None)


def boot_diff_mean(a, b, inv, F, BI):
    """mean(b) − mean(a)，按簇重采样（a = base，b = ours）。"""
    cnt = np.bincount(inv, minlength=F).astype(float)
    ma = np.bincount(inv, weights=a, minlength=F)[BI].sum(1) / cnt[BI].sum(1)
    mb = np.bincount(inv, weights=b, minlength=F)[BI].sum(1) / cnt[BI].sum(1)
    return float(b.mean() - a.mean()), [float(np.percentile(mb - ma, 2.5)), float(np.percentile(mb - ma, 97.5))]


def boot_diff_ap(y, sa, sb, inv, F, BI):
    members = [np.flatnonzero(inv == f) for f in range(F)]
    d = []
    for row in BI:
        idx = np.concatenate([members[f] for f in row])
        d.append(E.ap_score(y[idx], sb[idx]) - E.ap_score(y[idx], sa[idx]))
    return float(E.ap_score(y, sb) - E.ap_score(y, sa)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    rows = E.load_jsonl(MT.DATA['vat'])
    y = np.array([int(r['inout']) == 1 for r in rows])
    ours = np.load(OUT / 'test_per_instance.npz')
    assert list(ours['id']) == [r['id'] for r in rows] and np.array_equal(ours['inout'], y)
    arms = {'ours': dict(l2=ours['l2'], auc=ours['auc'], p_in=ours['p_in'])}
    metrics = {'ours': json.load(open(OUT / 'test_scored.json'))['metrics']}
    _, met, per = MT.score_gazelle(E, 'vat')
    # score_gazelle 不返回 insts；p_in 直接从官方 dump 取（与它内部 _replay_instances 的输入相同）
    dump = np.load(G / 'results/vat_official/official_inout_vitl14.npz')
    arms['gazelle'] = dict(per_rows(rows, None, per), p_in=np.asarray(dump['inout_pred'], float))
    metrics['gazelle'] = met
    insts = OS.replay_vat(sorted(glob.glob(str(G / 'results/omnigf/vat_test_dump_shard*.npz'))), rows)
    m, per = E.score_instances(insts, MT.SIGMA, ())
    (metrics['omnigf'],) = m.values()
    arms['omnigf'] = per_rows(rows, insts, per)
    for k, a in arms.items():  # 自检：逐实例数组重算的汇总数应与 eval_gaze 的汇总一致
        assert abs(np.nanmean(a['l2'][y]) - metrics[k]['l2']) < 1e-9, (k, 'l2')
        assert abs(E.ap_score(y, a['p_in']) - metrics[k]['ap_inout']) < 1e-9, (k, 'ap')
    seq = np.array([r['seq'] for r in rows])
    rng = np.random.default_rng(CP.BOOT_SEED)
    _, inv_all = np.unique(seq, return_inverse=True)
    F_all = int(inv_all.max()) + 1
    BI_all = rng.integers(0, F_all, (CP.BOOT_B, F_all))
    inv_in = np.unique(seq[y], return_inverse=True)[1]
    F_in = int(inv_in.max()) + 1
    BI_in = np.random.default_rng(CP.BOOT_SEED).integers(0, F_in, (CP.BOOT_B, F_in))
    comps = []
    for base in ('gazelle', 'omnigf'):
        b, o = arms[base], arms['ours']
        c = dict(baseline=base, rel_l2_improvement=CP.boot_ratio(b['l2'][y], o['l2'][y], seq[y]))
        pt, ci = boot_diff_mean(b['auc'][y], o['auc'][y], inv_in, F_in, BI_in)
        c['auc_diff'] = dict(point=pt, ci95=ci)
        pt, ci = boot_diff_ap(y, b['p_in'], o['p_in'], inv_all, F_all, BI_all)
        c['ap_diff'] = dict(point=pt, ci95=ci)
        comps.append(c)
        print(base, json.dumps(c, ensure_ascii=False), flush=True)
    keys = ('l2', 'auc', 'ap_inout', 'n', 'n_in_frame')
    out = dict(id='C22-5-VAT-COMPARE', metrics={k: {kk: v.get(kk) for kk in keys} for k, v in metrics.items()}, comparisons=comps,
               bootstrap=dict(cluster='seq (clip)', n_clusters_all=F_all, n_clusters_in=F_in, resamples=CP.BOOT_B, seed=CP.BOOT_SEED))
    (OUT / 'compare.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    ci = lambda d: f"{d['point']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"  # noqa: E731
    L = ['| 行 | 画面内 L2 | AUC | AP in/out |', '|---|---:|---:|---:|']
    for k, lab in (('ours', 'Ours（flowET_d6 → VAT，H=0）'), ('gazelle', 'Gaze-LLE ViT-L（官方，回放）'), ('omnigf', 'OmniGF（官方权重，自评）')):
        mm = metrics[k]
        L.append(f'| {lab} | {mm["l2"]:.4f} | {mm["auc"]:.4f} | {mm["ap_inout"]:.4f} |')
    L += ['', '| 对照 | L2 相对改善（正 = 我们更好） | AUC 差 | AP 差 |', '|---|---|---|---|']
    for c in comps:
        L.append(f'| 对 {c["baseline"]} | {ci(c["rel_l2_improvement"])} | {ci(c["auc_diff"])} | {ci(c["ap_diff"])} |')
    L.append(f'\n> 按片段 cluster bootstrap {CP.BOOT_B} 次、seed {CP.BOOT_SEED}；未作多重比较校正。L2 / AUC 只算画面内（{F_in} 个片段），AP 用全部样本（{F_all} 个片段）。'
             '\n> 口径差：Gaze-LLE / OmniGF 的点 = 64×64 热图 argmax，我们是连续坐标（样本聚类 hyps[0]）；我们的 AUC 热图 = 10000 样本计数 + σ=6 格平滑（val 选定）。')
    (OUT / 'compare.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
