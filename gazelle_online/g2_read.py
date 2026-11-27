"""读 G2 判定（flow 为方法），gt 口径为判定、eye 口径并排照报（C6）。只调用冻结的 g2_eval.gate，不改它。
基线集合 = {B1(s0), B2(s0), B3 MDN, B4 MCL, B5 diff} ∪ {官方 ckpt} ∪ {B1/B2 lr3e-4 补充臂}（C2/C7，seed 1 不进）。
约束 C 四种口径并排（C3/C7）：B1 s0 argmax / hyps[0]，B1 lr3e-4 argmax / hyps[0]。
python -I g2_read.py [--method flow|flowET|flowET_s1]  ->  results/g2/gate_<method>_{gt,eye}.json + 打印
  --method flowET：眼部 token 变体（bundle_flowET_*），基线集合不变（flow 本身不进基线集合）。
  --method flowET_s1：flowET 的第二个训练 seed（bundle_flowET_s1_*，C14），同上。
  --method rayET / rayET_s1 / ray_nomarch / ray_noeye / flowET_d6：射线头主臂、seed 1、两个消融与容量对照（C15，bundle_<method>_*），同上。
  --method raytokET / raytokET_s1：v2 射线 token（C16），同上。
  --method eyefuse / eyehr（及 _s1）：眼部 token 变体 2（C17），同上。
  --method mixflowET / mixflowET_s1：集合原生混合 flow 头（C21，bundle_mixflowET_*，假设为原生口径），同上。
  --method flow_s1 / flowET_d6_s1 / flowET_d6_s2：预注册 flow 的 seed 1、主方法 d6 的 seed 1/2（C22，bundle_<method>_*），只作 seed 方差，同上。
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/setmetrics')
import g2_eval as E  # noqa: E402

R = f'{G}/results/g2'
SETS = {
    'gt': dict(B1='bundle_B1_b1_s0', B2='bundle_B2_b1_s0', OFFICIAL='../setmetrics/official_gazelle_vitl_bundle_v2',
               B1_lr3e4='bundle_B1_b1_dinov2l_sd_lr3e-4_s0_ep14_gt', B2_lr3e4='bundle_B2_b1_dinov2l_sd_lr3e-4_s0_ep14_gt',
               B3='bundle_mdn_gt', B4='bundle_mcl_gt', B5='bundle_diff_gt', method='bundle_{M}_gt',
               B1_h0='bundle_check_b1_s0', B1_lr3e4_h0='bundle_check_b1_lr3e-4_gt'),
    'eye': dict(B1='bundle_B1_b1_dinov2l_sd_s0_ep14_eye', B2='bundle_B2_b1_dinov2l_sd_s0_ep14_eye', OFFICIAL='bundle_B1_official_eyebox',
                B1_lr3e4='bundle_B1_b1_dinov2l_sd_lr3e-4_s0_ep14_eye', B2_lr3e4='bundle_B2_b1_dinov2l_sd_lr3e-4_s0_ep14_eye',
                B3='bundle_mdn_eye', B4='bundle_mcl_eye', B5='bundle_diff_eye', method='bundle_{M}_eye',
                B1_h0='bundle_check_b1_s0_eye', B1_lr3e4_h0='bundle_check_b1_lr3e-4_eye'),
}
MAIN = ('B1', 'B2', 'B3', 'B4', 'B5', 'OFFICIAL', 'B1_lr3e4', 'B2_lr3e4')


def load(p):
    return E._load(f'{R}/{p}.npz')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='flow', choices=['flow', 'flowET', 'flowET_s1', 'rayET', 'rayET_s1', 'ray_nomarch', 'ray_noeye', 'flowET_d6', 'raytokET', 'raytokET_s1', 'eyefuse', 'eyehr', 'eyefuse_s1', 'eyehr_s1', 'mixflowET', 'mixflowET_s1', 'flow_s1', 'flowET_d6_s1', 'flowET_d6_s2'])
    MN = ap.parse_args().method
    ctx = E.load_context()
    allres = {}
    for box, S in SETS.items():
        B = {k: load(v) for k, v in S.items() if k != 'method'}
        method = load(S['method'].replace('{M}', MN))
        base = {k: B[k] for k in MAIN}
        res = E.gate(method, base, ctx=ctx)
        # 约束 C：几种口径（只取 C 一项；额外基线不影响主判定，因为只读 'C'）
        cs = {}
        for name in ('B1', 'B1_h0', 'B1_lr3e4', 'B1_lr3e4_h0'):
            sb = dict(base)
            sb['B1x'] = B[name] if name not in base else base[name]
            cs[name] = E.gate(method, {**base, 'B1': sb['B1x']}, std_baseline='B1', ctx=ctx)['C']
        res['C_variants'] = cs
        # 每臂一览
        sel_all, sel_sep = ctx[3]['all'], ctx[3]['M_sep']
        arms = {}
        for k, b in {**B, MN.upper(): method}.items():
            V = np.asarray(b['nll_variants'], float)
            arms[k] = dict(cov_all=float(np.asarray(b['cov0.1@5'], float)[sel_all].mean()), cov_sep=float(np.asarray(b['cov0.1@5'], float)[sel_sep].mean()),
                           nll_all=float(V[sel_all].mean(0).min()), nll_sep=float(V[sel_sep].mean(0).min()),
                           auc=float(np.asarray(b['auc'], float).mean()), avg_l2=float(np.asarray(b['avg_l2_fix'], float).mean()), min_l2=float(np.asarray(b['min_l2_fix'], float).mean()))
        res['arms'] = arms
        allres[box] = res
        json.dump(res, open(f'{R}/gate_{MN}_{box}.json', 'w'), indent=1, ensure_ascii=False)

    for box, res in allres.items():
        print(f'\n################ 口径 {box}{"（判定）" if box == "gt" else "（并排，不进判定）"}  verdict={res["verdict"]}  A={res["A"]} B={res["B"]} C={res["C_ok"]}')
        print(f'{"arm":12s} {"cov全":>7s} {"cov Msep":>9s} {"NLL全":>7s} {"NLL Msep":>9s} {"AUC":>7s} {"AvgL2":>7s} {"MinL2":>7s}')
        for k, a in res['arms'].items():
            print(f'{k:12s} {a["cov_all"]:7.4f} {a["cov_sep"]:9.4f} {a["nll_all"]:7.4f} {a["nll_sep"]:9.4f} {a["auc"]:7.4f} {a["avg_l2"]:7.4f} {a["min_l2"]:7.4f}')
        for sn in ('all', 'M_sep'):
            s = res['subsets'][sn]
            p1, p2 = s['P1'], s['P2']
            print(f'-- {sn} (n={s["n"]})')
            print(f'   P1 最强 baseline={p1["baseline"]} cov {p1["cov_base"]:.4f} → 方法 {p1["cov_method"]:.4f}  漏检率相对降 {p1["delta1_miss_reduction"]:+.3f}（门槛 0.10）CI95 [{p1["ci95"][0]:+.3f},{p1["ci95"][1]:+.3f}]  {"PASS" if p1["point_pass"] else "fail"}')
            print(f'   P2 最强 baseline={p2["baseline"]} NLL {p2["nll_base"]:.4f} → 方法 {p2["nll_method"]:.4f}  改进 {p2["improvement"]:+.4f}（门槛 {p2["threshold"]:.4f}，LOO 参照 {p2["nll_ref"]:.3f}）CI95 [{p2["ci95"][0]:+.4f},{p2["ci95"][1]:+.4f}]  {"PASS" if p2["point_pass"] else "fail"}')
            print(f'   CI 下界均 >0: {s["ci_lower_positive"]}')
        print('-- 约束 C（全体；AUC 不低于 -0.005，Avg/Min L2 不高于 +0.005）')
        for name, c in res['C_variants'].items():
            print(f'   对 {name:12s} ' + '  '.join(f'{k} {v["method"]:.4f} vs {v["baseline"]:.4f} (Δ{v["delta"]:+.4f}{"" if v["ok"] else " ✗"})' for k, v in c.items()) + f'   → {"OK" if all(v["ok"] for v in c.values()) else "不满足"}')


if __name__ == '__main__':
    main()
