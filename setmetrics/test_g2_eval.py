"""G2 适配器与判定逻辑自检。python -I -u test_g2_eval.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_eval as E  # noqa: E402
import metrics as M  # noqa: E402

G_ = E.G_
ok = []


def check(name, cond, info=''):
    ok.append(bool(cond))
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{info}]' if info else ''), flush=True)


ctx = E.load_context()
inst, fidx, hw, subsets, F = ctx
N = len(inst)
sep = subsets['M_sep']

# 1. pmf_from_samples
S = np.random.default_rng(0).random((1000, 2))
P = E.pmf_from_samples(S, 4)
check('pmf_from_samples sums to 1', abs(P.sum() - 1) < 1e-12 and (P > 0).all())
nll_u = M.nll(P, np.random.default_rng(1).random((10, 2)))
check('uniform samples -> NLL close to ln 4096', abs(nll_u - np.log(4096)) < 0.3, f'{nll_u:.3f}')

# 2. 格式校验
good = {'hyps': np.random.default_rng(0).random((N, 5, 2)), 'w': np.tile(np.array([.4, .3, .15, .1, .05]), (N, 1)), 'samples': np.random.default_rng(1).random((N, 10000, 2))}
E.validate(good, N)
bad_cases = {
    'K=4': {**good, 'hyps': good['hyps'][:, :4], 'w': np.tile(np.array([.4, .3, .2, .1]), (N, 1))},
    'w not sum 1': {**good, 'w': good['w'] * 0.9},
    'w ascending': {**good, 'w': good['w'][:, ::-1]},
    'samples out of range': {**good, 'samples': good['samples'] * 1.5},
    'samples count': {**good, 'samples': good['samples'][:, :5000]},
}
for nm, p in bad_cases.items():
    try:
        E.validate(p, N); check(f'validate rejects {nm}', False)
    except AssertionError:
        check(f'validate rejects {nm}', True)

# 3. 阈值算术与协议一致（以官方 Gaze-LLE 的 S1 数字为 baseline）
base = E.official_bundle()


def mk(cov_fn=None, nll_delta=None, auc_delta=0.0, avg_delta=0.0):
    m = {k: np.array(v, copy=True) for k, v in base.items()}
    c = base['cov0.1@5'].astype(float)
    m['cov0.1@5'] = cov_fn(c) if cov_fn else c
    d = np.zeros(N) if nll_delta is None else nll_delta
    m['nll_variants'] = (base['nll_variants'][:, [0]] - (d if np.ndim(d) == 0 else d[:, None]))
    m['nll_variant_names'] = np.array(['cf'])
    m['auc'] = base['auc'] + auc_delta
    m['avg_l2_fix'] = base['avg_l2_fix'] + avg_delta
    return m


R0 = E.gate(mk(), {'B1': base}, required=('B1',), ctx=ctx)
a, s = R0['subsets']['all'], R0['subsets']['M_sep']
check('baseline cov values match prereg (0.852125 / 0.743701)', abs(a['P1']['cov_base'] - 0.852125) < 1e-5 and abs(s['P1']['cov_base'] - 0.743701) < 1e-5)
check('cov thresholds match prereg (0.8669 / 0.7693)', abs(a['P1']['cov_base'] + 0.1 * (1 - a['P1']['cov_base']) - 0.8669) < 1e-4 and abs(s['P1']['cov_base'] + 0.1 * (1 - s['P1']['cov_base']) - 0.7693) < 1e-4)
check('NLL thresholds match prereg v2 (0.0337 / 0.0598)', abs(a['P2']['threshold'] - 0.0337) < 5e-5 and abs(s['P2']['threshold'] - 0.0598) < 5e-5, f"{a['P2']['threshold']:.5f}/{s['P2']['threshold']:.5f}")
check('official sample-pipeline NLL base match prereg v2 (5.774448 / 6.470558)', abs(a['P2']['nll_base'] - 5.774448) < 1e-5 and abs(s['P2']['nll_base'] - 6.470558) < 1e-5, f"{a['P2']['nll_base']:.6f}/{s['P2']['nll_base']:.6f}")
check('strongest NLL baseline variant is raw', a['P2']['baseline'].endswith('raw') and s['P2']['baseline'].endswith('raw'))
check('copy of baseline -> FAIL with zero improvement', R0['verdict'] == 'FAIL' and abs(a['P1']['delta1_miss_reduction']) < 1e-12 and abs(a['P2']['improvement']) < 1e-12)

# 4. 判定分支
R = E.gate(mk(lambda c: c + (1 - c) * 0.15, 0.12), {'B1': base}, required=('B1',), ctx=ctx)
check('improved everywhere, std unchanged -> PASS', R['verdict'] == 'PASS' and R['A'] and R['B'] and R['C_ok'], f"d1={R['subsets']['all']['P1']['delta1_miss_reduction']:.3f}")
R = E.gate(mk(lambda c: c + (1 - c) * 0.15 * (~sep), np.where(sep, 0.0, 0.12)), {'B1': base}, required=('B1',), ctx=ctx)
check('improved only outside M_sep -> FAIL (B)', R['verdict'] == 'FAIL' and R['A'] and not R['B'])
R = E.gate(mk(lambda c: c + (1 - c) * 0.15, 0.12, auc_delta=-0.01), {'B1': base}, required=('B1',), ctx=ctx)
check('AUC -0.010 -> FAIL (C)', R['verdict'] == 'FAIL' and not R['C_ok'] and R['A'] and R['B'])
R = E.gate(mk(lambda c: c + (1 - c) * 0.15, 0.12, auc_delta=-0.004, avg_delta=0.004), {'B1': base}, required=('B1',), ctx=ctx)
check('std degradation within 0.005 -> still PASS', R['verdict'] == 'PASS')
R = E.gate(mk(lambda c: c + (1 - c) * 0.15, 0.12, avg_delta=0.006), {'B1': base}, required=('B1',), ctx=ctx)
check('AvgL2 +0.006 -> FAIL (C)', R['verdict'] == 'FAIL' and not R['C_ok'])
R = E.gate(mk(lambda c: c + (1 - c) * 0.15, 0.012), {'B1': base}, required=('B1',), ctx=ctx)
check('NLL gain below threshold (0.012 < 0.0337) -> FAIL', R['verdict'] == 'FAIL' and not R['subsets']['all']['P2']['point_pass'])

# 置信区间分支：逐帧符号噪声，使点估计达标但区间跨 0
nf = len(np.unique(fidx))
sg = np.where(np.random.default_rng(0).permutation(nf) < round(nf * 0.5094), 1.0, -1.0)
noisy = mk(lambda c: c + sg[fidx] * 0.9, 0.12)
R = E.gate(noisy, {'B1': base}, required=('B1',), ctx=ctx)
a = R['subsets']['all']
check('CI branch exercised: point d1>=0.10 but CI lower <= 0', a['P1']['point_pass'] and a['P1']['ci95'][0] <= 0, f"d1={a['P1']['delta1_miss_reduction']:.3f} ci={[round(x, 3) for x in a['P1']['ci95']]}")
check('CI crossing 0 on all -> FAIL (A)', R['verdict'] == 'FAIL' and not R['A'])

R = E.gate(mk(), {'B1': base}, ctx=ctx)
check('missing baselines -> INCOMPLETE', R['verdict'] == 'INCOMPLETE' and set(R['missing_baselines']) == {'B2', 'B3', 'B4', 'B5'})

# 5. 端到端空跑：用 S1 的热图当“方法”，核对适配器与 S1 一致
H = np.load(f'{G_}/results/repro_test/full.npz')['heatmaps'].astype(np.float32)
hy = np.zeros((N, 5, 2)); smp = np.zeros((N, 10000, 2))
for i in range(N):
    pk = M.nms_peaks(H[i], 5)
    hy[i] = M.CELLS[pk]
    smp[i] = M.sample_hyps(M.heat_to_pmf(H[i], 1.0), np.random.default_rng([777, i]), 10000)
pred = {'hyps': hy, 'w': np.tile(np.array([5, 4, 3, 2, 1]) / 15.0, (N, 1)), 'samples': smp}
bundle = E.evaluate_prediction(pred)
s1 = np.load(f'{G_}/results/setmetrics/s1_per_instance.npz')
check('round trip: cov0.10@5 equals S1 nms exactly', np.allclose(bundle['cov0.1@5'], s1['nms/cov0.1@5'], atol=1e-12), f"{bundle['cov0.1@5'].mean():.4f}")
check('round trip: minL2@5 equals S1 nms exactly', np.allclose(bundle['minL2@5'], s1['nms/minL2@5'], atol=1e-12))
check('round trip: AvgL2/MinL2 (point=hyps[0]=argmax) equal S1', abs(bundle['avg_l2_fix'].mean() - 0.09687) < 1e-3 and abs(bundle['min_l2_fix'].mean() - 0.03999) < 1e-3,
      f"{bundle['avg_l2_fix'].mean():.5f}/{bundle['min_l2_fix'].mean():.5f}")
d_nll = bundle['nll_variants'][:, 0].mean() - base['nll_variants'][:, 0].mean()
check('round trip: same-pipeline NLL within 0.02 of official bundle (raw)', abs(d_nll) < 0.02, f'diff {d_nll:+.4f} (sigma_s chosen {bundle["sigma_s_chosen"]})')
check('round trip: sample-pmf AUC within 0.003 of official 0.9584 (M=10000)', abs(bundle['auc'].mean() - 0.9584) < 0.003, f"{bundle['auc'].mean():.4f}")
Rr = E.gate(bundle, {'B1': base}, required=('B1',), ctx=ctx)
check('round trip gate runs; dry-run result', Rr['verdict'] in ('PASS', 'FAIL'), f"verdict {Rr['verdict']}, d1 all {Rr['subsets']['all']['P1']['delta1_miss_reduction']:+.3f}")

print(f'\n{sum(ok)}/{len(ok)} passed')
sys.exit(0 if all(ok) else 1)
