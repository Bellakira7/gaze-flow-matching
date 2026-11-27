"""射线头的 hold-out 汇总与 C15 判定（只用训练 hold-out，不碰测试集）。
python -I compare_ray.py [--base flowET_s0_rayeval] [--names ray_eye_s0,ray_eye_nomarch_s0,ray_noeye_s0,ray_eye_s1,flow_eye_d6_s0]
读 experiments/holdout_eval/<name>_shard*of*.npz（eval_ray_holdout.py 输出，含 hyp0 / y / center / wh）。
  * 各臂：NLL（σ_s 取该臂最优）、cov10@5、cov05@5、点估计 L2；机制量 = 点估计相对"头→标注"射线的方向误差（度）与相对径向误差
    |‖p−c‖−‖y−c‖| / ‖y−c‖（像素空间，排除归一化 ‖y−c‖<0.05；与 tools/ray_decomp_analysis.py 同口径）。
  * 各臂 − base 的逐实例配对差，bootstrap（B=2000，按实例重采样，同 compare_holdout.py）95% CI。
  * C15 规则：主方法 = ray_eye_s0 若其 NLL 比 base 低 ≥ 0.02；机制预测成立 = ray_eye_s0 − base 的相对径向误差 CI 上界 < 0。
  * 对拍：base 若是 flowET 重评，其 nll/cov/cov05/l2 必须与已有 flow_eye_s0 的 shard 逐位相同。
"""
from pathlib import Path
import argparse
import glob

import numpy as np

G = str(Path(__file__).resolve().parents[1])
D = f'{G}/experiments/holdout_eval'
KEYS = ('nll', 'cov', 'cov05', 'l2')


def load(name, extra=True):
    fs = sorted(f for f in glob.glob(f'{D}/{name}_shard*of*.npz') if 'smoke' not in f)
    assert fs, f'没有 {name} 的 shard'
    P = [np.load(f) for f in fs]
    idx = np.concatenate([p['idx'] for p in P])
    o = np.argsort(idx)
    assert len(idx) == 3762 and len(set(idx.tolist())) == 3762, (name, len(idx))
    ks = KEYS + (('hyp0', 'y', 'center', 'wh', 'p_out') if extra else ())
    return {k: np.concatenate([p[k] for p in P])[o] for k in ks}


def mech(d):
    """-> 方向误差（度）[N]、相对径向误差 [N]、有效掩码 [N]。"""
    wh = d['wh']
    vy, vp = (d['y'] - d['center']) * wh, (d['hyp0'] - d['center']) * wh
    ok = np.linalg.norm(d['y'] - d['center'], axis=1) >= 0.05
    ry, rp = np.linalg.norm(vy, axis=1), np.linalg.norm(vp, axis=1)
    cosang = (vy * vp).sum(1) / np.maximum(ry * rp, 1e-9)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1))), np.abs(rp - ry) / np.maximum(ry, 1e-9), ok


def boot(x, rng, B=2000):
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='flowET_s0_rayeval')
    ap.add_argument('--names', default='ray_eye_s0,ray_eye_nomarch_s0,ray_noeye_s0,ray_eye_s1,flow_eye_d6_s0')
    ap.add_argument('--main', default='', help='C16 等：对 --base 判定的候选臂名（如 raytok_eye_s0）')
    a = ap.parse_args()
    base = load(a.base)
    if a.base == 'flowET_s0_rayeval' and glob.glob(f'{D}/flow_eye_s0_shard*of*.npz'):
        old = load('flow_eye_s0', extra=False)
        same = all(np.array_equal(base[k], old[k]) for k in KEYS)
        print(f'[对拍] flowET 重评 vs 已有 flow_eye_s0：nll/cov/cov05/l2 {"逐位相同" if same else "**不同**"}')
    names = [n for n in a.names.split(',') if glob.glob(f'{D}/{n}_shard*of*.npz')]
    arms = {a.base: base, **{n: load(n) for n in names}}
    rng = np.random.default_rng(0)
    print(f'\n{"臂":22s} {"NLL↓":>7s} {"σ_s":>3s} {"cov10↑":>7s} {"cov05↑":>7s} {"点L2↓":>7s} {"方向误差°↓":>10s} {"相对径向↓":>9s} {"P(out)":>7s}')
    for n, d in arms.items():
        m = d['nll'].mean(0)
        j = int(np.argmin(m))
        ang, rad, ok = mech(d)
        po = np.nanmean(d['p_out']) if np.isfinite(d['p_out']).any() else float('nan')
        print(f'{n:22s} {m[j]:7.4f} {(1, 2, 3)[j]:3d} {d["cov"].mean():7.4f} {d["cov05"].mean():7.4f} {d["l2"].mean():7.4f} {ang[ok].mean():10.3f} {rad[ok].mean():9.4f} {po:7.4f}')
    verdict = {}
    for n in names:
        d = arms[n]
        jb, jd = int(np.argmin(base['nll'].mean(0))), int(np.argmin(d['nll'].mean(0)))
        ang0, rad0, ok0 = mech(base)
        ang1, rad1, ok1 = mech(d)
        ok = ok0 & ok1
        print(f'\n{n} − {a.base}（配对；NLL/L2/误差 负 = {n} 更好，cov 正 = 更好）：')
        rows = (('NLL', d['nll'][:, jd] - base['nll'][:, jb]), ('cov10', d['cov'] - base['cov']), ('cov05', d['cov05'] - base['cov05']),
                ('点L2', d['l2'] - base['l2']), ('方向误差°', (ang1 - ang0)[ok]), ('相对径向', (rad1 - rad0)[ok]))
        for k, x in rows:
            mu, lo, hi = boot(x, rng)
            print(f'  {k:8s} Δ {mu:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]{"  跨0" if lo < 0 < hi else ""}')
            verdict[(n, k)] = (mu, lo, hi)
    if 'ray_eye_s0' in names and a.base == 'flowET_s0_rayeval':
        dn = verdict[('ray_eye_s0', 'NLL')][0]
        rr = verdict[('ray_eye_s0', '相对径向')]
        print('\n== C15 判定（hold-out）==')
        print(f'主方法选择：ray_eye_s0 − {a.base} 的 NLL 差 {dn:+.4f} → {"射线头为主方法（≤ −0.02）" if dn <= -0.02 else "主方法仍为 flowET（差未达 −0.02）"}')
        print(f'机制预测（相对径向误差降低，CI 上界 < 0）：Δ {rr[0]:+.4f} [{rr[1]:+.4f}, {rr[2]:+.4f}] → {"成立" if rr[2] < 0 else "不成立"}')
    if a.main and a.main in names:
        dn = verdict[(a.main, 'NLL')]
        rr = verdict[(a.main, '相对径向')]
        print(f'\n== {a.main} 对 {a.base} 的判定（hold-out；C16/C17：NLL 差 ≤ −0.02 才算胜出，机制 = 相对径向误差 CI 上界 < 0）==')
        print(f'NLL 差 {dn[0]:+.4f} [{dn[1]:+.4f}, {dn[2]:+.4f}] → {"胜出（≤ −0.02）" if dn[0] <= -0.02 else "未胜出"}')
        print(f'相对径向误差 Δ {rr[0]:+.4f} [{rr[1]:+.4f}, {rr[2]:+.4f}] → {"机制成立" if rr[2] < 0 else "机制不成立"}')


if __name__ == '__main__':
    main()
