"""MAINTABLE-20261109：主表的一次性 test 读数（GF test + VAT test），口径见 docs/maintable-freeze-20261109.md。

  prepare           锁定 checkpoint / 数据 / 运行代码的 sha256，写 manifest.json（不读 test）
  worker --part P   DLC 上跑：8 卡同机对拍 SFT val → prompt 对拍 → 本分片的 test 推理（只落 preds，不打分）
  readout           三个分片都完成后，在 CPU 上一次性打分、出主表（这一步才「读」test）

运行时代码先复制进隔离快照 workdir/maintable_20261109_audit 并 commit，DLC 跑快照里的副本；项目目录只作数据与产物根。
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grpo_val_report as R  # noqa: E402

ROOT = Path(str(Path(__file__).resolve().parents[1]))
AUDIT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/maintable_20261109'
EXP = ROOT / 'experiments'
SIGMA = 3.0  # KDE 带宽（格），验证集上选定，见 kde_sigma_selection.md
DATA = {'gf': ROOT / 'data/vqa/gazefollow_test.jsonl', 'vat': ROOT / 'data/vqa/vat_test.jsonl'}
GF_VAL = ROOT / 'data/vqa/gazefollow_val.jsonl'
# 行名 -> (adapter, 锁定依据)。adapter=None 是零样本基座。
MODELS = {
    'zeroshot': (None, 'Qwen3-VL-4B-Instruct 基座'),
    'sft_v1': (EXP / 'sft_v1/v0-20261105-141107/checkpoint-3000', 'select_val_sel.json（GF val_sel）'),
    'grpo_B_full': (EXP / 'grpo_B_full_tau030_16gpu/attempt_56/v0-20261108-172822/checkpoint-2300', 'cp2.json（GF val_sel 35 点）'),
    'grpo_C_full': (EXP / 'grpo_C_full_tau030_16gpu/attempt_24/v0-20261108-130459/checkpoint-2200', 'cp2.json（GF val_sel 35 点）'),
    'sft_mix': (EXP / 'sft_mix_v1/v0-20261107-185905/checkpoint-4000', 'sft_mix_v1/select.json（GF val_sel + VAT val）'),
    'grpo_B_mix': (EXP / 'grpo_B_mix_tau030_8gpu/attempt_18/v0-20261108-172918/checkpoint-2000', 'mix_selection.json（19 点）'),
    'grpo_C_mix': (EXP / 'grpo_C_mix_tau030_8gpu/attempt_27/v0-20261108-172918/checkpoint-3000', 'mix_selection.json（19 点）'),
}
# 三个 DLC 分片，各 8 卡；VAT test 31978 条约 55 min/模型，GF test 4782 条约 8 min/模型
PARTS = {
    'P1': [(m, 'gf') for m in MODELS] + [('zeroshot', 'vat')],
    'P2': [('sft_mix', 'vat'), ('grpo_B_mix', 'vat'), ('grpo_C_mix', 'vat')],
    'P3': [('sft_v1', 'vat'), ('grpo_B_full', 'vat'), ('grpo_C_full', 'vat')],
}
RUNTIME = ('eval_gaze.py', 'check_prompt_parity.py', 'cp1_r16.py', 'cp_split.py', 'grpo_val_report.py', 'maintable.py')
# 主表的配对比较（读数前写定；全部报告，不在 test 上挑）：(数据集, 对照, 被比较)
COMPARISONS = [
    ('gf', 'sft_v1', 'grpo_B_full'), ('gf', 'sft_v1', 'grpo_C_full'), ('gf', 'grpo_B_full', 'grpo_C_full'),
    ('gf', 'sft_mix', 'grpo_B_mix'), ('gf', 'sft_mix', 'grpo_C_mix'), ('gf', 'grpo_B_mix', 'grpo_C_mix'),
    ('gf', 'gazelle', 'grpo_C_full'), ('gf', 'gazelle', 'grpo_B_full'), ('gf', 'gazelle', 'grpo_C_mix'), ('gf', 'gazelle', 'grpo_B_mix'),
    ('vat', 'sft_mix', 'grpo_B_mix'), ('vat', 'sft_mix', 'grpo_C_mix'), ('vat', 'grpo_B_mix', 'grpo_C_mix'),
    ('vat', 'gazelle', 'grpo_C_mix'), ('vat', 'gazelle', 'grpo_B_mix'),
]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def audit_commit():
    marker = AUDIT / 'SNAPSHOT_COMMIT.txt'
    if marker.exists():
        return marker.read_text().strip()
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=AUDIT, text=True).strip()


def pred_path(model, ds):
    return OUT / 'test' / model / ds / 'preds.jsonl'


def prepare():
    cp = R.load_module()
    # 锁定点必须与已落盘的选点报告一致
    cp2 = json.loads((ROOT / 'results/validation_20261109/cp2.json').read_text())
    assert cp2['locked_steps'] == {'SFT': 3000, 'B': 2300, 'C': 2200}
    for arm, key in (('SFT', 'sft_v1'), ('B', 'grpo_B_full'), ('C', 'grpo_C_full')):
        assert Path(cp2['splits']['val_gate']['table'][arm]['infer_meta']['adapter']).resolve() == MODELS[key][0].resolve()
    mix = json.loads((ROOT / 'results/validation_20261109/mix_selection.json').read_text())
    for arm in ('B', 'C'):
        src = Path(mix['selected'][arm]['sources']['gfval']['path'])
        meta = json.loads(Path(str(src) + '.meta.json').read_text())
        assert Path(meta['adapter']).resolve() == MODELS[f'grpo_{arm}_mix'][0].resolve(), arm
    sm = json.loads((ROOT / 'results/sft_mix_v1/ckpt4000_gfval_greedy/preds.jsonl.meta.json').read_text())
    assert Path(sm['adapter']).resolve() == MODELS['sft_mix'][0].resolve()
    manifest = {'id': 'MAINTABLE-20261109', 'kde_sigma_cells': SIGMA, 'parts': PARTS, 'comparisons': COMPARISONS,
                'models': {}, 'data_sha256': {}, 'runtime_sha256': {}}
    for name, (adapter, why) in MODELS.items():
        entry = {'adapter': None if adapter is None else str(adapter), 'locked_by': why}
        if adapter is not None:
            entry['adapter_sha256'] = cp.sha256_file(adapter / 'adapter_model.safetensors')
            assert json.loads((adapter / 'args.json').read_text())['max_pixels'] == 262144, name
        manifest['models'][name] = entry
    for path in (*DATA.values(), GF_VAL):
        manifest['data_sha256'][path.name] = cp.sha256_file(path)
    for name in RUNTIME:
        manifest['runtime_sha256'][name] = cp.sha256_file(AUDIT / 'tools' / name)
    write(AUDIT / 'manifest.json', manifest)
    write(OUT / 'manifest.json', manifest)
    print(f'manifest ready: {len(MODELS)} models, parts {list(PARTS)}', flush=True)


def check_runtime(cp, manifest):
    for name, digest in manifest['runtime_sha256'].items():
        assert cp.sha256_file(AUDIT / 'tools' / name) == digest, name
    for path in (*DATA.values(), GF_VAL):
        assert cp.sha256_file(path) == manifest['data_sha256'][path.name], path


def validate_preds(path, data, adapter):
    meta = json.loads(Path(str(path) + '.meta.json').read_text())
    assert Path(meta['data']).resolve() == data.resolve()
    assert (meta['adapter'] is None) if adapter is None else (Path(meta['adapter']).resolve() == adapter.resolve())
    assert meta['max_pixels'] == 262144 and meta['num_samples'] == 0
    assert meta['prompt_versions'] == ['v1'] and meta['system'] is None
    assert Path(meta['model']).resolve() == (ROOT / 'ckpt/Qwen3-VL-4B-Instruct').resolve()
    assert R.read_unique(path).keys() == R.read_unique(data).keys()
    return meta


def infer(pred, data, adapter):
    pred.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-I', str(AUDIT / 'tools/eval_gaze.py'), 'infer', '--data', str(data),
           '--model', str(ROOT / 'ckpt/Qwen3-VL-4B-Instruct'), '--out', str(pred),
           '--gpus', '0,1,2,3,4,5,6,7', '--batch-size', '8', '--max-pixels', '262144']
    if adapter is not None:
        cmd[cmd.index('--out'):cmd.index('--out')] = ['--adapter', str(adapter)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def worker(part):
    cp = R.load_module()
    manifest = json.loads((AUDIT / 'manifest.json').read_text())
    check_runtime(cp, manifest)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 8, 'Require exactly 8 devices'
    # 同机对拍：重推 SFT v1 的 GF val greedy，必须与 DSW 落盘结果逐条相同，否则不读 test
    parity = OUT / f'parity_{part}/preds.jsonl'
    sft = MODELS['sft_v1'][0]
    if not (parity.parent / 'PASS.json').exists():
        infer(parity, GF_VAL, sft)
        validate_preds(parity, GF_VAL, sft)
        old, new = R.read_unique(ROOT / 'results/sft_v1/ckpt3000_val_greedy/preds.jsonl'), R.read_unique(parity)
        diff = sum(old[i]['greedy']['text'] != new[i]['greedy']['text'] for i in old)
        assert diff == 0, f'parity FAIL: {diff} texts differ'
        dp = max(abs(old[i]['greedy']['p_in_io'] - new[i]['greedy']['p_in_io']) for i in old)
        write(parity.parent / 'PASS.json', {'n': len(old), 'text_diff': 0, 'p_in_io_max_diff': dp})
        print(f'PARITY PASS {len(old)} people, text diff 0, p_in_io max diff {dp}', flush=True)
    for ds in sorted({d for _, d in PARTS[part]}):
        check = subprocess.run([str(ROOT / 'envs/swift/bin/python'), str(AUDIT / 'tools/check_prompt_parity.py'),
                                '--data', str(DATA[ds]), '--max-pixels', '262144', '--n', '40'],
                               cwd=ROOT, capture_output=True, text=True)
        write(OUT / f'parity_{part}/prompt_parity_{ds}.log', check.stdout + check.stderr)
        if check.returncode != 0 or 'mismatches 0  pixel_values max|Δ| 0.0' not in check.stdout:
            raise ValueError(f'{ds}: prompt parity failed: {check.stdout[-800:]} {check.stderr[-800:]}')
    for model, ds in PARTS[part]:
        adapter, data = MODELS[model][0], DATA[ds]
        if adapter is not None:
            assert cp.sha256_file(adapter / 'adapter_model.safetensors') == manifest['models'][model]['adapter_sha256']
        pred = pred_path(model, ds)
        valid = pred.parent / 'VALID.json'
        if valid.exists():
            assert json.loads(valid.read_text())['preds_sha256'] == cp.sha256_file(pred)
            validate_preds(pred, data, adapter)
            continue
        print(f'MAINTABLE {part} {model} {ds} start', flush=True)
        infer(pred, data, adapter)
        validate_preds(pred, data, adapter)
        write(valid, {'model': model, 'ds': ds, 'n': len(R.read_unique(pred)), 'data_sha256': cp.sha256_file(data),
                      'adapter_sha256': manifest['models'][model].get('adapter_sha256'),
                      'preds_sha256': cp.sha256_file(pred), 'commit': audit_commit()})
        print(f'MAINTABLE {part} {model} {ds} VALID', flush=True)
    write(OUT / f'PART_{part}_COMPLETE.json', {'part': part, 'items': PARTS[part], 'commit': audit_commit()})
    print(f'MAINTABLE {part} COMPLETE', flush=True)


# ------------------------------------------------------------------------------------------------------------------
# readout：一次性打分
# ------------------------------------------------------------------------------------------------------------------
def score_ours(E, ds, model):
    rows = E.load_jsonl(DATA[ds])
    recs = {r['id']: r for r in E.load_jsonl(pred_path(model, ds))}
    insts = [E.make_instance(r, recs[r['id']]) for r in rows]
    m, per = E.score_instances(insts, SIGMA, ())
    (metrics,) = m.values()
    return rows, metrics, per


def score_gazelle(E, ds):
    """官方 Gaze-LLE ViT-L 已落盘热图回放（与 eval_gaze selfcheck [3] 同一路径；点 = argmax 格子左上角，官方口径）。"""
    import numpy as np
    rows = E.load_jsonl(DATA[ds])
    if ds == 'gf':
        dump = np.load(ROOT / 'results/repro_test/full.npz')
        hms = dump['heatmaps'].astype(np.float32)
        pts = [(ix / E.RES, iy / E.RES) for iy, ix in (np.unravel_index(h.argmax(), h.shape) for h in hms)]
        insts = E._replay_instances(rows, hms, pts, [1.0] * len(rows))
    else:
        dump = np.load(ROOT / 'results/vat_official/official_inout_vitl14.npz')
        hms, pts = [None] * len(rows), [None] * len(rows)
        heat = dump['heatmaps']  # NpzFile 每次 [] 都会重新解压整个数组，先取出来
        for k, n in enumerate(dump['in_idx']):
            h = heat[k].astype(np.float32)
            hms[n] = h
            iy, ix = np.unravel_index(h.argmax(), h.shape)
            pts[n] = (ix / E.RES, iy / E.RES)
        insts = E._replay_instances(rows, hms, pts, list(dump['inout_pred']))
        for i in insts:  # 官方对 GT 画面外样本不出点，只参与 AP；两种 AP 用同一个 in/out 头
            if i['heatmap'] is None:
                i['g_pt'] = E.CENTER
            i['p_in']['dir'] = i['p_in']['io']
    m, per = E.score_instances(insts, SIGMA, ())
    (metrics,) = m.values()
    return rows, metrics, per


def readout():
    cp = R.load_module()
    sys.path.insert(0, str(AUDIT / 'tools'))
    import eval_gaze as E
    manifest = json.loads((AUDIT / 'manifest.json').read_text())
    check_runtime(cp, manifest)
    for part in PARTS:
        assert (OUT / f'PART_{part}_COMPLETE.json').exists(), f'{part} not complete'
    sources = {}
    for part, items in PARTS.items():
        for model, ds in items:
            pred = pred_path(model, ds)
            validate_preds(pred, DATA[ds], MODELS[model][0])
            sha = cp.sha256_file(pred)
            assert sha == json.loads((pred.parent / 'VALID.json').read_text())['preds_sha256']
            sources[f'{model}:{ds}'] = {'path': str(pred), 'sha256': sha}
    import numpy as np
    res, arrays, clusters = {'gf': {}, 'vat': {}}, {}, {}
    for ds in ('gf', 'vat'):
        have = {m for m, d in sum(PARTS.values(), []) if d == ds}
        names = [m for m in MODELS if m in have] + ['gazelle']
        for name in names:
            rows, metrics, per = score_gazelle(E, ds) if name == 'gazelle' else score_ours(E, ds, name)
            res[ds][name] = metrics
            ins = [(r, p) for r, p in zip(rows, per) if r['inout']]
            arrays[ds, name] = np.array([p['avg_l2'] for _, p in ins])
            key = [r['frame_idx'] if ds == 'gf' else r['seq'] for r, _ in ins]  # GF 按图、VAT 按片段（131 个）聚类
            assert clusters.setdefault(ds, key) == key
            print(ds, name, {k: metrics.get(k) for k in ('auc', 'avg_l2', 'min_l2', 'l2', 'ap_inout_dir', 'ap_inout')}, flush=True)
    comps = [{'ds': ds, 'baseline': b, 'model': m, 'rel_l2_improvement': cp.boot_ratio(arrays[ds, b], arrays[ds, m], clusters[ds])}
             for ds, b, m in COMPARISONS]
    out = {'id': 'MAINTABLE-20261109', 'commit': audit_commit(), 'kde_sigma_cells': SIGMA, 'metrics': res,
           'comparisons': comps, 'sources': sources,
           'bootstrap': {'gf_cluster': 'frame_idx', 'vat_cluster': 'seq (clip)', 'resamples': cp.BOOT_B, 'seed': cp.BOOT_SEED}}
    write(OUT / 'main_table.json', out)
    write(OUT / 'main_table.md', render(out))
    print(render(out), flush=True)


LABEL = {'zeroshot': '零样本 Qwen3-VL-4B', 'sft_v1': 'SFT（只 GF）', 'grpo_B_full': 'GRPO B（只 GF）',
         'grpo_C_full': 'GRPO C（只 GF）', 'sft_mix': 'SFT（GF+VAT）', 'grpo_B_mix': 'GRPO B（GF+VAT）',
         'grpo_C_mix': 'GRPO C（GF+VAT）', 'gazelle': 'Gaze-LLE ViT-L 官方（自评）'}


def render(out):
    def ci(r):
        return f'{r["point"]:+.2%} [{r["ci95"][0]:+.2%}, {r["ci95"][1]:+.2%}]'
    gf, vat = out['metrics']['gf'], out['metrics']['vat']
    lines = ['# 主表：GF test / VAT test 一次性读数（MAINTABLE-20261109）', '',
             f'口径见 docs/maintable-freeze-20261109.md。eval_gaze/1，prompt v1，greedy 8 卡 batch 8，max_pixels=262144；'
             f'AUC 为单点 KDE σ={out["kde_sigma_cells"]:g} 格；快照 `{out["commit"]}`。', '',
             '## GazeFollow test（4782 人，约 10 标注/人）', '',
             '| 模型 | AUC | Avg L2 | Min L2 | 中心回退率 |', '|---|---:|---:|---:|---:|']
    for n, m in gf.items():
        lines.append(f'| {LABEL[n]} | {m["auc"]:.4f} | {m["avg_l2"]:.4f} | {m["min_l2"]:.4f} | {m["center_fallback_rate_in_frame"]:.2%} |')
    lines += ['', '## VAT test（31978 条，画面内 20655）', '',
              '| 模型 | AUC | L2 | AP（`<dir>`，主） | AP（`<io>`） | 画面内回退率 |', '|---|---:|---:|---:|---:|---:|']
    for n, m in vat.items():
        note = '†' if n in ('zeroshot', 'sft_v1', 'grpo_B_full', 'grpo_C_full') else ''
        lines.append(f'| {LABEL[n]}{note} | {m["auc"]:.4f} | {m["l2"]:.4f} | {m["ap_inout_dir"]:.4f} | {m["ap_inout"]:.4f} | '
                     f'{m["center_fallback_rate_in_frame"]:.2%} |')
    lines += ['', '† 没用 VAT 训练、没见过画面外样本，VAT 列是跨数据集迁移；AP 仅供参考。',
              'Gaze-LLE 行：GF 用 gazelle_dinov2_vitl14、VAT 用 gazelle_dinov2_vitl14_inout，已落盘热图回放；L2 取 argmax 格子左上角（官方口径，无 +0.5）；两种 AP 是同一个 in/out 头。', '',
              '## 配对比较（L2 相对改善，正 = 被比较者更好；GF 按图、VAT 按片段 cluster bootstrap）', '',
              '| 数据集 | 对照 | 被比较 | 相对改善（95% CI） |', '|---|---|---|---|']
    for c in out['comparisons']:
        lines.append(f'| {c["ds"].upper()} | {LABEL[c["baseline"]]} | {LABEL[c["model"]]} | {ci(c["rel_l2_improvement"])} |')
    lines += ['', '比较清单读数前写定、全部报告；未作多重比较校正；单种子。', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'worker', 'readout'))
    parser.add_argument('--part', choices=tuple(PARTS))
    a = parser.parse_args()
    os.chdir(ROOT)
    if a.mode == 'worker':
        if a.part is None:
            parser.error('worker requires --part')
        worker(a.part)
    else:
        {'prepare': prepare, 'readout': readout}[a.mode]()
