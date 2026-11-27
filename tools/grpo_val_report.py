#!/usr/bin/env python
"""GRPO 扫参报告：只计算 val_sel，不读取 val_gate 的指标。

/usr/local/bin/python -I tools/grpo_val_report.py \
  --preds C_t30:50=results/grpo/<run>/ckpt50_val_greedy/preds.jsonl ... \
  --out results/grpo/<run>/select_val_sel.json --report results/grpo/<run>/val_sel_report.md

label 中冒号前为 run 名；每个 run 独立选 checkpoint。--pair 左标签=右标签
报告右模型相对左模型的改善及按帧配对 bootstrap 区间（2000 次，seed 0）。
沿用冻结的 eval_gaze.py / cp_split.py / cp1_r16.py，不修改评分或切分。
这是扫参报告，不作 CP2 判定；选出的 checkpoint 的区间没有校正选择偏差。
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_module():
    spec = importlib.util.spec_from_file_location('cp1_r16', HERE / 'cp1_r16.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_unique(path):
    records = {}
    with open(path) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['id'] in records:
                raise ValueError(f'{path}: duplicate id {row["id"]}')
            records[row['id']] = row
    return records


def evaluate(path, rows, ids, cp):
    path = Path(path).resolve()
    meta = json.loads(Path(str(path) + '.meta.json').read_text())
    if meta.get('num_samples') != 0 or meta.get('max_pixels') != 262144:
        raise ValueError(f'{path}: expected greedy, max_pixels=262144')
    if Path(meta['data']).resolve() != Path(cp.G, 'data/vqa/gazefollow_val.jsonl').resolve():
        raise ValueError(f'{path}: metadata refers to a different validation dataset')
    if meta.get('system') is not None or meta.get('prompt_versions') != ['v1']:
        raise ValueError(f'{path}: expected prompt v1 and no custom system prompt')
    records = read_unique(path)
    if records.keys() != rows.keys():
        raise ValueError(f'{path}: predictions must cover full val exactly; missing={len(rows.keys() - records.keys())}, extra={len(records.keys() - rows.keys())}')
    values, statuses = [], []
    for sample_id in ids:
        point, status = cp.E.completion_to_point(records[sample_id]['greedy']['text'])
        row = rows[sample_id]
        values.append(cp.E.l2_stats(cp.E._sp(point), row['gt_xs_norm'], row['gt_ys_norm'])[0])
        statuses.append(status)
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'{path}: non-finite L2')
    info = {
        'preds': str(path), 'preds_sha256': cp.sha256_file(path),
        'infer_meta': meta, 'avg_l2': float(values.mean()),
        'center_fallback_rate': sum(s in ('parse_fail', 'pred_out') for s in statuses) / len(ids),
        'clamped_rate': statuses.count('clamped') / len(ids),
    }
    adapter = Path(meta['adapter'])
    train_args = adapter / 'args.json'
    if train_args.is_file():
        config = json.loads(train_args.read_text())
        info['training_config'] = {key: config.get(key) for key in (
            'learning_rate', 'beta', 'num_generations', 'per_device_train_batch_size',
            'gradient_accumulation_steps', 'reward_funcs', 'reward_weights', 'seed', 'data_seed')}
        info['training_config']['n_ranks'] = len(list(adapter.glob('rng_state_*.pth')))
    return values, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preds', nargs='+', required=True, help='run:step=path')
    ap.add_argument('--baseline', default=str(ROOT / 'results/sft_v1/ckpt3000_val_greedy/preds.jsonl'))
    ap.add_argument('--pair', nargs='*', default=[], help='left_label=right_label; positive means right is better')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report')
    args = ap.parse_args()
    cp = load_module()
    val = ROOT / 'data/vqa/gazefollow_val.jsonl'
    rows = read_unique(val)
    split = cp.SPLIT.load_split(str(ROOT / 'data/vqa/cp2_split'), str(val))
    if cp.sha256_file(val) != cp.SPLIT.VAL_SHA256:
        raise ValueError('validation dataset differs from preregistration')
    ids = split['val_sel']
    frames = [rows[i]['frame_idx'] for i in ids]
    baseline, baseline_info = evaluate(args.baseline, rows, ids, cp)
    arrays, table, groups = {}, [], {}
    for item in args.preds:
        label, sep, path = item.partition('=')
        if not sep or not label or label in arrays:
            raise ValueError(f'invalid or duplicate candidate: {item}')
        values, info = evaluate(path, rows, ids, cp)
        for key in ('model', 'max_pixels', 'num_samples', 'system', 'prompt_versions'):
            left, right = baseline_info['infer_meta'].get(key), info['infer_meta'].get(key)
            if key == 'model':
                left, right = str(Path(left).resolve()), str(Path(right).resolve())
            if left != right:
                raise ValueError(f'{label}: inference setting {key} differs from SFT')
        arrays[label] = values
        group = label.rsplit(':', 1)[0] if ':' in label else 'candidates'
        info.update(label=label, group=group, improvement_vs_sft=cp.boot_ratio(baseline, values, frames))
        table.append(info)
        groups.setdefault(group, []).append(info)
    selected = {name: min(items, key=lambda x: x['avg_l2'])['label'] for name, items in groups.items()}
    comparisons = []
    for pair in args.pair:
        left, sep, right = pair.partition('=')
        if not sep or left not in arrays or right not in arrays:
            raise ValueError(f'pair must refer to candidate labels: {pair}')
        configs = {row['label']: row.get('training_config', {}) for row in table}
        different_topology = any(configs[left].get(key) != configs[right].get(key)
                                 for key in ('n_ranks', 'gradient_accumulation_steps'))
        comparisons.append({'left': left, 'right': right, 'different_training_topology': different_topology,
                            'relative_improvement': cp.boot_ratio(arrays[left], arrays[right], frames)})
    result = {
        'scope': 'val_sel_only_exploratory_not_CP2', 'n_people': len(ids), 'n_frames': len(set(frames)),
        'bootstrap': {'cluster': 'frame_idx', 'resamples': 2000, 'seed': 0},
        'val_sha256': cp.sha256_file(val), 'baseline': baseline_info,
        'selection_rule': 'minimum val_sel greedy Avg L2 per run; ties follow input order',
        'table': table, 'selected': selected, 'comparisons': comparisons,
        'code_sha256': {p.name: cp.sha256_file(p) for p in (Path(__file__), HERE / 'cp1_r16.py', HERE / 'cp_split.py', HERE / 'eval_gaze.py')},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    lines = ['# GRPO val_sel 扫参报告', '',
             f'{len(ids)} 人 / {len(set(frames))} 帧；SFT checkpoint-3000 Avg L2 = {baseline.mean():.6f}。', '',
             '只用于选配置和 checkpoint；不读取 val_gate 指标，不作 CP2 判定。区间未校正 checkpoint 选择偏差。', '',
             '| candidate | Avg L2 | 改善 vs SFT | 95% CI | 回退率 |',
             '|---|---:|---:|---|---:|']
    for row in table:
        improvement = row['improvement_vs_sft']
        lo, hi = improvement['ci95']
        lines.append(f'| {row["label"]} | {row["avg_l2"]:.6f} | {improvement["point"]:+.2%} | [{lo:+.2%}, {hi:+.2%}] | {row["center_fallback_rate"]:.2%} |')
    lines += ['', '按每个 run 的 val_sel 最低 L2 选点：' + '；'.join(selected.values()) + '。']
    for pair in comparisons:
        improvement = pair['relative_improvement']
        lo, hi = improvement['ci95']
        lines += ['', f'{pair["right"]} 相对 {pair["left"]}：{improvement["point"]:+.2%} [{lo:+.2%}, {hi:+.2%}]。正值表示右模型更好。']
        if pair['different_training_topology']:
            lines += ['两次训练的卡数或梯度累积不同；该比较只能作筛选参考，不能将全部差异归因于 τ 或奖励设计。']
    lines += ['', f'数据、预测和评分代码的 SHA256 记录在 `{out.name}`。', '']
    report = '\n'.join(lines)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
    print(report)


if __name__ == '__main__':
    main()
