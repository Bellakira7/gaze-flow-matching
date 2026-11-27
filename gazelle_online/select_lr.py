"""按 G2.clarifications_20261022.md §C4 的预声明规则，从 3-epoch 筛选结果里为每个头选 lr。
python -I select_lr.py   ->  experiments/queue_selection.json  （stdout 打印对比表）
MDN/Flow：hold-out 最后一个 epoch 的 nll_best 更低者；MCL：cov10 更高者。平手取 1e-3。
"""
from pathlib import Path
import json
import sys

G = str(Path(__file__).resolve().parents[1])
LRS = ['1e-3', '3e-4']
RULE = {'flow': ('nll_best', min), 'mdn': ('nll_best', min), 'mcl': ('cov10', max)}
sel = {}
for head, (key, pick) in RULE.items():
    vals = {}
    for lr in LRS:
        rows = [json.loads(l) for l in open(f'{G}/experiments/scr_{head}_lr{lr}_s0/metrics.jsonl')]
        assert len(rows) == 3 and rows[-1]['epoch'] == 2, (head, lr, len(rows))
        vals[lr] = rows[-1][key]
        print(f'{head:5s} lr={lr:5s} {key}={vals[lr]:.4f}  (nll_best {rows[-1]["nll_best"]:.4f}  cov10 {rows[-1]["cov10"]:.4f}  pt-L2 {rows[-1]["l2_pt"]:.4f})')
    best = pick(vals.values())
    sel[head] = LRS[0] if vals[LRS[0]] == best else LRS[1]
    print(f'  -> {head}: lr={sel[head]}')
json.dump(sel, open(f'{G}/experiments/queue_selection.json', 'w'), indent=1)
