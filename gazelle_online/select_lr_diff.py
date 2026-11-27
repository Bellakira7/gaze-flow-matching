"""B5(diff) 的 lr 选择，规则同 select_lr.py / prereg/G2.clarifications_20261022.md §C4：hold-out 最后一个 epoch 的 nll_best 更低者，平手取 1e-3。
python -I select_lr_diff.py  ->  experiments/queue_selection_diff.json
"""
from pathlib import Path
import json

G = str(Path(__file__).resolve().parents[1])
vals = {}
for lr in ('1e-3', '3e-4'):
    rows = [json.loads(l) for l in open(f'{G}/experiments/scr_diff_lr{lr}_s0/metrics.jsonl')]
    assert len(rows) == 3 and rows[-1]['epoch'] == 2, (lr, len(rows))
    vals[lr] = rows[-1]['nll_best']
    print(f'diff lr={lr:5s} nll_best={vals[lr]:.4f} cov10 {rows[-1]["cov10"]:.4f} pt-L2 {rows[-1]["l2_pt"]:.4f}')
sel = '1e-3' if vals['1e-3'] <= vals['3e-4'] else '3e-4'
print('  -> diff: lr=' + sel)
json.dump({'diff': sel}, open(f'{G}/experiments/queue_selection_diff.json', 'w'), indent=1)
