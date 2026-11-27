import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CachedGazeFollow, FEAT_DIMS, build_cached_gazelle, collate_train, evaluate)  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--backbone', default='dinov2_vitl14', choices=list(FEAT_DIMS))
p.add_argument('--data_path', required=True, help='含 *_preprocessed.json 的目录')
p.add_argument('--feat_dir', required=True)
p.add_argument('--ckpt_save_dir', default='./experiments')
p.add_argument('--exp_name', default='train_gazefollow_cached')
p.add_argument('--max_epochs', type=int, default=15)
p.add_argument('--batch_size', type=int, default=60)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--n_workers', type=int, default=16)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--no_flip_view', action='store_true')
p.add_argument('--eval_only', action='store_true')
p.add_argument('--ckpt_path', default=None, help='eval_only 时加载的 decoder 权重（官方 .pt 或本脚本存的 epoch_N.pt）')
p.add_argument('--dump', default=None, help='eval_only 时把测试集热图/逐实例指标存成 npz')
args = p.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
dev = 'cuda'

model = build_cached_gazelle(args.backbone).to(dev)
test_ds = CachedGazeFollow(args.data_path, args.feat_dir, 'test')

if args.eval_only:
    model.load_gazelle_state_dict(torch.load(args.ckpt_path, weights_only=True))
    res = evaluate(model, test_ds, dev, dump=args.dump)
    print(json.dumps(res, indent=1))
    sys.exit(0)

exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(exp_dir)
json.dump(vars(args), open(os.path.join(exp_dir, 'args.json'), 'w'), indent=1)
print('Learnable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

train_ds = CachedGazeFollow(args.data_path, args.feat_dir, 'train', flip_view=not args.no_flip_view)
print(f'train instances {len(train_ds)}, flip view {train_ds.has_flip}, test frames {len(test_ds)}')
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_train,
                                       num_workers=args.n_workers, persistent_workers=True)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

log = open(os.path.join(exp_dir, 'metrics.jsonl'), 'a')
for epoch in range(args.max_epochs):
    model.train()
    for it, (feats, bboxes, heatmaps) in enumerate(train_dl):
        optimizer.zero_grad()
        preds = model({'images': feats.to(dev).float(), 'bboxes': [[b] for b in bboxes]})
        loss = loss_fn(torch.stack(preds['heatmap']).squeeze(dim=1), heatmaps.to(dev))
        loss.backward()
        optimizer.step()
        if it % 50 == 0:
            print(f'TRAIN EPOCH {epoch}, iter {it}/{len(train_dl)}, loss={loss.item():.4f}', flush=True)
    scheduler.step()

    ckpt = os.path.join(exp_dir, f'epoch_{epoch}.pt')
    torch.save(model.get_gazelle_state_dict(), ckpt)
    res = evaluate(model, test_ds, dev)
    res['epoch'] = epoch
    log.write(json.dumps(res) + '\n')
    log.flush()
    print(f"EVAL EPOCH {epoch}: AUC={res['auc']:.4f} AvgL2={res['avg_l2']:.4f} MinL2={res['min_l2']:.4f} | "
          f"fixed(+0.5): AvgL2={res['avg_l2_fix']:.4f} MinL2={res['min_l2_fix']:.4f}", flush=True)
print('done', exp_dir)
