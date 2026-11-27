"""Gaze-LLE 在线特征版训练（冻结 backbone 在线前向，不缓存特征），8 卡 DDP。

torchrun --nproc_per_node 8 train_online.py --exp_name b1_dinov2l_sd --backbone dinov2_vitl14

与官方 scripts/train_gazefollow.py 的配方对齐：Adam lr 1e-3、cosine(T_max=epochs, eta_min=1e-7，按 epoch 步进)、BCE、15 epoch、
官方 GazeDataset 的增强（random_crop / 翻转 / bbox 抖动，各 p=0.5）。与官方的差别（逐条列出，别默默扩散）：
  1. 全局 batch 64（官方 60），8 卡各 8；
  2. backbone 用 fp16 autocast 前向（官方 fp32）；实测与 fp32 特征相对 L2 差 0.2%、token 余弦 1.0000，吞吐 3.6x；
  3. 训练集是 ShijianDeng 镜像 + 眼睛为中心的正方形头框（边长 0.12*sqrt(WH)），不是原始 GazeFollow 的真值头框；
  4. **不在测试集上选 epoch**：每个 epoch 只在训练内 hold-out（按图片内容哈希切 3%）上评测；测试集由单独脚本、对选定 ckpt 评一次。
"""
from pathlib import Path
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_cached')
from common import FEAT_DIMS, IN_SIZE, LocalDinoV2Backbone, _instance_metrics  # noqa: E402  (顺带把官方 gazelle 加进 sys.path)
from gazelle.dataloader import GazeDataset, collate_fn  # noqa: E402
from gazelle.model import GazeLLE  # noqa: E402


class AmpBackbone(LocalDinoV2Backbone):
    """冻结 backbone：fp16 autocast 前向、no_grad、输出转回 fp32（解码器仍是 fp32，与官方一致）。"""

    def forward(self, x):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            y = super().forward(x)
        return y.float()


class SDGazeDataset(GazeDataset):
    """官方 GazeDataset + 按 frame 的 hold-out 标记过滤。fit = 训练（带增强，返回 heatmap）；holdout = 评测格式（无增强）。"""

    def __init__(self, path, mode, transform):
        super().__init__('gazefollow', path, 'train', transform)
        want = mode == 'holdout'
        self.data_idxs = [(i, j) for (i, j) in self.data_idxs if self.data[i]['holdout'] == want]
        if want:
            self.aug, self.split = False, 'test'  # __getitem__ 据此返回评测格式 (img, bbox, gazex, gazey, inout, h, w)


def build_model(name):
    assert name in FEAT_DIMS, name
    return GazeLLE(AmpBackbone(name))


def decoder_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith('backbone')}


def load_decoder_state(model, sd):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith('backbone') for k in missing), (missing[:3], unexpected[:3])


@torch.no_grad()
def eval_holdout(model, ds, rank, world, bs, n_workers):
    model.eval()
    sub = torch.utils.data.Subset(ds, list(range(rank, len(ds), world)))
    dl = torch.utils.data.DataLoader(sub, batch_size=bs, collate_fn=collate_fn, num_workers=n_workers)
    res = []
    for imgs, bboxes, gazex, gazey, _, heights, widths in dl:
        preds = model({'images': imgs.cuda(non_blocking=True), 'bboxes': [[b] for b in bboxes]})
        hm = torch.stack(preds['heatmap']).squeeze(dim=1).float().cpu().numpy()
        for i in range(hm.shape[0]):
            res.append(_instance_metrics((hm[i], gazex[i], gazey[i], heights[i], widths[i])))
    allres = [None] * world
    dist.all_gather_object(allres, np.array(res).reshape(-1, 5))
    r = np.concatenate(allres).mean(0)
    return dict(zip(['auc', 'avg_l2', 'min_l2', 'avg_l2_fix', 'min_l2_fix'], map(float, r)), n=int(sum(len(a) for a in allres)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='dinov2_vitl14', choices=list(FEAT_DIMS))
    ap.add_argument('--data_path', default=f'{G}/data/gazefollow_train_sd')
    ap.add_argument('--exp_name', required=True)
    ap.add_argument('--exp_root', default=f'{G}/experiments')
    ap.add_argument('--max_epochs', type=int, default=15)
    ap.add_argument('--batch_size', type=int, default=64, help='全局 batch')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--n_workers', type=int, default=10, help='每卡 dataloader 进程数')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max_steps', type=int, default=0, help='每 epoch 最多多少 step（smoke）')
    ap.add_argument('--limit_holdout', type=int, default=0, help='hold-out 只评前 N 个（smoke）')
    args = ap.parse_args()

    rank, world, lrank = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(lrank)
    dist.init_process_group('nccl')
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    assert args.batch_size % world == 0
    per_rank_bs = args.batch_size // world

    exp_dir = os.path.join(args.exp_root, args.exp_name)
    last = os.path.join(exp_dir, 'last.pt')
    if rank == 0:
        if os.path.exists(last) and not args.resume:
            sys.exit(f'{exp_dir} 已有 last.pt；要续训请加 --resume，否则换 exp_name（别复用目录）')
        os.makedirs(exp_dir, exist_ok=True)
        json.dump(vars(args), open(os.path.join(exp_dir, 'args.json'), 'w'), indent=1)
    dist.barrier()

    model = build_model(args.backbone).cuda()
    for p in model.backbone.parameters():
        p.requires_grad = False
    transform = model.backbone.get_transform(IN_SIZE)
    train_ds = SDGazeDataset(args.data_path, 'fit', transform)
    hold_ds = SDGazeDataset(args.data_path, 'holdout', transform)
    if args.limit_holdout:
        hold_ds.data_idxs = hold_ds.data_idxs[:args.limit_holdout]
    if rank == 0:
        print(f'backbone {args.backbone}  可训练参数 {sum(p.numel() for p in model.parameters() if p.requires_grad)}  '
              f'train {len(train_ds)} heads  hold-out {len(hold_ds)} heads  全局 batch {args.batch_size}', flush=True)

    sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True, seed=args.seed)
    dl = torch.utils.data.DataLoader(train_ds, batch_size=per_rank_bs, sampler=sampler, collate_fn=collate_fn, num_workers=args.n_workers,
                                     persistent_workers=True, prefetch_factor=4, drop_last=True)
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[lrank], broadcast_buffers=False)
    loss_fn = nn.BCELoss()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs, eta_min=1e-7)

    start = 0
    if args.resume and os.path.exists(last):
        ck = torch.load(last, map_location='cpu', weights_only=True)
        load_decoder_state(model, ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start = ck['epoch'] + 1
        if rank == 0:
            print(f'resume from epoch {start}', flush=True)

    log = open(os.path.join(exp_dir, 'metrics.jsonl'), 'a') if rank == 0 else None
    for epoch in range(start, args.max_epochs):
        sampler.set_epoch(epoch)
        model.train()
        model.backbone.eval()
        t0, t_data, n_img = time.time(), 0.0, 0
        t_prev = time.time()
        for it, batch in enumerate(dl):
            t_data += time.time() - t_prev
            imgs, bboxes, _, _, _, _, _, heatmaps = batch
            opt.zero_grad(set_to_none=True)
            preds = ddp({'images': imgs.cuda(non_blocking=True), 'bboxes': [[b] for b in bboxes]})
            loss = loss_fn(torch.stack(preds['heatmap']).squeeze(dim=1), heatmaps.cuda(non_blocking=True))
            loss.backward()
            opt.step()
            n_img += imgs.shape[0] * world
            if rank == 0 and it % 50 == 0:
                el = time.time() - t0
                print(f'TRAIN EPOCH {epoch} iter {it}/{len(dl)} loss {loss.item():.4f}  {n_img / el:.0f} img/s  等数据占比 {t_data / el:.2f}', flush=True)
            if args.max_steps and it + 1 >= args.max_steps:
                break
            t_prev = time.time()
        sched.step()

        res = eval_holdout(model, hold_ds, rank, world, per_rank_bs, 4)
        res.update(epoch=epoch, train_time_s=round(time.time() - t0, 1), lr_next=sched.get_last_lr()[0])
        if rank == 0:
            torch.save(decoder_state(model), os.path.join(exp_dir, f'epoch_{epoch}.pt'))
            torch.save({'model': decoder_state(model), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': epoch}, last + '.tmp')
            os.replace(last + '.tmp', last)
            log.write(json.dumps(res) + '\n')
            log.flush()
            print(f"HOLD-OUT EPOCH {epoch}: AUC={res['auc']:.4f}  AvgL2={res['avg_l2']:.4f} (修正 {res['avg_l2_fix']:.4f})  n={res['n']}  用时 {res['train_time_s']}s", flush=True)
        dist.barrier()
    if rank == 0:
        print('done', exp_dir, flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
