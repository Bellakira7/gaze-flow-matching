"""【眼部 token 变体 2（C17）】与 train_heads_eye.py 逐行相同，唯一差别：模型换成 gaze_eye2.EyeGazeModel2（--variant fuse | hires），
hires 时数据集换成 gaze_eye2.HiResCropDataset（多返回一个原图 crop，传给 trunk）；去掉 --head（固定 flow）。
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port 29690 train_eye2.py --variant fuse --lr 3e-4 --exp_name eyefuse_s0
以下为 train_heads_eye.py 原 docstring：
【眼部 token 变体】与 train_heads.py 逐行相同，唯一差别：模型换成 gaze_eye.EyeGazeModel（trunk 追加头部 crop token，见 gaze_eye.py），多一个 --eye_kw。
原 docstring：B3 MDN / B4 MCL / 我方法 flow 的训练入口（与 train_online.py 同数据、同增强、同 backbone 前向、同 batch/epoch/lr/cosine；
唯一差别 = 头与 loss，以及 grad clip 5.0）。train_online.py 保持原样不动（B1 两个 seed 用它训练）。

torchrun --nproc_per_node 8 train_heads.py --head flow --exp_name flow_s0
单卡调试： CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.run --nproc_per_node 1 train_heads.py --head flow --exp_name dbg --exp_root /tmp/gz_dbg --max_steps 200 --max_epochs 2 --limit_holdout 256

hold-out 每个 epoch 评一次：每实例 1000 个样本 -> pmf（σ_s ∈ {1,2,3} 格取均值最低者）-> NLL；cov0.10@5、点估计误差。
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
sys.path.insert(0, f'{G}/gazelle_online')
from train_online import AmpBackbone, SDGazeDataset  # noqa: E402  (顺带 import 了 common / gazelle)
from common import IN_SIZE  # noqa: E402
from gazelle.dataloader import collate_fn  # noqa: E402
import gaze_eval as GE  # noqa: E402
from gaze_models import GazeModel  # noqa: E402,F401
from gaze_eye2 import EyeGazeModel2, HiResCropDataset  # noqa: E402


@torch.no_grad()
def eval_holdout(model, ds, rank, world, bs, n_workers, n_samples, hires=False):
    model.eval()
    sub = torch.utils.data.Subset(ds, list(range(rank, len(ds), world)))
    dl = torch.utils.data.DataLoader(sub, batch_size=bs, collate_fn=collate_fn, num_workers=n_workers)
    rows = []
    gen = torch.Generator(device='cuda').manual_seed(1234 + rank)
    for batch in dl:
        imgs, bboxes, gazex, gazey = batch[:4]
        crops = batch[7].cuda(non_blocking=True) if hires else None
        tok = model.trunk.tokens(imgs.cuda(non_blocking=True), [[b] for b in bboxes], crops=crops)
        S = model.head.sample(tok, n_samples, gen=gen).float().cpu().numpy()
        for i in range(S.shape[0]):
            y = np.array([gazex[i][0], gazey[i][0]])
            rows.append(GE.holdout_instance(S[i], y))
    allrows = [None] * world
    dist.all_gather_object(allrows, rows)
    return GE.summarize([r for rr in allrows for r in rr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['fuse', 'hires'])
    ap.add_argument('--head_kw', default='{}', help='JSON，传给 FlowHead 的构造参数')
    ap.add_argument('--eye_kw', default='{}', help='JSON，传给 EyeTrunk（crop_size / margin）')
    ap.add_argument('--backbone', default='dinov2_vitl14')
    ap.add_argument('--data_path', default=f'{G}/data/gazefollow_train_sd')
    ap.add_argument('--exp_name', required=True)
    ap.add_argument('--exp_root', default=f'{G}/experiments')
    ap.add_argument('--max_epochs', type=int, default=15)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--clip', type=float, default=5.0)
    ap.add_argument('--n_workers', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--holdout_samples', type=int, default=1000)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--limit_holdout', type=int, default=0)
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
            sys.exit(f'{exp_dir} 已有 last.pt；要续训请加 --resume，否则换 exp_name')
        os.makedirs(exp_dir, exist_ok=True)
        json.dump(vars(args), open(os.path.join(exp_dir, 'args.json'), 'w'), indent=1)
    dist.barrier()

    eye_kw = json.loads(args.eye_kw)
    model = EyeGazeModel2(AmpBackbone(args.backbone), args.variant, eye_kw=eye_kw, **json.loads(args.head_kw)).cuda()
    hires = args.variant == 'hires'
    if hires:
        ds_kw = dict(margin=eye_kw.get('margin', 1.25), crop_size=eye_kw.get('crop_size', 224))
        mk_ds = lambda mode, tf: HiResCropDataset(args.data_path, mode, tf, **ds_kw)  # noqa: E731
    else:
        mk_ds = lambda mode, tf: SDGazeDataset(args.data_path, mode, tf)  # noqa: E731
    for p in model.trunk.backbone.parameters():
        p.requires_grad = False
    transform = model.trunk.backbone.get_transform(IN_SIZE)
    train_ds = mk_ds('fit', transform)
    hold_ds = mk_ds('holdout', transform)
    if args.limit_holdout:
        hold_ds.data_idxs = hold_ds.data_idxs[:args.limit_holdout]
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in model.head.parameters())
    if rank == 0:
        print(f'eye2 variant {args.variant} {args.head_kw}  可训练参数 {n_par}（头 {n_head}）  train {len(train_ds)}  hold-out {len(hold_ds)}  全局 batch {args.batch_size}', flush=True)

    sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True, seed=args.seed)
    dl = torch.utils.data.DataLoader(train_ds, batch_size=per_rank_bs, sampler=sampler, collate_fn=collate_fn, num_workers=args.n_workers,
                                     persistent_workers=True, prefetch_factor=4, drop_last=True)
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[lrank], broadcast_buffers=False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs, eta_min=1e-7)

    start = 0
    if args.resume and os.path.exists(last):
        ck = torch.load(last, map_location='cpu', weights_only=True)
        model.load_decoder_state(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start = ck['epoch'] + 1
        if rank == 0:
            print(f'resume from epoch {start}', flush=True)

    log = open(os.path.join(exp_dir, 'metrics.jsonl'), 'a') if rank == 0 else None
    for epoch in range(start, args.max_epochs):
        sampler.set_epoch(epoch)
        model.train()
        model.trunk.backbone.eval()
        t0, n_img = time.time(), 0
        for it, batch in enumerate(dl):
            imgs, bboxes, gazex, gazey = batch[0], batch[1], batch[2], batch[3]
            gaze = torch.tensor([[gx[0], gy[0]] for gx, gy in zip(gazex, gazey)], dtype=torch.float32).clamp(0, 1).cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crops = batch[8].cuda(non_blocking=True) if hires else None
            loss, logs = ddp(imgs.cuda(non_blocking=True), [[b] for b in bboxes], gaze, crops)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip)
            opt.step()
            n_img += imgs.shape[0] * world
            if rank == 0 and it % 50 == 0:
                print(f'TRAIN EPOCH {epoch} iter {it}/{len(dl)} loss {loss.item():.4f} gnorm {gn.item():.2f} {json.dumps({k: round(v, 4) for k, v in logs.items()})}  '
                      f'{n_img / (time.time() - t0):.0f} img/s', flush=True)
            if args.max_steps and it + 1 >= args.max_steps:
                break
        sched.step()

        t1 = time.time()
        res = eval_holdout(model, hold_ds, rank, world, per_rank_bs, 4, args.holdout_samples, hires=hires)
        res.update(epoch=epoch, train_time_s=round(t1 - t0, 1), eval_time_s=round(time.time() - t1, 1), loss_last=loss.item())
        if rank == 0:
            torch.save(model.decoder_state(), os.path.join(exp_dir, f'epoch_{epoch}.pt'))
            torch.save({'model': model.decoder_state(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': epoch}, last + '.tmp')
            os.replace(last + '.tmp', last)
            log.write(json.dumps(res) + '\n')
            log.flush()
            print(f"HOLD-OUT EPOCH {epoch}: NLL={res['nll_best']:.4f}(σ={res['sigma_best']:g}) cov10@5={res['cov10']:.4f} cov05@5={res['cov05']:.4f} "
                  f"pt-L2={res['l2_pt']:.4f} n={res['n']}  训练 {res['train_time_s']}s 评测 {res['eval_time_s']}s", flush=True)
        dist.barrier()
    if rank == 0:
        print('done', exp_dir, flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
