"""forecast 可行性训练（C19）。与 train_heads_eye.py 同骨架（DDP、Adam、cosine 按 epoch、clip 5、冻结 backbone），差别：VAT 数据、从 flowET_d6 初始化、两组 lr。
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port 29700 train_forecast.py --H 30 --ctx 0 --exp_name fc_h30_s
每个 epoch 末在 val 上评一次（1000 样本），打印联合 NLL（σ_s=1）与 hit@5；正式比较用 eval_forecast.py。
C20：--scheme c20（fit 32 / val 4 个节目）+ --head flow|mdn|mcl|heatmap + --eye 0|1 + --depth + --init（各头自己的 GazeFollow s0 ckpt），模型 = gaze_forecast.ForecastAny。
"""
from pathlib import Path
import argparse
import json
import math
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
sys.path.insert(0, f'{G}/setmetrics')
from train_online import AmpBackbone  # noqa: E402
from common import IN_SIZE  # noqa: E402
import gaze_eval as GE  # noqa: E402
import metrics as M  # noqa: E402
from g2_eval import pmf_from_samples  # noqa: E402
from gaze_forecast import VATForecast, ForecastModel, ForecastAny, collate  # noqa: E402


@torch.no_grad()
def eval_val(model, ds, rank, world, bs, n_samples):
    model.eval()
    sub = torch.utils.data.Subset(ds, list(range(rank, len(ds), world)))
    dl = torch.utils.data.DataLoader(sub, batch_size=bs, collate_fn=collate, num_workers=4)
    gen = torch.Generator(device='cuda').manual_seed(1234 + rank)
    rows = []
    for imgs, bn, y, inn, crops, _ in dl:
        bb, cr = [[b] for b in bn], crops.cuda(non_blocking=True)
        tok = model.tokens(imgs.cuda(non_blocking=True), bb, cr) if isinstance(model, ForecastAny) else model.trunk.tokens(imgs.cuda(non_blocking=True), bb, cr)
        po = torch.sigmoid(model.out_logit(tok)).clamp(1e-4, 1 - 1e-4).cpu().numpy()
        mask = inn > 0.5
        if not mask.any():
            S = None
        elif isinstance(model, ForecastAny):
            S = model.sample(tok[mask.cuda()], n_samples, gen=gen, steps=16, gamma=3.0, solver='heun').float().cpu().numpy().astype(np.float64)
        else:
            S = model.head.sample(tok[mask.cuda()], n_samples, gen=gen, steps=16, gamma=3.0, solver='heun').float().cpu().numpy().astype(np.float64)
        si = 0
        for i in range(len(bn)):
            if mask[i]:
                yy = y[i].numpy().astype(np.float64)
                nl = M.nll(pmf_from_samples(S[si], 1.0), yy[None])
                hy, _ = GE.samples_to_hyps(S[si], 1.0)
                d = np.hypot(*(hy - yy).T)
                rows.append((nl - math.log(1 - po[i]), float(d.min() <= 0.10), float(d[0]), 1.0))
                si += 1
            else:
                rows.append((-math.log(po[i]), np.nan, np.nan, 0.0))
    allrows = [None] * world
    dist.all_gather_object(allrows, rows)
    a = np.array([r for rr in allrows for r in rr])
    return dict(n=len(a), joint_nll=float(a[:, 0].mean()), hit5=float(np.nanmean(a[:, 1])), l2=float(np.nanmean(a[:, 2])), frac_in=float(a[:, 3].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--H', type=int, required=True)
    ap.add_argument('--ctx', default='0', help='逗号分隔的上下文帧偏移，最后一个必须是 0')
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--scheme', default='c19', choices=['c19', 'c20'])
    ap.add_argument('--head', default='flow', choices=['flow', 'mdn', 'mcl', 'heatmap'], help='c20 才用')
    ap.add_argument('--eye', type=int, default=1, help='c20 才用：1 = 眼部 token trunk')
    ap.add_argument('--init', default=f'{G}/experiments/flow_eye_d6_s0/epoch_14.pt')
    ap.add_argument('--exp_name', required=True)
    ap.add_argument('--exp_root', default=f'{G}/experiments/forecast')
    ap.add_argument('--max_epochs', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_out', type=float, default=1e-3)
    ap.add_argument('--clip', type=float, default=5.0)
    ap.add_argument('--n_workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sample_every', type=int, default=6)
    ap.add_argument('--val_H', type=int, default=30, help='epoch 末 val 评测用的 horizon（fc_h0 也看 t+30）')
    ap.add_argument('--val_samples', type=int, default=1000)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--limit_val', type=int, default=0)
    args = ap.parse_args()

    rank, world, lrank = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(lrank)
    dist.init_process_group('nccl')
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    per_rank_bs = args.batch_size // world
    ctx = [int(v) for v in args.ctx.split(',')]

    exp_dir = os.path.join(args.exp_root, args.exp_name)
    if rank == 0:
        if os.path.exists(os.path.join(exp_dir, f'epoch_{args.max_epochs - 1}.pt')):
            sys.exit(f'{exp_dir} 已训完')
        os.makedirs(exp_dir, exist_ok=True)
        json.dump(vars(args), open(os.path.join(exp_dir, 'args.json'), 'w'), indent=1)
    dist.barrier()

    if args.scheme == 'c20':
        model = ForecastAny(AmpBackbone('dinov2_vitl14'), args.head, eye=bool(args.eye), C=len(ctx), **({'depth': args.depth} if args.head == 'flow' else {})).cuda()
    else:
        model = ForecastModel(AmpBackbone('dinov2_vitl14'), C=len(ctx), depth=args.depth).cuda()
    model.load_init(args.init)
    for p in model.trunk.backbone.parameters():
        p.requires_grad = False
    tf = model.trunk.backbone.get_transform(IN_SIZE)
    train_ds = VATForecast('fit', args.H, ctx, tf, args.sample_every, scheme=args.scheme)
    val_ds = VATForecast('val', args.val_H, ctx, tf, args.sample_every, scheme=args.scheme)
    if args.limit_val:
        val_ds.items = val_ds.items[:args.limit_val]
    if rank == 0:
        print(f'H={args.H} ctx={ctx} scheme={args.scheme} head={args.head} eye={args.eye}  可训练参数 {sum(p.numel() for p in model.parameters() if p.requires_grad)}  fit {len(train_ds)}（{len(train_ds.tracks)} 轨迹） val {len(val_ds)}（{len(val_ds.tracks)} 轨迹）', flush=True)

    sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True, seed=args.seed)
    dl = torch.utils.data.DataLoader(train_ds, batch_size=per_rank_bs, sampler=sampler, collate_fn=collate, num_workers=args.n_workers,
                                     persistent_workers=True, prefetch_factor=4, drop_last=True)
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[lrank], broadcast_buffers=False)
    train_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    out_p = [p for n, p in train_params if n.startswith('out_')]
    rest_p = [p for n, p in train_params if not n.startswith('out_')]
    opt = torch.optim.Adam([{'params': rest_p, 'lr': args.lr}, {'params': out_p, 'lr': args.lr_out}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs, eta_min=1e-7)

    log = open(os.path.join(exp_dir, 'metrics.jsonl'), 'a') if rank == 0 else None
    for epoch in range(args.max_epochs):
        sampler.set_epoch(epoch)
        model.train()
        model.trunk.backbone.eval()
        t0, n_img = time.time(), 0
        for it, (imgs, bn, y, inn, crops, _) in enumerate(dl):
            opt.zero_grad(set_to_none=True)
            loss, logs = ddp(imgs.cuda(non_blocking=True), [[b] for b in bn], y.cuda(non_blocking=True), inn.cuda(non_blocking=True), crops.cuda(non_blocking=True))
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for _, p in train_params], args.clip)
            opt.step()
            n_img += imgs.shape[0] * world
            if rank == 0 and it % 20 == 0:
                print(f'TRAIN EPOCH {epoch} iter {it}/{len(dl)} loss {loss.item():.4f} gnorm {gn.item():.2f} {json.dumps({k: round(v, 4) for k, v in logs.items()})}  {n_img / (time.time() - t0):.0f} inst/s', flush=True)
            if args.max_steps and it + 1 >= args.max_steps:
                break
        sched.step()
        t1 = time.time()
        res = eval_val(model, val_ds, rank, world, per_rank_bs, args.val_samples)
        res.update(epoch=epoch, train_time_s=round(t1 - t0, 1), eval_time_s=round(time.time() - t1, 1))
        if rank == 0:
            torch.save(model.decoder_state(), os.path.join(exp_dir, f'epoch_{epoch}.pt'))
            log.write(json.dumps(res) + '\n')
            log.flush()
            print(f"VAL EPOCH {epoch}: joint-NLL={res['joint_nll']:.4f} hit5={res['hit5']:.4f} L2={res['l2']:.4f} frac_in={res['frac_in']:.3f} n={res['n']}  训练 {res['train_time_s']}s 评测 {res['eval_time_s']}s", flush=True)
        dist.barrier()
    if rank == 0:
        print('done', exp_dir, flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
