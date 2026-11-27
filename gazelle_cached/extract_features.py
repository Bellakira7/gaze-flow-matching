import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FEAT_DIMS, IN_SIZE, LocalDinoV2Backbone  # noqa: E402


class ImgDS(torch.utils.data.Dataset):
    def __init__(self, rows, paths, root, transform, flip):
        self.rows, self.paths, self.root, self.transform, self.flip = rows, paths, root, transform, flip

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        r = self.rows[k]
        img = Image.open(os.path.join(self.root, self.paths[r])).convert('RGB')
        out = {'orig': self.transform(img)}
        if self.flip:
            out['flip'] = self.transform(TF.hflip(img))
        return r, out


def collate(batch):
    rows = [b[0] for b in batch]
    views = {v: torch.stack([b[1][v] for b in batch]) for v in batch[0][1]}
    return rows, views


def worker(rank, world, args, n_gpu):
    dev = torch.device(f'cuda:{rank % n_gpu}')
    torch.set_num_threads(2)
    paths = json.load(open(os.path.join(args.out_dir, f'{args.split}_index.json')))
    bb = LocalDinoV2Backbone(args.backbone).to(dev).eval()
    transform = bb.get_transform(IN_SIZE)
    views = ['orig', 'flip'] if args.flip else ['orig']
    mm = {v: np.load(os.path.join(args.out_dir, f'{args.split}_{v}.npy'), mmap_mode='r+') for v in views}

    my_rows = list(range(rank, len(paths), world))
    prog_file = os.path.join(args.out_dir, f'progress_{args.split}_rank{rank}.txt')
    done = int(open(prog_file).read()) if (args.resume and os.path.exists(prog_file)) else 0
    ds = ImgDS(my_rows[done:], paths, args.data_path, transform, args.flip)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.n_workers, collate_fn=collate)

    with torch.inference_mode():
        for step, (rows, imgs) in enumerate(dl):
            for v in views:
                f = bb(imgs[v].to(dev, non_blocking=True))
                mm[v][rows] = f.half().cpu().numpy()
            done += len(rows)
            if step % 20 == 0:
                for v in views:
                    mm[v].flush()
                open(prog_file, 'w').write(str(done))
                if rank == 0:
                    print(f'[rank0] {done}/{len(my_rows)}', flush=True)
    for v in views:
        mm[v].flush()
    open(prog_file, 'w').write(str(done))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', required=True, help='GazeFollow data_new 目录（含 *_preprocessed.json）')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--split', default='test', choices=['train', 'test'])
    p.add_argument('--backbone', default='dinov2_vitl14', choices=list(FEAT_DIMS))
    p.add_argument('--flip', action='store_true', help='同时抽水平翻转视图（训练集用）')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_workers', type=int, default=8)
    p.add_argument('--n_proc', type=int, default=0, help='0 = 用满可见 GPU')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='只抽前 N 张（smoke）')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frames = json.load(open(os.path.join(args.data_path, f'{args.split}_preprocessed.json')))
    paths = [f['path'] for f in frames]
    if args.limit:
        paths = paths[:args.limit]
    idx_file = os.path.join(args.out_dir, f'{args.split}_index.json')
    if args.resume and os.path.exists(idx_file):
        assert json.load(open(idx_file)) == paths, 'resume 时 index 与当前数据不一致'
    json.dump(paths, open(idx_file, 'w'))

    C, (H, W) = FEAT_DIMS[args.backbone], (IN_SIZE[0] // 14, IN_SIZE[1] // 14)
    for v in (['orig', 'flip'] if args.flip else ['orig']):
        f = os.path.join(args.out_dir, f'{args.split}_{v}.npy')
        if not (args.resume and os.path.exists(f)):
            np.lib.format.open_memmap(f, mode='w+', dtype=np.float16, shape=(len(paths), C, H, W)).flush()

    n_gpu = torch.cuda.device_count()
    world = args.n_proc or n_gpu
    print(f'{len(paths)} images, {world} procs on {n_gpu} GPUs, feat [{C},{H},{W}] fp16', flush=True)
    mp.spawn(worker, args=(world, args, n_gpu), nprocs=world, join=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
