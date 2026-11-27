"""ShijianDeng/gazefollow 的 train.parquet -> 官方格式的 GazeFollow 训练集（图片文件 + train_preprocessed.json）。

python -I -u build_gazefollow_train_from_sd.py [--limit N]

- 该镜像一行一张图、一个眼点、一个注视点，没有头框 -> 用「眼睛为中心的正方形」补框（边长 = frac*sqrt(W*H)，
  见 results/headbox/eval_with_boxes.log：frac=0.12 在官方 ckpt 上与真值框几乎等价）。
- 同一张图（图片字节 sha1 相同）的多行合并成一个 frame 的多个 head，特征只抽一次，hold-out 也按 frame 切，避免泄漏。
- hold-out：sha1 前 8 位 % 100 < HOLD_PCT（按内容确定性切分，与行序无关）。G2 预注册规定调参只能用训练 hold-out。
- 注意：图像是镜像的缩放版（大量图短边 512），不是原始文件；坐标全是归一化的，所以不受影响。
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter

import pyarrow.parquet as pq
from PIL import Image

G = str(Path(__file__).resolve().parents[1])
PARQUET = f'{G}/data/hf_shijiandeng_train/train.parquet'
OUT = f'{G}/data/gazefollow_train_sd'
HOLD_PCT = 3
RE_XY = re.compile(r'\(x, y\): \(([0-9.eE+-]+), ([0-9.eE+-]+)\)')
EXT = {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp'}


def eye_square(eye, W, H, frac):
    s = frac * (W * H) ** 0.5
    cx, cy = eye[0] * W, eye[1] * H
    return [max(cx - s / 2, 0.0), max(cy - s / 2, 0.0), min(cx + s / 2, W), min(cy + s / 2, H)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frac', type=float, default=0.12)
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 行（smoke）')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    img_dir = os.path.join(a.out, 'train_sd')
    os.makedirs(img_dir, exist_ok=True)
    pf = pq.ParquetFile(PARQUET)
    frames, by_sha = [], {}
    n_rows, fmt = 0, Counter()
    for g in range(pf.metadata.num_row_groups):
        for r in pf.read_row_group(g).to_pylist():
            if a.limit and n_rows >= a.limit:
                break
            n_rows += 1
            assert len(r['images']) == 1 and len(r['texts']) == 1, (n_rows, len(r['images']), len(r['texts']))
            t = r['texts'][0]
            assert t['source'] == 'gazefollow', t['source']
            eye = [float(v) for v in RE_XY.search(t['user']).groups()]
            gaze = [float(v) for v in RE_XY.search(t['assistant']).groups()]
            assert all(0.0 <= v <= 1.0 for v in eye + gaze), (n_rows, eye, gaze)
            raw = r['images'][0]['bytes']
            sha = hashlib.sha1(raw).hexdigest()
            if sha not in by_sha:
                im = Image.open(io.BytesIO(raw))
                W, H = im.size
                ext = EXT[im.format]
                fmt[im.format] += 1
                path = f'train_sd/{len(frames):06d}_{sha[:10]}.{ext}'
                with open(os.path.join(a.out, path), 'wb') as f:
                    f.write(raw)
                by_sha[sha] = len(frames)
                frames.append({'path': path, 'width': W, 'height': H, 'sha1': sha, 'heads': [],
                               'holdout': int(sha[:8], 16) % 100 < HOLD_PCT})
            fr = frames[by_sha[sha]]
            W, H = fr['width'], fr['height']
            box = eye_square(eye, W, H, a.frac)
            fr['heads'].append({'bbox': box, 'bbox_norm': [box[0] / W, box[1] / H, box[2] / W, box[3] / H],
                                'eye_norm': eye, 'inout': 1,
                                'gazex': [gaze[0] * W], 'gazey': [gaze[1] * H],
                                'gazex_norm': [gaze[0]], 'gazey_norm': [gaze[1]]})
        if a.limit and n_rows >= a.limit:
            break
        if g % 100 == 0:
            print(f'row group {g}/{pf.metadata.num_row_groups}  rows {n_rows}  unique frames {len(frames)}', flush=True)

    json.dump(frames, open(os.path.join(a.out, 'train_preprocessed.json'), 'w'))
    hold = [f for f in frames if f['holdout']]
    nh = Counter(len(f['heads']) for f in frames)
    print(f'rows {n_rows}  unique frames {len(frames)}  (重复图合并 {n_rows - len(frames)} 行)  格式 {dict(fmt)}')
    print(f'每帧 head 数分布 {dict(sorted(nh.items()))}')
    print(f'hold-out: {len(hold)} frames / {sum(len(f["heads"]) for f in hold)} heads；train: {len(frames) - len(hold)} frames')
    Ws = Counter((f['width'], f['height']) for f in frames)
    print('最常见分辨率', Ws.most_common(5))


if __name__ == '__main__':
    main()
