"""把 HF 镜像 vikhyatk/gazefollow 的 test parquet 转成官方 Gaze-LLE 的 `test_preprocessed.json` + 图像文件。

口径严格照 gazelle/data_prep/preprocess_gazefollow.py 的 TEST 部分：
  - 一个 head = 一个实例，gaze 为该实例所有标注者的 (x, y) 列表（原文件顺序）
  - bbox 先换成像素，再 swap(min>max) + 夹到图内，bbox_norm = bbox / (width, height)
  - width/height 取真实图像尺寸；inout 恒为 1
  - 同一张图上的多个 head 合并到同一个 frame（官方按 path 聚合）
图像字节**原样落盘**（不重编码），避免引入额外的特征漂移。

用法: python -I build_gazefollow_test_from_hf.py <test.parquet> <out_dir(必须为空或不存在)>
"""
import collections
import hashlib
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def main(pq_path, out_dir):
    if os.path.exists(out_dir) and os.listdir(out_dir):
        sys.exit(f'{out_dir} 非空，拒绝覆盖')
    os.makedirs(os.path.join(out_dir, 'test2_hf'), exist_ok=True)
    pf = pq.ParquetFile(pq_path)

    frames, key2frame = [], {}
    n_inst, n_annot, swapped, clamped, gaze_oob, path_kinds = 0, [], 0, 0, 0, collections.Counter()
    sizes = collections.Counter()
    for rg in range(pf.metadata.num_row_groups):
        for row in pf.read_row_group(rg, columns=['image', 'gazes']).to_pylist():
            data, src_path = row['image']['bytes'], row['image'].get('path')
            sha = hashlib.sha1(data).hexdigest()
            if sha not in key2frame:
                im = Image.open(io.BytesIO(data))
                width, height = im.size
                fmt = (im.format or 'JPEG').lower()
                assert fmt in ('jpeg', 'png'), fmt
                rel = f'test2_hf/{len(frames):05d}_{sha[:10]}.{"jpg" if fmt == "jpeg" else "png"}'
                with open(os.path.join(out_dir, rel), 'wb') as f:
                    f.write(data)
                path_kinds['has_src_path' if src_path else 'no_src_path'] += 1
                key2frame[sha] = len(frames)
                sizes[(width, height)] += 1
                frames.append({'path': rel, 'heads': [], 'width': width, 'height': height, 'src_path': src_path})
            fr = key2frame[sha]
            frame = frames[fr]
            width, height = frame['width'], frame['height']

            g = row['gazes']
            assert len(g) > 0
            hb = [(a['head_bbox']['xmin'], a['head_bbox']['ymin'], a['head_bbox']['xmax'], a['head_bbox']['ymax']) for a in g]
            assert all(b == hb[0] for b in hb), '同一实例内各标注的 head_bbox 不一致'  # 官方也有同样的 assert
            eyes = {(a['eye']['x'], a['eye']['y']) for a in g}
            assert len(eyes) == 1, '同一实例内 eye 不一致'

            xmin, ymin, xmax, ymax = [v * f for v, f in zip(hb[0], (width, height, width, height))]
            if xmin > xmax:
                xmin, xmax = xmax, xmin
                swapped += 1
            if ymin > ymax:
                ymin, ymax = ymax, ymin
                swapped += 1
            c = (max(xmin, 0), max(ymin, 0), min(xmax, width), min(ymax, height))
            clamped += int(c != (xmin, ymin, xmax, ymax))
            xmin, ymin, xmax, ymax = c

            gx = [a['gaze']['x'] for a in g]
            gy = [a['gaze']['y'] for a in g]
            gaze_oob += sum((x < 0 or x > 1 or y < 0 or y > 1) for x, y in zip(gx, gy))
            frame['heads'].append({
                'bbox': [xmin, ymin, xmax, ymax],
                'bbox_norm': [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)],
                'gazex': [x * float(width) for x in gx],
                'gazey': [y * float(height) for y in gy],
                'gazex_norm': [float(x) for x in gx],
                'gazey_norm': [float(y) for y in gy],
                'inout': 1,
                'num_annot': len(gx),
                'head_id': len(frame['heads']),
                'eye_norm': list(next(iter(eyes))),
            })
            n_inst += 1
            n_annot.append(len(gx))

    for fr in frames:
        fr['num_heads'] = len(fr['heads'])
    json.dump(frames, open(os.path.join(out_dir, 'test_preprocessed.json'), 'w'))
    na = np.array(n_annot)
    print(f'frames {len(frames)} | instances {n_inst} | multi-person frames {sum(f["num_heads"] > 1 for f in frames)}')
    print(f'annotations/instance: min {na.min()} median {int(np.median(na))} max {na.max()} | total {na.sum()} | hist {dict(sorted(collections.Counter(n_annot).items()))}')
    print(f'bbox swapped {swapped}, clamped {clamped}, gaze outside [0,1]: {gaze_oob}')
    print(f'src path available: {dict(path_kinds)} | distinct image sizes {len(sizes)}, top {sizes.most_common(5)}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
