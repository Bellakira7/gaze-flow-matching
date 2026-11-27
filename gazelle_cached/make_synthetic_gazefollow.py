import os
import sys

import numpy as np
from PIL import Image, ImageDraw

out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/gf_synth'
rng = np.random.RandomState(0)
W, H = 640, 480


def make_img(path):
    arr = rng.randint(0, 255, (H, W, 3)).astype(np.uint8) // 3
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    for _ in range(6):
        x0, y0 = rng.randint(0, W - 80), rng.randint(0, H - 80)
        d.rectangle([x0, y0, x0 + rng.randint(30, 80), y0 + rng.randint(30, 80)], fill=tuple(int(c) for c in rng.randint(80, 255, 3)))
    os.makedirs(os.path.dirname(os.path.join(out, path)), exist_ok=True)
    im.save(os.path.join(out, path), quality=90)


def head():
    cx, cy, s = rng.randint(100, W - 100), rng.randint(80, H - 120), rng.randint(40, 80)
    return cx, cy, s, [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]


rows = []
for i in range(40):
    path = f'train/00000000/{i:08d}.jpg'
    make_img(path)
    for k in range(2 if i % 5 == 0 else 1):
        cx, cy, s, bb = head()
        gx, gy = np.clip(rng.rand(2), 0.02, 0.98)
        inout = 0 if (i == 3) else (-1 if i == 7 else 1)
        rows.append([path, k, bb[0], bb[1], s, s, cx / W, cy / H, gx, gy, *bb, inout, 'synth', 'meta'])
with open(os.path.join(out, 'train_annotations_release.txt'), 'w') as f:
    for r in rows:
        f.write(','.join(str(v) for v in r) + '\n')

rows = []
for i in range(12):
    path = f'test2/00000000/{i:08d}.jpg'
    make_img(path)
    for k in range(2 if i % 4 == 0 else 1):
        cx, cy, s, bb = head()
        centers = rng.rand(2 if i % 2 == 0 else 1, 2) * 0.8 + 0.1
        for a in range(10):
            gx, gy = np.clip(centers[a % len(centers)] + rng.randn(2) * 0.03, 0.0, 1.0)
            rows.append([path, k, bb[0], bb[1], s, s, cx / W, cy / H, gx, gy, *bb, 'synth', 'meta'])
with open(os.path.join(out, 'test_annotations_release.txt'), 'w') as f:
    for r in rows:
        f.write(','.join(str(v) for v in r) + '\n')
print('wrote', out)
