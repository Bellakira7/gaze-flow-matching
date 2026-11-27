#!/usr/bin/env python
"""CP1 / CP2 共用的 val 切分（prereg/CP1_CP2.prereg.md §1）。

  /usr/local/bin/python -I tools/cp_split.py [--data data/vqa/gazefollow_val.jsonl] [--out data/vqa/cp2_split]

规则（逐字照 §1，只用 frame_idx，不接触任何标注或预测）:
  h = int(hashlib.sha1(f'cp2|{frame_idx}'.encode()).hexdigest()[:8], 16) % 2
  h == 0 -> val_sel（挑 checkpoint）    h == 1 -> val_gate（判定）
按帧切，同一帧的多个人不跨组。

产物（--out 目录，已存在且内容不同会拒绝覆盖）:
  val_sel.ids / val_gate.ids   每行一个样本 id，保持 val 文件原序
  split.json                   val 的 sha256、两组的人数/帧数、两个 .ids 的 sha256

校验（任何一项不过就非零退出）:
  1. val 的 sha256 == prereg §1 里冻结的值
  2. 规模 == prereg §0 披露的值：val_sel 1705 人 / 1606 帧，val_gate 1675 人 / 1564 帧
  3. 两组的人、帧都零重叠；两组并集 == 全 val（prereg §5.4）
  4. 与 train 图像零重叠沿用已有核对，这里不重做
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

G = str(Path(__file__).resolve().parents[1])
VAL_SHA256 = 'b4bbe926e49a5148ba468b30b77ea8dd626f516a7e2da0959b5765a69ee88865'  # prereg §1
EXPECT = {'val_sel': (1705, 1606), 'val_gate': (1675, 1564)}  # (人, 帧)，prereg §0


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def group_of(frame_idx):
    h = int(hashlib.sha1(f'cp2|{frame_idx}'.encode()).hexdigest()[:8], 16) % 2
    return 'val_sel' if h == 0 else 'val_gate'


def load_split(split_dir, val_path=None):
    """供 cp1_r16.py 用：-> {'val_sel': [id...], 'val_gate': [id...]}；顺带核对 split.json 里记的 val / ids 的 sha256。"""
    meta = json.load(open(os.path.join(split_dir, 'split.json')))
    if val_path is not None and sha256_file(val_path) != meta['val_sha256']:
        sys.exit(f'val 文件 sha256 与 {split_dir}/split.json 记录的不一致，拒绝使用这份切分')
    out = {}
    for g in ('val_sel', 'val_gate'):
        p = os.path.join(split_dir, g + '.ids')
        if sha256_file(p) != meta[g]['ids_sha256']:
            sys.exit(f'{p} 的 sha256 与 split.json 记录的不一致（被改过？）')
        out[g] = [l.rstrip('\n') for l in open(p) if l.strip()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=f'{G}/data/vqa/gazefollow_val.jsonl')
    ap.add_argument('--out', default=f'{G}/data/vqa/cp2_split')
    a = ap.parse_args()

    val_sha = sha256_file(a.data)
    if val_sha != VAL_SHA256:
        sys.exit(f'val sha256 {val_sha} != prereg 冻结值 {VAL_SHA256}；这不是预注册的那份 val，不切')
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    ids = [r['id'] for r in rows]
    assert len(set(ids)) == len(ids), 'val 里有重复 id'

    groups = {'val_sel': [], 'val_gate': []}
    frames = {'val_sel': set(), 'val_gate': set()}
    for r in rows:
        g = group_of(r['frame_idx'])
        groups[g].append(r['id'])
        frames[g].add(r['frame_idx'])

    fails = []
    for g, (n_exp, f_exp) in EXPECT.items():
        n, f = len(groups[g]), len(frames[g])
        ok = (n, f) == (n_exp, f_exp)
        print(f'  {"PASS" if ok else "FAIL"} {g}: {n} 人 / {f} 帧（prereg 披露 {n_exp} / {f_exp}）')
        if not ok:
            fails.append(f'{g} 规模')
    both_p = set(groups['val_sel']) & set(groups['val_gate'])
    both_f = frames['val_sel'] & frames['val_gate']
    union_ok = set(groups['val_sel']) | set(groups['val_gate']) == set(ids) and len(groups['val_sel']) + len(groups['val_gate']) == len(ids)
    for ok, msg in ((not both_p, f'人零重叠（重叠 {len(both_p)}）'), (not both_f, f'帧零重叠（重叠 {len(both_f)}）'),
                    (union_ok, f'两组并集 == 全 val（{len(ids)} 人）')):
        print(f'  {"PASS" if ok else "FAIL"} {msg}')
        if not ok:
            fails.append(msg)
    if fails:
        sys.exit('切分校验未通过，不落盘: ' + '; '.join(fails))

    os.makedirs(a.out, exist_ok=True)
    meta = {'rule': "h = int(sha1(f'cp2|{frame_idx}').hexdigest()[:8], 16) % 2; 0 -> val_sel, 1 -> val_gate",
            'prereg': 'prereg/CP1_CP2.prereg.md §1', 'val': os.path.abspath(a.data), 'val_sha256': val_sha, 'n_val': len(ids)}
    for g in ('val_sel', 'val_gate'):
        p = os.path.join(a.out, g + '.ids')
        txt = ''.join(i + '\n' for i in groups[g])
        if os.path.exists(p) and open(p).read() != txt:
            sys.exit(f'{p} 已存在且内容不同，拒绝覆盖（预注册之后切分不能变）')
        with open(p, 'w') as f:
            f.write(txt)
        meta[g] = {'n_people': len(groups[g]), 'n_frames': len(frames[g]), 'ids_sha256': sha256_file(p)}
    json.dump(meta, open(os.path.join(a.out, 'split.json'), 'w'), indent=1)
    print('->', a.out)


if __name__ == '__main__':
    main()
