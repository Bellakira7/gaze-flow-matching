#!/usr/bin/env python
"""Gaze-R1 评测脚本：GazeFollow / VideoAttentionTarget（/ ChildPlay）上的 AUC、L2、in/out AP。

三个子命令（/usr/local/bin/python；输入是 tools/convert_to_vqa.py 产出的 jsonl）:

  infer     用 HF transformers 跑 Qwen3-VL（可挂 LoRA adapter），把每条样本的回答写进 preds.jsonl
              python eval_gaze.py infer --data data/vqa/gazefollow_test.jsonl --model ckpt/Qwen3-VL-4B-Instruct \\
                  [--adapter <ckpt dir>] --out results/x/preds.jsonl [--gpus 0,1,2,3] [--max-pixels 262144] \\
                  [--num-samples 16 --temperature 1.0 --seed 0]
              采样可复现：每条样本单独 generate，生成前用 row_seed(--seed, 该条 id) 重置 RNG，所以某条的 K 个采样只取决于
              (seed, id, 模型, 输入)，与 --gpus 分片数、batch 组成、断点续跑切点无关（prereg/CP1_CP2.prereg.md §5.2）。
              greedy 仍按 batch 一起跑，路径与加种子之前逐位相同。
  score     从 preds.jsonl 算指标（纯 CPU，可反复重算，换 KDE 带宽不用重跑推理）
              python eval_gaze.py score --data data/vqa/gazefollow_test.jsonl --preds results/x/preds.jsonl \\
                  --out results/x/metrics.json [--dump-instances results/x/inst.jsonl]
  selfcheck 校验指标代码本身（**开跑任何评测之前先过**，见下）

------------------------------------------------------------------------------------------------------------------
指标口径（与 Gaze-LLE 官方 gazelle/utils.py 逐项对应；坐标全部是 [0,1] 归一化坐标，L2 不乘图像尺寸）
  GazeFollow（每人约 10 个标注）
    Avg L2 : ‖pred − mean(标注)‖        Min L2 : min_i ‖pred − 标注_i‖
    AUC    : 在**原图分辨率**上，目标图 = 各标注点所在像素置 1（int(x*W)，夹到 W-1；重复像素只算一次），其余 0；
             热图 64×64 双线性插值到 (H,W) 后算 ROC-AUC
  VAT / ChildPlay（单标注，GT 在画面内的样本才算 AUC 和 L2）
    L2     : ‖pred − gt‖
    AUC    : 在 64×64 热图上，目标图 = 以 gt*64 为中心的 (6σ+1) 方窗（σ=3，官方实现里右/下界夹到 63），ROC-AUC
    AP     : 全部样本（含画面外）上 average_precision(inout_gt, P(in))
  热图：模型输出的是点，不是热图 -> 由 K 个采样点做各向同性高斯 KDE 得到 64×64 热图（格子中心 (i+.5)/64，
        与双线性插值 align_corners=False 的像素中心一致）。K=1（只有 greedy）时就是一个高斯。
        带宽 --kde-sigma-cells（默认 3 格，与 VAT 的 GT 高斯同宽）是协议里的自由参数：**只能在 hold-out 上选，不能在 test 上调**。
        官方热图模型的 L2 取 argmax 格子的**左上角** idx/64（没有 +0.5），所以热图法在口径上略吃亏；
        我们输出连续坐标，没有这个离散化偏差（W6 主表需要说明）。
  P(in)  : 用 `<io>` 之后那一个 token 上 {in, out} 的概率（--inout-conf io，对应 TODO）；
           另存 `<dir>` 之后第一个 token 上 {数字, none} 的概率（dir 法）。
           ⚠️ 答案顺序是 dir→pt→io，画面外样本的 dir 就是 none，所以 io 位置的概率几乎被前文决定（接近 0/1，AP 的分辨力会偏低）；
           dir 法是真正做出 in/out 决定的位置。两个都报，主指标默认 io。
  解析失败 / 回答 out 却没给点：点按**画面中心 (0.5,0.5)** 计分，单独统计比例，不丢样本。
  预测点超出 [0,1000]：夹到边界并计数（只会有利于模型，所以计数要看）。
  K>1 采样时额外报：greedy、best-of-k（按每个采样的 L2 取最小）、avg-of-k（k 个采样 L2 的均值）、K 点 KDE 的 AUC。
------------------------------------------------------------------------------------------------------------------
selfcheck 做四件事（任何一件不过就返回非零）:
  1. 合成用例：解析器边界、AUC 秩公式 vs sklearn、L2、KDE、AP
  2. 与官方 gazelle.utils 逐实例对拍：同一批热图喂我的实现和官方实现，数值逐位一致
  3. 回放官方 Gaze-LLE ViT-L checkpoint 已落盘的热图（results/repro_test/full.npz、results/vat_official/*.npz）
     经我的评分代码，必须复现官方数字（GazeFollow 0.958 / 0.099 / 0.041；VAT 0.937 / 0.103 / 0.903，论文口径容差 0.001）
  4. 管线冒烟：真值答案喂回去（解析失败应为 0、L2 只剩量化误差、AP=1）；全解析失败 = 全部按中心点计分
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter

import numpy as np

G = str(Path(__file__).resolve().parents[1])
OFFICIAL_UTILS = str(Path(__file__).resolve().parents[1] / 'third_party/gazelle/gazelle/utils.py')
CENTER = (0.5, 0.5)
RES = 64  # 热图分辨率（与 Gaze-LLE / VAT 协议一致）
VERSION = 'eval_gaze/1'

# ------------------------------------------------------------------------------------------------------------------
# 解析（与 tools/swift_gaze_rewards.py 的 parse 同一套语法；那边依赖 swift，这里不能 import，改一边要改另一边）
# ------------------------------------------------------------------------------------------------------------------
_FULL = re.compile(
    r'^\s*<dir>\s*(?P<dir>none|[-+]?\d+(?:\.\d+)?)\s*</dir>\s*'
    r'(?:<pt>\s*(?P<x>[-+]?\d+(?:\.\d+)?)\s*,\s*(?P<y>[-+]?\d+(?:\.\d+)?)\s*</pt>\s*)?'
    r'<io>\s*(?P<io>in|out)\s*</io>\s*$', re.IGNORECASE)


def parse_completion(text):
    """-> dict(ok, io, theta, x, y)；x,y 是 0-1000 相对坐标原值。语法违规（含 in 缺点、out 带点）一律 ok=False。"""
    bad = {'ok': False, 'io': None, 'theta': None, 'x': None, 'y': None}
    if not isinstance(text, str):
        return bad
    m = _FULL.match(text.strip())
    if m is None:
        return bad
    io = m.group('io').lower()
    x, y = m.group('x'), m.group('y')
    if (io == 'in') != (x is not None):
        return bad
    d = m.group('dir').lower()
    return {'ok': True, 'io': io, 'theta': None if d == 'none' else float(d) % 360.0,
            'x': None if x is None else float(x), 'y': None if y is None else float(y)}


def completion_to_point(text):
    """-> (pt|None, status)。status: ok / clamped / pred_out / parse_fail。pt 是 [0,1] 归一化坐标。"""
    p = parse_completion(text)
    if not p['ok']:
        return None, 'parse_fail'
    if p['io'] == 'out':
        return None, 'pred_out'
    x, y = p['x'] / 1000.0, p['y'] / 1000.0
    st = 'ok' if (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) else 'clamped'
    return (min(1.0, max(0.0, x)), min(1.0, max(0.0, y))), st


# ------------------------------------------------------------------------------------------------------------------
# 指标
# ------------------------------------------------------------------------------------------------------------------
def l2_stats(pt, gx, gy):
    """pt=(x,y) 归一化；gx/gy 为标注列表 -> (avg_l2, min_l2)。官方口径：归一化坐标下直接算。"""
    gx, gy = np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)
    avg = math.hypot(pt[0] - gx.mean(), pt[1] - gy.mean())
    mn = float(np.hypot(pt[0] - gx, pt[1] - gy).min())
    return avg, mn


def kde_heatmap(pts, sigma_cells=3.0, res=RES):
    """各向同性高斯 KDE：格子中心 (i+.5)/res，带宽 sigma_cells/res。float64，避免远处值在 float32 下下溢成大片并列。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    c = (np.arange(res) + 0.5) / res
    s2 = 2.0 * (sigma_cells / res) ** 2
    dx2 = (c[None, :] - pts[:, 0:1]) ** 2  # K x res（列方向 = x）
    dy2 = (c[None, :] - pts[:, 1:2]) ** 2  # K x res（行方向 = y）
    return (np.exp(-dy2 / s2)[:, :, None] * np.exp(-dx2 / s2)[:, None, :]).sum(0)


def _auc_rank(scores, pos_idx):
    """ROC-AUC，正类在 flat 下标 pos_idx（唯一）。Mann-Whitney U，并列算 0.5，与 sklearn roc_auc_score 等价。"""
    scores = scores.ravel()
    mask = np.ones(scores.size, dtype=bool)
    mask[pos_idx] = False
    neg = np.sort(scores[mask])
    pos = scores[pos_idx]
    lo = np.searchsorted(neg, pos, 'left')
    hi = np.searchsorted(neg, pos, 'right')
    return float(((lo + 0.5 * (hi - lo)).mean()) / neg.size)


def resize_bilinear(hm, h, w):
    """与官方 gazefollow_auc 相同的 torch 双线性插值（align_corners=False）；保持输入 dtype（float32/float64）。"""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(hm))[None, None]
    return torch.nn.functional.interpolate(t, (h, w), mode='bilinear').squeeze().numpy()


def auc_gazefollow(hm, gx, gy, h, w):
    pos = set()
    for x, y in zip(gx, gy):
        if x >= 0:
            xi, yi = int(x * float(w)), int(y * float(h))
            pos.add(min(yi, h - 1) * w + min(xi, w - 1))
    return _auc_rank(resize_bilinear(hm, h, w), np.fromiter(pos, dtype=np.int64))


def auc_vat(hm, gx, gy):
    assert hm.shape == (RES, RES)
    sigma = 3
    x, y = gx * RES, gy * RES
    ul = [max(0, int(x - 3 * sigma)), max(0, int(y - 3 * sigma))]
    br = [min(int(x + 3 * sigma + 1), RES - 1), min(int(y + 3 * sigma + 1), RES - 1)]  # 官方实现的 -1 夹取，保持一致
    tgt = np.zeros((RES, RES), dtype=bool)
    tgt[ul[1]:br[1], ul[0]:br[0]] = True
    return _auc_rank(hm, np.flatnonzero(tgt.ravel()))


def ap_score(y_true, y_score):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_score))


# ------------------------------------------------------------------------------------------------------------------
# 实例构造与打分
# ------------------------------------------------------------------------------------------------------------------
PROTO = {'gazefollow': 'gf', 'vat': 'vat', 'childplay': 'vat'}  # ChildPlay 沿用 VAT 协议（单标注、64×64 方窗 AUC、AP in/out）


def row_seed(seed, row_id):
    """逐样本确定性种子：由 --seed 与该条的 id 共同决定。
    用 sha256 而不是 hash()（str 的 hash 每个进程加盐，跨进程/跨分片不一致）。取前 15 个十六进制位 < 2**60，torch.manual_seed 可直接吃。"""
    return int(hashlib.sha256(f'{seed}|{row_id}'.encode()).hexdigest()[:15], 16)


def dataset_of(row_id):
    return row_id.split('/')[0]


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def make_instance(row, rec):
    """row: convert_to_vqa 的一行；rec: preds.jsonl 的一行（greedy[, samples]）。"""
    ds = dataset_of(row['id'])
    inst = {'id': row['id'], 'ds': ds, 'proto': PROTO[ds], 'W': row['width'], 'H': row['height'], 'inout': int(row['inout']),
            'gx': list(row['gt_xs_norm']) if row['inout'] else [], 'gy': list(row['gt_ys_norm']) if row['inout'] else []}
    g = rec['greedy']
    inst['g_pt'], inst['g_status'] = completion_to_point(g['text'])
    inst['samples'] = [completion_to_point(s['text']) for s in rec.get('samples', [])]
    # in/out 置信度：token 概率；缺失时退回解析出的硬标签，再不行给 0.5
    p = parse_completion(g['text'])
    hard = None if not p['ok'] else (1.0 if p['io'] == 'in' else 0.0)
    inst['p_in'], inst['p_src'] = {}, {}
    for k in ('io', 'dir'):
        v = g.get('p_in_' + k)
        inst['p_in'][k], inst['p_src'][k] = (v, 'token') if v is not None else ((hard, 'hard') if hard is not None else (0.5, 'neutral'))
    inst['heatmap'] = None
    return inst


def _sp(pt):
    return CENTER if pt is None else pt


def _auc_job(job):
    proto, hm, pts, sigma, gx, gy, W, H = job
    import torch
    torch.set_num_threads(1)
    if hm is not None:
        hm = hm.astype(np.float32) if hm.dtype == np.float16 else hm
    else:
        hm = kde_heatmap(pts, sigma)
    return auc_gazefollow(hm, gx, gy, H, W) if proto == 'gf' else auc_vat(hm, gx[0], gy[0])


def _mean_sem(v):
    v = np.asarray(v, dtype=np.float64)
    return (float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))) if len(v) > 1 else ((float(v.mean()), 0.0) if len(v) else (float('nan'), 0.0))


def score_instances(insts, kde_sigma=3.0, k_list=(2, 4, 8, 16), n_proc=None):
    """-> (metrics: {dataset: {...}}, per_instance: list[dict])。"""
    from multiprocessing import Pool
    ks = [len(i['samples']) for i in insts]
    K = min(ks) if ks else 0
    assert K == max(ks), f'各样本的采样数不一致: {Counter(ks)}'
    per = []
    jobs, job_of = [], {}
    for n, i in enumerate(insts):
        r = {'id': i['id'], 'ds': i['ds'], 'inout': i['inout'], 'g_status': i['g_status']}
        if i['inout']:
            r['avg_l2'], r['min_l2'] = l2_stats(_sp(i['g_pt']), i['gx'], i['gy'])
            if K:
                sl = [l2_stats(_sp(p), i['gx'], i['gy']) for p, _ in i['samples']]
                r['s_avg_l2'], r['s_min_l2'] = [a for a, _ in sl], [b for _, b in sl]
                r['s_status'] = [s for _, s in i['samples']]
            gx, gy = (i['gx'], i['gy'])
            if i['heatmap'] is not None:
                jobs.append((i['proto'], i['heatmap'], None, kde_sigma, gx, gy, i['W'], i['H']))
                job_of[len(jobs) - 1] = (n, 'auc')
            else:
                jobs.append((i['proto'], None, [_sp(i['g_pt'])], kde_sigma, gx, gy, i['W'], i['H']))
                job_of[len(jobs) - 1] = (n, 'auc_greedy' if K else 'auc')
                if K:
                    jobs.append((i['proto'], None, [_sp(p) for p, _ in i['samples']], kde_sigma, gx, gy, i['W'], i['H']))
                    job_of[len(jobs) - 1] = (n, 'auc')
        per.append(r)
    n_proc = n_proc or min(32, os.cpu_count() or 1)
    if len(jobs) > 64 and n_proc > 1:
        with Pool(n_proc) as pool:
            res = pool.map(_auc_job, jobs, chunksize=16)
    else:
        res = [_auc_job(j) for j in jobs]
    for ji, v in enumerate(res):
        n, key = job_of[ji]
        per[n][key] = v

    metrics = {}
    for ds in sorted({i['ds'] for i in insts}):
        idx = [n for n, i in enumerate(insts) if i['ds'] == ds]
        sub, per_s = [insts[n] for n in idx], [per[n] for n in idx]
        proto = sub[0]['proto']
        ins = [r for r in per_s if r['inout']]
        m = {'protocol': proto, 'n': len(sub), 'n_in_frame': len(ins)}
        st = Counter(i['g_status'] for i in sub)
        st_in = Counter(r['g_status'] for r in ins)
        m['greedy_status'] = dict(st)
        m['parse_fail_rate'] = st['parse_fail'] / len(sub)
        m['center_fallback_rate_in_frame'] = (st_in['parse_fail'] + st_in['pred_out']) / max(1, len(ins))
        m['pred_out_on_in_frame_rate'] = st_in['pred_out'] / max(1, len(ins))
        m['clamped_rate'] = st['clamped'] / len(sub)
        if ins:
            m['auc'], m['auc_sem'] = _mean_sem([r['auc'] for r in ins])
            if proto == 'gf':
                (m['avg_l2'], m['avg_l2_sem']), (m['min_l2'], m['min_l2_sem']) = _mean_sem([r['avg_l2'] for r in ins]), _mean_sem([r['min_l2'] for r in ins])
            else:
                m['l2'], m['l2_sem'] = _mean_sem([r['avg_l2'] for r in ins])
            if K:
                m['K'] = K
                m['auc_greedy'], _ = _mean_sem([r['auc_greedy'] for r in ins])
                ss = Counter(s for r in ins for s in r['s_status'])
                tot = sum(ss.values())
                m['sample_parse_fail_rate'] = ss['parse_fail'] / tot
                m['sample_pred_out_rate'] = ss['pred_out'] / tot
                A = np.array([r['s_avg_l2'] for r in ins])  # n x K
                Mn = np.array([r['s_min_l2'] for r in ins])
                for k in [k for k in k_list if k <= K]:
                    m[f'best_of_{k}'] = {'avg_l2': float(A[:, :k].min(1).mean()), 'min_l2': float(Mn[:, :k].min(1).mean())}
                    m[f'avg_of_{k}'] = {'avg_l2': float(A[:, :k].mean()), 'min_l2': float(Mn[:, :k].mean())}
        if proto == 'vat':
            y = np.array([i['inout'] for i in sub])
            for k in ('io', 'dir'):
                s = np.array([i['p_in'][k] for i in sub])
                m['ap_inout' if k == 'io' else 'ap_inout_dir'] = ap_score(y, s)
                m[f'inout_acc_{k}'] = float(((s >= 0.5) == (y == 1)).mean())
                m[f'conf_src_{k}'] = dict(Counter(i['p_src'][k] for i in sub))
        metrics[ds] = m
    return metrics, per


# ------------------------------------------------------------------------------------------------------------------
# score 子命令
# ------------------------------------------------------------------------------------------------------------------
def cmd_score(a):
    rows = load_jsonl(a.data)
    recs = {}
    for p in a.preds:
        for r in load_jsonl(p):
            recs[r['id']] = r
    missing = [r['id'] for r in rows if r['id'] not in recs]
    if missing:
        sys.exit(f'{len(missing)}/{len(rows)} 条样本没有预测（例 {missing[:3]}），拒绝打分——不要悄悄丢样本')
    insts = [make_instance(r, recs[r['id']]) for r in rows]
    t0 = time.time()
    metrics, per = score_instances(insts, a.kde_sigma_cells, [int(k) for k in a.k_list.split(',') if k])
    meta = {'version': VERSION, 'data': os.path.abspath(a.data), 'preds': [os.path.abspath(p) for p in a.preds],
            'kde_sigma_cells': a.kde_sigma_cells, 'score_seconds': round(time.time() - t0, 1)}
    for p in a.preds:
        mp = p + '.meta.json'
        if os.path.exists(mp):
            meta.setdefault('infer_meta', []).append(json.load(open(mp)))
    out = {'meta': meta, 'metrics': metrics}
    print_metrics(metrics)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, 'w'), indent=1, ensure_ascii=False)
        print('->', a.out)
    if a.dump_instances:
        with open(a.dump_instances, 'w') as f:
            for r in per:
                f.write(json.dumps(r) + '\n')


def print_metrics(metrics):
    for ds, m in metrics.items():
        print(f'== {ds}  (protocol {m["protocol"]})  n={m["n"]}  in-frame={m["n_in_frame"]}')
        keys = ['auc', 'avg_l2', 'min_l2', 'l2', 'ap_inout', 'ap_inout_dir', 'inout_acc_io', 'inout_acc_dir', 'parse_fail_rate',
                'center_fallback_rate_in_frame', 'pred_out_on_in_frame_rate', 'clamped_rate']
        for k in keys:
            if k in m:
                sem = f' ± {m[k + "_sem"]:.4f}' if k + '_sem' in m else ''
                print(f'   {k:32s} {m[k]:.4f}{sem}')
        for k, v in m.items():
            if k.startswith(('best_of_', 'avg_of_')):
                print(f'   {k:32s} avg_l2 {v["avg_l2"]:.4f}  min_l2 {v["min_l2"]:.4f}')
        for k in ('K', 'auc_greedy', 'sample_parse_fail_rate', 'sample_pred_out_rate'):
            if k in m:
                print(f'   {k:32s} {m[k]}')


# ------------------------------------------------------------------------------------------------------------------
# infer 子命令（HF transformers）
# ------------------------------------------------------------------------------------------------------------------
def rescale_image(img, max_pixels):
    """与 swift/template/vision_utils.rescale_image 逐行一致：超过 max_pixels 才缩，缩到 (int(h'), int(w'))。"""
    import torchvision.transforms as T
    if not max_pixels or max_pixels <= 0 or img.width * img.height <= max_pixels:
        return img
    ratio = img.width / img.height
    h = math.sqrt(max_pixels / ratio)
    return T.Resize((int(h), int(h * ratio)))(img)


def prepare_image(img, max_pixels):
    """复现 ms-swift(qwen3_vl) 训练时对图像做的事：rescale_image(max_pixels) -> qwen_vl_utils.fetch_image
    （smart_resize 到 32 的倍数，最小 4 个 token、最大 16384 个 token，PIL resize）-> image_processor(do_resize=False)。
    后一步由 configure_processor 把 HF processor 的 size 放宽到同样的上下限来保证是空操作。"""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    img = rescale_image(img.convert('RGB'), max_pixels)
    f = 32  # patch 16 * merge 2
    h, w = smart_resize(img.height, img.width, factor=f, min_pixels=4 * f * f, max_pixels=16384 * f * f)
    return img.resize((w, h))


def configure_processor(proc):
    """HF processor 默认 min_pixels=65536 会放大小图，而 swift 训练时 do_resize=False（图已被 prepare_image 取整）。"""
    proc.image_processor.size = {'shortest_edge': 4 * 32 * 32, 'longest_edge': 16384 * 32 * 32}
    return proc


def build_hf_inputs(processor, images, contents, system=None):
    """contents[i] 是 convert_to_vqa 的 user 文本，形如 '<image>\\n...'。<image> 换成图像，其后的 '\\n...' 原样保留
    （ms-swift 的 qwen3_vl template 也是这样展开的；tools/check_prompt_parity.py 校验过逐 token 一致）。"""
    msgs = []
    for img, c in zip(images, contents):
        assert c.startswith('<image>'), c[:20]
        m = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': c[len('<image>'):]}]}]
        if system:
            m.insert(0, {'role': 'system', 'content': [{'type': 'text', 'text': system}]})
        msgs.append(m)
    processor.tokenizer.padding_side = 'left'
    return processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                         return_tensors='pt', padding=True)


def _find_tag_value(toks, name):
    """答案里 <name>VALUE 的 VALUE 所在 token 下标；找不到返回 None。
    tokenizer 会把相邻标签粘成一个 token（'</pt><io>' -> '</', 'pt', '><', 'io', '>'），所以不能按 ['<', name, '>'] 匹配，
    而是找 name、'>' 连续出现且前一个 token 以 '<' 结尾。"""
    for i in range(1, len(toks) - 2):
        if toks[i] == name and toks[i + 1] == '>' and toks[i - 1].endswith('<'):
            return i + 2
    return None


def analyze_generation(tok, gen_ids, logits_steps, b, ids):
    """gen_ids: 第 b 条的生成 token 列表；logits_steps: 逐步 logits 的 list[[B,V]]。返回 (text, p_in_io, p_in_dir)。"""
    import torch
    text = tok.decode(gen_ids, skip_special_tokens=True)
    toks = tok.convert_ids_to_tokens(gen_ids)
    p_io = p_dir = None
    t = _find_tag_value(toks, 'io')
    if t is not None and t < len(logits_steps) and toks[t] in ('in', 'out'):
        lg = logits_steps[t][b].float()
        p_io = float(torch.softmax(lg[[ids['in'], ids['out']]], 0)[0])
    t = _find_tag_value(toks, 'dir')
    if t is not None and t < len(logits_steps) and (toks[t] == 'none' or toks[t].isdigit()):
        lp = torch.log_softmax(logits_steps[t][b].float(), -1)
        num = torch.logsumexp(lp[ids['digits']], 0)
        p_dir = float(torch.sigmoid(num - lp[ids['none']]))  # P(数字) / (P(数字)+P(none))
    return text, p_io, p_dir


def cmd_infer(a):
    if a.gpus and a.shard is None:
        return launch_shards(a)

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    rows = load_jsonl(a.data)
    if a.limit:
        rows = rows[:a.limit]
    if a.shard is not None:
        rows = rows[a.shard::a.nshards]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)['id'] for l in open(a.out) if l.strip()}
    todo = [r for r in rows if r['id'] not in done]
    print(f'[infer] shard={a.shard} rows={len(rows)} done={len(done)} todo={len(todo)}', flush=True)
    if not todo:
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, device_map='cuda:0',
                                                            attn_implementation=a.attn).eval()
    if a.adapter:
        # 系统 peft 0.21 的 dispatch_torchao 会因厂商 torchao 0.11 版本过低而抛 ImportError（与 LoRA 无关，见 docs/rl-framework-setup.md）。
        # 不动系统环境，直接让这个检查返回 False；用 envs/swift 的 python（peft 0.18）跑则不需要。
        import peft.tuners.lora.torchao as _pt
        _pt.is_torchao_available = lambda: False
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload().eval()
    proc = configure_processor(AutoProcessor.from_pretrained(a.model))
    tok = proc.tokenizer
    ids = {'in': tok.convert_tokens_to_ids('in'), 'out': tok.convert_tokens_to_ids('out'),
           'none': tok.convert_tokens_to_ids('none'), 'digits': [tok.convert_tokens_to_ids(str(d)) for d in range(10)]}
    assert all(isinstance(v, int) and v != tok.unk_token_id for k, v in ids.items() if k != 'digits') and None not in ids['digits'], ids
    print(f'[infer] loaded in {time.time() - t0:.0f}s', flush=True)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, i):
            return todo[i], prepare_image(Image.open(todo[i]['images'][0]), a.max_pixels)  # max_pixels 必须与训练一致

    dl = torch.utils.data.DataLoader(DS(), batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=lambda b: list(zip(*b)))
    if a.shard is None:
        write_meta(a, rows)

    n_done, t0 = 0, time.time()
    with open(a.out, 'a') as fout, torch.inference_mode():
        for brows, bimgs in dl:
            inp = build_hf_inputs(proc, list(bimgs), [r['messages'][0]['content'] for r in brows], a.system).to(model.device)
            plen = inp['input_ids'].shape[1]
            common = dict(max_new_tokens=a.max_new_tokens, return_dict_in_generate=True, output_logits=True)
            g = model.generate(**inp, do_sample=False, **common)
            recs = [{'id': r['id']} for r in brows]
            for b in range(len(brows)):
                text, pio, pdir = analyze_generation(tok, g.sequences[b, plen:].tolist(), g.logits, b, ids)
                recs[b]['greedy'] = {'text': text, 'p_in_io': pio, 'p_in_dir': pdir}
            if a.num_samples:
                K = a.num_samples
                for b, r in enumerate(brows):
                    # 逐条单独 generate（batch=1，无 padding）+ 生成前按 (seed, id) 重置 RNG：K 个采样只取决于这一条本身，
                    # 不随分片数 / batch 组成 / 续跑切点变化（HF generate 不支持逐行 generator，只能这样隔离）
                    inp1 = build_hf_inputs(proc, [bimgs[b]], [r['messages'][0]['content']], a.system).to(model.device)
                    plen1 = inp1['input_ids'].shape[1]
                    torch.manual_seed(row_seed(a.seed, r['id']))
                    # top_k=0：关掉 generation_config 里默认的 top_k=20，否则采样分布被悄悄截断
                    s = model.generate(**inp1, do_sample=True, temperature=a.temperature, top_p=a.top_p, top_k=0,
                                       num_return_sequences=K, **common)
                    recs[b]['samples'] = []
                    for k in range(K):
                        text, pio, pdir = analyze_generation(tok, s.sequences[k, plen1:].tolist(), s.logits, k, ids)
                        recs[b]['samples'].append({'text': text, 'p_in_io': pio, 'p_in_dir': pdir})
            for r in recs:
                fout.write(json.dumps(r, ensure_ascii=False) + '\n')
            fout.flush()
            n_done += len(brows)
            if (n_done // a.batch_size) % 20 == 0:
                print(f'[infer] shard={a.shard} {n_done}/{len(todo)}  {n_done / (time.time() - t0):.2f} img/s', flush=True)
    print(f'[infer] done {n_done} in {time.time() - t0:.0f}s', flush=True)


def write_meta(a, rows):
    meta = {'version': VERSION, 'model': a.model, 'adapter': a.adapter, 'max_pixels': a.max_pixels, 'num_samples': a.num_samples,
            'temperature': a.temperature, 'top_p': a.top_p, 'seed': a.seed, 'system': a.system, 'data': os.path.abspath(a.data),
            'prompt_versions': sorted({r.get('prompt_version') for r in rows}), 'started': time.strftime('%F %T')}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(meta, open(a.out + '.meta.json', 'w'), indent=1)


def launch_shards(a):
    """--gpus 0,1,2,3：每张卡起一个子进程各跑 1/n 的样本，结束后合并成 --out。"""
    gpus = a.gpus.split(',')
    n = len(gpus)
    write_meta(a, load_jsonl(a.data)[:a.limit] if a.limit else load_jsonl(a.data))
    base = [sys.executable, os.path.abspath(__file__), 'infer', '--data', a.data, '--model', a.model, '--batch-size', str(a.batch_size),
            '--num-workers', str(a.num_workers), '--max-new-tokens', str(a.max_new_tokens), '--num-samples', str(a.num_samples),
            '--temperature', str(a.temperature), '--top-p', str(a.top_p), '--seed', str(a.seed), '--attn', a.attn, '--limit', str(a.limit)]
    for flag, v in (('--adapter', a.adapter), ('--max-pixels', a.max_pixels), ('--system', a.system)):
        if v:
            base += [flag, str(v)]
    procs, outs = [], []
    for i, g in enumerate(gpus):
        o = f'{a.out}.shard{i}'
        outs.append(o)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        procs.append(subprocess.Popen(base + ['--out', o, '--shard', str(i), '--nshards', str(n)], env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(f'有分片失败 {codes}；已完成部分保留在 {a.out}.shard*，重跑同一命令会续跑')
    with open(a.out, 'a') as f:
        for o in outs:
            if os.path.exists(o):
                f.write(open(o).read())
                os.remove(o)
    print('merged ->', a.out)


# ------------------------------------------------------------------------------------------------------------------
# selfcheck 子命令
# ------------------------------------------------------------------------------------------------------------------
class _Check:
    def __init__(self):
        self.fails = []

    def ok(self, cond, msg):
        print(('  PASS ' if cond else '  FAIL ') + msg)
        if not cond:
            self.fails.append(msg)


def _load_official():
    import importlib.util
    spec = importlib.util.spec_from_file_location('official_gazelle_utils', OFFICIAL_UTILS)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _replay_instances(rows, heatmaps, points, p_in):
    """把官方热图模型的输出包装成 instance：点 = argmax 格子左上角 idx/64（官方 L2 口径）。"""
    insts = []
    for r, hm, pt, p in zip(rows, heatmaps, points, p_in):
        i = make_instance(r, {'greedy': {'text': r['solution'], 'p_in_io': p}})  # text 只是占位，下面覆盖
        i['g_pt'], i['g_status'], i['heatmap'] = pt, 'ok', hm
        insts.append(i)
    return insts


def cmd_selfcheck(a):
    C = _Check()
    rng = np.random.RandomState(0)

    print('[1] 合成用例')
    P = parse_completion
    C.ok(P('<dir>41</dir><pt>709,495</pt><io>in</io>')['ok'], '标准 in 答案可解析')
    C.ok(P(' <dir>359</dir> <pt>5, 12</pt> <io>in</io> ')['ok'], '空白容忍')
    C.ok(P('<dir>none</dir><io>out</io>')['ok'] and P('<dir>none</dir><io>out</io>')['x'] is None, 'out 答案可解析且无点')
    for bad in ['', 'in', '<dir>10</dir><io>in</io>', '<dir>none</dir><pt>1,2</pt><io>out</io>', '<pt>1,2</pt><dir>3</dir><io>in</io>',
                '<dir>3</dir><pt>1,2</pt><io>maybe</io>', 'The answer is <dir>3</dir><pt>1,2</pt><io>in</io>', None]:
        C.ok(not P(bad)['ok'], f'语法违规判失败: {bad!r}')
    C.ok(completion_to_point('<dir>359</dir><pt>1200,-5</pt><io>in</io>') == ((1.0, 0.0), 'clamped'), '越界点夹到 [0,1] 并标 clamped')
    C.ok(completion_to_point('<dir>none</dir><io>out</io>') == (None, 'pred_out'), 'out -> 无点')
    C.ok(completion_to_point('garbage') == (None, 'parse_fail'), '垃圾 -> parse_fail')

    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(f'{G}/ckpt/Qwen3-VL-4B-Instruct')
        for ans, want_dir, want_io in [('<dir>41</dir><pt>709,495</pt><io>in</io>', '4', 'in'), ('<dir>none</dir><io>out</io>', 'none', 'out'),
                                       ('<dir>359</dir><pt>5,12</pt><io>in</io>', '3', 'in')]:
            toks = tk.convert_ids_to_tokens(tk(ans, add_special_tokens=False).input_ids)
            td, ti = _find_tag_value(toks, 'dir'), _find_tag_value(toks, 'io')
            C.ok(td is not None and ti is not None and toks[td] == want_dir and toks[ti] == want_io,
                 f'tokenizer 下定位 <dir>/<io> 的值 token：{ans} -> {toks[td] if td else None}, {toks[ti] if ti else None}')
    except Exception as e:  # noqa: BLE001
        C.ok(False, f'无法加载 tokenizer: {e!r}')

    avg, mn = l2_stats((0.5, 0.5), [0.5, 0.8], [0.5, 0.9])
    C.ok(abs(avg - math.hypot(0.15, 0.2)) < 1e-12 and mn == 0.0, 'Avg/Min L2 手算值')

    from sklearn.metrics import roc_auc_score
    worst = 0.0
    for _ in range(300):
        n = int(rng.randint(50, 3000))
        s = np.round(rng.rand(n), int(rng.choice([1, 2, 6])))  # 故意制造并列
        pos = rng.choice(n, int(rng.randint(1, 8)), replace=False)
        y = np.zeros(n)
        y[pos] = 1
        worst = max(worst, abs(_auc_rank(s, pos) - roc_auc_score(y, s)))
    C.ok(worst < 1e-12, f'秩公式 AUC == sklearn（含并列），最大差 {worst:.1e}')

    hm = kde_heatmap([(0.30, 0.70)], 3.0)
    iy, ix = np.unravel_index(hm.argmax(), hm.shape)
    C.ok((ix, iy) == (int(0.30 * RES), int(0.70 * RES)) and hm.min() > 0, f'KDE 峰值在点所在格子 {(ix, iy)}，且无下溢到 0')
    hm2 = kde_heatmap([(0.5, 0.5), (0.5, 0.5)], 3.0)
    C.ok(np.allclose(hm2, 2 * kde_heatmap([(0.5, 0.5)], 3.0)), 'KDE 对重复点线性叠加')
    C.ok(auc_vat(kde_heatmap([(0.4, 0.4)]), 0.4, 0.4) > 0.99 and auc_vat(kde_heatmap([(0.9, 0.9)]), 0.1, 0.1) < 0.5, 'VAT AUC：点对时≈1，点错时<0.5')
    C.ok(auc_gazefollow(kde_heatmap([(0.4, 0.4)]), [0.4], [0.4], 375, 500) > 0.99, 'GazeFollow AUC：点对时≈1')
    y = np.array([1, 1, 0, 0, 1, 0])
    C.ok(ap_score(y, np.array([.9, .8, .1, .2, .7, .3])) == 1.0 and ap_score(y, 1 - np.array([.9, .8, .1, .2, .7, .3])) < 0.6, 'AP：完美排序=1，反序<0.6')

    print('[2] 与官方 gazelle.utils 逐实例对拍（同一批热图）')
    gf_rows = load_jsonl(f'{G}/data/vqa/gazefollow_test.jsonl')
    vat_rows = load_jsonl(f'{G}/data/vqa/vat_test.jsonl')
    gf_dump = np.load(f'{G}/results/repro_test/full.npz')
    vat_dump = np.load(f'{G}/results/vat_official/official_inout_vitl14.npz')
    gf_hm = gf_dump['heatmaps'].astype(np.float32)
    vat_hm = vat_dump['heatmaps'].astype(np.float32)
    C.ok(len(gf_rows) == len(gf_hm) == 4782, f'GazeFollow 实例数 {len(gf_rows)} == 4782 == 热图数 {len(gf_hm)}')
    C.ok(len(vat_rows) == len(vat_dump['inout_gt']) == 31978, f'VAT 实例数 {len(vat_rows)} == 31978')
    C.ok(np.array_equal(np.array([r['inout'] for r in vat_rows]), vat_dump['inout_gt']), 'VAT：jsonl 的 inout 与官方回放逐条一致')
    in_idx = vat_dump['in_idx']
    C.ok(np.array_equal(np.flatnonzero([r['inout'] for r in vat_rows]), in_idx), 'VAT：in-frame 下标与官方回放一致')
    try:
        import torch
        off = _load_official()
        sub = rng.choice(len(gf_rows), 120, replace=False)
        d_auc = d_avg = d_min = 0.0
        for n in sub:
            r, hm = gf_rows[n], torch.from_numpy(gf_hm[n])
            ref_auc = off.gazefollow_auc(hm, r['gt_xs_norm'], r['gt_ys_norm'], r['height'], r['width'])
            ref_avg, ref_min = off.gazefollow_l2(hm, r['gt_xs_norm'], r['gt_ys_norm'])
            iy, ix = np.unravel_index(gf_hm[n].argmax(), gf_hm[n].shape)
            avg, mn = l2_stats((ix / RES, iy / RES), r['gt_xs_norm'], r['gt_ys_norm'])
            d_auc = max(d_auc, abs(auc_gazefollow(gf_hm[n], r['gt_xs_norm'], r['gt_ys_norm'], r['height'], r['width']) - ref_auc))
            d_avg, d_min = max(d_avg, abs(avg - ref_avg)), max(d_min, abs(mn - ref_min))
        C.ok(d_auc < 1e-9 and d_avg < 1e-9 and d_min < 1e-9, f'GazeFollow 120 个实例：AUC 最大差 {d_auc:.1e}、Avg L2 {d_avg:.1e}、Min L2 {d_min:.1e}')
        d_auc = d_l2 = 0.0
        for k in rng.choice(len(in_idx), 300, replace=False):
            r, hm = vat_rows[in_idx[k]], torch.from_numpy(vat_hm[k])
            gx, gy = r['gt_xs_norm'][0], r['gt_ys_norm'][0]
            iy, ix = np.unravel_index(vat_hm[k].argmax(), vat_hm[k].shape)
            d_auc = max(d_auc, abs(auc_vat(vat_hm[k], gx, gy) - off.vat_auc(hm, gx, gy)))
            d_l2 = max(d_l2, abs(l2_stats((ix / RES, iy / RES), [gx], [gy])[0] - off.vat_l2(hm, gx, gy)))
        C.ok(d_auc < 1e-9 and d_l2 < 1e-9, f'VAT 300 个实例：AUC 最大差 {d_auc:.1e}、L2 {d_l2:.1e}')
    except Exception as e:  # noqa: BLE001
        C.ok(False, f'无法加载官方 gazelle.utils 对拍: {e!r}')

    print('[3] 回放官方 Gaze-LLE ViT-L 热图 -> 我的评分代码 -> 对照官方数字')
    pts = []
    for hm in gf_hm:
        iy, ix = np.unravel_index(hm.argmax(), hm.shape)
        pts.append((ix / RES, iy / RES))
    m, per = score_instances(_replay_instances(gf_rows, gf_hm, pts, [1.0] * len(gf_rows)))
    g = m['gazefollow']
    ref = gf_dump['per_instance']
    print(f'   GazeFollow  AUC {g["auc"]:.4f}  Avg L2 {g["avg_l2"]:.4f}  Min L2 {g["min_l2"]:.4f}')
    d = np.array([[p['auc'], p['avg_l2'], p['min_l2']] for p in per]) - ref[:, :3]
    # 官方落盘热图是 fp16：最大值处并列时 argmax 取到另一个格子。要求「L2 有差异的实例」恰好都是 fp16 并列实例，其余逐位一致
    dif = np.flatnonzero(np.abs(d[:, 1:]).max(1) > 1e-9)
    tied = np.array([(gf_dump['heatmaps'][k] == gf_dump['heatmaps'][k].max()).sum() > 1 for k in dif], dtype=bool)
    C.ok(len(dif) == 0 or tied.all(), f'GazeFollow L2 与官方落盘值有差异的实例 {len(dif)}/{len(d)}，其中 fp16 并列 {int(tied.sum())}；其余 {len(d) - len(dif)} 个逐位一致')
    C.ok(np.abs(d[:, 0]).mean() < 1e-4, f'GazeFollow AUC 逐实例 vs 官方落盘值（fp16 热图）：均值 |Δ| {np.abs(d[:, 0]).mean():.1e}，最大 {np.abs(d[:, 0]).max():.1e}')
    C.ok(abs(g['auc'] - 0.958) <= 1e-3 and abs(g['avg_l2'] - 0.099) <= 1e-3 and abs(g['min_l2'] - 0.041) <= 1e-3,
         f'GazeFollow 对论文 0.958 / 0.099 / 0.041（容差 0.001）：{g["auc"]:.4f} / {g["avg_l2"]:.4f} / {g["min_l2"]:.4f}')

    vpts, vhm, vp = [None] * len(vat_rows), [None] * len(vat_rows), vat_dump['inout_pred']
    for k, n in enumerate(in_idx):
        vhm[n] = vat_hm[k]
        iy, ix = np.unravel_index(vat_hm[k].argmax(), vat_hm[k].shape)
        vpts[n] = (ix / RES, iy / RES)
    insts = _replay_instances(vat_rows, vhm, vpts, list(vp))
    for i in insts:  # 官方对 GT 画面外的样本不出点；这里只用于 AP
        if i['heatmap'] is None:
            i['g_pt'] = CENTER
    m, per = score_instances(insts)
    v = m['vat']
    print(f'   VAT         AUC {v["auc"]:.4f}  L2 {v["l2"]:.4f}  AP {v["ap_inout"]:.4f}')
    ref = vat_dump['in_frame_metrics']
    d = np.array([[p['auc'], p['avg_l2']] for p in per if p['inout']]) - ref
    dif = np.flatnonzero(np.abs(d[:, 1]) > 1e-9)
    tied = np.array([(vat_dump['heatmaps'][k] == vat_dump['heatmaps'][k].max()).sum() > 1 for k in dif], dtype=bool)
    C.ok(len(dif) == 0 or tied.all(), f'VAT L2 与官方落盘值有差异的实例 {len(dif)}/{len(d)}，其中 fp16 并列 {int(tied.sum())}；其余 {len(d) - len(dif)} 个逐位一致')
    C.ok(np.abs(d[:, 0]).mean() < 1e-4, f'VAT AUC 逐实例 vs 官方落盘值（fp16 热图）：均值 |Δ| {np.abs(d[:, 0]).mean():.1e}，最大 {np.abs(d[:, 0]).max():.1e}')
    C.ok(abs(v['auc'] - 0.937) <= 1e-3 and abs(v['l2'] - 0.103) <= 1e-3 and abs(v['ap_inout'] - 0.903) <= 1e-3,
         f'VAT 对论文 0.937 / 0.103 / 0.903（容差 0.001）：{v["auc"]:.4f} / {v["l2"]:.4f} / {v["ap_inout"]:.4f}')

    print('[4] 管线冒烟：真值答案 / 全部解析失败')
    for name, rows in (('gazefollow_test', gf_rows), ('vat_test', vat_rows)):
        gold = [make_instance(r, {'greedy': {'text': r['solution'], 'p_in_io': 1.0 if r['inout'] else 0.0}}) for r in rows]
        m, _ = score_instances(gold)
        ds = 'gazefollow' if 'gazefollow' in name else 'vat'
        mm = m[ds]
        l2 = mm['avg_l2'] if ds == 'gazefollow' else mm['l2']
        C.ok(mm['parse_fail_rate'] == 0 and l2 < 1e-3, f'{name} 真值答案：解析失败 0，L2 {l2:.5f}（只剩 0-1000 量化误差，应 < 0.001），AUC(单高斯) {mm["auc"]:.4f}')
        if ds == 'vat':
            C.ok(mm['ap_inout'] == 1.0 and mm['ap_inout_dir'] == 1.0, f'{name} 真值答案：AP=1')
        bad = [make_instance(r, {'greedy': {'text': '??'}}) for r in rows]
        m, _ = score_instances(bad)
        mm = m[ds]
        exp = float(np.mean([math.hypot(0.5 - np.mean(r['gt_xs_norm']), 0.5 - np.mean(r['gt_ys_norm'])) for r in rows if r['inout']]))
        l2 = mm['avg_l2'] if ds == 'gazefollow' else mm['l2']
        C.ok(mm['parse_fail_rate'] == 1.0 and abs(l2 - exp) < 1e-9, f'{name} 全部解析失败 = 中心点基线：L2 {l2:.4f}（独立算得 {exp:.4f}），不丢样本 n={mm["n"]}')

    print('[5] 合成预测：失败/out/越界/采样路径的计数与数值（手算）')
    base = {'width': 400, 'height': 300, 'inout': 1, 'gt_xs_norm': [0.2, 0.4], 'gt_ys_norm': [0.5, 0.5]}  # mean=(0.3,0.5)
    rows5 = [dict(base, id=f'gazefollow/test/x{i}') for i in range(4)]
    texts = ['<dir>10</dir><pt>300,500</pt><io>in</io>',   # 恰在标注均值上: avg 0, min 0.1
             'blah',                                       # 解析失败 -> 中心 (0.5,0.5)
             '<dir>none</dir><io>out</io>',                # 回答 out -> 中心
             '<dir>10</dir><pt>1500,500</pt><io>in</io>']  # 越界 -> 夹到 (1.0,0.5)
    pr = {r['id']: {'greedy': {'text': t, 'p_in_io': 0.9}, 'samples': [{'text': t}, {'text': texts[0]}, {'text': 'blah'}]}
          for r, t in zip(rows5, texts)}
    insts = [make_instance(r, pr[r['id']]) for r in rows5]
    m, per = score_instances(insts, k_list=(2, 3))
    g = m['gazefollow']
    exp_avg = [0.0, math.hypot(0.2, 0.0), math.hypot(0.2, 0.0), math.hypot(0.7, 0.0)]
    exp_min = [0.1, math.hypot(0.1, 0.0), math.hypot(0.1, 0.0), math.hypot(0.6, 0.0)]
    C.ok(abs(g['avg_l2'] - np.mean(exp_avg)) < 1e-12 and abs(g['min_l2'] - np.mean(exp_min)) < 1e-12, f'greedy Avg/Min L2 = 手算 {np.mean(exp_avg):.4f} / {np.mean(exp_min):.4f}')
    C.ok((g['parse_fail_rate'], g['pred_out_on_in_frame_rate'], g['clamped_rate'], g['center_fallback_rate_in_frame']) == (0.25, 0.25, 0.25, 0.5),
         '解析失败 / out / 越界 / 中心回退 比例 = 1/4, 1/4, 1/4, 1/2')
    # 采样：每个实例 3 个采样 = [自身, 真值答案, 解析失败(=中心)]
    s_avg = np.array([[a, 0.0, math.hypot(0.2, 0.0)] for a in exp_avg])
    C.ok(abs(g['best_of_2']['avg_l2'] - np.minimum(s_avg[:, 0], s_avg[:, 1]).mean()) < 1e-12 and abs(g['best_of_3']['avg_l2']) < 1e-12
         and abs(g['avg_of_3']['avg_l2'] - s_avg.mean()) < 1e-12 and g['K'] == 3, 'best-of-k / avg-of-k = 手算')
    C.ok(abs(g['sample_parse_fail_rate'] - (1 + 1 * 4) / 12) < 1e-12, '采样解析失败率 = 5/12')
    vrows = [dict(id=f'vat/test/y{i}', width=100, height=100, inout=i % 2, **({'gt_xs_norm': [0.5], 'gt_ys_norm': [0.5]} if i % 2 else {})) for i in range(6)]
    vp = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]  # inout=[0,1,0,1,0,1]：完美排序
    insts = [make_instance(r, {'greedy': {'text': '<dir>1</dir><pt>500,500</pt><io>in</io>' if r['inout'] else '<dir>none</dir><io>out</io>', 'p_in_io': p}})
             for r, p in zip(vrows, vp)]
    m, _ = score_instances(insts)
    C.ok(m['vat']['ap_inout'] == 1.0 and m['vat']['n_in_frame'] == 3 and m['vat']['l2'] < 1e-12 and m['vat']['auc'] > 0.99,
         'VAT：AP=1、只对 GT-in 的 3 个样本算 L2/AUC')

    print('[6] 逐样本采样种子 row_seed（prereg §5.2）')
    C.ok(row_seed(0, 'gazefollow/val/abc') == 781806472528550614 and row_seed(1, 'gazefollow/val/abc') == 63030687044023304,
         '与固定字面值一致（sha256 派生，不依赖进程的 hash 加盐）')
    ids6 = [f'gazefollow/val/{i}' for i in range(20000)]
    sd = [row_seed(0, i) for i in ids6]
    C.ok(len(set(sd)) == len(sd), f'{len(ids6)} 个不同 id 的种子互不相同')
    C.ok(sd == [row_seed(0, i) for i in ids6] and [row_seed(0, i) for i in reversed(ids6)][::-1] == sd,
         '同 (seed,id) 恒同值，且与遍历顺序无关（所以与分片数 / 续跑切点无关）')
    C.ok(all(row_seed(0, i) != row_seed(1, i) for i in ids6[:2000]), 'seed 变了，种子变')
    C.ok(max(sd) < 2 ** 60, '种子 < 2**60（torch.manual_seed 可直接吃）')

    print('\nselfcheck', 'PASS' if not C.fails else f'FAIL ({len(C.fails)})')
    for f in C.fails:
        print('  -', f)
    sys.exit(1 if C.fails else 0)


# ------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)

    s = sp.add_parser('infer')
    s.add_argument('--data', required=True)
    s.add_argument('--model', default=f'{G}/ckpt/Qwen3-VL-4B-Instruct')
    s.add_argument('--adapter', default=None, help='peft LoRA adapter 目录（swift 的 checkpoint-N），加载后 merge')
    s.add_argument('--out', required=True)
    s.add_argument('--gpus', default=None, help='如 0,1,2,3：每卡一个分片进程，完成后合并')
    s.add_argument('--shard', type=int, default=None)
    s.add_argument('--nshards', type=int, default=1)
    s.add_argument('--batch-size', type=int, default=8)
    s.add_argument('--num-workers', type=int, default=4)
    s.add_argument('--max-pixels', type=int, default=None, help='必须与训练时 swift 的 --max_pixels 一致（smoke SFT 用 262144）')
    s.add_argument('--max-new-tokens', type=int, default=64)
    s.add_argument('--num-samples', type=int, default=0, help='>0 时额外采样 K 条（AUC 的 KDE、best-of-K 用）')
    s.add_argument('--temperature', type=float, default=1.0)
    s.add_argument('--top-p', type=float, default=1.0)
    s.add_argument('--seed', type=int, default=0, help='采样种子；每条样本的实际种子 = row_seed(seed, id)，写进 preds.jsonl.meta.json')
    s.add_argument('--system', default=None)
    s.add_argument('--attn', default='sdpa')
    s.add_argument('--limit', type=int, default=0)

    s = sp.add_parser('score')
    s.add_argument('--data', required=True)
    s.add_argument('--preds', required=True, nargs='+')
    s.add_argument('--out', default=None)
    s.add_argument('--dump-instances', default=None)
    s.add_argument('--kde-sigma-cells', type=float, default=3.0)
    s.add_argument('--k-list', default='2,4,8,16')

    sp.add_parser('selfcheck')

    a = ap.parse_args()
    {'infer': cmd_infer, 'score': cmd_score, 'selfcheck': cmd_selfcheck}[a.cmd](a)


if __name__ == '__main__':
    main()
