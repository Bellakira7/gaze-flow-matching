"""C22 第 5 项：flow 线的标准 VAT（当前帧 gaze following）读数。细则见 results/vat_standard/RULES.md。
模型 = experiments/forecast/c20_ours_h0（ForecastAny flow + 眼部 token + 出画头，H=0），不重训。

  # 逐实例推理（val：C20 的 4 个 val 节目，H=0，sample_every=6；test：data/vqa/vat_test.jsonl 全部 31978 行，须 --read_test）
  python -I -u eval_vat_standard.py infer --split val  --ckpt <epoch_k.pt> --name h0_ep<k> --n 1000
  python -I -u eval_vat_standard.py infer --split test --ckpt <epoch_k.pt> --name h0 --read_test --nshards 4 --shard i
  # val 上选 epoch（联合 NLL，1000 样本）与 AUC 平滑 σ（10000 样本）→ results/vat_standard/selection.json
  python -I eval_vat_standard.py select
  # 合并 test 分片并用冻结 eval_gaze 打分 → results/vat_standard/test_scored.json
  python -I eval_vat_standard.py score --name h0
  # 打分包装对拍：OmniGF 已落盘 VAT 热图走同一个 score_rows，应复现 omnigf_table.json
  python -I eval_vat_standard.py check_omnigf

每个 shard 的 npz：idx（test = jsonl 行号；val = VATForecast.items 下标）、inn、y（归一化真值，画面外为 0.5）、logit_out（出画 logit，fp32）、
counts（10000 个样本在 64×64 格的计数，uint16；点估计 / 密度都由它确定性导出：samples_to_hyps 与 pmf_from_samples 只经过 bincount）。
已存在且能读、行数对的输出一律跳过，不覆盖；0 字节或坏文件删掉重跑。
"""
from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

G = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{G}/gazelle_online')
sys.path.insert(0, f'{G}/setmetrics')
sys.path.insert(0, f'{G}/tools')
import metrics as M  # noqa: E402

OUT = f'{G}/results/vat_standard'
CKPT_DIR = f'{G}/experiments/forecast/c20_ours_h0'
EPOCHS = range(8)
SIGMAS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
TIE = 1e-4


def shard_path(name, split, shard, nshards, limit=0):
    return f'{OUT}/{name}_{split}_shard{shard}of{nshards}' + (f'_smoke{limit}' if limit else '') + '.npz'


def load_ok(p, n=None):
    """能读、键齐、（给 n 时）行数对 -> dict；否则 None（0 字节 / 坏文件顺手删掉）。"""
    if not os.path.exists(p):
        return None
    try:
        d = dict(np.load(p))
        assert {'idx', 'inn', 'y', 'logit_out', 'counts'} <= set(d) and (n is None or len(d['idx']) == n)
        return d
    except Exception as e:  # noqa: BLE001
        print(f'[warn] {p} 不可用（{type(e).__name__}: {e}），删掉重跑', flush=True)
        os.remove(p)
        return None


# ====================================================================== 推理
class RowsDS:
    """vat_test.jsonl 的行 -> 与 VATForecast.__getitem__ 相同的元组（用 jsonl 的 bbox_norm 与图像，不做增强）。"""

    def __init__(self, rows, idx, transform, margin=1.25, crop_size=224):
        self.rows, self.idx, self.tf, self.margin, self.crop_size = rows, idx, transform, margin, crop_size

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        from gaze_eye2 import hires_crop
        r = self.rows[self.idx[i]]
        img = Image.open(r['images'][0]).convert('RGB')
        bn = [float(v) for v in r['bbox_norm']]
        inn = int(r['inout']) == 1
        y = [float(r['gt_x_norm']), float(r['gt_y_norm'])] if inn else [0.5, 0.5]
        crops = torch.stack([hires_crop(img, bn, self.margin, self.crop_size)])
        return self.tf(img), bn, torch.tensor(y, dtype=torch.float32), torch.tensor(float(inn)), crops, int(self.idx[i])


def cmd_infer(a):
    import torch
    from train_online import AmpBackbone
    from common import IN_SIZE
    from gaze_forecast import VATForecast, ForecastAny, collate
    import eval_gaze as E
    assert a.split != 'test' or a.read_test, '读 VAT 测试集须显式 --read_test（C22 第 5 项：只读一次）'
    out = shard_path(a.name, a.split, a.shard, a.nshards, a.limit)
    sel = json.load(open(f'{G}/experiments/sampler_selection.json'))
    bb = AmpBackbone('dinov2_vitl14')
    tf = bb.get_transform(IN_SIZE)
    if a.split == 'val':
        ds = VATForecast('val', 0, (0,), tf, sample_every=6, scheme='c20')
        items = list(range(len(ds)))[a.shard::a.nshards]
        if a.limit:
            items = items[:a.limit]

        class Sub(torch.utils.data.Dataset):
            def __len__(self):
                return len(items)

            def __getitem__(self, i):
                r = ds[items[i]]
                return (*r[:5], items[i])
        data = Sub()
    elif a.split == 'jsonl_smoke':  # 只给 RowsDS 冒烟：同格式的 VAT 训练节目 jsonl（Gaze-R1 的 vat_val_s3），不碰 test
        rows = E.load_jsonl(f'{G}/data/vqa/vat_val_s3.jsonl')
        idx = list(range(len(rows)))[a.shard::a.nshards][:a.limit or 64]
        data = RowsDS(rows, idx, tf)
    else:
        rows = E.load_jsonl(f'{G}/data/vqa/vat_test.jsonl')
        assert len(rows) == 31978, len(rows)
        idx = list(range(len(rows)))[a.shard::a.nshards]
        if a.limit:
            idx = idx[:a.limit]
        data = RowsDS(rows, idx, tf)
    if load_ok(out, len(data)) is not None:
        print(f'SKIP {out}（已存在，{len(data)} 行）', flush=True)
        return
    model = ForecastAny(bb, 'flow', eye=True, depth=6).cuda().eval()
    model.load_decoder_state(torch.load(a.ckpt, map_location='cpu', weights_only=True))
    dl = torch.utils.data.DataLoader(data, batch_size=a.bs, collate_fn=collate, num_workers=6)
    gen = torch.Generator(device='cuda').manual_seed(1234 + a.shard)
    R = {k: [] for k in ('idx', 'inn', 'y', 'logit_out', 'counts')}
    t0 = time.time()
    print(f'[{a.name}/{a.split} shard {a.shard}/{a.nshards}] {len(data)} 行  ckpt {a.ckpt}  n={a.n}  sampler {sel["flow"]}', flush=True)
    with torch.no_grad():
        for b, (imgs, bn, y, inn, crops, meta) in enumerate(dl):
            tok = model.tokens(imgs.cuda(), [[x] for x in bn], crops.cuda())
            lo = model.out_logit(tok).float().cpu().numpy()
            S = model.sample(tok, a.n, gen=gen, **sel['flow']).float().cpu().numpy().astype(np.float64)
            for i in range(len(bn)):
                c = np.bincount(M.cell_index(np.clip(S[i], 0, 1 - 1e-9)), minlength=M.NCELL)
                R['counts'].append(c.reshape(M.G, M.G).astype(np.uint16))
                R['idx'].append(int(meta[i]))
                R['inn'].append(bool(inn[i] > 0.5))
                R['y'].append(y[i].numpy())
                R['logit_out'].append(float(lo[i]))
            if b % 10 == 0:
                print(f'[{a.name}/{a.split} shard {a.shard}] {len(R["idx"])}/{len(data)}  {time.time() - t0:.0f}s', flush=True)
    os.makedirs(OUT, exist_ok=True)
    tmp = out[:-4] + '.tmp.npz'
    np.savez(tmp, **{k: np.array(v) for k, v in R.items()})
    os.replace(tmp, out)
    assert load_ok(out, len(data)) is not None, f'{out} 写完读不回来（cpfs ENOSPC？）'
    print('saved', out, f'{time.time() - t0:.0f}s', flush=True)


# ====================================================================== 由计数导出点 / 密度（与 gaze_eval.samples_to_hyps、g2_eval.pmf_from_samples 逐行相同，只是起点换成计数）
def hyps_from_counts(cnt, sigma_h):
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(cnt.astype(float), sigma_h, mode='constant')
    pk = M.nms_peaks(sm, 5, 0.10)
    P = sm.ravel() / sm.sum()
    w = np.array([M.disc_mass(P, i, 0.10) for i in pk])
    hyps = M.CELLS[pk]
    if len(pk) < 5:
        pad = 5 - len(pk)
        hyps = np.concatenate([hyps, np.repeat(hyps[:1], pad, 0)])
        w = np.concatenate([w, np.zeros(pad)])
    o = np.argsort(-w, kind='stable')
    hyps, w = hyps[o], w[o]
    return hyps, w / max(w.sum(), 1e-12)


def pmf_from_counts(cnt, sigma, lam=M.LAMBDA):
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(cnt.astype(float), sigma, mode='constant')
    P = sm.ravel() / sm.sum()
    return (1 - lam) * P + lam / M.NCELL


def _point_job(job):
    cnt, sigma_h = job
    return hyps_from_counts(cnt, sigma_h)[0][0]


def _val_auc_job(job):
    import eval_gaze as E
    cnt, y = job
    return [E.auc_vat(pmf_from_counts(cnt, s).reshape(M.G, M.G), float(y[0]), float(y[1])) for s in SIGMAS]


def merge(name, split, nshards, n_expect=None):
    parts = []
    for s in range(nshards):
        d = load_ok(shard_path(name, split, s, nshards))
        assert d is not None, f'缺 {shard_path(name, split, s, nshards)}'
        parts.append(d)
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    o = np.argsort(d['idx'], kind='stable')
    d = {k: v[o] for k, v in d.items()}
    assert len(np.unique(d['idx'])) == len(d['idx']), '分片间有重复行'
    if n_expect is not None:
        assert np.array_equal(d['idx'], np.arange(n_expect)), f'分片没恰好覆盖 0..{n_expect - 1}'
    return d


# ====================================================================== val：选 epoch 与 σ
def joint_nll(d):
    po = np.clip(1 / (1 + np.exp(-d['logit_out'].astype(np.float64))), 1e-4, 1 - 1e-4)
    v = []
    for i in range(len(d['idx'])):
        if d['inn'][i]:
            P = pmf_from_counts(d['counts'][i], 1.0)
            v.append(-math.log(P[M.cell_index(np.clip(d['y'][i][None].astype(np.float64), 0, 1 - 1e-9))[0]]) - math.log(1 - po[i]))
        else:
            v.append(-math.log(po[i]))
    return float(np.mean(v))


def cmd_select(a):
    from sklearn.metrics import average_precision_score
    res = {'rules': 'results/vat_standard/RULES.md', 'epochs': {}}
    for e in EPOCHS:
        d = merge(f'h0_ep{e}', 'val', a.nshards)
        res['epochs'][e] = dict(n=int(len(d['idx'])), n_in=int(d['inn'].sum()), joint_nll=joint_nll(d))
        print(f'epoch {e}: val 联合 NLL {res["epochs"][e]["joint_nll"]:.4f}  n={len(d["idx"])}', flush=True)
    best = min(EPOCHS, key=lambda e: res['epochs'][e]['joint_nll'])
    res['epoch'] = best
    tag = f'h0_ep{best}_n10000'
    d = merge(tag, 'val', a.nshards)
    inn = d['inn']
    with Pool(32) as pool:
        A = np.array(pool.map(_val_auc_job, [(d['counts'][i], d['y'][i]) for i in np.flatnonzero(inn)], chunksize=8))
        pts = np.array(pool.map(_point_job, [(d['counts'][i], json.load(open(f'{G}/experiments/sampler_selection.json'))['sigma_h']) for i in np.flatnonzero(inn)], chunksize=8))
    auc = A.mean(0)
    top = auc.max()
    tied = [s for s, v in zip(SIGMAS, auc) if top - v < TIE]
    sigma = min(tied, key=lambda s: (abs(s - 3.0), s))
    p_in = 1 / (1 + np.exp(d['logit_out'].astype(np.float64)))
    res.update(sigma=sigma, val_n10000=dict(tag=tag, n=int(len(inn)), n_in=int(inn.sum()), auc_by_sigma={f'{s:g}': float(v) for s, v in zip(SIGMAS, auc)},
                                           l2=float(np.hypot(*(pts - d['y'][inn]).T).mean()), ap_inout=float(average_precision_score(inn.astype(int), p_in))))
    json.dump(res, open(f'{OUT}/selection.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)


# ====================================================================== test：打分
def score_rows(rows, hms, pts, pin):
    """冻结 eval_gaze 的打分：与 tools/omnigf_score.py 对 OmniGF 的调用相同（热图原样进 64×64 方窗 AUC；点给定；AP 用 p_in）。"""
    import eval_gaze as E
    insts = E._replay_instances(rows, hms, pts, pin)
    for i in insts:  # 只有一个 in/out 分数；不覆盖的话 'dir' 会退回 jsonl 里 solution 的真值标签
        i['p_in']['dir'] = i['p_in']['io']
    m, per = E.score_instances(insts, 3.0, ())
    (metrics,) = m.values()
    return metrics, per


def cmd_score(a):
    import eval_gaze as E
    out = f'{OUT}/test_scored.json'
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f'SKIP {out}（已存在）', flush=True)
        print(open(out).read())
        return
    sel = json.load(open(f'{OUT}/selection.json'))
    sigma, sigma_h = sel['sigma'], json.load(open(f'{G}/experiments/sampler_selection.json'))['sigma_h']
    rows = E.load_jsonl(f'{G}/data/vqa/vat_test.jsonl')
    d = merge(a.name, 'test', a.nshards, len(rows))
    assert np.array_equal(d['inn'], np.array([int(r['inout']) == 1 for r in rows])), 'inout 与 jsonl 不一致'
    print(f'合并 {len(rows)} 行，σ={sigma:g}（val 选定），σ_h={sigma_h:g}；开始导出点 / 热图', flush=True)
    t0 = time.time()
    with Pool(32) as pool:
        pts = pool.map(_point_job, [(c, sigma_h) for c in d['counts']], chunksize=64)
    print(f'点导出完 {time.time() - t0:.0f}s', flush=True)
    hms = [pmf_from_counts(c, sigma).reshape(M.G, M.G) for c in d['counts']]
    pin = (1 / (1 + np.exp(d['logit_out'].astype(np.float64)))).tolist()
    print(f'热图导出完 {time.time() - t0:.0f}s；打分', flush=True)
    metrics, per = score_rows(rows, hms, [(float(x), float(y)) for x, y in pts], pin)
    res = dict(model=f'{CKPT_DIR}/epoch_{sel["epoch"]}.pt', epoch=sel['epoch'], sigma=sigma, sigma_h=sigma_h, n=len(rows), metrics=metrics)
    np.savez(f'{OUT}/test_per_instance.npz', id=np.array([r['id'] for r in rows]), inout=d['inn'], p_in=np.array(pin), point=np.array(pts),
             l2=np.array([p.get('avg_l2', np.nan) for p in per]), auc=np.array([p.get('auc', np.nan) for p in per]))
    json.dump(res, open(out, 'w'), indent=1, ensure_ascii=False)
    print({k: metrics.get(k) for k in ('auc', 'l2', 'ap_inout', 'ap_inout_dir', 'n', 'n_in_frame')}, flush=True)


def cmd_check_omnigf(a):
    """OmniGF 已落盘的 VAT 热图走 score_rows，应与 results/omnigf/omnigf_table.json 的 VAT 行逐位相同。"""
    import glob
    import eval_gaze as E
    import omnigf_score as O
    rows = E.load_jsonl(f'{G}/data/vqa/vat_test.jsonl')
    if a.limit:  # 冒烟：只取第一个分片覆盖到的人（partial）
        insts = O.replay_vat(sorted(glob.glob(f'{G}/results/omnigf/vat_test_dump_shard*.npz'))[:1], rows, partial=True)
    else:
        insts = O.replay_vat(sorted(glob.glob(f'{G}/results/omnigf/vat_test_dump_shard*.npz')), rows)
    ids = {i['id'] for i in insts}
    sub = [r for r in rows if r['id'] in ids]
    by = {i['id']: i for i in insts}
    m_ref, _ = E.score_instances(insts, 3.0, ())
    (m_ref,) = m_ref.values()
    m, _ = score_rows(sub, [by[r['id']]['heatmap'] for r in sub], [by[r['id']]['g_pt'] for r in sub], [by[r['id']]['p_in']['io'] for r in sub])
    keys = ('auc', 'l2', 'ap_inout', 'ap_inout_dir', 'n', 'n_in_frame')
    print('omnigf_score 路径', {k: m_ref.get(k) for k in keys})
    print('score_rows 路径  ', {k: m.get(k) for k in keys})
    assert all(m[k] == m_ref[k] for k in keys), '两条路径不一致'
    print('一致')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('infer')
    p.add_argument('--split', required=True, choices=['val', 'test', 'jsonl_smoke'])
    p.add_argument('--ckpt', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--read_test', action='store_true')
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--n', type=int, default=10000)
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--limit', type=int, default=0)
    p = sp.add_parser('select')
    p.add_argument('--nshards', type=int, default=1)
    p = sp.add_parser('score')
    p.add_argument('--name', default='h0')
    p.add_argument('--nshards', type=int, default=4)
    p = sp.add_parser('check_omnigf')
    p.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    {'infer': cmd_infer, 'select': cmd_select, 'score': cmd_score, 'check_omnigf': cmd_check_omnigf}[a.cmd](a)


if __name__ == '__main__':
    main()
