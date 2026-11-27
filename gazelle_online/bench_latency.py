"""干净的延迟 / 效率数字（TODO P1：之前的 flow 0.43 s/实例、B5 1.2 s/实例是与训练抢卡时测的，含竞争）。
CUDA_VISIBLE_DEVICES=<独占的一张卡> python -I -u bench_latency.py [--n_inst 40] [--warm 5] [--out results/paper/latency]
口径：测试集真实图像（取前 n_inst 个不同的帧，每帧第 1 个头），batch=1 个实例，**图像 -> 假设**的端到端；各段 cuda.synchronize 后计时，
报中位数 / p90（ms）。backbone = 冻结 DINOv2-L fp16 autocast（与训练/评测一致），解码器 fp32。设定同读测试集时：flow = Heun/16 步/γ=3，B5 = 64 步 × M_draw 64，N=10000 样本。
分段：encode（图像 -> token；B1 为图像 -> 热图；flowET 含头框 crop 的第二次 backbone 前向）、sample（token -> N 个样本；N=64/1000/10000）、hyps（样本 -> 5 个假设，CPU；MCL/B5/B1 用原生假设）。
**只报耗时，不暗示"N=64 时质量也够"：质量只在 N=10000 的协议下评过。** 开始与结束时各记一次该卡上的其它进程（独占才算"干净"）。
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

G = str(Path(__file__).resolve().parents[1])
for p in ('gazelle_online', 'gazelle_cached', 'setmetrics'):
    sys.path.insert(0, f'{G}/{p}')
from train_online import AmpBackbone  # noqa: E402
from common import IN_SIZE  # noqa: E402
import gaze_eval as GE  # noqa: E402
from gaze_models import GazeModel, HeatmapGaze, GRID  # noqa: E402
from gaze_eye import EyeGazeModel  # noqa: E402
import gaze_diffusion as GD  # noqa: E402,F401  (注册 'diff' 头)
from eval_holdout_ckpt import heat_native_hyps  # noqa: E402
from predict_test import head_box  # noqa: E402

SEL = json.load(open(f'{G}/experiments/sampler_selection.json'))
SIGMA_H = SEL['sigma_h']
METHODS = [  # 名称, 头, 实验目录, sample 的额外参数, 假设来源
    ('B1 heatmap', 'heatmap', 'b1_dinov2l_sd_s0', {}, 'native'),
    ('B3 MDN', 'mdn', 'mdn_s0', {}, 'cluster'),
    ('B4 MCL', 'mcl', 'mcl_s0', {}, 'native'),
    ('B5 heatmap diffusion', 'diff', 'diff_s0', SEL['diff'], 'native'),
    ('Ours (flow)', 'flow', 'flow_s0', SEL['flow'], 'cluster'),
    ('Ours+eye token (flowET)', 'flow_eye', 'flow_eye_s0', SEL['flow'], 'cluster'),
    ('Ours+eye token, 6-layer head (flowET_d6)', 'flow_eye', 'flow_eye_d6_s0', SEL['flow'], 'cluster'),  # C22 第 6 项
]
HEAD_KW = {'flow_eye_d6_s0': {'depth': 6}}  # 与训练 args.json 的 head_kw 一致
NS = (64, 1000, 10000)


def gpu_procs():
    """本进程所用物理卡上的所有计算进程（pid, 显存）；CUDA_VISIBLE_DEVICES 须是单张卡的序号。"""
    try:
        idx = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
        q = lambda *args: subprocess.run(['nvidia-smi', *args], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()  # noqa: E731
        uuid = q('-i', idx, '--query-gpu=uuid', '--format=csv,noheader')[0].strip()
        return [r for r in q('--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader') if r.split(',')[0].strip() == uuid]
    except Exception as e:  # noqa: BLE001
        return [f'nvidia-smi 失败: {e}']


def tm(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, (time.perf_counter() - t) * 1e3


def build(bb, head, exp):
    if head == 'heatmap':
        m = HeatmapGaze(bb)
        miss, unexp = m.load_state_dict(torch.load(f'{G}/experiments/{exp}/epoch_14.pt', map_location='cpu', weights_only=True), strict=False)
        assert not unexp and all(k.startswith('backbone.') for k in miss), (miss[:3], unexp[:3])
    else:
        kw = HEAD_KW.get(exp, {})
        m = EyeGazeModel(bb, 'flow', **kw) if head == 'flow_eye' else GazeModel(bb, head, **kw)
        m.load_decoder_state(torch.load(f'{G}/experiments/{exp}/epoch_14.pt', map_location='cpu', weights_only=True))
    return m.cuda().eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_inst', type=int, default=40)
    ap.add_argument('--warm', type=int, default=5)
    ap.add_argument('--out', default=f'{G}/results/paper/latency')
    a = ap.parse_args()
    procs0 = gpu_procs()
    real = AmpBackbone('dinov2_vitl14').cuda().eval()
    tf = real.get_transform(IN_SIZE)
    frames = json.load(open(f'{G}/data/gazefollow_test_hf/test_preprocessed.json'))
    order = list(range(a.warm + a.n_inst))
    imgs = [tf(Image.open(f'{G}/data/gazefollow_test_hf/{frames[i]["path"]}').convert('RGB')).unsqueeze(0).cuda() for i in order]
    boxes = [[[head_box(frames[i], frames[i]['heads'][0], 'gt', 0.12)]] for i in order]
    gen = torch.Generator(device='cuda').manual_seed(2026)
    res = {}
    for name, head, exp, skw, hyp_mode in METHODS:
        model = build(real, head, exp)
        npar = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(('backbone.', 'trunk.backbone.')))
        rows = {k: [] for k in ('encode', *[f'sample{n}' for n in NS], 'hyps', *[f'e2e{n}' for n in NS])}
        for j in range(a.warm + a.n_inst):
            x, bx = imgs[j], boxes[j]
            if head == 'heatmap':
                hm, t_enc = tm(lambda: torch.cat(model.forward({'images': x, 'bboxes': bx})['heatmap'], 0))
                tok = hm
            else:
                tok, t_enc = tm(lambda: model.trunk.tokens(x, bx))
            t_s = {}
            for n in NS:
                if head == 'heatmap':
                    def f(n=n):
                        h = hm.flatten(1).double().clamp_min(1e-12)
                        P = (1 - 1e-3) * h / h.sum(1, keepdim=True) + 1e-3 / h.shape[1]
                        cells = torch.multinomial(P.float(), n, replacement=True, generator=gen)
                        jit = torch.rand(cells.shape + (2,), device=cells.device, generator=gen)
                        return torch.stack([(cells % GRID + jit[..., 0]) / GRID, (cells // GRID + jit[..., 1]) / GRID], -1)
                else:
                    def f(n=n):
                        return model.head.sample(tok, n, gen=gen, **skw)
                S, t_s[n] = tm(f)
            S = S.float().cpu().numpy()  # 最后一个 n（10000）的样本，用于假设提取

            def hyps():
                if hyp_mode == 'cluster':
                    return GE.samples_to_hyps(S[0].astype(np.float64), SIGMA_H)
                if head == 'heatmap':
                    return heat_native_hyps(tok[0].float().cpu().numpy())
                return model.head.predict(tok if head != 'diff' else None)
            t0 = time.perf_counter()
            hyps()
            torch.cuda.synchronize()
            t_h = (time.perf_counter() - t0) * 1e3
            if j >= a.warm:
                rows['encode'].append(t_enc)
                rows['hyps'].append(t_h)
                for n in NS:
                    rows[f'sample{n}'].append(t_s[n])
                    rows[f'e2e{n}'].append(t_enc + t_s[n] + t_h)
        res[name] = dict(params_no_backbone=int(npar), **{k: dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)), n=len(v)) for k, v in rows.items()})
        print(f'{name:26s} enc {res[name]["encode"]["median"]:7.1f} | sample@64/1k/10k '
              + '/'.join(f'{res[name][f"sample{n}"]["median"]:.1f}' for n in NS) + f' | hyps {res[name]["hyps"]["median"]:.1f} | e2e@10k {res[name]["e2e10000"]["median"]:.1f} ms', flush=True)
        del model
        torch.cuda.empty_cache()
    procs1 = gpu_procs()
    meta = dict(device=torch.cuda.get_device_name(0), cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), n_inst=a.n_inst, warm=a.warm, batch='1 实例',
                sampler=SEL, sigma_h=SIGMA_H, gpu_procs_before=procs0, gpu_procs_after=procs1, time=time.strftime('%F %T'))
    json.dump(dict(meta=meta, res=res), open(a.out + '.json', 'w'), indent=1, ensure_ascii=False)
    L = ['| 方法 | 可训练参数(不含 backbone) | encode | sample N=64 | N=1000 | N=10000 | 假设提取 | **端到端 N=10000（中位 / p90）** |', '|---|---:|---:|---:|---:|---:|---:|---|']
    for name, r in res.items():
        L.append(f'| {name} | {r["params_no_backbone"]:,} | {r["encode"]["median"]:.0f} | ' + ' | '.join(f'{r[f"sample{n}"]["median"]:.0f}' for n in NS)
                 + f' | {r["hyps"]["median"]:.0f} | **{r["e2e10000"]["median"]:.0f} / {r["e2e10000"]["p90"]:.0f}** |')
    open(a.out + '.md', 'w').write('\n'.join(L) + '\n\n> 单位 ms，batch=1 实例，图像→假设端到端；' + f'{meta["device"]}；采集前该卡的其它进程：{procs0}；采集后：{procs1}。\n')
    print('\n'.join(L))
    print('wrote', a.out + '.json/.md')


if __name__ == '__main__':
    main()
