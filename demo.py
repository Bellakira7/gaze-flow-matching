"""Continuous-video inference with the paper's six-layer flow checkpoint.

Head tracks are supplied separately as normalized, zero-based per-frame boxes.
No ground-truth gaze coordinates are used. Requires ffmpeg and ffprobe on PATH.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--head-boxes', type=Path, required=True, help='CSV: frame,xmin,ymin,xmax,ymax; normalized coordinates; one person')
    ap.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/flowET_d6_s0_epoch14.pt')
    ap.add_argument('--backbone-weights', type=Path, help='Offline DINOv2 ViT-L/14 .pth; otherwise use official download/cache')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0, help='Optional first N frames for a quick check')
    ap.add_argument('--samples', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    print('config: '+json.dumps(vars(args),default=str),flush=True)
    if args.samples < 1 or args.limit < 0:
        ap.error('--samples must be positive and --limit must be nonnegative')
    for binary in ['ffmpeg','ffprobe']:
        if not shutil.which(binary):
            ap.error(f'{binary} is required on PATH')
    for path in [args.video,args.head_boxes,args.checkpoint]:
        if not path.is_file():
            ap.error(f'Missing input: {path}')
    if (args.out/'demo.mp4').exists():
        ap.error('Output demo.mp4 already exists; choose a new --out directory')
    if args.backbone_weights:
        if not args.backbone_weights.is_file():
            ap.error(f'Missing backbone weights: {args.backbone_weights}')
        os.environ['DINOV2_WEIGHTS']=str(args.backbone_weights.resolve())
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(ROOT/'gazelle_online'))
    import numpy as np
    from PIL import Image
    import torch
    from train_online import AmpBackbone
    from gaze_eye import EyeGazeModel
    import gaze_eval as GE
    from visualization import sha,render,video_writer

    if not torch.cuda.is_available():
        ap.error('A CUDA-compatible GPU is required for this release')
    checkpoint_hash=sha(args.checkpoint)
    expected='200072de520ca356e69591121a63e3ef4979b046b08d4166fe3564a5e4503657'
    if checkpoint_hash!=expected:
        ap.error('Checkpoint does not match the released flowET_d6 seed-0 weights')
    with args.head_boxes.open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    assert [int(r['frame']) for r in rows]==list(range(len(rows))), 'CSV frame IDs must be 0,1,... without gaps'
    boxes=np.array([[float(r[k]) for k in ['xmin','ymin','xmax','ymax']] for r in rows])
    assert boxes.shape==(len(rows),4) and len(rows)>0
    assert np.isfinite(boxes).all() and np.all((boxes>=0)&(boxes<=1))
    assert np.all(boxes[:,2:]>boxes[:,:2]), 'Head boxes must have positive width and height'
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,nb_frames','-of','json',str(args.video)]))['streams'][0]
    width,height=int(probe['width']),int(probe['height'])
    num,den=map(int,probe['avg_frame_rate'].split('/'))
    if num<=0 or den<=0:
        ap.error('Input video must report a positive frame rate')
    fps=num/den
    recorded_count=str(probe.get('nb_frames',''))
    frame_count=int(recorded_count) if recorded_count.isdigit() else len(rows)
    assert len(rows)==frame_count, 'Provide exactly one head box per video frame'
    total=min(frame_count,args.limit) if args.limit else frame_count
    if width%2 or height%2:
        ap.error('Input width and height must be even for H.264 yuv420p output')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    backbone=AmpBackbone('dinov2_vitl14').cuda().eval()
    model=EyeGazeModel(backbone,'flow',depth=6).cuda().eval()
    model.load_decoder_state(torch.load(args.checkpoint,map_location='cpu',weights_only=True))
    transform=backbone.get_transform((448,448))
    print('model ready: checkpoint hash verified; '+torch.cuda.get_device_name(0),flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    reader=subprocess.Popen(['ffmpeg','-v','error','-threads','2','-i',str(args.video),
        '-frames:v',str(total),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    writer=video_writer(args.out/'demo.mp4',width,height+112,fps)
    predictions=[];hypotheses=[];masses=[];forward_times=[]
    started=time.monotonic()
    try:
        with torch.inference_mode():
            for index in range(total):
                raw=reader.stdout.read(width*height*3)
                assert len(raw)==width*height*3, f'Video ended before frame {index}'
                image=Image.frombytes('RGB',(width,height),raw)
                x=transform(image).unsqueeze(0).cuda()
                gen=torch.Generator(device='cuda').manual_seed(args.seed)
                torch.cuda.synchronize();tick=time.monotonic()
                tokens=model.trunk.tokens(x,[[boxes[index].tolist()]])
                samples=model.head.sample(tokens,args.samples,gen=gen,steps=16,gamma=3.0,solver='heun')[0].float().cpu().numpy()
                forward_times.append(time.monotonic()-tick)
                hyps,weights=GE.samples_to_hyps(samples.astype(np.float64),1.0)
                assert np.isfinite(samples).all() and np.isfinite(hyps).all() and np.allclose(weights.sum(),1)
                predictions.append(samples);hypotheses.append(hyps);masses.append(weights)
                frame=render(image,boxes[index],samples,hyps,weights,index,index,total,fps)
                writer.stdin.write(frame.tobytes())
                if index in [0,total//2,total-1]:
                    frame.save(args.out/f'preview_{index:04d}.jpg',quality=92)
                if index%10==0 or index==total-1:
                    print(f'[{index+1}/{total}] elapsed={time.monotonic()-started:.1f}s',flush=True)
    finally:
        reader.stdout.close();writer.stdin.close()
        read_status=reader.wait();write_status=writer.wait()
    assert read_status==write_status==0
    np.savez_compressed(args.out/'predictions.npz',samples=np.stack(predictions),hyps=np.stack(hypotheses),w=np.stack(masses),frames=np.arange(total))
    stats={'model':'flowET_d6','training_seed':0,'epoch':14,'checkpoint_sha256':checkpoint_hash,
        'video_sha256':sha(args.video),'head_boxes_sha256':sha(args.head_boxes),
        'frames':total,'output_fps':fps,'seconds':total/fps,'samples_per_frame':args.samples,
        'frame_seed':args.seed,'temporal_smoothing':False,'head_boxes':'supplied normalized track',
        'gaze_annotations_used':False,'wall_seconds':time.monotonic()-started,
        'median_forward_seconds':float(np.median(forward_times)),
        'output_sha256':sha(args.out/'demo.mp4')}
    (args.out/'stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    print('done: '+str(args.out/'demo.mp4'),flush=True)


if __name__=='__main__':
    main()
