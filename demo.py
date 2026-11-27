"""Video-only gaze demo with automatic face-derived head detection and tracking."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CHECKPOINT_SHA256 = '200072de520ca356e69591121a63e3ef4979b046b08d4166fe3564a5e4503657'


def read_boxes(path):
    """Legacy single-person CSV; validate before allocating the gaze model."""
    import numpy as np
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or [int(r['frame']) for r in rows] != list(range(len(rows))):
        raise ValueError('CSV frame IDs must be 0,1,... without gaps')
    boxes = np.array([[float(r[k]) for k in ['xmin','ymin','xmax','ymax']] for r in rows])
    if not (np.isfinite(boxes).all() and np.all((boxes >= 0) & (boxes <= 1)) and np.all(boxes[:,2:] > boxes[:,:2])):
        raise ValueError('CSV boxes must be finite normalized xyxy with positive area')
    return boxes


def video_info(path):
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,nb_frames:stream_side_data=rotation',
        '-of','json',str(path)]))['streams'][0]
    width, height = int(probe['width']), int(probe['height'])
    rotation = next((float(s['rotation']) for s in probe.get('side_data_list',[]) if 'rotation' in s), 0)
    if round(rotation) % 180 == 90:
        width, height = height, width
    num, den = map(int, probe['avg_frame_rate'].split('/'))
    if num <= 0 or den <= 0:
        raise ValueError('Input video must report a positive average frame rate')
    count = str(probe.get('nb_frames',''))
    return width, height, num/den, int(count) if count.isdigit() else None


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/flowET_d6_s0_epoch14.pt')
    ap.add_argument('--backbone-weights', type=Path, help='Offline DINOv2 ViT-L/14 weights; otherwise official cache/download')
    ap.add_argument('--out', type=Path, help='Default: output/<video stem>_<timestamp>')
    ap.add_argument('--head-boxes', type=Path, help='Optional legacy CSV: frame,xmin,ymin,xmax,ymax (normalized, one person)')
    ap.add_argument('--target', default='all', help='all, largest, leftmost, rightmost, or a positive track ID; spatial choices lock one ID per shot')
    ap.add_argument('--det-threshold', type=float, default=.8)
    ap.add_argument('--det-size', type=int, default=960, help='Maximum detector input side in pixels')
    ap.add_argument('--min-face-size', type=int, default=24, help='Minimum face side in original video pixels')
    ap.add_argument('--max-heads', type=int, default=8, help='Maximum face detections retained per frame')
    ap.add_argument('--track-max-age', type=int, default=15, help='Retain unmatched IDs for this many frames')
    ap.add_argument('--track-predict-gap', type=int, default=2, help='Emit predicted boxes for at most this many missed frames')
    ap.add_argument('--limit', type=int, default=0, help='First N frames only (0: entire video)')
    ap.add_argument('--samples', type=int, default=10000, help='Flow samples per person per frame')
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    if args.samples < 1 or args.limit < 0 or args.det_size < 32 or args.min_face_size < 1 or args.max_heads < 1:
        ap.error('Invalid sample/frame/detector limits')
    if not 0 < args.det_threshold <= 1 or not 0 <= args.track_predict_gap <= args.track_max_age:
        ap.error('Require 0 < det-threshold <= 1 and 0 <= track-predict-gap <= track-max-age')
    if args.head_boxes and args.target != 'all':
        ap.error('--target applies to automatic detection, not supplied --head-boxes')
    for binary in ['ffmpeg','ffprobe']:
        if not shutil.which(binary):
            ap.error(f'{binary} is required on PATH')
    for path in [args.video,args.checkpoint,args.head_boxes,args.backbone_weights]:
        if path is not None and not path.is_file():
            ap.error(f'Missing input: {path}')
    if args.out is None:
        args.out = ROOT/'output'/f'{args.video.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if args.out.exists() and any(args.out.iterdir()):
        ap.error('Output directory is not empty; choose a new --out directory')
    return args


def main():
    args = parse_args()
    print('config: '+json.dumps(vars(args),default=str), flush=True)
    # Adapted from experiment-protocol eval template: prevent injected DDP state.
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):
        os.environ.pop(key, None)
    if args.backbone_weights:
        os.environ['DINOV2_WEIGHTS'] = str(args.backbone_weights.resolve())
    sys.path.insert(0, str(ROOT/'gazelle_online'))
    import numpy as np
    from PIL import Image
    import torch
    from train_online import AmpBackbone
    from gaze_eye import EyeGazeModel
    import gaze_eval as GE
    from head_tracking import YuNetHeads, HeadTracker, TargetSelector, SceneCutDetector, MODEL_SHA256
    from visualization import sha, render_tracks, video_writer

    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA-compatible GPU is required')
    checkpoint_hash = sha(args.checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint does not match the released flowET_d6 seed-0 weights')
    boxes = read_boxes(args.head_boxes) if args.head_boxes else None
    selector = TargetSelector(args.target)
    tracker = HeadTracker(args.track_max_age, args.track_predict_gap)
    cut_detector = SceneCutDetector()
    detector = YuNetHeads(args.det_threshold,args.det_size,args.min_face_size,args.max_heads) if boxes is None else None
    width, height, fps, frame_count = video_info(args.video)
    if boxes is not None and frame_count is not None and len(boxes) != frame_count:
        raise ValueError('Provide exactly one head box per video frame')
    total = min(frame_count,args.limit) if frame_count and args.limit else (args.limit or frame_count)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    backbone = AmpBackbone('dinov2_vitl14').cuda().eval()
    model = EyeGazeModel(backbone,'flow',depth=6).cuda().eval()
    model.load_decoder_state(torch.load(args.checkpoint,map_location='cpu',weights_only=True))
    transform = backbone.get_transform((448,448))
    print('model ready: checkpoint hash verified; '+torch.cuda.get_device_name(0), flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    pred_dir = args.out/'predictions'
    pred_dir.mkdir()
    decode = ['ffmpeg','-v','error','-threads','2','-i',str(args.video),'-map','0:v:0','-an','-sn','-dn']
    if args.limit:
        decode += ['-frames:v',str(args.limit)]
    decode += ['-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','-']
    reader = subprocess.Popen(decode, stdout=subprocess.PIPE)
    writer = video_writer(args.out/'demo.partial.mp4',width,height+112,fps)
    frames, forward_times, track_times, selected_ids = [], [], [], set()
    index = 0
    started = time.monotonic()
    try:
        with (args.out/'head_tracks.csv').open('w',newline='',encoding='utf-8') as stream, torch.inference_mode():
            csv_writer = csv.writer(stream)
            csv_writer.writerow(['frame','track_id','xmin','ymin','xmax','ymax','source','last_detection_score','missed_frames','selected'])
            while not args.limit or index < args.limit:
                raw = reader.stdout.read(width*height*3)
                if not raw:
                    break
                if len(raw) != width*height*3:
                    raise RuntimeError(f'Truncated video frame {index}')
                image = Image.frombytes('RGB',(width,height),raw)
                tick = time.monotonic()
                cut, det_count = False, None
                if detector is not None:
                    rgb = np.asarray(image)
                    cut = cut_detector.update(rgb)
                    if cut:
                        tracker.reset()
                        selector.reset()
                    tracks = tracker.update(detector.detect(rgb))
                    det_count = detector.last_detection_count
                    selected = selector.select(tracks)
                else:
                    if index >= len(boxes):
                        raise ValueError('CSV ended before video')
                    tracks = [dict(track_id=1,bbox=boxes[index].tolist(),source='supplied',last_detection_score=None,missed_frames=0)]
                    selected = tracks
                track_times.append(time.monotonic()-tick)
                ids = [t['track_id'] for t in selected]
                selected_ids.update(ids)
                for track in tracks:
                    csv_writer.writerow([index,track['track_id'],*track['bbox'],track['source'],
                        track['last_detection_score'],track['missed_frames'],int(track['track_id'] in ids)])
                predictions, hypotheses, masses = [], [], []
                if selected:
                    x = transform(image).unsqueeze(0).cuda()
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    tokens = model.trunk.tokens(x,[[t['bbox'] for t in selected]])
                    # One head per sampler call bounds attention memory and keeps
                    # random draws independent of changes in the number of people.
                    for person in range(len(selected)):
                        gen = torch.Generator(device='cuda').manual_seed(args.seed)
                        samples = model.head.sample(tokens[person:person+1],args.samples,gen=gen,
                            steps=16,gamma=3.0,solver='heun')[0].float().cpu().numpy()
                        hyps, weights = GE.samples_to_hyps(samples.astype(np.float64),1.0)
                        if not (np.isfinite(samples).all() and np.isfinite(hyps).all() and np.isfinite(weights).all() and np.allclose(weights.sum(),1)):
                            raise RuntimeError(f'Invalid prediction at frame {index}, track {ids[person]}')
                        predictions.append(samples)
                        hypotheses.append(hyps)
                        masses.append(weights)
                    forward_times.append(time.monotonic()-tick)
                samples = np.asarray(predictions,dtype=np.float32).reshape(-1,args.samples,2)
                hyps = np.asarray(hypotheses,dtype=np.float32).reshape(-1,5,2)
                weights = np.asarray(masses,dtype=np.float32).reshape(-1,5)
                # Stream per-frame arrays: prediction RAM does not grow with video length.
                np.savez_compressed(pred_dir/f'{index:06d}.npz',frame=index,track_ids=np.asarray(ids,dtype=np.int64),
                    boxes=np.asarray([t['bbox'] for t in selected],dtype=np.float32).reshape(-1,4),samples=samples,hyps=hyps,w=weights)
                frame = render_tracks(image,selected,samples,hyps,weights,index,total,fps,automatic=detector is not None)
                writer.stdin.write(frame.tobytes())
                if index == 0 or (total and index in [total//2,total-1]):
                    frame.save(args.out/f'preview_{index:04d}.jpg',quality=92)
                frames.append(dict(frame=index,scene_cut=cut,detections=det_count,emitted_tracks=len(tracks),
                    selected_ids=ids,predicted_boxes=sum(t['source']=='predicted' for t in selected)))
                if index % 10 == 0 or (total and index == total-1):
                    print(f'[{index+1}/{total or "?"}] heads={len(selected)} elapsed={time.monotonic()-started:.1f}s',flush=True)
                index += 1
            reader.stdout.close()
            writer.stdin.close()
            read_status, write_status = reader.wait(), writer.wait()
            if read_status or write_status:
                raise RuntimeError(f'ffmpeg failed: reader={read_status}, writer={write_status}')
            if not index:
                raise ValueError('No video frames decoded')
            if frame_count and index != (min(frame_count,args.limit) if args.limit else frame_count):
                raise ValueError(f'Expected {total} frames, decoded {index}')
            if boxes is not None and not args.limit and index != len(boxes):
                raise ValueError('CSV/video length mismatch')
    finally:
        # Do not wait on a decoder blocked on its stdout pipe after an exception.
        for process in (reader,writer):
            if process.poll() is None:
                process.terminate()
        for process in (reader,writer):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    (args.out/'demo.partial.mp4').rename(args.out/'demo.mp4')
    frame.save(args.out/f'preview_{index-1:04d}.jpg',quality=92)
    (args.out/'frames.json').write_text(json.dumps(frames,indent=2)+'\n')
    stats = {'model':'flowET_d6','training_seed':0,'epoch':14,'checkpoint_sha256':checkpoint_hash,
        'video_sha256':sha(args.video),'head_boxes_sha256':sha(args.head_boxes) if args.head_boxes else None,
        'frames':index,'output_fps':fps,'seconds':index/fps,'samples_per_person_frame':args.samples,
        'frame_seed':args.seed,'gaze_temporal_smoothing':False,'head_box_smoothing':detector is not None,
        'head_boxes':'YuNet faces expanded 1.5x, center shifted up 0.1 face height; Kalman/Hungarian tracking' if detector else 'supplied normalized track',
        'detector_sha256':MODEL_SHA256 if detector else None,'target':args.target,
        'track_ids':sorted(selected_ids),'person_frames':sum(len(f['selected_ids']) for f in frames),
        'frames_without_selected_heads':sum(not f['selected_ids'] for f in frames),
        'predicted_person_frames':sum(f['predicted_boxes'] for f in frames),
        'scene_cut_frames':[f['frame'] for f in frames if f['scene_cut']],
        'gaze_annotations_used':False,'wall_seconds':time.monotonic()-started,
        'median_forward_seconds_per_nonempty_frame':float(np.median(forward_times)) if forward_times else None,
        'median_detection_tracking_seconds':float(np.median(track_times)),
        'output_sha256':sha(args.out/'demo.mp4'),'config':vars(args),
        'output_timing':'one output frame per decoded frame, constant average input fps; audio omitted'}
    (args.out/'stats.json').write_text(json.dumps(stats,indent=2,default=str)+'\n')
    print('done: '+str(args.out/'demo.mp4'),flush=True)


if __name__ == '__main__':
    main()
