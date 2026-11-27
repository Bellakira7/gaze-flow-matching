from pathlib import Path
import hashlib
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
import matplotlib

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def video_writer(path, width, height, fps):
    return subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps),
        '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast',
        '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)

def render(image, box, samples, hyps, weights, frame_num, count, total, fps):
    w, h = image.size
    xy = (np.clip(samples.astype(float), 0, 1-1e-9)*64).astype(int)
    hist = np.bincount(xy[:, 1]*64+xy[:, 0], minlength=4096).reshape(64, 64)
    density = gaussian_filter(hist.astype(float), 1.5)
    density /= max(density.max(), 1e-12)
    rgba = matplotlib.colormaps['magma'](density)
    rgba[..., 3] = np.where(density > .04, .6*density**.55, 0)
    overlay = Image.fromarray((rgba*255).astype(np.uint8)).resize((w,h), Image.Resampling.BILINEAR)
    result = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
    d = ImageDraw.Draw(result)
    font = lambda size: ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'), size)
    bb = np.asarray(box)*[w,h,w,h]
    d.rectangle(tuple(bb), outline='#54dfdc', width=4)
    center = ((bb[0]+bb[2])/2, (bb[1]+bb[3])/2)
    top = tuple(hyps[0]*[w,h])
    d.line([center,top], fill='#ffce68', width=3)
    for k in reversed(range(5)):
        x,y = hyps[k]*[w,h]
        radius = 8 + 10*np.sqrt(weights[k])
        d.ellipse((x-radius,y-radius,x+radius,y+radius), fill='#0c1424',outline='#ffce68',width=2)
        d.text((x,y),str(k+1),font=font(16),fill='#ffce68',anchor='mm')
    canvas = Image.new('RGB',(w,h+112),'#0c1424')
    canvas.paste(result,(0,64))
    d = ImageDraw.Draw(canvas)
    d.text((20,8),'flowET_d6  |  Continuous-video inference',font=font(24),fill='#f3f6fb')
    d.text((20,39),f'seed 0 / epoch 14   |   frame {frame_num}   |   {count+1}/{total}   |   playback {fps:g} fps',font=font(16),fill='#a9b7cc')
    d.text((20,h+72),'Cyan: supplied head track   |   Gold: predicted gaze targets   |   No temporal smoothing',font=font(17),fill='#e0e7f2')
    d.text((20,h+95),f'{len(samples):,} samples/frame  |  Heun 16 steps  |  Ground-truth gaze is not an input',font=font(12),fill='#a9b7cc')
    return canvas
