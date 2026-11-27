from pathlib import Path
import json, subprocess, sys
import numpy as np
from PIL import Image
ROOT=Path('/cpfs01/shared/public/groups/closeloop/ace.yang/gaze/demos/AUTO-HEAD-20261127')
sys.path.insert(0,str(ROOT/'code'))
from demo import video_info
from visualization import video_writer,render_tracks
odd=ROOT/'odd.mp4'
writer=video_writer(odd,321,353,30)
for i in range(2):writer.stdin.write(Image.new('RGB',(321,353),(i*100,0,0)).tobytes())
writer.stdin.close();assert writer.wait()==0
assert video_info(odd)==(322,354,30.0,2)
rotated=ROOT/'rotated.mp4'
subprocess.run(['ffmpeg','-v','error','-y','-display_rotation','90','-i',str(ROOT/'blank.mp4'),'-c','copy',str(rotated)],check=True)
w,h,fps,n=video_info(rotated)
raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(rotated),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'])
assert (w,h)==(360,640) and len(raw)==w*h*3
out={'odd_dimensions_pad_to_even':True,'rotated_phone_video_dimensions':True,'rotated_dimensions':[w,h]}
(ROOT/'io_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
