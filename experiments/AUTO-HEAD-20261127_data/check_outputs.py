"""Engineering artifact checks; adapted from the experiment-protocol eval template."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
BASE=Path('/cpfs01/shared/public/groups/closeloop/ace.yang/gaze/demos/AUTO-HEAD-20261127')

def check(name,expected):
    root=BASE/name
    stats=json.loads((root/'stats.json').read_text())
    frames=json.loads((root/'frames.json').read_text())
    rows=list(csv.DictReader((root/'head_tracks.csv').open()))
    assert stats['frames']==len(frames)==expected
    assert [f['frame'] for f in frames]==list(range(expected))
    files=sorted((root/'predictions').glob('*.npz'))
    assert len(files)==expected
    by_frame={i:[] for i in range(expected)}
    for r in rows:
        b=np.array([float(r[k]) for k in ['xmin','ymin','xmax','ymax']])
        assert np.isfinite(b).all() and np.all((b>=0)&(b<=1)) and np.all(b[2:]>b[:2])
        assert r['source'] in ('detected','predicted','supplied')
        assert 0<int(r['missed_frames'])<=2 if r['source']=='predicted' else int(r['missed_frames'])==0
        if int(r['selected']):by_frame[int(r['frame'])].append(r)
    person_frames=0
    for i,path in enumerate(files):
        with np.load(path,allow_pickle=False) as p:
            ids=p['track_ids'].tolist()
            assert ids==frames[i]['selected_ids']==[int(r['track_id']) for r in by_frame[i]]
            assert len(ids)==len(set(ids))
            assert int(p['frame'])==i
            n=len(ids);person_frames+=n
            assert p['boxes'].shape==(n,4)
            assert p['samples'].shape==(n,10000,2)
            assert p['hyps'].shape==(n,5,2) and p['w'].shape==(n,5)
            assert all(np.isfinite(p[k]).all() for k in ['boxes','samples','hyps','w'])
            assert np.allclose(p['w'].sum(axis=1),1) and np.all(p['w']>=0)
            if n:
                np.testing.assert_allclose(p['boxes'],[[float(r[k]) for k in ['xmin','ymin','xmax','ymax']] for r in by_frame[i]],atol=1e-7)
    assert person_frames==stats['person_frames']
    assert stats['output_sha256']==hashlib.sha256((root/'demo.mp4').read_bytes()).hexdigest()
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height,avg_frame_rate,duration','-of','json',str(root/'demo.mp4')]))['streams'][0]
    assert int(probe['nb_read_frames'])==expected
    assert abs(float(probe['duration'])-expected/stats['output_fps'])<.001
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(root/'demo.mp4'),'-f','null','-'],check=True)
    if name=='blank_smoke':
        assert person_frames==0 and stats['frames_without_selected_heads']==expected
    if name=='supplied_smoke':
        assert person_frames==expected and all(r['source']=='supplied' for r in rows)
    track_summary={}
    for r in rows:
        key=r['track_id']
        t=track_summary.setdefault(key,{'first_frame':int(r['frame']),'last_frame':int(r['frame']),'person_frames':0,'predicted_frames':0})
        t['last_frame']=int(r['frame']);t['person_frames']+=1;t['predicted_frames']+=r['source']=='predicted'
    return {'run':name,'checks_passed':True,'decoded_frames':expected,'person_frames':person_frames,
        'output_probe':probe,'tracks':track_summary,'stats':stats}

print('config: '+json.dumps({'source':str(BASE),'protocol':'automatic-demo-engineering-v1','aggregation':'per-run frame/person-frame counts; no accuracy score'}),flush=True)
results=[check(name,n) for name,n in [('auto_smoke',2),('supplied_smoke',2),('blank_smoke',2),('full',360)]]
result={'protocol':'automatic-demo-engineering-v1','dataset':'VideoAttentionTarget MLB Interview 18341_18942 excerpt (360 frames) and synthetic black input','aggregation':'per-run frame/person-frame counts; no accuracy score','results':results}
(BASE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({r['run']:{'frames':r['decoded_frames'],'person_frames':r['person_frames'],'tracks':r['tracks']} for r in results},indent=2),flush=True)
