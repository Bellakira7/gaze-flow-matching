"""Automatic face-derived head boxes and short-term, geometry-based tracking.

YuNet runs on CPU. Head boxes are expanded face detections, not a separately
trained whole-head detector. Track IDs are local to the video, not identities.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
MODEL = ROOT/'third_party/yunet/face_detection_yunet_2023mar.onnx'
MODEL_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'


def face_to_head(box, width, height, expansion=(1.5, 1.5)):
    """Pixel xywh face -> normalized, clipped xyxy head approximation."""
    x, y, w, h = map(float, box)
    cx, cy = x+w/2, y+h/2-h*.10
    hw, hh = w*expansion[0]/2, h*expansion[1]/2
    return np.clip(np.array([cx-hw, cy-hh, cx+hw, cy+hh])/[width,height,width,height],0,1)


class YuNetHeads:
    def __init__(self, score_threshold=.8, input_size=960, min_face_size=24, max_heads=8):
        import cv2
        if not hasattr(cv2,'FaceDetectorYN_create'):
            raise RuntimeError('OpenCV >= 4.8 is required; install requirements.txt')
        if not MODEL.is_file() or hashlib.sha256(MODEL.read_bytes()).hexdigest()!=MODEL_SHA256:
            raise RuntimeError('Bundled YuNet weights are missing or corrupt; restore third_party/yunet')
        cv2.setNumThreads(2)
        self.cv2=cv2
        self.detector=cv2.FaceDetectorYN_create(str(MODEL),'',(320,320),score_threshold,.3,5000)
        self.input_size=input_size
        self.min_face_size=min_face_size
        self.max_heads=max_heads
        self.last_detection_count=0

    def detect(self, rgb):
        cv2=self.cv2
        height,width=rgb.shape[:2]
        scale=min(1.,self.input_size/max(width,height))
        dw,dh=max(1,round(width*scale)),max(1,round(height*scale))
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        if (dw,dh)!=(width,height):
            bgr=cv2.resize(bgr,(dw,dh),interpolation=cv2.INTER_AREA)
        self.detector.setInputSize((dw,dh))
        _,faces=self.detector.detect(bgr)
        result=[]
        for face in [] if faces is None else faces:
            x,y,w,h=face[:4]*np.array([width/dw,height/dh,width/dw,height/dh])
            if min(w,h)<self.min_face_size:
                continue
            box=face_to_head((x,y,w,h),width,height)
            if np.any(box[2:]<=box[:2]):
                continue
            result.append(np.r_[box,float(face[-1])])
        result.sort(key=lambda row:float(row[4]),reverse=True)
        self.last_detection_count=len(result)
        return np.asarray(result[:self.max_heads],dtype=float).reshape(-1,5)


def _measurement(box):
    return np.r_[(box[:2]+box[2:])/2,box[2:]-box[:2]]


def _box(state):
    size=np.clip(state[2:4],.001,1.)
    return np.clip(np.r_[state[:2]-size/2,state[:2]+size/2],0,1)


def _iou(a,b):
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:])
    overlap=float(np.prod(np.maximum(hi-lo,0)))
    aa=float(np.prod(np.maximum(a[2:]-a[:2],0)))
    bb=float(np.prod(np.maximum(b[2:]-b[:2],0)))
    return overlap/max(aa+bb-overlap,1e-12)


@dataclass
class _Track:
    id: int
    state: np.ndarray
    covariance: np.ndarray
    score: float
    missed: int = 0
    hits: int = 1


class HeadTracker:
    """Constant-velocity Kalman filter + gated Hungarian association.

    Unmatched tracks may supply a predicted box for at most predict_gap frames.
    They remain eligible for reassociation until max_age, then expire. IDs are
    never recycled, including after a scene reset. There is no face recognition.
    """
    def __init__(self,max_age=15,predict_gap=2):
        if not 0<=predict_gap<=max_age:
            raise ValueError('Require 0 <= predict_gap <= max_age')
        self.max_age=max_age;self.predict_gap=predict_gap
        self.tracks=[];self.next_id=1
        self.F=np.eye(8);self.F[:4,4:]=np.eye(4)
        self.H=np.eye(4,8)
        self.Q=np.diag([1e-5]*4+[1e-6]*4)
        self.R=np.eye(4)*1e-4

    def reset(self):
        self.tracks.clear()

    def update(self,detections):
        detections=np.asarray(detections,dtype=float).reshape(-1,5)
        if not np.isfinite(detections).all():
            raise ValueError('Non-finite detections')
        for track in self.tracks:
            track.state=self.F@track.state
            track.covariance=self.F@track.covariance@self.F.T+self.Q
            track.missed+=1
        cost=np.full((len(self.tracks),len(detections)),1e6)
        for i,track in enumerate(self.tracks):
            box=_box(track.state)
            for j,row in enumerate(detections):
                other=row[:4]
                a=_measurement(box);b=_measurement(other)
                relative=float(np.linalg.norm(a[:2]-b[:2])/max(np.linalg.norm(a[2:]),.01))
                ratio=np.maximum(a[2:],b[2:])/np.maximum(np.minimum(a[2:],b[2:]),1e-4)
                overlap=_iou(box,other)
                if (overlap>=.05 or relative<=.5) and np.max(ratio)<3:
                    cost[i,j]=1-overlap+.25*relative
        matched=set()
        if cost.size:
            ri,ci=linear_sum_assignment(cost)
            for i,j in zip(ri,ci):
                if cost[i,j]>=1e6:
                    continue
                track=self.tracks[i];z=_measurement(detections[j,:4])
                gain=np.linalg.solve(self.H@track.covariance@self.H.T+self.R,self.H@track.covariance).T
                track.state+=gain@(z-self.H@track.state)
                residual=np.eye(8)-gain@self.H
                track.covariance=residual@track.covariance@residual.T+gain@self.R@gain.T
                track.score=float(detections[j,4]);track.missed=0;track.hits+=1
                matched.add(int(j))
        self.tracks=[t for t in self.tracks if t.missed<=self.max_age]
        # Deterministic left-to-right IDs for simultaneous new detections.
        for j in sorted(set(range(len(detections)))-matched,key=lambda k:float(detections[k,0])):
            row=detections[j]
            self.tracks.append(_Track(self.next_id,np.r_[_measurement(row[:4]),np.zeros(4)],np.eye(8)*.001,float(row[4])))
            self.next_id+=1
        output=[]
        for track in sorted(self.tracks,key=lambda t:t.id):
            box=_box(track.state)
            if track.missed>self.predict_gap or np.any(box[2:]<=box[:2]):
                continue
            output.append({'track_id':track.id,'bbox':box.tolist(),
                'source':'detected' if track.missed==0 else 'predicted',
                'last_detection_score':track.score,'missed_frames':track.missed})
        return output


class SceneCutDetector:
    """Conservative color/displacement heuristic; does not guarantee shot detection."""
    def __init__(self):
        self.previous=None

    def update(self,rgb):
        # A small fixed grid avoids an OpenCV dependency in tracker unit tests.
        ys=np.linspace(0,rgb.shape[0]-1,36).astype(int)
        xs=np.linspace(0,rgb.shape[1]-1,64).astype(int)
        small=rgb[ys][:,xs].astype(np.float32)/255
        cut=False
        if self.previous is not None:
            change=float(np.mean(np.abs(small-self.previous)))
            def histogram(image):
                bins=np.minimum((image*7).astype(int),7)
                ids=bins[...,0]*64+bins[...,1]*8+bins[...,2]
                hist=np.bincount(ids.ravel(),minlength=512).astype(float)
                return hist/hist.sum()
            a=histogram(small);b=histogram(self.previous)
            distance=float(np.sqrt(max(0,1-np.sqrt(a*b).sum())))
            cut=change>.25 and distance>.65
        self.previous=small
        return cut


class TargetSelector:
    """Default all; one-person modes lock an ID and never silently switch people."""
    def __init__(self,target='all'):
        if target not in ['all','largest','leftmost','rightmost'] and not (target.isdigit() and int(target)>0):
            raise ValueError('--target must be all, largest, leftmost, rightmost, or a positive track ID')
        self.target=target;self.locked=int(target) if target.isdigit() else None

    def reset(self):
        if not self.target.isdigit():self.locked=None

    def select(self,tracks):
        if self.target=='all':return tracks
        if self.locked is None and tracks:
            if self.target=='largest':
                chosen=max(tracks,key=lambda t:np.prod(np.diff(np.asarray(t['bbox']).reshape(2,2),axis=0)))
            else:
                chosen=sorted(tracks,key=lambda t:t['bbox'][0])[0 if self.target=='leftmost' else -1]
            self.locked=chosen['track_id']
        return [t for t in tracks if t['track_id']==self.locked]
