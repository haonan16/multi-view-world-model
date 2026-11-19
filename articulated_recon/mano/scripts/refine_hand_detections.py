"""Re-detect hands in observation-derived crops, preserving original coordinates.

Crops are determined by previous RGB detections, never fitted or GT geometry.
Candidate agreement is measured in original image pixels, not handedness scores.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np,mediapipe as mp
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def candidates(detector,rgb,previous):
    good=previous[:,2]>0
    if good.sum()<5:return []
    H,W=rgb.shape[:2];xy=previous[good,:2];center=(xy.min(0)+xy.max(0))/2
    radius=max(40,float(np.ptp(xy,axis=0).max())*.85)
    out=[]
    for angle in [0,90,180,270]:
        M=cv2.getRotationMatrix2D(tuple(center),angle,384/(2*radius));M[:,2]+=np.array([192,192])-center
        im=cv2.warpAffine(rgb,M,(384,384),borderMode=cv2.BORDER_CONSTANT)
        result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(im)))
        inv=cv2.invertAffineTransform(M)
        for lm,handed in zip(result.hand_landmarks,result.handedness):
            p=np.array([[x.x*384,x.y*384] for x in lm]);p=p@inv[:,:2].T+inv[:,2]
            if np.linalg.norm(p[0]-previous[0,:2])>radius*.8:continue
            out.append(dict(angle=angle,xy=p,side=handed[0].category_name))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--detector-model',type=Path,required=True);p.add_argument('--detector-input',type=Path);p.add_argument('--frames',type=int,nargs='+',required=True);a=p.parse_args()
    case=a.run/'inputs'/a.case;z=np.load(case/'obs.npz');rgb=z['rgb']
    detector_input=a.detector_input or case/'detector.npz'
    kp=np.load(detector_input)['kp'][:,:rgb.shape[1]]
    out=a.run/'diagnostics'/'crop-detector'/a.case;out.mkdir(parents=True,exist_ok=True)
    opts=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(a.detector_model)),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.3,min_hand_presence_confidence=.3)
    arrays={}
    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as det:
        for t in a.frames:
            rows=[]
            for v in range(rgb.shape[1]):
                cs=candidates(det,rgb[t,v],kp[t,v]);arrays[f'{t}_{v}']=np.array([c['xy'] for c in cs]);tiles=[]
                if not cs:continue
                xy=kp[t,v,:,:2];lo=np.maximum(0,np.floor(xy.min(0)-35).astype(int));hi=np.minimum(rgb.shape[3:1:-1],np.ceil(xy.max(0)+35).astype(int))
                for label,pts in [('Original',xy)]+[(str(c['angle'])+' '+c['side'],c['xy']) for c in cs]:
                    im=rgb[t,v].copy()
                    for j in range(5,9):
                        x,y=np.rint(pts[j]).astype(int);cv2.circle(im,(x,y),2,(255,0,0),-1)
                    crop=cv2.resize(im[lo[1]:hi[1],lo[0]:hi[0]],(240,240));cv2.putText(crop,f'v{v} {label}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);tiles.append(crop)
                while len(tiles)<5:tiles.append(np.zeros((240,240,3),np.uint8))
                rows.append(np.concatenate(tiles[:5],1))
            if rows:Image.fromarray(np.concatenate(rows,0)).save(out/f'{t:03d}.jpg')
    np.savez_compressed(out/'candidates.npz',**arrays)
if __name__=='__main__':main()
