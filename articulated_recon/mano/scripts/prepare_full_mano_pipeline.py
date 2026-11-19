"""Build fitting-only inputs from RGB landmarks and supplied depth/cameras.

This prepares calibrated/RGB-D diagnostics; it does not estimate cameras or depth.
Source inputs must contain detector, triangulation and initialization NPZs.
"""
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from bootstrap_runtime import bootstrap

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--upstream',type=Path,default=ROOT.parent)
    ap.add_argument('--cases',nargs='+',choices=['button','block','bottle'],default=['button','block','bottle'])
    a=ap.parse_args()
    run=a.run.resolve(); src=a.source.resolve()
    run.mkdir(parents=True,exist_ok=True)
    if run == src:raise ValueError('Use a new output run directory')
    bootstrap(run, a.upstream)
    rng=np.random.default_rng(0)
    for key in a.cases:
        out=run/'inputs'/key; out.mkdir(parents=True,exist_ok=True)
        for name in ['obs.npz','ann.json','sam3.npz','sam3-execution.json','sam3d-execution.json']:
            p=src/'inputs'/key/name
            if p.exists(): shutil.copy2(p,out/name)
        if (src/'inputs'/key/'sam3d').exists():
            shutil.copytree(src/'inputs'/key/'sam3d',out/'sam3d',dirs_exist_ok=True)
        protocol=json.loads((src/'inputs'/key/'protocol.json').read_text())
        base=src/'inputs'/key
        if not (base/'detector.npz').exists() and 'baseline_directory' in protocol:
            base=Path(protocol['baseline_directory'])
        for name in ['detector.npz','triangulation.npz','initialization.npz']:
            shutil.copy2(base/name,out/name)
        z=np.load(out/'obs.npz')
        d,K,C=z['depth_m'],z['K'],z['c2w']; T,V,H,W=d.shape
        kp=np.load(base/'detector.npz')['kp'][:,:V]
        triang=np.load(base/'triangulation.npz'); xyz=triang['xyz']
        init=np.load(base/'initialization.npz')
        assert len(init['q'])==T and kp.shape[:2]==(T,V)
        points=np.zeros((T,768,3),np.float32);views=np.zeros((T,768),np.int64);counts=np.zeros(T,np.int64)
        areas=np.zeros((T,V),np.int64)
        # RGB-only spatial selection; no fitted RGB-D surface or GT mask.
        for t in range(T):
            fp=[];fv=[];tree=cKDTree(xyz[t])
            for v in range(V):
                good=kp[t,v,:,2]>0
                if good.sum()<4:
                    # Missing RGB detections use the existing RGB-only temporal
                    # initialization solely as a search window, not as depth.
                    pc=(xyz[t]-C[v,:3,3])@C[v,:3,:3];uv=pc@K[v].T
                    xy=uv[:,:2]/np.maximum(uv[:,2:],1e-6)
                else:
                    xy=kp[t,v,good,:2]
                m=np.zeros((H,W),np.uint8)
                hull=cv2.convexHull(np.rint(xy).astype(np.int32))
                cv2.fillConvexPoly(m,hull,1);m=cv2.dilate(m,np.ones((7,7),np.uint8))
                y,x=np.nonzero((m>0)&(d[t,v]>0))
                rays=np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(K[v]).T
                p=(rays*d[t,v,y,x,None])@C[v,:3,:3].T+C[v,:3,3]
                keep=tree.query(p)[0]<.045
                p=p[keep];areas[t,v]=len(p);fp.append(p);fv.append(np.full(len(p),v))
            if not fp: continue
            p=np.concatenate(fp);vi=np.concatenate(fv)
            ix=rng.choice(len(p),min(len(p),768),replace=False);n=len(ix)
            points[t,:n]=p[ix];views[t,:n]=vi[ix];counts[t]=n
        np.savez_compressed(out/'hand-input.npz',initial_q=init['q'],initial_translation=init['translation'],
                            points=points,point_views=views,counts=counts,foreground_counts=areas)
        protocol.update(hand_fit_sha256=None,hand_method='direct pushed RobotFitter port to MANO',
                        fixed_for_ablation=['camera calibration','table plane','20 object keyframes'],
                        hand_initialization='Same RGB-only detector triangulation + IK initialization as previous run; zero shared shape',
                        hand_foreground='RGB detector convex hull (projected RGB triangulation when detection missing), 3px dilation, measured depth within 45mm of RGB-triangulated joints; not GT masks',
                        limitations=['Public RH20T depth compression remains',
                                     'MANO PCA bounds are statistical bounds, not anatomical joint limits',
                                     'Known publisher camera calibration replaces unavailable robot-proprioception ICP',
                                     'Button annotated as one static assembly; no separately specified cap joint'])
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        print(key,T,'frames',V,'training views',counts.min(),'minimum foreground points',flush=True)


if __name__=='__main__':main()
