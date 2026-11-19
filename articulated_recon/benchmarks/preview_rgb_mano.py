"""Observed RGB, fresh RGB/depth seed, and actual port output; no oracle overlay."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'mano')]


def preview(run,case,models):
    import cv2
    import imageio.v2 as imageio
    import torch
    from mano_recon.model import MANOHand,translation_base
    from mano_recon.render_fast import render
    run,models=Path(run).resolve(),Path(models).resolve()
    inp=run/'inputs'/case;fitdir=run/'hand-results'/case
    receipt=json.loads((fitdir/'execution.json').read_text())
    if receipt['status']!='complete':raise ValueError('Actual port has not completed')
    with np.load(inp/'obs.npz') as z:rgb,k,c,times=[z[n].copy() for n in ['rgb','K','c2w','timestamps']]
    with np.load(inp/'hand-input.npz') as z:q,tr=z['initial_q'].copy(),z['initial_translation'].copy()
    with np.load(inp/'human-evidence.npz') as z:kp,visible=z['keypoints'].copy(),z['visible'].copy()
    with np.load(fitdir/'fit.npz') as z:vertices,joints=z['vertices'].copy(),z['joints'].copy()
    side=json.loads((inp/'protocol.json').read_text())['mano_side']
    hand=MANOHand(models,side=side,components=q.shape[1]-3,device='cpu')
    with torch.no_grad():seed,seed_j=hand(torch.tensor(q),base=translation_base(torch.tensor(tr)))
    seed,seed_j=seed.numpy(),seed_j.numpy();faces=hand.faces.numpy()
    media=run/'media';media.mkdir(exist_ok=True)
    fps=float(1/np.median(np.diff(times)));h,w=rgb.shape[2:4]
    output=media/(case+'-source-seed-fit.mp4');samples=set(np.rint(np.linspace(0,len(times)-1,4)).astype(int));stills=[]
    with imageio.get_writer(output,fps=fps,codec='libx264',quality=8,macro_block_size=1) as writer:
        for t,source in enumerate(rgb[:,0]):
            panels=[source.copy()]
            for vs,color in [(seed[t],(80,180,255)),(vertices[t],(80,240,120))]:
                panel=source.copy()
                if visible[t]:
                    mesh,depth=render(vs,faces,k[0],c[0],size=(h,w),color=color)
                    valid=depth>0
                    panel[valid]=np.clip(.35*panel[valid]+.65*mesh[valid],0,255).astype(np.uint8)
                else:cv2.putText(panel,'Unobserved',(15,h-20),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,90,90),2)
                panels.append(panel)
            for title,panel in zip(['Recorded RGB','Fresh RGB/depth seed','Amani port fit'],panels):
                cv2.rectangle(panel,(0,0),(w,35),(20,20,20),-1)
                cv2.putText(panel,title,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
            row=np.concatenate(panels,axis=1);writer.append_data(row)
            if t in samples:stills.append(row)
    imageio.imwrite(media/(case+'-source-seed-fit.png'),np.concatenate(stills,axis=0))
    def repro(j):
        p=j@k[0].T;uv=p[...,:2]/np.maximum(p[...,2:],.01)
        return np.linalg.norm(uv-kp[:,0,:,:2],axis=-1).mean(-1)
    before,after=repro(seed_j),repro(joints)
    report=dict(case=case,visible_frames=int(visible.sum()),frames=len(visible),mano_side=side,
        detector_reprojection_mean_px_before=float(before[visible].mean()),detector_reprojection_mean_px_after=float(after[visible].mean()),
        detector_reprojection_median_px_before=float(np.median(before[visible])),detector_reprojection_median_px_after=float(np.median(after[visible])),
        interpretation='Agreement with predicted RGB landmarks, not GT geometry/contact accuracy',
        per_frame_before_px=before.tolist(),per_frame_after_px=after.tolist(),video=str(output),
        overlay='Complete hand mesh overlay, without depth-occlusion hiding; missing frames not rendered')
    (media/(case+'-reprojection.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=['bottle','block']);a=p.parse_args()
    for case in a.cases:preview(a.run,case,a.models)
