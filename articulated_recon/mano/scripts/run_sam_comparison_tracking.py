"""Amani seg_track.py adapted only for the isolated input/output layout.

Seed each fit camera from the preliminary fit's selected SAM3 instance. Run the
actual SAM3 video tracker forwards and backwards over the same 20 keyframes.
Static objects are excluded by Amani's method, explicitly recorded here.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--case',choices=['all','button','block','bottle'],default='all')
    ap.add_argument('--seed-arm',default='sam3d-upstream')
    ap.add_argument('--all-frames',action='store_true',help='Track every source frame and retain the same object keyframes for fitting')
    args=ap.parse_args();run=args.run.resolve()
    sys.path.insert(0,str(run/'runtime'))
    from amani_comparison.objects_v2 import Sam3Masks
    model=None;proc=None
    for key in ['button','block','bottle']:
        if args.case not in ('all',key):continue
        case=run/'inputs'/key;outdir=run/'results'/args.seed_arm/key
        meta=json.loads((case/'protocol.json').read_text());ids=meta['keyframe_ids']
        source_rgb=np.load(case/'obs.npz')['rgb']
        rgb=source_rgb if args.all_frames else source_rgb[ids];T,V,H,W=rgb.shape[:4]
        retained={int(t):k for k,t in enumerate(ids)} if args.all_frames else {k:k for k in range(len(ids))}
        with open(outdir/'objects-fit.pkl','rb') as f:res=pickle.load(f)
        sam=Sam3Masks(outdir/'sam3-keyframes.npz',H,W)
        prompts=[];rec={k:[] for k in ['p','t','v','score','box','mask']};seeds=[]
        started=time.monotonic()
        for ob in res:
            if ob['type']!='free':continue
            if model is None:
                from transformers import Sam3TrackerVideoModel,Sam3TrackerVideoProcessor
                model=Sam3TrackerVideoModel.from_pretrained('facebook/sam3',dtype=torch.bfloat16).to('cuda').eval()
                proc=Sam3TrackerVideoProcessor.from_pretrained('facebook/sam3')
            pi=len(prompts);prompts.append('__track__'+ob['name'])
            for v in range(V):
                seed=next(((t,s[v]) for t,s in sorted(ob['sel'].items()) if v in s),None)
                if seed is None:continue
                seed_keyframe,sid=seed;ts=int(ids[seed_keyframe]) if args.all_frames else seed_keyframe;m0=sam.mask(sid,raw=True)
                seeds.append(dict(object=ob['name'],view=v,keyframe=int(seed_keyframe),source_frame=int(ids[seed_keyframe]),instance=int(sid)))
                with torch.no_grad():
                    sess=proc.init_video_session(video=[rgb[t,v] for t in range(T)],inference_device='cuda',dtype=torch.bfloat16)
                    proc.add_inputs_to_inference_session(inference_session=sess,frame_idx=ts,obj_ids=1,input_masks=m0.astype(np.float32))
                    outs={}
                    for rev in (False,True):
                        for output in model.propagate_in_video_iterator(sess,start_frame_idx=ts,reverse=rev):
                            mk=proc.post_process_masks([output.pred_masks],original_sizes=[[H,W]],binarize=True)[0]
                            outs[output.frame_idx]=mk[0,0].cpu().numpy().astype(bool) if mk.ndim==4 else mk[0].cpu().numpy().astype(bool)
                for t,m in outs.items():
                    if t not in retained:continue
                    if m.sum()<15:continue
                    ys,xs=np.nonzero(m)
                    for k,val in [('p',pi),('t',retained[t]),('v',v),('score',1.),('box',np.array([xs.min(),ys.min(),xs.max(),ys.max()],np.float32)),('mask',np.packbits(m))]:rec[k].append(val)
        n=len(rec['p'])
        np.savez_compressed(outdir/'sam3_track.npz',prompts=np.array(prompts or ['__none__']),
                            p=np.array(rec['p'],int),t=np.array(rec['t'],int),v=np.array(rec['v'],int),score=np.array(rec['score']),
                            box=np.array(rec['box']).reshape(-1,4),mask=np.array(rec['mask'],np.uint8).reshape(n,(H*W+7)//8),shape=np.array([H,W]))
        audit=dict(status='complete',masks=n,seeds=seeds,seconds=time.monotonic()-started,
                   tracker_used=bool(prompts),static_objects_skipped=[o['name'] for o in res if o['type']!='free'],
                   transformers=importlib.metadata.version('transformers') if model is not None else None,source='Amani seg_track.py API and seeding logic; path adaptation',
                   heldout_camera_used=False,source_frame_ids=ids,source_frames_processed=T,
                   learned_SAM3_video_tracker=bool(prompts),tracker='SAM3 learned video tracker',all_source_frames=args.all_frames)
        (outdir/'sam3-track-execution.json').write_text(json.dumps(audit,indent=2))
        print(key,json.dumps(audit),flush=True)


if __name__=='__main__':main()
