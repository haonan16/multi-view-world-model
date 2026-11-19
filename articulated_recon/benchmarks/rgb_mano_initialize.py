"""Prepare a fresh hand-only RobotFitter run from learned RGB geometry.

MediaPipe relative3D supplies pose hypotheses; learned source depth supplies
wrist translation. SAM3 hand pixels reject object/background samples. Missing
or ambiguous evidence remains unobserved. No archived hand state is accepted.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]),
              str(Path(__file__).resolve().parents[1]/'mano')]
from artirecon.input_protocol import load_manifest, validate_manifest, file_sha256
from source_geometry import MaskCandidates, associate_masks, default_config, fit_table, lift, validate_geometry


def choose_track(xy,frame,object_centres,shape):
    """Task-object proximity plus source-image continuity, with a missing state."""
    diagonal=float(np.linalg.norm(shape));T=len(object_centres)
    states=[np.r_[np.flatnonzero(frame==t),-1] for t in range(T)]
    costs=[];back=[]
    for t,rows in enumerate(states):
        emission=[]
        for row in rows:
            if row<0:emission.append(1.);continue
            target=object_centres[t]
            if np.isfinite(target).all():
                emission.append(min(3.,np.linalg.norm(xy[row,[4,8]]-target,axis=1).min()/(.2*diagonal)))
            else:emission.append(.5)
        if t==0:costs.append(np.array(emission));back.append(np.zeros(len(rows),int));continue
        matrix=np.empty((len(states[t-1]),len(rows)))
        for i,old in enumerate(states[t-1]):
            for j,new in enumerate(rows):
                matrix[i,j]=.3 if min(old,new)<0 else min(4.,np.linalg.norm(xy[old,0]-xy[new,0])/(.12*diagonal))
        total=costs[-1][:,None]+matrix
        back.append(np.argmin(total,axis=0));costs.append(np.array(emission)+total.min(0))
    chosen=np.full(T,-1,int);idx=int(np.argmin(costs[-1]))
    for t in range(T-1,-1,-1):chosen[t]=states[t][idx];idx=back[t][idx]
    return chosen


def wrist_depth(depth,mask,xy,radius=10,min_points=12,max_iqr=.05):
    """Use hand-only local depth; reject unreliable/missing observed support."""
    h,w=depth.shape;x,y=np.rint(xy).astype(int)
    region=np.zeros((h,w),bool)
    if not (0<=x<w and 0<=y<h):return None
    region[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)]=True
    values=depth[region&mask&np.isfinite(depth)&(depth>0)]
    if len(values)<min_points:return None
    if np.percentile(values,75)-np.percentile(values,25)>max_iqr:return None
    return float(np.median(values))


def score_hand_side(hypotheses, labels, scores, policy='rgb_evidence'):
    """Soft detector-side prior; its scores are not joint/metric uncertainty."""
    if policy not in {'rgb_evidence','reprojection_only'}:raise ValueError('Unknown hand-side policy')
    labels=np.char.lower(np.asarray(labels,dtype=str));scores=np.asarray(scores,float)
    if len(labels)==0 or scores.shape!=labels.shape or not np.isfinite(scores).all():
        raise ValueError('Hand-side ranking needs valid selected RGB candidates')
    right=np.clip(np.where(labels=='right',scores,1-scores),.05,.95)
    result={}
    for side,h in hypotheses.items():
        probability=right if side=='right' else 1-right
        prior=float(-np.log(probability).mean())
        score=h['selection_score']
        if policy=='rgb_evidence':score+=2.*prior+.1*h['relative3d_rmse_mm']
        result[side]=dict(total_score=float(score),reprojection_px=h['selection_score'],
            relative3d_rmse_mm=h['relative3d_rmse_mm'],detector_side_penalty=prior,
            policy=policy,interpretation='Declared heuristic score, not calibrated posterior probability')
    return result


def fit_pose_hypothesis(models,side,world,xy,k,translation,visible,device,iterations=200):
    import torch
    from mano_recon.model import MANOHand,translation_base
    hand=MANOHand(models,side=side,components=12,samples=256,device=device)
    target=torch.as_tensor(world,dtype=torch.float32,device=device)
    image=torch.as_tensor(xy,dtype=torch.float32,device=device)
    K=torch.as_tensor(k,dtype=torch.float32,device=device)
    seen=torch.as_tensor(visible,dtype=torch.float32,device=device)
    tr=torch.as_tensor(translation,dtype=torch.float32,device=device)
    with torch.no_grad():
        _,neutral=hand(torch.zeros(1,15,device=device))
    rotations=[]
    for points in world:
        U,_,Vh=np.linalg.svd(neutral[0].cpu().numpy().T@points)
        R=Vh.T@np.diag([1,1,np.linalg.det(Vh.T@U.T)])@U.T
        rotations.append(Rotation.from_matrix(R).as_rotvec())
    q=torch.nn.Parameter(torch.as_tensor(np.c_[rotations,np.zeros((len(world),12))],dtype=torch.float32,device=device))
    beta=torch.nn.Parameter(torch.zeros(1,10,device=device))
    optimizer=torch.optim.Adam([q,beta],lr=.025)
    for step in range(iterations):
        optimizer.zero_grad();_,joints=hand(q,beta)
        rel=torch.nn.functional.smooth_l1_loss(joints,target,reduction='none',beta=.01).mean((1,2))/.01
        camera=joints+tr[:,None];projection=camera@K.T;uv=projection[...,:2]/projection[...,2:].clamp_min(.01)
        repro=torch.nn.functional.smooth_l1_loss(uv,image,reduction='none',beta=3.).mean((1,2))/10.
        loss=((rel+repro)*seen).sum()/seen.sum().clamp_min(1)+.005*beta.square().mean()+.0001*q[:,3:].square().mean()
        loss.backward();optimizer.step()
        with torch.no_grad():q[:,3:].clamp_(-3,3);beta.clamp_(-3,3)
    with torch.no_grad():
        _,joints=hand(q,beta);camera=joints+tr[:,None];p=camera@K.T;uv=p[...,:2]/p[...,2:].clamp_min(.01)
        error=torch.linalg.norm(uv-image,dim=-1).mean(-1).cpu().numpy()
        relative_error=torch.sqrt((joints-target).square().mean((1,2))).cpu().numpy()*1000
    return dict(q=q.detach().cpu().numpy(),betas=beta.detach().cpu().numpy(),reprojection_px=error,
                selection_score=float(np.median(error[visible])),relative3d_rmse_mm=float(np.median(relative_error[visible])))


def merge_manifests(paths):
    combined={}
    for path in paths:
        report=load_manifest(path)
        if report['condition']!='rgb_only' or report['task_mode']!='reconstruction':
            raise ValueError('Fresh hand preparation accepts RGB-only reconstruction ancestry')
        data=json.loads(path.read_text())
        for a in data['artifacts']:
            if a['id'] not in report['method_artifact_ids']:continue
            if 'path' in a:a['path']=str((path.parent/a['path']).resolve())
            if a['id'] in combined:
                for key in ['kind','origin','sha256','parent_ids']:
                    if combined[a['id']].get(key)!=a.get(key):raise ValueError('Conflicting input ancestry: '+a['id'])
            else:combined[a['id']]=a
    return dict(schema_version=1,condition='rgb_only',task_mode='reconstruction',artifacts=list(combined.values()),method_inputs=[])


def prepare(*,geometry,landmarks,masks,run,case,models,object_prompt,device='cuda',side_policy='rgb_evidence'):
    import cv2
    geometry,landmarks,masks,run,models=map(lambda p:Path(p).resolve(),[geometry,landmarks,masks,run,models])
    out=run/'inputs'/case
    if out.exists():raise FileExistsError(out)
    load_manifest(geometry/'input-manifest.json',[('depth',geometry/'geometry.npz'),('cameras',geometry/'geometry.npz')])
    load_manifest(landmarks/'input-manifest.json',[('hand_initialization',landmarks/'landmarks.npz')])
    load_manifest(masks/'input-manifest.json',[('masks',masks/'masks.npz')])
    manifest=merge_manifests([geometry/'input-manifest.json',landmarks/'input-manifest.json',masks/'input-manifest.json'])
    with np.load(geometry/'geometry.npz',allow_pickle=False) as z:data={n:z[n].copy() for n in z.files}
    depth,k,timestamps=validate_geometry(data);T,H,W=depth.shape
    with np.load(masks/'masks.npz',allow_pickle=False) as z:maskdata={n:z[n].copy() for n in z.files}
    candidates=MaskCandidates(maskdata,T,(H,W),timestamps)
    with np.load(landmarks/'landmarks.npz',allow_pickle=False) as z:det={n:z[n].copy() for n in z.files}
    if not np.array_equal(det['timestamps'],timestamps):raise ValueError('Detector/geometry frame timing differs')
    config=default_config();config['object_prompt']=object_prompt
    object_rows,object_audit=associate_masks(candidates,T,config)
    centres=np.full((T,2),np.nan)
    for t,row in enumerate(object_rows):
        if row is not None:centres[t]=np.argwhere(candidates.mask(row)).mean(0)[::-1]
    selected=choose_track(det['xy'],det['frame'],centres,(H,W))
    xy=np.zeros((T,21,2),np.float32);world=np.zeros((T,21,3),np.float32)
    visible=selected>=0;translation=np.zeros((T,3),np.float32);depth_valid=np.zeros((T,1,H,W),bool)
    foreground=np.zeros((T,768,3),np.float32);counts=np.zeros(T,np.int64)
    rng=np.random.default_rng(0);table_points=[];frame_audit=[]
    for t,row in enumerate(selected):
        hands=candidates.union(candidates.rows(t,'hand',.3))
        objects=candidates.union(candidates.rows(t,object_prompt,.3))
        table=candidates.union(candidates.rows(t,'tabletop',.3))
        if len(table_points)<5 and table.any():
            table_points.append(lift(depth[t],k,table&~hands&~objects,1000,rng))
        if row<0:frame_audit.append(dict(frame=t,observed=False,reason='No unambiguous continuous RGB hand candidate'));continue
        xy[t]=det['xy'][row];world[t]=det['world_relative'][row]
        hull=np.zeros((H,W),np.uint8)
        cv2.fillConvexPoly(hull,cv2.convexHull(np.rint(xy[t]).astype(np.int32)),1)
        local=cv2.dilate(hull,np.ones((7,7),np.uint8)).astype(bool)&hands&~objects
        z=wrist_depth(depth[t],local,xy[t,0])
        if z is None:
            visible[t]=False;frame_audit.append(dict(frame=t,candidate=int(row),observed=False,reason='Wrist depth has insufficient/ambiguous hand-only support'));continue
        translation[t]=np.r_[xy[t,0],1.]@np.linalg.inv(k).T*z
        reliable=local&np.isfinite(depth[t])&(depth[t]>0)&(abs(depth[t]-z)<.08)
        depth_valid[t,0]=reliable
        points=lift(depth[t],k,reliable,768,rng);counts[t]=len(points);foreground[t,:len(points)]=points
        frame_audit.append(dict(frame=t,candidate=int(row),observed=True,handedness_prediction=str(det['side'][row]),wrist_depth_m=z,points=len(points)))
    if visible.sum()<3:raise ValueError('Too few RGB and hand-only learned-depth observations for MANO initialization')
    # Numerical padding for an optimizer batch, not invented observations.
    good=np.flatnonzero(visible)
    for t in np.flatnonzero(~visible):
        near=good[np.argmin(abs(good-t))];world[t]=world[near];xy[t]=xy[near];translation[t]=translation[near]
    start=time.monotonic()
    hypotheses={side:fit_pose_hypothesis(models,side,world,xy,k,translation,visible,device) for side in ['left','right']}
    out.mkdir(parents=True)
    np.savez_compressed(out/'initial-hypotheses.npz',**{s+'_q':h['q'] for s,h in hypotheses.items()},
                        **{s+'_betas':h['betas'] for s,h in hypotheses.items()},
                        **{s+'_reprojection_px':h['reprojection_px'] for s,h in hypotheses.items()},selected_rows=selected,visible=visible)
    ranking=score_hand_side(hypotheses,det['side'][selected[visible]],det['handedness_score'][selected[visible]],side_policy)
    (out/'side-hypotheses.json').write_text(json.dumps(ranking,indent=2))
    ordered=sorted(hypotheses,key=lambda s:ranking[s]['total_score']);side=ordered[0]
    if ranking[ordered[1]]['total_score']-ranking[side]['total_score']<.5:
        raise ValueError('MANO side remains ambiguous under RGB reprojection; preserve hypotheses rather than force a side')
    fit=hypotheses[side]
    if not table_points:raise ValueError('No learned table mask supports a fresh table estimate')
    normal,offset,table_audit=fit_table(np.concatenate(table_points),config)
    np.savez_compressed(out/'obs.npz',rgb=data['rgb'],depth_m=data['depth_m'],K=data['K'],c2w=data['c2w'],timestamps=timestamps)
    np.savez_compressed(out/'hand-input.npz',initial_q=fit['q'],initial_translation=translation,
        points=foreground,point_views=np.zeros((T,768),np.int64),counts=counts,foreground_counts=counts[:,None])
    kp=np.concatenate([xy,np.repeat(visible[:,None,None],21,axis=1)],axis=-1)[:,None].astype(np.float32)
    np.savez_compressed(out/'human-evidence.npz',keypoints=kp,depth_valid=depth_valid,visible=visible)
    protocol=dict(case=case,mano_side=side,fps=float(1/np.median(np.diff(timestamps))),
        table=dict(normal=normal.tolist(),offset=float(offset)),
        hand_initialization='Fresh RGB MediaPipe relative3D plus learned UniDepth wrist depth; both MANO sides tested using RGB reprojection',
        side_hypothesis_reprojection_px={s:h['selection_score'] for s,h in hypotheses.items()},
        side_ranking=ranking,input_rgb_mirrored=False,
        initial_betas_policy='Hypothesis beta used for ranking only; port refits one shared beta from zero',
        hand_foreground='Learned SAM3 hand mask intersected with RGB landmark hull; object masks excluded; within80mm of supported wrist depth',
        missing_policy='Numerical nearest-visible seed padding only; missing frames remain visible=false and counts=0',
        task_association='Whole-sequence DP, RGB task-object proximity plus wrist continuity; no supplied pose/geometry',
        table_audit=table_audit,frames=frame_audit,
        limitations=['Single observed camera; depth and hidden fingers remain model estimates','Handedness scores are not joint confidence',
                     'No fingertip-object contact constraint added at this initialization stage'])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2,allow_nan=False))
    (run/'enable-human-evidence.json').write_text(json.dumps({'fresh_rgb_landmarks':True,'masked_predicted_depth':True}))
    depth_ids=[a['id'] for a in manifest['artifacts'] if a['kind']=='depth' and a.get('path')==str(geometry/'geometry.npz')]
    camera_ids=[a['id'] for a in manifest['artifacts'] if a['kind']=='cameras' and a.get('path')==str(geometry/'geometry.npz')]
    if not depth_ids or not camera_ids:raise ValueError('Learned geometry artifacts not bound')
    model_ids=[]
    for tested_side in ['left','right']:
        aid='mano_'+tested_side+'_weights';model_ids.append(aid)
        model_path=models/f'MANO_{tested_side.upper()}.npz'
        manifest['artifacts'].append(dict(id=aid,kind='model_prior',origin='prior',scope='global',prior_type='mano_model',
            parent_ids=[],path=str(model_path),sha256=file_sha256(model_path)))
    specs=[('hand_observation_depth','depth','obs.npz',depth_ids),('hand_observation_cameras','cameras','obs.npz',camera_ids),
           ('hand_initialization','hand_initialization','hand-input.npz',['rgb_hand_landmarks','sam3_masks']+model_ids+depth_ids+camera_ids),
           ('hand_rgb_evidence','hand_initialization','human-evidence.npz',['rgb_hand_landmarks','sam3_masks']+depth_ids+camera_ids),
           ('hand_table_support','support','protocol.json',['sam3_masks','rgb_hand_landmarks']+model_ids+depth_ids+camera_ids)]
    for aid,kind,path,parents in specs:
        manifest['artifacts'].append(dict(id=aid,kind=kind,origin='model_prediction',parent_ids=parents,path=path,sha256=file_sha256(out/path)))
    manifest['method_inputs']=[x[0] for x in specs]
    manifest['metric_scale']=dict(status='estimated',artifact_ids=depth_ids,description='Learned UniDepth metric depth; not measured scale')
    manifest['robot_registration']=dict(status='unresolved',artifact_ids=[],description='Human source camera not registered to target robot')
    validation=validate_manifest(manifest,out,verify_files=True)
    (out/'input-manifest.json').write_text(json.dumps(manifest,indent=2))
    result=dict(status='complete',case=case,visible_frames=int(visible.sum()),total_frames=T,mano_side=side,
        code_sha256=file_sha256(__file__),
        initialization_seconds=time.monotonic()-start,side_hypothesis_reprojection_px=protocol['side_hypothesis_reprojection_px'],
        side_ranking=ranking,
        input_protocol=validation,completion_meaning='Prepared a fresh RGB-only MANO seed and port inputs, not final fitting or contact success')
    (out/'preparation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    failures=[]
    for job in json.loads(args.requests.read_text()):
        try:prepare(**job,models=args.models,run=args.run,device=args.device)
        except Exception as error:
            out=args.run/'inputs'/job['case'];out.mkdir(parents=True,exist_ok=True)
            # Never overwrite an existing successful result after refusing reuse.
            name='failed-retry.json' if (out/'preparation.json').exists() else 'preparation.json'
            (out/name).write_text(json.dumps(dict(status='failed',error=repr(error),case=job['case']),indent=2))
            failures.append(dict(case=job['case'],error=repr(error)));print(failures[-1],flush=True)
    if failures:raise RuntimeError('Some fresh hand preparations failed: '+repr(failures))


if __name__=='__main__':main()
