"""Execute the mechanically ported pushed RobotFitter, then export downstream states."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mano_recon.model import translation_base
from mano_recon.run_integrity import output_records, verify_cached_outputs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--input-manifest',type=Path,help='Validate the declared input lineage and hashes; otherwise provenance remains unverified')
    ap.add_argument('--models',type=Path,default=Path(os.environ['MANO_MODEL_DIR']) if 'MANO_MODEL_DIR' in os.environ else None)
    ap.add_argument('--case',choices=['button','block','bottle'],required=True)
    ap.add_argument('--iterations',type=int,default=300);ap.add_argument('--device',default='cuda')
    ap.add_argument('--calibrate',action='store_true',help='Run the pushed table and kinematic depth self-calibration stages')
    ap.add_argument('--refine-from',type=Path,help='Diagnostic: run the source joint-refinement stage from a completed fit')
    ap.add_argument('--finger-hypotheses',action='store_true',help='Compare index/middle whole-chain assignments in RGB evidence')
    ap.add_argument('--limit-frames',type=int);a=ap.parse_args();run=a.run.resolve()
    case=run/'inputs'/a.case
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from artirecon.input_protocol import load_manifest, unverified_report, ProtocolError
    from artirecon.scene_exports import fit_timestamps, mano_fit_scene
    input_protocol=unverified_report()
    if a.input_manifest:
        if a.calibrate:
            raise ProtocolError('Represent self-calibration as a separate declared input stage before using --input-manifest')
        required=[('depth',case/'obs.npz'),('cameras',case/'obs.npz'),
                  ('hand_initialization',case/'hand-input.npz'),('support',case/'protocol.json')]
        if (run/'enable-human-evidence.json').exists():
            required.append(('hand_initialization',case/'human-evidence.npz'))
        if a.refine_from:required.append(('hand_initialization',a.refine_from))
        input_protocol=load_manifest(a.input_manifest,required)
    os.environ['AMANI_DEVICE']=a.device
    sys.path.insert(0,str(run/'runtime'))
    from amani_comparison.fit_robot_mano import RobotFitter
    from amani_comparison.mano_adapter import MANORobot
    from amani_comparison.geom import table_calibrate
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    out=run/('hand-smoke' if a.limit_frames else 'hand-results')/a.case;out.mkdir(parents=True,exist_ok=True)
    if (out/'execution.json').exists() and not a.limit_frames:
        previous_audit=json.loads((out/'execution.json').read_text())
        if previous_audit.get('status')=='complete':
            if a.input_manifest and previous_audit.get('input_protocol',{}).get('manifest_sha256')!=input_protocol['manifest_sha256']:
                raise ProtocolError('Existing hand fit has different/unverified input provenance; use a fresh run directory')
            if a.input_manifest:
                verify_cached_outputs(previous_audit, {'fit': out/'fit.npz', 'frozen_hand': case/'frozen-hand.npz',
                                                      'scene_sequence': out/'scene-sequence.json'})
            if not a.input_manifest:print('Input provenance unverified for existing legacy fit',a.case,flush=True)
            print('Existing complete hand fit',a.case,flush=True);return
    z=np.load(case/'obs.npz');h=np.load(case/'hand-input.npz');meta=json.loads((case/'protocol.json').read_text())
    T=a.limit_frames or len(z['depth_m']);D=z['depth_m'][:T]
    timestamps,timestamp_method=fit_timestamps(z,T,meta.get('fps'),require_supplied=bool(a.input_manifest))
    obs=SimpleNamespace(depth=D,K=z['K'],c2w=z['c2w'],T=T,V=D.shape[1])
    logs=[]
    def log(s):
        print(a.case,s,flush=True);logs.append(s)
        (out/'progress.log').write_text('\n'.join(logs))
    fg=[h['points'][t,:h['counts'][t]].copy() for t in range(T)]
    fv=[h['point_views'][t,:h['counts'][t]] for t in range(T)]
    n=np.array(meta['table']['normal']);d=meta['table']['offset']
    table_scale=np.ones((T,obs.V));offsets=np.zeros(obs.V)
    if a.calibrate:
        original_C=obs.c2w.copy()
        table_scale,(n,d),noise=table_calibrate(obs,log=log)
        offsets,normal=table_calibrate.offsets
        obs.depth=obs.depth*table_scale[:,:,None,None].astype(np.float32)
        obs.c2w=obs.c2w.copy();obs.c2w[:,:3,3]-=offsets[:,None]*normal
        for t in range(T):
            fg[t]=obs.c2w[fv[t],:3,3]+table_scale[t,fv[t],None]*(fg[t]-original_C[fv[t],:3,3])
        meta['table']=dict(normal=np.asarray(n).tolist(),offset=float(d))
    e1=np.cross(n,[0.,0.,1.]) if abs(n[2])<.9 else np.cross(n,[1.,0.,0.]);e1/=np.linalg.norm(e1)
    prep=dict(n=n,d=d,e1=e1,e2=np.cross(n,e1),origin=-d*n,
              fg=fg,fg_view=fv)
    robot=MANORobot(str(a.models or run/'assets/mano'),h['initial_q'][:T],h['initial_translation'][:T],device=a.device)
    fitter=RobotFitter(obs,prep,robot=robot,device=a.device,known_base=np.eye(4),log=log,calib=a.calibrate)
    human_evidence=(run/'enable-human-evidence.json').exists()
    visible=np.ones(T,bool)
    if human_evidence:
        if a.calibrate:raise ValueError('Validate the evidence repair with fixed sensor calibration first')
        from types import MethodType
        from mano_recon.hand_evidence import attach_human_evidence,human_loss
        evidence=np.load(case/'human-evidence.npz');visible=evidence['visible'][:T]
        attach_human_evidence(fitter,evidence['keypoints'][:T],evidence['depth_valid'][:T],visible)
        fitter.finger_pair_hypotheses=a.finger_hypotheses
        fitter.loss=MethodType(human_loss,fitter)
        original_optimise=fitter.optimise
        def optimise_observed(yaw,xy,q,frames,**kwargs):
            if len(set(frames))==1 and not visible[frames[0]]:
                Q=robot.initial[frames[0]:frames[0]+1].expand(len(q),-1).clone()
                with torch.no_grad():
                    L,ch,dl=fitter.loss(Q,fitter.base_T(yaw,xy),frames)
                return yaw.detach(),xy.detach(),Q,L,ch,dl
            return original_optimise(yaw,xy,q,frames,**kwargs)
        fitter.optimise=optimise_observed
    start=time.monotonic()
    if a.refine_from:
        if a.calibrate or not human_evidence:raise ValueError('Refinement diagnostic requires fixed-calibration human evidence')
        previous=np.load(a.refine_from)
        Q=torch.tensor(np.concatenate([previous['q'][:T],previous['translation'][:T]],-1),device=a.device)
        with torch.no_grad():robot.betas.copy_(torch.as_tensor(previous['betas'],device=a.device))
        _,_,Q,L,ch,dl=fitter.optimise(torch.zeros(1,device=a.device),torch.zeros(1,2,device=a.device),Q,list(range(T)),iters=a.iterations,opt_base=False,lr=.01,s_sched=(.008,.008),smooth=0.)
        result=dict(q=Q.cpu().numpy(),loss=L.cpu().numpy(),chamfer=ch.cpu().numpy(),depth=dl.cpu().numpy())
    else:result=fitter.run(iters=a.iterations,n_starts=10,seed=0)
    q=result['q'][:,:-3];tr=result['q'][:,-3:];beta=robot.betas.detach()
    with torch.no_grad():
        qt=torch.tensor(q,device=a.device);bt=translation_base(torch.tensor(tr,device=a.device))
        vertices,joints=robot.hand(qt,beta,bt);P,N=robot.hand.surface(qt,bt,beta)
    vertices,joints=vertices.cpu().numpy(),joints.cpu().numpy()
    np.savez_compressed(out/'fit.npz',q=q,translation=tr,betas=beta.cpu().numpy(),vertices=vertices,joints=joints,
                        loss=result['loss'],chamfer=result['chamfer'],depth=result['depth'])
    np.savez_compressed(out/'hand-sequence.npz',vertices=vertices,joints=joints,faces=robot.hand.faces.cpu().numpy(),
                        timestamps=timestamps,visible=visible)
    # Export shared state, not invented object/contact estimates. Visibility by
    # itself is insufficient: point-count evidence controls observed flags.
    scene=mano_fit_scene(q=q,base=bt.cpu().numpy(),betas=beta.cpu().numpy(),
                        timestamps=timestamps,point_counts=h['counts'][:T],visible=visible,
                        model_uri=(a.models or run/'assets/mano').resolve()/f'MANO_{robot.hand.side.upper()}.npz',
                        input_protocol=input_protocol,support_normal=n,support_offset=d,
                        timestamp_method=timestamp_method,task_description=f'Fit the human hand in the {a.case} sequence')
    scene_path=out/'scene-sequence.json'
    scene_path.write_text(json.dumps(scene,indent=2,allow_nan=False))
    if a.calibrate:
        robscale=result['depth_scale']
        np.savez_compressed((out if a.limit_frames else case)/'fit-calibration.npz',c2w=obs.c2w,table_scale=table_scale,
                            robot_scale=robscale,total_scale=table_scale*robscale,offsets=offsets)
    tcp=(joints[:,4]+joints[:,8])/2
    frame=np.tile(np.eye(4),(T,1,1));frame[:,:3,:3]=Rotation.from_rotvec(q[:,:3]).as_matrix();frame[:,:3,3]=tcp
    # Unobserved hands must not remove object depth or trigger a fictitious grasp.
    points=P.cpu().numpy();normals=N.cpu().numpy();width=np.linalg.norm(joints[:,4]-joints[:,8],axis=-1)
    points[~visible]=1e3;tcp[~visible]=1e3;frame[~visible,:3,3]=1e3;width[~visible]=.15
    exported_hand=(out if a.limit_frames else case)/'frozen-hand.npz'
    np.savez_compressed(exported_hand,q=q,base=bt.cpu().numpy(),points=points,normals=normals,
                        hand_frames=frame,pinch_points=tcp,pinch_width=width,visible=visible)
    meta['hand_fit_sha256']=hashlib.sha256((out/'fit.npz').read_bytes()).hexdigest()
    # A verified input bundle stays byte-identical for downstream checks/cache reuse.
    if not a.limit_frames:((out if a.input_manifest else case)/'protocol.json').write_text(json.dumps(meta,indent=2))
    audit=dict(status='complete',frames=T,source_frames_fitted=True,iterations=a.iterations,forward_starts=10,
               shared_scene=str(scene_path),shared_scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
               timestamps_provenance=timestamp_method,
               initial_starts=24,backward_starts=6,seconds=time.monotonic()-start,device=a.device,
               heldout_used=False,GT_hand_masks_used=None,GT_hand_poses_used=None,previous_RGBD_fit_used=False,
               shared_shape=beta.cpu().numpy().tolist(),depth_self_calibration=a.calibrate,
               human_evidence_constraints=human_evidence,proprioception_camera_ICP=False,provenance='runtime/amani_comparison/robotfitter-mano-provenance.json')
    audit.update(finger_pair_hypotheses=a.finger_hypotheses,refinement_source=str(a.refine_from.resolve()) if a.refine_from else None)
    if a.refine_from:audit.update(forward_starts=0,initial_starts=0,backward_starts=0,previous_RGBD_fit_used=True,refinement_source_sha256=hashlib.sha256(a.refine_from.read_bytes()).hexdigest())
    audit['static_camera_refinement']=(case/'static-calibration.json').exists()
    audit['input_protocol']=input_protocol
    audit['output_artifacts']=output_records({'fit': out/'fit.npz', 'frozen_hand': exported_hand,
                                            'scene_sequence': out/'scene-sequence.json'})
    if input_protocol.get('condition') in {'rgb_only','calibrated_rgb'}:
        audit.update(GT_hand_masks_used=False,GT_hand_poses_used=False)
    (out/'execution.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
