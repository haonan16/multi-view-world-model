"""Compile recorded-observation reconstructions into robot TCP/gripper targets.
No simulator calls or evaluation labels. Metric trajectories are rigidly mapped
into a reachable table frame; support correction and grasp priors are explicit.
"""
from pathlib import Path
import json,hashlib,argparse
import numpy as np,trimesh
from scipy.spatial.transform import Rotation,Slerp
from scipy.linalg import polar
ROOT=Path(__file__).resolve().parents[1]
RUN=None
OUT=None

def grasp_rotation(approach,closing):
 a=np.asarray(approach,float);a/=np.linalg.norm(a);c=np.asarray(closing,float);c-=a*(a@c);c/=np.linalg.norm(c)
 return np.stack([np.cross(c,a),c,a],1)

def main():
 global RUN,OUT
 parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--cases',nargs='+',default=['block','bottle','button']);parser.add_argument('--scene-arm',default='part-refined');parser.add_argument('--hand-arm',default='hand-selected');args=parser.parse_args();RUN=args.run;OUT=args.out
 for case in args.cases:
  out=OUT/'inputs'/case;out.mkdir(parents=True,exist_ok=True)
  if (out/'frozen-plan.json').exists():raise RuntimeError('This plan is frozen; refusing to overwrite it after evaluation')
  src=RUN/'results'/args.scene_arm/case/'scene.json';scene=json.loads(src.read_text());ob=scene['objects'][0];A=np.array(ob['poses']);v=np.array(ob['vertices']);f=np.array(ob['faces']);times=np.array(scene['timestamps']);n=np.array(scene['table']['normal']);n/=np.linalg.norm(n);d=scene['table']['offset'];R0,S0=polar(A[0,:3,:3]);v=v@S0.T
  rotations=np.stack([polar(P[:3,:3])[0] for P in A]);stretch_error=max(np.abs(polar(P[:3,:3])[1]-S0).max() for P in A);assert stretch_error<1e-4
  delta=A[-1,:3,3]-A[0,:3,3];horizontal=delta-n*(delta@n)
  if np.linalg.norm(horizontal)>.05:axis=horizontal/np.linalg.norm(horizontal)
  else:axis=np.cross([0,1.,0],n);axis/=np.linalg.norm(axis)
  mapping=np.stack([axis,np.cross(n,axis),n]);origin=A[0,:3,3]-n*(A[0,:3,3]@n+d)
  motion=np.tile(np.eye(4),(len(A),1,1));motion[:,:3,:3]=mapping@rotations;motion[:,:3,3]=(A[:,:3,3]-origin)@mapping.T
  if case=='block':motion[:,0,3]-=np.linalg.norm(horizontal)/2
  # Materialize a stable bottom at the observed table. Geometry/pose fitting
  # sometimes leaves a floating start or a below-table final target.
  initial_bottom=(v@motion[0,:3,:3].T+motion[0,:3,3])[:,2].min();motion[:,2,3]-=initial_bottom
  corrections=[]
  for i,P in enumerate(motion):
   bottom=(v@P[:3,:3].T+P[:3,3])[:,2].min();shift=max(0,.002-bottom);P[2,3]+=shift;corrections.append(shift)
  hand_path=RUN/args.hand_arm/case/'hand-sequence.npz'
  hand=np.load(hand_path);j=hand['joints'];jm=(j-origin)@mapping.T
  if case=='block':jm[:,:,0]-=np.linalg.norm(horizontal)/2
  jm[:,:,2]-=initial_bottom
  frames=np.array(scene['frame_ids']);held=np.array(ob['held']);segments=[];targets=[];grips=[];phases=[];source_time=[];freq=20
  def add(target,grip,phase,duration,source=-1):
   targets.append(target.copy());grips.append(float(grip));phases.append(phase);source_time.append(float(source));segments.append(float(duration))
  def ramp(start,end,grip,phase,duration,source=-1):
   count=max(1,round(duration*freq));r=Slerp([0,1],Rotation.from_matrix([start[:3,:3],end[:3,:3]]))
   for k in range(1,count+1):
    u=k/count;P=np.eye(4);P[:3,:3]=r(u).as_matrix();P[:3,3]=(1-u)*start[:3,3]+u*end[:3,3];add(P,grip,phase,1/freq,source)
  assumptions=['Known Panda robot; reconstructed metric object geometry and table plane.', 'Physics friction/density are assumed, not identified from video.', 'Controller uses proprioception; object state is logged only for evaluation.']
  if case!='button':
   hi=np.flatnonzero(held);start=int(hi[0]) if len(hi) else 0;end=int(hi[-1]) if len(hi) else len(A)-1
   lo,up=v.min(0),v.max(0);center=(lo+up)/2;ext=up-lo
   if case=='block':
    closing_axis=int(np.argmin(ext[:2]));closing=motion[start,:3,closing_axis];closing[2]=0;approach=np.array([0.,0.,-1.]);contact_local=center
   else:
    closing_axis=int(np.argmin(ext[:2]));closing=motion[start,:3,closing_axis];closing[2]=0;approach=motion[start,:3,1-closing_axis];approach[2]=0
    if approach[0]<0:approach=-approach
    contact_local=center.copy();contact_local[2]=lo[2]+.40*ext[2]
   tcp=np.eye(4);tcp[:3,:3]=grasp_rotation(approach,closing);tcp[:3,3]=motion[start,:3,:3]@contact_local+motion[start,:3,3]
   relative=np.linalg.inv(motion[start])@tcp;pre=tcp.copy();pre[:3,3]-=tcp[:3,2]*.12;pre[2,3]+=.025
   # First pose is approached from the robot's measured initial TCP by the runner.
   add(pre,1,'pregrasp',1.5);ramp(pre,tcp,1,'approach',1.2);ramp(tcp,tcp,-1,'close',.8)
   prev=tcp
   for i in range(start+1,end+1):
    dest=motion[i]@relative;duration=max(times[i]-times[i-1],np.linalg.norm(dest[:3,3]-prev[:3,3])/.12)
    ramp(prev,dest,-1,'transport',duration,times[i]);prev=dest
   if case=='block':
    # Release at the final reconstructed location with a small support clearance.
    goal=motion[-1].copy();world=v@goal[:3,:3].T+goal[:3,3];goal[2,3]+=.003-world[:,2].min();dest=goal@relative
    ramp(prev,dest,-1,'place',.8,times[-1]);ramp(dest,dest,1,'release',.7,times[-1]);retreat=dest.copy();retreat[2,3]+=.15;ramp(dest,retreat,1,'retreat',1.0)
   else:
    ramp(prev,prev,-1,'hold',1.0,times[-1]);goal=motion[-1]
   events=dict(grasp_source_time=float(times[start]),last_held_source_time=float(times[end]),release_observed=bool(case=='block'),source='Saved Amani object association/held-state estimate; not GT contacts',interior_unheld_keyframes=(np.flatnonzero(~held[start:end+1])+start).tolist(),execution_assumption='One continuous robot grasp between first and last held estimates')
   assumptions.append('Robot-compatible grasp chosen on reconstructed geometry; human finger joint angles are not directly copied.')
  else:
   # Button interaction is known from the clip. Recover the lateral press site
   # from the closest observed index fingertip; articulation travel is unobserved.
   capstart=ob['parts'][-1]['face_start'];cv=np.unique(f[capstart:]);capv=v[cv];center=capv.mean(0);center[2]=capv[:,2].max();world=motion[0,:3,:3]@center+motion[0,:3,3]
   valid=hand['visible'];dist=np.linalg.norm(jm[:,8]-world,axis=1);dist[~valid]=np.inf;t=int(dist.argmin());site=world.copy();site[:2]=.5*(site[:2]+jm[t,8,:2]);tcp=np.eye(4);tcp[:3,:3]=grasp_rotation([0,0,-1],[0,1,0]);tcp[:3,3]=site
   pre=tcp.copy();pre[2,3]+=.14;touch=tcp.copy();touch[2,3]+=.012;press=tcp.copy();press[2,3]-=.012
   add(pre,-1,'prepress',1.5);ramp(pre,touch,-1,'approach',1.2);ramp(touch,press,-1,'press',1.0);ramp(press,press,-1,'hold',.6);ramp(press,pre,-1,'retract',1.2);goal=motion[0]
   events=dict(press_source_frame=t,press_source_time=float(hand['timestamps'][t]),source='Closest visible reconstructed index fingertip to fitted cap',cap_stroke_measured=False)
   assumptions.extend(['Button cap has an assumed 6 mm passive prismatic travel and spring.', 'Closed parallel gripper used as a pusher; not a human fingertip.'])
   bodyfaces=f[:capstart];capfaces=f[capstart:];trimesh.Trimesh(v,bodyfaces,process=True).export(out/'housing.obj');trimesh.Trimesh(v,capfaces,process=True).export(out/'cap.obj')
  mesh=trimesh.Trimesh(v,f,process=False);mesh.export(out/'object.obj')
  np.savez_compressed(out/'plan.npz',tcp_world=np.array(targets),gripper=np.array(grips),phase=np.array(phases),duration_s=np.array(segments),source_time_s=np.array(source_time),object_poses=motion,object_times=times,goal_pose=goal,hand_joints=jm,hand_times=hand['timestamps'])
  manifest=dict(case=case,source_scene=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),input_branch=(RUN/'branch.json').read_text() if (RUN/'branch.json').exists() else 'Recorded real RGB-D reconstruction; no world-model predictions',GT_pose_labels_used=False,planner_has_simulator_access=False,control_frequency=freq,events=events,assumptions=assumptions,source_to_table_rotation=mapping.tolist(),source_origin=origin.tolist(),support_initial_translation_m=float(-initial_bottom),per_keyframe_support_correction_m=corrections,collision='box' if case=='block' else 'convex hull of fitted bottle mesh' if case=='bottle' else 'separate housing and cap convex hulls',stretch_consistency_error=float(stretch_error),goal_pose=goal.tolist(),dimensions_m=np.ptp(v,axis=0).tolist())
  manifest.update(source_hand=str(hand_path),source_hand_sha256=hashlib.sha256(hand_path.read_bytes()).hexdigest(),planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scene_arm=args.scene_arm,hand_arm=args.hand_arm,human_fingers_drive_robot_grasp=bool(case=='button'))
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(case,len(targets),events,'support shift',-initial_bottom,flush=True)
if __name__=='__main__':main()
