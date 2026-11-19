"""Execute compiled TCP/gripper targets through Panda IK and native SAPIEN physics.
No object pose resets, welds, or simulated-state replanning after initialization.
"""
import argparse,json,time,traceback,importlib.metadata
from pathlib import Path
import numpy as np,torch,sapien,gymnasium as gym,trimesh,imageio.v2 as imageio
from scipy.spatial.transform import Rotation,Slerp
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

def arr(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
def pose(P):
 q=Rotation.from_matrix(P[:3,:3]).as_quat();return sapien.Pose(P[:3,3],q[[3,0,1,2]])
def serial(x):
 if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [serial(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 return x

def save(p,x):p.write_text(json.dumps(serial(x),indent=2,allow_nan=False))
@register_env('RecordedMANOActions-v1',max_episode_steps=2000,override=True)
class Env(BaseEnv):
 SUPPORTED_ROBOTS=['panda']
 def __init__(self,*args,case_dir=None,friction=1.,**kwargs):
  self.case_dir=Path(case_dir);self.manifest=json.loads((self.case_dir/'manifest.json').read_text());self.plan=np.load(self.case_dir/'plan.npz');self.case=self.manifest['case'];self.friction=friction;super().__init__(*args,robot_uids='panda',**kwargs)
 @property
 def _default_sensor_configs(self):return []
 @property
 def _default_human_render_camera_configs(self):
  return CameraConfig('render_camera',sapien_utils.look_at([.65,-.70,.60],[-.08,0,.15]),640,480,1.0,.01,10)
 def _load_agent(self,options):super()._load_agent(options,sapien.Pose(p=[-.615,0,0]))
 def _load_scene(self,options):
  self.table_scene=TableSceneBuilder(self,robot_init_qpos_noise=0);self.table_scene.build();mat=sapien.physx.PhysxMaterial(self.friction,self.friction,0.);P=self.plan['object_poses'][0]
  color=sapien.render.RenderMaterial(base_color=[.84,.56,.2,1],roughness=.6)
  if self.case=='button':
   ab=self.scene.create_articulation_builder();ab.initial_pose=pose(P);root=ab.create_link_builder();root.set_name('button_housing');cap=ab.create_link_builder(root);cap.set_name('button_cap');cap.set_joint_name('button_slide');axis=sapien.Pose(q=[2**-.5,0,2**-.5,0]);cap.set_joint_properties('prismatic',limits=[[0,.006]],pose_in_parent=axis,pose_in_child=axis,damping=2,friction=.05)
   for b,name,col in [(root,'housing',color),(cap,'cap',sapien.render.RenderMaterial(base_color=[.2,.65,.3,1]))]:
    b.add_visual_from_file(str(self.case_dir/(name+'.obj')),material=col);b.add_convex_collision_from_file(str(self.case_dir/(name+'.obj')),material=mat,density=400)
   self.button=ab.build(name='button',fix_root_link=True)
  else:
   b=self.scene.create_actor_builder();b.initial_pose=pose(P);mesh=trimesh.load(self.case_dir/'object.obj',force='mesh')
   b.add_visual_from_file(str(self.case_dir/'object.obj'),material=color)
   if self.case=='block':b.add_box_collision(pose=sapien.Pose(mesh.bounds.mean(0)),half_size=mesh.extents/2,material=mat,density=500)
   else:
    collision_parts=self.manifest.get('collision_parts',[])
    if collision_parts:
     for filename in collision_parts:
      if Path(filename).name!=filename or Path(filename).suffix!='.obj':raise ValueError('Invalid collision part path')
      b.add_convex_collision_from_file(str(self.case_dir/filename),material=mat,density=300)
    else:b.add_convex_collision_from_file(str(self.case_dir/'object.obj'),material=mat,density=300)
   self.object=b.build(name='manipulated_object')
 def _initialize_episode(self,env_idx,options):
  self.table_scene.initialize(env_idx)
  # The table builder's low default home intersects a tall bottle at reset.
  # Start the known robot above the workspace before executing any action.
  self.agent.reset(np.array([0.,0.,0.,-np.pi/2,0.,np.pi/2,np.pi/4,.04,.04]))
  if self.case=='button':
   self.button.set_qpos(torch.zeros((1,1),device=self.device));self.button.set_qvel(torch.zeros((1,1),device=self.device))
   joint=self.button.get_active_joints()[0];joint.set_drive_properties(stiffness=1800,damping=8);joint.set_drive_target(0.)
  else:self.object.set_pose(Pose.create(pose(self.plan['object_poses'][0])))
 def evaluate(self):return dict(success=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device))
 def _get_obs_extra(self,info):return {}

def run(case_dir,out,friction=1.,require_frozen=False):
 frozen_metadata=None
 if require_frozen:
  from freeze_plan import claim_execution
  frozen_metadata=claim_execution(case_dir,Path(__file__),friction,out)
 out.mkdir(parents=True,exist_ok=False);plan=np.load(case_dir/'plan.npz');manifest=json.loads((case_dir/'manifest.json').read_text());case=manifest['case'];env=gym.make('RecordedMANOActions-v1',case_dir=str(case_dir),friction=friction,num_envs=1,obs_mode='none',control_mode='pd_ee_pose',reward_mode='none',render_mode='rgb_array',sim_backend='cpu',human_render_camera_configs={'shader_pack':'minimal'},sim_config={'control_freq':20,'sim_freq':120});e=env.unwrapped;records=[];actions=[];writer=None;t0=time.time()
 try:
  env.reset(seed=0);freq=e.control_freq;writer=imageio.get_writer(out/'physics.mp4',fps=freq,codec='libx264',quality=8,macro_block_size=16)
  root=arr(e.agent.robot.pose.to_transformation_matrix())[0];targets=plan['tcp_world'];durations=plan['duration_s'];grips=plan['gripper'];phases=plan['phase'];source_time=plan['source_time_s'];initial=arr(e.agent.tcp_pose.to_transformation_matrix())[0]
  if case!='button':initial_object=arr(e.object.pose.to_transformation_matrix())[0]
  else:initial_object=plan['object_poses'][0]
  def step(P,g,phase,source):
   local=np.linalg.inv(root)@P;action=np.r_[local[:3,3],Rotation.from_matrix(local[:3,:3]).as_euler('XYZ'),g].astype(np.float32);env.step(action);actions.append(action)
   tcp=arr(e.agent.tcp_pose.to_transformation_matrix())[0];qpos=arr(e.agent.robot.qpos)[0];contacts=[]
   for ct in e.scene.sub_scenes[0].get_contacts():
    names=[b.entity.name for b in ct.bodies]
    if any(('manipulated_object' in n or 'button_cap' in n) for n in names):contacts.append(dict(bodies=names,separation=[float(p.separation) for p in ct.points],impulse=[float(np.linalg.norm(p.impulse)) for p in ct.points]))
   row=dict(step=len(records),time_s=len(records)/freq,phase=str(phase),source_time_s=float(source),tcp=tcp,qpos=qpos,target_tcp=P,tcp_error_mm=float(np.linalg.norm(tcp[:3,3]-P[:3,3])*1000),contacts=contacts)
   if case!='button':row.update(object_pose=arr(e.object.pose.to_transformation_matrix())[0],grasped=bool(arr(e.agent.is_grasping(e.object))[0]))
   else:row.update(cap_qpos_m=float(arr(e.button.qpos)[0,0]))
   records.append(row);frame=arr(e.render_rgb_array())[0].astype(np.uint8);writer.append_data(frame)
   if len(records)%40==0:imageio.imwrite(out/f'frame-{len(records):03d}.png',frame)
  # Smooth startup uses only robot proprioception, not object state.
  r=Slerp([0,1],Rotation.from_matrix([initial[:3,:3],targets[0,:3,:3]]));count=round(float(durations[0])*freq)
  for i in range(1,count+1):
   u=i/count;P=np.eye(4);P[:3,:3]=r(u).as_matrix();P[:3,3]=(1-u)*initial[:3,3]+u*targets[0,:3,3];step(P,grips[0],phases[0],-1)
  for i in range(1,len(targets)):
   for _ in range(max(1,round(float(durations[i])*freq))):step(targets[i],grips[i],phases[i],source_time[i])
  for _ in range(30):step(targets[-1],grips[-1],'final_hold',source_time[-1])
  writer.close();writer=None;np.save(out/'actions.npy',np.array(actions));save(out/'records.json',records)
  summary=dict(status='complete',case=case,wall_seconds=time.time()-t0,control_freq=freq,steps=len(records),mean_tcp_error_mm=float(np.mean([r['tcp_error_mm'] for r in records])),max_tcp_error_mm=max(r['tcp_error_mm'] for r in records),friction=friction,object_pose_updates_after_initialization=0,artificial_grasp_constraints=False,sim_state_used_by_planner=False,robot_initialization='raised Panda home; avoids tall-object intersection at reset')
  if frozen_metadata is not None:summary.update(frozen_plan=frozen_metadata,post_rollout_refinement=False)
  if case!='button':
   obj=np.array([r['object_pose'] for r in records]);goal=plan['goal_pose'];lift=obj[:,2,3]-initial_object[2,3];err=np.linalg.norm(obj[-1,:3,3]-goal[:3,3]);grasp=np.array([r['grasped'] for r in records]);stable=np.max(np.linalg.norm(np.diff(obj[-20:,:3,3],axis=0),axis=1))*freq<.01
   success=(grasp.any() and lift.max()>.025 and err<.025 and ((not grasp[-1] and stable) if case=='block' else grasp[-1]))
   summary.update(ever_grasped=bool(grasp.any()),final_grasped=bool(grasp[-1]),max_lift_mm=float(lift.max()*1000),goal_position_error_mm=float(err*1000),final_object_stable=bool(stable),success=bool(success),final_object_pose=obj[-1],goal_pose=goal)
  else:
   stroke=np.array([r['cap_qpos_m'] for r in records]);summary.update(max_cap_travel_mm=float(stroke.max()*1000),final_cap_travel_mm=float(stroke[-1]*1000),success=bool(stroke.max()>.004 and stroke[-1]<.001),physical_parameters_assumed=True)
  save(out/'summary.json',summary);print(json.dumps(serial(summary)),flush=True)
 except Exception:
  save(out/'failure.json',dict(error=traceback.format_exc()));raise
 finally:
  if writer is not None:writer.close()
  if records and not (out/'records.json').exists():save(out/'partial-records.json',records)
  env.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',nargs='+',default=['block','bottle','button']);p.add_argument('--friction',type=float,default=1.);p.add_argument('--require-frozen',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 save(a.output/'versions.json',{k:importlib.metadata.version(k) for k in ['mani_skill','sapien','torch','numpy']})
 for c in a.cases:
  try:run(a.inputs/c,a.output/c,a.friction,a.require_frozen)
  except Exception as exc:print(c,traceback.format_exc(),flush=True)
if __name__=='__main__':main()
