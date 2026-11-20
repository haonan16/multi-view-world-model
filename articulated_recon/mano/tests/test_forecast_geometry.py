"""Generated fixtures exercise fitting, not recorded-scene accuracy claims."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
import forecast_geometry as fg


def fixture(views=2,scales=(1.,1.,1.),rotation=True):
    t=len(scales);h,w=60,80
    half=np.array([.07,.035,.03])
    poses=np.tile(np.eye(4),(t,1,1));poses[:,2,3]=.8
    poses[:,0,3]=np.linspace(0,.06,t);poses[:,1,3]=np.linspace(0,-.05,t)
    if rotation:
        poses[:,:3,:3]=Rotation.from_rotvec(np.column_stack([np.zeros(t),np.zeros(t),np.linspace(0,.45,t)])).as_matrix()
    k=np.tile([[110.,0,40],[0,110.,30],[0,0,1]],(t,views,1,1))
    c=np.tile(np.eye(4),(t,views,1,1))
    if views>1:c[:,1,0,3]=.15
    mesh,faces=fg.geometry_arrays(dict(type='box',half_extents_m=half))
    depth=np.full((t,views,h,w),2.)
    masks=[];rows=[];vv=[]
    for i in range(t):
        for v in range(views):
            world=mesh@poses[i,:3,:3].T+poses[i,:3,3]
            rendered=fg.raster_depth(world,faces,k[i,v],c[i,v],(h,w))
            mask=rendered>0;depth[i,v,mask]=rendered[mask]
            masks.append(np.packbits(mask));rows.append(i);vv.append(v)
    depth/=np.asarray(scales)[:,None,None,None]
    c[:,:,:3,3]/=np.asarray(scales)[:,None,None]
    values=dict(rgb=np.zeros((t,views,h,w,3),np.uint8),depth=depth,K=k,c2w=c,timestamps=np.arange(t)*.2)
    mask_values=dict(mask=np.stack(masks),t=np.asarray(rows),v=np.asarray(vv),p=np.zeros(len(rows),int),
        score=np.ones(len(rows))*.95,shape=np.array([h,w]),prompts=np.array(['wooden block on patterned mat','hand']),
        timestamps=values['timestamps'])
    provenance=dict(method='Synthetic first-image model fixture')
    initial=fg.scene_from_arrays(timestamps=[0.],coordinate_frame=dict(name='source_camera',handedness='right',
        up_axis='-y',pose_convention='T_world_local'),metric_scale=dict(status='estimated',provenance=provenance),
        actors=[],objects=[fg.object_from_arrays(object_id='block',poses=poses[:1],observed=[True],
            primitive=dict(type='box',half_extents_m=half),provenance=provenance)],
        support=dict(id='table',normal_world=[0,-1,0],offset_m=.03,confidence=None,
                     provenance=dict(method='initial predicted plane',confidence_status='unknown')),
        contacts=[],uncertainty=dict(status='unestimated'),task=dict(description='Lift block',target=None))
    config=fg.default_config();config.update(points_per_view=96,boundary_samples=32,surface_grid=3,max_nfev=60)
    return values,mask_values,initial,poses,config


def test_actual_joint_fit_estimates_shared_shape_and_free_lift_rotation():
    values,masks,initial,truth,config=fixture()
    frames=fg.prepare_evidence(values,masks,config)
    fit=fg.fit_shared_block(initial,frames,values['timestamps'],'helper_requested_cameras',9.,config)
    assert fit['report']['final_objective']<fit['report']['initial_objective']
    assert fit['half_extents_m'].shape==(3,) and fit['poses'].shape==(3,4,4)
    np.testing.assert_allclose(fit['scales'],1.) # supplied arbitrary anchor must not rescale requested-camera depth
    np.testing.assert_allclose(fit['poses'][0],truth[0])
    assert fit['poses'][-1,1,3]<-.025 # no forced rest on the initial table
    assert np.linalg.norm(fit['poses'][-1,:3,:3]-fit['poses'][0,:3,:3])>.05
    assert fit['prediction_supported'].tolist()==[False,True,True]
    scene=fg.export_scene(initial,fit,values['timestamps'],['predicted_forecast_depth'],'block','Lift block')
    fg.validate_scene(scene,require_metric_scale=True)
    assert scene['objects'][0]['observed']==[True,False,False]
    assert scene['objects'][0]['prediction_supported']==[False,True,True]
    assert scene['objects'][0]['prediction_provenance']['generated'] is True
    assert scene['contacts']==[] and scene['support']==initial['support']


def test_unposed_frame_scales_are_estimated_separately_from_one_shape():
    values,masks,initial,truth,config=fixture(views=1,scales=(2.,3.,2.5),rotation=False)
    frames=fg.prepare_evidence(values,masks,config)
    fit=fg.fit_shared_block(initial,frames,values['timestamps'],'source_unposed',2.,config)
    assert fit['scales'][0]==2.
    assert fit['scales'][1]>2.35
    assert fit['report']['final_objective']<fit['report']['initial_objective']
    assert np.linalg.norm(fit['poses'][1,:3,3]-truth[1,:3,3])<.12
    assert 'per-time' in fit['report']['scale_mode']


def test_ambiguous_and_fully_occluded_frames_remain_unsupported():
    values,masks,initial,truth,config=fixture(views=1,rotation=False)
    extra=np.zeros((60,80),bool);extra[5:12,5:12]=True
    masks['mask']=np.concatenate([masks['mask'],[np.packbits(extra)]])
    for name,value in [('t',1),('v',0),('p',0),('score',.95)]:
        masks[name]=np.r_[masks[name],value]
    frames=fg.prepare_evidence(values,masks,config)
    assert frames[1][0]['status']=='ambiguous_candidates' and not frames[1][0]['known']
    fit=fg.fit_shared_block(initial,frames,values['timestamps'],'source_unposed',1.,config)
    assert not fit['prediction_supported'][1]
    np.testing.assert_array_equal(fit['poses'][1],fit['poses'][0])
    # Cover the sole object observation with a hand mask: no depth is free space.
    hand=masks['mask'][0].copy()
    clean=deepcopy(masks);clean['mask']=np.concatenate([clean['mask'],[hand]])
    for name,value in [('t',0),('v',0),('p',1),('score',.95)]:clean[name]=np.r_[clean[name],value]
    hidden=fg.prepare_evidence(values,clean,config)[0][0]
    assert len(hidden['points'])==0 and len(hidden['boundary_uv'])==0


def test_initial_scale_ratio_uses_only_first_image_predictions():
    anchor=np.ones((10,12))*2
    depth=np.ones((5,6))*.5
    scale,report=fg.initial_scale_anchor(anchor,[0,0,12,10],depth,np.ones((5,6),bool))
    assert scale==4. and report['pixels']==30
    assert report['measured_metric_depth_used'] is False
    assert report['interpretation'].startswith('Alignment of two predictions')


def test_duplicate_mask_candidates_are_deduplicated_without_inventing_identity():
    values,masks,_,_,config=fixture(views=1)
    masks['mask']=np.concatenate([masks['mask'],masks['mask'][:1]])
    for key in ['t','v','p','score']:masks[key]=np.r_[masks[key],masks[key][0]]
    assoc=fg.associate_masks(masks,(60,80),values['timestamps'],1,config)
    assert assoc[0][0]['known'] and len(assoc[0][0]['candidate_rows'])==1


def test_exported_overlays_have_nullable_contact_evidence_not_events(tmp_path):
    values,masks,initial,_,config=fixture(views=1)
    frames=fg.prepare_evidence(values,masks,config)
    fit=fg.fit_shared_block(initial,frames,values['timestamps'],'source_unposed',1.,config)
    report=fg.render_and_measure(tmp_path,values,frames,fit)
    assert len(list((tmp_path/'overlays').glob('*.png')))==3
    assert report[1]['views'][0]['hand_to_object_mask_gap_px'] is None
    assert report[1]['views'][0]['hand_to_box_min_signed_distance_m'] is None


def test_optional_soft_plane_prior_reduces_penetration_without_pose_snap():
    values,masks,initial,truth,config=fixture(views=2,rotation=False)
    frames=fg.prepare_evidence(values,masks,config)
    baseline=fg.fit_shared_block(initial,frames,values['timestamps'],'helper_requested_cameras',1.,config)
    config['nonpenetration_weight']=10.
    refined=fg.fit_shared_block(initial,frames,values['timestamps'],'helper_requested_cameras',1.,config)
    before=min(baseline['report']['estimated_plane_clearance_m'])
    after=min(refined['report']['estimated_plane_clearance_m'])
    assert after>before+1e-4
    np.testing.assert_array_equal(refined['poses'][0],truth[0])
    assert refined['report']['plane_snap_used'] is False
    assert refined['report']['nonpenetration_hard_guarantee'] is False
    assert refined['report']['nonpenetration_soft_prior_weight']==10.


def full_input_fixture(tmp_path,branch):
    path=Path(__file__).with_name('test_world_model_forecast.py')
    spec=importlib.util.spec_from_file_location('forecast_protocol_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args,_=module.fixture_inputs(tmp_path);args['frames']=5
    module.sf.prepare_forecast(**args)
    condition=args['output'];helper=tmp_path/'generated';helper.mkdir()
    with np.load(condition/'condition.npz') as z:arrays={k:z[k].copy() for k in z.files}
    count=len(arrays['timestamps'])
    _,known=module.wm.source_tensor(arrays['rgb'],forecast=True,total_frames=count)
    rgb=module.wm.restore_observed_source(np.full((count,4,160,320,3),77,np.uint8),arrays['rgb'],known)
    np.savez_compressed(helper/'observations.npz',rgb=rgb,observed_mask=known,
                        **{k:v for k,v in arrays.items() if k!='rgb'})
    manifest=json.loads((condition/'input-manifest.json').read_text())
    for a in manifest['artifacts']:
        if 'path' in a:a['path']=str((condition/a['path']).resolve())
    manifest['artifacts'].append(dict(id='world_model_weights',kind='model_prior',origin='prior',
        prior_type='learned_model_weights',scope='global',parent_ids=[]))
    manifest['artifacts'].extend(module.sf.generated_records(helper,arrays['timestamps']))
    manifest['method_inputs'].extend(['generated_source_forecast','generated_helper_rgb'])
    fg.write_json(helper/'input-manifest.json',manifest)
    for a in manifest['artifacts']:
        if 'path' in a:a['path']=str((helper/a['path']).resolve())
    initial=fg.load_scene(args['scene_path']);obj=initial['objects'][0]
    vertices,faces=fg.geometry_arrays(obj['geometry'])
    views=1 if branch=='source_unposed' else 4
    k=np.repeat(arrays['K'][None,:views],count,axis=0)
    c=np.repeat(arrays['c2w'][None,:views],count,axis=0)
    depth=np.ones((count,views,160,320));packed=[];tt=[];vv=[]
    for t in range(count):
        pose=np.array(obj['T_world_object'][0]);pose[0,3]+=.015*t;pose[1,3]-=.01*t
        world=vertices@pose[:3,:3].T+pose[:3,3]
        for v in range(views):
            rendered=fg.raster_depth(world,faces,k[t,v],c[t,v],(160,320))
            mask=rendered>0;depth[t,v,mask]=rendered[mask]
            packed.append(np.packbits(mask));tt.append(t);vv.append(v)
    if branch!='helper_requested_cameras':depth/=2;c[:,:,:3,3]/=2
    pred=tmp_path/'predictions';pred.mkdir()
    np.savez_compressed(pred/'predictions.npz',rgb=rgb[:,:views],depth=depth,K=k,c2w=c,
        timestamps=arrays['timestamps'],source_frame_indices=arrays['source_frame_indices'],observed_mask=known[:,:views])
    parents=['generated_source_forecast']+([] if views==1 else ['generated_helper_rgb'])
    pm=deepcopy(manifest)
    for name,kind in [('da3_depth','depth'),('da3_cameras','cameras')]:
        pm['artifacts'].append(dict(id=name,kind=kind,origin='model_prediction',parent_ids=parents,
            path=str(pred/'predictions.npz'),sha256=fg.file_sha256(pred/'predictions.npz')))
    pm['method_inputs']=['da3_depth','da3_cameras']
    if branch!='helper_requested_cameras':pm['metric_scale']=dict(status='unresolved',artifact_ids=[],description='arbitrary model scale')
    fg.write_json(pred/'input-manifest.json',pm)
    fg.write_json(pred/'execution.json',dict(status='complete',condition=branch,task_mode='forecast',
                                            output_sha256=fg.file_sha256(pred/'predictions.npz')))
    masks=tmp_path/'masks';masks.mkdir()
    np.savez_compressed(masks/'masks.npz',mask=np.stack(packed),t=tt,v=vv,p=np.zeros(len(tt),int),
        score=np.ones(len(tt))*.9,shape=[160,320],prompts=['wooden block on patterned mat','hand'],
        timestamps=arrays['timestamps'],observed_mask=known[:,:views],source_frame_indices=arrays['source_frame_indices'])
    mm=deepcopy(manifest);mm['artifacts'].append(dict(id='forecast_sam3_masks',kind='masks',origin='model_prediction',
        parent_ids=parents,path=str(masks/'masks.npz'),sha256=fg.file_sha256(masks/'masks.npz')))
    mm['method_inputs']=['forecast_sam3_masks'];fg.write_json(masks/'input-manifest.json',mm)
    config=fg.default_config();config.update(max_nfev=2,points_per_view=32,boundary_samples=16,surface_grid=2)
    config_path=tmp_path/'config.json';fg.write_json(config_path,config)
    return dict(condition=condition,helper=helper,predictions=pred,masks=masks,output=tmp_path/'fit',config_path=config_path)


@pytest.mark.parametrize('branch',['source_unposed','helper_unposed','helper_requested_cameras'])
def test_real_producer_consumer_contract_exports_valid_forecast_bundle(tmp_path,branch):
    args=full_input_fixture(tmp_path,branch)
    report=fg.run(**args)
    assert report['status']=='complete' and report['scene_sha256']==fg.file_sha256(args['output']/'scene-sequence.json')
    assert report['agent_calls']==0 and report['GT_geometry_accuracy'] is None
    scene=fg.load_scene(args['output']/'scene-sequence.json')
    assert scene['objects'][0]['observed']==[True,False,False,False,False]
    assert scene['objects'][0]['prediction_supported'][0] is False
    assert scene['objects'][0]['prediction_provenance']['generated'] is True
    verified=fg.load_manifest(args['output']/'input-manifest.json',[('object_geometry',args['output']/'scene-sequence.json')])
    assert verified['task_mode']=='forecast'
    if branch=='source_unposed':assert 'generated_helper_rgb' not in verified['method_artifact_ids']
    with np.load(args['output']/'geometry.npz') as z:
        assert z['source_frame_indices'].tolist()==[0,-1,-1,-1,-1]
        assert int(z['observed_mask'].sum())==1


def test_source_branch_rejects_masks_that_consume_generated_helpers(tmp_path):
    args=full_input_fixture(tmp_path,'source_unposed')
    path=args['masks']/'input-manifest.json';manifest=json.loads(path.read_text())
    manifest['artifacts'][-1]['parent_ids'].append('generated_helper_rgb');fg.write_json(path,manifest)
    with pytest.raises(ValueError,match='source-only masks'):
        fg.run(**args)
    assert not args['output'].exists()


def rest_fixture(count=5):
    block=np.zeros((80,140),bool);block[25:45,60:80]=True
    hand=np.zeros_like(block);hand[20:45,1:9]=True
    view=dict(object_mask=block,hand_mask=hand,known=True,object_score=.95,
              hand_scores=[.95],hand_rows=[1],selected_row=0)
    frames=[[deepcopy(view)] for _ in range(count)]
    config=fg.default_config();config['initial_rest_prior']=dict(fg.default_initial_rest_prior(),enabled=True)
    return frames,np.arange(count)*.2,config


@pytest.mark.parametrize('change,reason',[
    ('motion','object_image_motion'),('approach','hand_approaching'),
    ('area','object_area_changed'),('missing_object','missing_or_ambiguous_object'),
    ('missing_hand','missing_hand_is_unknown'),('gap','temporal_gap')])
def test_rest_prefix_stops_at_motion_approach_or_unknown_and_never_resumes(change,reason):
    frames,times,config=rest_fixture()
    if change=='motion':frames[2][0]['object_mask']=np.roll(frames[2][0]['object_mask'],10,axis=1)
    elif change=='approach':frames[2][0]['hand_mask']=np.roll(frames[2][0]['hand_mask'],15,axis=1)
    elif change=='area':
        frames[2][0]['object_mask'][:]=False;frames[2][0]['object_mask'][23:47,58:82]=True
    elif change=='missing_object':frames[2][0]['known']=False
    elif change=='missing_hand':frames[2][0]['hand_mask'][:]=False;frames[2][0]['hand_scores']=[]
    else:times[2:]+=.5
    anchor,report=fg.initial_rest_regime(frames,times,config)
    assert anchor.tolist()==[True,True,False,False,False]
    assert report['stop_frame']==2 and report['stop_reason']==reason
    assert report['depth_or_fitted_poses_used'] is False


def test_rest_default_config_stays_legacy_and_unknown_initial_hand_cannot_start_anchor():
    frames,times,config=rest_fixture()
    assert 'initial_rest_prior' not in fg.default_config()
    disabled,report=fg.initial_rest_regime(frames,times,fg.default_config())
    assert not disabled.any() and report['status']=='disabled'
    frames[0][0]['hand_mask'][:]=False;frames[0][0]['hand_scores']=[]
    anchor,report=fg.initial_rest_regime(frames,times,config)
    assert not anchor.any() and report['stop_frame']==0


def test_rest_is_an_in_optimizer_pose_constraint_and_does_not_change_camera_estimates():
    values,masks,initial,truth,config=fixture(views=1,rotation=False)
    # The image mask is stationary at frame 1 while independent depth/K drift.
    masks['mask'][1]=masks['mask'][0]
    values['depth'][1]=values['depth'][0]*1.2
    values['K'][1,0,:2,:2]*=1.2
    hand=np.zeros((60,80),bool);hand[15:40,:5]=True
    masks['mask']=np.concatenate([masks['mask'],np.tile(np.packbits(hand),(3,1))])
    for key,extra in [('t',[0,1,2]),('v',[0,0,0]),('p',[1,1,1]),('score',[.95]*3)]:masks[key]=np.r_[masks[key],extra]
    before_k=values['K'].copy()
    frames=fg.prepare_evidence(values,masks,config)
    baseline=fg.fit_shared_block(initial,frames,values['timestamps'],'helper_requested_cameras',1.,config)
    config['initial_rest_prior']=dict(fg.default_initial_rest_prior(),enabled=True)
    rested=fg.fit_shared_block(initial,frames,values['timestamps'],'helper_requested_cameras',1.,config)
    assert np.linalg.norm(baseline['poses'][1,:3,3]-truth[0,:3,3])>.01
    np.testing.assert_array_equal(rested['poses'][1],truth[0])
    assert rested['report']['initial_rest_prior']['anchored_frame_indices']==[0,1]
    assert rested['report']['free_pose_frame_indices']==[2]
    assert rested['report']['initial_rest_prior']['postfit_pose_snap'] is False
    assert rested['poses'][2,0,3]>rested['poses'][1,0,3]+.02
    np.testing.assert_array_equal(values['K'],before_k)
