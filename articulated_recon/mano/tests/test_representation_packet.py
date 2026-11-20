"""Source-only packet/instrumentation invariants on generated tiny fixtures."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parents[2]/'benchmarks'
sys.path.insert(0, str(BENCHMARKS))
import representation_packet as packets
import representation_agent_tools as checks
import representation_agent_harness as harness


def observations(count=7):
    h, w = 18, 20
    y, x = np.mgrid[:h,:w]
    rgb = np.stack([np.full((1,h,w,3),20+t,np.uint8) for t in range(count)])
    depth = np.stack([1+.001*x+.0001*y*y+.003*t for t in range(count)])[:,None]
    k = np.broadcast_to(np.array([[40.,0,10],[0,40.,9],[0,0,1]]), (count,1,3,3)).copy()
    c = np.broadcast_to(np.eye(4),(count,1,4,4)).copy()
    c[:,0,0,3] = np.linspace(0,.1,count)
    mask = np.zeros((count,1,h,w),bool)
    mask[:,:,3:15,4:16] = True
    return dict(rgb=rgb,depth_m=depth,K=k,c2w=c,timestamps=np.arange(count)*.1,
                object_masks=mask,object_mask_known=np.ones((count,1),bool),
                occlusion_mask=np.zeros_like(mask))


def source(tmp_path, origin='model_prediction'):
    folder = tmp_path/'source'
    folder.mkdir()
    data = observations()
    np.savez_compressed(folder/'rgb.npz',rgb=data['rgb'],timestamps=data['timestamps'])
    np.savez_compressed(folder/'geometry.npz',**data)
    sha = packets.file_sha256
    manifest = dict(schema_version=1,condition='rgb_only',task_mode='reconstruction',
        method_inputs=['depth','cameras','masks'],
        metric_scale=dict(status='estimated',artifact_ids=['depth'],description='model-predicted scale'),
        artifacts=[dict(id='rgb',kind='rgb',origin='observed_rgb',generated=False,parent_ids=[],
                        path='rgb.npz',sha256=sha(folder/'rgb.npz'),time_range_s=[0,.6])])
    for name, kind in [('depth','depth'),('cameras','cameras'),('masks','masks')]:
        manifest['artifacts'].append(dict(id=name,kind=kind,origin=origin,parent_ids=['rgb'],
            path='geometry.npz',sha256=sha(folder/'geometry.npz')))
    packets.write_json(folder/'input-manifest.json',manifest)
    return folder, data


def make_packet(tmp_path, folder, condition, budget=5, **kwargs):
    return packets.build_packet(observations=folder/'geometry.npz',manifest=folder/'input-manifest.json',
        output=tmp_path/condition,condition=condition,budget=budget,
        config=packets.default_config('explicit-test-model',sys.executable),
        weak_hull_initializer=condition=='B_object',frontend_context_frame_ids=list(range(7)),**kwargs)


def test_dynamic_native_points_do_not_get_reconstructed_through_approximate_k():
    data = observations()
    native = np.ones(data['depth_m'].shape+(3,),np.float64)*.713456789123
    data['points_world'] = native.copy()
    data['K_pinhole_rmse_px'] = np.ones((7,1))*2.7
    normalized = packets.normalize_observations(data)
    np.testing.assert_array_equal(normalized['points_world'],native)
    np.testing.assert_array_equal(normalized['c2w'],data['c2w'])
    np.testing.assert_array_equal(normalized['K_pinhole_rmse_px'],data['K_pinhole_rmse_px'])


def test_missing_mask_and_invalid_depth_stay_unknown():
    data = observations()
    del data['object_masks'], data['object_mask_known']
    data['depth_m'][0,0,0,0] = np.nan
    normalized = packets.normalize_observations(data)
    assert not normalized['object_mask_known'].any()
    assert not normalized['depth_valid'][0,0,0,0]
    with pytest.raises(ValueError,match='missing is unknown'):
        packets.compact_initializer(normalized,object_id='object',metric_status='estimated',coordinate_frame={})


def test_occluded_geometry_is_neither_foreground_penalty_nor_depth_error():
    rendered = np.array([[1.,5.],[1.,5.]])
    depth = np.ones((2,2))
    mask = np.ones((2,2),bool)
    occlusion = np.array([[False,True],[False,True]])
    result = checks.agreement(rendered,depth,mask,True,occlusion,np.ones((2,2),bool))
    assert result['visible_mask_iou']==1
    assert result['predicted_depth_agreement_median_m']==0
    assert result['depth_overlap_pixels']==2
    missing = checks.agreement(rendered,depth,mask,False,occlusion,np.ones((2,2),bool))
    assert missing['visible_mask_iou'] is None
    hidden = checks.agreement(rendered,depth,mask,True,np.ones((2,2),bool),np.ones((2,2),bool))
    assert hidden['predicted_depth_agreement_median_m'] is None


def test_matched_packets_share_evidence_prompt_config_and_preserve_context(tmp_path):
    folder,_ = source(tmp_path)
    a = make_packet(tmp_path,folder,'A_raw')
    b = make_packet(tmp_path,folder,'B_object')
    assert a['native_frame_ids']==b['native_frame_ids']==[0,2,3,4,6]
    assert a['frontend_context_frame_ids']==list(range(7))
    af,bf = [packets.verify_packet(tmp_path/n) for n in ['A_raw','B_object']]
    assert af['common_evidence_sha256']==bf['common_evidence_sha256']
    assert af['prompt_sha256']==bf['prompt_sha256']
    assert af['config_sha256']==bf['config_sha256']
    assert not (tmp_path/'A_raw'/'initializer-scene.json').exists()
    with np.load(tmp_path/'B_object'/'representation-targets.npz') as z:
        assert not z['shape_known'].any()
        assert not z['rotation_known'].any()
        assert not z['contact_known'].any()
    suite = harness.prepare_suite([tmp_path/'A_raw',tmp_path/'B_object'],tmp_path/'suite')
    assert suite['agent_status']=='not_started'
    assert len(suite['runs'])==2


def test_source_checker_renders_shared_geometry_without_gt(tmp_path):
    folder,_ = source(tmp_path)
    packet = make_packet(tmp_path,folder,'B_object')
    root = tmp_path/'B_object'
    result = checks.check_scene(root,root/'initializer-scene.json',root/'output/check-001')
    assert result['status']=='source_check_complete'
    assert result['GT_used'] is False
    assert result['gt_geometry_accuracy'] is None
    assert len(result['per_frame'])==5
    assert len(list((root/'output/check-001').glob('*overlay.png')))==5
    scene = json.loads((root/'initializer-scene.json').read_text())
    scene['objects'][0]['T_world_object'][0][0][0] = 2
    with pytest.raises(ValueError,match='orthonormal'):
        checks.validate_export(scene,packet)


def test_raw_ground_truth_ancestry_is_rejected_before_packet_materialization(tmp_path):
    folder,_ = source(tmp_path,origin='ground_truth')
    with pytest.raises(ValueError,match='forbidden ground_truth'):
        make_packet(tmp_path,folder,'A_raw')
    assert not (tmp_path/'A_raw').exists()


def test_frozen_input_mutation_is_detected(tmp_path):
    folder,_ = source(tmp_path)
    make_packet(tmp_path,folder,'A_raw')
    (tmp_path/'A_raw/images/000-source-00000.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen packet input changed'):
        packets.verify_packet(tmp_path/'A_raw')


def test_explicit_structured_initializer_cannot_hide_dense_frame_preprocessing(tmp_path):
    folder,data = source(tmp_path)
    normalized = packets.normalize_observations(data)
    ids = packets.select_frames(7,5)
    coordinate = dict(name='predicted_world',handedness='right',up_axis='unknown',pose_convention='T_world_local')
    scene = packets.compact_initializer({k:v[ids] for k,v in normalized.items()},object_id='task_object',
                                       metric_status='estimated',coordinate_frame=coordinate)
    scene_path, receipt_path = tmp_path/'init.json',tmp_path/'receipt.json'
    packets.write_json(scene_path,scene)
    receipt = dict(source_observations_sha256=packets.file_sha256(folder/'geometry.npz'),
        native_frame_ids=list(range(7)), frontend_context_frame_ids=list(range(7)),
        input_scope='selected_frames_only',producer_sha256='a'*64,initializer_sha256=packets.file_sha256(scene_path))
    packets.write_json(receipt_path,receipt)
    with pytest.raises(ValueError,match='native_frame_ids'):
        make_packet(tmp_path,folder,'B_object',initializer=scene_path,initializer_receipt=receipt_path)
    receipt['native_frame_ids']=ids.tolist()
    packets.write_json(receipt_path,receipt)
    packet = make_packet(tmp_path,folder,'B_object',initializer=scene_path,initializer_receipt=receipt_path)
    assert packet['initializer_kind']=='source_model_mesh_rigid_track'


def test_track_selection_keeps_source_correspondence_ids_and_gauge(tmp_path):
    times = np.arange(7)*.1
    tracks = np.arange(7*3*3,dtype=float).reshape(7,3,3)
    path = tmp_path/'tracks.npz'
    np.savez(path,tracks3d=tracks,timestamps=times,query_frame=[0,1,6],
             query_pixels=[[1,2],[3,4],[5,6]],visible=np.ones((7,3),bool),
             coordinate_frame_name=np.asarray('predicted_world'))
    result = packets.selected_tracks(path,np.array([0,3,6]),times,'predicted_world')
    np.testing.assert_array_equal(result['tracks3d'],tracks[[0,3,6]][:,[0,2]])
    assert result['query_column_ids'].tolist()==[0,2]
    with pytest.raises(ValueError,match='gauge'):
        packets.selected_tracks(path,np.array([0,3,6]),times,'different_world')


def test_feature_targets_default_to_unknown_not_missing_key_means_true(tmp_path):
    folder,_=source(tmp_path)
    make_packet(tmp_path,folder,'B_object')
    scene=json.loads((tmp_path/'B_object/initializer-scene.json').read_text())
    scene['objects'][0]['provenance']={'method':'unspecified'}
    scene['objects'][0]['geometry']['provenance']={'method':'unspecified'}
    features=packets.feature_targets(scene)
    assert not features['shape_known'].any()
    assert not features['rotation_known'].any()
    scene['objects'][0]['provenance'].update(rotation_status='estimated_where_observed',
                                           rotation_observed=[True,False,True,False,True])
    assert packets.feature_targets(scene)['rotation_known'][:,0].tolist()==[True,False,True,False,True]


def test_cli_accounting_distinguishes_turns_requests_tools_and_unknown_cost(tmp_path):
    path=tmp_path/'events.jsonl'
    events=[dict(type='item.started',item=dict(id='tool0',type='command_execution')),
            dict(type='item.completed',item=dict(id='tool0',type='command_execution',exit_code=0)),
            dict(type='item.completed',item=dict(id='tool0',type='command_execution',exit_code=0)),
            dict(type='item.completed',item=dict(id='tool1',type='file_change',status='failed')),
            dict(type='turn.completed',usage=dict(input_tokens=300,cached_input_tokens=100,output_tokens=50))]
    path.write_text('\n'.join(json.dumps(x) for x in events)+'\n{"partial":')
    counts=harness.event_accounting(path)
    assert counts['tool_calls_started']==2
    assert counts['failed_tool_calls']==1
    assert counts['model_requests'] is None and counts['monetary_cost'] is None
    assert counts['input_tokens']==300 and counts['output_tokens']==50
    config=packets.default_config('test',sys.executable)
    config['max_tool_calls']=1
    assert harness.budget_exceeded(counts,1,config)=='tool_calls'


def test_checkpoints_preserve_first_observed_candidate_and_reject_bad_scene(tmp_path):
    folder,_=source(tmp_path)
    packet=make_packet(tmp_path,folder,'B_object')
    path=tmp_path/'B_object/initializer-scene.json'
    checkpoints=tmp_path/'checkpoints'
    checkpoints.mkdir()
    records=[]
    harness.checkpoint(path,packet,checkpoints,records,1.,20)
    harness.checkpoint(path,packet,checkpoints,records,2.,20)
    assert len(records)==1 and records[0]['first_observed_elapsed_seconds']==1.
    path.write_text('{}')
    harness.checkpoint(path,packet,checkpoints,records,3.,20)
    assert len(records)==1


def test_fresh_packet_rejects_unlisted_evidence_prior_results_and_repeat_execution(tmp_path):
    folder,_=source(tmp_path)
    make_packet(tmp_path,folder,'A_raw')
    root=tmp_path/'A_raw'
    extra=root/'old-reconstruction.json'
    extra.write_text('{}')
    with pytest.raises(ValueError,match='Unlisted'):
        harness.assert_fresh_packet(root)
    extra.unlink()
    (root/'output').mkdir()
    (root/'output/preflight.txt').write_text(harness.PREFLIGHT_TEXT)
    harness.assert_fresh_packet(root)
    prior=root/'output/scene-sequence.json'
    prior.write_text('{}')
    with pytest.raises(ValueError,match='not empty'):
        harness.assert_fresh_packet(root)
    prior.unlink()
    (root/'execution-receipt.json').write_text('{}')
    with pytest.raises(ValueError,match='execution receipt'):
        harness.assert_fresh_packet(root)


def test_original_source_indices_survive_strided_frontend_and_track_selection(tmp_path):
    data=observations()
    data['source_frame_indices']=np.arange(7)*5
    normalized=packets.normalize_observations(data)
    assert normalized['source_frame_indices'].tolist()==[0,5,10,15,20,25,30]
    path=tmp_path/'tracks.npz'
    values=dict(tracks3d=np.ones((7,3,3)),timestamps=data['timestamps'],query_frame=[0,1,6],
                query_frame_indexing=np.asarray('geometry_rows'),coordinate_frame_name=np.asarray('predicted_world'))
    np.savez(path,**values)
    result=packets.selected_tracks(path,np.array([0,3,6]),data['timestamps'],'predicted_world',data['source_frame_indices'])
    assert result['native_frame_ids'].tolist()==[0,15,30]
    assert result['geometry_row_ids'].tolist()==[0,3,6]
    assert result['query_frame'].tolist()==[0,30]
