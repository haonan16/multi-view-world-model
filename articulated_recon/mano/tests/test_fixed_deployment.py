from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_registered_plan import scene as example_scene, materialized_inputs
from benchmarks.fixed_deployment import virtual_transform, create, attach
from benchmarks.registered_plan import validate_registration
from artirecon.input_protocol import file_sha256, load_manifest


def scene_in_camera():
    scene=example_scene()
    scene['coordinate_frame']['name']='source_camera'
    rotation=Rotation.from_euler('x',.7).as_matrix()
    trans=np.array([.2,.3,.8])
    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=trans
    scene['objects'][0]['T_world_object']=(transform@np.asarray(scene['objects'][0]['T_world_object'])).tolist()
    n=rotation@np.array([0,0,1.])
    scene['support']['normal_world']=n.tolist();scene['support']['offset_m']=-float(n@trans)
    return scene


def test_virtual_placement_preserves_geometry_and_places_estimated_plane():
    scene=scene_in_camera();original=deepcopy(scene)
    transform=virtual_transform(scene)
    r=transform[:3,:3]
    np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-12)
    assert np.linalg.det(r)==pytest.approx(1)
    positions=np.asarray(scene['objects'][0]['T_world_object'])[:,:3,3]
    mapped=positions@r.T+transform[:3,3]
    np.testing.assert_allclose(mapped[0],[.45,0,.025],atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(np.diff(mapped,axis=0),axis=1),
                               np.linalg.norm(np.diff(positions,axis=0),axis=1),atol=1e-12)
    assert scene==original


def fixture(tmp_path):
    scene_path,manifest_path,_=materialized_inputs(tmp_path)
    scene_path.write_text(json.dumps(scene_in_camera()))
    manifest=json.loads(manifest_path.read_text())
    manifest['artifacts']=[a for a in manifest['artifacts'] if a['id']!='registration']
    manifest['method_inputs'].remove('registration')
    manifest['robot_registration']={'status':'unresolved','artifact_ids':[], 'description':'No target robot registration yet'}
    manifest['artifacts'].append({'id':'source_scene','kind':'object_geometry','origin':'model_prediction',
        'parent_ids':['depth'],'path':str(scene_path),'sha256':file_sha256(scene_path)})
    manifest['method_inputs'].append('source_scene')
    manifest_path.write_text(json.dumps(manifest))
    return scene_path,manifest_path


def test_same_deployment_bytes_are_bound_without_replacing_variant_scene(tmp_path):
    scene_path,manifest_path=fixture(tmp_path)
    deployment=create(scene_path,manifest_path,'synthetic-camera',tmp_path/'deployment')
    assert deployment['deployment_type']=='virtual'
    validate_registration(deployment,'source_camera')
    receipt=attach(manifest_path,tmp_path/'deployment/input-manifest.json',tmp_path/'variant/manifest.json')
    report=load_manifest(receipt['manifest'])
    assert report['execution_ready']
    variant=json.loads((tmp_path/'variant/manifest.json').read_text())
    records={a['id']:a for a in variant['artifacts']}
    assert records['source_scene']['sha256']==file_sha256(scene_path)
    assert records['deployment__fixed_deployment']['sha256']==file_sha256(tmp_path/'deployment/deployment.json')


def test_cannot_attach_full_video_calibration_to_forecast(tmp_path):
    scene_path,manifest_path=fixture(tmp_path)
    create(scene_path,manifest_path,'synthetic-camera',tmp_path/'deployment')
    manifest=json.loads(manifest_path.read_text());manifest['task_mode']='forecast'
    # Forecast validity is tested by input_protocol; use an empty permitted RGB horizon here.
    manifest['observed_until_s']=8.23
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        attach(manifest_path,tmp_path/'deployment/input-manifest.json',tmp_path/'variant.json')


def test_multi_object_setup_needs_explicit_target():
    scene=scene_in_camera();scene['objects'].append(deepcopy(scene['objects'][0]))
    with pytest.raises(ValueError,match='explicit target'):virtual_transform(scene)


def forecast_fixture(tmp_path):
    from test_world_model_forecast import fixture_inputs, refresh_artifact_hashes
    args,_=fixture_inputs(tmp_path)
    scene=json.loads(args['scene_path'].read_text())
    scene['support']={'id':'table','normal_world':[0.,-1.,0.], 'offset_m':.1,
        'confidence':None,'provenance':{'method':'Synthetic first-image plane','artifact_ids':['depth'],
                                      'confidence_status':'uncalibrated'}}
    args['scene_path'].write_text(json.dumps(scene))
    refresh_artifact_hashes(args,args['scene_path'])
    return args


def test_forecast_deployment_requires_explicit_opt_in(tmp_path):
    args=forecast_fixture(tmp_path)
    with pytest.raises(ValueError,match='explicit first_image_forecast'):
        create(args['scene_path'],args['scene_manifest'],'synthetic-camera',tmp_path/'deployment')
    assert not (tmp_path/'deployment').exists()


def test_first_image_transform_is_fixed_across_generated_future_methods(tmp_path):
    args=forecast_fixture(tmp_path)
    original=args['scene_path'].read_bytes()
    deployment=create(args['scene_path'],args['scene_manifest'],'synthetic-camera',tmp_path/'deployment',
                      first_image_forecast=True)
    assert deployment['initialization_mode']=='strict_first_image_forecast'
    assert deployment['observed_until_s']==2.
    assert deployment['source_image_sha256']==file_sha256(args['geometry_bundle']/'rgb.npz')
    assert deployment['source_input_manifest_sha256']==file_sha256(args['scene_manifest'])
    assert args['scene_path'].read_bytes()==original
    np.testing.assert_allclose(deployment['T_robot_scene'],virtual_transform(json.loads(original)))
    validate_registration(deployment,'source_camera')
    for name in ['forecast-a','forecast-b']:
        manifest=json.loads(args['scene_manifest'].read_text())
        predicted=tmp_path/f'{name}.npz'
        np.savez(predicted,rgb=np.zeros((4,12,16,3),np.uint8),timestamps=[2.,3.,4.,5.])
        manifest['artifacts'].append(dict(id='predicted_future',kind='rgb',origin='model_prediction',
            generated=True,parent_ids=['source'],path=str(predicted),sha256=file_sha256(predicted),
            time_range_s=[2.,5.]))
        manifest['method_inputs'].append('predicted_future')
        method=tmp_path/f'{name}-inputs.json';method.write_text(json.dumps(manifest))
        path=tmp_path/name/'registered.json'
        receipt=attach(method,tmp_path/'deployment/input-manifest.json',path)
        assert load_manifest(path)['execution_ready']
        assert receipt['shared_registration_sha256']==file_sha256(tmp_path/'deployment/deployment.json')
        attached=json.loads(path.read_text())
        record=next(a for a in attached['artifacts'] if a['id']=='predicted_future')
        assert record['generated'] is True and record['time_range_s']==[2.,5.]


@pytest.mark.parametrize('part',['recorded_rgb','depth','camera','scene','generated_future'])
def test_forecast_deployment_rejects_future_ancestry_with_valid_hashes(tmp_path,part):
    from test_world_model_forecast import refresh_artifact_hashes
    args=forecast_fixture(tmp_path)
    root=args['geometry_bundle']
    if part=='recorded_rgb':
        with np.load(root/'rgb.npz') as z:rgb=z['rgb'].copy()
        np.savez(root/'rgb.npz',rgb=np.repeat(rgb,2,axis=0),timestamps=[2.,3.])
        refresh_artifact_hashes(args,root/'rgb.npz')
    elif part in {'depth','camera'}:
        path=root/'geometry.npz'
        with np.load(path) as z:arrays={key:z[key].copy() for key in z.files}
        if part=='depth':arrays['depth_m']=np.repeat(arrays['depth_m'],2,axis=0)
        else:arrays['c2w']=np.repeat(arrays['c2w'],2,axis=0)
        np.savez(path,**arrays);refresh_artifact_hashes(args,path)
    elif part=='scene':
        scene=json.loads(args['scene_path'].read_text());scene['timestamps']=[2.,3.]
        args['scene_path'].write_text(json.dumps(scene));refresh_artifact_hashes(args,args['scene_path'])
    else:
        path=root/'future.npz';np.savez(path,rgb=np.zeros((2,12,16,3),np.uint8),timestamps=[2.,3.])
        manifest=json.loads(args['scene_manifest'].read_text())
        manifest['artifacts'].append(dict(id='future',kind='rgb',origin='model_prediction',generated=True,
            parent_ids=['source'],path=str(path),sha256=file_sha256(path),time_range_s=[2.,3.]))
        manifest['method_inputs'].append('future');args['scene_manifest'].write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        create(args['scene_path'],args['scene_manifest'],'synthetic-camera',tmp_path/'deployment',first_image_forecast=True)
    assert not (tmp_path/'deployment').exists()


def test_forecast_opt_in_rejects_reconstruction_inputs(tmp_path):
    scene_path,manifest_path=fixture(tmp_path)
    with pytest.raises(ValueError,match='RGB-only forecast ancestry'):
        create(scene_path,manifest_path,'synthetic-camera',tmp_path/'deployment',first_image_forecast=True)
    assert not (tmp_path/'deployment').exists()


def test_forecast_attachment_rejects_different_cutoff_for_same_image(tmp_path):
    args=forecast_fixture(tmp_path)
    create(args['scene_path'],args['scene_manifest'],'synthetic-camera',tmp_path/'deployment',
           first_image_forecast=True)
    manifest=json.loads(args['scene_manifest'].read_text());manifest['observed_until_s']=3.
    path=tmp_path/'different-cutoff.json';path.write_text(json.dumps(manifest))
    # Generic forecast lineage allows an observation before the cutoff, but
    # this strict first-image deployment must retain its actual anchor time.
    assert load_manifest(path)['task_mode']=='forecast'
    with pytest.raises(ValueError,match='same observed-time cutoff'):
        attach(path,tmp_path/'deployment/input-manifest.json',tmp_path/'attached.json')
    assert not (tmp_path/'attached.json').exists()
