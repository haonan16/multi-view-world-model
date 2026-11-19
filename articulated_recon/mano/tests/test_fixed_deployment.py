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
