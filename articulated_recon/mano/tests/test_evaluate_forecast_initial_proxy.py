import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[2]/'benchmarks')]
from benchmarks import evaluate_forecast_initial_proxy as audit
from artirecon.input_protocol import file_sha256
from representation_agent_tools import geometry_arrays


def seals(tmp_path):
    runs=[]
    for i in range(4):
        scene=tmp_path/f'{i}-scene.json';scene.write_text('{}')
        status=tmp_path/f'{i}-status.json';status.write_text(json.dumps(dict(status='completed',final_scene_valid=True,final_scene_sha256=file_sha256(scene))))
        runs.append(dict(id=str(i),scene_path=str(scene),status_path=str(status),scene_sha256=file_sha256(scene),status_sha256=file_sha256(status)))
    seal=tmp_path/'sealed-suite.json';seal.write_text(json.dumps(dict(all_agents_sealed=True,suite_sha256='suite',runs=runs)))
    return seal


def box(size=.04):return geometry_arrays(dict(type='box',half_extents_m=[size/2]*3))


def export_inputs(tmp_path):
    seal=seals(tmp_path);rgb=np.zeros((1,6,8,3),np.uint8)
    np.savez(tmp_path/'rgb.npz',rgb=rgb,timestamps=np.array([0.]))
    manifest=dict(schema_version=1,condition='rgb_only',task_mode='forecast',observed_until_s=0.,method_inputs=['rgb'],
        artifacts=[dict(id='rgb',kind='rgb',origin='observed_rgb',generated=False,parent_ids=[],time_range_s=[0,0],path='rgb.npz',sha256=file_sha256(tmp_path/'rgb.npz'))])
    (tmp_path/'input-manifest.json').write_text(json.dumps(manifest))
    camera=np.eye(4);camera[:3,3]=[.1,.2,.3]
    np.savez(tmp_path/'cameras.npz',rgb=np.stack([rgb,rgb+255]),c2w=camera[None],camera_ids=np.array(['source']))
    v,f=box();poses=np.tile(np.eye(4),(2,1,1));poses[0,:3,3]=[.4,.5,.6];poses[1,:3,3]=[9,9,9]
    raw=dict(timestamps=[0.,99.],table=dict(normal=[0,0,1],offset=0),objects=[dict(vertices=v.tolist(),faces=f.tolist(),poses=poses.tolist())])
    (tmp_path/'reference.json').write_text(json.dumps(raw))
    return dict(sealed_suite=seal,observed_manifest=tmp_path/'input-manifest.json',raw_reference=tmp_path/'reference.json',
                source_cameras=tmp_path/'cameras.npz',source_view=0,source_camera_id='source',output=tmp_path/'proxy')


def test_all_seals_are_checked_before_evaluator_files_are_opened(tmp_path,monkeypatch):
    path=seals(tmp_path);payload=json.loads(path.read_text());payload['all_agents_sealed']=False;path.write_text(json.dumps(payload))
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('Reference opened before all agents froze'))
    with pytest.raises(ValueError,match='All four'):
        audit.export_proxy(sealed_suite=path,observed_manifest='missing',raw_reference='missing',source_cameras='missing',source_view=0,source_camera_id='source',output=tmp_path/'out')


def test_packaged_seals_reject_candidate_mutation(tmp_path):
    seal=seals(tmp_path);out=tmp_path/'portable';audit.package_seals(seal,out)
    assert len(audit.verify_seals(out/'sealed-suite.json')[1])==4
    (out/'2-scene.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='changed'):audit.verify_seals(out/'sealed-suite.json')


def test_identical_mesh_has_zero_exact_surface_distance():
    v,f=box();r=audit.compare_meshes(v,f,v,f,np.eye(3),samples=256)
    assert r['surface_centroid_error_mm']==0 and r['proxy_axis_extent_MAE_mm']==0
    assert r['symmetric_surface_mean_mm']<1e-7


def test_metric_bias_is_not_removed_by_alignment():
    v,f=box();r=audit.compare_meshes(v+[1,0,0],f,v,f,np.eye(3),samples=256)
    assert r['surface_centroid_error_mm']==pytest.approx(1000)
    assert r['symmetric_surface_mean_mm']>950
    assert r['proxy_axis_extent_MAE_mm']<1e-10


def test_shape_inflation_is_measured_without_rescaling():
    v,f=box();r=audit.compare_meshes(v*2,f,v,f,np.eye(3),samples=256)
    assert r['proxy_axis_extent_MAE_mm']==pytest.approx(40)
    assert r['surface_centroid_error_mm']<1e-10
    assert r['symmetric_surface_mean_mm']>10


def test_point_triangle_distance_covers_face_edge_and_corner():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0]]);f=np.array([[0,1,2]])
    d=audit.point_mesh_distances(np.array([[.2,.3,2.],[.5,-1,0],[-1,-1,0]]),v,f)
    np.testing.assert_allclose(d,[2,1,np.sqrt(2)])


def test_export_uses_camera_inverse_only_and_no_future_proxy(tmp_path):
    args=export_inputs(tmp_path);report=audit.export_proxy(**args)
    assert report['initial_rgb_pixel_identity']=='exact native uint8 equality'
    assert not report['future_motion_scored'] and not report['fitted_alignment']
    with np.load(args['output']/'initial-proxy.npz') as z:
        assert z['pose'].shape==(4,4);np.testing.assert_allclose(z['pose'][:3,3],[.3,.3,.3])
        assert 'poses' not in z and 'goal_pose' not in z


def test_rgb_mismatch_refuses_geometry_reference_without_approximate_matching(tmp_path,monkeypatch):
    args=export_inputs(tmp_path)
    np.savez(args['source_cameras'],rgb=np.ones((1,1,6,8,3),np.uint8),c2w=np.eye(4)[None])
    original=audit.read
    def checked(path):
        if Path(path)==args['raw_reference']:pytest.fail('Geometry opened after pixel mismatch')
        return original(path)
    monkeypatch.setattr(audit,'read',checked)
    with pytest.raises(ValueError,match='initial RGB differ'):audit.export_proxy(**args)
    assert not args['output'].exists()
