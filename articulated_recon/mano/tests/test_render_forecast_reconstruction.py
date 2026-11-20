"""Timing, 3D interpolation and actual MP4 delivery on generated fixtures."""
import importlib.util
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
import render_forecast_reconstruction as render


def test_thirty_fps_display_holds_three_fps_rgb_without_extra_predictions():
    times=np.arange(17)/3
    queries,rows,phases=render.timeline(times,30,.5,.5)
    assert len(queries)==190 # 5.333-second prediction horizon + two 0.5-second holds
    assert set(rows)==set(range(17))
    assert np.all(rows[:15]==0) and np.all(rows[-15:]==16)
    assert np.all(queries[:15]==0) and np.all(queries[-15:]==times[-1])
    assert queries.min()==0 and queries.max()==times[-1]
    assert phases[:15]==['initial hold']*15 and phases[-15:]==['final hold']*15
    # Every RGB query is a whole native row. No synthetic image time is inserted.
    assert rows.dtype.kind in 'iu'


def test_only_supported_3d_poses_interpolate_and_stay_rigid():
    times=np.array([0.,1.,2.]);poses=np.tile(np.eye(4),(3,1,1));poses[:,0,3]=[0,1,2]
    pose,label=render.supported_pose(poses,times,np.ones(3,bool),.5)
    assert pose[0,3]==.5 and label=='interpolated 3D pose'
    np.testing.assert_allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3))
    missing,label=render.supported_pose(poses,times,np.array([True,False,True]),.5)
    assert missing is None and label=='unsupported interval'
    native,label=render.supported_pose(poses,times,np.ones(3,bool),.5,interpolate=False)
    assert native[0,3]==0 and label=='native fitted pose'


def fixture_module():
    path=Path(__file__).with_name('test_forecast_geometry.py')
    spec=importlib.util.spec_from_file_location('forecast_renderer_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_orbit_contains_projected_3d_plane_cloud_and_shared_box():
    module=fixture_module();values,_,scene,_,_=module.fixture(views=2)
    scene['objects'][0]['T_world_object']=np.tile(scene['objects'][0]['T_world_object'],(3,1,1)).tolist()
    scene['objects'][0]['observed']=[True,False,False]
    scene['objects'][0]['prediction_supported']=[False,True,True]
    display=render.display_geometry(scene,scene['objects'][0])
    arrays=dict(rgb=values['rgb'],depth_m=values['depth'],K=values['K'],c2w=values['c2w'])
    points,colors=render.predicted_cloud(arrays,0,display,max_points=400)
    assert len(points)>0 and display['plane'] is not None
    image=render.orbit_panel(display,points,colors,display['poses'][0],.2,(240,320))
    changed=render.orbit_panel(display,points,colors,display['poses'][0],.8,(240,320))
    assert image.shape==(240,320,3) and image.dtype==np.uint8
    assert len(np.unique(image.reshape(-1,3),axis=0))>10
    assert not np.array_equal(image,changed)


def test_actual_encoded_video_decodes_and_records_native_hold_semantics(tmp_path):
    module=fixture_module();args=module.full_input_fixture(tmp_path,'helper_requested_cameras')
    module.fg.run(**args)
    before=render.file_sha256(args['helper']/'observations.npz')
    result=render.render(helper=args['helper'],bundle=args['output'],output=tmp_path/'video',
        fps=30,initial_hold_s=1/30,final_hold_s=1/30,panel_width=384,max_points=400)
    assert result['status']=='complete' and result['display_fps']==30
    assert result['display_frames']==26
    assert result['rgb_interpolation'] is False and result['depth_interpolation'] is False
    assert result['hand_mesh_present'] is False and result['GT_used'] is False
    assert result['source_frame_for_display'][-1]==4
    assert (tmp_path/'video/forecast-reconstruction.mp4').stat().st_size>1000
    assert len(list((tmp_path/'video').glob('preview-*.png')))==3
    assert render.file_sha256(args['helper']/'observations.npz')==before


def test_mesh_geometry_and_nonzero_local_origin_are_preserved_in_the_orbit():
    module=fixture_module();_,_,scene,_,_=module.fixture(views=1)
    obj=scene['objects'][0]
    vertices=np.array([[.03,.02,.01],[.13,.02,.01],[.03,.11,.01],[.03,.02,.09]])
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    obj['geometry']=dict(type='mesh',vertices_m=vertices.tolist(),faces=faces.tolist())
    display=render.display_geometry(scene,obj)
    np.testing.assert_array_equal(display['vertices'],vertices)
    np.testing.assert_array_equal(display['faces'],faces)
    np.testing.assert_allclose(display['target'],[.08,.065,.85])
    image=render.orbit_panel(display,np.empty((0,3)),np.empty((0,3),np.uint8),display['poses'][0],.4,(240,320))
    # Same local bounds/framing, different surface: the renderer must not draw
    # the tetrahedron as its enclosing box.
    box_v,box_f=render.geometry_arrays(dict(type='box',half_extents_m=np.ptp(vertices,axis=0)/2))
    box_v+=(vertices.min(0)+vertices.max(0))/2
    alternate=deepcopy(obj);alternate['geometry']=dict(type='mesh',vertices_m=box_v.tolist(),faces=box_f.tolist())
    box_display=render.display_geometry(scene,alternate)
    np.testing.assert_allclose(box_display['target'],display['target'])
    box_image=render.orbit_panel(box_display,np.empty((0,3)),np.empty((0,3),np.uint8),display['poses'][0],.4,(240,320))
    assert np.count_nonzero(np.any(image!=box_image,axis=2))>100
    assert len(display['edges'])==6


def test_task_selection_can_choose_cylinder_and_rejects_ambiguous_multiobject_scene():
    module=fixture_module();_,_,scene,_,_=module.fixture(views=1)
    distractor=deepcopy(scene['objects'][0]);distractor['id']='background_box'
    obj=scene['objects'][0];obj['geometry']=dict(type='cylinder',radius_m=.04,height_m=.13)
    scene['objects'].insert(0,distractor)
    assert render.task_object(scene,'block') is obj
    with pytest.raises(ValueError,match='multiple objects'):render.task_object(scene)
    with pytest.raises(ValueError,match='missing'):render.task_object(scene,'absent')
    display=render.display_geometry(scene,obj)
    np.testing.assert_allclose(display['local_aabb_extent'],[.08,.08,.13])
    assert len(display['vertices'])>8 and len(display['faces'])>12
    image=render.orbit_panel(display,np.empty((0,3)),np.empty((0,3),np.uint8),display['poses'][0],.5,(180,240))
    assert image.shape==(180,240,3) and np.any(image[:,:,0]>180)


def test_actual_agent_mesh_video_keeps_helper_images_without_fitted_cameras(tmp_path):
    module=fixture_module();args=module.full_input_fixture(tmp_path,'source_unposed')
    module.fg.run(**args)
    # Create a new synthetic sealed mesh result. Real frozen output scenes are
    # never modified by the renderer.
    scene_path=args['output']/'scene-sequence.json';scene=json.loads(scene_path.read_text())
    half=np.array(scene['objects'][0]['geometry']['half_extents_m'])
    vertices=np.array([[-1,-1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])*half
    scene['objects'][0]['geometry']=dict(type='mesh',vertices_m=vertices.tolist(),
        faces=[[0,2,1],[0,1,3],[0,3,2],[1,2,3]],provenance=dict(method='synthetic mesh fixture'))
    scene_path.write_text(json.dumps(scene))
    sha=render.file_sha256(scene_path)
    manifest_path=args['output']/'input-manifest.json';manifest=json.loads(manifest_path.read_text())
    for record in manifest['artifacts']:
        if record.get('path')=='scene-sequence.json':record['sha256']=sha
    manifest_path.write_text(json.dumps(manifest))
    receipt_path=args['output']/'execution.json';receipt=json.loads(receipt_path.read_text());receipt['scene_sha256']=sha
    receipt_path.write_text(json.dumps(receipt))
    source,arrays,loaded,obj,_=render.load_inputs(args['helper'],args['output'],object_id='block')
    assert source['rgb'].shape[1]==4 and arrays['K'].shape[1]==1 and obj['geometry']['type']=='mesh'
    result=render.render(helper=args['helper'],bundle=args['output'],output=tmp_path/'mesh-video',
        fps=10,initial_hold_s=0,final_hold_s=.1,panel_width=160,max_points=100,object_id='block')
    assert result['object_geometry_type']=='mesh' and result['object_vertices']==4 and result['object_faces']==4
    assert result['scene_sha256']==sha and render.file_sha256(scene_path)==sha
    assert result['display_frames']==9 and result['display_fps']==10
    assert (tmp_path/'mesh-video/forecast-reconstruction.mp4').stat().st_size>1000
