"""Timing, 3D interpolation and actual MP4 delivery on generated fixtures."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

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
