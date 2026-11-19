import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    p=Path(__file__).resolve().parents[2]/'benchmarks/world_model_views.py'
    spec=importlib.util.spec_from_file_location('wm_views',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_virtual_cameras_are_rigid_source_relative_and_aim_at_estimated_target():
    m=module();target=np.array([.1,.03,.8]);cams=m.virtual_cameras(target)
    np.testing.assert_allclose(cams[0],np.eye(4))
    r=cams[:,:3,:3]
    np.testing.assert_allclose(r.transpose(0,2,1)@r,np.broadcast_to(np.eye(3),(4,3,3)),atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(r),1.,atol=1e-12)
    for c in cams[1:]:
        pc=(target-c[:3,3])@c[:3,:3]
        np.testing.assert_allclose(pc[:2],0,atol=1e-12)
        assert pc[2]>0


def test_full_source_video_cannot_fill_unknown_views():
    m=module();rgb=np.full((9,4,8,3),255,np.uint8)
    pack,known=m.source_tensor(rgb)
    assert pack.shape==(1,4,9,3,4,8)
    assert known[:,0].all() and not known[:,1:].any()
    assert np.all(pack[0,0]==1) and not pack[:,1:].any()


def test_forecast_encodes_only_one_source_frame():
    m=module();rgb=np.full((1,4,8,3),255,np.uint8)
    pack,known=m.source_tensor(rgb,forecast=True,total_frames=9)
    assert known.sum()==1
    assert not pack[:,1:].any() and not pack[:,:,1:].any()
    with pytest.raises(ValueError,match='exactly one'):
        m.source_tensor(np.repeat(rgb,9,axis=0),forecast=True,total_frames=9)


@pytest.mark.parametrize('target',[[0,0,-1],[0,0,0],[0,float('nan'),1]])
def test_invalid_camera_target_rejected(target):
    with pytest.raises(ValueError):module().virtual_cameras(target)
