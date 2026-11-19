import pytest
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'actions'))
from prepare_actions import grasp_rotation

def test_grasp_frame_is_right_handed_and_has_requested_approach():
    for a,c in [([0,0,-1],[0,1,0]),([1,0,0],[0,1,0])]:
        R=grasp_rotation(a,c)
        np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-10)
        assert np.linalg.det(R)>.999999
        np.testing.assert_allclose(R[:,2],a)

def test_generated_actions_preserve_rigid_poses_and_support():
    root=ROOT/'runs/recorded-data-actions-20261112/inputs'
    if not all((root/case/'plan.npz').exists() for case in ['block','bottle','button']):
        pytest.skip('Historical recorded action outputs not installed')
    for case in ['block','bottle','button']:
        z=np.load(root/case/'plan.npz')
        p=z['tcp_world'];assert np.isfinite(p).all()
        np.testing.assert_allclose(p[:,:3,:3].transpose(0,2,1)@p[:,:3,:3],np.broadcast_to(np.eye(3),(len(p),3,3)),atol=1e-6)
        assert np.all(z['duration_s']>0)
        assert set(z['gripper']).issubset({-1.,1.})
        # ManiSkill interprets absolute rotations with intrinsic XYZ convention.
        np.testing.assert_allclose(Rotation.from_euler('XYZ',Rotation.from_matrix(p[:,:3,:3]).as_euler('XYZ')).as_matrix(),p[:,:3,:3],atol=1e-6)
        if case=='block':assert 'release' in z['phase']
        if case=='bottle':assert 'release' not in z['phase']
