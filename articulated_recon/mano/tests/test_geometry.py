import os
from pathlib import Path
import numpy as np
import pytest
import torch

from mano_recon.fitting import SequenceFitter, project
from mano_recon.model import MANOHand, translation_base
from mano_recon.observations import Observations

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["left", "right"])
def hand(request):
    path = Path(os.environ.get("MANO_MODEL_DIR", str(ROOT / "assets/mano")))
    if not (path / "MANO_RIGHT.npz").exists():
        pytest.skip("User-licensed MANO files not installed")
    return MANOHand(path, side=request.param, samples=128)


def test_wrist_transform_units_and_shared_shape(hand):
    q = torch.zeros(2, hand.dof)
    translation = torch.tensor([[.1,.2,.3], [-.2,.1,.4]])
    v, j = hand(q, base=translation_base(translation))
    assert torch.allclose(j[:, 0], translation, atol=1e-6)
    assert torch.allclose(v[1]-v[0], translation[1]-translation[0], atol=1e-6)
    assert .05 < (v[0].amax(0)-v[0].amin(0)).max() < .3
    assert j.shape == (2,21,3)
    with pytest.raises(ValueError, match="shared"):
        hand(q, betas=torch.zeros(2,10))


def test_shape_and_pose_gradients_agree_with_finite_difference(hand):
    q = torch.zeros(1, hand.dof, requires_grad=True)
    beta = torch.zeros(1,10,requires_grad=True)
    p, n = hand.surface(q, betas=beta)
    loss = p.square().sum()
    loss.backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(beta.grad).all()
    assert q.grad[:,3:].norm() > 0 and beta.grad.norm() > 0
    assert torch.allclose(n.norm(dim=-1),torch.ones_like(n[...,0]),atol=1e-5)
    eps = 1e-3
    for name, index, analytic in [("pose",3,q.grad[0,3]),("shape",0,beta.grad[0,0])]:
        values=[]
        for sign in [-1,1]:
            qq, bb = q.detach().clone(),beta.detach().clone()
            if name=="pose": qq[0,index]+=sign*eps
            else: bb[0,index]+=sign*eps
            values.append(hand.surface(qq,betas=bb)[0].square().sum())
        numeric=(values[1]-values[0])/(2*eps)
        assert torch.allclose(analytic,numeric,rtol=.03,atol=1e-4)


def observation_file(tmp_path, **changes):
    # A 2-metre fronto-parallel plane, seen from two translated cameras.
    T,V,H,W=2,2,8,8
    K=np.tile(np.array([[10,0,4],[0,10,4],[0,0,1]],np.float32),(V,1,1))
    c2w=np.tile(np.eye(4,dtype=np.float32),(V,1,1));c2w[1,0,3]=.1
    payload=dict(depth=np.full((T,V,H,W),2,np.float32),hand_mask=np.ones((T,V,H,W),bool),
                 K=K,c2w=c2w,timestamps=np.array([0.,.1]),depth_type="camera_z_m",camera_convention="opencv_c2w")
    payload.update(changes)
    path=tmp_path/"obs.npz";np.savez(path,**payload)
    return path


def test_backprojection_matches_camera_projection(tmp_path):
    obs=Observations.load(observation_file(tmp_path))
    points,_=obs.select_views([1]).points(max_per_frame=10000)
    uv,z=project(torch.tensor(np.stack(points)),torch.tensor(obs.K[:,1:]),torch.linalg.inv(torch.tensor(obs.c2w[:,1:])))
    y,x=np.indices((8,8))
    expected=torch.tensor(np.stack((x.ravel(),y.ravel()),-1),dtype=torch.float32)
    assert torch.allclose(uv[0,0],expected,atol=1e-5)
    assert torch.allclose(z,torch.full_like(z,2))


def test_missing_depth_is_not_free_space(tmp_path, hand):
    obs=Observations.load(observation_file(tmp_path,depth=np.full((2,2,8,8),np.nan)))
    fitter=SequenceFitter(hand,obs)
    q=torch.zeros(2,hand.dof)
    tr=torch.tensor([[0.,0.,1.],[0.,0.,1.]])
    _,terms=fitter.loss(q,tr,torch.zeros(1,10))
    assert terms['depth']==0 and terms['coverage']==0


def test_zero_confidence_removes_depth_evidence(tmp_path,hand):
    obs=Observations.load(observation_file(tmp_path,confidence=np.zeros((2,2,8,8))))
    _,terms=SequenceFitter(hand,obs).loss(torch.zeros(2,hand.dof),torch.tensor([[0.,0.,1.],[0.,0.,1.]]),torch.zeros(1,10))
    assert terms['depth']==0 and terms['coverage']==0


def test_reject_unknown_units_and_invalid_camera(tmp_path):
    with pytest.raises(ValueError,match="camera_z_m"):
        Observations.load(observation_file(tmp_path,depth_type="millimetres"))
    bad=np.tile(np.eye(4),(2,1,1));bad[0,0,0]=-1
    with pytest.raises(ValueError,match="proper"):
        Observations.load(observation_file(tmp_path,c2w=bad))


def test_heldout_depth_does_not_enter_fit(tmp_path):
    obs=Observations.load(observation_file(tmp_path))
    expected=obs.select_views([0]).points()[0]
    obs.depth[:,1]*=100
    actual=obs.select_views([0]).points()[0]
    assert all(np.array_equal(a,b) for a,b in zip(expected,actual))
