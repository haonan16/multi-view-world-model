import os
import pytest
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from mano_recon.hand_evidence import attach_human_evidence,human_loss
from mano_recon.robotfitter_adapter import MANORobot
from mano_recon.model import translation_base


def test_rgb_anchor_rejects_drift_and_unknown_depth_is_not_a_target():
    root=Path(__file__).resolve().parents[1]
    models=Path(os.environ.get('MANO_MODEL_DIR', str(root/'assets/mano')))
    if not (models/'MANO_RIGHT.npz').exists():
        pytest.skip('User-licensed MANO files not installed')
    from mano_recon.amani_objects.geom import DepthViews
    r=MANORobot(models,np.zeros((1,15),np.float32),np.array([[0,0,1]],np.float32),device='cpu',samples=64)
    base=torch.eye(4)[None];q=r.initial.clone();P,_=r.surface(q,base)
    C=np.stack([np.eye(4),np.eye(4)]);C[1,0,3]=.2
    K=np.repeat(np.array([[[200,0,32],[0,200,32],[0,0,1]]],np.float32),2,0)
    views=DepthViews(np.ones((1,2,64,64),np.float32),K,C,device='cpu')
    _,j=r.hand(q[:,:-3],r.betas,translation_base(q[:,-3:]));u,v,z=views.project(j)
    kp=torch.stack([u,v,torch.ones_like(u)],-1).detach().numpy()
    f=SimpleNamespace(dev='cpu',robot=r,views=views,F=P.detach(),Fn=torch.tensor([64]),lo=r.lo,hi=r.hi,
                      n=torch.tensor([0.,0.,1.]),prep={'d':-.7})
    attach_human_evidence(f,kp,np.zeros((1,2,64,64),bool),[True])
    good=human_loss(f,q,base,[0])[0]
    moved=q.clone();moved[:,-3]+=.05
    assert human_loss(f,moved,base,[0])[0]>good+.1
    views.D.fill_(3.)
    torch.testing.assert_close(human_loss(f,q,base,[0])[0],good)
    moved.requires_grad_(True);human_loss(f,moved,base,[0])[0].sum().backward()
    assert moved.grad[0,-3]>0 # gradient descent moves back toward RGB evidence
    # A detector can interchange whole finger chains in just one view. The
    # correspondence option should stop that view penalizing correct geometry.
    order=list(range(21));order[5:9],order[9:13]=order[9:13],order[5:9]
    f.human_keypoints[0,1]=f.human_keypoints[0,1,order].clone()
    inconsistent=human_loss(f,q,base,[0])[0]
    f.finger_pair_hypotheses=True
    corrected=human_loss(f,q,base,[0])[0]
    assert corrected < inconsistent
    assert (corrected-good).abs().max()<.01
