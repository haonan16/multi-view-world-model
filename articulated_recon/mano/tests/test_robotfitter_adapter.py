import os
import pytest
from pathlib import Path
import numpy as np
import torch
from mano_recon.model import translation_base
from mano_recon.robotfitter_adapter import MANORobot, declared_mano_side


@pytest.mark.parametrize('side',['left','right'])
def test_free_wrist_and_shape_gradients_match_mano(side):
    root=Path(__file__).resolve().parents[1]
    models=Path(os.environ.get('MANO_MODEL_DIR', str(root/'assets/mano')))
    if not (models/f'MANO_{side.upper()}.npz').exists():
        pytest.skip('User-licensed MANO files not installed')
    r=MANORobot(models,np.zeros((2,15),np.float32),np.array([[.1,.2,.3],[.2,.3,.4]],np.float32),device='cpu',samples=64,side=side)
    assert r.hand.side==side
    q=r.initial.clone().requires_grad_(True);r.betas.requires_grad_(True)
    base=torch.eye(4)[None].repeat(2,1,1)
    p,n=r.surface(q,base)
    expected,_=r.hand.surface(q[:,:-3],translation_base(q[:,-3:]),r.betas)
    torch.testing.assert_close(p,expected)
    torch.testing.assert_close(p[1]-p[0],torch.tensor([.1,.1,.1]).expand_as(p[0]),atol=1e-6,rtol=1e-5)
    p.square().mean().backward()
    assert torch.isfinite(q.grad).all() and q.grad[:,-3:].abs().sum()>0
    assert torch.isfinite(r.betas.grad).all() and r.betas.grad.abs().sum()>0


def test_strict_rgb_side_cannot_silently_default_to_right():
    assert declared_mano_side({})=='right'
    assert declared_mano_side({'mano_side':'left'},True)=='left'
    assert declared_mano_side({'mano_side':'right'},True)=='right'
    with pytest.raises(ValueError,match='explicit mano_side'):
        declared_mano_side({},True)
    with pytest.raises(ValueError,match='left or right'):
        declared_mano_side({'mano_side':'unknown'},True)
