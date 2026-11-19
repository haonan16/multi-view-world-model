import os
import pytest
from pathlib import Path
import torch
from mano_recon.model import MANOHand

def test_pca_expansion_preserves_saved_geometry_and_gradients():
    root=Path(__file__).resolve().parents[1]
    models=Path(os.environ.get('MANO_MODEL_DIR', str(root/'assets/mano')))
    if not (models/'MANO_RIGHT.npz').exists():
        pytest.skip('User-licensed MANO files not installed')
    torch.manual_seed(4)
    low=MANOHand(models,components=12,samples=32)
    full=MANOHand(models,components=45,samples=32)
    q=torch.randn(2,15)*.25
    expanded=torch.cat([q,torch.zeros(2,33)],1).requires_grad_(True)
    a,aj=low(q);b,bj=full(expanded)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(aj,bj,atol=1e-6,rtol=1e-5)
    b.square().sum().backward()
    assert torch.isfinite(expanded.grad).all()
    assert expanded.grad[:,15:].abs().sum()>0
