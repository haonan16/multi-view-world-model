import numpy as np
from mano_recon.render import render as reference
from mano_recon.render_fast import render


def test_dense_renderer_preserves_visibility_depth_and_shading():
    rng=np.random.default_rng(17)
    vertices=rng.normal(0,.18,(360,3));vertices[:,2]+=1
    faces=np.arange(360).reshape(-1,3)
    K=np.array([[120.,0,64],[0,120.,48],[0,0,1.]])
    C=np.eye(4);C[:3,3]=[.04,-.02,.1]
    a,da=reference(vertices,faces,K,C,size=(96,128))
    b,db=render(vertices,faces,K,C,size=(96,128))
    assert np.array_equal(da>0,db>0)
    assert np.allclose(da,db,atol=2e-6)
    assert np.max(np.abs(a.astype(int)-b.astype(int)))<=1
