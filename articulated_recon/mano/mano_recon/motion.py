"""Interpolate rigid motion while preserving fitted affine mesh stretch."""
import numpy as np
from scipy.linalg import polar
from scipy.spatial.transform import Rotation, Slerp

def interpolate(poses,ids,T):
    rotations=[];stretch=[]
    for A in poses[:,:3,:3]:
        R,S=polar(A)
        if np.linalg.det(R)<0:raise ValueError('Reflected object transform; inspect before rendering')
        rotations.append(R);stretch.append(S)
    R=Slerp(ids,Rotation.from_matrix(rotations))(np.arange(T)).as_matrix()
    S=np.stack([np.interp(np.arange(T),ids,np.array(stretch)[:,i,j]) for i in range(3) for j in range(3)],axis=-1).reshape(T,3,3)
    p=np.tile(np.eye(4),(T,1,1));p[:,:3,:3]=R@S
    for i in range(3):p[:,i,3]=np.interp(np.arange(T),ids,poses[:,i,3])
    assert np.allclose(p[ids],poses,atol=2e-6), 'Playback changed fitted keyframes'
    return p
