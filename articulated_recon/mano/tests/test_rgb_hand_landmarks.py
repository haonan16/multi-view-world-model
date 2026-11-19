import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def module():
    p=Path(__file__).resolve().parents[2]/'benchmarks/rgb_hand_landmarks.py'
    s=importlib.util.spec_from_file_location('rgb_hand_landmarks',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_crop_inversion_preserves_world_units_and_centres_wrist():
    m=module()
    lm=[SimpleNamespace(x=i/100,y=i/50,z=0) for i in range(21)]
    world=[SimpleNamespace(x=1+i*.01,y=2+i*.02,z=3+i*.03) for i in range(21)]
    xy,xyz=m.native_landmarks(lm,world,np.array([[0.,.5,10.],[-.5,0.,20.]]),(100,100))
    np.testing.assert_allclose(xy[1],[11.,19.5])
    np.testing.assert_allclose(xyz[0],0)
    np.testing.assert_allclose(xyz[1],[.02,-.01,.03],atol=1e-6)


def test_duplicate_removal_does_not_treat_handedness_score_as_joint_confidence():
    m=module();native=dict(xy=np.zeros((21,2)),handedness_score=.5)
    duplicate=dict(xy=np.ones((21,2)),handedness_score=.99)
    other=dict(xy=np.ones((21,2))*100,handedness_score=.9)
    assert m.deduplicate([native,duplicate,other])==[native,other]
