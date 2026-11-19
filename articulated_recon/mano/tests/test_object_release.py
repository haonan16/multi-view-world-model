import importlib.util
from pathlib import Path
import numpy as np


def test_release_requires_two_actual_depth_centres():
    p=Path(__file__).resolve().parents[1]/'scripts/build_sam_comparison_runtime.py'
    spec=importlib.util.spec_from_file_location('builder',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    ns={'np':np};exec(m.guard_missing_centres(''),ns);f=ns['stable_3d_centres']
    c=np.zeros(3)
    assert not f({0:{'c':None},1:{'c':c}},c)
    assert f({0:{'c':None},1:{'c':c},2:{'c':np.array([.01,0,0])}},c)
    assert not f({0:{'c':c},1:{'c':np.array([.04,0,0])}},c)
