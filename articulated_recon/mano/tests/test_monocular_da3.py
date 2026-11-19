import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.monocular_da3 import metric_depth


def test_focal_conversion_uses_actual_network_grid_not_native_focal():
    k=np.array([[600.,0,320],[0,600,180],[0,0,1]])
    depth,scale=metric_depth(np.ones((180,320)),k,(360,640))
    assert scale==pytest.approx(1.)
    np.testing.assert_array_equal(depth,1.)
    depth,scale=metric_depth(np.ones((360,640)),k,(360,640))
    assert scale==pytest.approx(2.)
    np.testing.assert_array_equal(depth,2.)


def test_nonisotropic_resize_scales_each_focal_separately():
    k=np.array([[600.,0,320],[0,300,180],[0,0,1]])
    _,scale=metric_depth(np.ones((180,320)),k,(360,640))
    assert scale==pytest.approx(.75)


def test_invalid_focal_is_rejected():
    with pytest.raises(ValueError):metric_depth(np.ones((10,10)),np.zeros((3,3)),(10,10))
