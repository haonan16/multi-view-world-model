import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

path = Path(__file__).parents[2] / 'benchmarks'
sys.path.insert(0, str(path))
spec = importlib.util.spec_from_file_location('heldout_mano_audit', path/'heldout_mano_audit.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_relative_cameras_preserve_world_points_without_alignment():
    source = np.eye(4);source[:3, 3] = [1, 2, 3]
    view = np.eye(4);view[:3, 3] = [-1, 4, 2]
    view[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    points = np.array([[.1, .2, 1], [.3, -.2, 2.]])
    transformed = m.transform_points(points, m.source_to_view(source, view))
    assert np.allclose(m.transform_points(transformed, view), m.transform_points(points, source))
    assert np.allclose(m.transform_points(points, m.source_to_view(source, source)), points)


def test_rejects_improper_or_scaled_cameras():
    for scale in [-1, 2]:
        bad = np.eye(4);bad[0, 0] = scale
        with pytest.raises(ValueError, match='rigid'):
            m.source_to_view(np.eye(4), bad)


def test_object_interpolation_never_implies_observation_across_missing_state():
    times = np.arange(5, dtype=float)
    poses = np.repeat(np.eye(4)[None], 3, axis=0);poses[:, 0, 3] = [0, 2, 4]
    result, visible = m.interpolate_track(poses, np.array([0, 2, 4]), times[[0, 2, 4]],
                                          np.array([True, False, True]), times)
    assert visible.tolist() == [True, False, False, False, True]
    assert np.allclose(result[:, 0, 3], times)
    with pytest.raises(ValueError, match='contract'):
        m.interpolate_track(poses, np.array([0, 2, 4]), times[[0, 2, 4]]+.01,
                            np.ones(3, bool), times)


@pytest.mark.parametrize('mutation', ['scale', 'shear', 'nonfinite', 'homogeneous'])
def test_object_interpolation_rejects_implicit_pose_correction(mutation):
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    if mutation == 'scale': poses[0, :3, :3] *= 1.01
    elif mutation == 'shear': poses[0, 0, 1] = .01
    elif mutation == 'nonfinite': poses[0, 0, 3] = np.nan
    else: poses[0, 3, 0] = .01
    with pytest.raises(ValueError, match='rigid'):
        m.interpolate_track(poses, np.array([0, 1]), np.array([0., 1.]),
                            np.ones(2, bool), np.array([0., 1.]))
