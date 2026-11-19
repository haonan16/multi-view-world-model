import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_geometry_keyframes_retains_turning_point():
    m = module()
    times = np.arange(5, dtype=float)
    poses = np.tile(np.eye(4), (5, 1, 1))
    poses[:, 0, 3] = [0, 1, 2, 2, 2]
    poses[:, 1, 3] = [0, 0, 0, 1, 2]
    vertices = np.array([[-.1, -.1, -.1], [.1, .1, .1]])
    ids = m.geometry_keyframes(poses, times, vertices, 3)
    assert ids == [0, 2, 4]
    np.testing.assert_allclose(m.interpolate_poses(poses, times, ids), poses, atol=1e-12)


def module():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/keyframe_compression.py'
    spec = importlib.util.spec_from_file_location('keyframe_compression', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_two_endpoints_preserve_linear_motion_and_constant_angular_speed():
    m = module()
    times = np.linspace(0, 2, 9)
    poses = np.tile(np.eye(4), (9, 1, 1))
    poses[:, 0, 3] = .1 * times
    poses[:, :3, :3] = Rotation.from_euler('z', (times * .4)[:, None]).as_matrix()
    reconstructed = m.interpolate_poses(poses, times, [0, 8])
    np.testing.assert_allclose(reconstructed, poses, atol=1e-10)
    assert m.errors(poses, reconstructed)['translation_max_mm'] < 1e-8
    with pytest.raises(ValueError, match='endpoints'):
        m.interpolate_poses(poses, times, [1, 8])


def test_contact_intervals_preserve_release_boundaries():
    assert module().intervals([True, True, False, True, False]) == [(0, 1), (3, 3)]
