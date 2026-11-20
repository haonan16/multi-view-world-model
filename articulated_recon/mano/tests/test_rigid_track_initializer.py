import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[2] / 'benchmarks/rigid_track_initializer.py'
spec = importlib.util.spec_from_file_location('rigid_track_initializer', PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def example():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 3)) * [.04, .025, .015]
    a = .7
    transform = np.eye(4)
    transform[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    transform[:3, 3] = [.07, -.03, .05]
    y = x @ transform[:3, :3].T + transform[:3, 3]
    return x, y, transform


def test_correspondence_outliers_are_rejected_without_changing_shape():
    x, y, expected = example()
    y[:30] = np.random.default_rng(5).normal(size=(30, 3))
    pose, inliers, report = m.robust_rigid(x, y)
    assert np.allclose(pose, expected, atol=1e-10)
    assert inliers.sum() == 70 and not inliers[:30].any()
    assert report['residual_m'] < 1e-10
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1)


def test_hidden_frame_stays_unobserved_and_later_frame_recovers_directly():
    x, y, expected = example()
    points = np.stack([x, np.full_like(x, np.nan), y])
    result = m.fit_object_track(points, np.ones((3, len(x)), bool))
    assert result['observed'].tolist() == [True, False, True]
    assert np.allclose(result['relative_poses'][1], np.eye(4))
    assert np.allclose(result['relative_poses'][2], expected)
    assert not result['inliers'][1].any()


def test_collinear_tracks_do_not_create_fake_orientation():
    x = np.c_[np.linspace(0, .1, 30), np.zeros((30, 2))]
    pose, _, report = m.robust_rigid(x, x + .2)
    assert pose is None and report['status'] == 'rotation_unobservable'


def test_visibility_is_an_explicit_gate_not_implicit_probability_calibration():
    x, y, _ = example()
    with pytest.raises(ValueError, match='boolean'):
        m.fit_object_track(np.stack([x, y]), np.ones((2, len(x)), float))
    visibility = np.ones((2, len(x)), bool)
    visibility[1, 5:] = False
    result = m.fit_object_track(np.stack([x, y]), visibility)
    assert result['observed'].tolist() == [True, False]


def test_rigidity_does_not_remove_shared_metric_bias():
    x, _, _ = example()
    shift = np.array([.1, 0, 0])
    # Both views inherit a factor-two metric error. A perfect rigid residual
    # still returns twice the true displacement; this is deliberately NOT fixed.
    result = m.fit_object_track(np.stack([x * 2, (x + shift) * 2]), np.ones((2, len(x)), bool))
    assert result['observed'].all()
    assert np.allclose(result['relative_poses'][1, :3, 3], shift * 2)
    assert result['frames'][1]['residual_m'] < 1e-10


def test_shape_and_initial_placement_are_shared_and_contact_stays_unknown():
    x, y, _ = example()
    motion = m.fit_object_track(np.stack([x, y]), np.ones((2, len(x)), bool))
    first = np.eye(4)
    first[:3, :3] = np.diag([2., 3., 4.])
    first[:3, 3] = [.2, .1, .7]
    scene = m.initialize_scene(vertices_m=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]],
        first_pose=first, motion=motion, timestamps=[0, .1],
        coordinate_frame=dict(name='predicted_world', handedness='right', up_axis='unknown', pose_convention='T_world_local'),
        metric_scale=dict(status='estimated', provenance=dict(method='model_estimate')),
        provenance=dict(artifact_ids=['source_predictions']))
    obj = scene['objects'][0]
    assert scene['contacts'] == [] and scene['support'] is None
    assert np.allclose(obj['geometry']['local_linear'], np.diag([2, 3, 4]))
    assert np.allclose(np.asarray(obj['T_world_object'])[0, :3, 3], first[:3, 3])
    assert obj['provenance']['hidden_surface_status'] == 'model_completion_unverified'
    assert len(obj['geometry']['vertices_m']) == 3


def test_nonrigid_correspondences_do_not_get_explained_by_mesh_inflation():
    x, _, _ = example()
    pose, _, report = m.robust_rigid(x, x * 4)
    assert pose is None and report['status'] == 'no_rigid_consensus'
