"""Constraint changes feasible scene geometry, not planner commands or targets."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks import source_joint_fit as joint
from benchmarks import source_nonpenetrating_fit as constrained
from test_source_joint_fit import synthetic_scene_and_observations


def test_global_nonpenetration_changes_all_infeasible_frames_without_postfit_clamp():
    scene, observations = synthetic_scene_and_observations()
    before = deepcopy(scene)
    fitted, report = constrained.fit_shared_scene(scene, observations, constrained.default_config())
    assert scene == before
    assert report["initial_projection_frames"] == [1, 2, 3, 4]
    assert min(report["after"]["bottom_height_m"]) >= -1e-12
    assert report["after"]["maximum_penetration_m"] < 1e-12
    assert report["endpoint_specific_operation"] is False
    assert report["simulator_accessed"] is False
    assert fitted["joint_fit"]["enforced_in_optimization"] is True
    assert fitted["joint_fit"]["postfit_pose_changes"] is False
    assert fitted["objects"][0]["geometry"] != before["objects"][0]["geometry"]
    assert report["final_cost"] == pytest.approx(sum(report["after"]["normalized_soft_l1_cost"].values()))


def test_residual_weights_match_the_original_unconstrained_fit():
    scene, observations = synthetic_scene_and_observations()
    fitted, report = joint.fit_shared_scene(scene, observations, joint.default_config())
    obj = fitted["objects"][0]
    _, metrics = constrained.components(np.asarray(obj["geometry"]["half_extents_m"]), np.asarray(obj["T_world_object"])[:, :3, 3],
        scene=scene, observations=observations, config=joint.default_config())
    assert sum(metrics["normalized_soft_l1_cost"].values()) == pytest.approx(report["final_cost"])
    assert metrics["unoccluded_bbox_absolute_error_mean_px"] == pytest.approx(report["after"]["unoccluded_bbox_absolute_error_mean_px"])
    assert metrics["rgb_support_absolute_residual_mean_m"] == pytest.approx(report["after"]["rgb_support_absolute_residual_mean_m"])


def test_constraint_does_not_upgrade_unobserved_frames():
    scene, observations = synthetic_scene_and_observations()
    observations["evidence"][2]["observed"] = False
    fitted, _ = constrained.fit_shared_scene(scene, observations, constrained.default_config())
    assert fitted["objects"][0]["observed"][2] is False


def test_constraint_depends_on_declared_estimated_plane_not_a_hardcoded_world_axis():
    scene, observations = synthetic_scene_and_observations()
    # Rigidly change scene coordinates and camera calibration-compatible rays.
    # The API's table frame itself must be a proper basis for arbitrary normals.
    normal = np.array([.3, -.5, -.812403840463596])
    normal /= np.linalg.norm(normal)
    basis = constrained.table_basis(normal)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(basis[:, 2], normal)
    assert np.linalg.det(basis) == pytest.approx(1.)
