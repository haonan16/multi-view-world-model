"""Small numerical tests: confidence coverage and uncertain joint constraints."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artirecon.scene_sequence import scene_from_arrays, object_from_arrays
from benchmarks import source_joint_fit as joint


def test_rgb_support_does_not_read_current_depth_and_missing_hand_is_unknown():
    config = joint.default_config()
    times = np.arange(6) * .1
    centers = np.zeros((6, 2))
    gaps = np.array([30, 30, 0, 30, np.nan, 30])
    flags = joint.rgb_support_hypotheses(centers, gaps, times, np.ones(6, bool), config)
    assert flags.tolist() == [True, True, False, True, False, True]
    # No poses/depth enter this API; mask movement alone invalidates support.
    centers[1] += 10
    changed = joint.rgb_support_hypotheses(centers, gaps, times, np.ones(6, bool), config)
    assert not changed[0] and not changed[1]


def synthetic_scene_and_observations():
    frames = 5
    timestamps = np.arange(frames) * .1
    rotation = np.diag([1., -1., -1.])
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    poses[:, :3, :3] = rotation
    poses[:, 2, 3] = np.linspace(.96, 1., frames)
    provenance = {"method": "synthetic model estimate", "artifact_ids": ["depth"], "confidence_status": "uncalibrated"}
    scene = scene_from_arrays(timestamps=timestamps,
        coordinate_frame={"name": "source_camera", "handedness": "right", "up_axis": "unknown", "pose_convention": "T_world_local"},
        metric_scale={"status": "estimated", "provenance": provenance}, actors=[],
        objects=[object_from_arrays(object_id="block", poses=poses, observed=[True] * frames, provenance=provenance,
                                   primitive={"type": "box", "half_extents_m": [.025, .025, .04]})],
        support={"id": "table", "normal_world": [0, 0, -1], "offset_m": 1., "confidence": None, "provenance": provenance},
        contacts=[], uncertainty={"status": "unestimated"}, task={"description": "synthetic block", "target": None})
    scene["initial_support_prior"] = {"used": True, "per_frame_support_corrections_applied": False}
    k = np.array([[500., 0, 50], [0, 500., 40], [0, 0, 1]])
    corners = np.array(list(product((-.025, .025), repeat=3)))
    projected = (corners + [0, 0, .975]) @ k.T
    uv = projected[:, :2] / projected[:, 2:3]
    bbox = np.r_[uv.min(0), uv.max(0)]
    xy = np.array(list(product(np.linspace(-.02, .02, 6), repeat=2)))
    points = [np.c_[xy, np.repeat(.95 + i * .006, len(xy))] for i in range(frames)]
    observations = dict(points=points, weights=[np.ones(len(p)) for p in points], bounds=np.repeat(bbox[None], frames, axis=0),
                        occluded=np.zeros(frames, bool), mask_visible=np.ones(frames, bool), support=np.ones(frames, bool),
                        K=k, timestamps=timestamps, evidence=[{"observed": True} for _ in range(frames)])
    return scene, observations


def test_joint_fit_changes_shared_geometry_and_reduces_support_inconsistency_without_clamp():
    scene, observations = synthetic_scene_and_observations()
    raw = deepcopy(scene)
    fitted, report = joint.fit_shared_scene(scene, observations, joint.default_config())
    assert scene == raw
    assert report["after"]["maximum_penetration_m"] < report["before"]["maximum_penetration_m"] * .7
    assert report["after"]["rgb_support_absolute_residual_mean_m"] < report["before"]["rgb_support_absolute_residual_mean_m"]
    assert fitted["objects"][0]["geometry"] != scene["objects"][0]["geometry"]
    assert fitted["joint_fit"]["endpoint_clamped"] is False
    assert fitted["joint_fit"]["support_is_soft_hypothesis"] is True
    assert abs(report["after"]["endpoint_bottom_m"]) > 1e-7  # Soft residual, not exact table snapping.
    assert report["support_evidence_uses_3d_height"] is False
    assert report["GT_accuracy_measured"] is False
    assert report["simulator_accessed"] is False


def low_confidence_inputs():
    frames, height, width = 3, 40, 60
    depth = np.ones((frames, 1, height, width))
    rank = np.full_like(depth, .9)
    packed, tids, pids = [], [], []
    for frame in range(frames):
        block = np.zeros((height, width), bool); block[12:22, 15:30] = True
        hand = np.zeros_like(block); hand[12:22, 45:53] = True
        table = np.ones_like(block)
        depth[frame, 0][block] = .9
        rank[frame, 0][block] = .01
        for prompt, mask in enumerate([block, hand, table]):
            packed.append(mask); tids.append(frame); pids.append(prompt)
    times = np.arange(frames) * .1
    geometry = dict(depth_m=depth, confidence_rank=rank, valid=np.ones_like(depth, bool), timestamps=times,
                    K=np.array([[[100., 0, 30], [0, 100., 20], [0, 0, 1]]]), c2w=np.eye(4)[None])
    masks = dict(mask=np.packbits(np.array(packed).reshape(len(packed), -1), axis=1), t=np.array(tids), p=np.array(pids),
                 score=np.ones(len(packed)) * .9, shape=np.array([height, width]), timestamps=times,
                 prompts=np.array(["wooden block on patterned mat", "hand", "tabletop"]))
    return geometry, masks, dict(timestamps=times.tolist())


def test_per_object_rank_keeps_weak_critical_evidence_without_claiming_high_certainty():
    geometry, masks, scene = low_confidence_inputs()
    evidence = joint.prepare_evidence(geometry, masks, scene, joint.default_config())
    assert all(row["observed"] and row["weak_depth_evidence"] for row in evidence["evidence"])
    assert all(row["mean_global_rank"] < .5 for row in evidence["evidence"])
    assert all(len(points) > 0 for points in evidence["points"])


def test_soft_weighting_preserves_points_but_downweights_them():
    geometry, masks, scene = low_confidence_inputs()
    config = dict(joint.default_config(), point_policy="soft_rank")
    evidence = joint.prepare_evidence(geometry, masks, scene, config)
    assert all(row["observed"] for row in evidence["evidence"])
    assert all(np.all((weights >= .1) & (weights < .2)) for weights in evidence["weights"])
    assert all(row["confidence_is_calibrated_probability"] is False for row in evidence["evidence"])


def test_unknown_confidence_requires_neutral_all_policy_and_reports_unknown():
    geometry, masks, scene = low_confidence_inputs()
    geometry["confidence_available"] = np.asarray(False)
    geometry["confidence_rank"][:] = 1.
    for policy in ("object_rank50", "soft_rank"):
        with pytest.raises(ValueError, match="neutral weights are not confidence"):
            joint.prepare_evidence(geometry, masks, scene, dict(joint.default_config(), point_policy=policy))
    evidence = joint.prepare_evidence(geometry, masks, scene, dict(joint.default_config(), point_policy="all"))
    assert all(row["observed"] and row["confidence_available"] is False for row in evidence["evidence"])
    assert all(row["mean_global_rank"] is None and row["weak_depth_evidence"] is None for row in evidence["evidence"])
    assert all(np.all(weight == 1.) for weight in evidence["weights"])
