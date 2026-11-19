"""Model-output geometry tests use synthetic depths/masks, never real GT inputs."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artirecon.input_protocol import file_sha256, validate_manifest
from benchmarks import source_geometry as initializer
from benchmarks import source_geometry_v2 as initializer_v2


def inputs(frames=5, missing=(), ambiguous=(), corrupt_depth=False):
    height, width = 80, 100
    depth = np.ones((frames, 1, height, width), np.float32)
    rank = np.full(depth.shape, .9, np.float32)
    ts = np.array([7, 7.13, 7.6, 8.1, 8.23])[:frames]
    masks, tids, pids, scores = [], [], [], []
    for frame in range(frames):
        block = np.zeros((height, width), bool)
        block[34:46, 35:55] = True
        depth[frame, 0][block] = .9
        table = np.ones((height, width), bool)
        hand = np.zeros((height, width), bool)
        hand[34:46, 55:65] = True
        depth[frame, 0][hand] = .9
        if frame not in missing:
            masks.append(block); tids.append(frame); pids.append(0); scores.append(.95)
        if frame in ambiguous:
            other = np.zeros_like(block)
            other[34:46, 65:85] = True
            masks.append(other); tids.append(frame); pids.append(0); scores.append(.95)
        for prompt, mask in ((1, hand), (2, table)):
            masks.append(mask); tids.append(frame); pids.append(prompt); scores.append(.95)
        if corrupt_depth and frame == 2:
            # Most object pixels get a low-ranked systematic error. Rank50
            # retains a clear subset; all retains the wrong median surface.
            depth[frame, 0, 34:46, 35:47] = 1.2
            rank[frame, 0, 34:46, 35:47] = .1
    geometry = dict(depth_m=depth, confidence_rank=rank, valid=np.ones_like(depth, bool),
                    K=np.array([[[100, 0, 50], [0, 100, 40], [0, 0, 1.]]]),
                    c2w=np.eye(4)[None], timestamps=ts)
    candidates = dict(mask=np.packbits(np.array(masks).reshape(len(masks), -1), axis=1),
                      t=np.array(tids, int), p=np.array(pids, int), score=np.array(scores),
                      shape=np.array([height, width]), timestamps=ts,
                      prompts=np.array(["wooden block on patterned mat", "hand", "tabletop"]))
    return geometry, candidates


def config(mode="all"):
    return dict(initializer.default_config(), confidence_mode=mode, table_ransac_iterations=50)


def test_first_shape_is_shared_and_translation_only_is_explicit():
    geometry, masks = inputs()
    scene, report = initializer.initialize(geometry, masks, config())
    obj = scene["objects"][0]
    assert obj["geometry"]["type"] == "box"
    assert obj["observed"] == [True] * 5
    assert np.asarray(obj["geometry"]["half_extents_m"]).shape == (3,)
    assert scene["coordinate_frame"]["name"] == "source_camera"
    assert scene["metric_scale"]["status"] == "estimated"
    assert scene["motion_model"] == "shared_cuboid_fixed_orientation_translation_only"
    assert scene["orientation_estimated_over_time"] is False
    assert scene["initial_support_prior"]["used"] is True
    assert scene["initial_support_prior"]["per_frame_support_corrections_applied"] is False
    poses = np.array(obj["T_world_object"])
    np.testing.assert_allclose(poses, np.repeat(poses[:1], 5, axis=0), atol=1e-6)
    assert report["SAM3D_used"] is False
    assert report["supplied_calibration_used"] is False
    assert report["measured_depth_used"] is False
    assert report["simulator_accessed"] is False
    assert not any(item["kind"] == "grasp" for item in scene["contacts"])
    assert all(item["confidence"] is None for item in scene["contacts"])


def test_missing_mask_stays_unobserved_and_splits_contact_intervals():
    geometry, masks = inputs(missing=(2,))
    scene, report = initializer.initialize(geometry, masks, config())
    assert scene["objects"][0]["observed"] == [True, True, False, True, True]
    assert report["unobserved_frames"] == [2]
    assert report["evidence"][2]["pose_placeholder"] is True
    assert report["evidence"][2]["camera_z_median_m"] is None
    assert all(not (c["start_frame"] <= 2 <= c["end_frame"]) for c in scene["contacts"] if c["status"] != "unknown")
    assert any(c["status"] == "unknown" and c["start_frame"] == c["end_frame"] == 2 for c in scene["contacts"])


def test_first_frame_ambiguity_rejected_not_cherry_picked():
    geometry, masks = inputs(ambiguous=(0,))
    with pytest.raises(initializer.InitializationError, match="missing/ambiguous"):
        initializer.initialize(geometry, masks, config())


def test_confidence_ablation_keeps_identical_masks_and_exposes_wrong_depth():
    geometry, masks = inputs(corrupt_depth=True)
    all_scene, all_report = initializer.initialize(geometry, masks, config("all"))
    ranked_scene, ranked_report = initializer.initialize(geometry, masks, config("rank50"))
    assert [r["selected_row"] for r in all_report["association"]] == [r["selected_row"] for r in ranked_report["association"]]
    all_track = np.array(all_scene["objects"][0]["T_world_object"])[:, :3, 3]
    ranked_track = np.array(ranked_scene["objects"][0]["T_world_object"])[:, :3, 3]
    assert abs(all_track[2, 2] - all_track[0, 2]) > .2
    assert abs(ranked_track[2, 2] - ranked_track[0, 2]) < .001
    assert all_report["evidence"][2]["box_bottom_above_table_m"] < -.2
    assert ranked_report["evidence"][2]["points"] < all_report["evidence"][2]["points"]
    assert all_scene["initial_support_prior"]["per_frame_support_corrections_applied"] is False


@pytest.mark.parametrize("mutation, message", [
    (lambda d, m: d["c2w"][0, 0].__setitem__(3, 1), "identity camera"),
    (lambda d, m: m["timestamps"].__setitem__(2, 7.5), "timestamps"),
    (lambda d, m: d["confidence_rank"].__setitem__((0, 0, 0, 0), 2), "ranks"),
    (lambda d, m: m["t"].__setitem__(0, 50), "Malformed"),
])
def test_rejects_misaligned_or_non_source_data(mutation, message):
    geometry, masks = inputs()
    # Deliberately decouple the arrays; real independent NPZs do not share them.
    masks = deepcopy(masks)
    mutation(geometry, masks)
    with pytest.raises(initializer.InitializationError, match=message):
        initializer.initialize(geometry, masks, config())


def test_no_bottle_proxy_passes_as_block_initializer():
    geometry, masks = inputs()
    with pytest.raises(initializer.InitializationError, match="block-only"):
        initializer.initialize(geometry, masks, dict(config(), object_id="bottle"))


def test_table_ransac_ignores_distant_outliers():
    rng = np.random.default_rng(4)
    plane = np.c_[rng.uniform(-.5, .5, (500, 2)), rng.normal(1, .001, 500)]
    outliers = rng.uniform(-2, 2, (100, 3))
    normal, offset, report = initializer.fit_table(np.r_[plane, outliers], config())
    np.testing.assert_allclose(normal, [0, 0, -1], atol=.003)
    assert abs(offset - 1) < .003
    assert report["inlier_fraction"] > .8


def materialize(tmp_path):
    depth_dir, mask_dir = tmp_path / "depth", tmp_path / "masks"
    depth_dir.mkdir(); mask_dir.mkdir()
    geometry, masks = inputs()
    np.savez_compressed(depth_dir / "geometry.npz", **geometry)
    np.savez_compressed(mask_dir / "masks.npz", **masks)
    rgb = tmp_path / "rgb.dat"
    rgb.write_text("synthetic RGB bytes")
    root = {"id": "source_rgb", "kind": "rgb", "origin": "observed_rgb", "generated": False,
            "parent_ids": [], "path": str(rgb), "sha256": file_sha256(rgb), "time_range_s": [7, 8.23]}
    base = dict(schema_version=1, condition="rgb_only", task_mode="reconstruction", method_inputs=["source_rgb"],
                artifacts=[root], metric_scale={"status": "estimated", "artifact_ids": ["predicted_geometry"], "description": "Predicted metric depth"},
                robot_registration={"status": "unresolved", "artifact_ids": [], "description": "No robot registration"})
    depth_manifest = deepcopy(base)
    for name, kind in (("predicted_geometry", "depth"), ("predicted_cameras", "cameras")):
        depth_manifest["artifacts"].append({"id": name, "kind": kind, "origin": "model_prediction", "parent_ids": ["source_rgb"],
                                            "path": "geometry.npz", "sha256": file_sha256(depth_dir / "geometry.npz")})
        depth_manifest["method_inputs"].append(name)
    mask_manifest = deepcopy(base)
    mask_manifest["metric_scale"] = {"status": "unresolved", "artifact_ids": [], "description": "Masks only"}
    mask_manifest["artifacts"].append({"id": "sam3_masks", "kind": "masks", "origin": "model_prediction", "parent_ids": ["source_rgb"],
                                      "path": "masks.npz", "sha256": file_sha256(mask_dir / "masks.npz")})
    mask_manifest["method_inputs"].append("sam3_masks")
    for directory, manifest in ((depth_dir, depth_manifest), (mask_dir, mask_manifest)):
        (directory / "input-manifest.json").write_text(json.dumps(manifest))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config()))
    return dict(depth_bundle=depth_dir, mask_bundle=mask_dir, config_path=config_path, output=tmp_path / "output")


def test_materialized_scene_lineage_keeps_support_prior_and_registration_blocker(tmp_path):
    args = materialize(tmp_path)
    report = initializer.produce(**args)
    assert report["status"] == "initialized"
    manifest = json.loads((args["output"] / "input-manifest.json").read_text())
    validation = validate_manifest(manifest, args["output"], verify_files=True)
    assert validation["condition"] == "rgb_only"
    assert validation["manual_rgb_dependencies"] == []
    assert validation["oracle_dependencies"] == validation["calibration_dependencies"] == []
    assert validation["robot_registration"]["status"] == "unresolved"
    assert validation["execution_ready"] is False
    assert "source_block_initializer" in validation["method_artifact_ids"]
    assert (args["output"] / "scene-sequence.json").is_file()
    scene = json.loads((args["output"] / "scene-sequence.json").read_text())
    assert scene["initial_support_prior"]["used"] is True
    assert report["config"]["initial_table_support_prior"] is True


def test_rgb_mismatch_rejected_before_reconstruction(tmp_path):
    args = materialize(tmp_path)
    path = args["mask_bundle"] / "input-manifest.json"
    manifest = json.loads(path.read_text())
    other = tmp_path / "other-rgb.dat"
    other.write_text("different synthetic RGB bytes")
    manifest["artifacts"][0].update(path=str(other), sha256=file_sha256(other))
    path.write_text(json.dumps(manifest))
    with pytest.raises(initializer.InitializationError, match="identical isolated RGB"):
        initializer.produce(**args)
    assert not args["output"].exists()


def test_v2_fallback_is_explicit_and_uses_predicted_background_only():
    geometry, masks = inputs()
    keep = masks["p"] != 2
    for key in ("mask", "t", "p", "score"):
        masks[key] = masks[key][keep]
    with pytest.raises(initializer.InitializationError, match="Insufficient table"):
        initializer.initialize(geometry, masks, config())
    scene, report = initializer_v2.initialize(geometry, masks, initializer_v2.default_config())
    assert report["table"]["fallback_used"] is True
    assert report["table"]["semantic_table_certainty"] == "unestablished"
    assert report["table"]["selection_roi_xyxy"] == [0, 0, 100, 80]
    assert report["table"]["cached_plane_used"] is False
    assert "Dominant background" in scene["support"]["provenance"]["method"]
    np.testing.assert_allclose(scene["support"]["normal_world"], [0, 0, -1], atol=1e-6)
    assert abs(scene["support"]["offset_m"] - 1) < 1e-6
    assert scene["objects"][0]["observed"] == [True] * 5


def test_v2_does_not_change_semantic_table_result_when_supported():
    geometry, masks = inputs()
    first, first_report = initializer.initialize(geometry, masks, initializer.default_config())
    second, second_report = initializer_v2.initialize(geometry, masks, initializer_v2.default_config())
    np.testing.assert_array_equal(first["objects"][0]["T_world_object"], second["objects"][0]["T_world_object"])
    np.testing.assert_array_equal(first["objects"][0]["geometry"]["half_extents_m"], second["objects"][0]["geometry"]["half_extents_m"])
    assert second_report["table"]["fallback_used"] is False
    assert first_report["table"]["inliers"] == second_report["table"]["inliers"]


def test_v2_rank50_fallback_retains_same_ambiguous_state_rules():
    geometry, masks = inputs(missing=(2,))
    keep = masks["p"] != 2
    for key in ("mask", "t", "p", "score"):
        masks[key] = masks[key][keep]
    scene, report = initializer_v2.initialize(geometry, masks, dict(initializer_v2.default_config(), confidence_mode="rank50"))
    assert scene["objects"][0]["observed"][2] is False
    assert report["evidence"][2]["pose_placeholder"] is True
    assert report["table"]["fallback_used"] is True


def test_v2_materializes_independent_rgb_only_lineage(tmp_path):
    args = materialize(tmp_path)
    args["config_path"].write_text(json.dumps(initializer_v2.default_config()))
    report = initializer_v2.produce(**args)
    assert report["input_protocol"]["condition"] == "rgb_only"
    assert report["input_protocol"]["robot_registration"]["status"] == "unresolved"
