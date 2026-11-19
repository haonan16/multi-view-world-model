"""Static-source self-calibration tests contain no measured real depth or GT."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks import static_depth_calibration as calibration
from artirecon.input_protocol import file_sha256, validate_manifest


def inputs():
    frames, height, width = 5, 80, 100
    drift = np.array([1., 1.1, .9, 1.05, .95])
    depth = np.ones((frames, 1, height, width), np.float32) * drift[:, None, None, None]
    masks, tids, pids = [], [], []
    for frame in range(frames):
        block = np.zeros((height, width), bool); block[34:46, 35:55] = True
        hand = np.zeros_like(block); hand[34:46, 70:80] = True
        table = np.ones_like(block)
        depth[frame, 0][block] *= .9
        for prompt, mask in enumerate([block, hand, table]):
            masks.append(mask); tids.append(frame); pids.append(prompt)
    times = np.arange(frames) * .1
    geometry = dict(depth_m=depth, valid=np.ones_like(depth, bool), confidence_rank=np.ones_like(depth) * .8,
                    K=np.array([[[100., 0, 50], [0, 100., 40], [0, 0, 1]]]), c2w=np.eye(4)[None], timestamps=times)
    mask_data = dict(mask=np.packbits(np.array(masks).reshape(len(masks), -1), axis=1), t=np.array(tids), p=np.array(pids),
                     score=np.ones(len(masks)) * .9, shape=np.array([height, width]), timestamps=times,
                     prompts=np.array(["wooden block on patterned mat", "hand", "tabletop"]))
    rgb = np.full((frames, height, width, 3), 100, np.uint8)
    return geometry, mask_data, rgb, drift


def config():
    return dict(calibration.default_config(), minimum_region_pixels=1000, fit_points_per_frame=300, heldout_points_per_frame=300)


def test_recovers_temporal_scale_in_frame_zero_gauge_without_mutating_raw_depth():
    geometry, masks, rgb, drift = inputs()
    original = geometry["depth_m"].copy()
    corrected, scales, report = calibration.calibrate(geometry, masks, rgb, config())
    assert report["status"] == "accepted"
    assert scales[0] == 1.
    np.testing.assert_allclose(scales, 1 / drift, rtol=1e-6)
    np.testing.assert_array_equal(geometry["depth_m"], original)
    np.testing.assert_allclose(corrected["depth_m"][:, 0, 0, 0], 1., atol=1e-6)
    np.testing.assert_array_equal(corrected["K"], geometry["K"])
    assert report["object_pose_used"] is False
    assert report["simulator_accessed"] is False
    assert max(row["corrected_heldout_absolute_median_m"] for row in report["rows"]) < 1e-6
    assert any(row["raw_heldout_absolute_median_m"] > .05 for row in report["rows"])


def test_out_of_range_scale_rejects_instead_of_clipping():
    geometry, masks, rgb, _ = inputs()
    geometry["depth_m"][2] *= 3
    _, scales, report = calibration.calibrate(geometry, masks, rgb, config())
    assert report["status"] == "rejected"
    assert scales[2] < config()["minimum_scale"]
    assert {"frame": 2, "reason": "scale outside fixed bounds"} in report["rejected_frames"]


def test_static_rgb_contradiction_rejects_the_capture_assumption():
    geometry, masks, rgb, _ = inputs()
    rgb[3] = 200
    _, _, report = calibration.calibrate(geometry, masks, rgb, config())
    assert report["status"] == "rejected"
    assert {"frame": 3, "reason": "source RGB static-background check failed"} in report["rejected_frames"]


def test_scale_only_cannot_hide_nonplanar_heldout_depth_error():
    geometry, masks, rgb, _ = inputs()
    rng = np.random.default_rng(18)
    geometry["depth_m"][2] += rng.normal(0, .05, geometry["depth_m"][2].shape)
    _, _, report = calibration.calibrate(geometry, masks, rgb, config())
    assert report["status"] == "rejected"
    assert {"frame": 2, "reason": "held-out background residual above criterion"} in report["rejected_frames"]


def test_compact_producer_keeps_raw_depth_and_rgb_only_lineage(tmp_path):
    geometry, masks, rgb, _ = inputs()
    source_dir, mask_dir = tmp_path / "source", tmp_path / "masks"
    source_dir.mkdir(); mask_dir.mkdir()
    np.savez_compressed(source_dir / "geometry.npz", **geometry)
    np.savez_compressed(source_dir / "rgb.npz", rgb=rgb, timestamps=geometry["timestamps"])
    np.savez_compressed(mask_dir / "masks.npz", **masks)
    source = dict(id="source_rgb", kind="rgb", origin="observed_rgb", generated=False,
                  parent_ids=[], path=str(source_dir / "rgb.npz"), sha256=file_sha256(source_dir / "rgb.npz"), time_range_s=[0., .4])
    records = [source]
    for aid, kind in (("predicted_geometry", "depth"), ("predicted_cameras", "cameras")):
        records.append(dict(id=aid, kind=kind, origin="model_prediction", parent_ids=["source_rgb"],
                            path="geometry.npz", sha256=file_sha256(source_dir / "geometry.npz")))
    manifest = dict(schema_version=1, condition="rgb_only", task_mode="reconstruction",
                    method_inputs=["source_rgb", "predicted_geometry", "predicted_cameras"], artifacts=records,
                    metric_scale=dict(status="estimated", artifact_ids=["predicted_geometry"], description="Synthetic model scale"),
                    robot_registration=dict(status="unresolved", artifact_ids=[], description="Not registered"))
    (source_dir / "input-manifest.json").write_text(json.dumps(manifest))
    mask_manifest = dict(schema_version=1, condition="rgb_only", task_mode="reconstruction",
                         method_inputs=["source_rgb", "sam3_masks"],
                         artifacts=[source,dict(id="sam3_masks", kind="masks", origin="model_prediction", parent_ids=["source_rgb"],
                            path="masks.npz", sha256=file_sha256(mask_dir / "masks.npz"))])
    (mask_dir / "input-manifest.json").write_text(json.dumps(mask_manifest))
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config()))
    raw_hash = file_sha256(source_dir / "geometry.npz")
    output = tmp_path / "calibrated"
    report = calibration.produce(depth_bundle=source_dir, mask_bundle=mask_dir, config_path=config_path,
                                 joint_config_paths=[], output=output)
    assert report["status"] == "accepted"
    assert file_sha256(source_dir / "geometry.npz") == raw_hash
    assert (output / "depth-calibration.npz").stat().st_size < 10000
    new_manifest = json.loads((output / "initializer/input-manifest.json").read_text())
    checked = validate_manifest(new_manifest, output / "initializer", verify_files=True)
    assert checked["oracle_dependencies"] == checked["calibration_dependencies"] == []
    assert checked["robot_registration"]["status"] == "unresolved"
    assert "raw_predicted_geometry" in checked["method_artifact_ids"]
    restored = calibration.materialize_recipe(source_dir / "geometry.npz", output / "depth-calibration.npz")
    expected, _, _ = calibration.calibrate(geometry, masks, rgb, config())
    np.testing.assert_array_equal(restored["depth_m"], expected["depth_m"])
    initializer = json.loads((output / "initializer/scene-sequence.json").read_text())
    obj = initializer["objects"][0]
    pose = np.asarray(obj["T_world_object"])[0]
    normal = np.asarray(initializer["support"]["normal_world"])
    radius = np.abs(normal @ pose[:3, :3]) @ np.asarray(obj["geometry"]["half_extents_m"])
    bottom = pose[:3, 3] @ normal + initializer["support"]["offset_m"] - radius
    assert abs(bottom) < 1e-8
    np.testing.assert_allclose(normal, report["reference_normal"])
    assert initializer["support"]["offset_m"] == report["reference_offset_m"]
    (source_dir / "geometry.npz").write_bytes(b"tampered raw prediction")
    with pytest.raises(ValueError, match="Raw depth changed"):
        calibration.materialize_recipe(source_dir / "geometry.npz", output / "depth-calibration.npz")


def test_recipe_does_not_override_reference_metric_gauge(tmp_path):
    geometry, _, _, _ = inputs()
    raw = tmp_path / "raw.npz"; np.savez_compressed(raw, **geometry)
    recipe = tmp_path / "recipe.npz"
    np.savez_compressed(recipe, recipe_kind="multiplicative_depth_scale_v1", raw_geometry_sha256=file_sha256(raw),
                        scale=np.full(len(geometry["timestamps"]), 1.1), timestamps=geometry["timestamps"])
    with pytest.raises(ValueError, match="frame-zero gauge"):
        calibration.materialize_recipe(raw, recipe)


def test_calibration_preserves_absent_confidence_marker(tmp_path):
    geometry, masks, rgb, _ = inputs()
    geometry["confidence_available"] = np.asarray(False)
    corrected, scales, _ = calibration.calibrate(geometry, masks, rgb, config())
    assert not corrected["confidence_available"].item()
    raw = tmp_path / "raw.npz"; np.savez_compressed(raw, **geometry)
    recipe = tmp_path / "recipe.npz"
    np.savez_compressed(recipe, recipe_kind="multiplicative_depth_scale_v1", raw_geometry_sha256=file_sha256(raw),
                        scale=scales, timestamps=geometry["timestamps"])
    assert not calibration.materialize_recipe(raw, recipe)["confidence_available"].item()
