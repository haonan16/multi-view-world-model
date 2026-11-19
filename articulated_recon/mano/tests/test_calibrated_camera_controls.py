import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def modules():
    root = Path(__file__).resolve().parents[2] / "benchmarks"
    result = []
    for name in ["calibrated_camera_controls", "world_model_views"]:
        spec = importlib.util.spec_from_file_location(name, root / (name + ".py"))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        result.append(module)
    return result


def fixture(tmp_path, source_camera="source"):
    m, _ = modules()
    rgb = np.arange(360*640*3, dtype=np.uint8).reshape(1, 360, 640, 3)
    np.savez(tmp_path / "source.npz", rgb=rgb, timestamps=[0.])
    k = np.repeat(np.array([[[800., 0, 640], [0, 820, 360], [0, 0, 1]]]), 4, axis=0)
    c = np.repeat(np.eye(4)[None], 4, axis=0)
    c[:, :3, 3] = [[.2, .3, .4], [1.2, .3, .4], [-.8, .3, .4], [.2, -.4, .4]]
    np.savez(tmp_path / "rig.npz", K=k, c2w=c, resolution_wh=[1280, 720], camera_ids=["source", "a", "b", "c"])
    source = {"id": "source", "kind": "rgb", "origin": "observed_rgb", "generated": False,
        "parent_ids": [], "path": "source.npz", "sha256": m.file_sha256(tmp_path / "source.npz"), "time_range_s": [0., 0.]}
    if source_camera is not None:
        source["camera_id"] = source_camera
    manifest = {"schema_version": 1, "condition": "calibrated_rgb", "task_mode": "reconstruction",
        "method_inputs": ["source", "rig"], "artifacts": [source,
            {"id": "rig", "kind": "cameras", "origin": "known_calibration", "parent_ids": [],
             "path": "rig.npz", "sha256": m.file_sha256(tmp_path / "rig.npz")}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return m, dict(input_manifest=tmp_path / "manifest.json", source_id="source", rig_id="rig",
                   timestamps=np.linspace(0, 5.3, 17), output=tmp_path / "published"), rgb


def test_camera_crop_preserves_exact_source_gauge_and_original_intrinsic_convention():
    m, _ = modules()
    k = np.repeat(np.array([[[800., 0, 640], [0, 820, 360], [0, 0, 1]]]), 4, axis=0)
    c = np.repeat(np.eye(4)[None], 4, axis=0); c[:, :3, 3] = [[1, 2, 3], [2, 2, 3], [0, 2, 3], [1, 3, 3]]
    result_k, result_c, crop = m.crop_rig(k, c, [1280, 720], [360, 640])
    np.testing.assert_allclose(result_k[0], [[200, 0, 160], [0, 205, 80], [0, 0, 1]])
    np.testing.assert_allclose(result_c[0], np.eye(4))
    np.testing.assert_allclose(result_c[1, :3, 3], [1, 0, 0])
    assert crop == [0, 20, 640, 320]
    bad = c.copy(); bad[1, 0, 0] = -1
    with pytest.raises(ValueError, match="proper rigid"):
        m.crop_rig(k, bad, [1280, 720], [360, 640])


def test_prepare_declares_calibration_and_only_one_observed_frame(tmp_path):
    pytest.importorskip("cv2")
    m, args, _ = fixture(tmp_path)
    report = m.prepare(**args)
    with np.load(args["output"] / "condition.npz") as z:
        assert z["rgb"].shape == (17, 160, 320, 3) and not z["rgb"][1:].any()
        assert np.array_equal(z["source_frame_indices"], [0] + [-1]*16)
    assert report["source_camera_id"] == "source"
    assert "verified" in report["source_camera_identity_policy"]
    _, validated = m.validate_generation(args["output"], "first_frame_diagnostic", 0, 0)
    assert validated["condition"] == "calibrated_rgb" and validated["calibration_dependencies"]


def test_wrong_source_camera_id_fails_before_writing_output(tmp_path):
    pytest.importorskip("cv2")
    m, args, _ = fixture(tmp_path, "wrong_camera")
    with pytest.raises(ValueError, match="camera ID differs"):
        m.prepare(**args)
    assert not args["output"].exists()


def test_missing_camera_id_keeps_caller_assumption_explicit(tmp_path):
    pytest.importorskip("cv2")
    m, args, _ = fixture(tmp_path, None)
    assert "caller declares" in m.prepare(**args)["source_camera_identity_policy"]


def test_calibrated_condition_cannot_enter_default_rgb_path(tmp_path):
    pytest.importorskip("cv2")
    m, args, _ = fixture(tmp_path); m.prepare(**args)
    _, generator = modules()
    with pytest.raises(ValueError, match="explicit.*opt-in"):
        generator.generation_preparation(args["output"])
    _, report = generator.generation_preparation(args["output"], conditioning="first_frame_diagnostic",
        text_guidance=0, hist_guidance=0, calibrated_camera_diagnostic=True)
    assert report["condition"] == "calibrated_rgb"
    with pytest.raises(ValueError, match="first-frame"):
        generator.generation_preparation(args["output"], text_guidance=0, hist_guidance=0,
                                        calibrated_camera_diagnostic=True)


def test_even_hash_valid_future_pixels_are_rejected(tmp_path):
    pytest.importorskip("cv2")
    m, args, _ = fixture(tmp_path); m.prepare(**args); out = args["output"]
    with np.load(out / "condition.npz") as z:
        arrays = {key: z[key].copy() for key in z.files}
    arrays["rgb"][1, 0, 0] = 1
    np.savez(out / "condition.npz", **arrays)
    digest = m.file_sha256(out / "condition.npz")
    prep = json.loads((out / "preparation.json").read_text()); prep["condition_sha256"] = digest
    (out / "preparation.json").write_text(json.dumps(prep))
    manifest = json.loads((out / "input-manifest.json").read_text()); manifest["artifacts"][-1]["sha256"] = digest
    (out / "input-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="zero future"):
        m.validate_generation(out, "first_frame_diagnostic", 0, 0)


def test_crossing_preserves_exact_pixels_times_and_declared_camera_ancestry(tmp_path):
    pytest.importorskip("cv2")
    m, args, _ = fixture(tmp_path); m.prepare(**args); published = args["output"]
    est = tmp_path / "estimated"; est.mkdir()
    with np.load(published / "condition.npz") as z:
        arrays = {key: z[key].copy() for key in z.files}
    pk, pc = arrays["K"].copy(), arrays["c2w"].copy()
    arrays["K"][:, 0, 0] *= 1.2; arrays["c2w"][1:, :3, 3] *= .5
    arrays["rgb"][1:] = 123  # Existing full-source condition: must not leak into crosses.
    np.savez(est / "condition.npz", **arrays)
    manifest = {"schema_version": 1, "condition": "rgb_only", "task_mode": "reconstruction",
        "method_inputs": ["virtual_camera_requests"], "artifacts": [
            {"id": "rgb", "kind": "rgb", "origin": "observed_rgb", "generated": False, "parent_ids": [],
             "path": str(tmp_path / "source.npz"), "sha256": m.file_sha256(tmp_path / "source.npz"), "time_range_s": [0, 0]},
            {"id": "virtual_camera_requests", "kind": "cameras", "origin": "model_prediction", "parent_ids": ["rgb"],
             "path": "condition.npz", "sha256": m.file_sha256(est / "condition.npz")}]}
    (est / "input-manifest.json").write_text(json.dumps(manifest))
    (est / "preparation.json").write_text(json.dumps({"condition_sha256": m.file_sha256(est / "condition.npz")}))
    original_hash = m.file_sha256(est / "condition.npz")
    m.cross(estimated_condition=est, published_condition=published, output=tmp_path / "cross")
    for name, expected_k, expected_c in [("estimatedK_publishedC", arrays["K"], pc), ("publishedK_virtualC", pk, arrays["c2w"])]:
        folder = tmp_path / "cross" / name
        with np.load(folder / "condition.npz") as z:
            np.testing.assert_array_equal(z["K"], expected_k); np.testing.assert_array_equal(z["c2w"], expected_c)
            np.testing.assert_array_equal(z["timestamps"], arrays["timestamps"])
            np.testing.assert_array_equal(z["rgb"][0], arrays["rgb"][0]); assert not z["rgb"][1:].any()
        prep, report = m.validate_generation(folder, "first_frame_diagnostic", 0, 0)
        assert report["condition"] == "calibrated_rgb" and "full-source" in prep["camera_design_horizon"]
    assert m.file_sha256(est / "condition.npz") == original_hash
