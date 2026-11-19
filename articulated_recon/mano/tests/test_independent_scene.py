import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artirecon.input_protocol import file_sha256
from benchmarks import independent_scene
from benchmarks.freeze_submission import freeze_submission
from test_freeze_submission import case, refresh_metadata, write_json


def reference():
    vertices = np.array([[-.1, -.1, -.1], [.1, -.1, -.1], [0, .1, -.1], [0, 0, .1]])
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[:, :3, :3] = np.stack([Rotation.from_euler("z", angle, degrees=True).as_matrix() for angle in (0, 30)]) @ np.diag([1., 2., .5])
    poses[:, :3, 3] = [[.2, .1, -.02], [.3, .1, .1]]
    return {"timestamps": [0, 1], "table": {"normal": [0, 0, 1], "offset": 0},
            "objects": [{"vertices": vertices.tolist(), "faces": [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]],
                         "poses": poses.tolist()}]}


def transform(angle, shift):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("y", angle, degrees=True).as_matrix()
    pose[:3, 3] = shift
    return pose


def test_camera_transform_preserves_every_surface_and_table_distance():
    raw = reference()
    C, T = transform(25, [.1, .4, .2]), transform(-40, [.5, -.3, .1])
    result = independent_scene.transform_reference(raw, C, T)
    M = T @ np.linalg.inv(C)
    original_poses = np.asarray(raw["objects"][0]["poses"])
    vertices = np.asarray(raw["objects"][0]["vertices"])
    world = np.einsum("tij,pj->tpi", original_poses[:, :3, :3], vertices) + original_poses[:, None, :3, 3]
    expected = world @ M[:3, :3].T + M[:3, 3]
    actual = np.einsum("tij,pj->tpi", result["poses"][:, :3, :3], result["vertices"]) + result["poses"][:, None, :3, 3]
    assert np.allclose(actual, expected, atol=1e-12)
    assert np.allclose(actual @ result["support_normal"] + result["support_offset"], world[..., 2], atol=1e-12)
    assert result["support_minimum_m"][0] < 0  # Original penetration retained; no support snap.


def test_reference_shape_change_cannot_be_hidden_by_rigid_canonicalization():
    raw = reference()
    raw["objects"][0]["poses"][1][0][0] *= 1.1
    with pytest.raises(ValueError, match="shape varies"):
        independent_scene.transform_reference(raw, np.eye(4), np.eye(4))


def test_nonrigid_published_camera_rejected():
    C = np.eye(4)
    C[0, 0] = 2
    with pytest.raises(ValueError, match="rigid SE"):
        independent_scene.transform_reference(reference(), C, np.eye(4))


def test_evaluator_inputs_are_not_opened_before_frozen_submission_check(monkeypatch, tmp_path):
    def blocked(_):
        raise ValueError("plan not frozen")
    monkeypatch.setattr(independent_scene, "verify_submission", blocked)
    monkeypatch.setattr(independent_scene, "read", lambda _: pytest.fail("Evaluator or method file opened before freeze check"))
    with pytest.raises(ValueError, match="plan not frozen"):
        independent_scene.prepare(submission="unfrozen", raw_reference="secret", source_cameras="secret",
            source_view=0, deployment="deployment", output=tmp_path / "out", source_camera_id="source")


def test_postfreeze_constructor_attaches_reference_without_changing_plan(case):
    registration_path = case["output"].parent / "registration.dat"
    write_json(registration_path, {"schema": "deployment-registration-v1", "source_frame": "source_camera",
        "target_frame": "robot_base", "T_robot_scene": np.eye(4).tolist(), "deployment_type": "virtual",
        "procedure": "Choose one common pose from the RGB-only initialization before evaluation",
        "fixed_across_methods": True, "derived_from_evaluation_geometry": False,
        "source_camera_id": "source0", "source_coordinate_convention": "opencv"})
    manifest = json.loads(case["input_manifest"].read_text())
    next(a for a in manifest["artifacts"] if a["id"] == "registration")["sha256"] = file_sha256(registration_path)
    write_json(case["input_manifest"], manifest)
    scene = json.loads(case["scene"].read_text())
    scene["coordinate_frame"]["name"] = "source_camera"
    write_json(case["scene"], scene)
    refresh_metadata(case)
    freeze_submission(**case)
    plan_sha, seal_sha = file_sha256(case["plan"]), file_sha256(case["output"])
    raw = write_json(case["output"].parent / "raw-reference.json", reference())
    cameras = case["output"].parent / "published-camera.npz"
    np.savez_compressed(cameras, c2w=np.eye(4))
    out = case["output"].parent / "evaluation"
    report = independent_scene.prepare(submission=case["output"], raw_reference=raw, source_cameras=cameras,
        source_view=0, deployment=registration_path, output=out, source_camera_id="source0")
    assert not report["reference_is_exact_gt"] and not report["fitted_alignment"] and not report["support_snap"]
    assert report["initial_support_gap_m"] < 0
    assert (out / "evaluation-attachment.json").exists()
    assert file_sha256(case["plan"]) == plan_sha and file_sha256(case["output"]) == seal_sha
    config = json.loads((out / "actor-config.json").read_text())
    assert config["initial_pose_must_not_be_snapped"] and config["support_must_not_be_replaced_by_z0"]
    assert np.array_equal(config["object"]["initial_pose"], np.load(out / "fixture.npz")["initial_pose"])
