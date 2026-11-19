"""Shared actor/object states preserve metric geometry and evidence boundaries."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


MODULE = Path(__file__).resolve().parents[2] / "artirecon/scene_sequence.py"
SPEC = importlib.util.spec_from_file_location("scene_sequence", MODULE)
scene_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scene_module)
SceneValidationError = scene_module.SceneValidationError


def poses(frames):
    result = np.repeat(np.eye(4)[None], frames, axis=0)
    theta = np.linspace(0, 0.3, frames)
    result[:, 0, 0] = result[:, 1, 1] = np.cos(theta)
    result[:, 0, 1] = -np.sin(theta)
    result[:, 1, 0] = np.sin(theta)
    result[:, 0, 3] = np.linspace(0, 0.1, frames)
    return result


def make_scene(frames=11):
    provenance = {"method": "synthetic test", "artifact_ids": ["test-rgb"]}
    return scene_module.scene_from_arrays(
        timestamps=np.arange(frames, dtype=float),
        coordinate_frame={"name": "test-world", "handedness": "right", "up_axis": "+z",
                          "pose_convention": "T_world_local"},
        metric_scale={"status": "estimated", "provenance": provenance},
        actors=[scene_module.actor_from_arrays(
            actor_id="hand", actor_type="mano", q=np.zeros((frames, 15)), base=np.eye(4),
            observed=np.ones(frames, dtype=bool), model_uri="licensed:MANO_RIGHT",
            shared_shape=np.zeros((1, 10)), provenance=provenance)],
        objects=[scene_module.object_from_arrays(
            object_id="block", poses=poses(frames), observed=np.ones(frames, dtype=bool),
            primitive={"type": "box", "half_extents_m": [0.01, 0.02, 0.03]}, provenance=provenance)],
        support={"id": "table", "normal_world": [0, 0, 1], "offset_m": 0,
                 "confidence": 0.8, "provenance": provenance},
        contacts=[], uncertainty={"status": "unestimated"},
        task={"description": "synthetic pickup", "target": {
            "object_id": "block", "position_world_m": [0.1, 0, 0.03], "provenance": provenance}})


def contact(start=1, end=6, name="grasp", status="hypothesized"):
    return {"id": name, "a": "hand", "b": "block", "kind": "grasp",
            "start_frame": start, "end_frame": end, "status": status,
            "confidence": 0.7, "provenance": {"method": "synthetic contact schedule"}}


def test_mano_and_urdf_share_one_sequence_and_roundtrip(tmp_path):
    scene = make_scene()
    robot = scene_module.actor_from_arrays(
        actor_id="robot", actor_type="urdf", q=np.zeros((11, 2)), base=np.eye(4),
        observed=[False] * 11, model_uri="asset:panda.urdf", joint_names=["joint1", "finger1"],
        provenance={"method": "planned robot state"})
    scene["actors"].append(robot)
    scene["objects"][0]["observed"][3] = False
    scene["contacts"] = [contact()]
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene, allow_nan=False))
    loaded = scene_module.load_scene(path, require_metric_scale=True)
    assert loaded == scene
    assert loaded["objects"][0]["observed"][3] is False
    assert loaded["actors"][1]["observed"] == [False] * 11
    assert scene_module.validate_scene(loaded) == {
        "frames": 11, "actors": 2, "objects": 1, "contacts": 1, "metric_scale_status": "estimated"}


@pytest.mark.parametrize("change, message", [
    (lambda s: s["timestamps"].__setitem__(3, s["timestamps"][2]), "increase strictly"),
    (lambda s: s["objects"][0]["T_world_object"].pop(), "shape"),
    (lambda s: s["objects"][0]["observed"].pop(), "explicitly cover"),
    (lambda s: s["objects"][0].pop("observed"), "explicitly cover"),
    (lambda s: s["objects"][0]["T_world_object"][0][0].__setitem__(0, 2), "not orthonormal"),
    (lambda s: s["objects"][0]["T_world_object"][0][0].__setitem__(0, -1), "proper rotations"),
    (lambda s: s["actors"][0].__setitem__("shared_shape", [[0] * 10] * 11), "shared_shape"),
    (lambda s: s["actors"][0]["q"].pop(), "finite \\[T,D\\]"),
    (lambda s: s["objects"][0]["geometry"].__setitem__("per_frame_meshes", []), "per-frame geometry"),
    (lambda s: s["support"].__setitem__("normal_world", [0, 0, 2]), "unit length"),
    (lambda s: s["objects"][0].__setitem__("id", "hand"), "duplicate"),
    (lambda s: s["task"]["target"].__setitem__("object_id", "missing"), "identify an object"),
    (lambda s: s["metric_scale"].pop("status"), "metric_scale.status"),
    (lambda s: s.__setitem__("timestamps", ["bad"]), "must be numeric"),
    (lambda s: s["actors"][0].__setitem__("q", ["bad"]), "must be numeric"),
])
def test_reject_inconsistent_or_implicit_state(change, message):
    scene = make_scene()
    change(scene)
    with pytest.raises(SceneValidationError, match=message):
        scene_module.validate_scene(scene)


@pytest.mark.parametrize("update", [
    {"start_frame": 7, "end_frame": 2}, {"start_frame": -1}, {"end_frame": 11},
    {"start_frame": True}, {"b": "absent"}, {"a": "block"}, {"confidence": 2},
    {"status": "definitely_physical"}, {"provenance": {}}, {"a": []},
])
def test_reject_malformed_contact_intervals(update):
    scene = make_scene()
    item = contact()
    item.update(update)
    scene["contacts"] = [item]
    with pytest.raises(SceneValidationError):
        scene_module.validate_scene(scene)


def test_unresolved_scale_is_explicit_and_can_block_execution():
    scene = make_scene()
    scene["metric_scale"]["status"] = "unresolved"
    assert scene_module.validate_scene(scene)["metric_scale_status"] == "unresolved"
    with pytest.raises(SceneValidationError, match="scale is unresolved"):
        scene_module.validate_scene(scene, require_metric_scale=True)


def test_mesh_affine_conversion_preserves_geometry_without_per_frame_shapes():
    original = poses(5)
    stretch = np.array([[1.1, .02, 0], [.02, .8, 0], [0, 0, 1.3]])
    original[:, :3, :3] = original[:, :3, :3] @ stretch
    vertices = np.array([[0, 0, 0], [.01, 0, 0], [0, .02, 0]])
    ob = scene_module.object_from_arrays(
        object_id="mesh", poses=original, observed=[True, True, False, True, True],
        vertices_m=vertices, faces=np.array([[0, 1, 2]]), provenance={"method": "cached affine mesh fit"})
    reconstructed = np.asarray(ob["T_world_object"])
    shape = np.asarray(ob["geometry"]["local_linear"])
    for before, after in zip(original, reconstructed):
        np.testing.assert_allclose(vertices @ before[:3, :3].T + before[:3, 3],
                                   (vertices @ shape.T) @ after[:3, :3].T + after[:3, 3], atol=1e-10)
    assert ob["observed"][2] is False
    assert np.asarray(ob["geometry"]["vertices_m"]).shape == (3, 3)
    original[-1, :3, :3] *= 1.2
    with pytest.raises(SceneValidationError, match="per-frame mesh stretch"):
        scene_module.object_from_arrays(object_id="bad", poses=original, observed=[True] * 5,
                                        mesh_uri="mesh.ply", provenance={"method": "bad shape track"})


def test_five_contact_keyframes_keep_available_transitions_and_do_not_mutate_scene():
    scene = make_scene()
    scene["contacts"] = [contact()]
    before = deepcopy(scene)
    uniform = scene_module.select_keyframes(scene, budget=5, policy="uniform")
    events = scene_module.select_keyframes(scene, budget=5, policy="contact_events")
    assert uniform["indices"] == [0, 2, 5, 8, 10]
    assert uniform["omitted_contact_boundaries"] == [1, 7]
    assert events["indices"] == [0, 1, 4, 7, 10]
    assert events["omitted_contact_boundaries"] == []
    assert not events["interpolated"] and scene == before


def test_insufficient_event_budget_reports_every_omitted_boundary():
    scene = make_scene()
    scene["contacts"] = [contact(i, i, f"contact{i}") for i in range(1, 9)]
    selected = scene_module.select_keyframes(scene, budget=3, policy="contact_events")
    assert selected["indices"] == [0, 5, 10]
    assert selected["omitted_contact_boundaries"] == [1, 2, 3, 4, 6, 7, 8, 9]
    assert scene_module.select_keyframes(scene, budget=20)["indices"] == list(range(11))
    scene["contacts"] = [contact(status="unknown")]
    assert scene_module.select_keyframes(scene, policy="contact_events")["contact_boundaries"] == []
    with pytest.raises(SceneValidationError, match="two keyframes"):
        scene_module.select_keyframes(scene, budget=1)


def test_unknown_confidence_does_not_erase_hypothesized_contact_timing():
    scene = make_scene()
    item = contact()
    item["confidence"] = None
    item["provenance"]["confidence_status"] = "uncalibrated"
    scene["contacts"] = [item]
    assert scene_module.select_keyframes(scene, policy="contact_events")["contact_boundaries"] == [1, 7]
    scene["support"]["confidence"] = None
    scene["support"]["provenance"]["confidence_status"] = "unknown"
    scene_module.validate_scene(scene)
    item["provenance"].pop("confidence_status")
    with pytest.raises(SceneValidationError, match="explain missing confidence"):
        scene_module.validate_scene(scene)
