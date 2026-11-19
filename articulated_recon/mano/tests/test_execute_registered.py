"""Boundary and scheduling tests use a fake plant, not measured task outcomes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artirecon.input_protocol import ProtocolError, file_sha256
from benchmarks import execute_registered as executor
from benchmarks.freeze_submission import attach_evaluation, freeze_submission
from test_freeze_submission import case, refresh_metadata, write_json


def commands():
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, :3, 3] = [[.45, 0, .2], [.45, 0, .06], [.48, 0, .15], [.48, 0, .06]]
    poses[:, :3, :3] = Rotation.from_euler("XYZ", [np.pi, 0, 0]).as_matrix()
    durations = np.array([.5, .1, .05, .2])
    return dict(T_robot_tcp=poses, gripper_command=np.array([1., -1., -1., 1.]),
                gripper_width_m=np.array([.08, 0, 0, .08]),
                phase=np.array(["pregrasp", "close", "transport", "release"]),
                duration_s=durations, command_time_s=durations.cumsum(),
                source_time_s=np.array([-1., 0, 1, 2]), source_time_valid=np.array([False, True, True, True]))


def sealed(case, *, reference_x=.45):
    case["executor"] = Path(executor.__file__)
    config = executor.default_config("robot_base")
    write_json(case["controller_config"], config)
    with case["plan"].open("wb") as stream:
        np.savez(stream, **commands())
    refresh_metadata(case, command_format="registered-tcp-gripper-npz-v1", planner_config={"command_hz": 20})
    payload = freeze_submission(**case)
    root = case["output"].parent
    mesh = root / "reference.obj"
    mesh.write_text("\n".join("v " + " ".join(str(x) for x in vertex) for vertex in
                    [[-.025, -.025, -.025], [.025, -.025, -.025], [0, .025, -.025], [0, 0, .025]])
                    + "\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n")
    initial = np.eye(4); initial[:3, 3] = [reference_x, 0, .025]
    goal = initial.copy(); goal[0, 3] += .03
    actor = write_json(root / "actor-config.json", {
        "schema": "independent-rigid-evaluation-v1", "target_frame": "robot_base",
        "robot": {"uid": "panda", "base_pose": np.eye(4).tolist(), "root_frame": "robot_base"},
        "object": {"mesh": mesh.name, "initial_pose": initial.tolist(), "collision": "convex_hull_of_reference_mesh",
                   "density_kg_m3": 500., "friction": 1.},
        "support": {"type": "plane", "normal": [0., 0, 1.], "offset": 0., "friction": 1.},
        "goal_pose": goal.tolist(), "support_must_not_be_replaced_by_z0": True, "initial_pose_must_not_be_snapped": True})
    spec = write_json(root / "evaluation.json", {"schema_version": 1, "independent_scene": True,
        "artifacts": [{"id": name, "path": str(path), "sha256": file_sha256(path)}
                      for name, path in (("actor_config", actor), ("reference_mesh", mesh))]})
    receipt = root / "evaluation-attachment.json"
    attach_evaluation(submission=case["output"], evaluation_spec=spec, receipt=receipt)
    return payload, config, receipt


def test_schedule_uses_sealed_commands_and_initial_robot_pose_only():
    data, config = commands(), executor.default_config()
    initial = np.eye(4); initial[:3, 3] = [.3, -.1, .4]
    out = executor.control_schedule(data, config, initial)
    assert out["actions"].shape == (47, 7)  # 17 command ticks + 30 fixed settling ticks.
    assert out["timing"]["control_counts"] == [10, 2, 1, 4]
    assert np.allclose(out["targets"][9], data["T_robot_tcp"][0])
    assert np.allclose(out["targets"][10:12], data["T_robot_tcp"][1])
    assert np.allclose(out["targets"][-1], data["T_robot_tcp"][-1])
    assert np.allclose(Rotation.from_euler("XYZ", out["actions"][:, 3:6]).as_matrix(), out["targets"][:, :3, :3], atol=1e-6)
    assert np.all(out["actions"][10:13, -1] == -1)
    assert np.all(out["actions"][-30:, -1] == 1)
    assert np.all(out["phase"][-30:] == "final_hold")
    # Noncommand reconstructions or forecast goals cannot change scheduling.
    changed = copy.deepcopy(data)
    changed.update(T_robot_object=np.full((10, 4, 4), 100), goal_pose=np.full((4, 4), -999))
    assert np.array_equal(executor.control_schedule(changed, config, initial)["actions"], out["actions"])


def test_actual_registered_planner_samples_are_not_reinterpolated():
    from test_registered_plan import scene, registration
    from benchmarks import registered_plan
    arrays, _ = registered_plan.build_plan(scene(), registration(), registered_plan.default_config("block"))
    schedule = executor.control_schedule(arrays, executor.default_config("robot_base"), np.eye(4))
    counts = schedule["timing"]["control_counts"]
    assert counts[0] == 30 and set(counts[1:]) == {1}
    assert np.allclose(schedule["targets"][30:30 + len(counts) - 1], arrays["T_robot_tcp"][1:])
    assert schedule["timing"]["quantized_duration_s"] >= schedule["timing"]["original_duration_s"] - 1e-9


@pytest.mark.parametrize("normal", [[0, 0, 1], [0, 0, -1], [0, 1, 0], [.6, 0, .8]])
def test_collision_plane_keeps_orientation_and_offset(normal):
    normal = np.array(normal, float)
    result = executor.plane_collision_pose(normal, .07)
    assert np.allclose(result[:3, 0], normal)
    points = np.array([[0, 0, 0], [0, .6, -.2], [0, -.2, .3]])
    transformed = points @ result[:3, :3].T + result[:3, 3]
    assert np.allclose(transformed @ normal + .07, 0)


def test_commands_load_only_declared_fields_and_reject_wrong_frame(tmp_path):
    data = commands()
    data["source_time_s"][1:] = [-2., -1., 0.]  # Real timestamps may be negative; validity is explicit.
    path = tmp_path / "plan.npz"
    np.savez(path, **data, goal_pose=np.array([{"unsafe": "object array is never opened"}], dtype=object))
    metadata = dict(command_format="registered-tcp-gripper-npz-v1", coordinate_frame="panda_base", planner_config={"command_hz": 20})
    loaded = executor.load_commands(path, metadata, executor.default_config())
    assert set(loaded) == set(data) and not loaded["T_robot_tcp"].flags.writeable
    assert np.array_equal(loaded["source_time_s"], [-1., -2., -1., 0.])
    with pytest.raises(ValueError, match="coordinate frame"):
        executor.load_commands(path, metadata, executor.default_config("other_base"))
    data["duration_s"][0] = -1
    np.savez(path, **data)
    with pytest.raises(ValueError, match="timing"):
        executor.load_commands(path, metadata, executor.default_config())


class FakePose:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)

    def to_transformation_matrix(self):
        return self.matrix[None]


class FakePlant:
    """Emulates API shape only, not dynamics; no test results count as physics."""
    def __init__(self, evaluator, config):
        self.actions = []
        self.agent = SimpleNamespace(robot=SimpleNamespace(pose=FakePose(np.eye(4)), qpos=np.zeros((1, 9))),
                                     tcp_pose=FakePose(np.eye(4)), is_grasping=lambda obj: np.array([False]))
        self.object = SimpleNamespace(pose=FakePose(evaluator["config"]["object"]["initial_pose"]))
        self.scene = SimpleNamespace(sub_scenes=[SimpleNamespace(get_contacts=lambda: [])])

    def reset(self, seed):
        assert seed == 0

    def step(self, action):
        self.actions.append(action.copy())
        pose = np.eye(4); pose[:3, :3] = Rotation.from_euler("XYZ", action[3:6]).as_matrix(); pose[:3, 3] = action[:3]
        self.agent.tcp_pose = FakePose(pose)

    def close(self):
        pass


def test_executor_consumes_seal_once_and_logs_fixed_controls(case, monkeypatch):
    payload, config, receipt = sealed(case)
    monkeypatch.setattr(executor, "_runtime_versions", lambda config: {"test": "fake plant"})
    monkeypatch.setattr(executor, "make_environment", FakePlant)
    before = file_sha256(case["plan"])
    output = case["output"].parent / "run"
    summary = executor.run(submission=case["output"], evaluation_receipt=receipt, output=output)
    assert summary["steps"] == 47 and not summary["score"]["task_success"]
    assert not summary["object_or_contact_feedback_to_scheduler"]
    assert summary["object_pose_updates_after_initialization"] == 0
    assert file_sha256(case["plan"]) == before
    expected = executor.control_schedule(commands(), config, np.eye(4))["actions"]
    assert np.array_equal(np.load(output / "actions.npy"), expected)
    records = json.loads((output / "records.json").read_text())
    assert records[0]["time_s"] == .05 and records[-1]["time_s"] == 47 / 20
    with pytest.raises(FileExistsError):
        executor.run(submission=case["output"], evaluation_receipt=receipt, output=output.with_name("retry"))
    assert Path(payload["execution_receipt_path"]).exists()


def test_changed_reference_is_blocked_before_any_environment_or_claim(case, monkeypatch):
    payload, _, receipt = sealed(case)
    monkeypatch.setattr(executor, "make_environment", lambda *args: pytest.fail("Unsealed geometry reached simulator"))
    (receipt.parent / "reference.obj").write_text("changed reference")
    with pytest.raises(ProtocolError, match="changed"):
        executor.run(submission=case["output"], evaluation_receipt=receipt, output=receipt.parent / "run")
    assert not Path(payload["execution_receipt_path"]).exists()


def test_infrastructure_failure_retains_consumed_receipt_and_zero_step_evidence(case, monkeypatch):
    payload, _, receipt = sealed(case)
    monkeypatch.setattr(executor, "_runtime_versions", lambda config: {})
    def broken(*args):
        raise RuntimeError("synthetic construction failure")
    monkeypatch.setattr(executor, "make_environment", broken)
    output = receipt.parent / "run"
    with pytest.raises(RuntimeError, match="construction failure"):
        executor.run(submission=case["output"], evaluation_receipt=receipt, output=output)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["submitted_control_steps"] == failure["recorded_control_steps"] == 0
    assert failure["receipt_retained"] and not failure["automatic_retry"]
    assert Path(payload["execution_receipt_path"]).exists()


def test_runtime_versions_and_validation_code_are_bound_before_evaluation(monkeypatch):
    config = executor.default_config()
    executor.validate_config(config)
    monkeypatch.setattr(executor.importlib.metadata, "version", lambda name: "changed")
    with pytest.raises(ValueError, match="runtime differs"):
        executor._runtime_versions(config)
    config["validation_code_sha256"]["freeze_submission.py"] = "changed"
    with pytest.raises(ValueError, match="Validation implementation changed"):
        executor.validate_config(config)
