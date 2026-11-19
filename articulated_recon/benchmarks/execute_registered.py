"""Execute one sealed registered TCP plan in an independently attached scene.

Scheduling uses commands, fixed controller configuration and initial robot TCP
only. Object poses/contacts are logged and scored, never fed into scheduling or
grasp selection. ManiSkill's fixed IK/PD controller still uses proprioception.
This is interface enforcement, not process isolation or inverse dynamics.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import ProtocolError, file_sha256
from benchmarks.freeze_submission import claim_execution, verify_submission

HELPERS = ("freeze_submission.py", "../artirecon/input_protocol.py")


def default_config(target_frame="panda_base"):
    return {
        "schema": "panda-fixed-controller-v1", "target_frame": target_frame,
        "robot_uid": "panda", "control_mode": "pd_ee_pose", "sim_backend": "physx_cpu",
        "render_backend": "none", "control_hz": 20, "simulation_hz": 120,
        "initial_qpos": [0., 0., 0., -np.pi / 2, 0., np.pi / 2, np.pi / 4, .04, .04],
        "startup": "interpolate_initial_robot_tcp_to_first_target", "final_hold_s": 1.5,
        "duration_quantization": "nearest_control_tick_minimum_one_per_command",
        "object_density_kg_m3": 500., "object_friction": 1., "support_friction": 1., "restitution": 0.,
        "robot_urdf_sha256": "524c6986dbbabe244e05801f262862205e880834d01459b06ee273016d0579ba",
        "runtime_versions": {"mani_skill": "3.0.0b22", "sapien": "3.0.3", "torch": "2.14.1+cpu",
                             "numpy": "2.2.6", "scipy": "1.15.3"},
        "validation_code_sha256": {name: file_sha256(Path(__file__).parent / name) for name in HELPERS},
        "score_policy": {"minimum_lift_m": .025, "maximum_planar_goal_error_m": .025,
                         "maximum_final_speed_m_s": .01, "final_support_interval_m": [-.003, .003],
                         "stability_samples": 20},
    }


def _pose(value, name):
    value = np.asarray(value, float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5, rtol=0)):
        raise ValueError(f"{name} must be rigid SE(3)")
    return value


def validate_config(config):
    expected = default_config(config.get("target_frame"))
    if set(config) != set(expected) or not isinstance(config["target_frame"], str) or not config["target_frame"]:
        raise ValueError("Use a complete fixed controller config with a named target frame")
    for key in ("schema", "robot_uid", "control_mode", "sim_backend", "render_backend", "startup", "duration_quantization"):
        if config[key] != expected[key]:
            raise ValueError(f"Unsupported controller configuration: {key}")
    for key in ("control_hz", "simulation_hz"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["simulation_hz"] % config["control_hz"]:
        raise ValueError("Simulation frequency must be a multiple of control frequency")
    q = np.asarray(config["initial_qpos"], float)
    if q.shape != (9,) or not np.isfinite(q).all():
        raise ValueError("Panda initial_qpos must contain nine finite values")
    for key in ("final_hold_s", "object_density_kg_m3", "object_friction", "support_friction"):
        if not isinstance(config[key], (int, float)) or not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not isinstance(config["restitution"], (int, float)) or not 0 <= config["restitution"] <= 1:
        raise ValueError("Restitution must be in [0,1]")
    if config["validation_code_sha256"] != expected["validation_code_sha256"]:
        raise ValueError("Validation implementation changed since the controller config was written")
    if set(config["runtime_versions"]) != set(expected["runtime_versions"]):
        raise ValueError("Pin the complete controller runtime version set")
    policy = config["score_policy"]
    if set(policy) != set(expected["score_policy"]):
        raise ValueError("Declare the complete score policy before evaluation")
    for key in ("minimum_lift_m", "maximum_planar_goal_error_m", "maximum_final_speed_m_s"):
        if not np.isfinite(policy[key]) or policy[key] <= 0:
            raise ValueError("Evaluation thresholds must be positive")
    interval = np.asarray(policy["final_support_interval_m"], float)
    if interval.shape != (2,) or not np.isfinite(interval).all() or interval[1] < interval[0]:
        raise ValueError("Invalid final support interval")
    if type(policy["stability_samples"]) is not int or policy["stability_samples"] < 2:
        raise ValueError("At least two stability samples are required")


def load_commands(path, metadata, config):
    """Read only command fields; reconstructed object tracks/goals are not consumed."""
    if metadata.get("command_format") != "registered-tcp-gripper-npz-v1" or metadata.get("coordinate_frame") != config["target_frame"]:
        raise ValueError("Unsupported command format or coordinate frame")
    if metadata.get("planner_config", {}).get("command_hz") != config["control_hz"]:
        raise ValueError("Plan and controller sampling frequencies disagree")
    names = ("T_robot_tcp", "gripper_command", "gripper_width_m", "phase", "duration_s", "command_time_s", "source_time_s", "source_time_valid")
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.array(archive[name], copy=True) for name in names}
    targets = result["T_robot_tcp"]
    if targets.ndim != 3 or targets.shape[1:] != (4, 4) or not len(targets):
        raise ValueError("TCP commands must be nonempty [N,4,4]")
    for target in targets:
        _pose(target, "T_robot_tcp")
    n = len(targets)
    if any(result[k].shape != (n,) for k in names[1:]):
        raise ValueError("Command vectors have different lengths")
    for name in ("gripper_command", "gripper_width_m", "duration_s", "command_time_s", "source_time_s"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Nonfinite command field: {name}")
    if np.any(result["duration_s"] <= 0) or not np.allclose(result["command_time_s"], np.cumsum(result["duration_s"]), atol=1e-8, rtol=1e-7):
        raise ValueError("Invalid cumulative command timing")
    if not np.isin(result["gripper_command"], [-1., 1.]).all() or np.any(result["gripper_width_m"] < 0):
        raise ValueError("Expected explicit force-close/open gripper commands")
    allowed_phases = {"pregrasp", "approach", "close", "transport", "place", "release", "retreat", "hold"}
    if (result["phase"].dtype.kind != "U" or result["phase"][0] != "pregrasp"
            or not set(result["phase"]).issubset(allowed_phases)):
        raise ValueError("Commands must begin with an explicit pregrasp phase")
    if result["source_time_valid"].dtype != bool:
        raise ValueError("Source validity must be boolean")
    valid = result["source_time_valid"]
    if np.any(result["source_time_s"][~valid] != -1):
        raise ValueError("Invalid source clock sentinels")
    for value in result.values():
        value.setflags(write=False)
    return result


def control_schedule(commands, config, initial_robot_tcp):
    """Deterministic controls with no geometry, goal, contacts or evaluator arguments."""
    initial = _pose(initial_robot_tcp, "initial robot TCP")
    hz = config["control_hz"]
    counts = np.maximum(1, np.rint(commands["duration_s"] * hz).astype(int))
    targets, actions, indices, phases, source_times, valid = [], [], [], [], [], []
    first = commands["T_robot_tcp"][0]
    interpolation = Slerp([0, 1], Rotation.from_matrix([initial[:3, :3], first[:3, :3]]))

    def add(pose, index, phase=None):
        grip = commands["gripper_command"][index]
        action = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_euler("XYZ"), grip].astype(np.float32)
        targets.append(pose.copy()); actions.append(action); indices.append(index)
        phases.append(str(commands["phase"][index]) if phase is None else phase)
        source_times.append(float(commands["source_time_s"][index])); valid.append(bool(commands["source_time_valid"][index]))

    for index, count in enumerate(counts):
        for step in range(1, count + 1):
            pose = commands["T_robot_tcp"][index]
            if index == 0:
                u = step / count
                pose = np.eye(4)
                pose[:3, :3] = interpolation(u).as_matrix()
                pose[:3, 3] = initial[:3, 3] * (1 - u) + first[:3, 3] * u
            add(pose, index)
    for _ in range(max(1, round(config["final_hold_s"] * hz))):
        add(commands["T_robot_tcp"][-1], len(counts) - 1, "final_hold")
    return {"targets": np.asarray(targets), "actions": np.asarray(actions), "command_index": np.asarray(indices),
            "phase": np.asarray(phases), "source_time_s": np.asarray(source_times), "source_time_valid": np.asarray(valid),
            "timing": {"original_duration_s": float(commands["duration_s"].sum()),
                       "quantized_duration_s": float(counts.sum() / hz), "control_counts": counts.tolist(),
                       "final_hold_s": float((len(actions) - counts.sum()) / hz)}}


def plane_collision_pose(normal, offset):
    """SAPIEN plane local +X maps to the supplied positive-side normal."""
    n = np.asarray(normal, float)
    if n.shape != (3,) or not np.isfinite(n).all() or not np.isclose(np.linalg.norm(n), 1, atol=1e-6) or not np.isfinite(offset):
        raise ValueError("Support must be a unit-normal finite plane")
    tangent = np.array([0., 1., 0.]) if abs(n[1]) < .9 else np.array([0., 0., 1.])
    tangent -= n * (n @ tangent)
    tangent /= np.linalg.norm(tangent)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([n, tangent, np.cross(n, tangent)], axis=1)
    pose[:3, 3] = -float(offset) * n
    return _pose(pose, "support collision pose")


def load_evaluator(submission, receipt_path, config):
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get("schema") != "evaluation-attachment-v1" or receipt["submission"] != {"path": str(Path(submission).resolve()), "sha256": file_sha256(submission)}:
        raise ProtocolError("Evaluator attachment does not belong to this submission")
    for record in [receipt["evaluation_spec"], *receipt["reference_artifacts"]]:
        if file_sha256(record["path"]) != record["sha256"]:
            raise ProtocolError("Evaluator artifact changed after attachment")
    artifacts = {r["artifact_id"]: r for r in receipt["reference_artifacts"]}
    actor_path = Path(artifacts["actor_config"]["path"])
    actor = json.loads(actor_path.read_text())
    if actor.get("schema") != "independent-rigid-evaluation-v1" or actor["target_frame"] != config["target_frame"]:
        raise ValueError("Independent evaluator uses an incompatible frame/schema")
    if actor["robot"]["uid"] != "panda" or actor["robot"]["root_frame"] != config["target_frame"] or not np.allclose(_pose(actor["robot"]["base_pose"], "robot base"), np.eye(4), atol=1e-8):
        raise ValueError("The executor requires robot-base coordinates, with identity Panda placement")
    mesh = (actor_path.parent / actor["object"]["mesh"]).resolve()
    if mesh != Path(artifacts["reference_mesh"]["path"]).resolve() or actor["object"]["collision"] != "convex_hull_of_reference_mesh":
        raise ValueError("Object collision asset differs from attached reference mesh")
    if actor["object"]["density_kg_m3"] != config["object_density_kg_m3"] or actor["object"]["friction"] != config["object_friction"] or actor["support"]["friction"] != config["support_friction"]:
        raise ValueError("Evaluator physics differs from the presealed controller configuration")
    if actor["support"]["type"] != "plane" or actor.get("initial_pose_must_not_be_snapped") is not True or actor.get("support_must_not_be_replaced_by_z0") is not True:
        raise ValueError("Independent support/placement contract is missing")
    _pose(actor["object"]["initial_pose"], "reference initial pose"); _pose(actor["goal_pose"], "reference goal pose")
    plane_collision_pose(actor["support"]["normal"], actor["support"]["offset"])
    vertices = []
    for line in mesh.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError("Reference OBJ must contain finite vertices")
    return {"config": actor, "mesh": mesh, "vertices": vertices}


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _runtime_versions(config):
    actual = {name: importlib.metadata.version(name) for name in config["runtime_versions"]}
    if actual != config["runtime_versions"]:
        raise ValueError(f"Controller runtime differs from sealed versions: {actual}")
    return actual


def make_environment(evaluator, config):
    """No planner is imported here; native construction creates collisions only."""
    import sapien
    import torch
    import mani_skill
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.utils.structs import Pose
    urdf = Path(mani_skill.__file__).parent / "assets/robots/panda/panda_v2.urdf"
    if file_sha256(urdf) != config["robot_urdf_sha256"]:
        raise ValueError("Installed robot URDF differs from the sealed known model")

    def native_pose(matrix):
        q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        return sapien.Pose(matrix[:3, 3], q[[3, 0, 1, 2]])

    class IndependentPanda(BaseEnv):
        SUPPORTED_ROBOTS = ["panda"]

        @property
        def _default_sensor_configs(self):
            return []

        @property
        def _default_human_render_camera_configs(self):
            return []

        def _load_agent(self, options):
            super()._load_agent(options, sapien.Pose())

        def _load_scene(self, options):
            surface = evaluator["config"]["support"]
            support_material = sapien.physx.PhysxMaterial(config["support_friction"], config["support_friction"], config["restitution"])
            builder = self.scene.create_actor_builder()
            builder.add_plane_collision(native_pose(plane_collision_pose(surface["normal"], surface["offset"])), material=support_material)
            builder.initial_pose = sapien.Pose()
            self.support = builder.build_static(name="independent_support")
            material = sapien.physx.PhysxMaterial(config["object_friction"], config["object_friction"], config["restitution"])
            builder = self.scene.create_actor_builder()
            builder.initial_pose = native_pose(np.asarray(evaluator["config"]["object"]["initial_pose"]))
            builder.add_convex_collision_from_file(str(evaluator["mesh"]), material=material, density=config["object_density_kg_m3"])
            self.object = builder.build(name="reference_object")

        def _initialize_episode(self, env_idx, options):
            self.agent.reset(np.asarray(config["initial_qpos"]))
            self.object.set_pose(Pose.create(native_pose(np.asarray(evaluator["config"]["object"]["initial_pose"]))))

        def evaluate(self):
            return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

        def _get_obs_extra(self, info):
            return {}

    return IndependentPanda(robot_uids="panda", num_envs=1, obs_mode="none", reward_mode="none",
        control_mode=config["control_mode"], sim_backend=config["sim_backend"], render_backend="none", render_mode=None,
        sim_config={"control_freq": config["control_hz"], "sim_freq": config["simulation_hz"]})


def score_records(records, evaluator, config, release_requested):
    policy, hz = config["score_policy"], config["control_hz"]
    if len(records) < policy["stability_samples"]:
        raise ValueError("Insufficient recorded final stability samples")
    poses = np.asarray([r["object_pose"] for r in records]); initial = np.asarray(evaluator["config"]["object"]["initial_pose"])
    goal = np.asarray(evaluator["config"]["goal_pose"]); support = evaluator["config"]["support"]
    normal, offset = np.asarray(support["normal"]), support["offset"]
    delta = poses[-1, :3, 3] - goal[:3, 3]
    planar = np.linalg.norm(delta - normal * (normal @ delta))
    lift = np.max((poses[:, :3, 3] - initial[:3, 3]) @ normal)
    speed = np.max(np.linalg.norm(np.diff(poses[-policy["stability_samples"]:, :3, 3], axis=0), axis=1)) * hz
    bottom = np.min((evaluator["vertices"] @ poses[-1, :3, :3].T + poses[-1, :3, 3]) @ normal + offset)
    grasp = np.asarray([r["grasped"] for r in records], bool)
    low, high = policy["final_support_interval_m"]
    gates = {"ever_grasped": bool(grasp.any()), "lifted": bool(lift > policy["minimum_lift_m"]),
             "release_requested": bool(release_requested), "final_not_grasped": not bool(grasp[-1]),
             "stable": bool(speed < policy["maximum_final_speed_m_s"]),
             "planar_destination": bool(planar < policy["maximum_planar_goal_error_m"]),
             "physical_support": bool(low <= bottom <= high)}
    return {"task_success": all(gates.values()), "gates": gates,
            "metrics": {"peak_normal_lift_mm": float(lift * 1000), "planar_goal_error_mm": float(planar * 1000),
                        "xyz_goal_error_mm": float(np.linalg.norm(delta) * 1000), "signed_normal_error_mm": float(normal @ delta * 1000),
                        "rotation_error_deg": float(np.degrees(Rotation.from_matrix(poses[-1, :3, :3] @ goal[:3, :3].T).magnitude())),
                        "final_support_gap_mm": float(bottom * 1000), "final_speed_mm_s": float(speed * 1000)},
            "limitation": "Recorded RGB-D proxy with assumed physics; release/drop quality still requires visual/contact review"}


def _save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def run(*, submission, evaluation_receipt, output):
    frozen = verify_submission(submission)
    if Path(frozen["files"]["executor"]["path"]).resolve() != Path(__file__).resolve():
        raise ProtocolError("The frozen executor is not this execution bridge")
    config = json.loads(Path(frozen["files"]["controller_config"]["path"]).read_text())
    validate_config(config)
    commands = load_commands(frozen["files"]["plan"]["path"], frozen["plan_metadata"], config)
    evaluator = load_evaluator(submission, evaluation_receipt, config)
    versions = _runtime_versions(config)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    receipt = claim_execution(submission=submission, evaluation_receipt=evaluation_receipt, output=output)
    output.mkdir(parents=True)
    _save(output / "execution-inputs.json", {"claim": receipt, "runtime_versions": versions,
        "controller_config": config, "input_condition": frozen["input_protocol"]["condition"],
        "deployment": frozen["plan_metadata"].get("deployment_registration"),
        "policy": "Fixed TCP commands; proprioception inside fixed IK/PD only; evaluator state is logging/scoring only"})
    env, records, actual_actions = None, [], []
    start, stage = time.monotonic(), "environment_construction"
    try:
        env = make_environment(evaluator, config)
        env.reset(seed=0)
        base = _array(env.agent.robot.pose.to_transformation_matrix())[0]
        if not np.allclose(base, np.eye(4), atol=1e-6, rtol=0):
            raise ValueError("Robot base moved away from sealed target coordinates")
        initial = _array(env.agent.tcp_pose.to_transformation_matrix())[0]
        initial_object = _array(env.object.pose.to_transformation_matrix())[0]
        if not np.allclose(initial_object, evaluator["config"]["object"]["initial_pose"], atol=5e-6, rtol=0):
            raise ValueError("Independent object placement changed before execution")
        schedule = control_schedule(commands, config, initial)
        _save(output / "timing.json", schedule["timing"])
        stage = "fixed_command_execution"
        for index, action in enumerate(schedule["actions"]):
            env.step(action)
            actual_actions.append(action)
            tcp = _array(env.agent.tcp_pose.to_transformation_matrix())[0]
            contacts = []
            for contact in env.scene.sub_scenes[0].get_contacts():
                names = [body.entity.name for body in contact.bodies]
                if any("reference_object" in name for name in names):
                    contacts.append({"bodies": names, "separation_m": [float(p.separation) for p in contact.points],
                                     "impulse": [float(np.linalg.norm(p.impulse)) for p in contact.points]})
            record = {"step": index, "time_s": (index + 1) / config["control_hz"], "phase": str(schedule["phase"][index]),
                "source_command_index": int(schedule["command_index"][index]),
                "source_time_s": float(schedule["source_time_s"][index]), "source_time_valid": bool(schedule["source_time_valid"][index]),
                "target_tcp": schedule["targets"][index].tolist(), "tcp": tcp.tolist(),
                "qpos": _array(env.agent.robot.qpos)[0].tolist(),
                "object_pose": _array(env.object.pose.to_transformation_matrix())[0].tolist(),
                "grasped": bool(_array(env.agent.is_grasping(env.object))[0]), "contacts": contacts,
                "tcp_error_mm": float(np.linalg.norm(tcp[:3, 3] - schedule["targets"][index, :3, 3]) * 1000)}
            records.append(record)
        _save(output / "records.json", records)
        np.save(output / "actions.npy", np.asarray(actual_actions))
        release = bool(np.any((commands["phase"] == "release") & (commands["gripper_command"] > 0)))
        score = score_records(records, evaluator, config, release)
        summary = {"status": "complete", "physics_executed": True, "steps": len(records),
            "control_hz": config["control_hz"], "wall_seconds": time.monotonic() - start,
            "mean_tcp_error_mm": float(np.mean([r["tcp_error_mm"] for r in records])),
            "max_tcp_error_mm": float(max(r["tcp_error_mm"] for r in records)),
            "object_pose_updates_after_initialization": 0, "artificial_grasp_constraints": False,
            "object_or_contact_feedback_to_scheduler": False, "post_rollout_refinement": False,
            "proprioception_in_fixed_ik_pd": True, "inverse_dynamics_performed": False,
            "reference_is_exact_gt": False, "simulator_rendering": False,
            "records_sha256": file_sha256(output / "records.json"), "actions_sha256": file_sha256(output / "actions.npy"),
            "submission_sha256": file_sha256(submission), "score": score}
        _save(output / "summary.json", summary)
        return summary
    except Exception:
        if records and not (output / "records.json").exists():
            _save(output / "partial-records.json", records)
            np.save(output / "partial-actions.npy", np.asarray(actual_actions))
        _save(output / "failure.json", {"stage": stage, "recorded_control_steps": len(records),
            "submitted_control_steps": len(actual_actions), "error": traceback.format_exc(),
            "receipt_retained": True, "automatic_retry": False})
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="mode", required=True)
    config_parser = modes.add_parser("config")
    config_parser.add_argument("--target-frame", default="panda_base")
    config_parser.add_argument("--output", type=Path, required=True)
    execute = modes.add_parser("run")
    for name in ("submission", "evaluation-receipt", "output"):
        execute.add_argument("--" + name, type=Path, required=True)
    args = vars(p.parse_args())
    if args.pop("mode") == "config":
        _save(args["output"], default_config(args["target_frame"]))
    else:
        print(json.dumps(run(**args), indent=2))
