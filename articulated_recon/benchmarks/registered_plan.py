"""Compile a reconstructed rigid-object track into registered TCP/gripper targets.

No simulator, GT scene, object recentering, support snapping, IK, or inverse
dynamics is used. A declared deployment calibration is applied unchanged. This
adapter supports one continuous grasp, optionally followed by supported release.
It rejects missing/ambiguous contact state rather than inferring first/last grasp.
"""
import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import ProtocolError, file_sha256, load_manifest, require_execution_ready
from artirecon.scene_sequence import load_scene, validate_scene


class PlanError(ValueError):
    """The declared input is unsupported or geometrically infeasible."""


def default_config(object_id):
    """Fixed geometric priors, to be saved before comparing input variants."""
    return dict(schema_version=1, object_id=object_id, grasp_strategy="top_down",
                closing_axis_local=None, side_approach_axis_local=0, side_approach_sign=1,
                grasp_local_fraction=[.5, .5, .5], open_width_m=.08,
                minimum_grasp_width_m=.003, gripper_clearance_m=.005,
                approach_distance_m=.12, retreat_distance_m=.15, command_hz=20,
                max_translation_m_s=.12, max_rotation_rad_s=.8,
                pregrasp_duration_s=1.5, approach_duration_s=1.2, close_duration_s=.8,
                release_duration_s=.7, retreat_duration_s=1., hold_duration_s=1.,
                support_penetration_tolerance_m=.003,
                pregrasp_stationary_tolerance_m=.01, pregrasp_stationary_tolerance_rad=.05,
                workspace_bounds_robot_m=None)


def _pose(value, name):
    value = np.asarray(value, float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5, rtol=0)):
        raise PlanError(f"{name} must be a finite rigid SE(3) transform")
    return value


def validate_registration(registration, source_frame):
    if registration.get("schema") != "deployment-registration-v1":
        raise PlanError("Use a deployment-registration-v1 artifact; legacy evaluator frames cannot be relabeled")
    if registration.get("source_frame") != source_frame:
        raise PlanError("Calibration source_frame differs from reconstructed scene frame")
    if not isinstance(registration.get("target_frame"), str) or not registration["target_frame"].strip():
        raise PlanError("Calibration must name its robot target_frame")
    if registration.get("fixed_across_methods") is not True or registration.get("derived_from_evaluation_geometry") is not False:
        raise PlanError("Deployment calibration must be fixed across methods and independent of evaluation geometry")
    if registration.get("deployment_type") not in {"virtual", "measured"}:
        raise PlanError("Calibration must distinguish virtual placement from measured physical deployment")
    if not isinstance(registration.get("procedure"), str) or not registration["procedure"].strip():
        raise PlanError("Deployment calibration needs an explicit measurement/placement procedure")
    return _pose(registration.get("T_robot_scene"), "T_robot_scene")


def _config(config):
    expected = default_config(config.get("object_id"))
    if set(config) != set(expected) or config.get("schema_version") != 1:
        raise PlanError("Config must explicitly contain all default_config fields; do not change defaults silently")
    if not isinstance(config["object_id"], str) or not config["object_id"]:
        raise PlanError("Config must name a task object")
    if config["grasp_strategy"] not in {"top_down", "side"}:
        raise PlanError("Supported grasp_strategy: top_down or side")
    for field in ("closing_axis_local", "side_approach_axis_local"):
        value = config[field]
        if not (field == "closing_axis_local" and value is None) and (type(value) is not int or value not in (0, 1, 2)):
            raise PlanError(f"{field} must be an explicit local axis 0/1/2")
    if config["side_approach_sign"] not in (-1, 1):
        raise PlanError("side_approach_sign must be -1 or +1")
    fraction = np.asarray(config["grasp_local_fraction"], float)
    if fraction.shape != (3,) or not np.isfinite(fraction).all() or np.any((fraction < 0) | (fraction > 1)):
        raise PlanError("grasp_local_fraction must be three fractions in [0,1]")
    for key, value in config.items():
        if key.endswith(("_m", "_s", "_rad", "_rad_s", "_m_s")) and key != "workspace_bounds_robot_m":
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise PlanError(f"{key} must be finite and positive")
    if type(config["command_hz"]) is not int or not 1 <= config["command_hz"] <= 200:
        raise PlanError("command_hz must be an integer in [1,200]")
    bounds = config["workspace_bounds_robot_m"]
    if bounds is not None:
        bounds = np.asarray(bounds, float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
            raise PlanError("workspace_bounds_robot_m must be [minimum_xyz, maximum_xyz]")


def _vertices(geometry, external_vertices=None):
    kind = geometry["type"]
    if kind == "mesh":
        vertices = np.asarray(geometry.get("vertices_m", external_vertices), float)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3 or not np.isfinite(vertices).all():
            raise PlanError("Mesh needs finite vertices from its hash-verified shared asset")
        return vertices @ np.asarray(geometry.get("local_linear", np.eye(3))).T
    if kind == "box":
        half = np.asarray(geometry["half_extents_m"])
    elif kind == "cylinder":
        half = np.array([geometry["radius_m"], geometry["radius_m"], geometry["height_m"] / 2])
    elif kind == "sphere":
        half = np.repeat(geometry["radius_m"], 3)
    else:
        raise PlanError("Unsupported shared geometry")
    # Primitive corners are a conservative bounding envelope, not a replacement mesh.
    return np.asarray(list(product(*zip(-half, half))))


def _unit(vector, name):
    vector = np.asarray(vector, float)
    if np.linalg.norm(vector) < 1e-8:
        raise PlanError(f"Degenerate {name}")
    return vector / np.linalg.norm(vector)


def _grasp_rotation(approach, closing):
    # Same right-handed TCP convention as the legacy geometric planner: +Z
    # approaches the object and +Y is the parallel-jaw closing axis.
    approach = _unit(approach, "approach axis")
    closing = _unit(closing - approach * (approach @ closing), "closing axis")
    return np.stack([np.cross(closing, approach), closing, approach], axis=1)


def _grasp_interval(scene, object_id):
    contacts = [c for c in scene["contacts"] if c["kind"] == "grasp" and object_id in (c["a"], c["b"])]
    if len(contacts) != 1 or contacts[0]["status"] == "unknown":
        raise PlanError("Exactly one explicit continuous non-unknown grasp interval is required; no first/last fallback")
    contact = contacts[0]
    if contact["end_frame"] <= contact["start_frame"]:
        raise PlanError("Grasp interval must span at least two observations")
    return contact


def _pose_evidence(obj, times, task_mode, observed_until_s):
    """Keep recorded observations distinct from supported model forecasts."""
    observed = np.asarray(obj['observed'], bool)
    predicted = np.zeros(len(times), bool)
    if task_mode == 'reconstruction':
        return observed, predicted
    if task_mode != 'forecast':
        raise PlanError('Unknown planning task mode')
    if (not isinstance(observed_until_s, (int, float)) or isinstance(observed_until_s, bool)
            or not np.isfinite(observed_until_s)):
        raise PlanError('Forecast planning requires a finite observed-time cutoff')
    future = times > observed_until_s
    if times[0] != observed_until_s or np.count_nonzero(~future) != 1 or not observed[0]:
        raise PlanError('Strict forecast plan requires one observed initial object pose at the cutoff')
    if observed[future].any():
        raise PlanError('Generated future poses must not be relabeled as recorded observations')
    raw = np.asarray(obj.get('prediction_supported'))
    if raw.dtype.kind != 'b' or raw.shape != times.shape:
        raise PlanError('Forecast requires a complete boolean prediction_supported array')
    if raw[~future].any():
        raise PlanError('prediction_supported may be used only after the observed-time cutoff')
    provenance = obj.get('prediction_provenance')
    if (not isinstance(provenance, dict) or provenance.get('origin') != 'model_prediction'
            or provenance.get('generated') is not True
            or not isinstance(provenance.get('method'), str) or not provenance['method'].strip()
            or not isinstance(provenance.get('artifact_ids'), list) or not provenance['artifact_ids']
            or any(not isinstance(x, str) or not x for x in provenance['artifact_ids'])):
        raise PlanError('Forecast pose support requires explicit generated-model provenance and artifact IDs')
    return observed, raw.copy()


def _forecast_lineage(obj, times, predicted, active):
    """Bind claimed future support to actual consumed generated RGB ancestry."""
    ids = obj['prediction_provenance']['artifact_ids']
    if any(aid not in active for aid in ids):
        raise ProtocolError('Forecast pose provenance references an inactive or missing artifact')
    closure, todo = set(), list(ids)
    while todo:
        aid = todo.pop()
        if aid in closure:
            continue
        if aid not in active:
            raise ProtocolError('Forecast pose ancestry is not an active method dependency')
        closure.add(aid)
        todo.extend(active[aid]['parent_ids'])
    ranges = [a['time_range_s'] for aid, a in active.items() if aid in closure
              and a['kind'] == 'rgb' and a['origin'] == 'model_prediction'
              and a.get('generated') is True and 'time_range_s' in a]
    for t in times[predicted]:
        if not any(low <= t <= high for low, high in ranges):
            raise ProtocolError('Forecast pose lacks generated RGB ancestry covering its timestamp')


def build_plan(scene, registration, config, *, external_vertices=None,
               task_mode='reconstruction', observed_until_s=None):
    """Pure TCP compilation. All object positions change only by registration."""
    validate_scene(scene, require_metric_scale=True)
    _config(config)
    transform = validate_registration(registration, scene["coordinate_frame"]["name"])
    objects = [o for o in scene["objects"] if o["id"] == config["object_id"]]
    if len(objects) != 1:
        raise PlanError("Configured task object is missing")
    obj = objects[0]
    grasp = _grasp_interval(scene, obj["id"])
    start, end = grasp["start_frame"], grasp["end_frame"]
    times = np.asarray(scene["timestamps"], float)
    observed, predicted = _pose_evidence(obj, times, task_mode, observed_until_s)
    poses = transform @ np.asarray(obj["T_world_object"], float)
    release = end < len(times) - 1
    last = end + int(release)
    if not (observed | predicted)[:last + 1].all():
        raise PlanError("Unobserved or unsupported predicted object poses occur in the plan's required prefix")
    if release:
        support_id = scene["support"]["id"] if scene["support"] is not None else None
        supports = [c for c in scene["contacts"] if c["kind"] == "support"
                    and obj["id"] in (c["a"], c["b"]) and c["status"] != "unknown"
                    and ({c["a"], c["b"]} - {obj["id"]}) <= {support_id, "world"}
                    and c["start_frame"] <= last <= c["end_frame"]]
        if not supports or support_id is None:
            raise PlanError("Ending a grasp does not prove supported release; require an explicit support interval")
    before = poses[:start + 1]
    displacement = np.linalg.norm(before[:, :3, 3] - poses[0, :3, 3], axis=1).max()
    angle = (Rotation.from_matrix(poses[0, :3, :3]).inv() * Rotation.from_matrix(before[:, :3, :3])).magnitude().max()
    if displacement > config["pregrasp_stationary_tolerance_m"] or angle > config["pregrasp_stationary_tolerance_rad"]:
        raise PlanError("Object moves before declared grasp; cannot assume its initial evaluation pose equals pickup pose")
    vertices = _vertices(obj["geometry"], external_vertices)
    low, high = vertices.min(0), vertices.max(0)
    support = scene["support"]
    normal, offset, penetration = None, None, None
    if support is not None:
        normal = transform[:3, :3] @ np.asarray(support["normal_world"], float)
        offset = float(support["offset_m"] - normal @ transform[:3, 3])
        heights = (vertices @ poses[:last + 1, :3, :3].transpose(0, 2, 1) + poses[:last + 1, None, :3, 3]) @ normal + offset
        penetration = float(max(0., -heights.min()))
        if penetration > config["support_penetration_tolerance_m"]:
            raise PlanError(f"Estimated object penetrates supplied support by {penetration:.6f} m; no support lifting is permitted")
    if config["grasp_strategy"] == "top_down":
        if normal is None:
            raise PlanError("Top-down grasp requires an explicitly oriented support normal")
        approach = -normal
    else:
        approach = poses[start, :3, config["side_approach_axis_local"]] * config["side_approach_sign"]
    world_vertices = vertices @ poses[start, :3, :3].T
    candidates = []
    axes = range(3) if config["closing_axis_local"] is None else [config["closing_axis_local"]]
    for axis in axes:
        closing = poses[start, :3, axis]
        projected = closing - approach * (approach @ closing)
        if np.linalg.norm(projected) < 1e-6:
            continue
        rotation = _grasp_rotation(approach, closing)
        width = float(np.ptp(world_vertices @ rotation[:, 1]))
        candidates.append((width, axis, rotation))
    if not candidates:
        raise PlanError("No closing axis orthogonal to the configured approach")
    width, axis, rotation = min(candidates, key=lambda entry: (entry[0], entry[1]))
    if not config["minimum_grasp_width_m"] <= width <= config["open_width_m"] - config["gripper_clearance_m"]:
        raise PlanError(f"Estimated grasp width {width:.6f} m is outside the configured gripper envelope")
    local_point = low + np.asarray(config["grasp_local_fraction"]) * (high - low)
    tcp = np.eye(4)
    tcp[:3, :3] = rotation
    tcp[:3, 3] = poses[start, :3, :3] @ local_point + poses[start, :3, 3]
    relative = np.linalg.inv(poses[start]) @ tcp
    pre = tcp.copy()
    pre[:3, 3] -= tcp[:3, 2] * config["approach_distance_m"]
    targets, widths, grips, phases, durations, source_times, source_valid = [], [], [], [], [], [], []
    retiming = []

    def add(pose, opened, phase, duration, source=None):
        targets.append(pose.copy())
        widths.append(config["open_width_m"] if opened else 0.)
        grips.append(1. if opened else -1.)
        phases.append(phase)
        durations.append(float(duration))
        source_times.append(float(source) if source is not None else -1.)
        source_valid.append(source is not None)

    def ramp(first, second, opened, phase, duration, source_first=None, source_last=None):
        distance = np.linalg.norm(second[:3, 3] - first[:3, 3])
        radians = (Rotation.from_matrix(first[:3, :3]).inv() * Rotation.from_matrix(second[:3, :3])).magnitude()
        actual = max(duration, distance / config["max_translation_m_s"], radians / config["max_rotation_rad_s"])
        count = max(1, int(np.ceil(actual * config["command_hz"])))
        interpolation = Slerp([0., 1.], Rotation.from_matrix([first[:3, :3], second[:3, :3]]))
        for index in range(1, count + 1):
            fraction = index / count
            pose = np.eye(4)
            pose[:3, :3] = interpolation(fraction).as_matrix()
            pose[:3, 3] = first[:3, 3] * (1 - fraction) + second[:3, 3] * fraction
            source = source_first * (1 - fraction) + source_last * fraction if source_first is not None and source_last is not None else None
            add(pose, opened, phase, actual / count, source)
        if phase in {"transport", "place"}:
            retiming.append(dict(source_start_s=float(source_first), source_end_s=float(source_last),
                                 source_duration_s=float(duration), command_duration_s=float(actual)))

    add(pre, True, "pregrasp", config["pregrasp_duration_s"])
    ramp(pre, tcp, True, "approach", config["approach_duration_s"])
    ramp(tcp, tcp, False, "close", config["close_duration_s"], times[start], times[start])
    previous = tcp
    for index in range(start + 1, last + 1):
        destination = poses[index] @ relative
        ramp(previous, destination, False, "place" if release and index == last else "transport",
             times[index] - times[index - 1], times[index - 1], times[index])
        previous = destination
    if release:
        ramp(previous, previous, True, "release", config["release_duration_s"], times[last], times[last])
        retreat = previous.copy()
        retreat[:3, 3] -= previous[:3, 2] * config["retreat_distance_m"]
        ramp(previous, retreat, True, "retreat", config["retreat_duration_s"])
    else:
        ramp(previous, previous, False, "hold", config["hold_duration_s"], times[last], times[last])
    targets = np.asarray(targets)
    if config["workspace_bounds_robot_m"] is not None:
        bounds = np.asarray(config["workspace_bounds_robot_m"])
        if np.any(targets[:, :3, 3] < bounds[0]) or np.any(targets[:, :3, 3] > bounds[1]):
            raise PlanError("TCP targets leave the fixed deployment workspace envelope; no recentering is permitted")
    arrays = dict(T_robot_tcp=targets, gripper_width_m=np.array(widths), gripper_command=np.array(grips),
                  phase=np.array(phases), duration_s=np.array(durations), command_time_s=np.cumsum(durations),
                  source_time_s=np.array(source_times), source_time_valid=np.array(source_valid),
                  T_robot_object=poses, object_times_s=times,
                  object_observed=observed, object_prediction_supported=predicted,
                  T_object_tcp=relative, goal_pose=poses[last])
    report = dict(schema="registered-tcp-plan-v1", coordinate_frame=registration["target_frame"],
                  object_id=obj["id"], object_geometry_replaced=False, object_pose_corrections_m=0.,
                  registration_applied_only=True, source_grasp_interval=grasp,
                  first_grasp_frame=start, last_grasp_frame=end, release=release,
                  target_source_frame=last, target_source_time_s=float(times[last]),
                  grasp_width_m=width, closing_axis_local=axis, grasp_local_point_m=local_point.tolist(),
                  support_penetration_m=penetration, retiming=retiming,
                  command_count=len(targets), duration_s=float(sum(durations)),
                  feasibility={"gripper_envelope": "passed", "support_envelope": "passed" if support is not None else "unknown",
                               "workspace_envelope": "passed" if config["workspace_bounds_robot_m"] is not None else "unknown",
                               "panda_ik": "unmeasured", "robot_collision": "unknown", "force_closure": "unknown",
                               "initial_robot_to_pregrasp_path": "unknown", "inverse_dynamics": "not_performed"},
                  limitations=["TCP targets only; no Panda joint solution or physical execution is certified.",
                               "A bounding-envelope grasp is not a contact-quality or force-closure proof.",
                               "Zero closed-width is a force-closing gripper command, not a zero-width object estimate.",
                               "Object motion is preserved at source timestamps; segments may slow down to fixed speed limits.",
                               "source_time_valid=false marks synthetic approach/retreat; its -1 timestamp is a placeholder."],
                  simulator_feedback_used=False, simulator_accessed=False, inverse_dynamics_used=False)
    report['task_mode'] = task_mode
    report['object_pose_evidence'] = dict(observed=observed.tolist(), prediction_supported=predicted.tolist(),
        observed_until_s=observed_until_s if task_mode == 'forecast' else None,
        prediction_provenance=obj.get('prediction_provenance') if task_mode == 'forecast' else None,
        interpretation='observed denotes recorded evidence; prediction_supported denotes model-generated evidence, not ground truth')
    return arrays, report


def _artifact_path(artifact, manifest_dir):
    path = Path(artifact["path"])
    return (path if path.is_absolute() else manifest_dir / path).resolve()


def produce_plan(*, scene_path, input_manifest, registration_id, config_path, output):
    """Validate consumed lineage and write the plan plus producer hash links."""
    scene_path, input_manifest, config_path = map(lambda p: Path(p).resolve(), (scene_path, input_manifest, config_path))
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    protocol = load_manifest(input_manifest)
    require_execution_ready(protocol)
    manifest = json.loads(input_manifest.read_text())
    active = {a["id"]: a for a in manifest["artifacts"] if a["id"] in protocol["method_artifact_ids"]}
    if registration_id not in protocol["robot_registration"]["artifact_ids"] or registration_id not in active:
        raise ProtocolError("Registration artifact must be resolved and consumed by the method")
    registration_artifact = active[registration_id]
    if registration_artifact["kind"] != "robot_registration" or registration_artifact["origin"] == "ground_truth":
        raise ProtocolError("Registration must be a permitted deployment calibration artifact")
    registration_path = _artifact_path(registration_artifact, input_manifest.parent)
    registration = json.loads(registration_path.read_text())
    scene = load_scene(scene_path, require_metric_scale=True)
    # Apply the same geometry-ancestry and external-asset checks as the seal.
    from benchmarks.freeze_submission import _provenance_references, _scene_external_references
    _provenance_references(scene, set(active))
    _scene_external_references(scene, scene_path, active, input_manifest.parent)
    for artifact in active.values():
        if artifact["origin"] == "ground_truth" and artifact["kind"] in {"object_geometry", "support", "robot_registration"}:
            raise ProtocolError("Ground-truth geometry/support/registration are evaluator-only")
    config = json.loads(config_path.read_text())
    objects = [o for o in scene["objects"] if o["id"] == config.get("object_id")]
    vertices = None
    if len(objects) == 1 and "uri" in objects[0]["geometry"]:
        uri = objects[0]["geometry"]["uri"]
        if uri.startswith("artifact://"):
            mesh = _artifact_path(active[uri.removeprefix("artifact://")], input_manifest.parent)
        else:
            mesh = Path(uri)
            mesh = mesh if mesh.is_absolute() else scene_path.parent / mesh
        if mesh.suffix == ".npz":
            with np.load(mesh, allow_pickle=False) as archive:
                vertices = archive["vertices"].copy()
        else:
            import trimesh
            vertices = np.asarray(trimesh.load(mesh, process=False, force="mesh").vertices)
    if protocol['task_mode'] == 'forecast' and len(objects) == 1:
        times = np.asarray(scene['timestamps'], float)
        _, predicted = _pose_evidence(objects[0], times, 'forecast', manifest.get('observed_until_s'))
        _forecast_lineage(objects[0], times, predicted, active)
    arrays, report = build_plan(scene, registration, config, external_vertices=vertices,
        task_mode=protocol['task_mode'], observed_until_s=manifest.get('observed_until_s'))
    output.mkdir(parents=True)
    plan = output / "plan.npz"
    np.savez_compressed(plan, **arrays)
    metadata = {**report, "source_scene_sha256": file_sha256(scene_path),
                "input_manifest_sha256": protocol["manifest_sha256"], "plan_sha256": file_sha256(plan),
                "planner_sha256": file_sha256(__file__), "planner_config_sha256": file_sha256(config_path),
                "planner_config": config, "registration_artifact_id": registration_id,
                "registration_sha256": file_sha256(registration_path),
                "deployment_registration": {
                    "type": registration["deployment_type"], "origin": registration_artifact["origin"],
                    "status": protocol["robot_registration"]["status"], "source_frame": registration["source_frame"],
                    "target_frame": registration["target_frame"], "fixed_across_methods": True,
                    "independent_of_evaluation_geometry": True,
                },
                "command_format": "registered-tcp-gripper-npz-v1", "input_condition": protocol["condition"]}
    (output / "plan-metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    config = modes.add_parser("config")
    config.add_argument("--object-id", required=True)
    config.add_argument("--output", type=Path, required=True)
    run = modes.add_parser("compile")
    for name in ("scene-path", "input-manifest", "config-path", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--registration-id", required=True)
    args = vars(parser.parse_args())
    if args.pop("mode") == "config":
        payload = default_config(args["object_id"])
        with args["output"].open("x") as stream:
            json.dump(payload, stream, indent=2)
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(produce_plan(**args), indent=2))


if __name__ == "__main__":
    main()
