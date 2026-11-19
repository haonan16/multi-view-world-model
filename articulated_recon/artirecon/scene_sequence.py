"""Shared, explicit scene state for MANO/URDF reconstruction and planning.

Version 1 stores one shape per object and full SE(3) tracks in metres. ``observed``
marks direct evidence availability, not ground truth accuracy. Contact labels and
confidence are supplied estimates; validating this file does not prove physics.
Provenance references are checked against the separate input protocol by callers.

No renderer, model inference, time interpolation or simulator is used here.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np


class SceneValidationError(ValueError):
    """The scene does not satisfy the shared sequence contract."""


def _require(ok, message):
    if not ok:
        raise SceneValidationError(message)


def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")


def _numeric(value, name):
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SceneValidationError(f"{name} must be numeric") from exc


def _array(value, shape, name):
    result = _numeric(value, name)
    _require(result.shape == shape, f"{name} must have shape {shape}, got {result.shape}")
    _require(np.isfinite(result).all(), f"{name} contains non-finite values")
    return result


def _poses(value, frames, name):
    p = _array(value, (frames, 4, 4), name)
    _require(np.allclose(p[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0), f"{name} has invalid homogeneous rows")
    r = p[:, :3, :3]
    _require(np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-5, rtol=0),
             f"{name} rotation is not orthonormal; bake shared scale into geometry")
    _require(np.allclose(np.linalg.det(r), 1, atol=1e-5, rtol=0), f"{name} must contain proper rotations")
    return p


def _observed(value, frames, name):
    _require(isinstance(value, (list, np.ndarray)) and len(value) == frames,
             f"{name} must explicitly cover all {frames} frames")
    _require(all(isinstance(v, (bool, np.bool_)) for v in value), f"{name} must contain booleans")


def _provenance(value, name):
    _require(isinstance(value, dict) and bool(value), f"{name} needs explicit provenance")
    _text(value.get("method"), f"{name}.method")
    if "artifact_ids" in value:
        _require(isinstance(value["artifact_ids"], list), f"{name}.artifact_ids must be a list")
        for item in value["artifact_ids"]:
            _text(item, f"{name}.artifact_ids item")


def _confidence(value, name, provenance=None):
    if value is None:
        _require(isinstance(provenance, dict)
                 and provenance.get("confidence_status") in {"unknown", "uncalibrated", "unestimated"},
                 f"{name}=null requires provenance.confidence_status to explain missing confidence")
        return
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and np.isfinite(value) and 0 <= value <= 1, f"{name} must be a score in [0, 1]")


def _geometry(geometry, name):
    _require(isinstance(geometry, dict), f"{name} must be one shared geometry object")
    kind = geometry.get("type")
    _require(kind in {"mesh", "box", "sphere", "cylinder"}, f"{name}.type is unsupported")
    allowed = {"type", "provenance"}
    if kind == "mesh":
        allowed |= {"uri", "vertices_m", "faces", "local_linear"}
        uri = geometry.get("uri")
        inline = "vertices_m" in geometry or "faces" in geometry
        _require((uri is not None) != inline, f"{name} needs either mesh URI or inline vertices/faces")
        if uri is not None:
            _text(uri, f"{name}.uri")
        else:
            vertices = _numeric(geometry.get("vertices_m"), f"{name}.vertices_m")
            _require(vertices.ndim == 2 and vertices.shape[1:] == (3,) and len(vertices) >= 3,
                     f"{name}.vertices_m must be [N,3]")
            _require(np.isfinite(vertices).all(), f"{name}.vertices_m contains non-finite values")
            try:
                faces = np.asarray(geometry.get("faces"))
            except (TypeError, ValueError) as exc:
                raise SceneValidationError(f"{name}.faces must be integer triangle indices") from exc
            _require(faces.ndim == 2 and faces.shape[1:] == (3,) and len(faces) > 0,
                     f"{name}.faces must be [M,3]")
            _require(np.issubdtype(faces.dtype, np.integer) and faces.min() >= 0 and faces.max() < len(vertices),
                     f"{name}.faces contains invalid indices")
        if "local_linear" in geometry:
            stretch = _array(geometry["local_linear"], (3, 3), f"{name}.local_linear")
            _require(np.linalg.det(stretch) > 1e-12, f"{name}.local_linear must preserve orientation and volume")
    elif kind == "box":
        allowed.add("half_extents_m")
        _require((_array(geometry.get("half_extents_m"), (3,), f"{name}.half_extents_m") > 0).all(),
                 f"{name}.half_extents_m must be positive")
    else:
        allowed |= {"radius_m"}
        fields = ["radius_m"]
        if kind == "cylinder":
            allowed.add("height_m")
            fields.append("height_m")
        for field in fields:
            _require(float(_array(geometry.get(field), (), f"{name}.{field}")) > 0,
                     f"{name}.{field} must be positive")
    _require(not set(geometry) - allowed, f"{name} has unsupported/per-frame geometry fields: {set(geometry) - allowed}")
    _provenance(geometry.get("provenance"), f"{name}.provenance")


def validate_scene(scene, require_metric_scale=False):
    """Validate a JSON scene, returning counts; never repair or align it silently.

    ``require_metric_scale`` rejects unresolved scale, but estimated scale is
    still an estimate. Robot registration, collision checking and input leakage
    checks remain separate requirements before executing a plan.
    """
    _require(isinstance(scene, dict), "scene must be an object")
    _require(scene.get("schema_version") == 1, "schema_version must be 1")
    _require(scene.get("units") == {"length": "m", "time": "s", "angle": "rad"}, "units must be SI m/s/rad")
    frame = scene.get("coordinate_frame", {})
    _require(isinstance(frame, dict), "coordinate_frame must be an object")
    _text(frame.get("name"), "coordinate_frame.name")
    _require(frame.get("handedness") == "right", "coordinate_frame must be right handed")
    _require(frame.get("pose_convention") == "T_world_local", "pose_convention must be T_world_local")
    _require(frame.get("up_axis") in {"+x", "-x", "+y", "-y", "+z", "-z", "unknown"}, "up_axis must be explicit")
    timestamps = _numeric(scene.get("timestamps"), "timestamps")
    _require(timestamps.ndim == 1 and len(timestamps) > 0 and np.isfinite(timestamps).all(),
             "timestamps must be a nonempty finite sequence")
    _require((np.diff(timestamps) > 0).all(), "timestamps must increase strictly")
    frames = len(timestamps)
    scale = scene.get("metric_scale", {})
    _require(isinstance(scale, dict) and scale.get("status") in {"known", "estimated", "unresolved"},
             "metric_scale.status must be known, estimated or unresolved")
    _provenance(scale.get("provenance"), "metric_scale.provenance")
    _require(not require_metric_scale or scale["status"] != "unresolved", "metric scale is unresolved")
    _require(isinstance(scene.get("actors"), list) and isinstance(scene.get("objects"), list),
             "actors and objects must be lists")
    ids = {"world"}
    for category in ("actors", "objects"):
        for entry in scene[category]:
            _require(isinstance(entry, dict), f"{category} entries must be objects")
            name = entry.get("id")
            _text(name, f"{category}.id")
            _require(name not in ids, f"duplicate/reserved entity id: {name}")
            ids.add(name)
            _observed(entry.get("observed"), frames, f"{name}.observed")
            _provenance(entry.get("provenance"), f"{name}.provenance")
            if category == "objects":
                _geometry(entry.get("geometry"), f"{name}.geometry")
                _poses(entry.get("T_world_object"), frames, f"{name}.T_world_object")
            else:
                _require(entry.get("type") in {"mano", "urdf"}, f"{name}.type must be mano or urdf")
                _text(entry.get("model_uri"), f"{name}.model_uri")
                q = _numeric(entry.get("q"), f"{name}.q")
                _require(q.ndim == 2 and q.shape[0] == frames and q.shape[1] > 0 and np.isfinite(q).all(),
                         f"{name}.q must be finite [T,D]")
                _poses(entry.get("T_world_base"), frames, f"{name}.T_world_base")
                if entry["type"] == "mano":
                    _require(4 <= q.shape[1] <= 48, f"{name}.q must contain wrist axis-angle and 1..45 PCA coordinates")
                    _require(entry.get("state_convention") == "wrist_axis_angle_then_pca",
                             f"{name} needs explicit MANO state_convention")
                    _require(("shared_shape" in entry) != ("shape_ref" in entry), f"{name} needs one shared_shape or shape_ref")
                    if "shared_shape" in entry:
                        _array(entry["shared_shape"], (10,), f"{name}.shared_shape")
                    else:
                        _text(entry["shape_ref"], f"{name}.shape_ref")
                else:
                    _require(entry.get("state_convention") == "urdf_joint_values", f"{name} needs URDF joint convention")
                    names = entry.get("joint_names")
                    _require(isinstance(names, list) and len(names) == q.shape[1] and len(set(names)) == len(names),
                             f"{name}.joint_names must identify each q coordinate")
                    for joint in names:
                        _text(joint, f"{name}.joint_names item")
    support = scene.get("support")
    if support is not None:
        _require(isinstance(support, dict), "support must be an object or null")
        _text(support.get("id"), "support.id")
        _require(support["id"] not in ids, "support id duplicates another entity")
        ids.add(support["id"])
        normal = _array(support.get("normal_world"), (3,), "support.normal_world")
        _require(np.isclose(np.linalg.norm(normal), 1, atol=1e-6), "support normal must have unit length")
        _array(support.get("offset_m"), (), "support.offset_m")
        _confidence(support.get("confidence"), "support.confidence", support.get("provenance"))
        _provenance(support.get("provenance"), "support.provenance")
    contacts = scene.get("contacts")
    _require(isinstance(contacts, list), "contacts must be a list")
    contact_ids = set()
    for contact in contacts:
        _require(isinstance(contact, dict), "contact must be an object")
        name = contact.get("id")
        _text(name, "contact.id")
        _require(name not in contact_ids, f"duplicate contact id: {name}")
        contact_ids.add(name)
        _text(contact.get("a"), f"{name}.a")
        _text(contact.get("b"), f"{name}.b")
        _require(contact.get("a") in ids and contact.get("b") in ids and contact["a"] != contact["b"],
                 f"{name} must reference two distinct known entities")
        _require(contact.get("kind") in {"grasp", "touch", "support", "attachment"}, f"{name}.kind is unsupported")
        start, end = contact.get("start_frame"), contact.get("end_frame")
        _require(type(start) is int and type(end) is int and 0 <= start <= end < frames,
                 f"{name} has an invalid inclusive contact interval")
        _require(contact.get("status") in {"observed", "hypothesized", "unknown"}, f"{name}.status is invalid")
        _confidence(contact.get("confidence"), f"{name}.confidence", contact.get("provenance"))
        _provenance(contact.get("provenance"), f"{name}.provenance")
    uncertainty = scene.get("uncertainty", {})
    _require(isinstance(uncertainty, dict) and uncertainty.get("status") in {"unestimated", "estimated", "calibrated"},
             "uncertainty.status must be explicit")
    if uncertainty["status"] != "unestimated":
        _provenance(uncertainty.get("provenance"), "uncertainty.provenance")
    task = scene.get("task")
    _require(isinstance(task, dict), "task must be an object")
    _text(task.get("description"), "task.description")
    target = task.get("target")
    if target is not None:
        _require(isinstance(target, dict) and target.get("object_id") in {o["id"] for o in scene["objects"]},
                 "task.target must identify an object")
        _require(("T_world_object" in target) != ("position_world_m" in target), "target needs a pose or position")
        if "T_world_object" in target:
            _poses([target["T_world_object"]], 1, "task.target.T_world_object")
        else:
            _array(target["position_world_m"], (3,), "task.target.position_world_m")
        _provenance(target.get("provenance"), "task.target.provenance")
    return {"frames": frames, "actors": len(scene["actors"]), "objects": len(scene["objects"]),
            "contacts": len(contacts), "metric_scale_status": scale["status"]}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return deepcopy(value)


def scene_from_arrays(*, timestamps, coordinate_frame, metric_scale, actors, objects,
                      support, contacts, uncertainty, task):
    """Assemble caller-supplied states without inventing missing observations."""
    scene = _jsonable(dict(schema_version=1, units={"length": "m", "time": "s", "angle": "rad"},
                          coordinate_frame=coordinate_frame, metric_scale=metric_scale,
                          timestamps=timestamps, actors=actors, objects=objects, support=support,
                          contacts=contacts, uncertainty=uncertainty, task=task))
    validate_scene(scene)
    return scene


def object_from_arrays(*, object_id, poses, observed, provenance, mesh_uri=None,
                       vertices_m=None, faces=None, primitive=None):
    """Convert an existing rigid object track, preserving constant mesh stretch.

    Use one mesh URI, one inline mesh, or one primitive dictionary. A constant
    positive affine stretch is retained as mesh ``local_linear``; changing shape
    across frames is rejected. Primitive poses must already be SE(3).
    """
    p = _numeric(poses, "poses")
    _require(p.ndim == 3 and p.shape[1:] == (4, 4) and len(p) > 0, "poses must be [T,4,4]")
    _require(np.isfinite(p).all() and np.allclose(p[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0),
             "poses contain invalid affine transforms")
    inline = vertices_m is not None or faces is not None
    _require(sum([mesh_uri is not None, inline, primitive is not None]) == 1, "provide exactly one shared geometry")
    if primitive is not None:
        geometry = deepcopy(primitive)
        _poses(p, len(p), "primitive poses")
    else:
        geometry = {"type": "mesh", "uri": mesh_uri} if mesh_uri is not None else {
            "type": "mesh", "vertices_m": vertices_m, "faces": faces}
        _require((np.linalg.det(p[:, :3, :3]) > 1e-12).all(), "mesh poses must preserve orientation and volume")
        u, values, vh = np.linalg.svd(p[:, :3, :3])
        rotations = u @ vh
        stretch = vh.transpose(0, 2, 1) @ (values[:, :, None] * vh)
        _require(np.allclose(stretch, stretch[0], atol=1e-6, rtol=1e-5),
                 "per-frame mesh stretch changes shape; reconcile shape before exporting a rigid track")
        p = p.copy()
        p[:, :3, :3] = rotations
        if not np.allclose(stretch[0], np.eye(3), atol=1e-7, rtol=0):
            geometry["local_linear"] = stretch[0]
    geometry["provenance"] = deepcopy(provenance)
    result = _jsonable({"id": object_id, "geometry": geometry, "T_world_object": p,
                        "observed": observed, "provenance": provenance})
    _geometry(result["geometry"], object_id + ".geometry")
    _poses(result["T_world_object"], len(p), object_id + ".T_world_object")
    _observed(result["observed"], len(p), object_id + ".observed")
    return result


def actor_from_arrays(*, actor_id, actor_type, q, base, observed, model_uri,
                      provenance, shared_shape=None, shape_ref=None, joint_names=None):
    """Convert MANO or URDF state arrays; a supplied static base may be broadcast.

    MANO q is wrist axis-angle followed by PCA coefficients. Its translation is
    in ``base``. URDF q follows the explicit joint_names (revolute rad/prismatic m).
    """
    q = _numeric(q, "q")
    _require(q.ndim == 2 and len(q) > 0, "q must be [T,D]")
    base = _numeric(base, "base")
    if base.shape == (4, 4):
        base = np.broadcast_to(base, (len(q), 4, 4)).copy()
    result = dict(id=actor_id, type=actor_type, model_uri=model_uri, q=q,
                  T_world_base=base, observed=observed, provenance=provenance,
                  state_convention="wrist_axis_angle_then_pca" if actor_type == "mano" else "urdf_joint_values")
    if actor_type == "mano":
        if shared_shape is not None:
            shape = np.asarray(shared_shape)
            result["shared_shape"] = shape[0] if shape.shape == (1, 10) else shape
        if shape_ref is not None:
            result["shape_ref"] = shape_ref
    else:
        result["joint_names"] = joint_names
    return _jsonable(result)


def load_scene(path, require_metric_scale=False):
    scene = json.loads(Path(path).read_text())
    validate_scene(scene, require_metric_scale=require_metric_scale)
    return scene


def select_keyframes(scene, budget=5, policy="uniform"):
    """Choose existing frames, preserving endpoints and reporting dropped events.

    Uniform uses equally spaced frame indices. Contact events are start frames
    and the first frame after each inclusive interval; unknown contact intervals
    create no asserted events. With insufficient budget, farthest-time sampling
    chooses among event boundaries (ties choose the earliest frame), then fills
    remaining slots with farthest-time non-event frames. No interpolation occurs.
    """
    validate_scene(scene)
    _require(type(budget) is int and budget > 0, "budget must be a positive integer")
    _require(policy in {"uniform", "contact_events"}, "unknown keyframe policy")
    times = np.asarray(scene["timestamps"])
    frames = len(times)
    count = min(budget, frames)
    _require(count >= 2 or frames == 1, "at least two keyframes are required to preserve both endpoints")
    boundaries = set()
    for contact in scene["contacts"]:
        if contact["status"] != "unknown":
            boundaries.add(contact["start_frame"])
            if contact["end_frame"] + 1 < frames:
                boundaries.add(contact["end_frame"] + 1)
    if policy == "uniform":
        selected = set(np.rint(np.linspace(0, frames - 1, count)).astype(int).tolist())
    else:
        selected = {0, frames - 1}
        for candidates in (boundaries, set(range(frames))):
            remaining = candidates - selected
            while len(selected) < count and remaining:
                item = max(remaining, key=lambda i: (min(abs(times[i] - times[j]) for j in selected), -i))
                selected.add(item)
                remaining.remove(item)
    indices = sorted(selected)
    return {"policy": policy, "budget": budget, "indices": indices,
            "timestamps": times[indices].tolist(), "contact_boundaries": sorted(boundaries),
            "omitted_contact_boundaries": sorted(boundaries - selected),
            "interpolated": False}
