"""Attach a recorded RGB-D proxy in a frozen deployment frame after planning.

The planner never needs this module or its evaluator inputs. Published source
camera C is used only here: T_robot_recording = T_robot_source @ inv(C).
No fitted alignment, support snap, scale fit, endpoint shift or object recenter
is applied in world space. Local mesh origin canonicalization preserves every
surface point. The reference remains a reconstructed proxy, not exact GT CAD.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import ProtocolError, file_sha256
from artirecon.scene_sequence import validate_scene
from benchmarks.freeze_submission import attach_evaluation, verify_submission
from benchmarks.registered_plan import _pose, validate_registration


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def transform_reference(reference, source_c2w, deployment):
    """Pure exact coordinate conversion of one fixed-shape reference object."""
    C = _pose(source_c2w, "published source camera c2w")
    transform = _pose(deployment, "T_robot_source") @ np.linalg.inv(C)
    if len(reference.get("objects", [])) != 1:
        raise ValueError("Evaluator adapter currently requires one reference object")
    obj = reference["objects"][0]
    vertices, faces = np.asarray(obj["vertices"], float), np.asarray(obj["faces"])
    poses, times = np.asarray(obj["poses"], float), np.asarray(reference["timestamps"], float)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise ValueError("Reference mesh vertices must be finite [N,3]")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Reference faces must index mesh vertices")
    if times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Reference timestamps must increase strictly")
    if poses.shape != (len(times), 4, 4) or not np.isfinite(poses).all() or not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("Reference poses must be finite homogeneous matrices")
    rotations, stretches = [], []
    for matrix in poses[:, :3, :3]:
        u, singular, vt = np.linalg.svd(matrix)
        rotation, stretch = u @ vt, vt.T @ np.diag(singular) @ vt
        if np.linalg.det(rotation) <= 0 or singular.min() <= 1e-10:
            raise ValueError("Reference linear transforms must preserve orientation and volume")
        rotations.append(rotation)
        stretches.append(stretch)
    rotations, stretches = np.asarray(rotations), np.asarray(stretches)
    drift = np.max(np.linalg.norm(vertices @ (stretches - stretches[0]).transpose(0, 2, 1), axis=-1))
    if drift > 1e-6:
        raise ValueError("Reference shape varies through time; cannot silently fit one rigid asset")
    scaled = vertices @ stretches[0].T
    center = (scaled.min(0) + scaled.max(0)) / 2
    canonical_vertices = scaled - center
    canonical_poses = np.tile(np.eye(4), (len(times), 1, 1))
    canonical_poses[:, :3, :3] = rotations
    canonical_poses[:, :3, 3] = poses[:, :3, 3] + np.einsum("tij,j->ti", rotations, center)
    registered = transform @ canonical_poses
    table = reference["table"]
    normal, offset = np.asarray(table["normal"], float), float(table["offset"])
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(offset) or np.linalg.norm(normal) < 1e-10:
        raise ValueError("Reference table plane is invalid")
    length = np.linalg.norm(normal)
    normal, offset = normal / length, offset / length
    registered_normal = transform[:3, :3] @ normal
    registered_offset = offset - registered_normal @ transform[:3, 3]
    world_vertices = np.einsum("tij,pj->tpi", registered[:, :3, :3], canonical_vertices) + registered[:, None, :3, 3]
    raw_world = np.einsum("tij,pj->tpi", poses[:, :3, :3], vertices) + poses[:, None, :3, 3]
    direct = raw_world @ transform[:3, :3].T + transform[:3, 3]
    error = float(np.max(np.linalg.norm(world_vertices - direct, axis=-1)))
    if error > 1e-6:
        raise ValueError("Canonicalization changed reference surface positions")
    return dict(vertices=canonical_vertices, faces=faces, poses=registered, timestamps=times,
                source_c2w=C, T_robot_recording=transform, support_normal=registered_normal,
                support_offset=registered_offset,
                support_minimum_m=(world_vertices @ registered_normal + registered_offset).min(axis=1),
                canonicalization_max_surface_error_m=error, shared_shape_variation_m=float(drift))


def prepare(*, submission, raw_reference, source_cameras, source_view, deployment, output,
            source_camera_id, source_convention="opencv"):
    """Verify freeze and registration binding before opening evaluator inputs."""
    frozen = verify_submission(submission)  # Must be the first artifact read.
    output, deployment = Path(output).resolve(), Path(deployment).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = Path(frozen["files"]["input_manifest"]["path"])
    manifest = read(manifest_path)
    registration_id = frozen["plan_metadata"]["registration_artifact_id"]
    artifact = next(a for a in manifest["artifacts"] if a["id"] == registration_id)
    artifact_path = Path(artifact["path"])
    artifact_path = artifact_path if artifact_path.is_absolute() else manifest_path.parent / artifact_path
    if artifact_path.resolve() != deployment or file_sha256(deployment) != artifact["sha256"]:
        raise ProtocolError("Evaluator deployment differs from frozen planner registration")
    method_scene = read(frozen["files"]["scene"]["path"])
    registration = read(deployment)
    T = validate_registration(registration, method_scene["coordinate_frame"]["name"])
    if registration["source_frame"] != "source_camera":
        raise ProtocolError("This evaluator conversion requires the explicit source_camera reconstruction gauge")
    if registration["target_frame"] != frozen["plan_metadata"]["coordinate_frame"]:
        raise ProtocolError("Evaluator target frame differs from frozen commands")
    if source_convention != "opencv" or registration.get("source_coordinate_convention", source_convention) != source_convention:
        raise ProtocolError("Only explicitly declared OpenCV source-camera coordinates are supported")
    if not source_camera_id or registration.get("source_camera_id", source_camera_id) != source_camera_id:
        raise ProtocolError("Source camera identity differs from deployment metadata")
    if type(source_view) is not int or source_view < 0:
        raise ProtocolError("source_view must be a nonnegative integer")

    # Evaluator-only files become visible only after the preceding frozen-plan checks.
    reference = read(raw_reference)
    with np.load(source_cameras, allow_pickle=False) as archive:
        cameras = archive["c2w"]
    if cameras.shape == (4, 4):
        if source_view != 0:
            raise ProtocolError("A single-camera archive requires source_view=0")
        C = cameras
    elif cameras.ndim == 3 and cameras.shape[1:] == (4, 4) and source_view < len(cameras):
        C = cameras[source_view]
    else:
        raise ProtocolError("Camera archive must contain c2w [V,4,4] or [4,4]")
    data = transform_reference(reference, C, T)
    provenance = {"method": "Evaluation-only recorded RGB-D reconstruction transformed by frozen deployment and published source camera",
                  "artifact_ids": ["raw_reference", "published_source_camera"], "confidence_status": "unknown"}
    scene = {
        "schema_version": 1, "units": {"length": "m", "time": "s", "angle": "rad"},
        "coordinate_frame": {"name": registration["target_frame"], "handedness": "right", "up_axis": "+z", "pose_convention": "T_world_local"},
        "timestamps": data["timestamps"].tolist(), "metric_scale": {"status": "estimated", "provenance": provenance},
        "actors": [], "objects": [{"id": "reference_object", "geometry": {"type": "mesh", "vertices_m": data["vertices"].tolist(),
            "faces": data["faces"].tolist(), "provenance": provenance}, "T_world_object": data["poses"].tolist(),
            "observed": [True] * len(data["timestamps"]), "provenance": provenance}],
        "support": {"id": "reference_support", "normal_world": data["support_normal"].tolist(), "offset_m": float(data["support_offset"]),
                    "confidence": None, "provenance": provenance},
        "contacts": [], "uncertainty": {"status": "unestimated"},
        "task": {"description": "Independent recorded-reference object placement", "target": {"object_id": "reference_object",
                 "T_world_object": data["poses"][-1].tolist(), "provenance": provenance}},
    }
    validate_scene(scene, require_metric_scale=True)
    output.mkdir(parents=True)
    write(output / "scene.json", scene)
    np.savez_compressed(output / "fixture.npz", vertices=data["vertices"], faces=data["faces"],
        initial_pose=data["poses"][0], goal_pose=data["poses"][-1], poses=data["poses"], timestamps=data["timestamps"],
        support_normal=data["support_normal"], support_offset=data["support_offset"], T_robot_recording=data["T_robot_recording"])
    with (output / "reference.obj").open("x") as stream:
        for v in data["vertices"]:
            stream.write("v " + " ".join(format(float(x), ".17g") for x in v) + "\n")
        for f in data["faces"]:
            stream.write("f " + " ".join(str(int(x) + 1) for x in f) + "\n")
    actor_config = {"schema": "independent-rigid-evaluation-v1", "target_frame": registration["target_frame"],
        "robot": {"uid": "panda", "base_pose": np.eye(4).tolist(), "root_frame": registration["target_frame"]},
        "object": {"mesh": "reference.obj", "initial_pose": data["poses"][0].tolist(),
                   "collision": "convex_hull_of_reference_mesh", "density_kg_m3": 500., "friction": 1.},
        "support": {"type": "plane", "normal": data["support_normal"].tolist(), "offset": float(data["support_offset"]), "friction": 1.},
        "goal_pose": data["poses"][-1].tolist(), "physics_parameters_assumed": True,
        "support_must_not_be_replaced_by_z0": True, "initial_pose_must_not_be_snapped": True,
        "description": "Recorded RGB-D reconstruction proxy, not exact real-world ground truth"}
    write(output / "actor-config.json", actor_config)
    audit = {"schema": "independent-reference-coordinate-audit-v1", "evaluation_only": True,
        "submission_sha256": file_sha256(submission), "deployment_sha256": file_sha256(deployment),
        "raw_reference_sha256": file_sha256(raw_reference), "source_cameras_sha256": file_sha256(source_cameras),
        "source_camera_id": source_camera_id, "source_view": source_view, "source_convention": source_convention,
        "T_robot_recording": data["T_robot_recording"].tolist(), "published_source_c2w": C.tolist(),
        "mapping": "T_robot_source @ inverse(published_source_c2w)", "reference_is_exact_gt": False,
        "fitted_alignment": False, "support_snap": False,
        "deployment_hash_must_match_across_compared_methods": file_sha256(deployment),
        "initial_support_gap_m": float(data["support_minimum_m"][0]), "per_frame_support_minimum_m": data["support_minimum_m"].tolist(),
        "canonicalization_max_surface_error_m": data["canonicalization_max_surface_error_m"],
        "shared_shape_variation_m": data["shared_shape_variation_m"], "script_sha256": file_sha256(__file__),
        "limitation": "Reference camera/geometry/depth are evaluator-only; proxy geometry and supplied material properties have error"}
    write(output / "audit.json", audit)
    artifacts = []
    for aid, path in [("raw_reference", raw_reference), ("published_source_camera", source_cameras),
                      ("reference_scene", output / "scene.json"), ("fixed_fixture", output / "fixture.npz"),
                      ("reference_mesh", output / "reference.obj"), ("actor_config", output / "actor-config.json"),
                      ("coordinate_audit", output / "audit.json")]:
        path = Path(path).resolve()
        artifacts.append({"id": aid, "path": str(path), "sha256": file_sha256(path)})
    write(output / "evaluation-spec.json", {"schema_version": 1, "independent_scene": True,
          "reference_kind": "recorded_rgbd_reconstruction_proxy", "artifacts": artifacts})
    attach_evaluation(submission=submission, evaluation_spec=output / "evaluation-spec.json", receipt=output / "evaluation-attachment.json")
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("submission", "raw-reference", "source-cameras", "deployment", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-view", type=int, required=True)
    p.add_argument("--source-camera-id", required=True)
    p.add_argument("--source-convention", choices=("opencv",), default="opencv")
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))
