"""Joint block reconstruction with a global nonpenetrating-table feasible set.

The learned table is fixed, shared shape changes with all poses, and clearance is
nonnegative in EVERY frame during optimization. This is not an endpoint edit or
a planner correction. A wrong estimated table can still produce wrong geometry.
"""
import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import file_sha256, validate_manifest
from artirecon.scene_sequence import load_scene, validate_scene
from benchmarks import source_geometry as base
from benchmarks import source_joint_fit as joint
from benchmarks.static_depth_calibration import materialize_recipe


def default_config(point_policy="object_rank50"):
    return dict(schema_version=1, feasible_set="shared_table_nonpenetration_all_frames",
                initialization="project_negative_clearance_only_before_optimization",
                pose_trust_region="table_tangent_coordinates_and_nonnegative_clearance",
                joint_config=dict(joint.default_config(), point_policy=point_policy))


def validate_config(config):
    expected = default_config()
    if set(config) != set(expected): raise ValueError("Use the complete predeclared nonpenetration config")
    for key in expected:
        if key != "joint_config" and config[key] != expected[key]: raise ValueError(f"Unsupported constraint choice: {key}")
    joint.validate_config(config["joint_config"])


def table_basis(normal):
    axis = np.array([1., 0., 0.]) - normal * normal[0]
    if np.linalg.norm(axis) < 1e-6: axis = np.array([0., 1., 0.]) - normal * normal[1]
    axis /= np.linalg.norm(axis)
    return np.stack([axis, np.cross(normal, axis), normal], axis=1)


def components(half, positions, *, scene, observations, config):
    """The same five residual families/weights as source_joint_fit v1."""
    rotation = np.asarray(scene["objects"][0]["T_world_object"])[0, :3, :3]
    initial_half = np.asarray(scene["objects"][0]["geometry"]["half_extents_m"])
    normal, offset = np.asarray(scene["support"]["normal_world"]), scene["support"]["offset_m"]
    times = np.asarray(observations["timestamps"])
    corners = np.array(list(product((-1., 1.), repeat=3)))
    residuals = dict(depth=[], silhouette=[], support=[], acceleration=[], shape=[])
    raw_sdf, bbox_absolute, support_absolute, accelerations = [], [], [], []
    per_frame = []
    for frame, points in enumerate(observations["points"]):
        local = (points - positions[frame]) @ rotation
        difference = abs(local) - half
        sdf = np.linalg.norm(np.maximum(difference, 0), axis=1) + np.minimum(np.max(difference, axis=1), 0)
        point_weight = np.sqrt(observations["weights"][frame] * config["max_points_per_frame"] / max(1, len(points)))
        depth_loss = sdf * point_weight / config["depth_sigma_m"]
        world_corners = (corners * half) @ rotation.T + positions[frame]
        uvw = world_corners @ observations["K"].T
        uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], .05)
        bbox_error = np.r_[uv.min(0), uv.max(0)] - observations["bounds"][frame]
        if observations["mask_visible"][frame] and not observations["occluded"][frame]: bbox_absolute.extend(abs(bbox_error))
        if observations["occluded"][frame]: bbox_error = np.r_[np.maximum(bbox_error[:2], 0), np.minimum(bbox_error[2:], 0)]
        silhouette_loss = bbox_error / config["silhouette_sigma_px"] if observations["mask_visible"][frame] else np.zeros(4)
        bottom = normal @ positions[frame] + offset - abs(normal @ rotation) @ half
        support_loss = bottom / config["support_sigma_m"] if observations["support"][frame] else 0.
        per_frame.extend(np.r_[depth_loss, silhouette_loss, support_loss])
        residuals["depth"].extend(depth_loss); residuals["silhouette"].extend(silhouette_loss)
        residuals["support"].append(support_loss); raw_sdf.extend(abs(sdf))
        if observations["support"][frame]: support_absolute.append(abs(bottom))
    for frame in range(1, len(positions) - 1):
        dt0, dt1 = times[frame] - times[frame - 1], times[frame + 1] - times[frame]
        acceleration = ((positions[frame + 1] - positions[frame]) / dt1 - (positions[frame] - positions[frame - 1]) / dt0) / ((dt0 + dt1) / 2)
        residuals["acceleration"].extend(acceleration / config["acceleration_sigma_m_s2"])
        accelerations.append(np.linalg.norm(acceleration))
    residuals["shape"] = ((np.log(half) - np.log(initial_half)) / config["shape_log_sigma"]).tolist()
    values = np.r_[per_frame, residuals["acceleration"], residuals["shape"]]
    bottom = positions @ normal + offset - abs(normal @ rotation) @ half
    summary = dict(dimensions_m=(2 * half).tolist(), bottom_height_m=bottom.tolist(),
                   endpoint_bottom_m=float(bottom[-1]), maximum_penetration_m=float(max(0., -bottom.min())),
                   depth_surface_absolute_mean_m=float(np.mean(raw_sdf)) if raw_sdf else None,
                   unoccluded_bbox_absolute_error_mean_px=float(np.mean(bbox_absolute)) if bbox_absolute else None,
                   rgb_support_absolute_residual_mean_m=float(np.mean(support_absolute)) if support_absolute else None,
                   acceleration_norm_mean_m_s2=float(np.mean(accelerations)) if accelerations else None,
                   normalized_soft_l1_cost={key: float(np.sum(np.sqrt(1 + np.asarray(value)**2) - 1)) for key, value in residuals.items()})
    return values, summary


def fit_shared_scene(scene, observations, config):
    validate_config(config); validate_scene(scene, require_metric_scale=True)
    params = config["joint_config"]
    if len(scene["objects"]) != 1 or scene["objects"][0]["id"] != "block" or scene["objects"][0]["geometry"]["type"] != "box":
        raise ValueError("Single shared block cuboid only")
    obj = scene["objects"][0]
    poses = np.asarray(obj["T_world_object"], float)
    if not np.allclose(poses[:, :3, :3], poses[0, :3, :3], atol=1e-6): raise ValueError("Fixed-orientation baseline only")
    if not np.array_equal(observations["timestamps"], scene["timestamps"]): raise ValueError("Timestamp mismatch")
    frames = len(poses); normal = np.asarray(scene["support"]["normal_world"])
    offset = float(scene["support"]["offset_m"]); basis = table_basis(normal)
    extent_weights = abs(normal @ poses[0, :3, :3])
    initial_half = np.asarray(obj["geometry"]["half_extents_m"])
    initial_centers = poses[:, :3, 3]
    initial_coordinates = initial_centers @ basis
    original_clearance = initial_coordinates[:, 2] + offset - extent_weights @ initial_half
    initial_coordinates[:, 2] = np.maximum(original_clearance, 0.)

    def decode(x):
        half = np.exp(x[:3]); coordinates = x[3:].reshape(frames, 3).copy()
        coordinates[:, 2] += extent_weights @ half - offset
        return half, coordinates @ basis.T

    initial = np.r_[np.log(initial_half), initial_coordinates.ravel()]
    adjustment = params["maximum_pose_adjustment_m"]
    lower_pose, upper_pose = initial_coordinates - adjustment, initial_coordinates + adjustment
    lower_pose[:, 2] = 0.
    lower = np.r_[np.repeat(np.log(params["minimum_extent_m"] / 2), 3), lower_pose.ravel()]
    upper = np.r_[np.repeat(np.log(params["maximum_extent_m"] / 2), 3), upper_pose.ravel()]
    nres = sum(len(p) + 5 for p in observations["points"]) + 3 * max(frames - 2, 0) + 3
    sparsity = lil_matrix((nres, len(initial)), dtype=int); cursor = 0
    for frame, points in enumerate(observations["points"]):
        count = len(points) + 5
        sparsity[cursor:cursor + count, :3] = 1
        sparsity[cursor:cursor + count, 3 + frame * 3:6 + frame * 3] = 1
        cursor += count
    for frame in range(1, frames - 1):
        sparsity[cursor:cursor + 3, :3] = 1
        sparsity[cursor:cursor + 3, 3 + (frame - 1) * 3:3 + (frame + 2) * 3] = 1
        cursor += 3
    sparsity[cursor:cursor + 3, :3] = 1
    def evaluate(x):
        half, positions = decode(x)
        return components(half, positions, scene=scene, observations=observations, config=params)[0]
    started = time.monotonic()
    result = least_squares(evaluate, initial, jac_sparsity=sparsity.tocsr(), bounds=(lower, upper),
                           loss=params["robust_loss"], f_scale=1., max_nfev=params["max_nfev"], xtol=1e-6, ftol=1e-6, gtol=1e-6)
    fit_seconds = time.monotonic() - started
    half, positions = decode(result.x)
    output = deepcopy(scene); fitted_poses = poses.copy(); fitted_poses[:, :3, 3] = positions
    output["objects"][0]["T_world_object"] = fitted_poses.tolist()
    output["objects"][0]["geometry"]["half_extents_m"] = half.tolist()
    output["objects"][0]["observed"] = [row["observed"] for row in observations["evidence"]]
    method = "Joint shared cuboid/poses constrained above one estimated table in all frames"
    output["objects"][0]["provenance"]["method"] = method
    output["objects"][0]["geometry"]["provenance"]["method"] = method
    output["contacts"] = [c for c in output["contacts"] if c["kind"] != "support"]
    for index, (first, last) in enumerate(base.intervals(observations["support"])):
        output["contacts"].append(dict(id=f"rgb-support-{index}", a="block", b="table", kind="support", start_frame=first, end_frame=last,
            status="hypothesized", confidence=None, provenance=dict(method="Stationary source RGB object and distant hand masks, independent of 3D height", artifact_ids=["sam3_masks"], confidence_status="uncalibrated")))
    output["joint_fit"] = dict(point_policy=params["point_policy"], shared_table_nonpenetration=True, enforced_in_optimization=True,
        endpoint_clamped=False, postfit_pose_changes=False, simulator_feedback_used=False, full_orientation_tracking=False)
    validate_scene(output, require_metric_scale=True)
    def summary(h, p): return components(h, p, scene=scene, observations=observations, config=params)[1]
    report = dict(status="fitted", optimizer_success=bool(result.success), optimizer_message=result.message, evaluations=result.nfev,
        fit_seconds=fit_seconds, final_cost=float(result.cost), config=config, before=summary(initial_half, initial_centers),
        feasible_initialization=summary(*decode(initial)), after=summary(half, positions),
        initial_projection_frames=np.flatnonzero(original_clearance < 0).tolist(), initial_projection_distance_m=np.maximum(-original_clearance, 0).tolist(),
        initial_projection_only_for_optimizer_feasibility=True, pose_trust_region="Same 0.15m radius now in table-tangent coordinates and clearance; not world-axis bounds",
        observed_frames=sum(row["observed"] for row in observations["evidence"]), frames=frames, evidence=observations["evidence"],
        support_frame_indices=np.flatnonzero(observations["support"]).tolist(), GT_accuracy_measured=False, simulator_accessed=False,
        endpoint_specific_operation=False, losses_unchanged_from_joint_v1=True,
        limitations=["Nonpenetration is relative to a model-estimated plane, not ground-truth geometry.",
            "Support remains a soft RGB hypothesis; clearance zero is not proof of contact.", "Shape and fixed orientation remain approximations; hidden geometry is unvalidated."])
    return output, report


def produce(*, initializer_bundle, config_path, output, comparison_bundle=None):
    bundle, output = Path(initializer_bundle).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    manifest = base._materialized_manifest(bundle); active = {a["id"]: a for a in manifest["artifacts"]}
    current = [a["id"] for a in active.values() if a["kind"] == "object_geometry" and a.get("path") == str(bundle / "scene-sequence.json")]
    if len(current) != 1: raise ValueError("Bind exactly one initializer scene artifact")
    config = json.loads(Path(config_path).read_text()); validate_config(config)
    geometry_record = active["predicted_geometry"]
    if geometry_record.get("raw_geometry_artifact_id"):
        geometry = materialize_recipe(active[geometry_record["raw_geometry_artifact_id"]]["path"], geometry_record["path"])
    else:
        with np.load(geometry_record["path"], allow_pickle=False) as source: geometry = {name: source[name].copy() for name in source.files}
    scene = load_scene(bundle / "scene-sequence.json", require_metric_scale=True)
    with np.load(active["sam3_masks"]["path"], allow_pickle=False) as masks:
        observations = joint.prepare_evidence(geometry, masks, scene, config["joint_config"])
    fitted, report = fit_shared_scene(scene, observations, config)
    if comparison_bundle is not None:
        comparison_bundle = Path(comparison_bundle).resolve()
        prior_manifest = base._materialized_manifest(comparison_bundle)
        prior_records = {a["id"]: a for a in prior_manifest["artifacts"]}
        for aid in ("source_rgb", "predicted_geometry", "predicted_cameras", "sam3_masks"):
            if active[aid]["sha256"] != prior_records[aid]["sha256"]: raise ValueError("Comparison source evidence differs")
        prior = load_scene(comparison_bundle / "scene-sequence.json", require_metric_scale=True)
        prior_config = json.loads((comparison_bundle / "diagnostics.json").read_text())["config"]
        if prior_config != config["joint_config"]: raise ValueError("Comparison losses/config differ")
        p = prior["objects"][0]
        report["unconstrained_baseline"] = components(np.asarray(p["geometry"]["half_extents_m"]), np.asarray(p["T_world_object"])[:, :3, 3],
            scene=scene, observations=observations, config=config["joint_config"])[1]
        report["unconstrained_scene_sha256"] = file_sha256(comparison_bundle / "scene-sequence.json")
    output.mkdir(parents=True)
    path = output / "scene-sequence.json"; path.write_text(json.dumps(fitted, indent=2, allow_nan=False))
    manifest["artifacts"].append(dict(id="nonpenetrating_joint_fit", kind="object_geometry", origin="model_prediction", parent_ids=current + ["predicted_geometry", "predicted_cameras", "sam3_masks"], path="scene-sequence.json", sha256=file_sha256(path)))
    manifest["method_inputs"].append("nonpenetrating_joint_fit")
    report["input_protocol"] = validate_manifest(manifest, output, verify_files=True)
    report["producer_sha256"] = file_sha256(__file__); report["joint_evidence_producer_sha256"] = file_sha256(joint.__file__)
    report["config_sha256"] = file_sha256(config_path)
    (output / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (output / "fit-config.json").write_text(json.dumps(config, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); subs = parser.add_subparsers(dest="mode", required=True)
    cfg = subs.add_parser("config"); cfg.add_argument("--point-policy", choices=["all", "object_rank50", "soft_rank"], required=True)
    cfg.add_argument("--output", type=Path, required=True)
    run = subs.add_parser("run")
    for name in ("initializer-bundle", "config-path", "output"): run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--comparison-bundle", type=Path)
    args = vars(parser.parse_args())
    if args.pop("mode") == "config":
        with args["output"].open("x") as stream: json.dump(default_config(args["point_policy"]), stream, indent=2)
    else: print(json.dumps(produce(**args), indent=2))


if __name__ == "__main__": main()
