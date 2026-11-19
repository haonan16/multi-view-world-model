"""Joint shared-cuboid/translation fitting from RGB masks and estimated depth.

Support evidence is a soft RGB-stationarity/distant-hand hypothesis, computed
without consulting current 3D heights. No simulator or evaluation geometry is
accepted. This remains a fixed-orientation block baseline, not SAM3D or MANO.
"""
import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import file_sha256, validate_manifest
from artirecon.scene_sequence import load_scene, validate_scene
from benchmarks import source_geometry as base


def default_config():
    return dict(schema_version=1, point_policy="object_rank50", max_points_per_frame=64,
                minimum_points=16, confidence_floor=.1, depth_sigma_m=.04,
                silhouette_sigma_px=2., support_sigma_m=.015, shape_log_sigma=.7,
                acceleration_sigma_m_s2=2., stationarity_window_s=.4,
                stationarity_radius_px=2., distant_hand_px=12., occlusion_gap_px=6.,
                minimum_support_observations=3, maximum_pose_adjustment_m=.15,
                minimum_extent_m=.004, maximum_extent_m=.5, max_nfev=80,
                random_seed=0, robust_loss="soft_l1")


def validate_config(config):
    if set(config) != set(default_config()) or config["schema_version"] != 1:
        raise ValueError("Use the complete preregistered joint-fit config")
    if config["point_policy"] not in {"all", "object_rank50", "soft_rank"} or config["robust_loss"] != "soft_l1":
        raise ValueError("Unknown point policy or robust loss")
    for key, value in config.items():
        if key in {"schema_version", "point_policy", "random_seed", "robust_loss"}:
            continue
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid config value: {key}")
    if config["confidence_floor"] > 1:
        raise ValueError("confidence_floor cannot exceed one")


def rgb_support_hypotheses(centroids, hand_gaps, timestamps, visible, config):
    """Stationary source masks + distant hands; never reads depth or 3D poses."""
    centroids, hand_gaps, times, visible = map(np.asarray, (centroids, hand_gaps, timestamps, visible))
    flags = np.zeros(len(times), bool)
    for frame in range(len(times)):
        neighbors = np.flatnonzero(abs(times - times[frame]) <= config["stationarity_window_s"] / 2 + 1e-8)
        if (visible[frame] and np.isfinite(hand_gaps[frame]) and hand_gaps[frame] >= config["distant_hand_px"]
                and len(neighbors) >= config["minimum_support_observations"] and visible[neighbors].all()
                and np.max(np.linalg.norm(centroids[neighbors] - centroids[frame], axis=1)) <= config["stationarity_radius_px"]):
            flags[frame] = True
    return flags


def prepare_evidence(geometry, masks, source_scene, config):
    validate_config(config)
    depth, k, times = base.validate_geometry(geometry)
    ranked = base.confidence_available(geometry)
    if not ranked and config["point_policy"] != "all":
        raise ValueError("Rank policies require model confidence; neutral weights are not confidence")
    if not np.array_equal(times, source_scene["timestamps"]):
        raise ValueError("Initializer and new model observations have different timestamps")
    candidates = base.MaskCandidates(masks, len(depth), depth.shape[-2:], times)
    selected, association = base.associate_masks(candidates, len(depth), base.default_config())
    rng = np.random.default_rng(config["random_seed"])
    ranks = np.asarray(geometry["confidence_rank"])[:, 0]
    depth_valid = np.asarray(geometry["valid"], bool)[:, 0] & np.isfinite(depth) & (depth > 0)
    clouds, weights, bounds, centers, gaps, evidence = [], [], [], [], [], []
    for frame, row in enumerate(selected):
        object_mask = candidates.mask(row) if row is not None else np.zeros(depth.shape[-2:], bool)
        hand_mask = candidates.union(candidates.rows(frame, "hand", .3))
        visible = object_mask.any()
        if visible:
            yy, xx = np.where(object_mask)
            bounds.append([xx.min(), yy.min(), xx.max(), yy.max()])
            centers.append([xx.mean(), yy.mean()])
        else:
            bounds.append([0., 0., 0., 0.]); centers.append([0., 0.])
        gap = float(distance_transform_edt(~hand_mask)[object_mask].min()) if visible and hand_mask.any() else np.nan
        gaps.append(gap)
        mask = object_mask & ~hand_mask
        mask &= distance_transform_edt(mask) > 1
        mask &= depth_valid[frame]
        yy, xx = np.where(mask)
        point_ranks = ranks[frame, yy, xx]
        if config["point_policy"] == "object_rank50" and len(xx):
            selected_points = point_ranks >= np.median(point_ranks)
            xx, yy, point_ranks = xx[selected_points], yy[selected_points], point_ranks[selected_points]
        available = len(xx)
        if len(xx) > config["max_points_per_frame"]:
            sample = np.sort(rng.choice(len(xx), config["max_points_per_frame"], replace=False))
            xx, yy, point_ranks = xx[sample], yy[sample], point_ranks[sample]
        rays = np.c_[xx, yy, np.ones(len(xx))] @ np.linalg.inv(k).T
        points = rays * depth[frame, yy, xx, None]
        point_weights = (config["confidence_floor"] + (1 - config["confidence_floor"]) * point_ranks
                         if config["point_policy"] == "soft_rank" else np.ones(len(points)))
        clouds.append(points); weights.append(point_weights)
        evidence.append(dict(frame=frame, observed=bool(available >= config["minimum_points"]),
                             selected_candidate_row=row, available_points=available, sampled_points=len(points),
                             mean_global_rank=float(point_ranks.mean()) if ranked and len(points) else None,
                             weak_depth_evidence=bool(not len(points) or point_ranks.mean() < .5) if ranked else None,
                             confidence_available=ranked,
                             hand_gap_px=gap if np.isfinite(gap) else None, confidence_is_calibrated_probability=False))
    visible = np.array([row is not None for row in selected])
    support = rgb_support_hypotheses(centers, gaps, times, visible, config)
    return dict(points=clouds, weights=weights, bounds=np.array(bounds),
                occluded=(np.array(gaps) <= config["occlusion_gap_px"]) | ~np.isfinite(gaps),
                mask_visible=visible, support=support, K=k, timestamps=times, evidence=evidence, association=association)


def fit_shared_scene(scene, observations, config):
    """Numerical joint optimization; no frame-wise table clamp or shape recreation."""
    validate_config(config)
    validate_scene(scene, require_metric_scale=True)
    if len(scene["objects"]) != 1 or scene["objects"][0]["geometry"]["type"] != "box" or scene["objects"][0]["id"] != "block":
        raise ValueError("Joint baseline supports one shared block cuboid")
    obj = scene["objects"][0]
    poses = np.asarray(obj["T_world_object"], float)
    if not np.allclose(poses[:, :3, :3], poses[0, :3, :3], atol=1e-6):
        raise ValueError("This baseline does not optimize object rotation")
    frames, rotation = len(poses), poses[0, :3, :3]
    initial_half = np.asarray(obj["geometry"]["half_extents_m"], float)
    initial_centers = poses[:, :3, 3]
    normal = np.asarray(scene["support"]["normal_world"], float)
    offset = float(scene["support"]["offset_m"])
    times = np.asarray(observations["timestamps"], float)
    if not np.array_equal(times, scene["timestamps"]):
        raise ValueError("Actual observation timestamps differ from scene")
    corners = np.array(list(product((-1., 1.), repeat=3)))
    support_extent_weights = abs(normal @ rotation)
    nres = sum(len(p) + 4 + 1 for p in observations["points"]) + 3 * max(0, frames - 2) + 3
    sparsity = lil_matrix((nres, 3 + 3 * frames), dtype=int)
    cursor = 0
    for frame, points in enumerate(observations["points"]):
        count = len(points) + 5
        sparsity[cursor:cursor + count, :3] = 1
        sparsity[cursor:cursor + count, 3 + frame * 3:6 + frame * 3] = 1
        cursor += count
    for frame in range(1, frames - 1):
        sparsity[cursor:cursor + 3, 3 + (frame - 1) * 3:3 + (frame + 2) * 3] = 1
        cursor += 3
    sparsity[cursor:cursor + 3, :3] = 1

    def residual(parameters):
        half = np.exp(parameters[:3]); positions = parameters[3:].reshape(frames, 3)
        result = []
        for frame, points in enumerate(observations["points"]):
            local = (points - positions[frame]) @ rotation
            difference = abs(local) - half
            sdf = np.linalg.norm(np.maximum(difference, 0), axis=1) + np.minimum(np.max(difference, axis=1), 0)
            point_weight = np.sqrt(observations["weights"][frame] * config["max_points_per_frame"] / max(1, len(points)))
            result.extend(sdf * point_weight / config["depth_sigma_m"])
            world_corners = (corners * half) @ rotation.T + positions[frame]
            uvw = world_corners @ observations["K"].T
            uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], .05)
            predicted = np.r_[uv.min(0), uv.max(0)]
            bbox_error = predicted - observations["bounds"][frame]
            if observations["occluded"][frame]:
                bbox_error = np.r_[np.maximum(bbox_error[:2], 0), np.minimum(bbox_error[2:], 0)]
            result.extend(bbox_error / config["silhouette_sigma_px"] if observations["mask_visible"][frame] else np.zeros(4))
            bottom = normal @ positions[frame] + offset - support_extent_weights @ half
            result.append(bottom / config["support_sigma_m"] if observations["support"][frame] else 0.)
        for frame in range(1, frames - 1):
            dt0, dt1 = times[frame] - times[frame - 1], times[frame + 1] - times[frame]
            acceleration = ((positions[frame + 1] - positions[frame]) / dt1 - (positions[frame] - positions[frame - 1]) / dt0) / ((dt0 + dt1) / 2)
            result.extend(acceleration / config["acceleration_sigma_m_s2"])
        result.extend((np.log(half) - np.log(initial_half)) / config["shape_log_sigma"])
        return np.asarray(result)

    initial = np.r_[np.log(initial_half), initial_centers.ravel()]
    adjustment = config["maximum_pose_adjustment_m"]
    lower = np.r_[np.repeat(np.log(config["minimum_extent_m"] / 2), 3), (initial_centers - adjustment).ravel()]
    upper = np.r_[np.repeat(np.log(config["maximum_extent_m"] / 2), 3), (initial_centers + adjustment).ravel()]
    started = time.monotonic()
    result = least_squares(residual, initial, jac_sparsity=sparsity.tocsr(), bounds=(lower, upper),
                           loss=config["robust_loss"], f_scale=1., max_nfev=config["max_nfev"],
                           xtol=1e-6, ftol=1e-6, gtol=1e-6)
    output = deepcopy(scene)
    fitted_half = np.exp(result.x[:3]); fitted_centers = result.x[3:].reshape(frames, 3)
    fitted_poses = poses.copy(); fitted_poses[:, :3, 3] = fitted_centers
    output["objects"][0]["geometry"]["half_extents_m"] = fitted_half.tolist()
    output["objects"][0]["T_world_object"] = fitted_poses.tolist()
    output["objects"][0]["observed"] = [row["observed"] for row in observations["evidence"]]
    output["objects"][0]["provenance"]["method"] = "Joint shared cuboid/translation fit from source masks, predicted depth and uncertain RGB support"
    output["objects"][0]["geometry"]["provenance"]["method"] = output["objects"][0]["provenance"]["method"]
    output["contacts"] = [c for c in output["contacts"] if c["kind"] != "support"]
    for index, (first, last) in enumerate(base.intervals(observations["support"])):
        output["contacts"].append(dict(id=f"rgb-support-{index}", a="block", b="table", kind="support",
                                       start_frame=first, end_frame=last, status="hypothesized", confidence=None,
                                       provenance={"method": "Stationary RGB object mask and distant RGB hand masks; independent of 3D height",
                                                   "artifact_ids": ["sam3_masks"], "confidence_status": "uncalibrated"}))
    output["initial_support_prior"]["per_frame_support_corrections_applied"] = False
    output["joint_fit"] = dict(point_policy=config["point_policy"], uncertainty_calibrated=False,
                               support_is_soft_hypothesis=True, endpoint_clamped=False,
                               full_orientation_tracking=False, simulator_feedback_used=False)
    validate_scene(output, require_metric_scale=True)

    def diagnostics(half, positions):
        bottom = positions @ normal + offset - support_extent_weights @ half
        bbox_errors = []
        for frame in range(frames):
            projected = ((corners * half) @ rotation.T + positions[frame]) @ observations["K"].T
            uv = projected[:, :2] / projected[:, 2:3]
            bbox = np.r_[uv.min(0), uv.max(0)]
            if observations["mask_visible"][frame] and not observations["occluded"][frame]:
                bbox_errors.extend(abs(bbox - observations["bounds"][frame]))
        return dict(dimensions_m=(2 * half).tolist(), bottom_height_m=bottom.tolist(),
                    endpoint_bottom_m=float(bottom[-1]), maximum_penetration_m=float(max(0, -bottom.min())),
                    rgb_support_absolute_residual_mean_m=float(np.mean(abs(bottom[observations["support"]]))) if observations["support"].any() else None,
                    unoccluded_bbox_absolute_error_mean_px=float(np.mean(bbox_errors)) if bbox_errors else None)
    report = dict(status="fitted", optimizer_success=bool(result.success), optimizer_message=result.message,
                  evaluations=result.nfev, fit_seconds=time.monotonic() - started, final_cost=float(result.cost),
                  config=config, before=diagnostics(initial_half, initial_centers), after=diagnostics(fitted_half, fitted_centers),
                  observed_frames=sum(row["observed"] for row in observations["evidence"]), frames=frames,
                  support_frame_indices=np.flatnonzero(observations["support"]).tolist(), evidence=observations["evidence"],
                  support_evidence_uses_3d_height=False, GT_accuracy_measured=False, simulator_accessed=False,
                  limitations=["Metrics are source silhouette and estimated-support consistency, not GT shape or execution accuracy.",
                               "Box orientation remains fixed; hidden shape and physical grasp quality remain uncertain.",
                               "RGB support is an uncertain tabletop assumption, not measured contact.",
                               "Raw touch contacts are retained as hypotheses; no new continuous grasp is manufactured."])
    return output, report


def produce(*, initializer_bundle, config_path, output):
    bundle, output = Path(initializer_bundle).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(Path(config_path).read_text()); validate_config(config)
    manifest = base._materialized_manifest(bundle)
    artifacts = {a["id"]: a for a in manifest["artifacts"]}
    if Path(artifacts["source_block_initializer"]["path"]) != bundle / "scene-sequence.json":
        raise ValueError("Initializer scene is not bound to the materialized RGB lineage")
    scene = load_scene(bundle / "scene-sequence.json", require_metric_scale=True)
    with np.load(artifacts["predicted_geometry"]["path"], allow_pickle=False) as depth, np.load(artifacts["sam3_masks"]["path"], allow_pickle=False) as masks:
        observations = prepare_evidence(depth, masks, scene, config)
    fitted, report = fit_shared_scene(scene, observations, config)
    output.mkdir(parents=True)
    scene_path = output / "scene-sequence.json"
    scene_path.write_text(json.dumps(fitted, indent=2, allow_nan=False))
    (output / "fit-config.json").write_text(json.dumps(config, indent=2))
    manifest["artifacts"].append(dict(id="source_joint_fit", kind="object_geometry", origin="model_prediction",
                                      parent_ids=["source_block_initializer", "predicted_geometry", "predicted_cameras", "sam3_masks", "estimated_support"],
                                      path="scene-sequence.json", sha256=file_sha256(scene_path)))
    manifest["method_inputs"].append("source_joint_fit")
    report["input_protocol"] = validate_manifest(manifest, output, verify_files=True)
    report["producer_sha256"] = file_sha256(__file__)
    report["config_sha256"] = file_sha256(output / "fit-config.json")
    (output / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    config = modes.add_parser("config"); config.add_argument("--point-policy", choices=["all", "object_rank50", "soft_rank"], required=True)
    config.add_argument("--output", type=Path, required=True)
    run = modes.add_parser("run")
    for name in ("initializer-bundle", "config-path", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    if args.pop("mode") == "config":
        with args["output"].open("x") as stream: json.dump(dict(default_config(), point_policy=args["point_policy"]), stream, indent=2)
    else:
        report = produce(**args)
        print(json.dumps({k: report[k] for k in ("status", "optimizer_success", "evaluations", "fit_seconds", "observed_frames", "before", "after")}, indent=2))


if __name__ == "__main__":
    main()
