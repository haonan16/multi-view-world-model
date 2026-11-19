"""Source-only static-table calibration of temporal monocular depth scale.

Frame zero fixes the learned metric gauge. One multiplicative depth scale per
frame is estimated from background pixels, never object poses or simulator
outcomes. Raw predictions remain untouched; a compact correction recipe is saved.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import file_sha256, validate_manifest
from artirecon.scene_sequence import object_from_arrays, scene_from_arrays
from benchmarks import source_geometry as base
from benchmarks import source_joint_fit as joint


def default_config():
    return dict(schema_version=1, reference_frame=0, consensus_fraction=.5,
                minimum_region_pixels=2500, fit_points_per_frame=1000,
                heldout_points_per_frame=1000, exclusion_dilation_px=2,
                minimum_scale=.8, maximum_scale=1.25, maximum_rgb_median_delta=20.,
                maximum_heldout_plane_median_m=.01, random_seed=0,
                static_camera_assumption=True, correction="multiplicative_only")


def validate_config(config):
    if set(config) != set(default_config()) or config["schema_version"] != 1:
        raise ValueError("Use the full preregistered static-depth configuration")
    if config["reference_frame"] != 0 or config["correction"] != "multiplicative_only" or config["static_camera_assumption"] is not True:
        raise ValueError("This variant fixes frame-zero gauge and uses scale only")
    if not 0 < config["consensus_fraction"] <= 1 or not 0 < config["minimum_scale"] < 1 < config["maximum_scale"]:
        raise ValueError("Invalid consensus/scale bounds")


def materialize_recipe(raw_geometry_path, recipe_path):
    """Explicitly reconstruct arrays in memory; a recipe is not geometry.npz."""
    with np.load(recipe_path, allow_pickle=False) as recipe:
        if recipe["recipe_kind"].item() != "multiplicative_depth_scale_v1":
            raise ValueError("Unknown depth recipe")
        if file_sha256(raw_geometry_path) != recipe["raw_geometry_sha256"].item():
            raise ValueError("Raw depth changed after calibration")
        scales = recipe["scale"].copy()
        times = recipe["timestamps"].copy()
    with np.load(raw_geometry_path, allow_pickle=False) as raw:
        data = {name: raw[name].copy() for name in ("depth_m", "K", "c2w", "timestamps", "valid", "confidence_rank")}
        if "confidence_available" in raw:
            data["confidence_available"] = raw["confidence_available"].copy()
    if scales.shape != (len(data["timestamps"]),) or not np.array_equal(times, data["timestamps"]):
        raise ValueError("Recipe timestamps/scales mismatch raw geometry")
    if not np.isfinite(scales).all() or np.any(scales <= 0) or scales[0] != 1:
        raise ValueError("Invalid multiplicative scales/frame-zero gauge")
    data["depth_m"] *= scales[:, None, None, None]
    return data


def initialize_on_reference_plane(geometry, masks, normal, offset):
    """Fresh shared shape/translation seed using the SAME frame-zero plane.

    This repeats the v1 cuboid prior with a predeclared model-estimated plane,
    instead of refitting another plane and overwriting it after shape creation.
    """
    config = base.default_config()
    depth, k, times = base.validate_geometry(geometry)
    candidates = base.MaskCandidates(masks, len(times), depth.shape[-2:], times)
    selected, association = base.associate_masks(candidates, len(times), config)
    normal = np.asarray(normal, float)
    axis = np.array([1., 0., 0.]) - normal * normal[0]
    if np.linalg.norm(axis) < 1e-6: axis = np.array([0., 1., 0.]) - normal * normal[1]
    axis /= np.linalg.norm(axis)
    basis = np.stack([axis, np.cross(normal, axis), normal], axis=1)
    clouds, observed = [], []
    rng = np.random.default_rng(config["random_seed"])
    for frame, row in enumerate(selected):
        mask = candidates.mask(row) if row is not None else np.zeros(depth.shape[-2:], bool)
        mask &= ~candidates.union(candidates.rows(frame, "hand", .3))
        mask &= distance_transform_edt(mask) > config["mask_erosion_pixels"]
        mask &= np.asarray(geometry["valid"])[frame, 0]
        cloud = base.lift(depth[frame], k, mask, config["max_points_per_object"], rng)
        clouds.append(cloud); observed.append(row is not None and len(cloud) >= config["min_object_points"])
    if not observed[0]: raise ValueError("No supported first-frame shape evidence after calibration")
    initial = clouds[0] @ basis
    _, _, axes = np.linalg.svd(initial[:, :2] - np.median(initial[:, :2], axis=0), full_matrices=False)
    primary = axes[0]
    if primary[0] < 0: primary *= -1
    yaw = np.array([[primary[0], -primary[1], 0], [primary[1], primary[0], 0], [0, 0, 1.]])
    orientation = basis @ yaw
    low, high = np.quantile(clouds[0] @ orientation, [config["extent_quantile"], 1 - config["extent_quantile"]], axis=0)
    low[2] = -offset
    extents = high - low
    if np.any(extents < config["minimum_extent_m"]) or np.any(extents > config["maximum_extent_m"]):
        raise ValueError("Unsupported initial cuboid dimensions on frame-zero plane")
    center = orientation @ ((low + high) / 2)
    initial_median = np.median(clouds[0], axis=0)
    poses = np.repeat(np.eye(4)[None], len(times), axis=0); poses[:, :3, :3] = orientation
    previous = center.copy()
    for frame, cloud in enumerate(clouds):
        if observed[frame]: previous = center + np.median(cloud, axis=0) - initial_median
        poses[frame, :3, 3] = previous
    provenance = dict(method="Frame-zero predicted plane + scale-calibrated source depth/SAM3 cuboid initialization",
                      artifact_ids=["predicted_geometry", "predicted_cameras", "sam3_masks"], confidence_status="uncalibrated")
    scene = scene_from_arrays(timestamps=times,
        coordinate_frame=dict(name="source_camera", handedness="right", up_axis="unknown", pose_convention="T_world_local"),
        metric_scale=dict(status="estimated", provenance=provenance), actors=[],
        objects=[object_from_arrays(object_id="block", poses=poses, observed=observed, provenance=provenance,
                                   primitive=dict(type="box", half_extents_m=extents / 2))],
        support=dict(id="table", normal_world=normal, offset_m=offset, confidence=None, provenance=provenance),
        contacts=[], uncertainty=dict(status="estimated", provenance=provenance),
        task=dict(description="Initialize source block after static-background depth calibration", target=None))
    scene["motion_model"] = "shared_cuboid_fixed_orientation_translation_only"
    scene["orientation_estimated_over_time"] = False
    scene["initial_support_prior"] = dict(used=True, frame=0, source="Explicit initial tabletop support hypothesis on frame-zero predicted plane", per_frame_support_corrections_applied=False)
    scene["depth_self_calibration"] = dict(reference_frame=0, additive_offset_used=False, raw_predictions_preserved=True)
    return scene, dict(status="initialized", observed_frames=sum(observed), frames=len(times),
                       association=association, dimensions_m=extents.tolist(), support_plane_refitted=False,
                       initial_plane_matches_shape_prior=True, config=config)


def calibrate(geometry, masks, rgb, config):
    validate_config(config)
    depth, k, times = base.validate_geometry(geometry)
    if rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8:
        raise ValueError("Need the exact native RGB sequence accompanying predicted depth")
    candidates = base.MaskCandidates(masks, len(times), depth.shape[-2:], times)
    vote = np.zeros(depth.shape[-2:], np.int32)
    mask_rows = []
    for frame in range(len(times)):
        rows = candidates.rows(frame, "tabletop", .3)
        if len(rows):
            row = max(rows, key=lambda r: (candidates.mask(int(r)).sum(), candidates.score[r], -int(r)))
            vote += candidates.mask(int(row)); mask_rows.append(int(row))
    if not mask_rows:
        raise ValueError("No source-derived table region available for static calibration")
    region = vote >= np.ceil(len(mask_rows) * config["consensus_fraction"])
    if region.sum() < config["minimum_region_pixels"]:
        raise ValueError("Insufficient consensus table region")
    rng = np.random.default_rng(config["random_seed"])
    fitting, testing, rgb_delta, exclusion_fraction = [], [], [], []
    ray_inverse = np.linalg.inv(k)
    exclusions = []
    for frame in range(len(times)):
        excluded = candidates.union(candidates.rows(frame, "hand", .3))
        excluded |= candidates.union(candidates.rows(frame, "wooden block on patterned mat", .3))
        exclusions.append(binary_dilation(excluded, iterations=config["exclusion_dilation_px"]))
    raw_valid = np.asarray(geometry["valid"], bool)[:, 0]
    for frame in range(len(times)):
        mask = region & ~exclusions[frame] & raw_valid[frame] & np.isfinite(depth[frame]) & (depth[frame] > 0)
        yy, xx = np.where(mask)
        needed = config["fit_points_per_frame"] + config["heldout_points_per_frame"]
        if len(xx) < max(needed, config["minimum_region_pixels"]):
            raise ValueError(f"Frame {frame}: insufficient independent background fit/heldout pixels")
        ids = rng.choice(len(xx), needed, replace=False)
        xx, yy = xx[ids], yy[ids]
        points = (np.c_[xx, yy, np.ones(len(xx))] @ ray_inverse.T) * depth[frame, yy, xx, None]
        fitting.append(points[:config["fit_points_per_frame"]])
        testing.append(points[config["fit_points_per_frame"]:])
        stable = region & ~exclusions[frame] & ~exclusions[0]
        if stable.sum() < config["minimum_region_pixels"]:
            raise ValueError(f"Frame {frame}: insufficient static RGB overlap")
        difference = np.mean(abs(rgb[frame].astype(float) - rgb[0].astype(float)), axis=-1)
        rgb_delta.append(float(np.median(difference[stable])))
        exclusion_fraction.append(float(exclusions[frame][region].mean()))
    normal, _, plane_report = base.fit_table(fitting[0], base.default_config())
    # Exactly preserve frame-zero metric scale, using its fixed-normal median.
    offset = -float(np.median(fitting[0] @ normal))
    if offset <= 0:
        raise ValueError("Frame-zero table plane does not lie in front of the camera")
    scales, rows, rejected = [], [], []
    for frame, (fit_points, heldout) in enumerate(zip(fitting, testing)):
        denominator = -float(np.median(fit_points @ normal))
        if denominator <= 0:
            raise ValueError(f"Frame {frame}: invalid plane offset for scale estimation")
        scale = offset / denominator
        if frame == 0: scale = 1.
        scales.append(scale)
        before = heldout @ normal + offset
        after = (heldout * scale) @ normal + offset
        row = dict(frame=frame, timestamp_s=float(times[frame]), scale=float(scale),
                   fit_points=len(fit_points), heldout_points=len(heldout),
                   raw_heldout_signed_median_m=float(np.median(before)),
                   raw_heldout_absolute_median_m=float(np.median(abs(before))),
                   corrected_heldout_signed_median_m=float(np.median(after)),
                   corrected_heldout_absolute_median_m=float(np.median(abs(after))),
                   static_rgb_median_delta=rgb_delta[frame], dynamic_exclusion_fraction=exclusion_fraction[frame])
        if not config["minimum_scale"] <= scale <= config["maximum_scale"]:
            rejected.append(dict(frame=frame, reason="scale outside fixed bounds"))
        if rgb_delta[frame] > config["maximum_rgb_median_delta"]:
            rejected.append(dict(frame=frame, reason="source RGB static-background check failed"))
        if row["corrected_heldout_absolute_median_m"] > config["maximum_heldout_plane_median_m"]:
            rejected.append(dict(frame=frame, reason="held-out background residual above criterion"))
        rows.append(row)
    scales = np.asarray(scales, np.float32)
    corrected = {name: np.asarray(geometry[name]).copy() for name in ("depth_m", "K", "c2w", "timestamps", "valid", "confidence_rank")}
    if "confidence_available" in geometry:
        corrected["confidence_available"] = np.asarray(geometry["confidence_available"]).copy()
    corrected["depth_m"] *= scales[:, None, None, None]
    report = dict(status="accepted" if not rejected else "rejected", config=config, frames=len(times),
                  reference_frame=0, frame_zero_scale=float(scales[0]), reference_normal=normal.tolist(),
                  reference_offset_m=offset, reference_plane_fit=plane_report,
                  source_table_mask_rows=mask_rows, consensus_region_pixels=int(region.sum()),
                  semantic_region="Temporal consensus of largest SAM3 tabletop candidate per source frame",
                  static_mask_reused=True, dynamic_masks_excluded_per_frame=True,
                  RGB_stability_check_is_heuristic=True, rows=rows, rejected_frames=rejected,
                  metric_scale="Frame-zero learned metric gauge; no GT alignment", object_pose_used=False,
                  simulator_accessed=False, raw_depth_changed=False,
                  limitations=["A static camera/table is a capture assumption; RGB stability is not a proof.",
                               "Multiplicative correction cannot remove object-specific errors or all normal drift.",
                               "Held-out residuals assess shared predicted background, not ground-truth accuracy."])
    return corrected, scales, report


def produce(*, depth_bundle, mask_bundle, config_path, joint_config_paths, output):
    depth_bundle, mask_bundle, output = map(lambda x: Path(x).resolve(), (depth_bundle, mask_bundle, output))
    if output.exists(): raise FileExistsError(output)
    config = json.loads(Path(config_path).read_text()); validate_config(config)
    depth_manifest = base._materialized_manifest(depth_bundle)
    mask_manifest = base._materialized_manifest(mask_bundle)
    if depth_manifest["task_mode"] != "reconstruction" or mask_manifest["task_mode"] != "reconstruction":
        raise ValueError("Temporal calibration uses full recorded RGB; not valid as a first-frame forecast")
    depth_records = {a["id"]: a for a in depth_manifest["artifacts"]}
    mask_records = {a["id"]: a for a in mask_manifest["artifacts"]}
    if depth_records["predicted_cameras"]["path"] != depth_records["predicted_geometry"]["path"]:
        raise ValueError("Consumed K must be bound to the same raw geometry archive")
    if depth_records["source_rgb"]["sha256"] != mask_records["source_rgb"]["sha256"]:
        raise ValueError("Depth and masks consumed different source RGB")
    started = time.monotonic()
    with np.load(depth_records["predicted_geometry"]["path"], allow_pickle=False) as depth, \
         np.load(mask_records["sam3_masks"]["path"], allow_pickle=False) as masks, \
         np.load(depth_records["source_rgb"]["path"], allow_pickle=False) as source:
        corrected, scales, report = calibrate(depth, masks, source["rgb"], config)
        # Keep small arrays/candidates in memory for both matched fits.
        mask_data = {key: masks[key].copy() for key in masks.files}
    output.mkdir(parents=True)
    (output / "calibration-config.json").write_text(json.dumps(config, indent=2))
    report.update(seconds=time.monotonic() - started, producer_sha256=file_sha256(__file__),
                  config_sha256=file_sha256(output / "calibration-config.json"))
    (output / "calibration-diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(output / "depth-calibration.npz", scale=scales, timestamps=corrected["timestamps"],
                        normal=np.array(report["reference_normal"]), offset_m=report["reference_offset_m"],
                        raw_geometry_path=depth_records["predicted_geometry"]["path"],
                        raw_geometry_sha256=depth_records["predicted_geometry"]["sha256"],
                        recipe_kind="multiplicative_depth_scale_v1")
    if report["status"] != "accepted":
        return report
    manifest = deepcopy(depth_manifest)
    for artifact in manifest["artifacts"]:
        if artifact["id"] == "predicted_geometry": artifact["id"] = "raw_predicted_geometry"
        artifact["parent_ids"] = ["raw_predicted_geometry" if p == "predicted_geometry" else p for p in artifact["parent_ids"]]
    manifest["method_inputs"] = ["raw_predicted_geometry" if p == "predicted_geometry" else p for p in manifest["method_inputs"]]
    existing = {a["id"] for a in manifest["artifacts"]}
    manifest["artifacts"] += [a for a in mask_manifest["artifacts"] if a["id"] not in existing]
    manifest["artifacts"].append(dict(id="predicted_geometry", kind="depth", origin="model_prediction",
        parent_ids=["raw_predicted_geometry", "predicted_cameras", "sam3_masks", "source_rgb"],
        path=str(output / "depth-calibration.npz"), sha256=file_sha256(output / "depth-calibration.npz"),
        representation="multiplicative_depth_scale_v1: requires static_depth_calibration.materialize_recipe; not a raw geometry archive",
        raw_geometry_artifact_id="raw_predicted_geometry"))
    manifest["method_inputs"] += ["predicted_geometry", "sam3_masks"]
    manifest["metric_scale"] = dict(status="estimated", artifact_ids=["predicted_geometry"], description="Frame-zero learned metric gauge, temporal scales from static source background")
    initial, initializer_report = initialize_on_reference_plane(corrected, mask_data, report["reference_normal"], report["reference_offset_m"])
    init_dir = output / "initializer"; init_dir.mkdir()
    init_path = init_dir / "scene-sequence.json"; init_path.write_text(json.dumps(initial, indent=2, allow_nan=False))
    manifest["artifacts"].append(dict(id="static_calibrated_initializer", kind="object_geometry", origin="model_prediction",
        parent_ids=["predicted_geometry", "predicted_cameras", "sam3_masks"], path=str(init_path), sha256=file_sha256(init_path)))
    manifest["method_inputs"].append("static_calibrated_initializer")
    initializer_report["input_protocol"] = validate_manifest(manifest, init_dir, verify_files=True)
    (init_dir / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
    (init_dir / "diagnostics.json").write_text(json.dumps(initializer_report, indent=2, allow_nan=False))
    results = []
    for joint_config_path in joint_config_paths:
        joint_config = json.loads(Path(joint_config_path).read_text()); joint.validate_config(joint_config)
        observations = joint.prepare_evidence(corrected, mask_data, initial, joint_config)
        fitted, fit_report = joint.fit_shared_scene(initial, observations, joint_config)
        target = output / joint_config["point_policy"]; target.mkdir()
        path = target / "scene-sequence.json"; path.write_text(json.dumps(fitted, indent=2, allow_nan=False))
        final_manifest = deepcopy(manifest)
        final_manifest["artifacts"].append(dict(id="static_calibrated_joint_fit", kind="object_geometry", origin="model_prediction",
            parent_ids=["static_calibrated_initializer", "predicted_geometry", "predicted_cameras", "sam3_masks"],
            path=str(path), sha256=file_sha256(path)))
        final_manifest["method_inputs"].append("static_calibrated_joint_fit")
        fit_report["input_protocol"] = validate_manifest(final_manifest, target, verify_files=True)
        fit_report["joint_fitter_sha256"] = file_sha256(joint.__file__)
        fit_report["joint_config_sha256"] = file_sha256(joint_config_path)
        (target / "input-manifest.json").write_text(json.dumps(final_manifest, indent=2))
        (target / "diagnostics.json").write_text(json.dumps(fit_report, indent=2, allow_nan=False))
        results.append(dict(mode=joint_config["point_policy"], observed_frames=fit_report["observed_frames"],
                            before=fit_report["before"], after=fit_report["after"]))
    report["joint_results"] = results
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    config = modes.add_parser("config"); config.add_argument("--output", type=Path, required=True)
    run = modes.add_parser("run")
    for name in ("depth-bundle", "mask-bundle", "config-path", "output"): run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--joint-config-paths", type=Path, nargs="+", required=True)
    args = vars(parser.parse_args())
    if args.pop("mode") == "config":
        with args["output"].open("x") as stream: json.dump(default_config(), stream, indent=2)
    else:
        report = produce(**args)
        print(json.dumps({"status": report["status"], "rejected_frames": report["rejected_frames"],
                          "joint_results": report.get("joint_results", [])}, indent=2))


if __name__ == "__main__": main()
