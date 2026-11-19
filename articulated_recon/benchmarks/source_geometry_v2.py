"""V2 block initializer with an explicitly labeled dominant-plane fallback.

This is a cuboid/translation baseline, not SAM3D, full orientation tracking or
physical execution. Dataset depth/calibration and evaluator geometry are never
inputs. Ambiguous detections and failed depth fits remain explicitly unobserved.

The separate version preserves the byte-frozen v1 queued experiment. Algorithm
parameters and mask association are unchanged; only absent/unsupported semantic
table evidence may use the full image outside all hand/object masks instead.
The dominant background plane is a hypothesis, not a SAM3 table detection.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import file_sha256, validate_manifest
from artirecon.scene_sequence import object_from_arrays, scene_from_arrays


class InitializationError(ValueError):
    pass


def default_config():
    return dict(schema_version=2, object_id="block", object_prompt="wooden block on patterned mat",
                hand_prompt="hand", table_prompt="tabletop", confidence_mode="all",
                min_mask_pixels=16, max_object_image_fraction=.2, candidate_margin=.05,
                minimum_score=.3, duplicate_iou=.85, mask_erosion_pixels=1,
                min_object_points=16, max_points_per_object=4000,
                table_sample_frames=5, max_table_points_per_frame=1000,
                table_ransac_iterations=250, table_inlier_distance_m=.008,
                table_min_inlier_fraction=.25, table_min_points=100,
                initial_table_support_prior=True, extent_quantile=.05,
                minimum_extent_m=.004, maximum_extent_m=.5,
                support_distance_m=.012, contact_distance_m=.025,
                contact_image_distance_px=6., min_hand_points=12,
                contact_policy="touch_only", grasp_motion_min_m=.004,
                grasp_relative_motion_tolerance_m=.02, random_seed=0,
                table_fallback="dominant_plane_full_image_excluding_all_hand_object_candidates")


def validate_config(config):
    if set(config) != set(default_config()) or config["schema_version"] != 2:
        raise InitializationError("Use a complete saved initializer configuration")
    if config["table_fallback"] != "dominant_plane_full_image_excluding_all_hand_object_candidates":
        raise InitializationError("V2 supports only its explicitly declared dominant-plane fallback")
    if config["object_id"] != "block":
        raise InitializationError("This initializer is block-only; a cuboid does not model a bottle nozzle")
    if config["confidence_mode"] not in {"all", "rank50"}:
        raise InitializationError("confidence_mode must be all or rank50")
    if config["contact_policy"] not in {"touch_only", "proximity_motion_grasp"}:
        raise InitializationError("Unknown contact policy")
    if config["initial_table_support_prior"] is not True:
        raise InitializationError("This baseline requires its explicitly declared initial-support hypothesis")
    for key in ("min_mask_pixels", "min_object_points", "max_points_per_object", "table_sample_frames",
                "max_table_points_per_frame", "table_ransac_iterations", "table_min_points", "min_hand_points"):
        if type(config[key]) is not int or config[key] < 1:
            raise InitializationError(f"{key} must be a positive integer")
    for key in ("table_inlier_distance_m", "minimum_extent_m", "maximum_extent_m", "support_distance_m",
                "contact_distance_m", "contact_image_distance_px", "grasp_motion_min_m", "grasp_relative_motion_tolerance_m"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise InitializationError(f"{key} must be positive")
    for key in ("minimum_score", "max_object_image_fraction", "candidate_margin", "duplicate_iou", "table_min_inlier_fraction"):
        if not np.isfinite(config[key]) or not 0 < config[key] <= 1:
            raise InitializationError(f"{key} must be in (0,1]")
    if not 0 < config["extent_quantile"] < .5:
        raise InitializationError("extent_quantile must be in (0,.5)")
    if type(config["mask_erosion_pixels"]) is not int or config["mask_erosion_pixels"] < 0:
        raise InitializationError("mask_erosion_pixels must be a nonnegative integer")


def validate_geometry(data):
    """Accept only the explicitly isolated, single static source camera format."""
    depth = np.asarray(data["depth_m"])
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise InitializationError("Expected one source camera depth [T,1,H,W]")
    intrinsics, cameras = np.asarray(data["K"], float), np.asarray(data["c2w"], float)
    if (intrinsics.shape != (1, 3, 3) or not np.isfinite(intrinsics).all()
            or intrinsics[0, 0, 0] <= 0 or intrinsics[0, 1, 1] <= 0
            or not np.allclose(intrinsics[0, 2], [0, 0, 1], atol=1e-6)
            or cameras.shape != (1, 4, 4) or not np.allclose(cameras[0], np.eye(4), atol=1e-8)):
        raise InitializationError("Use predicted source K and identity camera gauge; supplied world extrinsics are unsupported")
    timestamps = np.asarray(data["timestamps"], float)
    if timestamps.shape != (len(depth),) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise InitializationError("Source timestamps must be finite, increasing and match depth")
    for name in ("valid", "confidence_rank"):
        if np.asarray(data[name]).shape != depth.shape:
            raise InitializationError(f"{name} shape differs from depth")
    rank = np.asarray(data["confidence_rank"])
    if not np.isfinite(rank).all() or np.any((rank < 0) | (rank > 1)):
        raise InitializationError("confidence_rank must contain finite [0,1] ranks, not probabilities")
    return depth[:, 0], intrinsics[0], timestamps


class MaskCandidates:
    def __init__(self, data, frames, shape, timestamps):
        self.data = data
        self.shape = tuple(int(x) for x in data["shape"])
        self.t = np.asarray(data["t"])
        self.p = np.asarray(data["p"])
        self.score = np.asarray(data["score"], float)
        self.prompts = list(np.asarray(data["prompts"]).tolist())
        self.packed = np.asarray(data["mask"])
        count = len(self.t)
        if self.shape != tuple(shape) or not np.array_equal(np.asarray(data["timestamps"]), timestamps):
            raise InitializationError("Mask grid/timestamps must match the exact source depth sequence")
        if (self.packed.shape != (count, (int(np.prod(shape)) + 7) // 8) or self.packed.dtype != np.uint8
                or self.p.shape != (count,) or self.score.shape != (count,)
                or not np.issubdtype(self.t.dtype, np.integer) or not np.issubdtype(self.p.dtype, np.integer)
                or (count and (self.t.min() < 0 or self.t.max() >= frames or self.p.min() < 0 or self.p.max() >= len(self.prompts)))
                or not np.isfinite(self.score).all() or np.any((self.score < 0) | (self.score > 1))):
            raise InitializationError("Malformed SAM3 candidate arrays")

    def rows(self, frame, prompt, threshold):
        if prompt not in self.prompts:
            raise InitializationError(f"Required task prompt absent: {prompt}")
        return np.flatnonzero((self.t == frame) & (self.p == self.prompts.index(prompt)) & (self.score >= threshold))

    def mask(self, row):
        return np.unpackbits(self.packed[row])[:np.prod(self.shape)].reshape(self.shape).astype(bool)

    def union(self, rows):
        result = np.zeros(self.shape, bool)
        for row in rows:
            result |= self.mask(int(row))
        return result


def associate_masks(candidates, frames, config):
    """Source-mask-only association; scores are heuristics, not probabilities."""
    selected, diagnostics = [], []
    previous = None
    shape = candidates.shape
    diagonal = np.linalg.norm(shape)
    for frame in range(frames):
        rows = candidates.rows(frame, config["object_prompt"], config["minimum_score"])
        table = candidates.union(candidates.rows(frame, config["table_prompt"], config["minimum_score"]))
        table_region = np.zeros(shape, bool)
        if table.any():
            y, x = np.where(table)
            table_region[y.min():y.max() + 1, x.min():x.max() + 1] = True
        masks = [(int(row), candidates.mask(row)) for row in rows]
        masks = [(row, mask) for row, mask in masks if config["min_mask_pixels"] <= mask.sum() <= np.prod(shape) * config["max_object_image_fraction"]]
        # SAM3 may emit overlapping alternatives for the same region. Retain the
        # highest detector score before judging ambiguity between different objects.
        unique = []
        for row, mask in sorted(masks, key=lambda pair: (-candidates.score[pair[0]], pair[0])):
            if all((mask & other).sum() / max(1, (mask | other).sum()) < config["duplicate_iou"] for _, other in unique):
                unique.append((row, mask))
        maximum_area = max([mask.sum() for _, mask in unique], default=1)
        scored = []
        for row, mask in unique:
            table_fraction = float((mask & table_region).sum() / mask.sum())
            continuity = 0.
            if previous is not None:
                iou = (mask & previous).sum() / max(1, (mask | previous).sum())
                distance = np.linalg.norm(np.argwhere(mask).mean(0) - np.argwhere(previous).mean(0))
                continuity = .5 * iou + .5 * np.exp(-distance / (.1 * diagonal))
            score = .4 * candidates.score[row] + .2 * mask.sum() / maximum_area + .15 * table_fraction + .25 * continuity
            scored.append(dict(row=row, selection_score=float(score), sam3_score=float(candidates.score[row]),
                               area_px=int(mask.sum()), table_bbox_fraction=table_fraction, continuity=float(continuity)))
        scored.sort(key=lambda row: (-row["selection_score"], row["row"]))
        ambiguous = len(scored) > 1 and scored[0]["selection_score"] - scored[1]["selection_score"] < config["candidate_margin"]
        chosen = scored[0]["row"] if scored and not ambiguous else None
        selected.append(chosen)
        diagnostics.append(dict(frame=frame, selected_row=chosen, ambiguous=ambiguous, candidates=scored,
                                reason="ambiguous" if ambiguous else "missing" if chosen is None else "selected"))
        # Missing detections never become observed by borrowing a previous mask.
        previous = candidates.mask(chosen) if chosen is not None else None
    return selected, diagnostics


def lift(depth, intrinsics, mask, maximum, rng):
    yy, xx = np.where(mask & np.isfinite(depth) & (depth > 0))
    if len(xx) > maximum:
        ids = np.sort(rng.choice(len(xx), maximum, replace=False))
        xx, yy = xx[ids], yy[ids]
    rays = np.c_[xx, yy, np.ones(len(xx))] @ np.linalg.inv(intrinsics).T
    return rays * depth[yy, xx, None]


def fit_table(points, config):
    points = np.asarray(points, float)
    if len(points) < config["table_min_points"]:
        raise InitializationError("Insufficient table-mask depth after object/hand exclusion and confidence filtering")
    rng = np.random.default_rng(config["random_seed"])
    best = np.zeros(len(points), bool)
    for _ in range(config["table_ransac_iterations"]):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        inliers = abs(points @ normal - sample[0] @ normal) < config["table_inlier_distance_m"]
        if inliers.sum() > best.sum():
            best = inliers
    if best.mean() < config["table_min_inlier_fraction"] or best.sum() < config["table_min_points"]:
        raise InitializationError("No adequately supported table plane in model depth")
    selected = points[best]
    center = np.median(selected, axis=0)
    _, _, axes = np.linalg.svd(selected - center, full_matrices=False)
    normal = axes[-1]
    offset = -float(np.median(selected @ normal))
    # Orient toward the camera, using the tabletop-camera capture assumption.
    if offset < 0:
        normal, offset = -normal, -offset
    residuals = abs(points @ normal + offset)
    return normal, offset, dict(points=len(points), inliers=int(best.sum()), inlier_fraction=float(best.mean()),
                               median_residual_m=float(np.median(residuals)),
                               inlier_residual_median_m=float(np.median(residuals[best])),
                               normal_orientation="Toward the source camera; explicit tabletop capture assumption")


def intervals(mask):
    result, start = [], None
    for frame, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = frame
        if not value and start is not None:
            result.append((start, frame - 1))
            start = None
    return result


def initialize(data, masks, config):
    """Return shared scene + diagnostics; caller provides model-only provenance."""
    validate_config(config)
    depth, intrinsics, timestamps = validate_geometry(data)
    candidates = MaskCandidates(masks, len(depth), depth.shape[-2:], timestamps)
    selected, association = associate_masks(candidates, len(depth), config)
    if selected[0] is None:
        raise InitializationError("First-frame block association is missing/ambiguous; no shared shape can be initialized")
    rng = np.random.default_rng(config["random_seed"])
    valid = np.asarray(data["valid"], bool)[:, 0] & np.isfinite(depth) & (depth > 0)
    if config["confidence_mode"] == "rank50":
        valid &= np.asarray(data["confidence_rank"])[:, 0] >= .5
    table_frames = np.unique(np.rint(np.linspace(0, len(depth) - 1, min(len(depth), config["table_sample_frames"]))).astype(int))
    table_points, fallback_points, sampling_report = [], [], []
    for frame in table_frames:
        mask = candidates.union(candidates.rows(frame, config["table_prompt"], config["minimum_score"]))
        exclusions = candidates.union(candidates.rows(frame, config["object_prompt"], config["minimum_score"]))
        exclusions |= candidates.union(candidates.rows(frame, config["hand_prompt"], config["minimum_score"]))
        exclusions = binary_dilation(exclusions, iterations=2)
        mask &= ~exclusions
        table_points.append(lift(depth[frame], intrinsics, mask & valid[frame], config["max_table_points_per_frame"], rng))
        # Independent fixed RNG means preparing the fallback never changes the
        # semantic-plane or object-point samples when semantic evidence works.
        fallback_rng = np.random.default_rng(config["random_seed"] + 1000 + int(frame))
        fallback_points.append(lift(depth[frame], intrinsics, ~exclusions & valid[frame],
                                    config["max_table_points_per_frame"], fallback_rng))
        sampling_report.append(dict(frame=int(frame), semantic_pixels=int((mask & valid[frame]).sum()),
                                    exclusion_fraction=float(exclusions.mean()),
                                    fallback_pixels=int((~exclusions & valid[frame]).sum())))
    fallback_reason = None
    try:
        normal, offset, table_report = fit_table(np.concatenate(table_points), config)
        plane_source = "SAM3 semantic tabletop pixels excluding hand/object candidates"
    except InitializationError as error:
        fallback_reason = str(error)
        normal, offset, table_report = fit_table(np.concatenate(fallback_points), config)
        plane_source = "Dominant background plane hypothesis from full image outside hand/object candidates"
    table_report.update(plane_source=plane_source, fallback_used=fallback_reason is not None,
                        fallback_reason=fallback_reason, sampled_frames=sampling_report,
                        selection_roi_xyxy=[0, 0, depth.shape[2], depth.shape[1]],
                        semantic_table_certainty="unestablished", cached_plane_used=False)
    axis = np.array([1., 0., 0.])
    axis -= normal * (normal @ axis)
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([0., 1., 0.]) - normal * normal[1]
    axis /= np.linalg.norm(axis)
    basis = np.stack([axis, np.cross(normal, axis), normal], axis=1)
    clouds, hands, evidence = [], [], []
    for frame, row in enumerate(selected):
        hand_mask = candidates.union(candidates.rows(frame, config["hand_prompt"], config["minimum_score"]))
        object_mask = candidates.mask(row) if row is not None else np.zeros(depth.shape[-2:], bool)
        clean = object_mask & ~hand_mask
        if config["mask_erosion_pixels"]:
            clean &= distance_transform_edt(clean) > config["mask_erosion_pixels"]
        cloud = lift(depth[frame], intrinsics, clean & valid[frame], config["max_points_per_object"], rng)
        hand_cloud = lift(depth[frame], intrinsics, hand_mask & ~object_mask & valid[frame], config["max_points_per_object"], rng)
        observed = row is not None and len(cloud) >= config["min_object_points"]
        clouds.append(cloud)
        hands.append(hand_cloud)
        image_gap = float(distance_transform_edt(~hand_mask)[object_mask].min()) if hand_mask.any() and object_mask.any() else None
        evidence.append(dict(frame=frame, timestamp_s=float(timestamps[frame]), observed=bool(observed),
                             selected_candidate_row=row, points=len(cloud), hand_points=len(hand_cloud),
                             mask_pixels=int(object_mask.sum()), clean_mask_pixels=int(clean.sum()),
                             confidence_retained_pixels=int((clean & valid[frame]).sum()),
                             mask_gap_px=image_gap,
                             camera_z_median_m=float(np.median(cloud[:, 2])) if len(cloud) else None,
                             camera_z_iqr_m=float(np.ptp(np.quantile(cloud[:, 2], [.25, .75]))) if len(cloud) else None,
                             below_table_point_fraction=float(np.mean(cloud @ normal + offset < 0)) if len(cloud) else None))
    if not evidence[0]["observed"]:
        raise InitializationError("First-frame block has insufficient unoccluded model-depth evidence")
    initial = clouds[0] @ basis
    xy = initial[:, :2] - np.median(initial[:, :2], axis=0)
    _, _, yaw_axes = np.linalg.svd(xy, full_matrices=False)
    primary = yaw_axes[0]
    if primary[0] < 0:
        primary *= -1
    yaw = np.array([[primary[0], -primary[1], 0], [primary[1], primary[0], 0], [0, 0, 1]])
    orientation = basis @ yaw
    first_local = clouds[0] @ orientation
    quantile = config["extent_quantile"]
    low, high = np.quantile(first_local, [quantile, 1 - quantile], axis=0)
    # Hidden extent along table normal is unobserved. This initial RGB-based
    # support hypothesis sets only frame-zero bottom to the estimated plane.
    low[2] = -offset
    extents = high - low
    if np.any(extents < config["minimum_extent_m"]) or np.any(extents > config["maximum_extent_m"]):
        raise InitializationError(f"First-frame cuboid extents are unsupported: {extents.tolist()} m")
    first_center = orientation @ ((low + high) / 2)
    observed = np.array([row["observed"] for row in evidence], bool)
    poses = np.repeat(np.eye(4)[None], len(depth), axis=0)
    poses[:, :3, :3] = orientation
    reference_median = np.median(clouds[0], axis=0)
    previous = first_center.copy()
    for frame in range(len(depth)):
        if observed[frame]:
            previous = first_center + np.median(clouds[frame], axis=0) - reference_median
        poses[frame, :3, 3] = previous
        evidence[frame]["pose_placeholder"] = not bool(observed[frame])
        evidence[frame]["box_bottom_above_table_m"] = float(normal @ previous + offset - extents[2] / 2)
    touch = np.zeros(len(depth), bool)
    support = np.zeros(len(depth), bool)
    contact_known = np.zeros(len(depth), bool)
    for frame, row in enumerate(evidence):
        support[frame] = observed[frame] and abs(row["box_bottom_above_table_m"]) <= config["support_distance_m"]
        contact_known[frame] = observed[frame] and len(hands[frame]) >= config["min_hand_points"]
        if contact_known[frame]:
            # Surface-to-surface proximity is a hypothesis, not true contact.
            distance = float(np.quantile(cKDTree(clouds[frame]).query(hands[frame], k=1)[0], .05))
            row["hand_object_distance_m"] = distance
            touch[frame] = distance <= config["contact_distance_m"] and row["mask_gap_px"] <= config["contact_image_distance_px"]
        else:
            row["hand_object_distance_m"] = None
        row["contact_observable"] = bool(contact_known[frame])
    grasp = np.zeros(len(depth), bool)
    if config["contact_policy"] == "proximity_motion_grasp":
        for frame in range(1, len(depth)):
            if touch[frame - 1:frame + 1].all():
                object_delta = poses[frame, :3, 3] - poses[frame - 1, :3, 3]
                hand_delta = np.median(hands[frame], axis=0) - np.median(hands[frame - 1], axis=0)
                if (np.linalg.norm(object_delta) >= config["grasp_motion_min_m"]
                        and np.linalg.norm(object_delta - hand_delta) <= config["grasp_relative_motion_tolerance_m"]):
                    grasp[frame - 1:frame + 1] = True
    provenance = {"method": "Source RGB SAM3 association + UniDepth metric depth/K; v2 fixed-shape translation initializer",
                  "artifact_ids": ["predicted_geometry", "predicted_cameras", "sam3_masks"],
                  "confidence_status": "uncalibrated"}
    contacts = []
    for kind, flag, counterpart in (("support", support, "table"), ("touch", touch, "world"), ("grasp", grasp, "world")):
        for index, (start, end) in enumerate(intervals(flag)):
            contacts.append(dict(id=f"{kind}-{index}", a="block", b=counterpart, kind=kind,
                                 start_frame=start, end_frame=end, status="hypothesized", confidence=None,
                                 provenance={**provenance, "method": f"Estimated {kind}: {config['contact_policy']}; not measured contact"}))
    for index, (start, end) in enumerate(intervals(~contact_known)):
        contacts.append(dict(id=f"contact-unknown-{index}", a="block", b="world", kind="touch",
                             start_frame=start, end_frame=end, status="unknown", confidence=None,
                             provenance={**provenance, "method": "Insufficient hand/object evidence; no contact asserted"}))
    scene = scene_from_arrays(
        timestamps=timestamps,
        coordinate_frame={"name": "source_camera", "handedness": "right", "up_axis": "unknown", "pose_convention": "T_world_local"},
        metric_scale={"status": "estimated", "provenance": provenance}, actors=[],
        objects=[object_from_arrays(object_id="block", poses=poses, observed=observed, provenance=provenance,
                                   primitive={"type": "box", "half_extents_m": extents / 2})],
        support={"id": "table", "normal_world": normal, "offset_m": offset, "confidence": None,
                 "provenance": {**provenance, "method": plane_source,
                                "dominant_plane_fallback_used": fallback_reason is not None,
                                "semantic_table_certainty": "unestablished"}},
        contacts=contacts, uncertainty={"status": "estimated", "provenance": provenance,
                                      "interpretation": "Evidence dispersion and ablation sensitivity, not calibrated metric covariance"},
        task={"description": "Initialize task-supplied wooden block from one recorded RGB camera", "target": None})
    scene["motion_model"] = "shared_cuboid_fixed_orientation_translation_only"
    scene["orientation_estimated_over_time"] = False
    scene["support_plane_hypothesis"] = {"method": plane_source, "fallback_used": fallback_reason is not None,
                                         "semantic_table_certainty": "unestablished"}
    scene["initial_support_prior"] = {"used": True, "frame": 0, "source": "Task-supplied RGB support hypothesis",
                                      "per_frame_support_corrections_applied": False}
    diagnostics = dict(status="initialized", confidence_mode=config["confidence_mode"], frames=len(depth),
                       observed_frames=int(observed.sum()), unobserved_frames=np.flatnonzero(~observed).tolist(),
                       association=association, evidence=evidence, table=table_report,
                       dimensions_m=extents.tolist(), initial_pose=poses[0].tolist(), config=config,
                       contact_intervals=contacts, scene_initialization_only=True, SAM3D_used=False,
                       full_orientation_tracking=False, measured_depth_used=False, supplied_calibration_used=False,
                       simulator_accessed=False, camera_z_convention="z-depth along predicted source pinhole rays",
                       confidence_interpretation="Global within-frame inverse uncertainty rank; rank50 keeps rank>=0.5, not a probability",
                       limitations=["A cuboid from visible first-frame surfaces cannot establish hidden shape accuracy.",
                                    "Initial support is a declared RGB-based hypothesis, not a measured object bottom.",
                                    "Camera is assumed static; source camera identity is a gauge, not robot registration.",
                                    "Depth centroid changes conflate translation, visibility, mask errors and depth bias.",
                                    "Fixed rotation is a baseline approximation, not recovered full SE3 motion.",
                                    "A world contact counterpart denotes an anonymous estimated hand, not fitted MANO geometry.",
                                    "Missing/ambiguous frames carry a placeholder pose with observed=false; planning must reject these gaps.",
                                    "Task text supplies the relevant object category; this is not automatic task relevance discovery."])
    return scene, diagnostics


def _materialized_manifest(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "input-manifest.json").read_text())
    report = validate_manifest(manifest, folder, verify_files=True)
    if report["condition"] != "rgb_only" or report["oracle_dependencies"] or report["calibration_dependencies"]:
        raise InitializationError("Initializer requires RGB-only model products; supplied K/depth are not allowed")
    for artifact in manifest["artifacts"]:
        if "path" in artifact:
            path = Path(artifact["path"])
            artifact["path"] = str((path if path.is_absolute() else folder / path).resolve())
    return manifest


def produce(*, depth_bundle, mask_bundle, config_path, output):
    depth_bundle, mask_bundle, output = map(lambda path: Path(path).resolve(), (depth_bundle, mask_bundle, output))
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    source = _materialized_manifest(depth_bundle)
    segmented = _materialized_manifest(mask_bundle)
    def named(manifest, name):
        return next(artifact for artifact in manifest["artifacts"] if artifact["id"] == name)
    if named(source, "source_rgb")["sha256"] != named(segmented, "source_rgb")["sha256"]:
        raise InitializationError("SAM3 and depth must consume identical isolated RGB bytes")
    for name, path in (("predicted_geometry", depth_bundle / "geometry.npz"),
                       ("predicted_cameras", depth_bundle / "geometry.npz"),
                       ("sam3_masks", mask_bundle / "masks.npz")):
        artifact = named(source if name != "sam3_masks" else segmented, name)
        if Path(artifact["path"]) != path or artifact["sha256"] != file_sha256(path):
            raise InitializationError(f"Actual consumed file is not bound to {name}")
    if source["task_mode"] != segmented["task_mode"]:
        raise InitializationError("Task modes differ")
    output.mkdir(parents=True)
    start = time.monotonic()
    try:
        with np.load(depth_bundle / "geometry.npz", allow_pickle=False) as depth, np.load(mask_bundle / "masks.npz", allow_pickle=False) as masks:
            scene, diagnostics = initialize(depth, masks, config)
        (output / "scene-sequence.json").write_text(json.dumps(scene, indent=2, allow_nan=False))
        (output / "initializer-config.json").write_text(json.dumps(config, indent=2, allow_nan=False))
        merged = deepcopy(source)
        current = {artifact["id"]: artifact for artifact in merged["artifacts"]}
        for artifact in segmented["artifacts"]:
            if artifact["id"] not in current:
                merged["artifacts"].append(artifact)
            elif artifact["id"] != "source_rgb" and artifact != current[artifact["id"]]:
                raise InitializationError(f"Conflicting artifact ID: {artifact['id']}")
        merged["artifacts"].extend([
            {"id": "source_block_initializer", "kind": "object_geometry", "origin": "model_prediction",
             "parent_ids": ["predicted_geometry", "predicted_cameras", "sam3_masks"],
             "path": "scene-sequence.json", "sha256": file_sha256(output / "scene-sequence.json"),
             "description": "Shared cuboid + translation-only approximation, not SAM3D",
             "initial_support_prior": "Algorithm/task hypothesis that frame-zero object rests on predicted table; no measured 3D value supplied"},
            {"id": "estimated_support", "kind": "support", "origin": "model_prediction",
             "parent_ids": ["predicted_geometry", "predicted_cameras", "sam3_masks"],
             "path": "scene-sequence.json", "sha256": file_sha256(output / "scene-sequence.json")},
        ])
        merged["method_inputs"] = list(dict.fromkeys(merged["method_inputs"] + segmented["method_inputs"] +
                                                       ["source_block_initializer", "estimated_support"]))
        validation = validate_manifest(merged, output, verify_files=True)
        (output / "input-manifest.json").write_text(json.dumps(merged, indent=2, allow_nan=False))
        diagnostics.update(seconds=time.monotonic() - start, input_protocol=validation,
                           scene_sha256=file_sha256(output / "scene-sequence.json"),
                           initializer_config_sha256=file_sha256(output / "initializer-config.json"),
                           producer_sha256=file_sha256(__file__))
    except (InitializationError, ValueError, KeyError) as error:
        diagnostics = dict(status="rejected", error=str(error), seconds=time.monotonic() - start,
                           config=config, simulator_accessed=False, producer_sha256=file_sha256(__file__))
        (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
        raise
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    config = modes.add_parser("config")
    config.add_argument("--confidence-mode", choices=["all", "rank50"], required=True)
    config.add_argument("--output", type=Path, required=True)
    run = modes.add_parser("run")
    for name in ("depth-bundle", "mask-bundle", "config-path", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    if args.pop("mode") == "config":
        payload = dict(default_config(), confidence_mode=args["confidence_mode"])
        with args["output"].open("x") as stream:
            json.dump(payload, stream, indent=2)
        print(json.dumps(payload, indent=2))
    else:
        result = produce(**args)
        print(json.dumps({key: result[key] for key in ("status", "observed_frames", "frames", "dimensions_m", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
