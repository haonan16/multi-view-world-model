"""Fresh SAM3D aspect ratios + one fitted metric size for a shared block.

Only canonical mesh proportions are used. SAM3D camera pose, MoGe metric scale,
and rendered-camera geometry do not enter fitting. Orientation stays the declared
source table/PCA baseline; this tests shape proportions, not mesh motion tracking.
"""
import argparse
from copy import deepcopy
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
from benchmarks import source_nonpenetrating_fit as physical
from benchmarks.static_depth_calibration import materialize_recipe


def default_config(point_policy="object_rank50"):
    return dict(schema_version=1, shape_prior="fresh_sam3d_canonical_mesh_bbox_aspect",
                shape_parameterization="one_positive_metric_scale_for_shared_xyz_ratio",
                canonical_axis_mapping="canonical_z_to_table_normal_xy_to_inherited_pca_basis",
                feasible_set="shared_table_nonpenetration_all_frames",
                initialization="geometric_mean_shape_scale_then_project_negative_clearance",
                joint_config=dict(joint.default_config(), point_policy=point_policy))


def validate_config(config):
    expected = default_config()
    if set(config) != set(expected): raise ValueError("Use complete predeclared aspect config")
    for key in expected:
        if key != "joint_config" and config[key] != expected[key]: raise ValueError(f"Unsupported aspect choice: {key}")
    joint.validate_config(config["joint_config"])


def canonical_aspect(vertices):
    vertices = np.asarray(vertices, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError("Need finite canonical mesh vertices [N,3]")
    dimensions = np.ptp(vertices, axis=0)
    if np.any(dimensions <= 0): raise ValueError("Degenerate canonical mesh extents")
    ratio = dimensions / np.exp(np.log(dimensions).mean())
    return ratio, dimensions


def fit_shared_scene(scene, observations, config, aspect):
    validate_config(config); validate_scene(scene, require_metric_scale=True)
    if len(scene["objects"]) != 1 or scene["objects"][0]["id"] != "block" or scene["objects"][0]["geometry"]["type"] != "box":
        raise ValueError("Block-only shared cuboid aspect ablation")
    aspect = np.asarray(aspect, float)
    if aspect.shape != (3,) or not np.isfinite(aspect).all() or np.any(aspect <= 0) or not np.isclose(np.prod(aspect), 1.):
        raise ValueError("Aspect must be finite positive xyz ratios with geometric mean one")
    params = config["joint_config"]; obj = scene["objects"][0]
    poses = np.asarray(obj["T_world_object"], float)
    if not np.allclose(poses[:, :3, :3], poses[0, :3, :3], atol=1e-6): raise ValueError("Inherited fixed orientation only")
    if not np.array_equal(observations["timestamps"], scene["timestamps"]): raise ValueError("Timestamp mismatch")
    initial_half = np.asarray(obj["geometry"]["half_extents_m"])
    initial_scalar = float(np.exp(np.log(initial_half / aspect).mean()))
    frames = len(poses); initial_centers = poses[:, :3, 3]
    normal = np.asarray(scene["support"]["normal_world"]); offset = scene["support"]["offset_m"]
    basis = physical.table_basis(normal); extent_weights = abs(normal @ poses[0, :3, :3])
    coordinates = initial_centers @ basis
    initial_clearance = coordinates[:, 2] + offset - extent_weights @ (aspect * initial_scalar)
    coordinates[:, 2] = np.maximum(initial_clearance, 0.)
    def decode(x):
        half = aspect * np.exp(x[0]); local = x[1:].reshape(frames, 3).copy()
        local[:, 2] += extent_weights @ half - offset
        return half, local @ basis.T
    initial = np.r_[np.log(initial_scalar), coordinates.ravel()]
    adjustment = params["maximum_pose_adjustment_m"]
    lower_pose, upper_pose = coordinates - adjustment, coordinates + adjustment; lower_pose[:, 2] = 0.
    minimum_scalar = np.max(params["minimum_extent_m"] / 2 / aspect)
    maximum_scalar = np.min(params["maximum_extent_m"] / 2 / aspect)
    if not minimum_scalar <= initial_scalar <= maximum_scalar: raise ValueError("Initial aspect-projected shape outside fixed size bounds")
    lower = np.r_[np.log(minimum_scalar), lower_pose.ravel()]
    upper = np.r_[np.log(maximum_scalar), upper_pose.ravel()]
    nres = sum(len(p) + 5 for p in observations["points"]) + 3 * max(0, frames - 2) + 3
    sparsity = lil_matrix((nres, len(initial)), dtype=int); cursor = 0
    for frame, points in enumerate(observations["points"]):
        count = len(points) + 5
        sparsity[cursor:cursor + count, 0] = 1
        sparsity[cursor:cursor + count, 1 + frame * 3:4 + frame * 3] = 1
        cursor += count
    for frame in range(1, frames - 1):
        sparsity[cursor:cursor + 3, 0] = 1
        sparsity[cursor:cursor + 3, 1 + (frame - 1) * 3:1 + (frame + 2) * 3] = 1
        cursor += 3
    sparsity[cursor:cursor + 3, 0] = 1
    def evaluate(x):
        half, positions = decode(x)
        return physical.components(half, positions, scene=scene, observations=observations, config=params)[0]
    started = time.monotonic()
    result = least_squares(evaluate, initial, jac_sparsity=sparsity.tocsr(), bounds=(lower, upper),
                           loss=params["robust_loss"], f_scale=1., max_nfev=params["max_nfev"], xtol=1e-6, ftol=1e-6, gtol=1e-6)
    fit_seconds = time.monotonic() - started
    half, positions = decode(result.x); output = deepcopy(scene)
    fitted_poses = poses.copy(); fitted_poses[:, :3, 3] = positions
    output["objects"][0]["T_world_object"] = fitted_poses.tolist()
    output["objects"][0]["geometry"]["half_extents_m"] = half.tolist()
    output["objects"][0]["observed"] = [row["observed"] for row in observations["evidence"]]
    for record in (output["objects"][0], output["objects"][0]["geometry"]):
        record["provenance"]["method"] = "Fresh SAM3D aspect ratios + one metric size + source-depth/mask joint motion fit"
        record["provenance"]["artifact_ids"] = list(dict.fromkeys(record["provenance"].get("artifact_ids", []) + ["sam3d_canonical_mesh"]))
    output["contacts"] = [c for c in output["contacts"] if c["kind"] != "support"]
    for index, (first, last) in enumerate(base.intervals(observations["support"])):
        output["contacts"].append(dict(id=f"rgb-support-{index}", a="block", b="table", kind="support", start_frame=first, end_frame=last,
            status="hypothesized", confidence=None, provenance=dict(method="Stationary source RGB object and distant hand masks, independent of 3D height", artifact_ids=["sam3_masks"], confidence_status="uncalibrated")))
    output["joint_fit"] = dict(point_policy=params["point_policy"], shared_table_nonpenetration=True, enforced_in_optimization=True,
        endpoint_clamped=False, postfit_pose_changes=False, simulator_feedback_used=False, full_orientation_tracking=False,
        one_metric_shape_scale=True, aspect_from_fresh_SAM3D=True, SAM3D_metric_pose_or_scale_used=False)
    output["shape_prior"] = dict(aspect_xyz=aspect.tolist(), fitted_half_extent_geometric_mean_m=float(np.exp(result.x[0])),
        mapping=config["canonical_axis_mapping"], shape_repr="Aspect-constrained cuboid, not full SAM3D mesh collision")
    validate_scene(output, require_metric_scale=True)
    def summary(h, p): return physical.components(h, p, scene=scene, observations=observations, config=params)[1]
    report = dict(status="fitted", optimizer_success=bool(result.success), optimizer_message=result.message, evaluations=result.nfev,
        fit_seconds=fit_seconds, final_cost=float(result.cost), config=config, before=summary(initial_half, initial_centers),
        feasible_initialization=summary(*decode(initial)), after=summary(half, positions),
        aspect_xyz=aspect.tolist(), initial_metric_scalar_m=initial_scalar, fitted_metric_scalar_m=float(np.exp(result.x[0])),
        initial_projection_frames=np.flatnonzero(initial_clearance < 0).tolist(), initial_projection_distance_m=np.maximum(-initial_clearance, 0).tolist(),
        observed_frames=sum(row["observed"] for row in observations["evidence"]), frames=frames,
        source_depth_masks_support_evidence_unchanged=True, SAM3D_metric_pose_or_scale_used=False,
        GT_accuracy_measured=False, simulator_accessed=False, endpoint_specific_operation=False,
        pose_trust_region="Same 0.15m table-tangent/clearance bounds as unconstrained-aspect feasible-set baseline",
        losses_unchanged_from_joint_v1=True,
        limitations=["SAM3D proportions are predictions, not known geometry.", "Canonical z-up is mapped to estimated table normal; XY alignment inherits the source initializer.",
            "Only a cuboid envelope with fixed orientation is fitted; this is not full SAM3D mesh tracking.", "Table and metric gauge remain estimated; geometric feasibility is not execution success."])
    return output, report


def produce(*, initializer_bundle, asset_bundle, config_path, comparison_bundle, output):
    import trimesh
    bundle, asset_bundle, output = map(lambda p: Path(p).resolve(), (initializer_bundle, asset_bundle, output))
    if output.exists(): raise FileExistsError(output)
    manifest = base._materialized_manifest(bundle); asset_manifest = base._materialized_manifest(asset_bundle)
    active = {a["id"]: a for a in manifest["artifacts"]}; asset = {a["id"]: a for a in asset_manifest["artifacts"]}
    for aid in ("source_rgb", "sam3_masks"):
        if active[aid]["sha256"] != asset[aid]["sha256"]: raise ValueError("SAM3D and depth must use identical source observations")
    selected = asset["selected_object_mask"]["selection"]
    if selected["frame_index"] != 0 or selected["object_prompt"] != "wooden block on patterned mat": raise ValueError("Need fresh frame-zero block asset")
    initializer_report = json.loads((bundle / "diagnostics.json").read_text())
    first_row = initializer_report["association"][0]["selected_row"]
    if selected["candidate_row_ids"][selected["selected_index"]] != first_row: raise ValueError("SAM3D asset targets a different RGB candidate")
    mesh_record = asset["sam3d_canonical_mesh"]
    mesh = trimesh.load(mesh_record["path"], process=False)
    aspect, canonical_dimensions = canonical_aspect(mesh.vertices)
    config = json.loads(Path(config_path).read_text()); validate_config(config)
    current = [a["id"] for a in active.values() if a["kind"] == "object_geometry" and a.get("path") == str(bundle / "scene-sequence.json")]
    if len(current) != 1: raise ValueError("Bind one actual source initializer")
    record = active["predicted_geometry"]
    if not record.get("raw_geometry_artifact_id"): raise ValueError("This ablation requires the fixed frame-zero static-calibrated depth")
    geometry = materialize_recipe(active[record["raw_geometry_artifact_id"]]["path"], record["path"])
    scene = load_scene(bundle / "scene-sequence.json", require_metric_scale=True)
    with np.load(active["sam3_masks"]["path"], allow_pickle=False) as masks:
        observations = joint.prepare_evidence(geometry, masks, scene, config["joint_config"])
    fitted, report = fit_shared_scene(scene, observations, config, aspect)
    comparison_bundle = Path(comparison_bundle).resolve(); previous_manifest = base._materialized_manifest(comparison_bundle)
    previous_artifacts = {a["id"]: a for a in previous_manifest["artifacts"]}
    for aid in ("source_rgb", "predicted_geometry", "predicted_cameras", "sam3_masks"):
        if active[aid]["sha256"] != previous_artifacts[aid]["sha256"]: raise ValueError("Comparison evidence differs")
    previous_config = json.loads((comparison_bundle / "diagnostics.json").read_text())["config"]["joint_config"]
    if previous_config != config["joint_config"]: raise ValueError("Comparison losses differ")
    previous = load_scene(comparison_bundle / "scene-sequence.json", require_metric_scale=True)["objects"][0]
    report["free_aspect_baseline"] = physical.components(np.asarray(previous["geometry"]["half_extents_m"]), np.asarray(previous["T_world_object"])[:, :3, 3],
        scene=scene, observations=observations, config=config["joint_config"])[1]
    report["canonical_dimensions"] = canonical_dimensions.tolist(); report["mesh_sha256"] = mesh_record["sha256"]
    # Merge only canonical mesh ancestry; camera mesh/metric pose are NOT method inputs.
    needed = set()
    def visit(aid):
        if aid in needed: return
        needed.add(aid)
        for parent in asset[aid]["parent_ids"]: visit(parent)
    visit("sam3d_canonical_mesh")
    for aid in needed:
        if aid in active:
            if active[aid].get("sha256") != asset[aid].get("sha256"): raise ValueError("Conflicting shared artifact")
        else: manifest["artifacts"].append(asset[aid])
    output.mkdir(parents=True); path = output / "scene-sequence.json"
    path.write_text(json.dumps(fitted, indent=2, allow_nan=False))
    manifest["artifacts"].append(dict(id="sam3d_aspect_joint_fit", kind="object_geometry", origin="model_prediction",
        parent_ids=current + ["predicted_geometry", "predicted_cameras", "sam3_masks", "sam3d_canonical_mesh"], path="scene-sequence.json", sha256=file_sha256(path)))
    manifest["method_inputs"].append("sam3d_aspect_joint_fit")
    report["input_protocol"] = validate_manifest(manifest, output, verify_files=True)
    report["producer_sha256"] = file_sha256(__file__); report["config_sha256"] = file_sha256(config_path)
    (output / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (output / "fit-config.json").write_text(json.dumps(config, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="mode", required=True)
    cfg = commands.add_parser("config"); cfg.add_argument("--point-policy", choices=["all", "object_rank50", "soft_rank"], required=True); cfg.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    for name in ("initializer-bundle", "asset-bundle", "config-path", "comparison-bundle", "output"): run.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    if args.pop("mode") == "config":
        with args["output"].open("x") as stream: json.dump(default_config(args["point_policy"]), stream, indent=2)
    else: print(json.dumps(produce(**args), indent=2))


if __name__ == "__main__": main()
