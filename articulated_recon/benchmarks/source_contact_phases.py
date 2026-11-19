"""Estimate grasp phases from contiguous RGB proximity, co-motion and lift.

This re-estimates evidence; it never merges earlier 3D contact intervals or fills
missing detections. Output contacts remain uncertain hypotheses, not force closure.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import file_sha256, validate_manifest
from artirecon.scene_sequence import load_scene, validate_scene
from benchmarks import source_geometry as base
from benchmarks.source_joint_fit import rgb_support_hypotheses


def default_config():
    return dict(schema_version=1, proximity_px=6., minimum_duration_s=.25,
                minimum_motion_object_diagonals=.5, moving_step_object_diagonals=.04,
                minimum_motion_cosine=.3, minimum_coupled_steps=3, minimum_lift_m=.01,
                maximum_before_support_gap_s=.5, support_stationarity_window_s=.4,
                support_stationarity_radius_px=2., support_distant_hand_px=12.,
                support_minimum_observations=3)


def mask_evidence(masks, times, config):
    candidates = base.MaskCandidates(masks, len(times), tuple(masks["shape"]), times)
    selected, association = base.associate_masks(candidates, len(times), base.default_config())
    object_centers, hand_centers, gaps, diagonals, valid = [], [], [], [], []
    for frame, row in enumerate(selected):
        object_mask = candidates.mask(row) if row is not None else np.zeros(candidates.shape, bool)
        alternatives = []
        if object_mask.any():
            xy = np.argwhere(object_mask)[:, ::-1]
            object_centers.append(xy.mean(0)); diagonals.append(np.linalg.norm(np.ptp(xy, axis=0)))
            for hand_row in candidates.rows(frame, "hand", .3):
                hand = candidates.mask(hand_row)
                if hand.any():
                    distance = float(distance_transform_edt(~hand)[object_mask].min())
                    alternatives.append((distance, -float(candidates.score[hand_row]), int(hand_row), np.argwhere(hand)[:, ::-1].mean(0)))
        else:
            object_centers.append([0., 0.]); diagonals.append(0.)
        if alternatives:
            best = min(alternatives, key=lambda item: item[:3])
            gaps.append(best[0]); hand_centers.append(best[3]); valid.append(True)
        else:
            gaps.append(np.nan); hand_centers.append([0., 0.]); valid.append(False)
    support_config = dict(stationarity_window_s=config["support_stationarity_window_s"],
                          stationarity_radius_px=config["support_stationarity_radius_px"],
                          distant_hand_px=config["support_distant_hand_px"],
                          minimum_support_observations=config["support_minimum_observations"])
    support = rgb_support_hypotheses(object_centers, gaps, times, valid, support_config)
    return dict(object_centers=np.asarray(object_centers), hand_centers=np.asarray(hand_centers), support=support,
                gaps=np.asarray(gaps), diagonals=np.asarray(diagonals), valid=np.array(valid), association=association)


def infer_phases(scene, evidence, config):
    if set(config) != set(default_config()) or config["schema_version"] != 1:
        raise ValueError("Use the complete fixed RGB-phase configuration")
    validate_scene(scene, require_metric_scale=True)
    if len(scene["objects"]) != 1 or scene["objects"][0]["id"] != "block":
        raise ValueError("Current phase estimator supports the single block task")
    obj = scene["objects"][0]
    times = np.asarray(scene["timestamps"], float)
    observed = np.asarray(obj["observed"], bool)
    known = observed & evidence["valid"] & np.isfinite(evidence["gaps"])
    near = known & (evidence["gaps"] <= config["proximity_px"])
    support = np.asarray(evidence["support"], bool) & known
    normal = np.asarray(scene["support"]["normal_world"])
    poses = np.asarray(obj["T_world_object"])
    heights = poses[:, :3, 3] @ normal
    rows, accepted = [], []
    for first, last in base.intervals(near):
        before = np.flatnonzero(support & (times <= times[first]))
        recent = before[times[first] - times[before] <= config["maximum_before_support_gap_s"]]
        after_supported = last + 1 < len(times) and support[last + 1] and known[last + 1]
        diagonal = float(np.median(evidence["diagonals"][first:last + 1]))
        object_path = evidence["object_centers"][first:last + 1]
        hand_path = evidence["hand_centers"][first:last + 1]
        displacement = float(np.linalg.norm(np.ptp(object_path, axis=0)))
        object_delta, hand_delta = np.diff(object_path, axis=0), np.diff(hand_path, axis=0)
        speed0, speed1 = np.linalg.norm(object_delta, axis=1), np.linalg.norm(hand_delta, axis=1)
        moving = (speed0 >= config["moving_step_object_diagonals"] * diagonal) & (speed1 > 1e-6)
        cosine = np.sum(object_delta[moving] * hand_delta[moving], axis=1) / (speed0[moving] * speed1[moving])
        motion_cosine = float(np.median(cosine)) if len(cosine) else None
        lift = float(heights[first:last + 1].max() - np.median(heights[recent])) if len(recent) else None
        reasons = []
        if times[last] - times[first] < config["minimum_duration_s"]: reasons.append("proximity too brief")
        if not len(recent): reasons.append("no recent independent support hypothesis")
        if not after_supported: reasons.append("first released frame lacks support/observation evidence")
        if displacement < config["minimum_motion_object_diagonals"] * diagonal: reasons.append("insufficient object motion")
        if len(cosine) < config["minimum_coupled_steps"] or motion_cosine is None or motion_cosine < config["minimum_motion_cosine"]:
            reasons.append("insufficient RGB hand/object co-motion")
        if lift is None or lift < config["minimum_lift_m"]: reasons.append("estimated lift not established; could be pushing")
        row = dict(start_frame=first, end_frame=last, start_s=float(times[first]), end_s=float(times[last]),
                   duration_s=float(times[last] - times[first]), object_displacement_px=displacement,
                   object_diagonal_px=diagonal, coupled_steps=len(cosine), median_motion_cosine=motion_cosine,
                   estimated_lift_m=lift, first_release_supported=bool(after_supported), accepted=not reasons, reasons=reasons)
        rows.append(row)
        if not reasons:
            accepted.append(dict(id=f"rgb-motion-grasp-{len(accepted)}", a="block", b="world", kind="grasp",
                                 start_frame=first, end_frame=last, status="hypothesized", confidence=None,
                                 provenance={"method": "Contiguous observed RGB proximity + co-motion + estimated lift + independent support",
                                             "artifact_ids": ["sam3_masks", "predicted_geometry", "predicted_cameras"],
                                             "confidence_status": "uncalibrated"}))
    output = deepcopy(scene)
    original = [c for c in output["contacts"] if c["kind"] == "grasp"]
    original_support = [c for c in output["contacts"] if c["kind"] == "support"]
    output["contacts"] = [c for c in output["contacts"] if c["kind"] not in {"grasp", "support"}] + accepted
    for index, (first, last) in enumerate(base.intervals(support)):
        output["contacts"].append(dict(id=f"rgb-phase-support-{index}", a="block", b="table", kind="support",
                                       start_frame=first, end_frame=last, status="hypothesized", confidence=None,
                                       provenance={"method": "Stationary RGB object and distant detected RGB hand, independently of 3D height",
                                                   "artifact_ids": ["sam3_masks"], "confidence_status": "uncalibrated"}))
    output["rgb_phase_estimation"] = dict(config=config, raw_grasp_hypotheses=original,
                                         filled_missing_frames=False, simulation_feedback_used=False)
    validate_scene(output, require_metric_scale=True)
    report = dict(status="estimated", candidates=rows, accepted_intervals=len(accepted),
                  missing_evidence_frames=np.flatnonzero(~known).tolist(), raw_grasp_hypotheses=original,
                  raw_support_hypotheses=original_support, support_evidence_uses_3d_height=False,
                  hypotheses_not_force_closure=True, missing_intervals_merged=False,
                  simulator_accessed=False, geometry_changed=False, config=config)
    return output, report


def produce(*, scene_bundle, config_path, output):
    bundle, output = Path(scene_bundle).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    manifest = base._materialized_manifest(bundle)
    active = {a["id"]: a for a in manifest["artifacts"]}
    scene_artifacts = [a["id"] for a in active.values() if a.get("kind") == "object_geometry" and a.get("path") == str(bundle / "scene-sequence.json")]
    if not scene_artifacts: raise ValueError("Current scene is not a hash-bound method artifact")
    config = json.loads(Path(config_path).read_text())
    scene = load_scene(bundle / "scene-sequence.json", require_metric_scale=True)
    with np.load(active["sam3_masks"]["path"], allow_pickle=False) as masks:
        evidence = mask_evidence(masks, np.asarray(scene["timestamps"]), config)
    result, report = infer_phases(scene, evidence, config)
    output.mkdir(parents=True)
    path = output / "scene-sequence.json"; path.write_text(json.dumps(result, indent=2, allow_nan=False))
    manifest["artifacts"].append(dict(id="rgb_contact_phases", kind="object_geometry", origin="model_prediction",
                                      parent_ids=scene_artifacts + ["sam3_masks"], path="scene-sequence.json", sha256=file_sha256(path)))
    manifest["method_inputs"].append("rgb_contact_phases")
    report["input_protocol"] = validate_manifest(manifest, output, verify_files=True)
    report["producer_sha256"] = file_sha256(__file__)
    (output / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    config = commands.add_parser("config"); config.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    for name in ("scene-bundle", "config-path", "output"): run.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    if args.pop("mode") == "config":
        with args["output"].open("x") as stream: json.dump(default_config(), stream, indent=2)
    else: print(json.dumps(produce(**args), indent=2))


if __name__ == "__main__": main()
