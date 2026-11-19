"""Generate and validate a staged comparison plan; never launch model jobs.

The default is eight primary block/bottle plans plus four optional calibration/
depth controls. Plans are not results or execution-ready preregistrations: actual
RGB hashes, model revisions, budgets, metric scale and robot registration still
need to be bound before any executable experiment is frozen.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import ProtocolError, validate_manifest


CONDITIONS = {"strict_rgb": "rgb_only", "calibrated_rgb": "calibrated_rgb", "rgbd_oracle": "rgbd_oracle"}
POLICIES = {"dense": None, "uniform": 5, "contact_events": 5}
INITIAL_SCENES = {"block": "RH20T", "bottle": "HO-Cap"}
METRICS = {
    "geometry": {"surface_error_mm": None, "contact_gap_mm": None, "clearance_error_mm": None,
                 "table_penetration_mm": None, "heldout_silhouette_iou": None},
    "motion": {"translation_error_mm": None, "rotation_error_deg": None, "endpoint_error_mm": None,
               "contact_interval_iou": None},
    "execution": {"first_attempt_success": None, "goal_error_mm": None, "collisions": None},
    "runtime_s": {"total": None, "world_model": None, "depth": None, "sam3": None, "sam3d": None,
                  "actor_fit": None, "object_fit": None, "planning": None, "execution": None},
    "agent_effort": {"calls": None, "input_tokens": None, "output_tokens": None,
                     "cost_usd": None, "refinement_steps": None},
}


def unmeasured_result():
    return {"status": "unmeasured", "metrics": deepcopy(METRICS)}


def _run(scene, condition, mode, views, policy, stage):
    return {
        "id": f"{scene}__{condition}__{mode}__{views}__{policy}",
        "scene_id": scene, "stage": stage, "method": "ours", "condition": condition,
        "task_mode": mode, "observed_scope": "full_clip" if mode == "reconstruction" else "first_frame",
        "source_view_count": 1, "view_condition": views,
        "source_temporal_stream": "observed_clip" if mode == "reconstruction" else "world_model_source_forecast",
        "keyframe_policy": policy, "keyframe_budget": POLICIES[policy],
        "sampling_stage": "reconstruction_input", "comparison_match": "matched_within_input_group",
        "status": "planned", "seed": 0,
    }


def build_matrix(scene_ids=("block", "bottle"), include_controls=True, include_forecast=False,
                 include_external=False):
    """Return small staged config only; optional later comparisons cost no compute."""
    if not scene_ids or len(set(scene_ids)) != len(scene_ids) or set(scene_ids) - set(INITIAL_SCENES):
        raise ValueError("Initial scene_ids must be a nonempty unique subset of block/bottle")
    rows = []
    for scene in scene_ids:
        for views, policy in (("source_only", "dense"), ("generated_helpers", "dense"),
                              ("generated_helpers", "uniform"), ("generated_helpers", "contact_events")):
            rows.append(_run(scene, "strict_rgb", "reconstruction", views, policy, "primary"))
        if include_controls:
            for condition in ("calibrated_rgb", "rgbd_oracle"):
                rows.append(_run(scene, condition, "reconstruction", "source_only", "dense", "controls"))
        if include_forecast:
            for views, policy in (("source_only", "dense"), ("generated_helpers", "dense"),
                                  ("generated_helpers", "uniform"), ("generated_helpers", "contact_events")):
                rows.append(_run(scene, "strict_rgb", "forecast", views, policy, "forecast_later"))
        if include_external:
            for method, scope in (("scenerig", "first_frame"), ("ovow", "full_clip")):
                rows.append({
                    "id": f"{scene}__external__{method}", "scene_id": scene, "stage": "external_later",
                    "method": method, "condition": "strict_rgb", "task_mode": "reconstruction",
                    "observed_scope": scope, "source_view_count": 1, "view_condition": "source_only",
                    "source_temporal_stream": "observed_clip" if scope == "full_clip" else "observed_first_frame",
                    "keyframe_policy": "native", "keyframe_budget": None, "sampling_stage": "native",
                    "comparison_match": "unmatched", "status": "planned", "seed": None,
                    "unmatched_reason": "SceneRig receives one frame; OVOW receives the clip. Native methods and budgets differ; no action retargeter is supplied.",
                })
    result = {
        "schema_version": 1, "status": "planning_only", "automatic_launch": False,
        "scenes": [{"id": name, "dataset": INITIAL_SCENES[name], "clip_binding": None} for name in scene_ids],
        "condition_mapping": deepcopy(CONDITIONS), "runs": rows,
        "future_scenes": [{"id": "peg_precision", "dataset": "ManiSkill", "status": "not_scheduled"},
                          {"id": "robocasa", "dataset": "RoboCasa", "status": "not_scheduled"}],
        "later_stages": ["separate first-frame forecast matrix", "external baseline reproductions",
                         "native sparse world-model generation"],
        "execution_requirements": {
            "metric_scale": "resolved from permitted method inputs",
            "robot_registration": "independent target-robot registration, never an assumed identity or evaluator scene transform",
            "source_hashes": "bind one recorded RGB view, camera selection, crop and timestamps before running",
            "model_revisions": "pin estimators, world model, masks, asset generator and fitter",
            "budgets": "predeclare seed, optimization steps, agent-call cap, hardware and timeout per matched group",
        },
        "evaluation": {"scene": "independent_fixed", "freeze_commands_before_execution": True,
                       "simulator_feedback_to_planner": False, "rollouts_per_variant": 1,
                       "outcome_based_selection": False, "oracle_data": "evaluation and explicit controls only"},
        "measurement_template": unmeasured_result(),
        "limitations": [
            "Primary sparse rows subsample reconstruction inputs. They do not establish speedup from native sparse world-model generation.",
            "Post-fit keyframe compression is a different diagnostic; cached-fit fidelity is not GT geometry accuracy.",
            "Equal RGB conditioning does not imply matched compute; report generation, reconstruction and agent costs separately.",
            "This file does not allocate GPUs, install models, run baselines or establish successful execution.",
        ],
    }
    validate_matrix(result)
    return result


def validate_matrix(matrix):
    """Validate a planning config, rejecting mode mixing and fabricated results."""
    if matrix.get("schema_version") != 1 or matrix.get("status") != "planning_only" or matrix.get("automatic_launch") is not False:
        raise ValueError("Expected schema_version=1 planning_only matrix with automatic_launch=false")
    if matrix.get("condition_mapping") != CONDITIONS:
        raise ValueError("Condition mapping must preserve strict_rgb -> rgb_only")
    scenes = matrix.get("scenes", [])
    if not scenes or any(s.get("id") not in INITIAL_SCENES for s in scenes):
        raise ValueError("Initial cases must be block/bottle; future scenes are not scheduled")
    scene_ids = {s["id"] for s in scenes}
    if len(scene_ids) != len(scenes):
        raise ValueError("Duplicate scene IDs")
    rows, ids, counts = matrix.get("runs"), set(), {}
    if not isinstance(rows, list) or not rows:
        raise ValueError("runs must be a nonempty list")
    for row in rows:
        name = row.get("id")
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError("Run IDs must be nonempty and unique")
        ids.add(name)
        if row.get("scene_id") not in scene_ids or row.get("condition") not in CONDITIONS:
            raise ValueError(f"{name}: unknown scene or condition")
        if row.get("status") != "planned" or row.get("source_view_count") != 1:
            raise ValueError(f"{name}: plans require one observed view and status=planned")
        if "results" in row or "metrics" in row:
            raise ValueError(f"{name}: planning config cannot contain claimed measurements")
        mode, method, scope = row.get("task_mode"), row.get("method"), row.get("observed_scope")
        if mode not in {"reconstruction", "forecast"}:
            raise ValueError(f"{name}: task mode must be explicit")
        if method == "ours":
            expected_scope = "full_clip" if mode == "reconstruction" else "first_frame"
            expected_stream = "observed_clip" if mode == "reconstruction" else "world_model_source_forecast"
            if scope != expected_scope or row.get("source_temporal_stream") != expected_stream:
                raise ValueError(f"{name}: reconstruction and forecast conditioning cannot be mixed")
            if row.get("view_condition") not in {"source_only", "generated_helpers"}:
                raise ValueError(f"{name}: unknown view condition")
            policy = row.get("keyframe_policy")
            if policy not in POLICIES or row.get("keyframe_budget") != POLICIES[policy]:
                raise ValueError(f"{name}: expected dense or five uniform/contact-event frames")
            if row.get("sampling_stage") != "reconstruction_input":
                raise ValueError(f"{name}: do not conflate input sampling with post-fit compression or sparse generation")
            if row.get("comparison_match") != "matched_within_input_group":
                raise ValueError(f"{name}: internal comparisons are grouped by input condition and task mode")
        elif method in {"scenerig", "ovow"}:
            if row.get("comparison_match") != "unmatched" or not row.get("unmatched_reason"):
                raise ValueError(f"{name}: native external baselines must disclose unmatched inputs/budgets")
            if scope != ("first_frame" if method == "scenerig" else "full_clip") or mode != "reconstruction":
                raise ValueError(f"{name}: incorrect external input scope")
            if row["condition"] != "strict_rgb" or row.get("keyframe_policy") != "native":
                raise ValueError(f"{name}: external RGB baseline must keep its declared native configuration")
        else:
            raise ValueError(f"{name}: unknown method")
        stage = row.get("stage")
        if stage not in {"primary", "controls", "forecast_later", "external_later"}:
            raise ValueError(f"{name}: unknown stage")
        counts[stage] = counts.get(stage, 0) + 1
    evaluation = matrix.get("evaluation", {})
    if (evaluation.get("scene") != "independent_fixed" or evaluation.get("freeze_commands_before_execution") is not True
            or evaluation.get("simulator_feedback_to_planner") is not False
            or evaluation.get("outcome_based_selection") is not False or evaluation.get("rollouts_per_variant") != 1):
        raise ValueError("Independent evaluation requires frozen commands, one rollout and no outcome-based refinement")
    if matrix.get("measurement_template") != unmeasured_result():
        raise ValueError("All planned metrics/runtime/agent effort must remain unmeasured (null)")
    if not matrix.get("execution_requirements", {}).get("robot_registration") or not matrix["execution_requirements"].get("metric_scale"):
        raise ValueError("Metric scale and independent robot registration are required execution gates")
    return {"status": "valid_plan", "run_count": len(rows), "stages": counts,
            "jobs_launched": 0, "measurements": "unmeasured"}


def validate_run_inputs(run, manifest, base_dir=None, verify_files=False):
    """Check declared inputs for one plan; return blockers rather than invent state.

    Generated RGB records use view_role='helper' or 'source_forecast'. These roles
    describe the declared consumed lineage, not a proof of model consistency.
    This checks inputs only; model revisions/resource budgets still need binding.
    """
    if run.get("condition") not in CONDITIONS:
        raise ProtocolError("Unknown experiment condition")
    if manifest.get("condition") != CONDITIONS[run["condition"]] or manifest.get("task_mode") != run.get("task_mode"):
        raise ProtocolError("Run condition/task mode does not match input manifest")
    report = validate_manifest(manifest, base_dir, verify_files)
    active = [a for a in manifest["artifacts"] if a["id"] in report["method_artifact_ids"]]
    source = [a for a in active if a["origin"] == "observed_rgb"]
    if len(source) != 1 or source[0].get("view_count", 1) != 1:
        raise ProtocolError("Comparison requires exactly one declared observed RGB camera")
    interval = source[0]["time_range_s"]
    if run["observed_scope"] == "first_frame" and interval[0] != interval[1]:
        raise ProtocolError("First-frame condition cannot consume a recorded source sequence")
    if run["observed_scope"] == "full_clip" and interval[1] <= interval[0]:
        raise ProtocolError("Full-clip reconstruction requires a nonzero observed interval")
    if run["task_mode"] == "forecast":
        cutoff = manifest["observed_until_s"]
        for artifact in active:
            if artifact["origin"] in {"sensor_depth", "ground_truth"}:
                bounds = artifact.get("time_range_s")
                if not isinstance(bounds, list) or len(bounds) != 2 or bounds[1] > cutoff:
                    raise ProtocolError("Forecast oracle controls cannot consume future/unbounded GT or sensor state")
    generated = [a for a in active if a["kind"] == "rgb" and a["origin"] == "model_prediction"]
    if any(a.get("view_role") not in {"helper", "source_forecast"} for a in generated):
        raise ProtocolError("Generated views need explicit helper/source_forecast roles")
    helpers = [a for a in generated if a["view_role"] == "helper"]
    if run.get("view_condition") == "source_only" and helpers:
        raise ProtocolError("Source-only condition consumes generated helper views")
    blockers = list(report["execution_blockers"])
    if not verify_files:
        blockers.append("input file identities have not been verified")
    if run.get("method") == "ours":
        if run.get("view_condition") == "generated_helpers" and not helpers:
            blockers.append("generated helper views are not materialized in consumed lineage")
        if run["task_mode"] == "forecast" and not any(a["view_role"] == "source_forecast" for a in generated):
            blockers.append("future source-view prediction is missing; monocular depth alone is not a motion forecast")
        for kind in ("depth", "cameras", "masks", "support"):
            if not any(a["kind"] == kind for a in active):
                blockers.append(f"{kind} stage is not materialized in consumed lineage")
        if run["condition"] == "rgbd_oracle" and not any(a["origin"] == "sensor_depth" for a in active):
            blockers.append("RGB-D oracle control has no sensor-depth input")
    blockers += ["pin model revisions and resource budgets before freezing the run",
                 "independent fixed evaluation scene and command seal are not validated by this input check"]
    return {"status": "inputs_validated", "run_id": run["id"], "input_protocol": report,
            "execution_ready": False, "execution_blockers": blockers, "jobs_launched": 0,
            "measurements": unmeasured_result()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="write a planning config; no models/jobs are run")
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--include-forecast", action="store_true")
    plan.add_argument("--include-external", action="store_true")
    plan.add_argument("--without-controls", action="store_true")
    check = sub.add_parser("validate")
    check.add_argument("matrix", type=Path)
    inputs = sub.add_parser("inputs", help="validate one run's declared input provenance")
    inputs.add_argument("matrix", type=Path)
    inputs.add_argument("--run-id", required=True)
    inputs.add_argument("--manifest", type=Path, required=True)
    inputs.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        matrix = build_matrix(include_controls=not args.without_controls,
                              include_forecast=args.include_forecast, include_external=args.include_external)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(matrix, stream, indent=2, allow_nan=False)
        report = validate_matrix(matrix)
        report["path"] = str(args.out.resolve())
    else:
        matrix = json.loads(args.matrix.read_text())
        report = validate_matrix(matrix)
        if args.command == "inputs":
            rows = [r for r in matrix["runs"] if r["id"] == args.run_id]
            if len(rows) != 1:
                raise ValueError("Run ID is not in this plan")
            report = validate_run_inputs(rows[0], json.loads(args.manifest.read_text()),
                                         args.manifest.parent, args.verify_files)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
