"""The staged plan cannot disguise forecasting, oracle inputs or missing results."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[2] / "benchmarks/experiment_matrix.py"
SPEC = importlib.util.spec_from_file_location("experiment_matrix", PATH)
matrix_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix_module)


def manifest(forecast=False):
    result = {
        "schema_version": 1, "condition": "rgb_only",
        "task_mode": "forecast" if forecast else "reconstruction",
        "artifacts": [{"id": "rgb", "kind": "rgb", "origin": "observed_rgb", "generated": False,
                       "parent_ids": [], "time_range_s": [0, 0 if forecast else 10]}],
        "method_inputs": ["rgb"],
        "metric_scale": {"status": "unresolved", "artifact_ids": [], "description": "Not fitted yet"},
        "robot_registration": {"status": "unresolved", "artifact_ids": [], "description": "No robot registration yet"},
    }
    if forecast:
        result["observed_until_s"] = 0
    for kind in ("depth", "cameras", "masks", "support"):
        result["artifacts"].append({"id": kind, "kind": kind, "origin": "model_prediction", "parent_ids": ["rgb"]})
        result["method_inputs"].append(kind)
    return result


def generated(data, role):
    data["artifacts"].append({"id": role, "kind": "rgb", "origin": "model_prediction",
                              "parent_ids": ["rgb"], "generated": True, "view_role": role})
    data["method_inputs"].append(role)


def row(mode="reconstruction", views="source_only", condition="strict_rgb"):
    return next(r for r in matrix_module.build_matrix(include_forecast=True)["runs"]
                if r["scene_id"] == "block" and r["task_mode"] == mode and r["view_condition"] == views
                and r["condition"] == condition and r["keyframe_policy"] == "dense")


def test_default_plan_is_frugal_and_contains_no_run_results():
    matrix = matrix_module.build_matrix()
    report = matrix_module.validate_matrix(matrix)
    assert report["stages"] == {"primary": 8, "controls": 4}
    assert report["jobs_launched"] == 0 and report["measurements"] == "unmeasured"
    assert not any(r["task_mode"] == "forecast" or r["method"] != "ours" for r in matrix["runs"])
    assert all(v is None for group in matrix["measurement_template"]["metrics"].values() for v in group.values())
    assert all(s["status"] == "not_scheduled" for s in matrix["future_scenes"])


def test_forecasts_and_native_external_baselines_are_explicit_later_stages():
    matrix = matrix_module.build_matrix(include_forecast=True, include_external=True)
    report = matrix_module.validate_matrix(matrix)
    assert report["stages"] == {"primary": 8, "controls": 4, "forecast_later": 8, "external_later": 4}
    forecasts = [r for r in matrix["runs"] if r["task_mode"] == "forecast"]
    assert all(r["observed_scope"] == "first_frame" and r["source_temporal_stream"] == "world_model_source_forecast" for r in forecasts)
    assert all(r["comparison_match"] == "unmatched" for r in matrix["runs"] if r["method"] != "ours")


@pytest.mark.parametrize("mutation", [
    lambda m: m.__setitem__("automatic_launch", True),
    lambda m: m["runs"][0].__setitem__("task_mode", "forecast"),
    lambda m: m["runs"][0].__setitem__("source_view_count", 4),
    lambda m: m["runs"][0].__setitem__("metrics", {"success": True}),
    lambda m: m["runs"][2].__setitem__("keyframe_budget", 4),
    lambda m: m["evaluation"].__setitem__("simulator_feedback_to_planner", True),
    lambda m: m["measurement_template"]["metrics"]["runtime_s"].__setitem__("total", 0),
])
def test_reject_mixed_conditioning_and_fabricated_measurements(mutation):
    matrix = matrix_module.build_matrix()
    mutation(matrix)
    with pytest.raises(ValueError):
        matrix_module.validate_matrix(matrix)


def test_input_check_surfaces_registration_scale_and_missing_helper_blockers():
    report = matrix_module.validate_run_inputs(row(views="generated_helpers"), manifest())
    assert not report["execution_ready"]
    assert "metric_scale is unresolved" in report["execution_blockers"]
    assert "robot_registration is unresolved" in report["execution_blockers"]
    assert any("helper views" in b for b in report["execution_blockers"])
    assert report["measurements"] == matrix_module.unmeasured_result()


def test_forecast_source_only_requires_predicted_motion_not_depth_alone():
    data = manifest(forecast=True)
    report = matrix_module.validate_run_inputs(row(mode="forecast"), data)
    assert any("not a motion forecast" in b for b in report["execution_blockers"])
    generated(data, "source_forecast")
    report = matrix_module.validate_run_inputs(row(mode="forecast"), data)
    assert not any("not a motion forecast" in b for b in report["execution_blockers"])
    generated(data, "helper")
    with pytest.raises(matrix_module.ProtocolError, match="Source-only"):
        matrix_module.validate_run_inputs(row(mode="forecast"), data)


def test_oracle_and_future_inputs_cannot_be_relabelled_as_strict_rgb():
    data = manifest()
    data["artifacts"].append({"id": "sensor", "kind": "depth", "origin": "sensor_depth", "parent_ids": []})
    data["artifacts"][1]["parent_ids"].append("sensor")
    with pytest.raises(matrix_module.ProtocolError, match="forbidden"):
        matrix_module.validate_run_inputs(row(), data)
    data = manifest(forecast=True)
    data["artifacts"][0]["time_range_s"] = [0, 10]
    with pytest.raises(matrix_module.ProtocolError, match="observed future"):
        matrix_module.validate_run_inputs(row(mode="forecast"), data)


def test_external_single_frame_input_is_not_a_full_video_comparison():
    matrix = matrix_module.build_matrix(include_external=True)
    external = next(r for r in matrix["runs"] if r["method"] == "scenerig")
    with pytest.raises(matrix_module.ProtocolError, match="cannot consume"):
        matrix_module.validate_run_inputs(external, manifest())
    bad = deepcopy(matrix)
    next(r for r in bad["runs"] if r["method"] == "scenerig")["comparison_match"] = "matched"
    with pytest.raises(ValueError, match="unmatched"):
        matrix_module.validate_matrix(bad)
