import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artirecon.input_protocol import (
    ProtocolError, load_manifest, require_execution_ready, validate_manifest,
)


def manifest():
    return {
        "schema_version": 1, "condition": "rgb_only", "task_mode": "reconstruction",
        "method_inputs": ["rgb", "depth"],
        "artifacts": [
            {"id": "rgb", "kind": "rgb", "origin": "observed_rgb", "generated": False,
             "parent_ids": [], "time_range_s": [0, 5]},
            {"id": "weights", "kind": "model_prior", "origin": "prior", "scope": "global",
             "prior_type": "learned_model_weights", "parent_ids": [], "revision": "frozen-test-model"},
            {"id": "depth", "kind": "depth", "origin": "model_prediction", "parent_ids": ["rgb", "weights"]},
        ],
        "metric_scale": {"status": "estimated", "artifact_ids": ["depth"], "description": "Metric model prediction"},
        "robot_registration": {"status": "unresolved", "artifact_ids": [], "description": "Robot base is not observed"},
    }


def add_oracle(m, origin="sensor_depth", kind="depth"):
    m["artifacts"].append({"id": "oracle", "kind": kind, "origin": origin, "parent_ids": []})


def materialize(m, root):
    for record in m["artifacts"]:
        if record["origin"] == "prior":
            continue
        path = root / (record["id"] + ".dat")
        path.write_bytes(record["id"].encode())
        record.update(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return m


def test_rgb_prediction_allowed_but_missing_robot_registration_blocks_execution():
    result = validate_manifest(manifest())
    assert result["condition"] == "rgb_only"
    assert not result["execution_ready"]
    with pytest.raises(ProtocolError, match="robot_registration"):
        require_execution_ready(result)


@pytest.mark.parametrize("condition", ["rgb_only", "calibrated_rgb"])
def test_model_prediction_cannot_hide_transitive_oracle_depth(condition):
    m = manifest()
    m["condition"] = condition
    add_oracle(m)
    m["artifacts"].append({"id": "cached_scene", "kind": "object_geometry",
                           "origin": "model_prediction", "parent_ids": ["rgb", "oracle"]})
    m["artifacts"][2]["parent_ids"].append("cached_scene")
    with pytest.raises(ProtocolError, match="forbidden sensor_depth"):
        validate_manifest(m)


def test_evaluator_only_oracle_is_allowed_until_it_becomes_a_dependency():
    m = manifest()
    add_oracle(m, "ground_truth", "object_geometry")
    result = validate_manifest(m)
    assert result["evaluation_only_artifact_ids"] == ["oracle"]
    assert result["oracle_dependencies"] == []
    m["method_inputs"].append("oracle")
    with pytest.raises(ProtocolError, match="forbidden ground_truth"):
        validate_manifest(m)


def test_oracle_condition_explicitly_reports_sensor_derived_estimate():
    m = manifest()
    m["condition"] = "rgbd_oracle"
    add_oracle(m)
    m["artifacts"][2]["parent_ids"].append("oracle")
    assert validate_manifest(m)["oracle_dependencies"] == ["oracle"]


def test_known_cameras_require_calibrated_diagnostic_label():
    m = manifest()
    m["artifacts"].append({"id": "camera", "kind": "cameras", "origin": "known_calibration", "parent_ids": []})
    m["artifacts"][2]["parent_ids"].append("camera")
    with pytest.raises(ProtocolError, match="forbidden known_calibration"):
        validate_manifest(m)
    m["condition"] = "calibrated_rgb"
    assert validate_manifest(m)["calibration_dependencies"] == ["camera"]


@pytest.mark.parametrize("kind", ["support", "object_geometry", "robot_registration"])
def test_per_scene_geometry_cannot_be_declared_as_global_prior(kind):
    m = manifest()
    m["artifacts"].append({"id": "hidden_gt", "kind": kind, "origin": "prior", "scope": "global",
                           "prior_type": "robot_urdf", "parent_ids": []})
    with pytest.raises(ProtocolError, match="never per-scene"):
        validate_manifest(m)


def test_forecast_rejects_future_observation_even_through_calibration_estimator():
    m = manifest()
    m.update(task_mode="forecast", observed_until_s=0)
    m["artifacts"][0]["time_range_s"] = [0, 0]
    m["artifacts"].append({"id": "future", "kind": "rgb", "origin": "observed_rgb", "generated": False,
                           "parent_ids": [], "time_range_s": [1, 5]})
    assert validate_manifest(m)["evaluation_only_artifact_ids"] == ["future"]
    m["artifacts"].append({"id": "cameras", "kind": "cameras", "origin": "model_prediction", "parent_ids": ["rgb", "future"]})
    m["artifacts"][2]["parent_ids"].append("cameras")
    with pytest.raises(ProtocolError, match="observed future RGB"):
        validate_manifest(m)


def test_generated_rgb_is_explicit_and_cannot_be_relabeled_observed():
    m = manifest()
    predicted = {"id": "generated", "kind": "rgb", "origin": "model_prediction", "generated": True,
                 "parent_ids": ["rgb", "weights"]}
    m["artifacts"].append(predicted)
    m["artifacts"][2]["parent_ids"].append("generated")
    validate_manifest(m)
    predicted["origin"] = "observed_rgb"
    with pytest.raises(ProtocolError, match="non-generated"):
        validate_manifest(m)


@pytest.mark.parametrize("failure", ["missing_parent", "cycle", "duplicate"])
def test_invalid_dependency_graphs_fail_closed(failure):
    m = manifest()
    if failure == "missing_parent":
        m["artifacts"][2]["parent_ids"].append("unknown")
    elif failure == "cycle":
        m["artifacts"].append({"id": "cycle", "kind": "cameras", "origin": "model_prediction", "parent_ids": ["depth"]})
        m["artifacts"][2]["parent_ids"].append("cycle")
    else:
        m["artifacts"].append(copy.deepcopy(m["artifacts"][0]))
    with pytest.raises(ProtocolError):
        validate_manifest(m)


def test_hash_verification_and_actual_runner_input_binding(tmp_path):
    m = materialize(manifest(), tmp_path)
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(m))
    report = load_manifest(path, [("depth", tmp_path / "depth.dat")])
    assert report["files_verified"] and report["bound_inputs"][0]["artifact_ids"] == ["depth"]
    with pytest.raises(ProtocolError, match="undeclared cameras"):
        load_manifest(path, [("cameras", tmp_path / "depth.dat")])
    (tmp_path / "depth.dat").write_bytes(b"different cached calibrated result")
    with pytest.raises(ProtocolError, match="SHA256 mismatch"):
        load_manifest(path)


def test_verified_estimated_registration_can_enable_execution(tmp_path):
    m = manifest()
    m["artifacts"].append({"id": "registration", "kind": "robot_registration", "origin": "model_prediction", "parent_ids": ["rgb", "depth"]})
    m["method_inputs"].append("registration")
    m["robot_registration"] = {"status": "estimated", "artifact_ids": ["registration"],
                                "description": "Register the observed robot model to the reconstructed scene"}
    m = materialize(m, tmp_path)
    report = validate_manifest(m, tmp_path, verify_files=True)
    require_execution_ready(report)
    unchecked = validate_manifest(m)
    assert unchecked["execution_ready"] and not unchecked["files_verified"]
    with pytest.raises(ProtocolError, match="verified inputs"):
        require_execution_ready(unchecked)
    m["metric_scale"]["status"] = "known"
    with pytest.raises(ProtocolError, match="model-only evidence"):
        validate_manifest(m)


def test_registration_requires_explicit_transform_evidence():
    m = manifest()
    m["robot_registration"] = {"status": "estimated", "artifact_ids": ["depth"], "description": "Incorrectly use depth as a transform"}
    with pytest.raises(ProtocolError, match="explicit robot_registration artifact"):
        validate_manifest(m)


def test_nonfinite_extra_metadata_fails_with_protocol_error():
    m = manifest()
    m["accidental_nan"] = float("nan")
    with pytest.raises(ProtocolError, match="finite JSON"):
        validate_manifest(m)
