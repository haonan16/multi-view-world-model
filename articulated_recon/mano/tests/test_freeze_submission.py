"""Regression checks for the pre-execution boundary; no simulator is invoked."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artirecon.input_protocol import ProtocolError, file_sha256, load_manifest
from benchmarks.freeze_submission import (
    attach_evaluation, claim_execution, freeze_submission, verify_submission,
)


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def case(tmp_path):
    def artifact(aid, kind, origin, parents, **extra):
        path = tmp_path / (aid + ".dat")
        path.write_text(aid)
        return dict(id=aid, kind=kind, origin=origin, parent_ids=parents,
                    path=path.name, sha256=file_sha256(path), **extra)

    manifest = {
        "schema_version": 1, "condition": "rgb_only", "task_mode": "reconstruction",
        "method_inputs": ["rgb", "depth", "registration"],
        "artifacts": [
            artifact("rgb", "rgb", "observed_rgb", [], generated=False, time_range_s=[0, 1]),
            {"id": "weights", "kind": "model_prior", "origin": "prior", "scope": "global",
             "prior_type": "learned_model_weights", "parent_ids": [], "revision": "test-model"},
            artifact("depth", "depth", "model_prediction", ["rgb", "weights"]),
            artifact("registration", "robot_registration", "model_prediction", ["rgb", "depth"]),
        ],
        "metric_scale": {"status": "estimated", "artifact_ids": ["depth"], "description": "Metric depth model"},
        "robot_registration": {"status": "estimated", "artifact_ids": ["registration"],
                               "description": "Observed robot registered to image reconstruction"},
    }
    manifest_path = write_json(tmp_path / "inputs.json", manifest)
    provenance = {"method": "Model fitting", "artifact_ids": ["depth"]}
    scene = {
        "schema_version": 1, "units": {"length": "m", "time": "s", "angle": "rad"},
        "coordinate_frame": {"name": "reconstruction", "handedness": "right", "up_axis": "+z",
                             "pose_convention": "T_world_local"},
        "timestamps": [0, 1], "metric_scale": {"status": "estimated", "provenance": provenance},
        "actors": [], "objects": [{"id": "block", "geometry": {"type": "box", "half_extents_m": [.01] * 3,
                                     "provenance": provenance}, "T_world_object": [np.eye(4).tolist()] * 2,
                                    "observed": [True, True], "provenance": provenance}],
        "support": None, "contacts": [], "uncertainty": {"status": "unestimated"},
        "task": {"description": "Move the observed block", "target": None},
    }
    scene_path = write_json(tmp_path / "scene.json", scene)
    paths = {"scene": scene_path, "input_manifest": manifest_path, "output": tmp_path / "submission.json"}
    for name in ("plan", "planner", "executor"):
        paths[name] = tmp_path / (name + ".dat")
        paths[name].write_text(name)
    paths["controller_config"] = write_json(tmp_path / "controller.json", {"target_frame": "robot_base"})
    paths["plan_metadata"] = tmp_path / "plan-metadata.json"
    refresh_metadata(paths)
    return paths


def refresh_metadata(paths, **changes):
    metadata = {
        "source_scene_sha256": file_sha256(paths["scene"]),
        "input_manifest_sha256": load_manifest(paths["input_manifest"])["manifest_sha256"],
        "plan_sha256": file_sha256(paths["plan"]), "planner_sha256": file_sha256(paths["planner"]),
        "coordinate_frame": "robot_base", "registration_artifact_id": "registration",
        "command_format": "test-opaque-commands", "simulator_feedback_used": False,
    }
    write_json(paths["plan_metadata"], dict(metadata, **changes))


def evaluator(case):
    root = case["output"].parent
    reference = root / "independent-scene.dat"
    reference.write_text("fixed held-out geometry")
    spec = write_json(root / "evaluator.json", {
        "schema_version": 1, "independent_scene": True,
        "artifacts": [{"id": "fixed_scene", "path": reference.name, "sha256": file_sha256(reference)}],
    })
    return spec, reference, root / "evaluation-attachment.json"


def test_ready_submission_seals_scene_and_plan_without_evaluator(case):
    payload = freeze_submission(**case)
    assert payload["input_protocol"]["execution_ready"]
    assert not payload["evaluation_scene_supplied_to_freeze"]
    assert verify_submission(case["output"]) == payload
    with pytest.raises(FileExistsError):
        freeze_submission(**case)


@pytest.mark.parametrize("field", ["plan", "scene", "planner", "executor", "controller_config"])
def test_changed_command_or_producer_or_scene_is_rejected(case, field):
    freeze_submission(**case)
    case[field].write_text("changed after sealing")
    with pytest.raises(ProtocolError, match="Sealed file changed"):
        verify_submission(case["output"])


def test_changed_active_depth_evidence_is_rejected(case):
    freeze_submission(**case)
    (case["output"].parent / "depth.dat").write_text("different estimated depth")
    with pytest.raises(ProtocolError, match="Sealed file changed"):
        verify_submission(case["output"])


def test_unresolved_robot_registration_blocks_before_sealing(case):
    m = json.loads(case["input_manifest"].read_text())
    m["robot_registration"] = {"status": "unresolved", "artifact_ids": [], "description": "No base observed"}
    write_json(case["input_manifest"], m)
    with pytest.raises(ProtocolError, match="robot_registration"):
        freeze_submission(**case)
    assert not case["output"].exists()


def test_scene_cannot_reference_evaluator_only_geometry(case):
    m = json.loads(case["input_manifest"].read_text())
    m["artifacts"].append({"id": "oracle", "kind": "object_geometry", "origin": "ground_truth", "parent_ids": []})
    write_json(case["input_manifest"], m)
    scene = json.loads(case["scene"].read_text())
    scene["objects"][0]["geometry"]["provenance"]["artifact_ids"] = ["oracle"]
    write_json(case["scene"], scene)
    with pytest.raises(ProtocolError, match="consumed method artifact_ids"):
        freeze_submission(**case)


def test_oracle_label_does_not_allow_evaluator_geometry_into_planning(case):
    m = json.loads(case["input_manifest"].read_text())
    m["condition"] = "rgbd_oracle"
    oracle = case["output"].parent / "oracle-geometry.dat"
    oracle.write_text("ground-truth object mesh")
    m["artifacts"].append({"id": "oracle", "kind": "object_geometry", "origin": "ground_truth", "parent_ids": [],
                           "path": oracle.name, "sha256": file_sha256(oracle)})
    m["method_inputs"].append("oracle")
    write_json(case["input_manifest"], m)
    with pytest.raises(ProtocolError, match="evaluator-only at plan submission"):
        freeze_submission(**case)


@pytest.mark.parametrize("changes", [{"source_scene_sha256": "stale"}, {"simulator_feedback_used": True},
                                     {"coordinate_frame": "other_base"}, {"registration_artifact_id": "depth"}])
def test_incompatible_plan_metadata_is_rejected(case, changes):
    refresh_metadata(case, **changes)
    with pytest.raises(ProtocolError):
        freeze_submission(**case)


def test_external_scene_mesh_must_be_declared_materialized_method_input(case):
    scene = json.loads(case["scene"].read_text())
    scene["objects"][0]["geometry"] = {"type": "mesh", "uri": "undeclared.obj",
                                       "provenance": {"method": "SAM3D", "artifact_ids": ["depth"]}}
    write_json(case["scene"], scene)
    with pytest.raises(ProtocolError, match="unsealed external scene asset"):
        freeze_submission(**case)


def test_evaluator_attaches_after_seal_and_first_claim_is_final(case):
    payload = freeze_submission(**case)
    sealed_sha, plan_sha = file_sha256(case["output"]), file_sha256(case["plan"])
    spec, _, receipt = evaluator(case)
    attach_evaluation(submission=case["output"], evaluation_spec=spec, receipt=receipt)
    assert file_sha256(case["output"]) == sealed_sha and file_sha256(case["plan"]) == plan_sha
    claim_execution(submission=case["output"], evaluation_receipt=receipt, output=receipt.parent / "run1")
    assert Path(payload["execution_receipt_path"]).is_file()
    with pytest.raises(FileExistsError):
        claim_execution(submission=case["output"], evaluation_receipt=receipt, output=receipt.parent / "run2")
    with pytest.raises(ProtocolError, match="after execution started"):
        attach_evaluation(submission=case["output"], evaluation_spec=spec, receipt=receipt.parent / "different.json")


def test_modified_eval_geometry_rejected_before_execution_claim(case):
    payload = freeze_submission(**case)
    spec, reference, receipt = evaluator(case)
    attach_evaluation(submission=case["output"], evaluation_spec=spec, receipt=receipt)
    reference.write_text("outcome-tuned geometry")
    with pytest.raises(ProtocolError, match="Sealed file changed"):
        claim_execution(submission=case["output"], evaluation_receipt=receipt, output=receipt.parent / "run")
    assert not Path(payload["execution_receipt_path"]).exists()


def test_eval_reference_cannot_reuse_planning_scene(case):
    freeze_submission(**case)
    spec, _, receipt = evaluator(case)
    spec_data = json.loads(spec.read_text())
    spec_data["artifacts"][0].update(path=str(case["scene"]), sha256=file_sha256(case["scene"]))
    write_json(spec, spec_data)
    with pytest.raises(ProtocolError, match="reuse a planner-input"):
        attach_evaluation(submission=case["output"], evaluation_spec=spec, receipt=receipt)


def test_relocating_seal_cannot_create_new_execution_identity(case):
    freeze_submission(**case)
    copy_path = case["output"].with_name("copied-submission.json")
    shutil.copyfile(case["output"], copy_path)
    with pytest.raises(ProtocolError, match="relocated"):
        verify_submission(copy_path)
