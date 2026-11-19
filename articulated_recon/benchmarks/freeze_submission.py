"""Bind a validated reconstruction to commands before evaluator data is attached.

This is an audit/immutability interface, not a planner, simulator adapter or
process sandbox. It cannot prove that a producer did not inspect other files.
The legacy executor builds its own reconstructed scene; sealing it here does
not turn that execution into independent-scene evaluation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import ProtocolError, file_sha256, load_manifest, require_execution_ready
from artirecon.scene_sequence import load_scene


def _read(path):
    return json.loads(Path(path).read_text())


def _record(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def _verify(record):
    if file_sha256(record["path"]) != record["sha256"]:
        raise ProtocolError(f"Sealed file changed: {record['path']}")


def _write_exclusive(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def _provenance_references(value, active, location="scene"):
    """Require numeric/geometry provenance to reference method-available evidence."""
    if isinstance(value, dict):
        if "provenance" in value:
            ids = value["provenance"].get("artifact_ids")
            if not isinstance(ids, list) or not ids or any(x not in active for x in ids):
                raise ProtocolError(f"{location}.provenance must reference consumed method artifact_ids")
        for key, item in value.items():
            if key != "provenance":
                _provenance_references(item, active, location + "." + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _provenance_references(item, active, f"{location}[{index}]")


def _scene_external_references(scene, scene_path, artifacts, manifest_dir):
    """External meshes/actor assets must be hash-bound method artifacts too."""
    paths = {}
    for aid, record in artifacts.items():
        if "path" in record:
            path = Path(record["path"])
            paths[aid] = (path if path.is_absolute() else manifest_dir / path).resolve()
    references = []
    for obj in scene["objects"]:
        if "uri" in obj["geometry"]:
            references.append((obj["geometry"]["uri"], obj["id"] + ".geometry"))
    for actor in scene["actors"]:
        references.append((actor["model_uri"], actor["id"] + ".model_uri"))
        if "shape_ref" in actor:
            references.append((actor["shape_ref"], actor["id"] + ".shape_ref"))
    for uri, name in references:
        if uri.startswith("artifact://"):
            aid = uri.removeprefix("artifact://")
            if aid not in paths:
                raise ProtocolError(f"{name} must reference a materialized method artifact")
        else:
            path = Path(uri)
            path = (path if path.is_absolute() else Path(scene_path).resolve().parent / path).resolve()
            if path not in paths.values():
                raise ProtocolError(f"{name} references an unsealed external scene asset: {uri}")


def freeze_submission(*, scene, input_manifest, plan, plan_metadata, planner,
                      executor, controller_config, output):
    """Seal existing commands; this function accepts no evaluation-scene input.

    The producer's plan_metadata declares source_scene_sha256,
    input_manifest_sha256 (the protocol's canonical digest), plan_sha256,
    planner_sha256, coordinate_frame, registration_artifact_id, command_format
    and simulator_feedback_used=False. These links are checked, not generated
    retroactively from an arbitrary old plan. Controller config must declare the
    same target_frame. The actual planner still needs to implement that frame.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = load_manifest(input_manifest)
    require_execution_ready(report)
    state = load_scene(scene, require_metric_scale=True)
    if not state["objects"]:
        raise ProtocolError("A manipulation submission requires reconstructed object geometry and motion")
    active = set(report["method_artifact_ids"])
    _provenance_references(state, active)
    manifest_path = Path(input_manifest).resolve()
    manifest = _read(manifest_path)
    active_artifacts = {a["id"]: a for a in manifest["artifacts"] if a["id"] in active}
    for artifact in active_artifacts.values():
        if artifact["origin"] == "ground_truth" and artifact["kind"] in {
            "object_geometry", "support", "robot_registration"
        }:
            raise ProtocolError("Ground-truth geometry/support/registration are evaluator-only at plan submission")
    _scene_external_references(state, scene, active_artifacts, manifest_path.parent)
    declared_scale = report["metric_scale"]["status"]
    if state["metric_scale"]["status"] == "known" and declared_scale == "estimated":
        raise ProtocolError("Scene cannot upgrade estimated metric scale to known")
    files = {name: _record(path) for name, path in {
        "scene": scene, "input_manifest": input_manifest, "plan": plan,
        "plan_metadata": plan_metadata, "planner": planner, "executor": executor,
        "controller_config": controller_config,
    }.items()}
    metadata = _read(plan_metadata)
    expected = {
        "source_scene_sha256": files["scene"]["sha256"],
        "input_manifest_sha256": report["manifest_sha256"],
        "plan_sha256": files["plan"]["sha256"],
        "planner_sha256": files["planner"]["sha256"],
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ProtocolError(f"Plan producer metadata does not bind {name} to these inputs")
    if metadata.get("simulator_feedback_used") is not False:
        raise ProtocolError("Plan must explicitly declare simulator_feedback_used=false")
    for name in ("coordinate_frame", "command_format"):
        if not isinstance(metadata.get(name), str) or not metadata[name].strip():
            raise ProtocolError(f"Plan metadata must declare {name}")
    if _read(controller_config).get("target_frame") != metadata["coordinate_frame"]:
        raise ProtocolError("Plan coordinate frame and controller target_frame disagree")
    registration_id = metadata.get("registration_artifact_id")
    if registration_id not in report["robot_registration"]["artifact_ids"]:
        raise ProtocolError("Plan must identify the resolved robot-registration evidence it used")

    evidence_files = []
    for artifact in manifest["artifacts"]:
        if artifact["id"] in active and "path" in artifact:
            path = Path(artifact["path"])
            path = path if path.is_absolute() else manifest_path.parent / path
            evidence_files.append(dict(artifact_id=artifact["id"], **_record(path)))
    payload = {
        "schema": "frozen-reconstruction-submission-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "submission_path": str(output), "files": files, "evidence_files": evidence_files,
        "input_protocol": report, "plan_metadata": metadata,
        "evaluation_attached": False, "evaluation_scene_supplied_to_freeze": False,
        "execution_receipt_path": str(output.with_suffix(output.suffix + ".execution-started.json")),
        "enforcement": "Checked declared provenance, source/plan/producer/controller hashes and resolved scale/registration. No evaluator input is accepted by freeze_submission.",
        "limitations": [
            "This is interface enforcement, not process isolation or proof of historical information access.",
            "Producer metadata is a declaration; hashes do not prove the commands were correctly derived from the scene.",
            "No command-format, IK, collision, contact, or simulator compatibility test is performed here.",
            "An independent evaluator adapter is still required; the legacy executor reconstructs its own evaluation geometry.",
        ],
    }
    _write_exclusive(output, payload)
    return payload


def verify_submission(submission):
    submission = Path(submission).resolve()
    payload = _read(submission)
    if payload.get("schema") != "frozen-reconstruction-submission-v1":
        raise ProtocolError("Unknown frozen submission schema")
    if payload.get("submission_path") != str(submission):
        raise ProtocolError("Submission was relocated; do not create a new execution identity by copying a seal")
    for record in list(payload["files"].values()) + payload["evidence_files"]:
        _verify(record)
    report = load_manifest(payload["files"]["input_manifest"]["path"])
    require_execution_ready(report)
    if report["manifest_sha256"] != payload["input_protocol"]["manifest_sha256"]:
        raise ProtocolError("Input protocol changed after submission")
    return payload


def attach_evaluation(*, submission, evaluation_spec, receipt):
    """Record evaluator-only artifacts after verifying the frozen submission.

    evaluation_spec declares independent_scene=true and artifacts [{id,path,
    sha256}]. Artifact content is opaque to planning and read only in this
    post-seal operation. This validates hashes and the declared boundary, not
    whether a reference scene is genuinely independent ground truth.
    """
    payload = verify_submission(submission)
    if Path(payload["execution_receipt_path"]).exists():
        raise ProtocolError("Cannot attach a different evaluator after execution started")
    spec_path = Path(evaluation_spec).resolve()
    spec = _read(spec_path)
    if spec.get("schema_version") != 1 or spec.get("independent_scene") is not True:
        raise ProtocolError("Evaluator specification must explicitly declare an independent scene")
    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ProtocolError("Evaluator must declare its fixed reference artifacts")
    references, ids = [], set()
    planning_paths = {r["path"] for r in payload["files"].values()} | {r["path"] for r in payload["evidence_files"]}
    for artifact in artifacts:
        aid = artifact.get("id")
        if not isinstance(aid, str) or not aid or aid in ids:
            raise ProtocolError("Evaluator artifact IDs must be nonempty and unique")
        ids.add(aid)
        path = Path(artifact["path"])
        path = path if path.is_absolute() else spec_path.parent / path
        record = _record(path)
        if record["sha256"] != artifact.get("sha256"):
            raise ProtocolError(f"Evaluator reference changed: {aid}")
        if record["path"] in planning_paths:
            raise ProtocolError("Evaluator reference must not reuse a planner-input file as its independent scene")
        references.append(dict(artifact_id=aid, **record))
    receipt_payload = {
        "schema": "evaluation-attachment-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "submission": _record(submission), "evaluation_spec": _record(spec_path),
        "reference_artifacts": references, "plans_changed": False,
        "independence": "Declared independent reference, not independently proven by this tool",
        "simulation_executed": False,
    }
    _write_exclusive(receipt, receipt_payload)
    return receipt_payload


def claim_execution(*, submission, evaluation_receipt, output):
    """Consume one sealed submission before evaluator construction, without running it."""
    payload = verify_submission(submission)
    receipt = _read(evaluation_receipt)
    if receipt.get("schema") != "evaluation-attachment-v1" or receipt["submission"] != _record(submission):
        raise ProtocolError("Evaluation attachment does not match this frozen submission")
    for record in [receipt["evaluation_spec"], *receipt["reference_artifacts"]]:
        _verify(record)
    claim = {
        "schema": "frozen-submission-execution-claim-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "submission": _record(submission), "evaluation_attachment": _record(evaluation_receipt),
        "output": str(Path(output).resolve()),
        "note": "Claim made before simulator construction; infrastructure failure retains this receipt for audit.",
    }
    _write_exclusive(payload["execution_receipt_path"], claim)
    return claim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    for name in ("scene", "input-manifest", "plan", "plan-metadata", "planner", "executor", "controller-config", "output"):
        freeze.add_argument("--" + name, type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--submission", type=Path, required=True)
    attach = commands.add_parser("attach-evaluation")
    for name in ("submission", "evaluation-spec", "receipt"):
        attach.add_argument("--" + name, type=Path, required=True)
    claim = commands.add_parser("claim")
    for name in ("submission", "evaluation-receipt", "output"):
        claim.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    function = {"freeze": freeze_submission, "verify": verify_submission,
                "attach-evaluation": attach_evaluation, "claim": claim_execution}[command]
    print(json.dumps(function(**args), indent=2))


if __name__ == "__main__":
    main()
