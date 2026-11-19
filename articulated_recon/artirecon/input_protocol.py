"""Validate declared input lineage before reconstruction or frozen planning.

This validates a supplied provenance graph and file identity, not the truth of a
producer's claim about an artifact. No purity is inferred from filenames. The
method_inputs closure is the information available to the method; other records
may describe evaluator-only data but cannot influence that closure unnoticed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


class ProtocolError(ValueError):
    """Input provenance is missing, inconsistent or disallowed."""


CONDITIONS = {"rgb_only", "calibrated_rgb", "rgbd_oracle"}
KINDS = {"rgb", "depth", "cameras", "masks", "support", "hand_initialization",
         "object_geometry", "robot_registration", "model_prior"}
ORIGINS = {"observed_rgb", "model_prediction", "known_calibration", "sensor_depth",
           "ground_truth", "prior", "manual_rgb"}
PRIOR_TYPES = {"mano_model", "robot_urdf", "learned_model_weights", "generic_physics"}
ORACLE_ORIGINS = {"sensor_depth", "ground_truth"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail(message):
    raise ProtocolError(message)


def _ids(value, name, allow_empty=True):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        _fail(f"{name} must be a list of nonempty artifact IDs")
    if len(set(value)) != len(value) or (not value and not allow_empty):
        _fail(f"{name} must contain unique artifact IDs" + (" and cannot be empty" if not allow_empty else ""))
    return value


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _path(record, base_dir):
    path = Path(record["path"])
    return (path if path.is_absolute() else Path(base_dir or ".") / path).resolve()


def validate_manifest(manifest, base_dir=None, verify_files=False):
    """Return a JSON-serializable validation report or raise ProtocolError.

    Required top-level fields: schema_version=1, condition, task_mode,
    method_inputs and artifacts. Each artifact has id/kind/origin/parent_ids.
    Observed RGB records additionally declare generated=False and time_range_s.
    Forecasts declare observed_until_s. For materialized inputs, verify_files
    requires matching path/sha256 records throughout the consumed lineage.

    metric_scale and robot_registration declarations are optional during fitting.
    Missing/unresolved declarations make execution_ready False, never an implicit
    identity registration or known scale. Call require_execution_ready before
    exporting executable commands.
    """
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        _fail("Expected input manifest schema_version=1")
    condition = manifest.get("condition")
    if condition not in CONDITIONS:
        _fail(f"Unknown condition {condition!r}")
    task_mode = manifest.get("task_mode")
    if task_mode not in {"reconstruction", "forecast"}:
        _fail("task_mode must be reconstruction or forecast")
    cutoff = manifest.get("observed_until_s")
    if task_mode == "forecast" and not _finite(cutoff):
        _fail("Forecasts require finite observed_until_s")
    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        _fail("artifacts must be a nonempty list")
    by_id = {}
    for record in records:
        if not isinstance(record, dict):
            _fail("Each artifact must be an object")
        aid, kind, origin = record.get("id"), record.get("kind"), record.get("origin")
        if not isinstance(aid, str) or not aid or aid in by_id:
            _fail(f"Missing or duplicate artifact ID {aid!r}")
        if kind not in KINDS or origin not in ORIGINS:
            _fail(f"{aid}: unknown kind/origin {kind!r}/{origin!r}")
        parents = _ids(record.get("parent_ids"), f"{aid}.parent_ids")
        if aid in parents:
            _fail(f"{aid}: self dependency")
        if origin == "prior":
            if kind != "model_prior" or record.get("scope") != "global" or record.get("prior_type") not in PRIOR_TYPES:
                _fail(f"{aid}: priors must be declared global model_prior assets, never per-scene geometry/support")
            if parents:
                _fail(f"{aid}: a global prior cannot depend on this scene")
        elif kind == "model_prior":
            _fail(f"{aid}: model_prior must have origin=prior")
        if origin == "observed_rgb":
            if kind != "rgb" or record.get("generated") is not False or parents:
                _fail(f"{aid}: observed RGB must be a non-generated root, not a generated/derived view")
            interval = record.get("time_range_s")
            if not isinstance(interval, list) or len(interval) != 2 or not all(_finite(x) for x in interval) or interval[1] < interval[0]:
                _fail(f"{aid}: observed RGB requires ordered finite time_range_s")
        if kind == "rgb" and origin == "model_prediction" and record.get("generated") is not True:
            _fail(f"{aid}: predicted RGB must explicitly declare generated=true")
        if origin in {"model_prediction", "manual_rgb"} and not parents:
            _fail(f"{aid}: derived evidence requires parent_ids")
        if origin == "manual_rgb" and kind not in {"masks", "hand_initialization"}:
            _fail(f"{aid}: manual_rgb describes RGB annotations, not supplied 3D geometry or calibration")
        if origin == "sensor_depth" and kind != "depth":
            _fail(f"{aid}: sensor_depth must be depth; derived products need declared parent lineage")
        if origin == "known_calibration" and kind not in {"cameras", "support", "robot_registration"}:
            _fail(f"{aid}: known_calibration cannot label per-object geometry or hand poses")
        if "path" in record and (not isinstance(record["path"], str) or not record["path"]):
            _fail(f"{aid}: path must be nonempty text")
        if "sha256" in record and (not isinstance(record["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None):
            _fail(f"{aid}: sha256 must be 64 lowercase hexadecimal characters")
        by_id[aid] = record
    for aid, record in by_id.items():
        for parent in record["parent_ids"]:
            if parent not in by_id:
                _fail(f"{aid}: unknown parent {parent!r}")

    visiting, ancestors = set(), {}

    def visit(aid):
        if aid in visiting:
            _fail(f"Dependency cycle at {aid}")
        if aid not in ancestors:
            visiting.add(aid)
            lineage = {aid}
            for parent in by_id[aid]["parent_ids"]:
                lineage.update(visit(parent))
            visiting.remove(aid)
            ancestors[aid] = lineage
        return ancestors[aid]

    # Validate disconnected evaluator records too, without admitting them to the method.
    for aid in by_id:
        visit(aid)
    roots = _ids(manifest.get("method_inputs"), "method_inputs", allow_empty=False)
    if any(aid not in by_id for aid in roots):
        _fail("method_inputs references an unknown artifact")
    active = set().union(*(ancestors[aid] for aid in roots))
    forbidden = ORACLE_ORIGINS | ({"known_calibration"} if condition == "rgb_only" else set())
    if condition == "rgbd_oracle":
        forbidden = set()
    for aid in sorted(active):
        record = by_id[aid]
        origin = record["origin"]
        if origin in forbidden:
            _fail(f"{condition} method consumes forbidden {origin} artifact {aid}; downstream model labels do not erase ancestry")
        if task_mode == "forecast" and origin == "observed_rgb" and record["time_range_s"][1] > cutoff:
            _fail(f"Forecast consumes observed future RGB from {aid} after observed_until_s={cutoff}")
        if origin in {"model_prediction", "manual_rgb"}:
            evidence = {by_id[x]["origin"] for x in ancestors[aid] if x != aid}
            accepted = {"observed_rgb"} | (ORACLE_ORIGINS if condition == "rgbd_oracle" else set())
            if not evidence.intersection(accepted):
                _fail(f"{aid}: derived artifact has no permitted observation ancestor")
        if verify_files:
            if record["kind"] == "model_prior" and "path" not in record:
                continue
            if "path" not in record or "sha256" not in record:
                _fail(f"{aid}: materialized method input requires path and sha256")
            path = _path(record, base_dir)
            if not path.is_file():
                _fail(f"{aid}: input file missing: {path}")
            if file_sha256(path) != record["sha256"]:
                _fail(f"{aid}: SHA256 mismatch: {path}")
    if not any(by_id[x]["origin"] == "observed_rgb" for x in active):
        _fail("The method must include an observed RGB source")

    blockers = []
    states = {}
    for name in ("metric_scale", "robot_registration"):
        state = manifest.get(name)
        if state is None:
            states[name] = {"status": "unresolved", "artifact_ids": [], "description": "Not declared"}
            blockers.append(f"{name} is not declared")
            continue
        if not isinstance(state, dict) or state.get("status") not in {"estimated", "known", "calibrated", "unresolved"}:
            _fail(f"{name}: expected status estimated/known/calibrated/unresolved")
        references = _ids(state.get("artifact_ids"), name + ".artifact_ids")
        if any(x not in active for x in references):
            _fail(f"{name}: evidence must be in the method's consumed lineage")
        if not isinstance(state.get("description"), str) or not state["description"].strip():
            _fail(f"{name}: a description of the scale/registration procedure is required")
        resolved = state["status"] != "unresolved"
        if resolved and not references:
            _fail(f"{name}: resolved state requires evidence artifact_ids")
        if resolved and name == "robot_registration" and not any(by_id[x]["kind"] == "robot_registration" for x in references):
            _fail("robot_registration: evidence must include an explicit robot_registration artifact")
        if resolved and name == "metric_scale" and all(by_id[x]["kind"] == "model_prior" for x in references):
            _fail("metric_scale: an actor model prior alone does not locate the scene at metric scale")
        if state["status"] in {"known", "calibrated"}:
            evidence = set().union(*(ancestors[x] for x in references))
            if not any(by_id[x]["origin"] in ORACLE_ORIGINS | {"known_calibration"} for x in evidence):
                _fail(f"{name}: model-only evidence must be labeled estimated, not known/calibrated")
        if not resolved:
            blockers.append(f"{name} is unresolved")
        states[name] = state
    try:
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ProtocolError("Manifest must contain finite JSON-serializable values") from error
    return {
        "status": "validated", "schema_version": 1, "condition": condition,
        "task_mode": task_mode, "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "method_artifact_ids": sorted(active), "method_inputs": roots,
        "evaluation_only_artifact_ids": sorted(set(by_id) - active),
        "oracle_dependencies": sorted(x for x in active if by_id[x]["origin"] in ORACLE_ORIGINS),
        "calibration_dependencies": sorted(x for x in active if by_id[x]["origin"] == "known_calibration"),
        "manual_rgb_dependencies": sorted(x for x in active if by_id[x]["origin"] == "manual_rgb"),
        "metadata_only_prior_ids": sorted(x for x in active if by_id[x]["kind"] == "model_prior" and "path" not in by_id[x]),
        "files_verified": bool(verify_files), "execution_ready": not blockers,
        "execution_blockers": blockers, **states,
        "limitations": ["Validation checks declared lineage and file identity, not estimator accuracy or provenance truth.",
                        "Predicted confidence ranks are not calibrated uncertainty unless separately established."],
    }


def require_execution_ready(report):
    """Gate commands without inventing a metric scale or robot base transform."""
    if report.get("status") != "validated" or not report.get("files_verified") or not report.get("execution_ready"):
        _fail("Execution requires verified inputs, resolved metric scale and robot registration: " +
              "; ".join(report.get("execution_blockers", ["input protocol is unverified"])))


def load_manifest(path, required_inputs=()):
    """Validate file hashes and bind each (kind, path) consumed by a runner.

    Required paths refer to actual runner inputs, preventing an unrelated clean
    manifest from labeling an old calibrated bundle as RGB-only.
    """
    path = Path(path).resolve()
    with path.open() as stream:
        manifest = json.load(stream)
    report = validate_manifest(manifest, base_dir=path.parent, verify_files=True)
    active = set(report["method_artifact_ids"])
    bound = []
    for kind, actual in required_inputs:
        actual = Path(actual).resolve()
        matches = [r["id"] for r in manifest["artifacts"] if r["id"] in active and r["kind"] == kind
                   and "path" in r and _path(r, path.parent) == actual]
        if not matches:
            _fail(f"Runner consumes undeclared {kind} input {actual}")
        bound.append({"kind": kind, "path": str(actual), "artifact_ids": matches})
    report["manifest_path"] = str(path)
    report["bound_inputs"] = bound
    return report


def unverified_report():
    return {"status": "unverified", "condition": None, "files_verified": False,
            "execution_ready": False, "execution_blockers": ["No input manifest was supplied"],
            "description": "Legacy run; RGB-only provenance has not been validated."}
