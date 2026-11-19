"""Array-only adapters from fitted states to the shared scene contract.

This exports an existing fit. It performs no model inference, object discovery,
contact inference, registration, or metric-scale recovery.
"""
from copy import deepcopy

import numpy as np

from .input_protocol import unverified_report
from .scene_sequence import SceneValidationError, actor_from_arrays, scene_from_arrays


def fit_timestamps(observations, frames, fps, *, require_supplied=False):
    """Use supplied second timestamps, or explicitly declare an FPS clock.

    The legacy observation NPZ convention calls timestamps in seconds
    ``timestamps``. Malformed supplied times are rejected, never replaced.
    """
    if not isinstance(frames, (int, np.integer)) or isinstance(frames, bool) or frames <= 0:
        raise SceneValidationError("frames must be a positive integer")
    if "timestamps" in observations:
        try:
            values = np.asarray(observations["timestamps"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise SceneValidationError("timestamps must be numeric seconds") from exc
        if values.ndim != 1:
            raise SceneValidationError("timestamps must be a [T] sequence in seconds")
        values = values[:frames]
        method = "observation timestamps in seconds"
    else:
        if require_supplied:
            raise SceneValidationError("A verified input run requires observation timestamps; no implicit FPS clock")
        if not isinstance(fps, (float, int, np.floating, np.integer)) or isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0:
            raise SceneValidationError("fps must be finite and positive when timestamps are absent")
        values = np.arange(frames, dtype=float) / fps
        method = "frame index divided by declared FPS; original acquisition timestamps unavailable"
    if values.shape != (frames,) or not np.isfinite(values).all() or not (np.diff(values) > 0).all():
        raise SceneValidationError("timestamps must be finite [T] and strictly increasing")
    return values, method


def declared_coordinate_frame(protocol, require_explicit=False):
    """Preserve the frame named by input preparation; never infer registration."""
    frame=protocol.get('coordinate_frame')
    if frame is None:
        if require_explicit:
            raise SceneValidationError('Verified hand inputs require an explicit coordinate_frame')
        return {'name':'reconstruction_world','handedness':'right','up_axis':'unknown','pose_convention':'T_world_local'}
    if (not isinstance(frame,dict) or not isinstance(frame.get('name'),str) or not frame['name'].strip()
            or frame.get('handedness')!='right' or frame.get('pose_convention')!='T_world_local'
            or frame.get('up_axis') not in {'+x','-x','+y','-y','+z','-z','unknown'}):
        raise SceneValidationError('Invalid declared hand coordinate_frame')
    return deepcopy(frame)


def mano_fit_scene(*, q, base, betas, timestamps, point_counts, visible,
                   model_uri, input_protocol=None, support_normal=None,
                   support_offset=None, timestamp_method="supplied timestamps in seconds",
                   task_description="Track a human hand through the observed sequence",
                   coordinate_frame=None):
    """Export MANO + an optional supplied table without inventing observations.

    ``visible`` alone is not evidence: a frame is observed only when visibility
    is true and at least one fitting point exists. Valid states from other frames
    remain in the track but are explicitly unobserved. The local licensed model
    is referenced by URI and is never copied into this output.
    """
    report = deepcopy(input_protocol if input_protocol is not None else unverified_report())
    frames = len(timestamps)
    counts = np.asarray(point_counts)
    visibility = np.asarray(visible)
    if counts.shape != (frames,) or not (np.issubdtype(counts.dtype, np.integer) or np.issubdtype(counts.dtype, np.floating)):
        raise SceneValidationError("point_counts must be a numeric [T] array")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise SceneValidationError("point_counts must contain finite nonnegative integers")
    if visibility.shape != (frames,) or visibility.dtype != np.dtype(bool):
        raise SceneValidationError("visible must be an explicit boolean [T] array")
    observed = (counts > 0) & visibility
    verified = report.get("status") == "validated" and report.get("files_verified") is True
    provenance = {
        "method": "Amani RobotFitter MANO port; fitted shared shape and time-varying hand state",
        "input_protocol_status": report.get("status", "unverified"),
        "condition": report.get("condition"),
        "files_verified": bool(report.get("files_verified", False)),
        "artifact_ids": report.get("method_artifact_ids", []),
        "observed_definition": "point_counts > 0 AND visible; evidence presence is not accuracy",
    }
    declared_scale = report.get("metric_scale", {})
    scale_status = declared_scale.get("status", "unresolved") if verified else "unresolved"
    if scale_status == "calibrated":
        scale_status = "known"
    if scale_status not in {"known", "estimated", "unresolved"}:
        scale_status = "unresolved"
    scale = {
        "status": scale_status,
        "provenance": {
            "method": declared_scale.get("description", "Metric scale not established by a verified input declaration"),
            "artifact_ids": declared_scale.get("artifact_ids", []) if verified else [],
            "input_protocol_status": report.get("status", "unverified"),
            "files_verified": verified,
            "source_status": declared_scale.get("status", "unresolved"),
        },
    }
    support = None
    if (support_normal is None) != (support_offset is None):
        raise SceneValidationError("support normal and offset must be supplied together")
    if support_normal is not None:
        support = {
            "id": "table", "normal_world": support_normal, "offset_m": support_offset,
            "confidence": None,
            "provenance": {
                **deepcopy(provenance),
                "method": "Support plane consumed by the hand fitter; no plane confidence calibrated",
                "confidence_status": "uncalibrated",
                "plane_convention": "normal_world dot x + offset_m = 0",
            },
        }
    actor = actor_from_arrays(
        actor_id="hand", actor_type="mano", q=q, base=base, shared_shape=betas,
        model_uri=str(model_uri), observed=observed, provenance=provenance)
    scene = scene_from_arrays(
        timestamps=timestamps,
        coordinate_frame=declared_coordinate_frame({'coordinate_frame':coordinate_frame}),
        metric_scale=scale, actors=[actor], objects=[], support=support, contacts=[],
        uncertainty={"status": "unestimated"},
        task={"description": task_description, "target": None})
    scene["input_protocol"] = report
    scene["timestamps_provenance"] = {"method": timestamp_method}
    scene["export_scope"] = {
        "actor_only": True, "objects_exported": False, "contacts_inferred": False,
        "robot_registration_established_by_export": False,
        "limitations": ["MANO hand-state export only; not an executed robot plan.",
                        "Support is an input plane, not a reconstructed finite table mesh."],
    }
    return scene
