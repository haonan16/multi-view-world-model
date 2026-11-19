"""Array-only export of native URDF fitter states to scene-sequence v1.

Native outputs are retained. This adapter does not infer timing, observations,
objects, contacts, metric scale, or target-robot registration.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .input_protocol import unverified_report
from .scene_sequence import SceneValidationError, actor_from_arrays, scene_from_arrays


def native_timestamps(result, metadata):
    """Preserve supplied second timestamps, or use an explicitly declared clock."""
    frames = len(result['robot']['q'])
    native, declared = result.get('timestamps'), metadata.get('timestamps')
    if native is not None:
        times = np.asarray(native, float)
        if declared is not None and not np.array_equal(times, np.asarray(declared, float)):
            raise SceneValidationError('Metadata timestamps disagree with native fit timestamps')
        method = 'Native fit timestamps in seconds, preserved without reindexing'
    elif declared is not None:
        times = np.asarray(declared, float)
        method = 'Explicit supplied timestamp metadata in seconds'
    elif 'control_hz' in metadata and 'steps' in result:
        hz = metadata['control_hz']
        if isinstance(hz, bool) or not isinstance(hz, (float, int)) or not np.isfinite(hz) or hz <= 0:
            raise SceneValidationError('Explicit control_hz must be positive and finite')
        times = np.asarray(result['steps'], float) / hz
        method = 'Native control-step indices divided by explicitly declared control_hz; control step zero is the epoch'
    else:
        raise SceneValidationError('No timestamps: supply native seconds, timestamp metadata, or explicit control_hz with native steps')
    if times.shape != (frames,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise SceneValidationError('Timestamps must be finite increasing seconds matching every fitted frame')
    return times, method


def urdf_fit_scene(*, q, base, joint_names, model_uri, timestamps, observed=None,
                   coordinate_frame=None, input_protocol=None, provenance=None,
                   support_normal=None, support_offset=None, timestamp_method='Explicit supplied timestamps in seconds'):
    """Convert existing states without changing order, transforms or timing.

    Missing observation flags remain false with an explicit unknown-evidence
    label. A finite optimized joint vector alone is not evidence of observation.
    """
    report = deepcopy(input_protocol if input_protocol is not None else unverified_report())
    verified = report.get('status') == 'validated' and report.get('files_verified') is True
    frames = len(timestamps)
    unknown_evidence = observed is None
    if unknown_evidence: observed = np.zeros(frames, bool)
    else:
        observed = np.asarray(observed)
        if observed.shape != (frames,) or observed.dtype != np.dtype(bool):
            raise SceneValidationError('Robot observed must be an explicit boolean [T] array')
    fit_provenance = deepcopy(provenance or {})
    fit_provenance.update(method=fit_provenance.get('method', 'Native Amani URDF RobotFitter state export'),
        input_protocol_status=report.get('status', 'unverified'), condition=report.get('condition'),
        files_verified=bool(report.get('files_verified', False)), artifact_ids=report.get('method_artifact_ids', []),
        observed_definition='Unknown original evidence; states retained but not asserted observed' if unknown_evidence else 'Explicit source-fitter observation flags preserved')
    scale = report.get('metric_scale', {})
    status = scale.get('status', 'unresolved') if verified else 'unresolved'
    if status == 'calibrated': status = 'known'
    if status not in {'known', 'estimated', 'unresolved'}: status = 'unresolved'
    metric_scale = dict(status=status, provenance=dict(method=scale.get('description', 'Metric scale has no verified input declaration'),
        artifact_ids=scale.get('artifact_ids', []) if verified else [], source_status=scale.get('status', 'unresolved'), files_verified=verified))
    frame = deepcopy(coordinate_frame) if coordinate_frame is not None else dict(name='legacy_native_frame_unverified',
        handedness='right', up_axis='unknown', pose_convention='T_world_local')
    support = None
    if (support_normal is None) != (support_offset is None): raise SceneValidationError('Supply support normal and offset together')
    if support_normal is not None:
        support = dict(id='table', normal_world=support_normal, offset_m=support_offset, confidence=None,
            provenance={**fit_provenance, 'method':'Native fitted support plane, exported without transformation', 'confidence_status':'uncalibrated'})
    actor = actor_from_arrays(actor_id='robot', actor_type='urdf', q=q, base=base, observed=observed,
        joint_names=list(joint_names), model_uri=str(model_uri), provenance=fit_provenance)
    scene = scene_from_arrays(timestamps=timestamps, coordinate_frame=frame, metric_scale=metric_scale,
        actors=[actor], objects=[], support=support, contacts=[], uncertainty=dict(status='unestimated'),
        task=dict(description='Preserve the fitted articulated robot over the observed sequence', target=None))
    scene['input_protocol'] = report
    scene['timestamps_provenance'] = dict(method=timestamp_method)
    scene['export_scope'] = dict(actor_only=True, objects_exported=False, contacts_inferred=False,
        robot_registration_established_by_export=False, unknown_original_observation_evidence=unknown_evidence,
        limitations=['URDF state export only; native object/articulation outputs remain in their original files.',
                     'A fitted robot base is preserved as a pose; this exporter does not certify deployment registration.'])
    return scene


def export_native_urdf(result, robot, directory, *, metadata_path=None, overwrite=False):
    """Hook for native runners; lack of metadata produces an explicit receipt.

    Optional scene-export-metadata.json may supply timestamps/control_hz and a
    coordinate frame/provenance. It cannot upgrade a legacy run to verified RGB.
    Only the native result's existing validated input_protocol can do that.
    """
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    output = directory / 'scene-sequence.json'
    receipt_path = directory / 'shared-scene-export.json'
    metadata = deepcopy(result.get('shared_scene_metadata', {}))
    metadata_path = Path(metadata_path) if metadata_path is not None else directory / 'scene-export-metadata.json'
    try:
        if not isinstance(metadata, dict): raise SceneValidationError('Native shared_scene_metadata must be an object')
        if metadata_path.exists():
            supplied = json.loads(metadata_path.read_text())
            if not isinstance(supplied, dict): raise SceneValidationError('Scene export metadata must be an object')
            metadata.update(supplied)
        if output.exists() and not overwrite:
            raise SceneValidationError('Shared scene already exists; explicit overwrite is required')
        timestamps, method = native_timestamps(result, metadata)
        fitted = result['robot']
        names = [entry['name'] for entry in robot.active]
        normal, offset = result.get('plane', (None, None))
        scene = urdf_fit_scene(q=fitted['q'], base=fitted['base'], joint_names=names, model_uri=robot.urdf,
            timestamps=timestamps, observed=fitted.get('observed'),
            coordinate_frame=result.get('coordinate_frame', metadata.get('coordinate_frame')),
            input_protocol=result.get('input_protocol'), provenance=result.get('provenance', metadata.get('provenance')),
            support_normal=normal, support_offset=offset, timestamp_method=method)
        scene['native_source'] = dict(label=result.get('label'), condition=result.get('condition'),
            control_step_indices=np.asarray(result['steps']).tolist() if 'steps' in result else None,
            native_object_count=len(result.get('objects', [])))
        output.write_text(json.dumps(scene, indent=2, allow_nan=False))
        receipt = dict(status='exported', path=str(output), frames=len(timestamps), actor_type='urdf',
            actor_only=True, native_objects_preserved_in_original_outputs=True, original_fit_modified=False,
            timestamp_method=method, input_protocol_status=scene['input_protocol'].get('status'),
            metric_scale_status=scene['metric_scale']['status'], robot_registration_established=False)
    except (SceneValidationError, KeyError, ValueError, TypeError) as error:
        receipt = dict(status='not_exported', reason=str(error), original_fit_modified=False,
            existing_shared_scene_preserved=output.exists(), existing_scene_must_not_be_treated_as_current=output.exists(),
            metadata_required='Actual seconds or explicit control_hz; optional coordinate_frame/provenance; no timing/calibration guesses')
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False))
    return receipt
