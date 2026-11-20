"""Evaluator-only Pi4 depth audit; never aligns or modifies model predictions.

Seal completed predictions before opening references. Measured color-grid depth
is resized with nearest-neighbor pixel-centre sampling; holes remain missing.
Confidence selection uses predictions alone, before seeing reference validity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
from artirecon.input_protocol import file_sha256, load_manifest
from depth_confidence_audit import error_metrics
from pi4_frontend import resize_rgb


PREDICTION_FILES = ('inference.json', 'input-manifest.json', 'geometry.npz')


def write_report(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _stable(files):
    for path, digest in files.items():
        if file_sha256(path) != digest:
            raise ValueError('File changed after sealing or during evaluation: ' + str(path))


def _prediction_identity(directory):
    directory = Path(directory).resolve()
    execution = json.loads((directory / 'inference.json').read_text())
    if (execution.get('status') != 'complete' or execution.get('geometry_schema') != 'pi4_dynamic_camera_v1'
            or execution.get('measured_depth_used') is not False
            or execution.get('known_intrinsics_supplied') is not False
            or execution.get('reference_alignment') is not False):
        raise ValueError('Require completed, unaligned RGB-only Pi4 inference')
    validation = load_manifest(directory / 'input-manifest.json',
        [('depth', directory / 'geometry.npz'), ('cameras', directory / 'geometry.npz')])
    if validation['condition'] != 'rgb_only' or validation['task_mode'] != 'reconstruction':
        raise ValueError('Require RGB-only reconstruction provenance')
    manifest = json.loads((directory / 'input-manifest.json').read_text())
    roots = [a for a in manifest['artifacts'] if a['id'] in validation['method_artifact_ids']
             and a['origin'] == 'observed_rgb']
    if len(roots) != 1 or roots[0]['id'] != 'source_rgb':
        raise ValueError('Require exactly one authenticated observed source RGB bundle')
    source_path = (directory / roots[0]['path']).resolve()
    files = {str(directory / name): file_sha256(directory / name) for name in PREDICTION_FILES}
    files[str(source_path)] = roots[0]['sha256']
    _stable(files)
    return execution, roots[0], source_path, files


def seal_predictions(predictions, output):
    """Read only completed model files; no evaluator reference is accepted here."""
    directory = Path(predictions).resolve()
    execution, source, _, files = _prediction_identity(directory)
    seal = dict(schema='pi4-evaluation-seal-v1', status='complete',
        prediction_directory=str(directory), files=files, source_rgb_sha256=source['sha256'],
        source_manifest_sha256=execution['source_manifest_sha256'],
        source_frame_indices=execution['source_frame_indices'],
        reference_accessed=False, purpose='Freeze completed predictions before evaluator reference access')
    write_report(output, seal)
    return seal


def bind_predictions(predictions, seal_path):
    directory, seal_path = Path(predictions).resolve(), Path(seal_path).resolve()
    seal = json.loads(seal_path.read_text())
    required = {str(directory / name) for name in PREDICTION_FILES}
    if (seal.get('schema') != 'pi4-evaluation-seal-v1' or seal.get('status') != 'complete'
            or seal.get('prediction_directory') != str(directory)
            or seal.get('reference_accessed') is not False
            or not required <= set(seal.get('files', {}))):
        raise ValueError('Missing valid completed prediction seal')
    _stable(seal['files'])
    execution, source, source_path, files = _prediction_identity(directory)
    if (files != seal['files'] or source['sha256'] != seal['source_rgb_sha256']
            or execution['source_manifest_sha256'] != seal['source_manifest_sha256']
            or execution['source_frame_indices'] != seal['source_frame_indices']):
        raise ValueError('Prediction identity differs from seal')
    files[str(seal_path)] = file_sha256(seal_path)
    with np.load(directory / 'geometry.npz', allow_pickle=False) as z:
        geometry = {k: z[k].copy() for k in ['rgb', 'timestamps', 'source_frame_indices',
            'local_points_m', 'depth_m', 'depth_valid', 'confidence', 'c2w']}
    with np.load(source_path, allow_pickle=False) as z:
        if set(z.files) != {'rgb', 'timestamps'}:
            raise ValueError('Source bundle contains non-RGB inputs')
        source_rgb, source_times = z['rgb'].copy(), z['timestamps'].copy()
    ids = np.asarray(geometry['source_frame_indices'])
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
            or np.any(ids < 0) or np.any(ids >= len(source_rgb)) or np.any(np.diff(ids) <= 0)
            or ids.tolist() != execution['source_frame_indices']
            or not np.array_equal(geometry['timestamps'], source_times[ids])
            or not np.array_equal(geometry['timestamps'], execution['timestamps_s'])):
        raise ValueError('Source frame IDs/timestamps differ from frozen inference')
    expected_rgb = resize_rgb(source_rgb[ids], execution['max_size'])
    if not np.array_equal(geometry['rgb'], expected_rgb[:, None]):
        raise ValueError('Predicted RGB grid differs from the exact declared source resize')
    depth = geometry['depth_m']
    shape = expected_rgb.shape[:3]
    if (depth.shape != (shape[0], 1, *shape[1:]) or geometry['local_points_m'].shape != depth.shape + (3,)
            or geometry['confidence'].shape != depth.shape or geometry['depth_valid'].shape != depth.shape
            or geometry['depth_valid'].dtype != bool
            or not np.array_equal(depth, geometry['local_points_m'][..., 2], equal_nan=True)):
        raise ValueError('Expected native metric local-camera Z on the model RGB grid')
    return execution, geometry, source, source_rgb.shape[1:3], files


def confidence_top_half(confidence, model_valid):
    """Midrank percentile >= .5, over model-valid pixels only; retain ties.

    Actual retained counts are reported. Equal scores can retain more than half;
    neither sensor depth validity nor semantic categories change the threshold.
    """
    c, valid = np.asarray(confidence), np.asarray(model_valid)
    if c.shape != valid.shape or valid.dtype != bool:
        raise ValueError('Confidence/validity grids differ')
    eligible = valid & np.isfinite(c) & (c >= 0) & (c <= 1)
    ranks = np.full(c.shape, np.nan)
    values = c[eligible]
    if len(values):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        # Average 1-based rank minus .5, divided by N: pixel-centred percentiles.
        midrank = (np.cumsum(counts) - counts / 2) / len(values)
        ranks[eligible] = midrank[inverse]
    return eligible & (ranks >= .5), ranks


def resize_reference(values, output_hw):
    """Nearest pixel-centre resampling; never interpolate depth across holes."""
    a = np.asarray(values)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError('Reference raster must be a two-dimensional integer image')
    h, w = output_hw
    if h < 1 or w < 1:
        raise ValueError('Invalid target raster dimensions')
    yy = np.minimum(np.floor((np.arange(h) + .5) * a.shape[0] / h).astype(int), a.shape[0] - 1)
    xx = np.minimum(np.floor((np.arange(w) + .5) * a.shape[1] / w).astype(int), a.shape[1] - 1)
    return a[yy[:, None], xx[None, :]]


def score_category(prediction, reference, category, model_valid, high):
    sensor_valid = np.isfinite(reference) & (reference > 0)
    paired = category & model_valid & sensor_valid
    selected = paired & high
    denominator = int(category.sum())
    return dict(category_pixels=denominator, sensor_valid_pixels=int((category & sensor_valid).sum()),
        model_valid_pixels=int((category & model_valid).sum()), paired_pixels=int(paired.sum()),
        paired_fraction=float(paired.sum() / denominator) if denominator else None,
        high_confidence_model_pixels=int((category & model_valid & high).sum()),
        all_valid=error_metrics(prediction[paired], reference[paired]),
        confidence_top50=error_metrics(prediction[selected], reference[selected]))


def camera_drift(c2w, timestamps, *, fixed_camera_documented=False):
    c = np.asarray(c2w, float)
    if c.shape != (len(timestamps), 1, 4, 4) or not np.isfinite(c).all():
        raise ValueError('Expected finite time-varying metric camera transforms')
    c = c[:, 0]
    r = c[:, :3, :3]
    if (not np.allclose(c[:, 3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(r.swapaxes(-1, -2) @ r, np.eye(3), atol=2e-3)
            or not np.allclose(np.linalg.det(r), 1, atol=2e-3)):
        raise ValueError('Camera drift requires proper SE3, never repaired poses')
    relative = np.linalg.inv(c[0])[None] @ c
    distance = np.linalg.norm(relative[:, :3, 3], axis=-1)
    angle = np.degrees(np.arccos(np.clip((np.trace(relative[:, :3, :3], axis1=1, axis2=2) - 1)/2, -1, 1)))
    return dict(status='fixed_camera_consistency_diagnostic' if fixed_camera_documented else 'predicted_camera_motion_only',
        fixed_camera_assumption=fixed_camera_documented, no_dataset_camera_used=True,
        max_translation_m=float(distance.max()), final_translation_m=float(distance[-1]),
        max_rotation_deg=float(angle.max()), final_rotation_deg=float(angle[-1]),
        per_frame=[dict(timestamp_s=float(t), translation_m=float(d), rotation_deg=float(a))
                   for t, d, a in zip(timestamps, distance, angle)],
        limitation='Relative to predicted first camera only; drift is an error diagnostic only under the documented fixed-camera assumption.')


def _reference_path(base, name):
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError('Reference path escapes its verified directory')
    return path


def evaluate(*, predictions, seal, reference, output):
    # No reference opened until completed predictions and all bound bytes pass.
    execution, geometry, source, native_hw, stable = bind_predictions(predictions, seal)
    reference = Path(reference).resolve()
    ref = json.loads(reference.read_text())
    stable[str(reference)] = file_sha256(reference)
    stable[str(Path(__file__).resolve())] = file_sha256(__file__)
    if (ref.get('schema') != 'hocap-depth-reference-v1' or ref.get('status') != 'verified'
            or ref.get('evaluator_only') is not True or ref.get('convention_verified') is not True
            or ref.get('depth_convention') != 'color-camera optical-axis Z'
            or ref.get('depth_unit') != 'millimeter' or ref.get('scale_to_m') != .001
            or ref.get('invalid_depth_value') != 0):
        raise ValueError('Require verified original aligned camera-Z millimetre depth reference')
    if (ref['source_rgb_bundle_sha256'] != source['sha256']
            or ref['source_input_manifest_sha256'] != execution['source_manifest_sha256']
            or ref['image_size'] != list(native_hw[::-1])):
        raise ValueError('Reference source identity or original RGB grid differs')
    # Source-manifest.json is the curator's provenance receipt. It is distinct
    # from the method's input-manifest.json hashed by the Pi4 producer.
    if not isinstance(ref.get('units_proof'), list) or not ref['units_proof']:
        raise ValueError('Verified depth units require pinned official-source evidence')
    for record in ref['units_proof'] + ([ref['metadata']] if ref.get('metadata') else []):
        # Curated shared format evidence is outside the individual case folder;
        # paths are reference-only, hash-bound and never added to method inputs.
        path = (reference.parent / record['path']).resolve()
        if file_sha256(path) != record['sha256']:
            raise ValueError('Reference convention/metadata evidence hash mismatch')
        stable[str(path)] = record['sha256']
    frames = ref['frames']
    ids = [f['source_frame_index'] for f in frames]
    originals = [f['original_frame_id'] for f in frames]
    times = [f['timestamp_s'] for f in frames]
    if (ids != ref['source_frame_indices'] or originals != ref['original_frame_ids'] or times != ref['timestamps_s']
            or len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids)
            or len(set(o - i for o, i in zip(originals, ids))) != 1):
        raise ValueError('Reference frame mapping is inconsistent')
    by_id = dict(zip(ids, frames))
    category_ids = ref.get('segmentation_categories')
    if category_ids is not None:
        if (ref.get('segmentation_verified') is not True or not isinstance(category_ids, dict)
                or not category_ids or 'all' in category_ids
                or any(not isinstance(v, list) or not v or any(type(i) is not int for i in v) for v in category_ids.values())):
            raise ValueError('Semantic categories require verified explicit publisher label IDs')
        flat = [i for values in category_ids.values() for i in values]
        if len(set(flat)) != len(flat):
            raise ValueError('Semantic category label IDs overlap')
    rows, missing, collected = [], [], {}
    for j, source_id in enumerate(geometry['source_frame_indices']):
        frame = by_id.get(int(source_id))
        if frame is None:
            missing.append(int(source_id))
            rows.append(dict(source_frame_index=int(source_id), status='reference_unavailable'))
            continue
        if frame['timestamp_s'] != float(geometry['timestamps'][j]):
            raise ValueError('Exact source/reference timestamp mismatch; no nearest-frame matching')
        path = _reference_path(reference.parent, frame['depth_path'])
        if file_sha256(path) != frame['depth_sha256']:
            raise ValueError('Reference depth hash mismatch')
        stable[str(path)] = frame['depth_sha256']
        with Image.open(path) as image:
            original = np.asarray(image).copy()
        if original.shape != tuple(native_hw) or original.dtype != np.uint16:
            raise ValueError('Expected original aligned uint16 depth PNG on the RGB grid')
        prediction = geometry['depth_m'][j, 0]
        measured = resize_reference(original, prediction.shape).astype(np.float64) * ref['scale_to_m']
        model_valid = geometry['depth_valid'][j, 0] & np.isfinite(prediction) & (prediction > 0)
        high, ranks = confidence_top_half(geometry['confidence'][j, 0], model_valid)
        categories = {'all': np.ones(prediction.shape, bool)}
        labels_status = 'unavailable: no verified publisher segmentation supplied'
        if category_ids is not None and frame.get('segmentation_path'):
            lp = _reference_path(reference.parent, frame['segmentation_path'])
            if file_sha256(lp) != frame['segmentation_sha256']:
                raise ValueError('Reference segmentation hash mismatch')
            stable[str(lp)] = frame['segmentation_sha256']
            with np.load(lp, allow_pickle=False) as z:
                labels = z[frame['segmentation_key']].copy()
            if labels.shape != tuple(native_hw):
                raise ValueError('Segmentation does not share the original source grid')
            labels = resize_reference(labels, prediction.shape)
            categories.update({name: np.isin(labels, values) for name, values in category_ids.items()})
            labels_status = 'verified publisher IDs; undeclared IDs excluded from semantic categories'
        scores = {name: score_category(prediction, measured, mask, model_valid, high) for name, mask in categories.items()}
        for name, mask in categories.items():
            paired = mask & model_valid & np.isfinite(measured) & (measured > 0)
            bucket = collected.setdefault(name, dict(all_p=[], all_r=[], high_p=[], high_r=[]))
            bucket['all_p'].append(prediction[paired]); bucket['all_r'].append(measured[paired])
            bucket['high_p'].append(prediction[paired & high]); bucket['high_r'].append(measured[paired & high])
        rows.append(dict(source_frame_index=int(source_id), original_frame_id=frame['original_frame_id'],
            timestamp_s=frame['timestamp_s'], status='evaluated', labels_status=labels_status,
            model_confidence_eligible_pixels=int(np.isfinite(ranks).sum()),
            high_confidence_pixels=int(high.sum()), categories=scores))
    aggregate = {}
    for name, bucket in collected.items():
        metrics = {label: error_metrics(np.concatenate(bucket[p]), np.concatenate(bucket[r]))
                   for label, p, r in [('all_valid', 'all_p', 'all_r'), ('confidence_top50', 'high_p', 'high_r')]}
        frames_with_category = [row['categories'][name] for row in rows if name in row.get('categories', {})]
        metrics['frames_evaluated'] = len(frames_with_category)
        metrics['coverage_counts'] = {key: sum(row[key] for row in frames_with_category)
            for key in ['category_pixels', 'sensor_valid_pixels', 'model_valid_pixels', 'paired_pixels', 'high_confidence_model_pixels']}
        for label in ['all_valid', 'confidence_top50']:
            values = [row[label]['mae_mm'] for row in frames_with_category if row[label]['pixels']]
            metrics[label]['mean_frame_mae_mm'] = float(np.mean(values)) if values else None
        aggregate[name] = metrics
    fixed = ref.get('fixed_camera') is True and bool(ref.get('fixed_camera_provenance'))
    report = dict(schema='pi4-measured-depth-audit-v1', evaluation_only=True,
        status='complete' if not missing else 'complete_with_missing_references',
        predictions=str(Path(predictions).resolve()), reference=str(reference),
        sequence=ref['sequence'], source_camera=ref['source_camera'],
        frames_requested=len(rows), frames_evaluated=len(rows) - len(missing), missing_source_frame_indices=missing,
        categories=aggregate, per_frame=rows,
        semantic_categories_status='verified categories on available labeled frames' if category_ids is not None else 'unavailable',
        camera_drift=camera_drift(geometry['c2w'], geometry['timestamps'], fixed_camera_documented=fixed),
        fixed_camera_provenance=ref.get('fixed_camera_provenance'),
        source_provenance_manifest_sha256=ref.get('source_manifest_sha256'),
        source_input_manifest_sha256=ref['source_input_manifest_sha256'],
        depth_comparison='Native Pi4 local camera Z in metres; original aligned depth PNG multiplied by0.001; no scaling/alignment fit',
        reference_resampling='Nearest pixel-centre sampling to model RGB size; no averaging or hole filling',
        confidence_policy='Per-frame model-valid finite scores only; midrank percentile>=0.5; retain ties; threshold precedes sensor validity and semantic masks',
        confidence_is_calibrated_metric_certainty=False, no_scale_correction=True, no_pose_alignment=True,
        predictions_modified=False, refinement_performed=False, simulator_used=False,
        limitations=['Sensor depth has missing pixels, boundary noise and finite accuracy; it is not exact geometry.',
                     'Depth and camera drift are separate diagnostics, not physical execution/contact success.',
                     'Shared clip pixels/frames are correlated; this is not an independent-sample statistical test.'],
        verified_input_hashes=stable, evaluator_sha256=file_sha256(__file__))
    _stable(stable)
    write_report(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('seal')
    a.add_argument('--predictions', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('evaluate')
    for name in ['predictions', 'seal', 'reference', 'output']:
        a.add_argument('--' + name, type=Path, required=True)
    args = vars(p.parse_args()); command = args.pop('command')
    result = seal_predictions(**args) if command == 'seal' else evaluate(**args)
    print(json.dumps({key: result[key] for key in ['schema', 'status']}, indent=2))


if __name__ == '__main__':
    main()
