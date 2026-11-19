"""Diagnostic: compress saved predicted poses, never score them as ground truth.

This tests retaining five states AFTER fitting. It does not measure sparse
world-model generation, reconstruction runtime savings, or robot success.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.scene_sequence import object_from_arrays, scene_from_arrays, select_keyframes


def interpolate_poses(poses, timestamps, indices):
    times = np.asarray(timestamps, float)
    ids = np.asarray(indices, int)
    if len(ids) < 2 or ids[0] != 0 or ids[-1] != len(times) - 1 or np.any(np.diff(ids) <= 0):
        raise ValueError('Retain both endpoints and strictly increasing keyframe indices')
    result = np.tile(np.eye(4), (len(times), 1, 1))
    result[:, :3, :3] = Slerp(times[ids], Rotation.from_matrix(poses[ids, :3, :3]))(times).as_matrix()
    for k in range(3):
        result[:, k, 3] = np.interp(times, times[ids], poses[ids, k, 3])
    return result


def intervals(mask):
    result, start = [], None
    for i, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = i
        if not value and start is not None:
            result.append((start, i - 1))
            start = None
    return result


def errors(reference, candidate):
    translation = np.linalg.norm(reference[:, :3, 3] - candidate[:, :3, 3], axis=1) * 1000
    rotation = (Rotation.from_matrix(reference[:, :3, :3]).inv() *
                Rotation.from_matrix(candidate[:, :3, :3])).magnitude() * 180 / np.pi
    return {'translation_mean_mm': float(translation.mean()), 'translation_max_mm': float(translation.max()),
            'rotation_mean_deg': float(rotation.mean()), 'rotation_max_deg': float(rotation.max())}


def geometry_keyframes(poses, timestamps, vertices, budget):
    """Retrospective compression: greedily preserve bounding-box point motion.

    This consumes the entire fitted trajectory. It is not a predictor of which
    frames to generate before the world model has run.
    """
    if budget < 2:
        raise ValueError('Retain at least two keyframes')
    import itertools
    corners = np.asarray(list(itertools.product(*zip(vertices.min(0), vertices.max(0)))))
    actual = corners @ poses[:, :3, :3].transpose(0, 2, 1) + poses[:, None, :3, 3]
    ids = [0, len(poses) - 1]
    while len(ids) < min(budget, len(poses)):
        approx = interpolate_poses(poses, timestamps, ids)
        predicted = corners @ approx[:, :3, :3].transpose(0, 2, 1) + approx[:, None, :3, 3]
        residual = np.linalg.norm(actual - predicted, axis=2).max(axis=1)
        residual[ids] = -1
        ids = sorted(ids + [int(np.argmax(residual))])
    return ids


def analyze(trajectory, budget=5):
    path = Path(trajectory).resolve()
    original_scene = json.loads(path.with_name('scene.json').read_text())
    object_data = original_scene['objects'][0]
    with np.load(path, allow_pickle=False) as data:
        poses, timestamps = data['poses'].copy(), data['timestamps'].copy()
        valid = data['valid'].astype(bool)
        contact = data['contact_prior'].astype(bool) if 'contact_prior' in data.files else None
    result = {'trajectory': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_description': original_scene.get('source'), 'frames': len(poses),
              'valid_frames': int(valid.sum()), 'condition': 'historical_calibrated_diagnostic',
              'ground_truth_accuracy_measured': False, 'simulation_executed': False,
              'generation_or_fitting_speedup_measured': False}
    if not valid.all():
        result.update(status='unsupported_frames', invalid_frame_indices=np.flatnonzero(~valid).tolist(),
                      reason='Visualization-filled poses cannot become valid planning keyframes')
        return result
    name = object_data['name']
    provenance = {'method': 'Previously saved predicted fit; supplied cameras/support',
                  'confidence_status': 'uncalibrated'}
    object_state = object_from_arrays(object_id=name, poses=poses, observed=valid,
                                     provenance=provenance, vertices_m=object_data['vertices'],
                                     faces=object_data['faces'])
    contacts = []
    if contact is not None:
        contacts += [{'id': f'support-{i}', 'a': name, 'b': 'world', 'kind': 'support',
                      'start_frame': start, 'end_frame': end, 'status': 'hypothesized',
                      'confidence': None, 'provenance': provenance}
                     for i, (start, end) in enumerate(intervals(contact))]
    held = np.asarray(object_data.get('held', [False] * len(poses)), bool)
    known = np.asarray(object_data.get('held_known', [False] * len(poses)), bool)
    # The legacy files have no fitted human hand model here. "world" is an
    # anonymous interaction counterpart, not a fabricated hand reconstruction.
    contacts += [{'id': f'held-{i}', 'a': name, 'b': 'world', 'kind': 'grasp',
                  'start_frame': start, 'end_frame': end, 'status': 'hypothesized',
                  'confidence': None, 'provenance': provenance}
                 for i, (start, end) in enumerate(intervals(held & known))]
    scene = scene_from_arrays(
        timestamps=timestamps,
        coordinate_frame={'name': 'legacy_reconstruction', 'handedness': 'right',
                          'up_axis': 'unknown', 'pose_convention': 'T_world_local'},
        metric_scale={'status': 'estimated', 'provenance': provenance}, actors=[], objects=[object_state],
        support=None, contacts=contacts, uncertainty={'status': 'unestimated'},
        task={'description': 'Keyframe compression of an existing predicted track', 'target': None})
    rigid = np.asarray(object_state['T_world_object'])
    vertices = np.asarray(object_data['vertices'], float)
    stretch = np.asarray(object_state['geometry'].get('local_linear', np.eye(3)))
    vertices = vertices @ stretch.T
    table = original_scene.get('table')
    result['variants'] = []
    for policy in ['uniform', 'contact_events', 'geometry_error']:
        if policy == 'geometry_error':
            selection = select_keyframes(scene, budget, 'contact_events')
            indices = geometry_keyframes(rigid, timestamps, vertices, budget)
            selection.update(policy=policy, indices=indices, timestamps=timestamps[indices].tolist(),
                             omitted_contact_boundaries=[i for i in selection['contact_boundaries'] if i not in indices],
                             full_fitted_sequence_consumed=True)
        else:
            selection = select_keyframes(scene, budget, policy)
        candidate = interpolate_poses(rigid, timestamps, selection['indices'])
        row = dict(selection, **errors(rigid, candidate))
        row['interpolation_for_diagnostic'] = 'Linear translation and shortest-arc SLERP at original timestamps'
        row['translation_error_per_frame_mm'] = (np.linalg.norm(rigid[:, :3, 3] - candidate[:, :3, 3], axis=1) * 1000).tolist()
        if table:
            normal, offset = np.asarray(table['normal']), float(table['offset'])
            minimum = lambda p: np.min((vertices @ p[:, :3, :3].transpose(0, 2, 1) + p[:, None, :3, 3]) @ normal + offset, axis=1)
            dense_bottom, sparse_bottom = minimum(rigid), minimum(candidate)
            row['support_diagnostic'] = {
                'against': 'same historical supplied support plane; not ground-truth geometry',
                'full_fit_max_penetration_mm': float(max(0, -dense_bottom.min()) * 1000),
                'compressed_max_penetration_mm': float(max(0, -sparse_bottom.min()) * 1000),
                'surface_height_change_max_mm': float(np.max(np.abs(sparse_bottom - dense_bottom)) * 1000)}
        result['variants'].append(row)
    result.update(status='complete', keyframes=min(budget, len(poses)),
                  retained_pose_fraction=min(budget, len(poses)) / len(poses),
                  contact_prior_available=contact is not None, grasp_prior_frames=int((held & known).sum()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, action='append', required=True)
    parser.add_argument('--budget', type=int, default=5)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = []
    for path in args.trajectory:
        try:
            rows.append(analyze(path, args.budget))
        except (ValueError, KeyError, OSError) as exc:
            rows.append({'trajectory': str(path), 'status': 'failed', 'reason': str(exc)})
    report = {'experiment': 'Saved-prediction keyframe compression diagnostic', 'budget': args.budget,
              'interpretation': 'Errors compare interpolated keyframes with the original fitted sequence. They are not GT errors or robot execution outcomes.',
              'selection': 'All input trajectories are reported, including unsupported and failed conversions.',
              'results': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
