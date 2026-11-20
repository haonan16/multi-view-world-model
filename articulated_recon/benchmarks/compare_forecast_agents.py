"""Compare sealed forecast-agent outputs against common predicted evidence.

This is an offline descriptive comparison, not an accuracy or success test.
Every run must have stopped before reference evidence is read. Scene hashes are
checked against the harness's terminal receipts. Source/reference depth and
cameras are fixed across candidates; a candidate cannot rescale its reference.
No simulator, future recorded trajectory, or private evaluator is loaded here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import load_manifest
from artirecon.scene_sequence import load_scene
from representation_agent_tools import agreement, digest, geometry_arrays, load_inputs, raster_depth, validate_export


TERMINAL = {'completed', 'failed', 'stopped_at_budget', 'interrupted',
            'completed_over_reported_budget', 'invalid_input_mutation'}
NUMERICAL_TOLERANCE_M = 1e-6


def read_json(path):
    return json.loads(Path(path).read_text())


def sealed_runs(suite_path):
    """Check every terminal receipt before touching any comparison evidence."""
    suite_path = Path(suite_path).resolve()
    suite_file = suite_path / 'suite.json'
    suite = read_json(suite_file)
    result = []
    for run in suite['runs']:
        status_path = suite_path / 'logs' / run['id'] / 'status.json'
        if not status_path.is_file():
            raise ValueError('Every agent must stop before comparison: ' + run['id'])
        status = read_json(status_path)
        if status.get('status') not in TERMINAL:
            raise ValueError('Agent is not terminal: ' + run['id'])
        workspace = Path(run['workspace']).resolve()
        if digest(workspace / 'frozen-inputs.json') != run['frozen_inputs_sha256']:
            raise ValueError('Frozen input inventory changed: ' + run['id'])
        invocation = read_json(workspace / 'execution-receipt.json')
        if invocation['suite_sha256'] != digest(suite_file) or invocation['run_id'] != run['id']:
            raise ValueError('Invocation receipt does not bind this suite/run')
        final = workspace / 'output' / 'scene-sequence.json'
        expected = status.get('final_scene_sha256')
        if expected is not None:
            if not final.is_file() or digest(final) != expected:
                raise ValueError('Final scene changed after terminal receipt: ' + run['id'])
        elif final.is_file():
            raise ValueError('Final scene is not sealed by terminal receipt: ' + run['id'])
        result.append(dict(run=run, status=status, workspace=workspace, scene_path=final,
                           status_sha256=digest(status_path), scene_sha256=expected))
    return suite, result


def reference_packet(path):
    packet, arrays = load_inputs(path)
    count, views, height, width = arrays['rgb'].shape[:4]
    if arrays['depth_m'].shape != (count, views, height, width):
        raise ValueError('Reference depth must share its RGB grid')
    if arrays['K'].shape != (count, views, 3, 3) or arrays['c2w'].shape != (count, views, 4, 4):
        raise ValueError('Reference needs explicit per-frame/view camera arrays')
    expected_observed = np.zeros((count, views), bool)
    expected_observed[0, 0] = True
    if not np.array_equal(arrays.get('observed_mask'), expected_observed):
        raise ValueError('Forecast reference must contain only one observed source image')
    ids = np.asarray(arrays.get('source_frame_indices'))
    if not np.array_equal(ids, np.r_[0, np.full(count - 1, -1)]):
        raise ValueError('Forecast future frames cannot carry recorded source indices')
    return packet, arrays


def summary(rows):
    iou = [r['visible_mask_iou'] for r in rows if r['visible_mask_iou'] is not None]
    depth = [r['predicted_depth_agreement_median_m'] for r in rows
             if r['predicted_depth_agreement_median_m'] is not None]
    return dict(frames_views=len(rows), mask_evidence_rows=len(iou), depth_evidence_rows=len(depth),
                visible_mask_iou_mean=float(np.mean(iou)) if iou else None,
                visible_mask_iou_min=float(np.min(iou)) if iou else None,
                predicted_depth_residual_median_m=float(np.median(depth)) if depth else None,
                depth_overlap_pixels=sum(r['depth_overlap_pixels'] for r in rows),
                visible_reference_pixels=sum(r.get('visible_reference_pixels', 0) for r in rows))


def score_scene(scene, packet, reference):
    """Use exactly the same reference depth/K/c2w for every output scene."""
    if scene['coordinate_frame'] != packet['coordinate_frame'] or not np.array_equal(
            scene['timestamps'], reference['timestamps']):
        raise ValueError('Comparison scene/reference timestamps or coordinate gauge differ')
    obj = next(o for o in scene['objects'] if o['id'] == packet['object_id'])
    vertices, faces = geometry_arrays(obj['geometry'])
    poses = np.asarray(obj['T_world_object'])
    supported = np.asarray(obj['observed'], bool) | np.asarray(
        obj.get('prediction_supported', np.zeros(len(poses), bool)), bool)
    rows = []
    count, views, h, w = reference['depth_m'].shape
    projection_valid = reference.get('camera_projection_valid', np.ones((count, views), bool))
    for t in range(count):
        world = vertices @ poses[t, :3, :3].T + poses[t, :3, 3]
        for v in range(views):
            if projection_valid[t, v]:
                rendered = raster_depth(world, faces, reference['K'][t, v], reference['c2w'][t, v], (h, w))
                row = agreement(rendered, reference['depth_m'][t, v], reference['object_masks'][t, v],
                    bool(reference['object_mask_known'][t, v]), reference['occlusion_mask'][t, v],
                    reference['depth_valid'][t, v])
            else:
                row = dict(status='reference_projection_unavailable', visible_mask_iou=None,
                    predicted_depth_agreement_median_m=None, depth_overlap_pixels=0)
            row.update(frame=t, view=v, timestamp_s=float(reference['timestamps'][t]),
                       pose_supported=bool(supported[t]),
                       evidence='observed_source_RGB_with_predicted_geometry' if t == 0 and v == 0
                       else 'generated_forecast_with_predicted_geometry')
            rows.append(row)
    per_view = [dict(view=v, **summary([r for r in rows if r['view'] == v])) for v in range(views)]
    return dict(reference_policy='Fixed reference depth_m/K/c2w; candidate depth-scale declarations are ignored',
        all_frame_summary_scope='All timeline entries, including unsupported placeholder poses; diagnostic only',
        source_view=summary([r for r in rows if r['view'] == 0]),
        initial_source=summary([r for r in rows if r['view'] == 0 and r['frame'] == 0]),
        source_future=summary([r for r in rows if r['view'] == 0 and r['frame'] > 0]),
        helper_views=summary([r for r in rows if r['view'] > 0]),
        per_view=per_view, per_frame_view=rows,
        interpretation='Agreement with a common set of model predictions; not true geometry accuracy or camera consistency proof.')


def shape_motion(scene, object_id, common_support=None):
    obj = next(o for o in scene['objects'] if o['id'] == object_id)
    vertices, faces = geometry_arrays(obj['geometry'])
    poses = np.asarray(obj['T_world_object'], float)
    times = np.asarray(scene['timestamps'], float)
    extent = np.ptp(vertices, axis=0)
    centre_local = (vertices.min(0) + vertices.max(0)) / 2
    centres = np.einsum('tij,j->ti', poses[:, :3, :3], centre_local) + poses[:, :3, 3]
    distances = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    angles = Rotation.from_matrix(poses[:-1, :3, :3].transpose(0, 2, 1) @ poses[1:, :3, :3]).magnitude()
    observed = np.asarray(obj['observed'], bool)
    supported = observed | np.asarray(obj.get('prediction_supported', np.zeros(len(times), bool)), bool)
    reliable_edges = supported[:-1] & supported[1:]
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    result = dict(shared_geometry_type=obj['geometry']['type'], local_aabb_extent_m=extent.tolist(),
        local_aabb_volume_m3=float(np.prod(extent)), vertices=len(vertices), faces=len(faces),
        edge_manifold_closed=bool(len(edge_counts) and np.all(edge_counts == 2)),
        shape_measurement_scope='Local AABB extent/volume; not true object volume or unseen-surface accuracy',
        trajectory_path_length_m=float(distances.sum()), net_displacement_m=float(np.linalg.norm(centres[-1] - centres[0])),
        maximum_step_m=float(distances.max()) if len(distances) else 0.,
        maximum_speed_m_s=float(np.max(distances / np.diff(times))) if len(distances) else None,
        total_rotation_rad=float(angles.sum()), support_flagged_frames=int(supported.sum()),
        supported_frame_indices=np.flatnonzero(supported).tolist(),
        unsupported_frame_indices=np.flatnonzero(~supported).tolist(),
        supported_adjacent_motion_m=float(distances[reliable_edges].sum()),
        supported_adjacent_pairs=int(reliable_edges.sum()),
        trajectory_interpretation='Generated forecast hypothesis; no future recorded trajectory is used as a target',
        common_predicted_table_clearance_m=None, common_predicted_table_penetration_frames=None)
    if common_support is not None:
        normal = np.asarray(common_support['normal_world'], float)
        normal = normal / np.linalg.norm(normal)
        # A shared, fixed predicted plane; never a candidate's own edited table.
        offsets = vertices @ np.einsum('tji,j->ti', poses[:, :3, :3], normal).T
        clearance = offsets.min(0) + poses[:, :3, 3] @ normal + common_support['offset_m']
        result.update(common_predicted_table_clearance_m=clearance.tolist(),
            common_predicted_table_min_clearance_m=float(clearance.min()),
            numerical_tolerance_m=NUMERICAL_TOLERANCE_M,
            common_predicted_table_penetration_frames=np.flatnonzero(clearance < -NUMERICAL_TOLERANCE_M).tolist(),
            table_interpretation='Plausibility relative to the initial model-predicted plane, not measured table accuracy')
    return result


def attach_common_supported(result, frame_count):
    """Use the same evidence-support intersection, never select by fit residual."""
    valid = [row for row in result['runs'] if row['scene_valid']]
    common = set(range(frame_count)) if valid else set()
    for row in valid:
        common.intersection_update(row['geometry_motion']['supported_frame_indices'])
    indices = sorted(common)
    result.update(common_supported_frame_indices=indices,
        common_supported_frame_count=len(indices), total_frame_count=frame_count,
        common_supported_valid_run_ids=[row['run_id'] for row in valid],
        common_supported_selection='Intersection of observed|prediction_supported across valid frozen scenes; no residual-based frame selection')
    for row in valid:
        for metrics in row['agreement'].values():
            selected = [r for r in metrics['per_frame_view'] if r['frame'] in common]
            metrics['common_supported'] = dict(frame_indices=indices,
                source_view=summary([r for r in selected if r['view'] == 0]),
                initial_source=summary([r for r in selected if r['view'] == 0 and r['frame'] == 0]),
                source_future=summary([r for r in selected if r['view'] == 0 and r['frame'] > 0]),
                helper_views=summary([r for r in selected if r['view'] > 0]))
        geometry = row['geometry_motion']
        clearances = geometry.get('common_predicted_table_clearance_m')
        if clearances is not None:
            geometry['common_supported_table_penetration_frames'] = [t for t in indices if clearances[t] < -NUMERICAL_TOLERANCE_M]
    return result


def compare(*, suite, reference, output, helper_reference=None, initial_scene_bundle=None, initial_scene_path=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    suite_info, sealed = sealed_runs(suite)  # Mandatory barrier before reading references.
    packet, obs = reference_packet(reference)
    if obs['rgb'].shape[1] != 1:
        raise ValueError('Primary reference must contain only the generated source stream')
    refs = [('common_source', Path(reference), packet, obs)]
    if helper_reference is not None:
        hp, ho = reference_packet(helper_reference)
        if (hp['coordinate_frame'] != packet['coordinate_frame'] or hp['object_id'] != packet['object_id']
                or not np.array_equal(ho['timestamps'], obs['timestamps'])
                or not np.array_equal(ho['rgb'][:, 0], obs['rgb'][:, 0])):
            raise ValueError('Helper comparison must use the same source pixels, time and gauge')
        refs.append(('common_helpers', Path(helper_reference), hp, ho))
    support = None
    support_hash = None
    if initial_scene_path is not None and initial_scene_bundle is None:
        raise ValueError('Explicit initial scene path still requires its manifest bundle')
    if initial_scene_bundle is not None:
        folder = Path(initial_scene_bundle)
        path = Path(initial_scene_path).resolve() if initial_scene_path is not None else folder / 'scene-sequence.json'
        report = load_manifest(folder / 'input-manifest.json', [('object_geometry', path)])
        initial = load_scene(path)
        if report['condition'] != 'rgb_only' or report['task_mode'] != 'forecast':
            raise ValueError('Common table must come from the RGB-only forecast initializer')
        if initial['timestamps'] != [float(obs['timestamps'][0])] or initial['coordinate_frame'] != packet['coordinate_frame']:
            raise ValueError('Common table must be the same first-image coordinate gauge')
        support, support_hash = initial['support'], digest(path)
    result = dict(schema_version=1, status='sealed_comparison_complete', suite_sha256=digest(Path(suite) / 'suite.json'),
        all_agent_outputs_sealed_before_reference_read=True, agent_config=suite_info['config'],
        source_accuracy=None, future_trajectory_accuracy=None, contact_accuracy=None, execution_success=None,
        independent_initial_RGBD_evaluation='not_run; requires separately authenticated evaluator-only initial reference',
        common_initial_scene_sha256=support_hash,
        references={name:dict(packet_sha256=digest(path / 'packet.json'), observations_sha256=digest(path / 'observations.npz'))
                    for name, path, _, _ in refs}, runs=[],
        limitations=['One attempt per condition is descriptive, not a reliable stochastic ranking.',
            'Reference depth, masks, cameras, metric scale and table are predictions, not ground truth.',
            'Helper-view agreement can reward internally inconsistent generated evidence.',
            'Camera estimation and reconstruction ambiguity remain unresolved by these scores.',
            'This script performs no action planning or independent physics evaluation.'])
    for entry in sealed:
        run, status = entry['run'], entry['status']
        metadata_path = entry['workspace'] / 'packet.json'
        inventory = read_json(entry['workspace'] / 'frozen-inputs.json')
        if digest(metadata_path) != inventory['files']['packet.json']:
            raise ValueError('Frozen packet metadata changed: ' + run['id'])
        metadata = read_json(metadata_path)
        preprocessing = status.get('preprocessing_timing_seconds', metadata.get('preprocessing_timing_seconds', {}))
        first_valid = status.get('first_valid_scene_seconds')
        first_valid_scope = 'terminal status measurement' if first_valid is not None else 'unreported'
        checkpoint_index = Path(suite) / 'logs' / run['id'] / 'checkpoints' / 'index.json'
        if first_valid is None and checkpoint_index.is_file():
            records = read_json(checkpoint_index)
            measured = [r['first_observed_elapsed_seconds'] for r in records
                        if isinstance(r.get('first_observed_elapsed_seconds'), (float, int))]
            if measured:
                first_valid = min(measured)
                first_valid_scope = 'earliest externally sampled valid scene checkpoint; not exact file-write time'
        row = dict(run_id=run['id'], condition=run['condition'], run_status=status['status'],
            evidence_branch=metadata.get('evidence_branch'),
            status_sha256=entry['status_sha256'], scene_sha256=entry['scene_sha256'],
            scene_valid=False, scene_error=None, effort=status.get('effort'),
            elapsed_seconds=status.get('elapsed_seconds'), wall_elapsed_seconds=status.get('wall_elapsed_seconds'),
            preprocessing_timing_seconds=preprocessing,
            initialization_preprocessing_seconds=preprocessing.get('structured_initializer'),
            initialization_preprocessing_status=('measured' if preprocessing.get('structured_initializer') is not None else
                'not_run' if metadata.get('condition') == 'A_raw' else 'unreported'),
            first_valid_scene_seconds=first_valid, first_valid_scene_timing_scope=first_valid_scope,
            distinct_valid_scene_checkpoints=status.get('distinct_valid_scene_checkpoints'),
            scene_edits=status.get('scene_edits'), edit_count_scope=status.get('edit_count_scope'))
        if entry['scene_sha256'] is not None and status['status'] != 'invalid_input_mutation':
            try:
                own_packet, own_obs = load_inputs(entry['workspace'])
                if not np.array_equal(own_obs['rgb'][:, 0], obs['rgb'][:, 0]):
                    raise ValueError('Agent did not receive the same generated source pixels as the reference')
                scene = read_json(entry['scene_path'])
                counts = validate_export(scene, own_packet)
                row.update(scene_valid=True, scene_counts=counts,
                    geometry_motion=shape_motion(scene, packet['object_id'], support),
                    agreement={name:score_scene(scene, p, a) for name, _, p, a in refs})
            except (ValueError, KeyError, StopIteration) as error:
                row.update(scene_valid=False, scene_error=str(error))
        elif entry['scene_sha256'] is None:
            row['scene_error'] = 'No sealed final scene'
        else:
            row['scene_error'] = 'Agent input mutation invalidated the run'
        result['runs'].append(row)
    attach_common_supported(result, len(obs['timestamps']))
    output.mkdir(parents=True)
    (output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['suite', 'reference', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--helper-reference', type=Path)
    parser.add_argument('--initial-scene-bundle', type=Path)
    parser.add_argument('--initial-scene-path', type=Path,
                        help='Explicit payload path when its manifest bundle stores the scene elsewhere')
    result = compare(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'], runs=len(result['runs']),
                          valid_scenes=sum(r['scene_valid'] for r in result['runs'])), indent=2))


if __name__ == '__main__':
    main()
