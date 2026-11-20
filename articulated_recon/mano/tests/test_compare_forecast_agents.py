"""The evaluator scores sealed outputs, never a live agent or moving reference."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import compare_forecast_agents as cf
from forecast_geometry import object_from_arrays, scene_from_arrays


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def fixture():
    frame = dict(name='source_camera', handedness='right', up_axis='-y', pose_convention='T_world_local')
    poses = np.tile(np.eye(4), (2, 1, 1)); poses[:, 2, 3] = 1
    provenance = dict(method='synthetic test only')
    obj = object_from_arrays(object_id='block', poses=poses, observed=[True, False],
        primitive=dict(type='box', half_extents_m=[.1, .08, .06]), provenance=provenance)
    obj['prediction_supported'] = [False, True]
    obj['prediction_provenance'] = dict(origin='model_prediction', generated=True,
                                        artifact_ids=['synthetic'], method='synthetic')
    scene = scene_from_arrays(timestamps=[0., .333333333333], coordinate_frame=frame,
        metric_scale=dict(status='estimated', provenance=provenance), actors=[], objects=[obj],
        support=None, contacts=[], uncertainty=dict(status='unestimated'), task=dict(description='Lift', target=None))
    packet = dict(coordinate_frame=frame, metric_scale_status='estimated', object_id='block',
                  timestamps=scene['timestamps'], condition='A_raw', evidence_branch='source_unposed',
                  preprocessing_timing_seconds=dict(structured_initializer=None))
    k = np.tile([[100., 0, 16], [0, 100., 16], [0, 0, 1]], (2, 1, 1, 1))
    camera = np.tile(np.eye(4), (2, 1, 1, 1))
    vertices, faces = cf.geometry_arrays(obj['geometry'])
    depth = np.stack([cf.raster_depth(vertices + [0, 0, 1], faces, k[t, 0], camera[t, 0], (32, 32))
                      for t in range(2)])[:, None]
    observations = dict(rgb=np.zeros((2, 1, 32, 32, 3), np.uint8), depth_m=depth, K=k, c2w=camera,
        depth_valid=depth > 0, object_masks=depth > 0, object_mask_known=np.ones((2, 1), bool),
        occlusion_mask=np.zeros(depth.shape, bool), camera_projection_valid=np.ones((2, 1), bool),
        timestamps=np.asarray(scene['timestamps']), observed_mask=np.array([[True], [False]]),
        source_frame_indices=np.array([0, -1]))
    return scene, packet, observations


def packet_folder(folder, packet, observations):
    folder.mkdir(parents=True)
    write(folder / 'packet.json', packet)
    np.savez_compressed(folder / 'observations.npz', **observations)
    write(folder / 'frozen-inputs.json', dict(files={name: cf.digest(folder / name)
        for name in ['packet.json', 'observations.npz']}))


def suite_folder(tmp_path, *, status='completed'):
    scene, packet, observations = fixture()
    workspace = tmp_path / 'agent'
    packet_folder(workspace, packet, observations)
    (workspace / 'output').mkdir()
    write(workspace / 'output' / 'scene-sequence.json', scene)
    suite = tmp_path / 'suite'; suite.mkdir()
    run = dict(id='one', workspace=str(workspace), condition='A_raw',
               frozen_inputs_sha256=cf.digest(workspace / 'frozen-inputs.json'))
    write(suite / 'suite.json', dict(runs=[run], config=dict(model='synthetic')))
    write(workspace / 'execution-receipt.json', dict(suite_sha256=cf.digest(suite / 'suite.json'), run_id='one'))
    log = suite / 'logs' / 'one'; log.mkdir(parents=True)
    write(log / 'status.json', dict(status=status, final_scene_sha256=cf.digest(workspace / 'output' / 'scene-sequence.json'),
        effort=dict(output_tokens=12, monetary_cost=None), elapsed_seconds=3.))
    return suite, workspace


def test_live_agent_rejected_before_reference_is_opened(tmp_path):
    suite, _ = suite_folder(tmp_path, status='running')
    with pytest.raises(ValueError, match='not terminal'):
        cf.compare(suite=suite, reference=tmp_path / 'nonexistent-evaluator', output=tmp_path / 'comparison')
    assert not (tmp_path / 'comparison').exists()


def test_changed_final_scene_cannot_be_evaluated(tmp_path):
    suite, workspace = suite_folder(tmp_path)
    path = workspace / 'output' / 'scene-sequence.json'
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='changed after terminal'):
        cf.sealed_runs(suite)


def test_same_reference_cannot_be_rescaled_by_candidate():
    scene, packet, obs = fixture()
    unchanged = cf.score_scene(scene, packet, obs)
    altered = deepcopy(scene)
    altered['forecast'] = dict(per_frame_depth_scale=[20., 20.])
    assert cf.score_scene(altered, packet, obs) == unchanged
    assert unchanged['source_view']['visible_mask_iou_mean'] == 1.
    assert unchanged['source_future']['depth_evidence_rows'] == 1
    assert unchanged['source_view']['predicted_depth_residual_median_m'] == 0.
    altered['objects'][0]['T_world_object'][1][0][3] += .25
    assert cf.score_scene(altered, packet, obs)['source_future']['visible_mask_iou_mean'] < .3


def test_unknown_or_occluded_evidence_is_not_a_zero_error():
    scene, packet, obs = fixture()
    obs['object_mask_known'][1, 0] = False
    obs['occlusion_mask'][0, 0] = True
    result = cf.score_scene(scene, packet, obs)
    assert result['source_view']['visible_mask_iou_mean'] is None
    assert result['source_view']['predicted_depth_residual_median_m'] is None
    assert result['source_view']['mask_evidence_rows'] == 0


def test_common_plane_clearance_uses_full_rotation_and_not_candidate_table():
    scene, _, _ = fixture()
    obj = scene['objects'][0]
    obj['T_world_object'][0][2][3] = .1
    obj['T_world_object'][1][2][3] = .1
    obj['T_world_object'][1] = np.asarray(obj['T_world_object'][1]).copy().tolist()
    pose = np.asarray(obj['T_world_object']); pose[1, :3, :3] = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    obj['T_world_object'] = pose.tolist()
    scene['support'] = dict(normal_world=[0, 0, 1], offset_m=999.)  # Must not enter this metric.
    result = cf.shape_motion(scene, 'block', dict(normal_world=[0, 0, 1], offset_m=0.))
    np.testing.assert_allclose(result['common_predicted_table_clearance_m'], [.04, 0], atol=1e-10)
    assert result['edge_manifold_closed'] is True
    assert result['supported_adjacent_pairs'] == 1


def test_compare_reports_effort_and_keeps_accuracy_unmeasured(tmp_path):
    suite, workspace = suite_folder(tmp_path)
    result = cf.compare(suite=suite, reference=workspace, output=tmp_path / 'comparison')
    assert result['runs'][0]['scene_valid']
    assert result['runs'][0]['effort']['output_tokens'] == 12
    assert result['runs'][0]['evidence_branch'] == 'source_unposed'
    assert result['runs'][0]['initialization_preprocessing_seconds'] is None
    assert result['runs'][0]['initialization_preprocessing_status'] == 'not_run'
    assert result['runs'][0]['first_valid_scene_seconds'] is None
    assert result['source_accuracy'] is None and result['execution_success'] is None
    assert result['future_trajectory_accuracy'] is None
    assert result['runs'][0]['agreement']['common_source']['source_view']['visible_mask_iou_mean'] == 1.


def test_reference_cannot_label_generated_future_as_observed(tmp_path):
    _, packet, obs = fixture(); obs['observed_mask'][1, 0] = True
    packet_folder(tmp_path / 'bad', packet, obs)
    with pytest.raises(ValueError, match='one observed source image'):
        cf.reference_packet(tmp_path / 'bad')


def test_failed_scene_retains_branch_and_measured_effort_metadata(tmp_path):
    suite, workspace = suite_folder(tmp_path, status='failed')
    path = suite / 'logs' / 'one' / 'status.json'; status = json.loads(path.read_text())
    status.update(final_scene_sha256=None, preprocessing_timing_seconds=dict(structured_initializer=4.2))
    write(path, status); (workspace / 'output' / 'scene-sequence.json').unlink()
    checkpoints = suite / 'logs' / 'one' / 'checkpoints'; checkpoints.mkdir()
    write(checkpoints / 'index.json', [dict(first_observed_elapsed_seconds=2.1)])
    result = cf.compare(suite=suite, reference=workspace, output=tmp_path / 'comparison')
    row = result['runs'][0]
    assert not row['scene_valid'] and row['evidence_branch'] == 'source_unposed'
    assert row['initialization_preprocessing_seconds'] == 4.2
    assert row['first_valid_scene_seconds'] == 2.1
    assert 'externally sampled' in row['first_valid_scene_timing_scope']


def test_explicit_initial_scene_path_is_bound_to_its_manifest(tmp_path, monkeypatch):
    suite, workspace = suite_folder(tmp_path)
    scene, _, _ = fixture(); scene['timestamps'] = scene['timestamps'][:1]
    obj = scene['objects'][0]
    for key in ['observed', 'prediction_supported', 'T_world_object']: obj[key] = obj[key][:1]
    scene_path = tmp_path / 'actual-initial-scene.json'; write(scene_path, scene)
    folder = tmp_path / 'manifest-only'; folder.mkdir()
    calls = []
    def bind(manifest, required):
        calls.append((manifest, required))
        return dict(condition='rgb_only', task_mode='forecast')
    monkeypatch.setattr(cf, 'load_manifest', bind)
    result = cf.compare(suite=suite, reference=workspace, initial_scene_bundle=folder,
        initial_scene_path=scene_path, output=tmp_path / 'comparison')
    assert calls == [(folder / 'input-manifest.json', [('object_geometry', scene_path)])]
    assert result['common_initial_scene_sha256'] == cf.digest(scene_path)


def test_common_support_intersection_excludes_placeholder_metrics_without_erasing_them():
    scene, packet, obs = fixture()
    first = deepcopy(scene); first['objects'][0]['T_world_object'][1][0][3] += .25
    second = deepcopy(scene); second['objects'][0]['prediction_supported'][1] = False
    rows = [dict(run_id=str(i), scene_valid=True, geometry_motion=cf.shape_motion(s, 'block'),
                 agreement=dict(common_source=cf.score_scene(s, packet, obs))) for i,s in enumerate([first, second])]
    result = cf.attach_common_supported(dict(runs=rows), 2)
    assert result['common_supported_frame_indices'] == [0]
    assert rows[0]['agreement']['common_source']['source_view']['visible_mask_iou_mean'] < 1.
    assert rows[0]['agreement']['common_source']['common_supported']['source_view']['visible_mask_iou_mean'] == 1.
    assert rows[1]['agreement']['common_source']['per_frame_view'][1]['pose_supported'] is False
    assert rows[0]['agreement']['common_source']['common_supported']['source_future']['mask_evidence_rows'] == 0


def test_numerical_plane_contact_is_not_counted_as_penetration():
    scene, _, _ = fixture()
    scene['objects'][0]['T_world_object'][0][2][3] = .06 - 1e-16
    scene['objects'][0]['T_world_object'][1][2][3] = .06 - 2e-6
    result = cf.shape_motion(scene, 'block', dict(normal_world=[0, 0, 1], offset_m=0.))
    assert result['common_predicted_table_clearance_m'][0] < 0  # Preserve the raw value.
    assert result['common_predicted_table_penetration_frames'] == [1]
    assert result['numerical_tolerance_m'] == 1e-6
