"""The camera-response audit consumes only t0 and authentic raw decoder output."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks import forecast_camera_audit as audit


def fixture(tmp_path):
    path = Path(__file__).with_name('test_world_model_forecast.py')
    spec = importlib.util.spec_from_file_location('camera_fixture', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    args, _ = m.fixture_inputs(tmp_path); args['frames'] = 5
    root = args['geometry_bundle']
    with np.load(root/'geometry.npz') as z: arrays = {k: z[k].copy() for k in z.files}
    arrays['valid'] = np.ones_like(arrays['depth_m'], bool)
    np.savez(root/'geometry.npz', **arrays); m.refresh_artifact_hashes(args, root/'geometry.npz')
    masks = np.zeros((3, 12, 16), bool)
    masks[0, 5:7, 7:9] = True; masks[1, 5:7, 5:7] = True; masks[2] = True
    np.savez(root/'masks.npz', mask=np.packbits(masks.reshape(3, -1), axis=1),
        shape=[12, 16], prompts=['wooden block on patterned mat', 'hand', 'tabletop'],
        t=np.zeros(3, dtype=int), p=np.arange(3), timestamps=[2.])
    manifest = json.loads(args['scene_manifest'].read_text())
    manifest['artifacts'].append(dict(id='sam3_masks', kind='masks', origin='model_prediction', parent_ids=['source'],
        path=str(root/'masks.npz'), sha256=audit.file_sha256(root/'masks.npz')))
    next(a for a in manifest['artifacts'] if a['id'] == 'scene')['parent_ids'].append('sam3_masks')
    m.write_json(args['scene_manifest'], manifest)
    m.sf.prepare_forecast(**args)
    condition = args['output']; helper = tmp_path/'generated'; helper.mkdir()
    with np.load(condition/'condition.npz') as z: condition_arrays = {k: z[k].copy() for k in z.files}
    _, known = m.wm.source_tensor(condition_arrays['rgb'], forecast=True, total_frames=5)
    raw = np.full((5, 4, 160, 320, 3), 77, np.uint8)
    rgb = m.wm.restore_observed_source(raw, condition_arrays['rgb'], known)
    common = {k: v for k, v in condition_arrays.items() if k != 'rgb'}
    np.savez(helper/'observations.npz', rgb=rgb, observed_mask=known, **common)
    np.savez(helper/'generated.npz', rgb=raw, observed_mask=known,
        **{k: v for k, v in common.items() if k != 'source_frame_indices'})
    manifest = json.loads((condition/'input-manifest.json').read_text())
    for record in manifest['artifacts']:
        if 'path' in record: record['path'] = str((condition/record['path']).resolve())
    manifest['artifacts'].append(dict(id='world_model_weights', kind='model_prior', origin='prior',
        prior_type='learned_model_weights', scope='global', parent_ids=[]))
    manifest['artifacts'].extend(m.sf.generated_records(helper, condition_arrays['timestamps']))
    manifest['method_inputs'].extend(['generated_source_forecast', 'generated_helper_rgb'])
    m.write_json(helper/'input-manifest.json', manifest)
    receipt = dict(status='complete', generated_sha256=audit.file_sha256(helper/'generated.npz'),
                   observations_sha256=audit.file_sha256(helper/'observations.npz'))
    m.write_json(helper/'generation.json', receipt)
    return helper, condition, rgb


def fake_matches(source, target, allowed):
    y, x = np.nonzero(allowed)
    ids = np.linspace(0, len(x)-1, 12).astype(int)
    a = np.c_[x[ids], y[ids]].astype(float)
    return a, a.copy(), dict(source_features=12, target_features=12, mutual_matches=12, homography_inliers=12)


def test_audit_uses_only_initial_predictions_and_excludes_dynamic_masks(tmp_path, monkeypatch):
    helper, condition, rgb = fixture(tmp_path); calls = []
    def matches(source, target, allowed):
        calls.append((source.copy(), target.copy(), allowed.copy()))
        return fake_matches(source, target, allowed)
    monkeypatch.setattr(audit, 'feature_matches', matches)
    out = tmp_path/'audit'; report = audit.run(helper, condition, out)
    assert len(calls) == 4 and [r['time_s'] for r in report['per_view']] == [2.]*4
    for source, _, allowed in calls:
        np.testing.assert_array_equal(source, rgb[0, 0])
        assert allowed.any() and not allowed.all()
        assert not allowed[80, 150]  # Initial object mask transformed through crop.
    assert (calls[0][1] == 77).all()  # Raw decoder, not restored source control.
    assert not report['future_recorded_rgb_used'] and not report['sensor_depth_used'] and not report['evaluator_used']
    assert (out/'initial-camera-response.png').exists()
    assert any(row['requested_reprojection_error_median_px'] > 0 for row in report['per_view'][1:])


def test_changed_generated_future_does_not_change_initial_measurements(tmp_path, monkeypatch):
    helper, condition, _ = fixture(tmp_path)
    monkeypatch.setattr(audit, 'feature_matches', fake_matches)
    first = audit.run(helper, condition, tmp_path/'before')
    for filename in ['observations.npz', 'generated.npz']:
        with np.load(helper/filename) as z: arrays = {k: z[k].copy() for k in z.files}
        arrays['rgb'][1:] = 11
        np.savez(helper/filename, **arrays)
    manifest = json.loads((helper/'input-manifest.json').read_text())
    for artifact in manifest['artifacts']:
        if artifact.get('path') == 'observations.npz': artifact['sha256'] = audit.file_sha256(helper/'observations.npz')
    (helper/'input-manifest.json').write_text(json.dumps(manifest))
    (helper/'generation.json').write_text(json.dumps(dict(status='complete',
        generated_sha256=audit.file_sha256(helper/'generated.npz'), observations_sha256=audit.file_sha256(helper/'observations.npz'))))
    second = audit.run(helper, condition, tmp_path/'after')
    assert first['per_view'] == second['per_view']


@pytest.mark.parametrize('change', ['raw_hash', 'receipt_status', 'raw_helpers', 'raw_timestamps', 'raw_shape'])
def test_decoder_control_requires_authentic_matching_packet(tmp_path, change):
    helper, condition, _ = fixture(tmp_path)
    receipt = json.loads((helper/'generation.json').read_text())
    if change == 'receipt_status': receipt['status'] = 'running'
    else:
        with np.load(helper/'generated.npz') as z: arrays = {k: z[k].copy() for k in z.files}
        if change in {'raw_hash', 'raw_helpers'}: arrays['rgb'][0, 1] = 99
        elif change == 'raw_timestamps': arrays['timestamps'] += .01
        else: arrays['rgb'] = arrays['rgb'][:-1]
        np.savez(helper/'generated.npz', **arrays)
        if change != 'raw_hash': receipt['generated_sha256'] = audit.file_sha256(helper/'generated.npz')
    (helper/'generation.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError): audit.run(helper, condition, tmp_path/'bad-audit')


def test_missing_matches_remain_unmeasured(tmp_path, monkeypatch):
    helper, condition, _ = fixture(tmp_path)
    monkeypatch.setattr(audit, 'feature_matches', lambda *args: (np.empty((0, 2)), np.empty((0, 2)),
        dict(source_features=0, target_features=0, mutual_matches=0, homography_inliers=0)))
    report = audit.run(helper, condition, tmp_path/'audit')
    assert all(row['status'] == 'insufficient_static_matches' for row in report['per_view'])
    assert all('requested_flow_fraction_median' not in row for row in report['per_view'])
