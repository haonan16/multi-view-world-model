"""Strict preparation rejects full-video ancestry before any model import."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artirecon.input_protocol import file_sha256, validate_manifest
from artirecon.scene_sequence import scene_from_arrays, object_from_arrays
from benchmarks import strict_forecast as sf
from benchmarks import world_model_views as wm


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2))


def fixture_inputs(tmp_path):
    root = tmp_path / 'initial'; root.mkdir()
    rgb = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(1, 12, 16, 3)
    np.savez(root/'rgb.npz', rgb=rgb, timestamps=[2.])
    k = np.array([[20., 0, 8], [0, 20., 6], [0, 0, 1]])
    np.savez(root/'geometry.npz', rgb=rgb[:, None], timestamps=[2.], depth_m=np.ones((1, 1, 12, 16)),
             K=k[None], c2w=np.eye(4)[None])
    pose = np.eye(4)[None]; pose[0, 2, 3] = .8
    prov = {'method': 'Synthetic first-image estimator', 'artifact_ids': ['depth']}
    scene = scene_from_arrays(timestamps=[2.],
        coordinate_frame={'name': 'source_camera', 'handedness': 'right', 'up_axis': '-y', 'pose_convention': 'T_world_local'},
        metric_scale={'status': 'estimated', 'provenance': prov}, actors=[],
        objects=[object_from_arrays(object_id='block', poses=pose, observed=[True],
            primitive={'type': 'box', 'half_extents_m': [.02]*3}, provenance=prov)],
        support=None, contacts=[], uncertainty={'status': 'unestimated'}, task={'description': 'Pick up the block'})
    write_json(root/'scene.json', scene)
    records = [dict(id='source', kind='rgb', origin='observed_rgb', generated=False, parent_ids=[],
                    path=str(root/'rgb.npz'), sha256=file_sha256(root/'rgb.npz'), time_range_s=[2., 2.]),
               *[dict(id=n, kind=n if n == 'depth' else 'cameras', origin='model_prediction', parent_ids=['source'],
                      path=str(root/'geometry.npz'), sha256=file_sha256(root/'geometry.npz')) for n in ['depth', 'camera']]]
    manifest = dict(schema_version=1, condition='rgb_only', task_mode='forecast', observed_until_s=2.,
        artifacts=records, method_inputs=['depth', 'camera'],
        metric_scale={'status': 'estimated', 'artifact_ids': ['depth'], 'description': 'Model depth scale'},
        robot_registration={'status': 'unresolved', 'artifact_ids': [], 'description': 'No robot registration'})
    write_json(root/'input-manifest.json', manifest)
    scene_manifest = deepcopy(manifest)
    scene_manifest['artifacts'].append(dict(id='scene', kind='object_geometry', origin='model_prediction',
        parent_ids=['depth', 'camera'], path=str(root/'scene.json'), sha256=file_sha256(root/'scene.json')))
    scene_manifest['method_inputs'] = ['scene']
    write_json(root/'scene-manifest.json', scene_manifest)
    return dict(geometry_bundle=root, scene_path=root/'scene.json', scene_manifest=root/'scene-manifest.json',
                output=tmp_path/'condition', object_id='block', task_text='Pick up the block', prompt_key='block_task',
                frames=17, dt_s=.2), rgb


def refresh_artifact_hashes(args, changed):
    for path in [args['geometry_bundle']/'input-manifest.json', args['scene_manifest']]:
        manifest = json.loads(path.read_text())
        for record in manifest['artifacts']:
            if record.get('path') == str(changed): record['sha256'] = file_sha256(changed)
        write_json(path, manifest)


def test_first_image_preparation_declares_horizon_without_future_pixels(tmp_path):
    args, rgb = fixture_inputs(tmp_path)
    prep = sf.prepare_forecast(**args)
    loaded, report = wm.generation_preparation(args['output'], conditioning='strict_forecast')
    assert report['task_mode'] == 'forecast' and loaded == prep
    with np.load(args['output']/'condition.npz') as z:
        assert z['rgb'].shape == (1, 160, 320, 3)
        assert z['source_frame_indices'].tolist() == [0] + [-1]*16
        np.testing.assert_allclose(z['timestamps'], 2 + np.arange(17)*.2)
        np.testing.assert_allclose(z['c2w'][0], np.eye(4))
        assert len(np.unique(z['c2w'].reshape(4, -1), axis=0)) == 4
        assert np.all(np.linalg.norm(z['c2w'][1:, :3, 3], axis=1) > 0)
        tensor, known = wm.source_tensor(z['rgb'], forecast=True, total_frames=17)
        assert known.sum() == 1 and known[0, 0]
        assert not tensor[:, 1:].any() and not tensor[:, :, 1:].any()
    assert prep['camera_design_horizon'] == 'Initial image only'
    assert 'RGB only' in prep['output_modality']


def test_forecast_cadence_has_no_implicit_default(tmp_path):
    args, _ = fixture_inputs(tmp_path)
    args.pop('dt_s')
    with pytest.raises(TypeError, match='dt_s'): sf.prepare_forecast(**args)


@pytest.mark.parametrize('part', ['source', 'depth', 'scene', 'camera', 'mode', 'future_ancestor'])
def test_reject_later_frames_even_if_outer_manifest_is_relabeled(tmp_path, part):
    args, rgb = fixture_inputs(tmp_path); root = args['geometry_bundle']
    if part == 'source':
        np.savez(root/'rgb.npz', rgb=np.repeat(rgb, 2, axis=0), timestamps=[2., 3.])
        refresh_artifact_hashes(args, root/'rgb.npz')
    elif part in {'depth', 'camera'}:
        with np.load(root/'geometry.npz') as z: arrays = {n: z[n].copy() for n in z.files}
        if part == 'depth': arrays['timestamps'] = np.array([2., 3.])
        else: arrays['c2w'] = np.repeat(np.eye(4)[None], 4, axis=0)
        np.savez(root/'geometry.npz', **arrays); refresh_artifact_hashes(args, root/'geometry.npz')
    elif part == 'scene':
        scene = json.loads((root/'scene.json').read_text()); scene['timestamps'] = [2., 3.]
        write_json(root/'scene.json', scene); refresh_artifact_hashes(args, root/'scene.json')
    else:
        manifest = json.loads(args['scene_manifest'].read_text())
        if part == 'mode': manifest['task_mode'] = 'reconstruction'
        else: manifest['artifacts'][0]['time_range_s'] = [2., 3.]
        write_json(args['scene_manifest'], manifest)
    with pytest.raises(ValueError): sf.prepare_forecast(**args)
    assert not args['output'].exists()


def test_full_video_manifest_cannot_be_promoted_by_first_image_conditioning(tmp_path):
    args, _ = fixture_inputs(tmp_path); sf.prepare_forecast(**args)
    prep_path = args['output']/'preparation.json'
    prep = json.loads(prep_path.read_text()); prep['mode'] = 'reconstruction'
    write_json(prep_path, prep)
    with pytest.raises(ValueError, match='dedicated first-image'):
        wm.generation_preparation(args['output'], conditioning='strict_forecast')


@pytest.mark.parametrize('kind', ['cameras', 'depth'])
def test_additional_temporal_ancestor_cannot_omit_timestamps(tmp_path, kind):
    args, _ = fixture_inputs(tmp_path); root = args['geometry_bundle']
    path = root/'extra.npz'
    arrays = {'c2w': np.repeat(np.eye(4)[None], 17, axis=0)} if kind == 'cameras' else {'depth': np.ones((17, 1, 12, 16))}
    np.savez(path, **arrays)
    manifest = json.loads(args['scene_manifest'].read_text())
    manifest['artifacts'].append(dict(id='extra', kind=kind, origin='model_prediction', parent_ids=['source'],
                                     path=str(path), sha256=file_sha256(path)))
    manifest['method_inputs'].append('extra')
    write_json(args['scene_manifest'], manifest)
    with pytest.raises(ValueError, match='singleton timestamp'): sf.prepare_forecast(**args)


@pytest.mark.parametrize('key', ['depth_m', 'points_world', 'local_points_m', 'uncertainty', 'confidence_rank'])
def test_temporal_payload_must_agree_with_singleton_timestamp(tmp_path, key):
    args, _ = fixture_inputs(tmp_path); root = args['geometry_bundle']
    path = root/'extra.npz'
    arrays = dict(depth_m=np.ones((1, 1, 12, 16)), timestamps=[2.])
    arrays[key] = np.ones((2, 1, 12, 16))
    np.savez(path, **arrays)
    manifest = json.loads(args['scene_manifest'].read_text())
    manifest['artifacts'].append(dict(id='extra', kind='depth', origin='model_prediction', parent_ids=['source'],
                                     path=str(path), sha256=file_sha256(path)))
    manifest['method_inputs'].append('extra')
    write_json(args['scene_manifest'], manifest)
    with pytest.raises(ValueError, match='temporal array'): sf.prepare_forecast(**args)


def test_explicit_legacy_depth_alias_is_supported(tmp_path):
    args, _ = fixture_inputs(tmp_path); path = args['geometry_bundle']/'geometry.npz'
    with np.load(path) as z: arrays = {n: z[n].copy() for n in z.files}
    arrays['depth'] = arrays.pop('depth_m')
    np.savez(path, **arrays); refresh_artifact_hashes(args, path)
    sf.prepare_forecast(**args)


def test_strict_input_cannot_enter_reconstruction_or_calibrated_modes(tmp_path):
    args, _ = fixture_inputs(tmp_path); sf.prepare_forecast(**args)
    with pytest.raises(ValueError, match='unsupported forecast'):
        wm.generation_preparation(args['output'])
    with pytest.raises(ValueError, match='calibrated diagnostic'):
        wm.generation_preparation(args['output'], conditioning='strict_forecast', calibrated_camera_diagnostic=True)


@pytest.mark.parametrize('change', ['future_pixels', 'rig', 'extra_ancestry', 'task'])
def test_reject_changed_preparation_even_with_updated_packet_hash(tmp_path, change):
    args, _ = fixture_inputs(tmp_path); sf.prepare_forecast(**args)
    out = args['output']; prep = json.loads((out/'preparation.json').read_text())
    manifest = json.loads((out/'input-manifest.json').read_text())
    if change in {'future_pixels', 'rig'}:
        with np.load(out/'condition.npz') as z: arrays = {n: z[n].copy() for n in z.files}
        if change == 'future_pixels': arrays['rgb'] = np.repeat(arrays['rgb'], 17, axis=0)
        else: arrays['c2w'][1] = np.eye(4)
        np.savez(out/'condition.npz', **arrays)
        prep['condition_sha256'] = file_sha256(out/'condition.npz')
        manifest['artifacts'][-1]['sha256'] = prep['condition_sha256']
    elif change == 'task': prep['task_text'] = 'Do something else'
    else:
        extra = deepcopy(manifest['artifacts'][0]); extra['id'] = 'extra_unused_source'
        manifest['artifacts'].append(extra)
    write_json(out/'preparation.json', prep); write_json(out/'input-manifest.json', manifest)
    with pytest.raises(ValueError): sf.validate_forecast(out)


def test_generated_source_is_never_restored_or_labeled_observed(tmp_path):
    args, _ = fixture_inputs(tmp_path); sf.prepare_forecast(**args)
    out = args['output']
    with np.load(out/'condition.npz') as z: arrays = {n: z[n].copy() for n in z.files}
    _, known = wm.source_tensor(arrays['rgb'], forecast=True, total_frames=17)
    raw = np.full((17, 4, 160, 320, 3), 51, np.uint8)
    observed = wm.restore_observed_source(raw, arrays['rgb'], known)
    assert np.array_equal(observed[0, 0], arrays['rgb'][0])
    assert (observed[1:, 0] == 51).all() and (observed[:, 1:] == 51).all()
    np.savez(out/'observations.npz', rgb=observed, observed_mask=known,
             **{k: v for k, v in arrays.items() if k != 'rgb'})
    records = sf.generated_records(out, arrays['timestamps'])
    source, helper = records
    assert source['view_role'] == 'source_forecast' and source['frame_indices'] == list(range(1, 17))
    assert source['view_indices'] == [0] and helper['view_indices'] == [1, 2, 3]
    assert all(r['generated'] and r['origin'] == 'model_prediction' for r in records)
    manifest = json.loads((out/'input-manifest.json').read_text())
    manifest['artifacts'] += [dict(id='world_model_weights', kind='model_prior', origin='prior', scope='global',
        prior_type='learned_model_weights', parent_ids=[])] + records
    manifest['method_inputs'] += [r['id'] for r in records]
    assert validate_manifest(manifest, out, verify_files=True)['task_mode'] == 'forecast'


def test_exact_task_text_and_key_binding(tmp_path):
    torch = pytest.importorskip('torch')
    path = tmp_path/'embeddings.pt'
    torch.save({'block': {'text': 'Pick up the block', 'embed': torch.zeros((1, 4096))}}, path)
    prep = {'prompt_key': 'block', 'task_text': 'Pick up the block'}
    result = sf.bind_task(prep, path, 'block')
    assert result['text'] == prep['task_text'] and result['embedding_sha256'] == file_sha256(path)
    with pytest.raises(ValueError, match='task text'):
        sf.bind_task(dict(prep, task_text='Move the bottle'), path, 'block')
    with pytest.raises(ValueError, match='declared task key'):
        sf.bind_task(prep, path, 'other')
    with pytest.raises(ValueError, match='null text'):
        sf.bind_task(prep, path, 'block', null_text=True)
