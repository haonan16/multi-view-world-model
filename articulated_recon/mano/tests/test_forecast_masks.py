"""CPU tests for frame/view routing and generated-evidence mask lineage."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks import forecast_masks as fm


def generated_fixture(tmp_path):
    path = Path(__file__).with_name('test_world_model_forecast.py')
    spec = importlib.util.spec_from_file_location('forecast_mask_fixture', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    args, _ = m.fixture_inputs(tmp_path); args['frames'] = 5
    m.sf.prepare_forecast(**args)
    condition = args['output']; helper = tmp_path/'generated'; helper.mkdir()
    with np.load(condition/'condition.npz') as z: arrays = {k: z[k].copy() for k in z.files}
    _, known = m.wm.source_tensor(arrays['rgb'], forecast=True, total_frames=5)
    raw = np.zeros((5, 4, 160, 320, 3), np.uint8)
    for t in range(5):
        for v in range(4): raw[t, v] = 10*t + v + 50
    rgb = m.wm.restore_observed_source(raw, arrays['rgb'], known)
    np.savez_compressed(helper/'observations.npz', rgb=rgb, observed_mask=known,
                        **{k: v for k, v in arrays.items() if k != 'rgb'})
    manifest = json.loads((condition/'input-manifest.json').read_text())
    for record in manifest['artifacts']:
        if 'path' in record: record['path'] = str((condition/record['path']).resolve())
    manifest['artifacts'].append(dict(id='world_model_weights', kind='model_prior', origin='prior',
                                     prior_type='learned_model_weights', scope='global', parent_ids=[]))
    manifest['artifacts'].extend(m.sf.generated_records(helper, arrays['timestamps']))
    manifest['method_inputs'].extend(['generated_source_forecast', 'generated_helper_rgb'])
    (helper/'input-manifest.json').write_text(json.dumps(manifest))
    checkpoint = tmp_path/'checkpoint'; checkpoint.mkdir(); (checkpoint/'config.json').write_text('{}')
    return helper, condition, checkpoint, rgb


def detections(image, prompt):
    if prompt == 'hand':
        # Installed SAM3's no-instance output can retain decoder resolution.
        return np.zeros((0, 288, 288), bool), np.zeros(0), np.zeros((0, 4))
    mask = np.zeros((1, *image.shape[:2]), bool); mask[0, 20:25, 30:36] = True
    return mask, np.array([.8]), np.array([[30, 20, 36, 25]])


def test_each_forecast_frame_view_routes_exact_pixels_and_retains_empty_prompts(tmp_path):
    helper, condition, checkpoint, rgb = generated_fixture(tmp_path)
    seen = []
    def predict(image, prompt):
        seen.append((image.copy(), prompt)); result = detections(image, prompt)
        image[:] = 0  # A processor mutation must not alter later input calls.
        return result
    out = tmp_path/'masks'
    receipt = fm.run(helper, condition, out, checkpoint, predict)
    assert receipt['status'] == 'complete' and receipt['processed_images'] == 20
    assert receipt['observed_images'] == 1 and receipt['generated_images'] == 19
    assert not receipt['temporal_tracking'] and not receipt['cross_view_identity_estimated']
    assert len(seen) == 60
    for index, (pixels, prompt) in enumerate(seen):
        t, rest = divmod(index, 12); v, p = divmod(rest, 3)
        np.testing.assert_array_equal(pixels, rgb[t, v])
        assert prompt == fm.DEFAULT_PROMPTS[p]
    with np.load(out/'masks.npz') as z:
        assert len(z['t']) == 40 and len(np.unique(z['candidate_row_id'])) == 40
        assert z['candidate_counts'].shape == (5, 4, 3) and not z['candidate_counts'][:, :, 1].any()
        np.testing.assert_array_equal(z['t'], np.repeat(np.arange(5), 8))
        np.testing.assert_array_equal(z['v'], np.tile(np.repeat(np.arange(4), 2), 5))
        np.testing.assert_array_equal(z['p'], np.tile([0, 2], 20))
        assert z['mask'].shape == (40, 160*320//8)
        mask = np.unpackbits(z['mask'][0])[:160*320].reshape(160, 320)
        assert mask[20:25, 30:36].all() and mask.sum() == 30
        assert z['observed_mask'].sum() == 1 and z['observed_mask'][0, 0]
        assert z['source_frame_indices'].tolist() == [0, -1, -1, -1, -1]
        assert z['image_rgb_sha256'].shape == (5, 4)


def test_all_and_source_only_products_keep_their_actual_generated_ancestry(tmp_path):
    helper, condition, checkpoint, _ = generated_fixture(tmp_path)
    out = tmp_path/'masks'; receipt = fm.run(helper, condition, out, checkpoint, detections)
    full = set(receipt['validation']['method_artifact_ids'])
    source = set(receipt['source_only_product']['validation']['method_artifact_ids'])
    assert {'generated_source_forecast', 'generated_helper_rgb'} <= full
    assert 'generated_source_forecast' in source and 'generated_helper_rgb' not in source
    assert receipt['generated_parent_ids'] == ['generated_source_forecast', 'generated_helper_rgb']
    for folder in [out, out/'source-only']:
        manifest = json.loads((folder/'input-manifest.json').read_text())
        assert manifest['task_mode'] == 'forecast' and manifest['observed_until_s'] == 2.
        roots = [a for a in manifest['artifacts'] if a['origin'] == 'observed_rgb']
        assert len(roots) == 1 and roots[0]['time_range_s'] == [2., 2.]
        mask = next(a for a in manifest['artifacts'] if a['id'] == 'forecast_sam3_masks')
        assert mask['origin'] == 'model_prediction'
    with np.load(out/'masks.npz') as full_masks, np.load(out/'source-only/masks.npz') as source_masks:
        keep = full_masks['v'] == 0
        for name in ['mask', 't', 'v', 'p', 'score', 'box', 'candidate_row_id']:
            np.testing.assert_array_equal(source_masks[name], full_masks[name][keep])
        assert source_masks['observed_mask'].shape == (5, 1)
        assert source_masks['frame_inference_seconds'].shape == (5, 1)


@pytest.mark.parametrize('change', ['false_observed', 'future_camera', 'pixels'])
def test_mask_preflight_rechecks_strict_inputs(tmp_path, change):
    helper, condition, _, _ = generated_fixture(tmp_path)
    if change == 'future_camera':
        prep = json.loads((condition/'preparation.json').read_text()); prep['mode'] = 'reconstruction'
        (condition/'preparation.json').write_text(json.dumps(prep))
    else:
        with np.load(helper/'observations.npz') as z: arrays = {k: z[k].copy() for k in z.files}
        if change == 'false_observed': arrays['observed_mask'][1, 0] = True
        else: arrays['rgb'][0, 0, 0, 0, 0] ^= 1
        np.savez_compressed(helper/'observations.npz', **arrays)
        manifest = json.loads((helper/'input-manifest.json').read_text())
        for artifact in manifest['artifacts']:
            if artifact.get('path') == 'observations.npz': artifact['sha256'] = fm.file_sha256(helper/'observations.npz')
        (helper/'input-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): fm.load_forecast(helper, condition)


def test_changed_input_fails_before_publishing_masks(tmp_path):
    helper, condition, checkpoint, _ = generated_fixture(tmp_path)
    changed = False
    def predict(image, prompt):
        nonlocal changed
        if not changed:
            with (helper/'observations.npz').open('ab') as stream: stream.write(b'changed')
            changed = True
        return detections(image, prompt)
    out = tmp_path/'masks'
    with pytest.raises(ValueError, match='input changed'): fm.run(helper, condition, out, checkpoint, predict)
    assert json.loads((out/'execution.json').read_text())['status'] == 'failed'
    assert not (out/'masks.npz').exists() and not (out/'input-manifest.json').exists()


def test_no_detection_is_recorded_without_inventing_masks(tmp_path):
    helper, condition, checkpoint, _ = generated_fixture(tmp_path)
    out = tmp_path/'masks'
    receipt = fm.run(helper, condition, out, checkpoint,
        lambda image, prompt: (np.zeros((0, 288, 288)), np.zeros(0), np.zeros((0, 4))))
    assert receipt['candidates'] == 0
    with np.load(out/'masks.npz') as z:
        assert z['mask'].shape == (0, 6400) and z['box'].shape == (0, 4)
        assert not z['candidate_counts'].any()


def test_invalid_prompts_and_existing_outputs_rejected(tmp_path):
    helper, condition, checkpoint, _ = generated_fixture(tmp_path)
    for prompts in [[], ['hand', 'hand'], ['']]:
        with pytest.raises(ValueError, match='prompts'): fm.load_forecast(helper, condition, prompts)
    with pytest.raises(FileExistsError): fm.run(helper, condition, helper, checkpoint, detections)
