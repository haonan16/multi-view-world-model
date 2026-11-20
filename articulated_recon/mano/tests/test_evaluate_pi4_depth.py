import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
spec = importlib.util.spec_from_file_location('evaluate_pi4_depth', ROOT / 'benchmarks/evaluate_pi4_depth.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fixture(tmp_path):
    from rgb_frontend import prepare, load_source_bundle
    from pi4_frontend import make_manifest
    source = tmp_path / 'original.npz'
    rgb = np.arange(3 * 24 * 32 * 3, dtype=np.uint8).reshape(3, 24, 32, 3)
    timestamps = np.array([10 / 30, 11 / 30, 12 / 30])
    np.savez(source, rgb=rgb, timestamps=timestamps)
    bundle = tmp_path / 'source'; prepare(source, bundle)
    manifest, source_row, _, _ = load_source_bundle(bundle)
    out = tmp_path / 'predictions'; out.mkdir()
    selected = np.array([0, 2]); images = m.resize_rgb(rgb[selected], 16)
    t, h, w, _ = images.shape
    depth = np.full((t, 1, h, w), 1.2, np.float32)
    local = np.zeros(depth.shape + (3,), np.float32); local[..., 2] = depth
    camera = np.tile(np.eye(4), (t, 1, 1, 1)); camera[1, 0, 0, 3] = .02
    np.savez(out / 'geometry.npz', rgb=images[:, None], timestamps=timestamps[selected],
             source_frame_indices=selected, local_points_m=local, depth_m=depth,
             depth_valid=np.ones(depth.shape, bool), confidence=np.broadcast_to(np.linspace(0, 1, w), depth.shape), c2w=camera)
    np.savez(out / 'raw-predictions.npz', unit_test_fixture=np.array(True))
    (out / 'rgb.npz').write_bytes((bundle / 'rgb.npz').read_bytes())
    result = make_manifest(manifest, source_row, out, {'revision':'synthetic-test-only'}, False)
    (out / 'input-manifest.json').write_text(json.dumps(result))
    execution = dict(status='complete', geometry_schema='pi4_dynamic_camera_v1',
        measured_depth_used=False, known_intrinsics_supplied=False, reference_alignment=False,
        source_manifest_sha256=m.file_sha256(bundle / 'input-manifest.json'),
        source_frame_indices=selected.tolist(), timestamps_s=timestamps[selected].tolist(), max_size=16)
    (out / 'inference.json').write_text(json.dumps(execution))
    seal = tmp_path / 'sealed.json'; m.seal_predictions(out, seal)
    refdir = tmp_path / 'reference'; refdir.mkdir()
    proof = refdir / 'synthetic-unit-convention.txt'; proof.write_text('Synthetic uint16 camera-Z test depth; millimetres')
    frames = []
    for i in selected:
        path = refdir / f'depth_{10+i:06d}.png'
        Image.fromarray(np.full((24, 32), 1000, np.uint16)).save(path)
        frames.append(dict(source_frame_index=int(i), original_frame_id=int(10+i),
            timestamp_s=float(timestamps[i]), depth_path=path.name, depth_sha256=m.file_sha256(path)))
    reference = dict(schema='hocap-depth-reference-v1', status='verified', evaluator_only=True,
        convention_verified=True, depth_convention='color-camera optical-axis Z', depth_unit='millimeter',
        scale_to_m=.001, invalid_depth_value=0, source_rgb_bundle_sha256=source_row['sha256'],
        source_manifest_sha256='a' * 64, source_input_manifest_sha256=execution['source_manifest_sha256'], image_size=[32, 24],
        sequence='synthetic-test', source_camera='test', source_frame_indices=selected.tolist(),
        original_frame_ids=[10, 12], timestamps_s=timestamps[selected].tolist(), frames=frames,
        fixed_camera=True, fixed_camera_provenance='synthetic fixed-camera test',
        units_proof=[dict(path=proof.name, sha256=m.file_sha256(proof))])
    path = refdir / 'reference-manifest.json'; path.write_text(json.dumps(reference))
    return dict(predictions=out, seal=seal, reference=path, output=tmp_path / 'report.json')


def test_metric_bias_is_not_aligned_away_and_native_indices_are_preserved(tmp_path):
    args = fixture(tmp_path); report = m.evaluate(**args)
    assert report['frames_evaluated'] == 2
    assert report['categories']['all']['all_valid']['mae_mm'] == pytest.approx(200, abs=.001)
    assert report['categories']['all']['all_valid']['bias_mm'] == pytest.approx(200, abs=.001)
    assert [f['original_frame_id'] for f in report['per_frame']] == [10, 12]
    assert report['camera_drift']['max_translation_m'] == pytest.approx(.02)
    assert report['no_scale_correction'] and not report['refinement_performed']
    assert report['semantic_categories_status'] == 'unavailable'


def test_reference_resize_uses_pixel_centres_and_preserves_zero_holes():
    raw = np.arange(16, dtype=np.uint16).reshape(4, 4) * 100
    raw[1, 1] = 0
    np.testing.assert_array_equal(m.resize_reference(raw, (2, 2)), [[0, 700], [1300, 1500]])
    resized = m.resize_reference(raw, (8, 8))
    np.testing.assert_array_equal(resized[2:4, 2:4], 0)
    assert set(np.unique(resized)) <= set(np.unique(raw))


def test_confidence_selection_does_not_use_reference_or_semantic_masks():
    confidence = np.array([[.1, .2, .3, .4]])
    valid = np.ones(confidence.shape, bool)
    high, ranks = m.confidence_top_half(confidence, valid)
    np.testing.assert_array_equal(high, [[False, False, True, True]])
    # Highest-confidence sensor pixel is missing. The threshold cannot move down.
    pred = np.ones(confidence.shape)
    reference = np.array([[1., 1., 1., 0.]])
    stats = m.score_category(pred, reference, valid, valid, high)
    assert stats['confidence_top50']['pixels'] == 1
    ties, _ = m.confidence_top_half(np.ones((2, 2)), np.ones((2, 2), bool))
    assert ties.all()  # Explicit tie policy, reported retention can exceed half.


def test_missing_reference_frames_are_reported_not_imputed(tmp_path):
    args = fixture(tmp_path)
    ref = json.loads(args['reference'].read_text())
    for key in ['frames', 'source_frame_indices', 'original_frame_ids', 'timestamps_s']:
        ref[key] = ref[key][:1]
    args['reference'].write_text(json.dumps(ref))
    report = m.evaluate(**args)
    assert report['status'] == 'complete_with_missing_references'
    assert report['frames_evaluated'] == 1 and report['missing_source_frame_indices'] == [2]
    assert report['per_frame'][1]['status'] == 'reference_unavailable'


def test_hash_seal_rejects_changed_prediction_before_reference_access(tmp_path):
    args = fixture(tmp_path)
    with (args['predictions'] / 'geometry.npz').open('ab') as stream:
        stream.write(b'changed')
    args['reference'] = tmp_path / 'never-open-this-missing-reference.json'
    with pytest.raises(ValueError, match='changed after sealing'):
        m.evaluate(**args)
    assert not args['output'].exists()


def test_incomplete_run_cannot_be_sealed(tmp_path):
    args = fixture(tmp_path)
    path = args['predictions'] / 'inference.json'
    run = json.loads(path.read_text()); run['status'] = 'running'; path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='completed'):
        m.seal_predictions(args['predictions'], tmp_path / 'invalid-seal.json')


def test_wrong_original_frame_mapping_or_timestamp_fails(tmp_path):
    args = fixture(tmp_path); ref = json.loads(args['reference'].read_text())
    ref['frames'][1]['timestamp_s'] += .001; ref['timestamps_s'][1] += .001
    args['reference'].write_text(json.dumps(ref))
    with pytest.raises(ValueError, match='timestamp mismatch'):
        m.evaluate(**args)


def test_source_provenance_receipt_is_not_confused_with_method_input_manifest(tmp_path):
    args = fixture(tmp_path); ref = json.loads(args['reference'].read_text())
    assert ref['source_manifest_sha256'] != ref['source_input_manifest_sha256']
    ref['source_input_manifest_sha256'] = ref['source_manifest_sha256']
    args['reference'].write_text(json.dumps(ref))
    with pytest.raises(ValueError, match='source identity'):
        m.evaluate(**args)


def test_reference_png_tampering_is_rejected(tmp_path):
    args = fixture(tmp_path); ref = json.loads(args['reference'].read_text())
    path = args['reference'].parent / ref['frames'][0]['depth_path']
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='depth hash mismatch'):
        m.evaluate(**args)


def test_semantic_ids_require_verification_and_unknown_ids_are_not_background(tmp_path):
    args = fixture(tmp_path); ref = json.loads(args['reference'].read_text())
    ref.update(segmentation_verified=True, segmentation_categories={'object':[1], 'hand':[2], 'background':[0]})
    for frame in ref['frames']:
        labels = np.full((24, 32), 99, np.int16); labels[:, :16] = 1
        p = args['reference'].parent / f"labels-{frame['source_frame_index']}.npz"
        np.savez(p, segmentation=labels)
        frame.update(segmentation_path=p.name, segmentation_key='segmentation', segmentation_sha256=m.file_sha256(p))
    args['reference'].write_text(json.dumps(ref))
    report = m.evaluate(**args)
    assert report['categories']['object']['all_valid']['pixels'] > 0
    assert report['categories']['background']['all_valid']['pixels'] == 0
    assert report['categories']['hand']['all_valid']['pixels'] == 0


def test_camera_drift_uses_relative_pose_without_changing_predicted_scale():
    c = np.tile(np.eye(4), (2, 1, 1, 1)); c[:, 0, :3, 3] = [[10, 0, 0], [10.1, 0, 0]]
    c[1, 0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    report = m.camera_drift(c, [0, 1], fixed_camera_documented=True)
    assert report['max_translation_m'] == pytest.approx(.1)
    assert report['max_rotation_deg'] == pytest.approx(90)
    c[1, 0, 0, 0] = 2
    with pytest.raises(ValueError, match='proper SE3'):
        m.camera_drift(c, [0, 1])
