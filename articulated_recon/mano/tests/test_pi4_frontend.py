import importlib.util
from pathlib import Path

import numpy as np
import pytest


def frontend():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/pi4_frontend.py'
    spec = importlib.util.spec_from_file_location('pi4_frontend', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predictions():
    poses = np.tile(np.eye(4, dtype=np.float32), (1, 2, 1, 1))
    poses[0, 1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    poses[0, 1, :3, 3] = [1, 2, 3]
    points = np.ones((1, 2, 2, 3, 3), dtype=np.float32)
    scalar = np.full((1, 2, 2, 3), .6, dtype=np.float32)
    raw = dict(local_points=points, camera_poses=poses,
               metric_scale=np.array([2.], dtype=np.float32), confidence=scalar.copy(),
               tracks_3d=points.copy() * 7, tracks_2d=points[..., :2].copy(),
               query_frame=np.array([0]), dynamic_mask=scalar.copy(),
               visibility=scalar.copy(), tracking_confidence=scalar.copy())
    raw['tracks_3d'][0, 1, 0, 0] = np.nan
    return raw


def test_metric_conversion_preserves_camera_motion_and_does_not_rescale_tracks():
    module = frontend()
    raw = predictions()
    module.validate_predictions(raw, (2, 2, 3, 3), 0)
    geometry = module.metric_geometry(raw, np.zeros((2, 2, 3, 3), np.uint8),
                                      np.array([0., .5]), np.array([0, 5]))
    np.testing.assert_allclose(geometry['local_points_m'], 2.)
    np.testing.assert_allclose(geometry['c2w'][1, 0, :3, 3], [2, 4, 6])
    np.testing.assert_allclose(geometry['points_world'][1, 0, 1, 2], [0, 6, 8])
    np.testing.assert_allclose(geometry['tracks_3d'][0, 0, 0, 0], [7, 7, 7])
    assert np.isnan(geometry['tracks_3d'][1, 0, 0, 0]).all()
    assert not geometry['track_finite'][1, 0, 0, 0]
    np.testing.assert_array_equal(raw['camera_poses'][0, 1, :3, 3], [1, 2, 3])
    assert geometry['rgb'].shape == (2, 1, 2, 3, 3)


@pytest.mark.parametrize('mutate', [
    lambda raw: raw['metric_scale'].__setitem__(0, -1),
    lambda raw: raw['camera_poses'].__setitem__((0, 0, 0, 0), 2),
    lambda raw: raw['query_frame'].__setitem__(0, 1),
    lambda raw: raw['confidence'].__setitem__((0, 0, 0, 0), np.nan),
])
def test_invalid_prediction_contract_is_rejected(mutate):
    raw = predictions()
    mutate(raw)
    with pytest.raises(ValueError):
        frontend().validate_predictions(raw, (2, 2, 3, 3), 0)


def test_fitted_intrinsics_are_diagnostic_and_residual_exposes_nonpinhole_rays():
    module = frontend()
    yy, xx = np.mgrid[:16, :32]
    depth = np.full(xx.shape, 3.)
    points = np.stack([(xx - 15) * depth / 50, (yy - 8) * depth / 55, depth], axis=-1)
    k, residual = module.pinhole_diagnostic(points[None])
    np.testing.assert_allclose(k[0], [[50, 0, 15], [0, 55, 8], [0, 0, 1]], atol=1e-10)
    assert residual[0] < 1e-10
    points[..., 0] += .0005 * (xx - 15) ** 2
    _, residual = module.pinhole_diagnostic(points[None])
    assert residual[0] > .1


def test_sampling_matches_native_stride_then_cap_and_rgb_resize():
    module = frontend()
    np.testing.assert_array_equal(module.select_frames(120, 3, 25), np.arange(0, 75, 3))
    rgb = np.full((2, 480, 640, 3), 97, np.uint8)
    resized = module.resize_rgb(rgb, 512)
    assert resized.shape == (2, 384, 512, 3)
    np.testing.assert_array_equal(resized, 97)


def test_track_selection_keeps_query_identity_metric_values_and_missing_observations():
    module = frontend()
    raw = predictions()
    tracks = module.extract_tracks(raw, np.array([0, 5]))
    assert tracks['tracks3d'].shape == (2, 2, 3)
    np.testing.assert_array_equal(tracks['query_pixels'], [[0, 0], [2, 1]])
    np.testing.assert_array_equal(tracks['query_frame'], [0, 0])
    np.testing.assert_allclose(tracks['tracks3d'][0], 7)
    assert np.isnan(tracks['tracks3d'][1, 0]).all()
    np.testing.assert_allclose(tracks['trackvisibility'], .6)
    with pytest.raises(ValueError, match='unique valid'):
        module.extract_tracks(raw, np.array([1, 1]))


def test_protocol_outputs_preserve_rgb_lineage_and_block_robot_execution(tmp_path):
    module = frontend()
    source = tmp_path / 'source.npz'
    np.savez(source, rgb=np.zeros((2, 16, 16, 3), np.uint8), timestamps=[0., 1.],
             sensor_depth=np.ones((2, 16, 16)))
    from rgb_frontend import prepare, load_source_bundle
    bundle = tmp_path / 'bundle'
    prepare(source, bundle)
    manifest, source_record, _, _ = load_source_bundle(bundle)
    (bundle / 'raw-predictions.npz').write_bytes(b'bound raw prediction fixture')
    (bundle / 'geometry.npz').write_bytes(b'bound geometry fixture')
    result = module.make_manifest(manifest, source_record, bundle,
                                  {'checkpoint_sha256': 'f' * 64, 'revision': 'public-release'}, True)
    report = module.validate_manifest(result, bundle, verify_files=True)
    assert report['oracle_dependencies'] == []
    assert report['calibration_dependencies'] == []
    assert not report['execution_ready']
    assert report['metric_scale']['status'] == 'estimated'
    assert report['robot_registration']['status'] == 'unresolved'
    with np.load(bundle / 'rgb.npz') as data:
        assert set(data.files) == {'rgb', 'timestamps'}


def test_contaminated_isolated_bundle_is_rejected(tmp_path):
    from rgb_frontend import prepare, load_source_bundle, sha256
    import json
    source = tmp_path / 'source.npz'
    np.savez(source, rgb=np.zeros((2, 16, 16, 3), np.uint8), timestamps=[0., 1.])
    bundle = tmp_path / 'bundle'
    prepare(source, bundle)
    np.savez(bundle / 'rgb.npz', rgb=np.zeros((2, 16, 16, 3), np.uint8),
             timestamps=[0., 1.], K=np.eye(3))
    manifest_path = bundle / 'input-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['artifacts'][0]['sha256'] = sha256(bundle / 'rgb.npz')
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Unexpected fields'):
        load_source_bundle(bundle)
