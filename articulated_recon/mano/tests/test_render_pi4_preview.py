import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/render_pi4_preview.py'
    spec = importlib.util.spec_from_file_location('render_pi4_preview', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_projection_uses_camera_to_world_and_nearest_point_occlusion():
    m = module()
    c2w = np.eye(4)
    c2w[:3, 3] = [5, 2, -1]
    points = np.array([[5, 2, 1], [5, 2, 3], [6, 2, 1], [5, 3, 1],
                       [5, 2, -2], [np.nan, 0, 0]])
    ids, uv, depth = m.project_points(points, c2w, 101, 101, fov_degrees=90)
    np.testing.assert_array_equal(ids, [0, 1, 2, 3])
    np.testing.assert_array_equal(uv[:2], [[50, 50], [50, 50]])
    assert uv[2, 0] > 50 and uv[3, 1] > 50
    np.testing.assert_allclose(depth, [2, 4, 2, 2])
    colors = np.array([[255, 0, 0], [0, 0, 255], [10, 20, 30], [40, 50, 60],
                       [255, 255, 255], [255, 255, 255]], dtype=np.uint8)
    image, visible = m.rasterize_points(points, colors, c2w, 101, 101, fov_degrees=90)
    np.testing.assert_array_equal(image[50, 50], [255, 0, 0])
    assert visible == 3


def test_projection_is_invariant_to_shared_rigid_world_coordinate_change():
    m = module()
    points = np.array([[-.2, -.1, 2], [.4, .3, 3], [.1, -.3, 2]])
    camera = np.eye(4)
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [2, -3, .5]
    transformed = points @ transform[:3, :3].T + transform[:3, 3]
    before = m.project_points(points, camera, 101, 81)
    after = m.project_points(transformed, transform @ camera, 101, 81)
    for first, second in zip(before, after):
        np.testing.assert_allclose(first, second)


def test_stride_five_timestamps_repeat_predictions_without_interpolation():
    m = module()
    timestamps = (53 + np.arange(0, 120, 5)) / 30
    times, indices, final_hold = m.display_schedule(timestamps, 30)
    np.testing.assert_array_equal(indices, np.repeat(np.arange(24), 5))
    np.testing.assert_allclose(times, np.arange(120) / 30)
    assert final_hold == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match='increasing'):
        m.display_schedule([0, 0], 30)


def test_one_fixed_point_sampling_and_depth_range_cover_whole_sequence():
    m = module()
    ids = m.sample_indices(4, 5, 7)
    np.testing.assert_array_equal(ids, [0, 3, 6, 9, 12, 15, 19])
    points = np.stack([np.stack(np.meshgrid(np.arange(5), np.arange(4)), -1)] * 2)
    points = np.concatenate([points, np.ones((*points.shape[:-1], 1))], -1).astype(float)
    points[1, ..., 2] += 1
    points[1, 0, 0] = np.nan
    camera = np.tile(np.eye(4), (2, 1, 1))
    display = m.fixed_display_geometry(points, camera, ids, percentile=0)
    assert display['valid'].shape == (2, 7)
    assert display['valid'][0].sum() == 7 and display['valid'][1].sum() == 6
    assert display['lower'][2] < 1 and display['upper'][2] > 2
    virtual, near, far = m.virtual_camera(display, 30)
    np.testing.assert_allclose(virtual[:3, :3].T @ virtual[:3, :3], np.eye(3), atol=1e-10)
    assert np.linalg.det(virtual[:3, :3]) == pytest.approx(1)
    assert 0 < near < far
    depth = np.array([[[1., 2.], [np.nan, -1.]], [[2., 4.], [3., 0.]]])
    limits = m.fixed_depth_limits(depth)
    first, second = [m.depth_colors(frame, limits) for frame in depth]
    np.testing.assert_array_equal(first[0, 1], second[0, 0])
    np.testing.assert_array_equal(first[1], 0)


def synthetic_run(tmp_path):
    """Small projection/encoding fixture; not a reconstruction quality claim."""
    path = tmp_path / 'synthetic'
    path.mkdir()
    yy, xx = np.mgrid[:32, :48]
    points = np.stack([(xx - 23.5) / 40, (yy - 15.5) / 40, np.ones_like(xx)], -1)
    rgb = np.stack([xx * 4, yy * 6, np.full_like(xx, 100)], -1).astype(np.uint8)
    times = np.array([1., 1 + 1 / 6])
    source_ids = np.array([0, 5])
    np.savez(path / 'geometry.npz', rgb=np.stack([rgb, rgb])[:, None],
             points_world=np.stack([points, points])[:, None],
             depth_m=np.ones((2, 1, 32, 48)), c2w=np.tile(np.eye(4), (2, 1, 1, 1)),
             timestamps=times, source_frame_indices=source_ids)
    record = dict(status='complete', geometry_schema='pi4_dynamic_camera_v1',
        measured_depth_used=False, known_intrinsics_supplied=False, reference_alignment=False,
        source_frame_indices=source_ids.tolist(), timestamps_s=times.tolist(), frames=2)
    (path / 'inference.json').write_text(json.dumps(record))
    return path


def test_synthetic_render_encodes_exact_hold_count_and_auditable_metadata(tmp_path):
    import cv2
    m = module()
    path = synthetic_run(tmp_path)
    output = tmp_path / 'preview'
    report = m.render(path, output, fps=30, max_points=500)
    assert report['display_prediction_indices'] == [0] * 5 + [1] * 5
    assert report['metric_scale'].startswith('estimated')
    assert not report['generated_rgb'] and not report['geometry_interpolated']
    assert not report['hidden_surfaces_completed'] and not report['reference_data_read']
    assert set(Path(p).name for p in report['input_hashes']) == {'geometry.npz', 'inference.json'}
    video = cv2.VideoCapture(str(output / 'preview.mp4'))
    try:
        assert video.get(cv2.CAP_PROP_FRAME_COUNT) == 10
        assert video.get(cv2.CAP_PROP_FPS) == 30
        assert video.get(cv2.CAP_PROP_FRAME_WIDTH) == 144
        assert video.get(cv2.CAP_PROP_FRAME_HEIGHT) == 68
    finally:
        video.release()
    with pytest.raises(FileExistsError):
        m.render(path, output)


def test_mismatched_inference_or_reference_usage_blocks_preview(tmp_path):
    m = module()
    path = synthetic_run(tmp_path)
    record = json.loads((path / 'inference.json').read_text())
    record['source_frame_indices'] = [0, 6]
    (path / 'inference.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='frame/time'):
        m.load_predictions(path)
    record['known_intrinsics_supplied'] = True
    (path / 'inference.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='source-only'):
        m.load_predictions(path)
