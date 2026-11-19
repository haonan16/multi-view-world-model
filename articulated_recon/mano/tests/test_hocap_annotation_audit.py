import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'benchmarks'))
from hocap_annotation_audit import published_camera, verify_reference_mesh, require_annotation_members, assert_stable
from hocap_object_audit import surface_comparison, summarize_frames


def test_camera_reference_world_gauge_is_not_per_camera_registration():
    tag, camera = np.eye(4), np.eye(4)
    tag[:3, 3] = [.1, -.2, .3]; camera[:3, 3] = [.4, .5, .6]
    result = published_camera({'tag_1': tag[:3].ravel().tolist(), 'source': camera[:3].ravel().tolist()}, 'source')
    assert np.allclose(result, np.linalg.inv(tag) @ camera)


def test_reference_mesh_must_match_declared_dataset_landmarks():
    vertices = np.arange(30.).reshape(1, 10, 3) / 1000
    joints = np.zeros((1, 21, 3)); labels = joints.copy()
    tips = [1, 2, 3, 4, 5]; labels[:, [4, 8, 12, 16, 20]] = vertices[:, tips]
    assert verify_reference_mesh(vertices, joints, labels, tips, np.array([True])).max() == 0
    with pytest.raises(ValueError, match='does not reproduce'):
        verify_reference_mesh(vertices, joints, labels, [0, 2, 3, 4, 5], np.array([True]))


def test_object_metric_keeps_translation_error_and_excludes_missing_frames():
    ref = np.array([[0., 0, 0], [.01, 0, 0]])
    row = surface_comparison(ref + [0, 0, .03], ref, ref.mean(0) + [0, 0, .03], ref.mean(0))
    assert row['surface_centroid_error_mm'] == pytest.approx(30)
    assert row['symmetric_sampled_surface_distance_mm'] == pytest.approx(30)
    frames = [dict(observed=True, subset='common', **row), dict(observed=False, subset='recovered')]
    assert summarize_frames(frames, 'all')['frames'] == 1
    assert summarize_frames(frames, 'recovered')['surface_centroid_error_mm'] is None


def test_unlisted_annotations_and_modified_input_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='absent from verification'):
        require_annotation_members({'meta.yaml': 'hash'}, 'label_000034.npz')
    from artirecon.input_protocol import file_sha256
    file = tmp_path/'input'; file.write_text('frozen')
    hashes = {str(file): file_sha256(file)}
    assert_stable(hashes)
    file.write_text('changed')
    with pytest.raises(ValueError, match='changed during evaluation'):
        assert_stable(hashes)
