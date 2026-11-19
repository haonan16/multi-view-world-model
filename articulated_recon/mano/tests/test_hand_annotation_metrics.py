import importlib.util
from pathlib import Path
import numpy as np
import pytest

path = Path(__file__).parents[2]/'benchmarks/hand_annotation_metrics.py'
spec = importlib.util.spec_from_file_location('hand_annotation_metrics', path)
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_translation_is_separate_from_wrist_relative_geometry():
    ref = np.arange(126).reshape(2, 21, 3) / 1000.
    pred = ref + [0, 0, .03]
    result = m.score_joints(pred, ref, np.ones(2, bool))
    assert result['absolute_joint']['mean_mm'] == pytest.approx(30)
    assert result['wrist_relative_joint']['mean_mm'] == pytest.approx(0, abs=1e-10)
    result = m.score_surface(pred, ref, pred[:, 0], ref[:, 0], np.ones(2, bool))
    assert result['absolute_corresponded_vertex']['mean_mm'] == pytest.approx(30)
    assert result['wrist_relative_corresponded_vertex']['mean_mm'] == pytest.approx(0, abs=1e-10)


def test_relative_score_does_not_procrustes_align_or_rescale():
    ref = np.zeros((1, 21, 3));ref[0, :, 0] = np.arange(21)*.001
    pred = ref*2
    result = m.score_joints(pred, ref, np.ones(1, bool))
    assert result['absolute_wrist']['mean_mm'] == 0
    assert result['wrist_relative_joint']['mean_mm'] == pytest.approx(10)


def test_absent_and_unobserved_hands_are_not_scored():
    ref = np.zeros((3, 21, 3));ref[1] = -1
    pred = np.ones_like(ref)
    result = m.score_joints(pred, ref, np.array([True, True, False]))
    assert result['frames'] == 1 and result['source_frame_indices'] == [0]
    ref[:] = -1
    with pytest.raises(ValueError, match='No mutually valid'):
        m.score_joints(pred, ref, np.ones(3, bool))
