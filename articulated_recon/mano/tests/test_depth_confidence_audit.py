import importlib.util
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).parents[2] / "benchmarks/depth_confidence_audit.py"
_spec = importlib.util.spec_from_file_location("depth_confidence_audit", _path)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def test_exact_binding_rejects_nearest_times_and_changed_pixels():
    rgb = np.zeros((2, 2, 3, 3), np.uint8)
    ts = np.array([0., .1])
    assert m.verify_exact_frames(rgb, ts, rgb.copy(), ts.copy())["frames"] == 2
    with pytest.raises(ValueError, match="timestamps"):
        m.verify_exact_frames(rgb, ts, rgb, ts + 1e-10)
    other = rgb.copy(); other[1, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="RGB"):
        m.verify_exact_frames(rgb, ts, other, ts)


def test_error_keeps_metric_bias_and_reports_missing_sensor_coverage():
    reference = np.ones((2, 2, 3)); reference[0, 0, 0] = 0
    prediction = np.ones_like(reference) * 2
    ranks = np.linspace(0, 1, reference.size).reshape(reference.shape)
    result = m.audit_category(prediction, reference, ranks, np.ones_like(reference, bool), bins=5)
    assert result["all"]["mae_mm"] == 1000  # No GT scale alignment.
    assert result["all"]["abs_rel"] == 1
    assert result["paired_pixels"] == 11
    assert result["sensor_valid_fraction_of_category"] == 11 / 12
    assert sum(b["pixels"] for b in result["equal_count_bins_low_to_high"]) == 11
    assert sum(b["pixels"] for b in result["within_frame_category_bins_low_to_high"]) == 11
    assert result["within_frame_category_bins_low_to_high"][0]["mean_frame_mae_mm"] == 1000


def test_high_confidence_can_be_worse_and_category_rank_not_recomputed():
    reference = np.ones((1, 2, 5)); ranks = np.linspace(.6, 1, 10).reshape(reference.shape)
    prediction = reference + ranks * .2
    result = m.audit_category(prediction, reference, ranks, np.ones_like(reference, bool))
    assert result["frozen_rank_top50"]["pixels"] == 10
    assert result["frozen_rank_bottom50"]["pixels"] == 0
    bins = result["equal_count_bins_low_to_high"]
    assert bins[0]["mae_mm"] < bins[-1]["mae_mm"]
    assert bins[0]["rank_min"] == .6


def test_excludes_overlap_without_relabeling_it_as_table():
    hand = np.array([True, True, False, False])
    obj = np.array([False, True, True, False])
    cats = m.exclusive_categories(hand, obj, np.ones(4, bool))
    assert cats["hand"].tolist() == [True, False, False, False]
    assert cats["object"].tolist() == [False, False, True, False]
    assert cats["table"].tolist() == [False, False, False, True]


def test_invalid_predictions_and_empty_categories_do_not_score_as_zeros():
    ref = np.ones((1, 2, 2)); pred = ref.copy(); pred[0, 0, 0] = np.nan
    ranks = np.ones_like(ref) * .5
    result = m.audit_category(pred, ref, ranks, np.ones_like(ref, bool))
    assert result["paired_pixels"] == 3
    empty = m.audit_category(pred, ref, ranks, np.zeros_like(ref, bool))
    assert empty["all"] == {"pixels": 0}
    assert empty["mean_frame_mae_mm"]["all"] is None


def test_metric_only_method_has_no_invented_confidence():
    reference = np.ones((2, 2, 2))
    result = m.metric_only_category(reference * 1.2, reference, np.ones_like(reference, bool))
    assert result["confidence_available"] is False
    assert result["all"]["mae_mm"] == pytest.approx(200)
    assert not any("rank" in k or "bin" in k for k in result)
    assert not any("rank" in k for k in result["all"])
