from pathlib import Path
import json
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks import source_depth_audit
from benchmarks.source_depth_audit import object_statistics
from test_static_depth_calibration import inputs


def test_raw_object_audit_preserves_metric_scale_and_does_not_select_confidence():
    geometry, masks, _, _ = inputs()
    geometry['confidence_available'] = np.asarray(False)
    geometry['confidence_rank'][:] = .001
    raw = geometry['depth_m'].copy()
    report = object_statistics(geometry, masks)
    assert report['confidence_available'] is False
    assert report['confidence_selection_used'] is False
    assert report['observed_frames'] == 5
    assert report['median_foreground_camera_z_m'] == np.median(raw[:, 0, 36, 40])
    np.testing.assert_array_equal(raw, geometry['depth_m'])
    geometry['depth_m'] *= 2.
    changed = object_statistics(geometry, masks)
    assert changed['median_foreground_camera_z_m'] == 2 * report['median_foreground_camera_z_m']
    assert changed['GT_accuracy_measured'] is False


def test_missing_depth_remains_missing_in_raw_comparison():
    geometry, masks, _, _ = inputs()
    geometry['valid'][2] = False
    report = object_statistics(geometry, masks)
    assert report['observed_frames'] == 4
    assert report['rows'][2]['camera_z_quantiles_10_50_90_m'] is None


def test_pending_inference_is_rejected_before_any_prediction_is_opened(tmp_path, monkeypatch):
    bundle = tmp_path / 'pending'; bundle.mkdir()
    (bundle / 'inference.json').write_text(json.dumps(dict(status='running', confidence_available=False)))
    def forbidden(*args, **kwargs):
        raise AssertionError('No prediction or manifest should be read before completion')
    monkeypatch.setattr(source_depth_audit.base, '_materialized_manifest', forbidden)
    with pytest.raises(ValueError, match='Wait for validated complete'):
        source_depth_audit.compare(unidepth_bundle=tmp_path / 'unidepth', da3_bundle=bundle,
            mask_bundle=tmp_path / 'masks', config_path=tmp_path / 'nonexistent-config', output=tmp_path / 'result')
    assert not (tmp_path / 'result').exists()
