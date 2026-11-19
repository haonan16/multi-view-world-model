"""The runner's shared-scene bridge cannot invent evidence or metric scale."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artirecon.scene_exports import fit_timestamps, mano_fit_scene
from artirecon.scene_sequence import SceneValidationError, load_scene


def fitted_arrays():
    q = np.zeros((4, 15))
    q[:, 0] = [0, .1, .2, .3]
    base = np.repeat(np.eye(4)[None], 4, axis=0)
    base[:, :3, 3] = [[.1, .2, .3], [.2, .3, .4], [.3, .4, .5], [.4, .5, .6]]
    return dict(q=q, base=base, betas=np.arange(10)[None] / 10,
                timestamps=np.array([2., 2.03, 2.07, 2.12]),
                point_counts=np.array([25, 0, 12, 4]),
                visible=np.array([True, True, False, True]),
                model_uri="/licensed/assets/MANO_RIGHT.npz", support_normal=[0, 0, 1], support_offset=-.2)


def verified_protocol(status="estimated"):
    return {"status": "validated", "files_verified": True, "condition": "rgb_only",
            "method_artifact_ids": ["rgb", "depth", "mano_init"],
            "metric_scale": {"status": status, "artifact_ids": ["depth"],
                             "description": "Predicted metric depth; scale remains estimated"},
            "robot_registration": {"status": "unresolved", "artifact_ids": [], "description": "Not available"},
            "execution_ready": False, "execution_blockers": ["robot_registration unresolved"]}


def test_export_preserves_state_and_marks_missing_evidence(tmp_path):
    arrays = fitted_arrays()
    scene = mano_fit_scene(**arrays)
    actor = scene["actors"][0]
    assert actor["observed"] == [True, False, False, True]
    np.testing.assert_array_equal(actor["q"], arrays["q"])
    np.testing.assert_array_equal(actor["T_world_base"], arrays["base"])
    np.testing.assert_array_equal(actor["shared_shape"], arrays["betas"][0])
    assert actor["model_uri"] == arrays["model_uri"]
    assert scene["timestamps"] == arrays["timestamps"].tolist()
    assert scene["objects"] == scene["contacts"] == []
    assert scene["support"]["confidence"] is None
    assert scene["support"]["offset_m"] == -.2
    assert scene["support"]["provenance"]["confidence_status"] == "uncalibrated"
    assert scene["metric_scale"]["status"] == "unresolved"
    assert scene["input_protocol"]["status"] == "unverified"
    assert scene["input_protocol"]["condition"] is None
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene, allow_nan=False))
    assert load_scene(path) == scene
    assert list(tmp_path.iterdir()) == [path]  # No licensed model copied.


def test_declared_verified_scale_and_condition_preserved_without_registration():
    report = verified_protocol()
    scene = mano_fit_scene(**fitted_arrays(), input_protocol=report)
    assert scene["metric_scale"]["status"] == "estimated"
    assert scene["actors"][0]["provenance"]["condition"] == "rgb_only"
    assert scene["input_protocol"]["robot_registration"]["status"] == "unresolved"
    assert scene["input_protocol"]["execution_ready"] is False
    assert scene["export_scope"]["robot_registration_established_by_export"] is False
    report["metric_scale"]["status"] = "known"
    assert scene["input_protocol"]["metric_scale"]["status"] == "estimated"


@pytest.mark.parametrize("change", [
    {"status": "unverified"}, {"files_verified": False}, {"files_verified": None}])
def test_unverified_scale_is_not_promoted_from_model_units(change):
    report = verified_protocol("known")
    report.update(change)
    scene = mano_fit_scene(**fitted_arrays(), input_protocol=report)
    assert scene["metric_scale"]["status"] == "unresolved"
    assert scene["metric_scale"]["provenance"]["artifact_ids"] == []


@pytest.mark.parametrize("field, value", [
    ("point_counts", [1, 2, 3]), ("point_counts", [1, -1, 0, 1]),
    ("point_counts", [1, 1.5, 0, 1]), ("point_counts", [1, np.nan, 0, 1]),
    ("visible", [1, 1, 1, 1]), ("visible", [True, False]),
    ("support_normal", None),
])
def test_invalid_evidence_or_partial_support_rejected(field, value):
    arrays = fitted_arrays()
    arrays[field] = value
    with pytest.raises(SceneValidationError):
        mano_fit_scene(**arrays)


def test_timestamps_preserve_actual_clock_instead_of_fps():
    times, provenance = fit_timestamps({"timestamps": [4, 4.04, 4.13, 4.3]}, 3, 30, require_supplied=True)
    np.testing.assert_array_equal(times, [4, 4.04, 4.13])
    assert "observation" in provenance
    legacy, provenance = fit_timestamps({}, 3, 30)
    np.testing.assert_allclose(legacy, np.arange(3) / 30)
    assert "unavailable" in provenance


def test_verified_fit_must_not_synthesize_timestamps():
    with pytest.raises(SceneValidationError, match="requires observation timestamps"):
        fit_timestamps({}, 3, 30, require_supplied=True)


@pytest.mark.parametrize("times", [[0, 1], [0, 1, 1], [0, np.nan, 2], [[0], [1], [2]], 0, ["bad"]])
def test_malformed_actual_clock_not_replaced_by_fps(times):
    with pytest.raises(SceneValidationError, match="timestamps"):
        fit_timestamps({"timestamps": times}, 3, 30)


@pytest.mark.parametrize("fps", [None, 0, -1, np.inf, "30", True])
def test_legacy_fallback_requires_positive_fps(fps):
    with pytest.raises(SceneValidationError, match="fps"):
        fit_timestamps({}, 3, fps)
