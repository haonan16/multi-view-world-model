from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artirecon.scene_sequence import scene_from_arrays, object_from_arrays
from benchmarks import source_contact_phases as phase


def inputs():
    frames = 11
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    poses[:, 2, 3] = .02
    poses[3:8, 2, 3] = [.04, .08, .1, .08, .04]
    provenance = {"method": "synthetic fixture", "artifact_ids": ["sam3_masks"], "confidence_status": "uncalibrated"}
    contacts = [dict(id=f"support-{i}", a="block", b="table", kind="support", start_frame=first, end_frame=last,
                     status="hypothesized", confidence=None, provenance=provenance) for i, (first, last) in enumerate([(0, 2), (8, 10)])]
    # An earlier depth threshold split is retained in the report, never used to
    # select the new phase boundaries.
    contacts += [dict(id=f"raw-grasp-{i}", a="block", b="world", kind="grasp", start_frame=first, end_frame=last,
                      status="hypothesized", confidence=None, provenance=provenance) for i, (first, last) in enumerate([(3, 4), (6, 7)])]
    scene = scene_from_arrays(timestamps=np.arange(frames) * .1,
        coordinate_frame={"name": "synthetic", "handedness": "right", "up_axis": "+z", "pose_convention": "T_world_local"},
        metric_scale={"status": "estimated", "provenance": provenance}, actors=[],
        objects=[object_from_arrays(object_id="block", poses=poses, observed=[True] * frames, provenance=provenance,
                                   primitive={"type": "box", "half_extents_m": [.02] * 3})],
        support={"id": "table", "normal_world": [0, 0, 1], "offset_m": 0, "confidence": None, "provenance": provenance},
        contacts=contacts, uncertainty={"status": "unestimated"}, task={"description": "synthetic pickup", "target": None})
    centers = np.c_[[0, 0, 0, 1, 6, 11, 16, 21, 21, 21, 21], np.zeros(frames)]
    evidence = dict(object_centers=centers, hand_centers=centers + [0, 2],
                    gaps=np.array([30, 30, 30, 1, 1, 1, 1, 1, 30, 30, 30]),
                    diagonals=np.ones(frames) * 20, valid=np.ones(frames, bool),
                    support=np.array([True, True, True, False, False, False, False, False, True, True, True]))
    return scene, evidence


def test_reestimates_from_contiguous_rgb_evidence_without_changing_geometry():
    scene, evidence = inputs()
    original = deepcopy(scene)
    result, report = phase.infer_phases(scene, evidence, phase.default_config())
    assert scene == original
    assert result["objects"] == scene["objects"]
    grasp = [c for c in result["contacts"] if c["kind"] == "grasp"]
    assert [(c["start_frame"], c["end_frame"]) for c in grasp] == [(3, 7)]
    assert grasp[0]["status"] == "hypothesized" and grasp[0]["confidence"] is None
    assert len(report["raw_grasp_hypotheses"]) == 2
    assert report["missing_intervals_merged"] is False


def test_missing_rgb_evidence_is_not_bridged():
    scene, evidence = inputs()
    evidence["valid"][5] = False
    result, report = phase.infer_phases(scene, evidence, phase.default_config())
    assert report["accepted_intervals"] == 0
    assert report["missing_evidence_frames"] == [5]
    assert not any(c["kind"] == "grasp" for c in result["contacts"])


def test_pushing_or_static_proximity_is_not_called_a_grasp():
    scene, evidence = inputs()
    for pose in scene["objects"][0]["T_world_object"]:
        pose[2][3] = .02
    _, report = phase.infer_phases(scene, evidence, phase.default_config())
    assert report["accepted_intervals"] == 0
    assert "estimated lift not established; could be pushing" in report["candidates"][0]["reasons"]


def test_release_requires_independent_supported_observation():
    scene, evidence = inputs()
    evidence["support"][8:] = False
    _, report = phase.infer_phases(scene, evidence, phase.default_config())
    assert report["accepted_intervals"] == 0
    assert "first released frame lacks support/observation evidence" in report["candidates"][0]["reasons"]
