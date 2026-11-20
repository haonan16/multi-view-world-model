"""Object motion from model point tracks, with explicit unobserved frames.

This stage estimates relative SE(3), not absolute placement or hidden shape.
A source-only first-frame mesh fit supplies the initial pose and shared shape.
The caller must authenticate that fit and the queried object identities. No
sensor depth, camera calibration, simulator, or evaluation target is read here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.scene_sequence import object_from_arrays, scene_from_arrays


@dataclass(frozen=True)
class RigidConfig:
    residual_threshold_m: float = .01
    min_inliers: int = 8
    min_inlier_fraction: float = .5
    hypotheses: int = 128
    min_span_m: float = .003
    min_second_axis_ratio: float = .02
    seed: int = 20261119

    def __post_init__(self):
        if (not np.isfinite(self.residual_threshold_m) or self.residual_threshold_m <= 0
                or type(self.min_inliers) is not int or self.min_inliers < 3
                or not 0 < self.min_inlier_fraction <= 1
                or type(self.hypotheses) is not int or self.hypotheses < 1
                or not np.isfinite(self.min_span_m) or self.min_span_m <= 0
                or not 0 < self.min_second_axis_ratio < 1):
            raise ValueError('Invalid rigid consensus configuration')


def _spread(points, config):
    singular = np.linalg.svd(points - points.mean(0), compute_uv=False) / np.sqrt(len(points))
    return (len(singular) >= 2 and singular[0] >= config.min_span_m
            and singular[1] >= config.min_second_axis_ratio * singular[0])


def _kabsch(source, destination, config):
    """Proper rotation; collinear or spatially tiny points do not fix orientation."""
    if len(source) < 3 or not _spread(source, config) or not _spread(destination, config):
        return None
    a, b = source.mean(0), destination.mean(0)
    u, _, vt = np.linalg.svd((source - a).T @ (destination - b))
    sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ sign @ u.T
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = r, b - r @ a
    return pose


def _errors(pose, source, destination):
    return np.linalg.norm(source @ pose[:3, :3].T + pose[:3, 3] - destination, axis=-1)


def robust_rigid(source, destination, config=RigidConfig()):
    """Unweighted RANSAC; model confidence is not assumed metric certainty.

    Returns a transform or None, original-row inliers, and a consistency report.
    A small residual verifies agreement between *predicted* correspondences;
    it cannot certify geometry, metric scale, physical contact, or correct IDs.
    """
    source, destination = np.asarray(source, float), np.asarray(destination, float)
    if source.ndim != 2 or source.shape[1:] != (3,) or source.shape != destination.shape:
        raise ValueError('Corresponding points must be matching [N,3] arrays')
    finite = np.isfinite(source).all(-1) & np.isfinite(destination).all(-1)
    a, b = source[finite], destination[finite]
    empty = np.zeros(len(source), bool)
    report = dict(candidates=int(finite.sum()), inliers=0, status='insufficient_correspondences',
                  residual_m=None, confidence_status='uncalibrated_model_consistency')
    if len(a) < config.min_inliers:
        return None, empty, report
    required = max(config.min_inliers, int(np.ceil(config.min_inlier_fraction * len(a))))
    rng = np.random.default_rng(config.seed)
    selections = [np.arange(len(a))]
    selections.extend(rng.choice(len(a), 3, replace=False) for _ in range(config.hypotheses))
    best_pose, best_mask, best_score = None, None, (-1, -np.inf)
    for selection in selections:
        pose = _kabsch(a[selection], b[selection], config)
        if pose is None:
            continue
        error = _errors(pose, a, b)
        inside = error <= config.residual_threshold_m
        count = int(inside.sum())
        score = (count, -float(np.median(error[inside])) if count else -np.inf)
        if score > best_score:
            best_pose, best_mask, best_score = pose, inside, score
    if best_pose is None:
        report['status'] = 'rotation_unobservable'
        return None, empty, report
    if best_score[0] < required:
        report['status'] = 'no_rigid_consensus'
        return None, empty, report
    # Re-estimate on consensus; never fit a new scale to make motion look rigid.
    for _ in range(5):
        refined = _kabsch(a[best_mask], b[best_mask], config)
        if refined is None:
            report['status'] = 'rotation_unobservable'
            return None, empty, report
        error = _errors(refined, a, b)
        new_mask = error <= config.residual_threshold_m
        if new_mask.sum() < required:
            report['status'] = 'no_rigid_consensus'
            return None, empty, report
        best_pose = refined
        if np.array_equal(new_mask, best_mask):
            break
        best_mask = new_mask
    inliers = empty.copy()
    inliers[np.flatnonzero(finite)] = new_mask
    report.update(status='rigid_consensus', inliers=int(new_mask.sum()),
                  residual_m=float(np.median(error[new_mask])),
                  residual_p95_m=float(np.quantile(error[new_mask], .95)))
    return best_pose, inliers, report


def fit_object_track(tracks3d, visible, *, config=RigidConfig()):
    """Relative poses to the query frame (index 0), never chained frame-to-frame.

    ``visible`` must already be a boolean evidence gate. Any threshold applied
    to model visibility probabilities must be recorded by the caller. Missing
    poses carry forward solely as an initialization placeholder and are marked
    unobserved. This function does not impute or label a contact state.
    """
    points, visible = np.asarray(tracks3d, float), np.asarray(visible)
    if points.ndim != 3 or points.shape[-1] != 3 or not len(points):
        raise ValueError('tracks3d must be nonempty [T,N,3]')
    if visible.shape != points.shape[:2] or visible.dtype.kind != 'b':
        raise ValueError('visible must be boolean [T,N] with explicit caller gating')
    poses = np.tile(np.eye(4), (len(points), 1, 1))
    observed = np.zeros(len(points), bool)
    inliers = np.zeros(points.shape[:2], bool)
    reports = []
    for t in range(len(points)):
        ids = np.flatnonzero(visible[0] & visible[t])
        pose, kept, report = robust_rigid(points[0, ids], points[t, ids], config)
        if pose is not None:
            poses[t], observed[t], inliers[t, ids] = pose, True, kept
        elif t:
            poses[t] = poses[t - 1]
        reports.append(dict(frame_index=t, **report))
    return dict(relative_poses=poses, observed=observed, inliers=inliers, frames=reports)


def initialize_scene(*, vertices_m, faces, first_pose, motion, timestamps,
                     coordinate_frame, metric_scale, provenance, object_id='task_object'):
    """Bind independently source-fitted mesh to relative rigid motion.

    A constant affine first pose (e.g. source-fitted mesh scale) is factored by
    the shared scene schema into SE(3) poses plus constant local mesh stretch.
    No support/contact relation is silently inferred from proximity.
    """
    first_pose = np.asarray(first_pose, float)
    relative = np.asarray(motion['relative_poses'], float)
    if first_pose.shape != (4, 4) or len(relative) != len(timestamps):
        raise ValueError('First mesh pose / trajectory timestamps do not match')
    details = dict(provenance, method='Shared source-fit mesh + robust predicted-point SE3',
                   confidence_status='uncalibrated',
                   rotation_status='estimated_where_observed',
                   rotation_observed=np.asarray(motion['observed'], bool).tolist(),
                   hidden_surface_status='model_completion_unverified',
                   fallback_pose_status='unobserved_carry_forward',
                   absolute_placement_status='inherited_source_prediction_not_corrected',
                   scale_status='inherited_source_prediction_not_corrected')
    obj = object_from_arrays(object_id=object_id, poses=relative @ first_pose,
        observed=motion['observed'], vertices_m=vertices_m, faces=faces, provenance=details)
    return scene_from_arrays(timestamps=timestamps, coordinate_frame=coordinate_frame,
        metric_scale=metric_scale, actors=[], objects=[obj], support=None, contacts=[],
        uncertainty=dict(status='estimated', provenance=details,
            interpretation='Rigid consensus is model consistency, not calibrated accuracy. '
                           'Hidden geometry, contact and unsupported poses remain unverified.'),
        task=dict(description='Recover a shared task object and its motion from model-derived RGB evidence', target=None))
