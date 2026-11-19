"""Frozen hand geometry metrics, without scale, rotation or pose alignment."""
import numpy as np
from scipy.spatial import cKDTree


def annotation_valid(joints):
    """HO-Cap's absent hand is all -1, not a real hand at negative coordinates."""
    joints = np.asarray(joints)
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        raise ValueError('Expected T,21,3 annotated joints')
    return np.isfinite(joints).all(axis=(1, 2)) & ~np.all(joints == -1, axis=(1, 2))


def _summarize(values):
    values = np.asarray(values, float)
    return {'mean_mm': float(values.mean() * 1000),
            'median_mm': float(np.median(values) * 1000),
            'p90_mm': float(np.quantile(values, .9) * 1000),
            'per_frame_mean_mm': (values.reshape(len(values), -1).mean(1) * 1000).tolist()}


def score_joints(predicted, reference, observed):
    p, r, seen = np.asarray(predicted), np.asarray(reference), np.asarray(observed)
    if p.shape != r.shape or p.ndim != 3 or p.shape[1:] != (21, 3) or seen.shape != (len(p),) or seen.dtype != bool:
        raise ValueError('Expected matched T,21,3 joints and explicit boolean observation flags')
    valid = seen & annotation_valid(r) & np.isfinite(p).all(axis=(1, 2))
    if not valid.any():
        raise ValueError('No mutually valid annotated/predicted hands')
    p, r = p[valid], r[valid]
    absolute = np.linalg.norm(p-r, axis=-1)
    relative = np.linalg.norm((p-p[:, :1])-(r-r[:, :1]), axis=-1)
    tips = [4, 8, 12, 16, 20]
    return dict(frames=int(valid.sum()), source_frame_indices=np.flatnonzero(valid).tolist(),
                absolute_joint=_summarize(absolute), absolute_wrist=_summarize(absolute[:, :1]),
                absolute_fingertips=_summarize(absolute[:, tips]),
                wrist_relative_joint=_summarize(relative), wrist_relative_fingertips=_summarize(relative[:, tips]),
                translation_diagnostic='Subtract each hand own wrist only; no rotation or scale alignment',
                rotation_alignment=False, scale_alignment=False)


def score_surface(predicted_vertices, reference_vertices, predicted_wrist, reference_wrist, valid):
    p, r = np.asarray(predicted_vertices), np.asarray(reference_vertices)
    if p.shape != r.shape or p.ndim != 3 or p.shape[2] != 3 or valid.shape != (len(p),) or valid.dtype != bool:
        raise ValueError('Expected matched T,V,3 shared-topology vertices and valid flags')
    if not valid.any() or not np.isfinite(p[valid]).all() or not np.isfinite(r[valid]).all():
        raise ValueError('No valid finite mesh comparison')
    p, r = p[valid], r[valid]
    rel_p = p-np.asarray(predicted_wrist)[valid, None]
    rel_r = r-np.asarray(reference_wrist)[valid, None]
    def chamfer(a, b):
        return np.array([.5*(cKDTree(y).query(x)[0].mean()+cKDTree(x).query(y)[0].mean())
                         for x, y in zip(a, b)])
    return dict(absolute_corresponded_vertex=_summarize(np.linalg.norm(p-r, axis=-1)),
                wrist_relative_corresponded_vertex=_summarize(np.linalg.norm(rel_p-rel_r, axis=-1)),
                absolute_symmetric_vertex_nn=_summarize(chamfer(p, r)),
                wrist_relative_symmetric_vertex_nn=_summarize(chamfer(rel_p, rel_r)),
                surface_metric='MANO corresponding-vertex error plus symmetric nearest-vertex distance; not point-to-triangle distance',
                alignment='None for absolute metrics; own-wrist translation subtraction only for relative metrics')
