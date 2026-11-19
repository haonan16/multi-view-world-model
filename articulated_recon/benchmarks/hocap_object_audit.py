"""Score frozen source-only meshes against HO-Cap CAD/poses, without alignment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from hocap_annotation_audit import (bind_hand, read_verified_annotations, write_report,
                                    require_annotation_members, assert_stable)
from heldout_mano_audit import source_to_view, transform_points
from depth_confidence_audit import verify_exact_frames
from artirecon.input_protocol import file_sha256, load_manifest
from mano_recon.run_integrity import artifact_record


def surface_comparison(predicted, reference, predicted_centroid, reference_centroid):
    """Unaligned metric distances. Centroids must use the same declared definition."""
    p, r = np.asarray(predicted), np.asarray(reference)
    if (p.ndim != 2 or r.ndim != 2 or p.shape[1] != 3 or r.shape[1] != 3
            or not len(p) or not len(r) or not np.isfinite(p).all() or not np.isfinite(r).all()):
        raise ValueError('Expected finite nonempty sampled surfaces')
    a, b = cKDTree(r).query(p)[0], cKDTree(p).query(r)[0]
    return dict(surface_centroid_error_mm=float(np.linalg.norm(predicted_centroid-reference_centroid)*1000),
                symmetric_sampled_surface_distance_mm=float(.5*(a.mean()+b.mean())*1000),
                predicted_to_CAD_mean_mm=float(a.mean()*1000), CAD_to_predicted_mean_mm=float(b.mean()*1000),
                symmetric_surface_p90_mm=float(.5*(np.quantile(a, .9)+np.quantile(b, .9))*1000))


def summarize_frames(frames, subset):
    selected = [f for f in frames if f['observed'] and (subset == 'all' or f['subset'] == subset)]
    return dict(frames=len(selected), **{k: float(np.mean([f[k] for f in selected])) if selected else None
        for k in ['surface_centroid_error_mm', 'symmetric_sampled_surface_distance_mm', 'symmetric_surface_p90_mm']})


def audit_objects(*, tracks, hand_run, case, annotations, verification, reference,
                  source_index, original_start, object_id, cad, output, samples=10000):
    import trimesh
    import yaml

    if len(tracks) != 2 or samples < 100:
        raise ValueError('Require ordered earlier/later tracks and at least100 surface samples')
    hand = bind_hand(hand_run, case)
    stable = dict(hand['hashes'])
    entries = []
    for folder in map(Path, tracks):
        audit = json.loads((folder / 'execution.json').read_text())
        protocol = load_manifest(folder / 'input-manifest.json',
            [('object_geometry', folder / name) for name in ['scene-sequence.json', 'fitted-mesh.ply', 'mesh-track.npz']])
        if (audit['status'] != 'complete' or protocol['condition'] != 'rgb_only'
                or audit['output_protocol']['manifest_sha256'] != protocol['manifest_sha256']
                or audit['original_hand_manifest_sha256'] != hand['manifest_sha256']
                or audit['frozen_hand'] != artifact_record(Path(hand_run) / 'inputs' / case / 'frozen-hand.npz')):
            raise ValueError('Object track must bind the same frozen source-only hand')
        if not {'mesh-track.npz', 'fitted-mesh.ply', 'scene-sequence.json'} <= set(audit['outputs']):
            raise ValueError('Object receipt omits required frozen outputs')
        for name, record in audit['outputs'].items():
            if artifact_record(folder / name) != record:
                raise ValueError('Frozen output changed: ' + name)
        scene = json.loads((folder / 'scene-sequence.json').read_text())
        if scene['coordinate_frame']['name'] != 'source_camera':
            raise ValueError('Object track must use source-camera coordinates')
        stable.update({str(folder / name): record['sha256'] for name, record in audit['outputs'].items()})
        stable.update({str(folder / name): file_sha256(folder / name) for name in ['execution.json', 'input-manifest.json']})
        with np.load(folder / 'mesh-track.npz') as z:
            pose, ids, ts, observed = [z[k].copy() for k in ['poses', 'frame_ids', 'timestamps', 'observed']]
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or observed.dtype != bool
                or observed.shape != ids.shape or pose.shape != (len(ids), 4, 4)
                or ids.min() < 0 or ids.max() >= len(hand['times']) or not np.all(np.diff(ids) > 0)
                or not np.array_equal(ts, hand['times'][ids])):
            raise ValueError('Object keyframes must be exact frozen source timestamps')
        for matrix in pose:
            source_to_view(np.eye(4), matrix)
        entries.append(dict(name=folder.name, folder=folder, mesh=trimesh.load(folder/'fitted-mesh.ply', force='mesh', process=False),
                            poses=pose, ids=ids, observed=observed, receipt=audit))
    if not np.array_equal(entries[0]['ids'], entries[1]['ids']):
        raise ValueError('Earlier/later tracks must use the same declared keyframes')
    # References first accessed after both completed source-only outputs are bound.
    raw = Path(annotations)
    stable.update({str(p): file_sha256(p) for p in [reference, verification, cad, __file__,
        Path(__file__).with_name('hocap_annotation_audit.py')]})
    hashes = read_verified_annotations(raw, verification)
    stable.update({str(raw / name): digest for name, digest in hashes.items()})
    require_annotation_members(hashes, 'meta.yaml', *[f'label_{i+original_start:06d}.npz' for i in entries[0]['ids']])
    if file_sha256(cad) not in hashes.values():
        raise ValueError('CAD is missing from verified archive extraction')
    with np.load(reference) as z:
        verify_exact_frames(hand['rgb'], hand['times'], z['rgb'][:, source_index], z['timestamps'])
        source_k = z['K'][source_index].copy()
    metadata = yaml.safe_load((raw / 'meta.yaml').read_text())
    object_index = metadata['object_ids'].index(object_id)
    gt = trimesh.load(cad, force='mesh', process=False)
    if not .01 < gt.extents.max() < 1:
        raise ValueError('Unexpected CAD scale: this adapter expects metres, with no rescaling')
    cad_points, _ = trimesh.sample.sample_surface(gt, samples, seed=100)
    reference_extent = np.sort(gt.bounding_box_oriented.primitive.extents)
    common = entries[0]['observed'] & entries[1]['observed']
    recovered = ~entries[0]['observed'] & entries[1]['observed']
    rows = []
    for entry in entries:
        mesh = entry['mesh']
        sampled, _ = trimesh.sample.sample_surface(mesh, samples, seed=101)
        frames = []
        for i, source_id in enumerate(entry['ids']):
            frame = dict(native_source_frame=int(source_id), original_frame=int(source_id)+original_start,
                         observed=bool(entry['observed'][i]), subset='common' if common[i] else 'recovered' if recovered[i] else 'other')
            if frame['observed']:
                with np.load(raw / f'label_{source_id+original_start:06d}.npz') as z:
                    if z['obj_class_names'][object_index] != object_id or not np.array_equal(z['cam_K'], source_k):
                        raise ValueError('Reference object/camera identity mismatch')
                    gt_pose = z['obj_poses'][object_index].copy()
                source_to_view(np.eye(4), gt_pose)
                frame.update(surface_comparison(transform_points(sampled, entry['poses'][i]),
                    transform_points(cad_points, gt_pose), transform_points(mesh.centroid, entry['poses'][i]),
                    transform_points(gt.centroid, gt_pose)))
            frames.append(frame)
        extent = np.sort(mesh.bounding_box_oriented.primitive.extents)
        folder = entry['folder']
        rows.append(dict(name=entry['name'], mesh_sha256=file_sha256(folder/'fitted-mesh.ply'),
                         track_sha256=file_sha256(folder/'mesh-track.npz'), observed_keyframes=int(entry['observed'].sum()),
                         same_frozen_hand=entry['receipt']['frozen_hand'], sorted_OBB_extent_mm=(extent*1000).tolist(),
                         sorted_extent_MAE_mm=float(abs(extent-reference_extent).mean()*1000),
                         watertight=bool(mesh.is_watertight), reference_watertight=bool(gt.is_watertight),
                         volume_ratio=abs(float(mesh.volume/gt.volume)) if mesh.is_watertight and gt.is_watertight else None,
                         subsets={s: summarize_frames(frames, s) for s in ['all', 'common', 'recovered']}, per_keyframe=frames))
    report = dict(schema='hocap-frozen-object-CAD-audit-packaged-v1', evaluation_only=True,
                  object_id=object_id, reference_CAD_sha256=file_sha256(cad), annotation_sha256=hashes,
                  reference_sha256=file_sha256(reference), verification_sha256=file_sha256(verification),
                  code_sha256=file_sha256(__file__), reference_sorted_OBB_extent_mm=(reference_extent*1000).tolist(),
                  units='CAD and camera-relative object poses in metres; no rescaling or registration correction',
                  same_ground_truth_for_both=True, no_pose_alignment=True, no_scale_alignment=True, simulator_accessed=False,
                  sampling=dict(count=samples, CAD_seed=100, predicted_seed=101, distribution='area-uniform surface'),
                  variants=rows, limits=['Nearest sampled surface approximates continuous point-to-triangle distance.',
                     'Surface-area centroids avoid arbitrary mesh-origin comparisons; no topology correspondence assumed.',
                     'Unobserved poses excluded; common and recovered sets reported separately.',
                     'Dataset CAD and pose annotations may contain residual error.'])
    assert_stable(stable)
    report['verified_input_hashes'] = stable
    write_report(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', type=Path, nargs=2, required=True)
    for name in ['hand-run', 'annotations', 'verification', 'reference', 'cad', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--source-index', type=int, required=True)
    parser.add_argument('--original-start', type=int, required=True)
    parser.add_argument('--object-id', required=True)
    parser.add_argument('--samples', type=int, default=10000)
    audit_objects(**vars(parser.parse_args()))
