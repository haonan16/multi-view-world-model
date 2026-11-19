"""Evaluator-only adapter for exact HO-Cap labels and frozen source-camera fits.

No reference arrays are returned to fitting. Licensed models and dataset files
stay outside the repository. This enforces an interface boundary, not OS isolation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).parents[1]), str(Path(__file__).parents[1] / 'mano')]
from artirecon.input_protocol import file_sha256, load_manifest
from mano_recon.run_integrity import verify_cached_outputs
from depth_confidence_audit import verify_exact_frames
from hand_annotation_metrics import annotation_valid, score_joints, score_surface
from heldout_mano_audit import source_to_view, transform_points


def read_verified_annotations(directory, verification):
    """Verify the extracted dataset files against the cached archive index."""
    directory, verification = Path(directory), Path(verification)
    receipt = json.loads(verification.read_text())
    if receipt.get('status') != 'verified':
        raise ValueError('Require verified cached annotation extraction')
    hashes = {}
    for record in receipt['files']:
        path = directory / Path(record['output']).name
        if path.name in hashes or file_sha256(path) != record['sha256']:
            raise ValueError('Duplicate name or changed annotation: ' + str(path))
        hashes[path.name] = record['sha256']
    return hashes


def require_annotation_members(hashes, *names):
    missing = set(names) - set(hashes)
    if missing:
        raise ValueError('Consumed annotations absent from verification receipt: ' + ', '.join(sorted(missing)))


def assert_stable(hashes):
    for path, expected in hashes.items():
        if file_sha256(path) != expected:
            raise ValueError('Input changed during evaluation: ' + str(path))


def bind_hand(run, case, allow_legacy_frame=False):
    """Verify completed source-only outputs before opening evaluator inputs."""
    run = Path(run).resolve()
    inp, fit = run / 'inputs' / case, run / 'hand-results' / case
    execution = json.loads((fit / 'execution.json').read_text())
    protocol = json.loads((inp / 'protocol.json').read_text())
    if (execution['status'] != 'complete' or execution['input_protocol']['condition'] != 'rgb_only'
            or execution['depth_self_calibration'] or execution['heldout_used']):
        raise ValueError('Require frozen RGB-only fit without calibration/refit using held-out views')
    report = load_manifest(inp / 'input-manifest.json', [('depth', inp / 'obs.npz'),
        ('cameras', inp / 'obs.npz'), ('hand_initialization', inp / 'hand-input.npz')])
    if report['manifest_sha256'] != execution['input_protocol']['manifest_sha256']:
        raise ValueError('Input manifest changed')
    verify_cached_outputs(execution, {'fit': fit / 'fit.npz', 'scene_sequence': fit / 'scene-sequence.json',
                                      'frozen_hand': inp / 'frozen-hand.npz'})
    frame = protocol.get('coordinate_frame', {}).get('name')
    if frame != 'source_camera' and not allow_legacy_frame:
        raise ValueError('Expected explicit source_camera; legacy identity-base evidence requires opt-in')
    with np.load(inp / 'obs.npz') as z:
        if z['rgb'].shape[1] != 1 or not np.array_equal(z['c2w'], np.eye(4)[None]):
            raise ValueError('Expected single source camera with identity transform')
        rgb, times = z['rgb'][:, 0].copy(), z['timestamps'].copy()
    with np.load(fit / 'fit.npz') as z:
        vertices, joints = z['vertices'].copy(), z['joints'].copy()
    with np.load(fit / 'hand-sequence.npz') as z:
        if not np.array_equal(vertices, z['vertices']) or not np.array_equal(times, z['timestamps']):
            raise ValueError('Frozen geometry/timestamps changed')
        visible = z['visible'].copy()
    paths = [fit / 'fit.npz', fit / 'hand-sequence.npz', fit / 'execution.json',
             inp / 'protocol.json', inp / 'obs.npz', inp / 'input-manifest.json', inp / 'frozen-hand.npz']
    return dict(name=run.name, side=protocol['mano_side'], vertices=vertices, joints=joints,
                visible=visible, rgb=rgb, times=times, run=run,
                hashes={str(p): file_sha256(p) for p in paths}, frame=frame,
                manifest_sha256=report['manifest_sha256'],
                legacy_frame_opt_in=frame != 'source_camera')


def published_camera(extrinsics, camera_id):
    """HO-Cap world is tag_1; saved extrinsics map cameras/tags into capture rig."""
    def matrix(values):
        result = np.array(list(values) + [0, 0, 0, 1], dtype=float).reshape(4, 4)
        source_to_view(np.eye(4), result)
        return result
    return np.linalg.inv(matrix(extrinsics['tag_1'])) @ matrix(extrinsics[camera_id])


def load_labels(directory, original_start, count, side):
    """Preserve absent-hand flags; labels already use the selected camera frame."""
    hand_id = {'right': 0, 'left': 1}[side]
    joints, pixels, intrinsics, objects = [], [], [], []
    for original in range(original_start, original_start + count):
        with np.load(Path(directory) / f'label_{original:06d}.npz') as z:
            joints.append(z['hand_joints_3d'][hand_id].copy())
            pixels.append(z['hand_joints_2d'][hand_id].copy())
            intrinsics.append(z['cam_K'].copy())
            objects.append(z['obj_poses'].copy())
    return np.stack(joints), np.stack(pixels), np.stack(intrinsics), np.stack(objects)


def verify_reference_mesh(vertices, generated_joints, labels, tip_vertices, valid, tolerance=1e-6):
    """Confirm declared GT landmark convention before allowing surface scores."""
    generated = np.asarray(generated_joints).copy()
    if (len(tip_vertices) != 5 or min(tip_vertices) < 0
            or max(tip_vertices) >= vertices.shape[1]):
        raise ValueError('Expected five valid reference fingertip indices')
    generated[:, [4, 8, 12, 16, 20]] = vertices[:, tip_vertices]
    errors = np.linalg.norm(generated[valid] - labels[valid], axis=-1)
    if not valid.any() or not np.isfinite(errors).all() or errors.max() > tolerance:
        raise ValueError('Declared reference MANO convention does not reproduce dataset labels')
    return errors


def audit_hands(*, runs, case, annotations, verification, reference, source_index,
                camera_ids, original_start, side, mano_models, reference_tip_vertices,
                output, allow_legacy_frame=False):
    import torch
    import yaml
    from mano_recon.model import MANOHand

    variants = [bind_hand(r, case, allow_legacy_frame) for r in runs]
    for v in variants[1:]:
        verify_exact_frames(v['rgb'], v['times'], variants[0]['rgb'], variants[0]['times'])
    raw, output = Path(annotations), Path(output)
    stable = {str(p): file_sha256(p) for p in [reference, verification, __file__,
        Path(__file__).with_name('hand_annotation_metrics.py'), Path(mano_models) / f'MANO_{side.upper()}.npz']}
    for variant in variants:
        stable.update(variant['hashes'])
    hashes = read_verified_annotations(raw, verification)
    stable.update({str(raw / name): digest for name, digest in hashes.items()})
    require_annotation_members(hashes, 'meta.yaml')
    metadata = yaml.safe_load((raw / 'meta.yaml').read_text())
    if side not in metadata['mano_sides']:
        raise ValueError('Requested side is absent from sequence metadata')
    count = len(variants[0]['times'])
    require_annotation_members(hashes, 'poses_m.npy', metadata['extrinsics'], metadata['subject_id'] + '.yaml',
        *[f'label_{i:06d}.npz' for i in range(original_start, original_start + count)])
    labels, pixels, known_k, _ = load_labels(raw, original_start, count, side)
    valid = annotation_valid(labels)
    with np.load(reference) as z:
        verify_exact_frames(variants[0]['rgb'], variants[0]['times'], z['rgb'][:, source_index], z['timestamps'])
        old_c, old_k = z['c2w'].copy(), z['K'].copy()
    if len(camera_ids) != len(old_c) or not np.array_equal(known_k, np.repeat(old_k[source_index][None], count, axis=0)):
        raise ValueError('Source camera identity/intrinsics differ from label convention')
    extrinsics = yaml.safe_load((raw / metadata['extrinsics']).read_text())['extrinsics']
    raw_c = np.stack([published_camera(extrinsics, c) for c in camera_ids])
    world_changes = old_c @ np.linalg.inv(raw_c)
    relative_errors = [float(abs(source_to_view(old_c[source_index], old_c[i]) -
                                 source_to_view(raw_c[source_index], raw_c[i])).max()) for i in range(len(old_c))]
    if max(relative_errors) > 2e-6 or abs(world_changes - world_changes[source_index]).max() > 2e-6:
        raise ValueError('Historical reference contains per-camera changes, not a common world gauge')
    projected = labels[valid] @ known_k[0].T
    pixel_error = np.linalg.norm(projected[..., :2] / projected[..., 2:] - pixels[valid], axis=-1)
    if pixel_error.max() >= 1.5:
        raise ValueError('3D labels do not project to their own annotated pixels')
    poses = np.load(raw / 'poses_m.npy')[{'right': 0, 'left': 1}[side], original_start:original_start+count]
    pose_valid = np.isfinite(poses).all(1) & ~np.all(poses == -1, axis=1)
    if poses.shape != (count, 51) or np.any(valid & ~pose_valid):
        raise ValueError('Dataset MANO parameter presence disagrees with joint labels')
    betas = np.asarray(yaml.safe_load((raw / (metadata['subject_id'] + '.yaml')).read_text())['betas'], np.float32)
    hand = MANOHand(mano_models, side=side, components=45, device='cpu')
    safe_poses = poses.copy(); safe_poses[~pose_valid] = 0
    p = torch.as_tensor(safe_poses, dtype=torch.float32)
    with torch.no_grad():
        # Do not call the wrist-recentered MANOHand.forward adapter for GT parameters.
        result = hand.mano(global_orient=p[:, :3], hand_pose=p[:, 3:48],
                           betas=torch.from_numpy(betas)[None].expand(count, -1), transl=p[:, 48:])
    world_vertices, world_joints = result.vertices.numpy(), result.joints.numpy()
    order = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    joints = np.concatenate([world_joints, world_vertices[:, reference_tip_vertices]], axis=1)[:, order]
    transform = np.linalg.inv(raw_c[source_index])
    reference_vertices, reference_joints = transform_points(world_vertices, transform), transform_points(joints, transform)
    errors = verify_reference_mesh(reference_vertices, reference_joints, labels, reference_tip_vertices, valid)
    upstream_tips = {'right': [745, 317, 444, 556, 673], 'left': [745, 317, 445, 556, 673]}
    rows = []
    for variant in variants:
        p, v, seen = variant['joints'], variant['vertices'], variant['visible']
        upstream, declared = p.copy(), p.copy()
        upstream[:, [4, 8, 12, 16, 20]] = v[:, upstream_tips[variant['side']]]
        declared[:, [4, 8, 12, 16, 20]] = v[:, reference_tip_vertices]
        rows.append(dict(name=variant['name'], side=variant['side'], hashes=variant['hashes'],
                         legacy_frame_opt_in=variant['legacy_frame_opt_in'],
                         original_exported_joint_convention=score_joints(p, labels, seen),
                         official_manopth_tip_convention=score_joints(upstream, labels, seen),
                         verified_dataset_vertex_definition=score_joints(declared, labels, seen),
                         surface=score_surface(v, reference_vertices, p[:, 0], labels[:, 0], seen & valid)))
    report = dict(schema='hocap-frozen-hand-annotation-audit-packaged-v1', evaluation_only=True,
                  source_camera=camera_ids[source_index], original_start=original_start, frames=count,
                  annotated_side=side, annotated_frames=int(valid.sum()), reference_joints_reproduced_max_mm=float(errors.max()*1000),
                  reference_tip_vertices=reference_tip_vertices, original_camera_relative_agreement_max=relative_errors,
                  original_world_to_legacy_world=world_changes[source_index].tolist(), annotation_sha256=hashes,
                  reference_sha256=file_sha256(reference), verification_sha256=file_sha256(verification),
                  model_sha256=file_sha256(Path(mano_models) / f'MANO_{side.upper()}.npz'),
                  code_sha256=file_sha256(__file__), metric_code_sha256=file_sha256(Path(__file__).with_name('hand_annotation_metrics.py')),
                  variants=rows, no_refitting=True, no_scale_correction=True, no_rotation_alignment=True,
                  units='metres before reporting millimetres; no additional model or label scaling',
                  limits=['Annotations are fitted MANO models, not exact human anatomy.',
                          'Legacy frame opt-in requires an independently audited fixed identity base; identity cameras alone do not prove this.',
                          'Common vertex definitions on opposite-chirality MANO meshes are diagnostics, not corrected handedness.',
                          'All joint definitions reported; no convention selected for the best score.'])
    assert_stable(stable)
    report['verified_input_hashes'] = stable
    write_report(output, report)
    return report


def write_report(output, report):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    for name in ['annotations', 'verification', 'reference', 'mano-models', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--source-index', type=int, required=True)
    parser.add_argument('--camera-ids', nargs='+', required=True)
    parser.add_argument('--original-start', type=int, required=True)
    parser.add_argument('--side', choices=['left', 'right'], required=True)
    parser.add_argument('--reference-tip-vertices', nargs=5, type=int, required=True)
    parser.add_argument('--allow-legacy-frame', action='store_true')
    audit_hands(**vars(parser.parse_args()))
