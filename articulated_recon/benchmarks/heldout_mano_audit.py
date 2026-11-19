"""Render frozen source-camera MANO geometry in held-out recorded cameras.

Published calibration is evaluator-only. There is no pose/scale correction,
alignment fit, hidden-depth occlusion, or optimization in this module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).parents[1]), str(Path(__file__).parents[1] / 'mano')]
from artirecon.input_protocol import file_sha256, load_manifest
from mano_recon.run_integrity import verify_cached_outputs, artifact_record
from depth_confidence_audit import verify_exact_frames


def source_to_view(source_c2w, view_c2w):
    """Published cameras have T_world_camera convention, in the same world."""
    for matrix in [source_c2w, view_c2w]:
        matrix = np.asarray(matrix)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-4)):
            raise ValueError('Expected a finite rigid T_world_camera')
    return np.linalg.inv(view_c2w) @ source_c2w


def transform_points(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def interpolate_track(poses, frame_ids, fitted_times, observed, native_times):
    """Visualization only: no extrapolation or interpolation through missing states."""
    from scipy.spatial.transform import Rotation, Slerp
    poses, frame_ids, observed = np.asarray(poses), np.asarray(frame_ids), np.asarray(observed)
    fitted_times, native_times = np.asarray(fitted_times), np.asarray(native_times)
    if (len(frame_ids) < 2 or poses.shape != (len(frame_ids), 4, 4) or observed.shape != frame_ids.shape
            or observed.dtype != bool or frame_ids.dtype.kind not in 'iu'
            or np.any(frame_ids < 0) or np.any(frame_ids >= len(native_times))
            or not np.array_equal(fitted_times, native_times[frame_ids])
            or frame_ids[0] != 0 or frame_ids[-1] != len(native_times)-1
            or not np.all(np.diff(fitted_times) > 0)):
        raise ValueError('Object keyframe/timestamp/validity contract mismatch')
    # Rotation.from_matrix otherwise silently projects scaled/sheared matrices
    # onto SO(3), which would alter the frozen reconstruction during evaluation.
    for pose in poses:
        source_to_view(np.eye(4), pose)
    result = np.repeat(np.eye(4)[None], len(native_times), axis=0)
    result[:, :3, :3] = Slerp(fitted_times, Rotation.from_matrix(poses[:, :3, :3]))(native_times).as_matrix()
    result[:, :3, 3] = np.column_stack([np.interp(native_times, fitted_times, poses[:, a, 3]) for a in range(3)])
    left = np.clip(np.searchsorted(fitted_times, native_times, side='right')-1, 0, len(frame_ids)-1)
    right = np.minimum(left+1, len(frame_ids)-1)
    visible = observed[left] & observed[right]
    visible[frame_ids] = observed
    return result, visible


def run(*, run_dir, case, reference, source_index, camera_ids, output, mesh_track=None):
    import cv2
    import imageio.v2 as imageio
    from mano_recon.render_fast import render

    run_dir, reference, output = map(lambda p: Path(p).resolve(), (run_dir, reference, output))
    inp, fit = run_dir / 'inputs' / case, run_dir / 'hand-results' / case
    execution = json.loads((fit / 'execution.json').read_text())
    protocol = json.loads((inp / 'protocol.json').read_text())
    scene = json.loads((fit / 'scene-sequence.json').read_text())
    if execution['status'] != 'complete' or execution['input_protocol']['condition'] != 'rgb_only':
        raise ValueError('Require a completed RGB-only fit before accessing held-out inputs')
    if execution['depth_self_calibration'] or execution['heldout_used']:
        raise ValueError('This audit requires the uncalibrated, source-only fit')
    validated = load_manifest(inp / 'input-manifest.json', [('depth', inp / 'obs.npz'),
        ('cameras', inp / 'obs.npz'), ('hand_initialization', inp / 'hand-input.npz')])
    if validated['manifest_sha256'] != execution['input_protocol']['manifest_sha256']:
        raise ValueError('Input manifest differs from the completed fit')
    if protocol['coordinate_frame']['name'] != 'source_camera' or scene['coordinate_frame']['name'] != 'source_camera':
        raise ValueError('Frozen vertices must explicitly be in the observed source camera frame')
    verify_cached_outputs(execution, {'fit': fit / 'fit.npz', 'scene_sequence': fit / 'scene-sequence.json',
                                      'frozen_hand': inp / 'frozen-hand.npz'})
    if file_sha256(fit / 'scene-sequence.json') != execution['shared_scene_sha256']:
        raise ValueError('Shared scene hash differs')
    with np.load(inp / 'obs.npz') as z:
        rgb, times, predicted_k, predicted_c = [z[k].copy() for k in ['rgb', 'timestamps', 'K', 'c2w']]
    if rgb.shape[1] != 1 or not np.allclose(predicted_c, np.eye(4)[None], atol=1e-8):
        raise ValueError('Require the single observed source-camera frame')
    with np.load(fit / 'fit.npz') as z:
        vertices = z['vertices'].copy()
    with np.load(fit / 'hand-sequence.npz') as z:
        if not np.array_equal(z['vertices'], vertices) or not np.array_equal(z['timestamps'], times):
            raise ValueError('Mesh/times differ from the frozen fit')
        faces, visible = z['faces'].copy(), z['visible'].copy()
    hashes = {str(p): file_sha256(p) for p in [fit / 'execution.json', fit / 'fit.npz',
              fit / 'hand-sequence.npz', inp / 'obs.npz', inp / 'frozen-hand.npz', fit / 'scene-sequence.json', reference]}
    object_data = None
    if mesh_track is not None:
        import trimesh
        track = Path(mesh_track).resolve()
        report = load_manifest(track / 'input-manifest.json', [('object_geometry', track / 'scene-sequence.json'),
                                                              ('object_geometry', track / 'fitted-mesh.ply')])
        audit = json.loads((track / 'execution.json').read_text())
        if (report['condition'] != 'rgb_only' or audit['status'] != 'complete'
                or audit['output_protocol']['manifest_sha256'] != report['manifest_sha256']
                or audit['original_hand_manifest_sha256'] != validated['manifest_sha256']
                or artifact_record(inp / 'frozen-hand.npz') != audit['frozen_hand']):
            raise ValueError('Object track must bind the same frozen RGB-only hand')
        if not {'mesh-track.npz', 'fitted-mesh.ply', 'scene-sequence.json'} <= set(audit['outputs']):
            raise ValueError('Object receipt omits required frozen outputs')
        for name, saved in audit['outputs'].items():
            if artifact_record(track / name) != saved:
                raise ValueError('Object output changed: '+name)
            hashes[str(track / name)] = file_sha256(track / name)
        object_scene = json.loads((track / 'scene-sequence.json').read_text())
        if object_scene['coordinate_frame']['name'] != 'source_camera':
            raise ValueError('Object geometry must share the frozen source-camera frame')
        with np.load(track / 'mesh-track.npz') as z:
            object_poses, object_visible = interpolate_track(z['poses'], z['frame_ids'], z['timestamps'], z['observed'], times)
            frame_ids = z['frame_ids'].tolist()
        mesh = trimesh.load(track / 'fitted-mesh.ply', force='mesh', process=False)
        object_data = (np.asarray(mesh.vertices), np.asarray(mesh.faces), object_poses, object_visible, frame_ids)
    # Evaluation data first opened after the fit/coordinate contract is checked.
    with np.load(reference) as z:
        observed, ref_times, k, c = [z[name].copy() for name in ['rgb', 'timestamps', 'K', 'c2w']]
    if len(camera_ids) != observed.shape[1] or not 0 <= source_index < len(camera_ids):
        raise ValueError('Camera identity metadata differs from the reference bundle')
    verify_exact_frames(rgb[:, 0], times, observed[:, source_index], ref_times)
    if len(vertices) != len(times) or visible.shape != times.shape:
        raise ValueError('Frozen hand frame count differs')
    transforms = [source_to_view(c[source_index], view) for view in c]
    order = [source_index] + [i for i in range(len(camera_ids)) if i != source_index]
    output.mkdir(parents=True, exist_ok=False)
    fps = 1 / float(np.median(np.diff(times)))
    h, w = observed.shape[2:4]
    video = output / 'heldout-mano.mp4'
    visible_counts = {camera_ids[v]: [] for v in order}
    with imageio.get_writer(video, fps=fps, codec='libx264', quality=8, macro_block_size=1) as writer:
        for frame in range(len(times)):
            panels = []
            for v in order:
                panel = observed[frame, v].copy()
                count = 0
                rendered = panel.copy()
                zbuffer = np.full((h, w), np.inf)
                if visible[frame]:
                    mesh, depth = render(transform_points(vertices[frame], transforms[v]), faces,
                                         k[v], np.eye(4), size=(h, w), color=(65, 225, 115))
                    selected = depth > 0
                    count = int(selected.sum())
                    rendered[selected] = mesh[selected]
                    zbuffer[selected] = depth[selected]
                if object_data is not None and object_data[3][frame]:
                    points = transform_points(object_data[0], object_data[2][frame])
                    mesh, depth = render(transform_points(points, transforms[v]), object_data[1], k[v], np.eye(4),
                                         size=(h, w), color=(240, 165, 55))
                    selected = (depth > 0) & (depth < zbuffer)
                    rendered[selected] = mesh[selected]
                    zbuffer[selected] = depth[selected]
                selected = np.isfinite(zbuffer)
                panel[selected] = (.45 * panel[selected] + .55 * rendered[selected]).astype(np.uint8)
                visible_counts[camera_ids[v]].append(count)
                role = 'Source RGB' if v == source_index else 'Held-out RGB'
                cv2.rectangle(panel, (0, 0), (w, 30), (20, 20, 20), -1)
                cv2.putText(panel, f'{role} | camera {camera_ids[v]}', (10, 22), 0, .55, (245, 245, 245), 1, cv2.LINE_AA)
                cv2.rectangle(panel, (0, h-28), (w, h), (20, 20, 20), -1)
                label = 'Frozen hand; published camera projection'
                if object_data is not None:
                    mode = 'fitted object keyframe' if frame in object_data[4] else 'interpolated object pose'
                    if not object_data[3][frame]:
                        mode = 'object association missing'
                    label = 'Frozen hand + '+mode
                cv2.putText(panel, f'{times[frame]:.2f}s | {label}', (10, h-8), 0, .49, (245, 245, 245), 1, cv2.LINE_AA)
                panels.append(panel)
            image = np.concatenate(panels, axis=1)
            writer.append_data(image)
            if frame in [0, len(times)//2, len(times)-1]:
                imageio.imwrite(output / f'frame-{frame:04d}.png', image)
    if any(file_sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError('Frozen fit or reference changed during the diagnostic')
    report = dict(schema='heldout-mano-projection-v1', evaluation_only=True,
                  input_condition='rgb_only', source_camera=camera_ids[source_index],
                  source_index=source_index, camera_order=[camera_ids[i] for i in order],
                  frames=len(times), fps=fps, observed_fit_frames=int(visible.sum()),
                  hashes=hashes, video_sha256=file_sha256(video), code_sha256=file_sha256(__file__),
                  T_view_source=[t.tolist() for t in transforms], reference_K=k.tolist(),
                  predicted_source_K=predicted_k.tolist(), projected_mesh_pixels=visible_counts,
                  pose_scale_alignment='NONE', refit_performed=False, hidden_depth_used=False,
                  source_projection='Published source K, same as held-out evaluation; no fit adjustment',
                  object_track=str(mesh_track) if mesh_track else None,
                  object_keyframes=object_data[4] if object_data else [],
                  object_interpolation='Linear translation + Slerp rotation between supported fitted keyframes, for display only' if object_data else None,
                  limitation='X-ray mesh overlay; published cameras only evaluate frozen geometry. No independent calibration correction. Views were never given to fitting.')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run-dir', 'reference', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--source-index', required=True, type=int)
    parser.add_argument('--camera-ids', required=True, nargs='+')
    parser.add_argument('--mesh-track', type=Path)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
