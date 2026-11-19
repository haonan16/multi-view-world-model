"""Recorded RGB, fitted hand/object overlay, and an estimated 3D scene view.

Object poses between fitted keyframes are explicitly labeled interpolated.
The table's drawn footprint is illustrative; its plane is the learned estimate.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1] / 'mano')]
from artirecon.input_protocol import file_sha256, load_manifest
from mano_recon.run_integrity import artifact_record
from source_mesh_track import verify_hand_stage


def preview(track, hand_run, case='bottle'):
    import cv2
    import imageio.v2 as imageio
    import trimesh
    from scipy.spatial.transform import Rotation, Slerp
    from mano_recon.model import MANOHand
    from mano_recon.render_fast import render
    track, hand_run = Path(track).resolve(), Path(hand_run).resolve()
    case_path, hand_path = hand_run / 'inputs' / case, hand_run / 'hand-results' / case
    verify_hand_stage(case_path, hand_path)
    load_manifest(track / 'input-manifest.json', [('object_geometry', track / 'scene-sequence.json'),
                                                ('object_geometry', track / 'fitted-mesh.ply')])
    audit = json.loads((track / 'execution.json').read_text())
    if audit['status'] != 'complete' or artifact_record(case_path / 'frozen-hand.npz') != audit['frozen_hand']:
        raise ValueError('Preview hand does not match the completed mesh fit')
    for name, saved in audit['outputs'].items():
        if artifact_record(track / name) != saved:
            raise ValueError('Mesh track output changed: ' + name)
    scene = json.loads((track / 'scene-sequence.json').read_text())
    hand_scene = json.loads((hand_path / 'scene-sequence.json').read_text())
    with np.load(case_path / 'obs.npz', allow_pickle=False) as z:
        rgb, k, times = [z[n].copy() for n in ['rgb', 'K', 'timestamps']]
    with np.load(track / 'mesh-track.npz', allow_pickle=False) as z:
        poses, ids, fitted_times, observed = [z[n].copy() for n in ['poses', 'frame_ids', 'timestamps', 'observed']]
    if not np.array_equal(times[ids], fitted_times):
        raise ValueError('Native source timestamps changed')
    with np.load(hand_path / 'fit.npz', allow_pickle=False) as z:
        hand_vertices = z['vertices'].copy()
    hand_visible = np.asarray(hand_scene['actors'][0]['observed'], bool)
    model_path = Path(hand_scene['actors'][0]['model_uri'])
    side = json.loads((case_path / 'protocol.json').read_text())['mano_side']
    hand = MANOHand(model_path.parent, side=side, components=12, device='cpu')
    hand_faces = hand.faces.numpy()
    mesh = trimesh.load(track / 'fitted-mesh.ply', force='mesh', process=False)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    rotations = Slerp(fitted_times, Rotation.from_matrix(poses[:, :3, :3]))(times).as_matrix()
    translations = np.column_stack([np.interp(times, fitted_times, poses[:, axis, 3]) for axis in range(3)])
    bracket = np.clip(np.searchsorted(fitted_times, times, side='right') - 1, 0, len(ids)-1)
    next_bracket = np.minimum(bracket+1, len(ids)-1)
    object_visible = observed[bracket] & observed[next_bracket]
    for i, source_id in enumerate(ids):
        object_visible[source_id] = observed[i]
    table = json.loads((case_path / 'protocol.json').read_text())['table']
    n, d = np.asarray(table['normal']), float(table['offset'])
    centre = np.median(translations[object_visible], axis=0)
    n /= np.linalg.norm(n)
    e1 = np.cross(n, [1., 0, 0])
    if np.linalg.norm(e1) < .1:
        e1 = np.cross(n, [0., 1, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    support_centre = centre - (centre @ n + d) * n
    table_vertices = support_centre + np.asarray([-.5*e1-.5*e2, .5*e1-.5*e2, .5*e1+.5*e2, -.5*e1+.5*e2])
    table_faces = np.asarray([[0, 1, 2], [0, 2, 3]])
    camera = np.eye(4)
    source_direction = -centre / np.linalg.norm(centre)
    source_direction = Rotation.from_rotvec(n*.5).apply(source_direction)
    camera[:3, 3] = centre + .65*source_direction + .2*n
    forward = centre-camera[:3, 3]
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, n)
    right /= np.linalg.norm(right)
    camera[:3, :3] = np.column_stack([right, np.cross(forward, right), forward])
    h, w = rgb.shape[2:4]
    virtual_k = np.asarray([[.9*w, 0, w/2], [0, .9*w, h/2], [0, 0, 1.]])
    media = track / 'media'
    media.mkdir(exist_ok=True)
    output = media / (case+'-rgb-overlay-3d.mp4')
    still_ids = set(np.rint(np.linspace(0, len(times)-1, 4)).astype(int))
    stills = []
    def raster(parts, intrinsics, pose, base=None):
        image = np.full((h, w, 3), 30, np.uint8) if base is None else base.copy()
        zbuffer = np.full((h, w), np.inf)
        for vs, fs, color in parts:
            colored, depth = render(vs, fs, intrinsics, pose, size=(h, w), color=color)
            use = (depth > 0) & (depth < zbuffer)
            zbuffer[use] = depth[use]
            image[use] = colored[use] if base is None else (.35*base[use]+.65*colored[use]).astype(np.uint8)
        return image
    fps = float(1/np.median(np.diff(times)))
    with imageio.get_writer(output, fps=fps, codec='libx264', quality=8, macro_block_size=1) as writer:
        for frame, source in enumerate(rgb[:, 0]):
            parts = []
            if object_visible[frame]:
                parts.append((vertices @ rotations[frame].T+translations[frame], faces, (235, 165, 70)))
            if hand_visible[frame]:
                parts.append((hand_vertices[frame], hand_faces, (80, 220, 150)))
            overlay = raster(parts, k[0], np.eye(4), source)
            view = raster([(table_vertices, table_faces, (170, 160, 140))]+parts, virtual_k, camera)
            panels = [source.copy(), overlay, view]
            key = ('object association missing' if not object_visible[frame] else
                   'fitted keyframe' if frame in ids else 'object pose interpolated')
            for title, panel in zip(['Recorded RGB', key, 'Estimated 3D; illustrative table extent'], panels):
                panel[:35] = 20
                cv2.putText(panel, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (250, 250, 250), 1, cv2.LINE_AA)
            row = np.concatenate(panels, axis=1)
            writer.append_data(row)
            if frame in still_ids:
                stills.append(row)
    imageio.imwrite(media/(case+'-rgb-overlay-3d.png'), np.concatenate(stills, axis=0))
    report = dict(video=str(output), fps=fps, source_frames=len(times), fitted_native_frame_ids=ids.tolist(),
        geometry='Actual fitted SAM3D topology, no box/cylinder substitute',
        intermediate_frames='SLERP rotation and linear translation between fitted keyframes; labeled interpolated',
        table='Estimated plane; one-metre drawn footprint is a display choice, not recovered table dimensions',
        measured_depth_used=False, GT_geometry_used=False, simulator_used=False,
        producer_sha256=file_sha256(__file__), inputs={p.name:file_sha256(p) for p in [track/'mesh-track.npz', track/'fitted-mesh.ply', hand_path/'fit.npz']})
    (media/'preview.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track', type=Path, required=True)
    p.add_argument('--hand-run', type=Path, required=True)
    p.add_argument('--case', default='bottle')
    a = p.parse_args()
    preview(a.track, a.hand_run, a.case)
