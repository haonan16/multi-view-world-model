"""Track a fresh RGB SAM3D mesh using Amani's mask/depth pose objective.

The hand fit is a separately authenticated derived input. This adapter does not
pretend that a merged downstream manifest equals the upstream hand manifest.
Single-view contact is unknown: no rigid grasp or assumed-rest attraction is
used, and no executable plan is produced. Geometry keeps the SAM3D topology.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1] / 'mano')]
from artirecon.input_protocol import file_sha256, load_manifest, validate_manifest
from artirecon.scene_sequence import load_scene, object_from_arrays, scene_from_arrays
from mano_recon.run_integrity import artifact_record, verify_cached_outputs
from rgb_mano_initialize import merge_manifests


def equal_observations(hand, geometry):
    """The frozen hand must inhabit these exact source pixels and metric gauge."""
    for name in ['rgb', 'depth_m', 'K', 'c2w', 'timestamps']:
        if name not in hand or name not in geometry or not np.array_equal(hand[name], geometry[name]):
            raise ValueError('Hand and object observations differ: ' + name)
    if hand['rgb'].shape[1] != 1 or hand['c2w'].shape != (1, 4, 4) or not np.allclose(hand['c2w'][0], np.eye(4)):
        raise ValueError('This adapter requires one observed source-camera gauge')
    times = hand['timestamps']
    if times.ndim != 1 or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('Explicit finite increasing source timestamps are required')


def verify_hand_stage(case, results):
    case, results = Path(case), Path(results)
    manifest = case / 'input-manifest.json'
    report = load_manifest(manifest, [('depth', case / 'obs.npz'), ('cameras', case / 'obs.npz'),
                                      ('support', case / 'protocol.json')])
    audit = json.loads((results / 'execution.json').read_text())
    if audit.get('status') != 'complete' or audit.get('input_protocol', {}).get('manifest_sha256') != report['manifest_sha256']:
        raise ValueError('Hand stage is incomplete or not bound to its original manifest')
    verify_cached_outputs(audit, {'fit': results / 'fit.npz', 'frozen_hand': case / 'frozen-hand.npz',
                                  'scene_sequence': results / 'scene-sequence.json'})
    return report


def sample_frames(count, budget):
    if count < 2 or budget < 2:
        raise ValueError('At least two source frames and fitting keyframes are required')
    return np.unique(np.rint(np.linspace(0, count - 1, min(count, budget))).astype(int))


def frame_observation_flags(selected):
    return np.asarray([bool(instances) for instances in selected], dtype=bool)


def global_candidate_row(selection):
    index, rows = selection.get('selected_index'), selection.get('candidate_row_ids', [])
    if not isinstance(index, int) or not 0 <= index < len(rows):
        raise ValueError('Selected SAM3D candidate lacks an explicit global row mapping')
    return int(rows[index])


def initial_shape_evidence(selected):
    """Unknown grasp/rest must never combine different poses into one shape fit."""
    if not selected or not selected[0]:
        raise ValueError('First-frame object evidence is required for shape initialization')
    return [selected[0]] + [{} for _ in selected[1:]]


def run(*, geometry, masks, asset, hand_run, case, output, keyframes=13, shape_iters=250, pose_iters=150, samples=1500):
    import torch
    import trimesh
    from PIL import Image
    from mano_recon.amani_objects.objects_v2 import ObjectsV2, Sam3Masks, mesh_samples

    start = time.monotonic()
    geometry, masks, asset, hand_run, output = map(lambda p: Path(p).resolve(), [geometry, masks, asset, hand_run, output])
    hand_case, hand_results = hand_run / 'inputs' / case, hand_run / 'hand-results' / case
    if output.exists():
        raise FileExistsError(output)
    hand_report = verify_hand_stage(hand_case, hand_results)
    load_manifest(geometry / 'input-manifest.json', [('depth', geometry / 'geometry.npz'), ('cameras', geometry / 'geometry.npz')])
    load_manifest(masks / 'input-manifest.json', [('masks', masks / 'masks.npz')])
    load_manifest(asset / 'input-manifest.json', [('object_geometry', asset / 'object.ply'),
                                               ('object_geometry', asset / 'object.json'), ('masks', asset / 'mask.png')])
    manifest = merge_manifests([hand_case / 'input-manifest.json', geometry / 'input-manifest.json',
                                masks / 'input-manifest.json', asset / 'input-manifest.json'])
    roots = [a for a in manifest['artifacts'] if a['origin'] == 'observed_rgb']
    if len(roots) != 1:
        raise ValueError('Hand, mesh and masks must descend from the identical source RGB')
    with np.load(hand_case / 'obs.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    with np.load(geometry / 'geometry.npz', allow_pickle=False) as z:
        equal_observations(data, z)
    if not np.array_equal(data['rgb'][0, 0], np.asarray(Image.open(asset / 'source.png').convert('RGB'))):
        raise ValueError('SAM3D source image differs from the source RGB')
    hand_scene = load_scene(hand_results / 'scene-sequence.json')
    if hand_scene['coordinate_frame']['name'] != 'source_camera' or not np.array_equal(hand_scene['timestamps'], data['timestamps']):
        raise ValueError('Hand scene frame/timing is not the source-camera sequence')
    protocol = json.loads((hand_case / 'protocol.json').read_text())
    selection = json.loads((asset / 'selection.json').read_text())
    if selection['frame_index'] != 0 or selection['source_rgb_sha256'] != roots[0]['sha256']:
        raise ValueError('SAM3D selection does not reference this first source RGB')
    with np.load(masks / 'masks.npz', allow_pickle=False) as z:
        mask_data = {k: z[k].copy() for k in z.files}
    if not np.array_equal(mask_data['timestamps'], data['timestamps']):
        raise ValueError('SAM3 source timing differs')
    H, W = data['depth_m'].shape[-2:]
    if tuple(mask_data['shape']) != (H, W):
        raise ValueError('SAM3 native image grid differs')
    mask = np.asarray(Image.open(asset / 'mask.png').convert('L')) > 0
    selected_row = global_candidate_row(selection)
    source_mask = np.unpackbits(mask_data['mask'][selected_row])[:H * W].reshape(H, W).astype(bool)
    if mask_data['t'][selected_row] != 0 or not np.array_equal(mask, source_mask):
        raise ValueError('SAM3D selected mask differs from source SAM3 candidate')
    pixels = np.argwhere(mask)[:, ::-1]
    hint_uv = pixels[np.argmin(np.linalg.norm(pixels - np.median(pixels, axis=0), axis=1))].tolist()
    ids = sample_frames(len(data['timestamps']), keyframes)
    remap = {int(t): i for i, t in enumerate(ids)}
    keep = np.isin(mask_data['t'], ids)
    selected_masks = {k: mask_data[k][keep] for k in ['p', 't', 'score', 'box', 'mask']}
    selected_masks.update(t=np.asarray([remap[int(t)] for t in selected_masks['t']]), v=np.zeros(keep.sum(), int),
                          prompts=mask_data['prompts'], shape=mask_data['shape'])
    hand_hash = artifact_record(hand_case / 'frozen-hand.npz')
    original_hand_ids = json.loads((hand_case / 'input-manifest.json').read_text())['method_inputs']
    manifest['artifacts'].append(dict(id='frozen_rgb_hand', kind='hand_initialization', origin='model_prediction',
        parent_ids=original_hand_ids, **hand_hash))
    method_ids = [a['id'] for a in manifest['artifacts'] if a['kind'] in {'depth', 'cameras', 'masks', 'object_geometry', 'support'}]
    manifest['method_inputs'] = list(dict.fromkeys(method_ids + ['frozen_rgb_hand']))
    manifest['metric_scale'] = dict(status='estimated', artifact_ids=['predicted_geometry'], description='Learned source UniDepth metric scale, not measured depth')
    manifest['robot_registration'] = dict(status='unresolved', artifact_ids=[], description='No target-robot registration or plan produced')
    validation = validate_manifest(manifest, output, verify_files=True)
    output.mkdir(parents=True)
    (output / 'input-manifest.json').write_text(json.dumps(manifest, indent=2))
    np.savez_compressed(output / 'sam3-keyframes.npz', **selected_masks)
    with np.load(hand_case / 'frozen-hand.npz', allow_pickle=False) as z:
        frozen = {k: z[k][ids].copy() for k in z.files}
    class Hand:
        def surface(self, q, base):
            return torch.as_tensor(frozen['points'], device=q.device), torch.as_tensor(frozen['normals'], device=q.device)
    obs = SimpleNamespace(depth=data['depth_m'][ids], K=data['K'], c2w=data['c2w'], T=len(ids), V=1)
    table = protocol['table']
    spec = dict(name=case, prompt=selection['object_prompt'], role='moved', shape='mesh',
                mesh=str(asset / 'object.ply'), mesh_json=str(asset / 'object.json'),
                hint=dict(t=0, v=0, uv=hint_uv), gate=.05)
    logs = []
    def log(message):
        logs.append(message)
        print(message, flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    fitter = ObjectsV2(obs, dict(n=table['normal'], d=table['offset']), Hand(), frozen,
                       Sam3Masks(output / 'sam3-keyframes.npz', H, W), dict(objects=[spec]), log=log)
    selected, centres, held = fitter.track(spec)
    if held.any():
        raise RuntimeError('Single-view contact unexpectedly asserted by author tracker')
    observed = frame_observation_flags(selected)
    if not observed[0] or observed.sum() < 2:
        raise RuntimeError('Insufficient observed source object masks to fit a sequence')
    # Author shape() calls a frame "rest" from its centre/held heuristic. With
    # unknown contact that label is insufficient to establish a shared pose.
    T0, half, mesh, rest = fitter.shape(spec, initial_shape_evidence(selected), centres, held)
    if rest != [0]:
        raise RuntimeError('Moving frames entered the first-frame-only shape fit')
    source_mesh = trimesh.load(asset / 'object.ply', force='mesh')
    if not np.array_equal(mesh.faces, source_mesh.faces) or len(mesh.vertices) != len(source_mesh.vertices):
        raise RuntimeError('SAM3D topology changed; primitive replacement is forbidden')
    Ps, Nm = mesh_samples(mesh, samples)
    Ps, Nm = Ps.astype(np.float32), Nm.astype(np.float32)
    poses = np.tile(np.eye(4), (len(ids), 1, 1))
    losses = np.zeros(len(ids))
    old_rest = os.environ.get('REST_W')
    # Unknown grasp does not imply the object is resting on a table.
    os.environ['REST_W'] = '0'
    try:
        poses[0], _ = fitter.fit_pose(0, [T0], Ps, Nm, selected[0], False,
                                      iters=shape_iters, opt_scale=True, extra=())
        scale = fitter.last_scale.copy()
        mesh.vertices = np.asarray(mesh.vertices) * scale
        Ps *= scale
        # Recompute surface normals after per-axis geometry refinement.
        Ps, Nm = mesh_samples(mesh, samples)
        Ps, Nm = Ps.astype(np.float32), Nm.astype(np.float32)
        poses[0], losses[0] = fitter.fit_pose(0, [poses[0]], Ps, Nm, selected[0], False, iters=pose_iters)
        for t in range(1, len(ids)):
            if not observed[t]:
                poses[t] = poses[t - 1]
                losses[t] = np.nan
                continue
            initial = [poses[t - 1]]
            points = [c['P'] for c in selected[t].values() if len(c['P'])]
            if points:
                shifted = poses[t - 1].copy()
                shifted[:3, 3] += np.median(np.concatenate(points), axis=0) - (poses[t - 1][:3, :3] @ Ps.mean(0) + poses[t - 1][:3, 3])
                initial.append(shifted)
            poses[t], losses[t] = fitter.fit_pose(t, initial, Ps, Nm, selected[t], False, iters=pose_iters)
            log(f'Native frame {ids[t]}: observed source-mask fit; loss={losses[t]:.5f}')
    finally:
        if old_rest is None:
            os.environ.pop('REST_W', None)
        else:
            os.environ['REST_W'] = old_rest
    mesh.export(output / 'fitted-mesh.ply')
    np.savez_compressed(output / 'mesh-track.npz', poses=poses, frame_ids=ids, timestamps=data['timestamps'][ids],
                        observed=observed, loss=losses, centres=centres, grasp_known=np.zeros(len(ids), bool),
                        shared_mesh_scale=scale)
    actors = copy.deepcopy(hand_scene['actors'])
    for actor in actors:
        for key in ['q', 'T_world_base', 'observed']:
            actor[key] = np.asarray(actor[key])[ids].tolist()
    provenance = dict(method='Fresh RGB SAM3D topology + Amani ObjectsV2 source-mask/depth pose fitting',
                      artifact_ids=manifest['method_inputs'], confidence_status='uncalibrated')
    scene = scene_from_arrays(timestamps=data['timestamps'][ids], coordinate_frame=hand_scene['coordinate_frame'],
        metric_scale=hand_scene['metric_scale'], actors=actors,
        objects=[object_from_arrays(object_id=case, poses=poses, observed=observed, provenance=provenance,
                                    mesh_uri=str(output / 'fitted-mesh.ply'))], support=hand_scene['support'], contacts=[],
        uncertainty=dict(status='estimated', provenance=provenance,
                         interpretation='Single-view grasp/contact unknown; no assumed-rest attraction or rigid-grasp constraints'),
        task=dict(description='Track the RGB-derived pump-bottle mesh', target=None))
    (output / 'scene-sequence.json').write_text(json.dumps(scene, indent=2, allow_nan=False))
    if artifact_record(hand_case / 'frozen-hand.npz') != hand_hash:
        raise ValueError('Frozen hand changed during fitting')
    validate_manifest(manifest, output, verify_files=True)
    consumed_ids = list(manifest['method_inputs'])
    for aid, name, parents in [('fitted_source_mesh', 'fitted-mesh.ply', consumed_ids),
                               ('source_mesh_track', 'mesh-track.npz', consumed_ids + ['fitted_source_mesh']),
                               ('source_mesh_scene', 'scene-sequence.json', consumed_ids + ['fitted_source_mesh', 'source_mesh_track'])]:
        manifest['artifacts'].append(dict(id=aid, kind='object_geometry', origin='model_prediction',
            parent_ids=parents, path=str(output / name), sha256=file_sha256(output / name)))
    manifest['method_inputs'] = ['source_mesh_scene']
    output_validation = validate_manifest(manifest, output, verify_files=True)
    (output / 'input-manifest.json').write_text(json.dumps(manifest, indent=2))
    audit = dict(status='complete', seconds=time.monotonic()-start, input_protocol=validation,
        original_hand_manifest_sha256=hand_report['manifest_sha256'], frozen_hand=hand_hash,
        output_protocol=output_validation,
        source_frame_ids=ids.tolist(), native_source_frames=len(data['timestamps']), observed_keyframes=int(observed.sum()),
        frame_count=len(ids), actual_camera_count=1, primitive_fallback=False,
        shape_evidence_native_frames=[0], shared_pose_extra_frames=[],
        selected_global_mask_row=selected_row,
        retained_sam3d_vertices=len(mesh.vertices), retained_sam3d_faces=len(mesh.faces),
        mesh_scale=scale.tolist(), shape_iterations=shape_iters, pose_iterations=pose_iters, samples=samples,
        grasp_status='unknown: author attachment requires two observed views', resting_attraction_used=False,
        initialization_support_prior='Author mesh initialization rests at learned first-frame support; later poses have no rest attraction',
        planning_ready=False, blocking_reasons=['Single-view grasp/contact unresolved', 'Robot registration unresolved'],
        GT_used=False, simulator_used=False, producer_sha256=file_sha256(__file__),
        completion_meaning='Source RGB mesh track fitted; not contact validation or executable robot plan',
        outputs={n:artifact_record(output / n) for n in ['fitted-mesh.ply', 'mesh-track.npz', 'scene-sequence.json']})
    (output / 'execution.json').write_text(json.dumps(audit, indent=2, allow_nan=False))
    (output / 'fit.log').write_text('\n'.join(logs))
    print(json.dumps(audit), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['geometry', 'masks', 'asset', 'hand-run', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', default='bottle')
    parser.add_argument('--keyframes', type=int, default=13)
    parser.add_argument('--shape-iters', type=int, default=250)
    parser.add_argument('--pose-iters', type=int, default=150)
    parser.add_argument('--samples', type=int, default=1500)
    run(**vars(parser.parse_args()))
