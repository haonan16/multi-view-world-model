"""Freeze source-only evidence for matched fresh-agent representation trials.

No evaluator, simulator, model inference, or agent is launched here. The compact
initializer is a deliberately untrained, deterministic representation adapter;
it is not an agent result or a recovered watertight object. Missing observations
remain unknown, including pixels hidden by a predicted hand mask.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from artirecon.input_protocol import file_sha256, load_manifest
from artirecon.scene_sequence import object_from_arrays, scene_from_arrays, validate_scene


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def select_frames(count, budget):
    if type(budget) is not int or budget < 2 or count < 2:
        raise ValueError('At least two selected source frames are required')
    return np.unique(np.rint(np.linspace(0, count - 1, min(count, budget))).astype(int))


def cameras_per_frame(data, key, count, views, size, allow_nonfinite=False):
    value = np.asarray(data[key], dtype=float)
    if value.shape == (views, size, size):
        value = np.broadcast_to(value, (count, views, size, size)).copy()
    if value.shape != (count, views, size, size) or (not allow_nonfinite and not np.isfinite(value).all()):
        raise ValueError(f'{key} must be [V,{size},{size}] or [T,V,{size},{size}]')
    return value


def normalize_observations(data, task_mode='reconstruction'):
    """Normalize dynamic predicted cameras without changing their gauge/scale."""
    rgb = np.asarray(data['rgb'])
    depth = np.asarray(data['depth_m'], dtype=float)
    if rgb.ndim == 4:
        rgb = rgb[:, None]
    if depth.ndim == 3:
        depth = depth[:, None]
    if rgb.ndim != 5 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or depth.shape != rgb.shape[:-1]:
        raise ValueError('Expected uint8 RGB[T,V,H,W,3] and predicted depth[T,V,H,W]')
    count, views, height, width = depth.shape
    if views != 1 and not (task_mode=='forecast' and views==4):
        raise ValueError('This source-only comparison currently accepts exactly one observed view')
    times = np.asarray(data['timestamps'], dtype=float)
    if times.shape != (count,) or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('Explicit increasing timestamps are required')
    k = cameras_per_frame(data, 'K', count, views, 3, allow_nonfinite='points_world' in data)
    c = cameras_per_frame(data, 'c2w', count, views, 4)
    if not np.allclose(c[..., 3, :], [0, 0, 0, 1]) or not np.allclose(
            c[..., :3, :3].swapaxes(-1, -2) @ c[..., :3, :3], np.eye(3), atol=1e-4):
        raise ValueError('Predicted cameras must be rigid c2w transforms')
    if (np.linalg.det(c[..., :3, :3]) < .999).any():
        raise ValueError('Camera rotations/intrinsics must preserve orientation')
    k_valid = np.isfinite(k).all(axis=(-2,-1))
    k_valid &= np.linalg.det(np.where(k_valid[...,None,None], k, np.eye(3))) > 0
    if not k_valid.all() and 'points_world' not in data:
        raise ValueError('Unavailable pinhole cameras require native predicted points_world')
    valid = np.isfinite(depth) & (depth > 0)
    if 'depth_valid' in data:
        supplied = np.asarray(data['depth_valid'], bool).reshape(depth.shape)
        valid &= supplied
    depth = np.where(valid, depth, 0).astype(np.float32)
    masks = np.zeros(depth.shape, bool)
    known = np.zeros((count, views), bool)
    if 'object_masks' in data:
        masks = np.asarray(data['object_masks'], bool).reshape(depth.shape)
        if 'object_mask_known' not in data:
            raise ValueError('object_masks require explicit object_mask_known[T,V]')
        known = np.asarray(data['object_mask_known'], bool).reshape(count, views)
    occ = np.asarray(data.get('occlusion_mask', np.zeros(depth.shape, bool)), bool).reshape(depth.shape)
    # Camera-space depth is positive Z, not distance along a normalized ray.
    y, x = np.mgrid[:height, :width]
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1)
    rays = np.einsum('tvij,hwj->tvhwi', np.linalg.inv(np.where(k_valid[...,None,None],k,np.eye(3))), pixels)
    points = rays * depth[..., None]
    points = np.einsum('tvij,tvhwj->tvhwi', c[..., :3, :3], points) + c[..., None, None, :3, 3]
    if 'points_world' in data:
        points = np.asarray(data['points_world']).copy()
        if points.shape != depth.shape + (3,):
            raise ValueError('points_world must share the predicted depth pixel grid')
        valid &= np.isfinite(points).all(axis=-1)
    points[~valid] = 0
    result = dict(rgb=rgb.copy(), depth_m=depth, K=k, c2w=c, timestamps=times,
                depth_valid=valid, points_world=points, object_masks=masks,
                object_mask_known=known, occlusion_mask=occ, camera_projection_valid=k_valid)
    frame_ids = np.asarray(data.get('source_frame_indices', np.arange(count)))
    if task_mode=='forecast':
        if not np.array_equal(frame_ids,np.r_[0,np.full(count-1,-1)]):
            raise ValueError('Forecast source indices must be 0 then -1 for generated frames')
        known=np.zeros((count,views),bool);known[0,0]=True
        if not np.array_equal(data.get('observed_mask'),known):
            raise ValueError('Only the initial source image may be observed')
        result['observed_mask']=known
    elif (frame_ids.shape != (count,) or frame_ids.dtype.kind not in 'iu'
            or (frame_ids < 0).any() or not (np.diff(frame_ids) > 0).all()):
        raise ValueError('source_frame_indices must identify strictly increasing original source frames')
    result['source_frame_indices'] = frame_ids.copy()
    for key in ['local_points_m', 'confidence', 'K_pinhole_rmse_px']:
        if key in data:
            value = np.asarray(data[key])
            if value.shape[:2] != (count, views) or not np.issubdtype(value.dtype, np.number):
                raise ValueError(f'{key} must share explicit [T,V] dimensions')
            result[key] = value.copy()
    return result


def add_sam3_masks(data, mask_path, prompt):
    """Unique source candidates only; ambiguous association is explicitly unknown."""
    with np.load(mask_path, allow_pickle=False) as z:
        masks = {name: z[name].copy() for name in z.files}
    if not np.array_equal(masks['timestamps'], data['timestamps']):
        raise ValueError('Predicted masks do not share source timestamps')
    height, width = data['depth_m'].shape[-2:]
    if tuple(masks['shape']) != (height, width):
        raise ValueError('Predicted masks do not share source pixel grid')
    prompts = list(masks['prompts'])
    if prompt not in prompts:
        raise ValueError('Object prompt absent from predicted masks')
    p = prompts.index(prompt)
    for t in range(len(data['timestamps'])):
        rows = np.flatnonzero((masks['t'] == t) & (masks['p'] == p) & (masks['score'] >= .3))
        if len(rows) == 1:
            data['object_masks'][t, 0] = np.unpackbits(masks['mask'][rows[0]])[:height*width].reshape(height, width)
            data['object_mask_known'][t, 0] = True
        if 'hand' in prompts:
            hands = np.flatnonzero((masks['t'] == t) & (masks['p'] == prompts.index('hand')) & (masks['score'] >= .3))
            for row in hands:
                data['occlusion_mask'][t, 0] |= np.unpackbits(masks['mask'][row])[:height*width].reshape(height, width).astype(bool)
    return data


def compact_initializer(data, *, object_id, metric_status, coordinate_frame):
    """Shared visible-surface hull + translation track, with no grasp assumption.

    The hull uses one frame, never a union of moving clouds. Identity rotations
    are an unestimated initial value. It does not claim complete hidden extent.
    """
    from scipy.spatial import ConvexHull, QhullError
    clouds, centres, observed = [], [], []
    for t in range(len(data['timestamps'])):
        valid = data['depth_valid'][t] & data['object_masks'][t] & ~data['occlusion_mask'][t]
        valid &= data['object_mask_known'][t, :, None, None]
        points = data['points_world'][t][valid]
        if len(points) >= 12:
            # Fixed robust trimming, identical for every sequence and budget.
            lo, hi = np.quantile(points, [.02, .98], axis=0)
            points = points[((points >= lo) & (points <= hi)).all(axis=1)]
        supported = len(points) >= 12
        clouds.append(points)
        centres.append(np.median(points, axis=0) if supported else None)
        observed.append(supported)
    first = next((i for i, value in enumerate(observed) if value), None)
    if first is None:
        raise ValueError('Compact initializer requires supported predicted object masks/depth; missing is unknown')
    points = clouds[first]
    points = points[np.unique(np.rint(np.linspace(0, len(points)-1, min(512, len(points)))).astype(int))]
    canonical = points - centres[first]
    try:
        hull = ConvexHull(canonical)
    except QhullError as exc:
        raise ValueError('Visible points are degenerate; do not invent thickness for compact initializer') from exc
    used = np.unique(hull.simplices)
    remap = np.full(len(points), -1, int)
    remap[used] = np.arange(len(used))
    vertices, faces = canonical[used], remap[hull.simplices]
    poses = np.repeat(np.eye(4)[None], len(centres), axis=0)
    # A fallback pose is required by the scene schema but remains unobserved.
    previous = centres[first]
    for t, centre in enumerate(centres):
        if centre is not None:
            previous = centre
        poses[t, :3, 3] = previous
    provenance = dict(method='selected_predicted_mask_depth_visible_hull_centroid_initializer',
                      artifact_ids=['selected_observations'], confidence_status='unestimated',
                      shape_source_frame=first, rotation_status='unestimated_identity_initial_value',
                      hidden_surface_status='unknown', fallback_pose_status='unobserved_carry_forward')
    obj = object_from_arrays(object_id=object_id, poses=poses, observed=observed,
                             vertices_m=vertices, faces=faces, provenance=provenance)
    return scene_from_arrays(timestamps=data['timestamps'], coordinate_frame=coordinate_frame,
        metric_scale=dict(status=metric_status, provenance=dict(method='upstream_predicted_scale',
                          artifact_ids=['selected_observations'])), actors=[], objects=[obj], support=None,
        contacts=[dict(id=f'{object_id}_support_unknown', a=object_id, b='world', kind='support',
                       start_frame=0, end_frame=len(centres)-1, status='unknown', confidence=None,
                       provenance=dict(method='not_inferred', confidence_status='unknown'))],
        uncertainty=dict(status='unestimated'),
        task=dict(description='Recover minimal task-critical shared object geometry and motion from source RGB/model predictions', target=None))


def feature_targets(scene):
    """Modular model-derived targets; unknown rotation/contact is never a label."""
    validate_scene(scene)
    count, objects = len(scene['timestamps']), scene['objects']
    n = len(objects)
    values = dict(object_ids=np.asarray([o['id'] for o in objects], dtype=str),
        canonical_extent=np.zeros((n, 3), np.float32), shape_known=np.zeros(n, bool),
        pose_translation=np.zeros((count, n, 3), np.float32),
        pose_rotation_6d=np.zeros((count, n, 6), np.float32),
        observed=np.zeros((count, n), bool), rotation_known=np.zeros((count, n), bool),
        contact_known=np.zeros((count, n), bool), timestamps=np.asarray(scene['timestamps']))
    for i, obj in enumerate(objects):
        poses = np.asarray(obj['T_world_object'])
        values['pose_translation'][:, i] = poses[:, :3, 3]
        values['pose_rotation_6d'][:, i] = poses[:, :3, :2].swapaxes(-1, -2).reshape(count, 6)
        values['observed'][:, i] = obj['observed']
        geometry = obj['geometry']
        if geometry['type'] == 'mesh' and 'vertices_m' in geometry:
            v = np.asarray(geometry['vertices_m']) @ np.asarray(geometry.get('local_linear', np.eye(3))).T
            values['canonical_extent'][i] = np.ptp(v, axis=0)
        # Visible hull extent is available, but full shape supervision is unknown.
        values['shape_known'][i] = geometry['provenance'].get('shape_target_status') == 'supported_model_pseudotarget'
        values['rotation_known'][:, i] = values['observed'][:, i] & (
            obj['provenance'].get('rotation_status') == 'supported_model_pseudotarget')
        if obj['provenance'].get('rotation_status') == 'estimated_where_observed':
            gate = np.asarray(obj['provenance'].get('rotation_observed', np.zeros(count)), bool)
            if gate.shape != (count,):
                raise ValueError('rotation_observed must explicitly cover every timestamp')
            values['rotation_known'][:, i] = values['observed'][:, i] & gate
    return values


def selected_tracks(path, ids, times, coordinate_name, source_frame_indices=None):
    """Preserve native model correspondences, querying only selected source frames."""
    with np.load(path, allow_pickle=False) as z:
        values = {k:z[k].copy() for k in z.files}
    points = values['tracks3d']
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[0] != len(times):
        raise ValueError('tracks3d must cover the original [T,N,3] sequence')
    if not np.array_equal(values.get('timestamps'), times):
        raise ValueError('Tracks must have the exact source timestamps')
    if str(values.get('coordinate_frame_name','')) != coordinate_name:
        raise ValueError('Tracks must explicitly declare the same coordinate-frame gauge')
    query_frames = np.asarray(values.get('query_frame'))
    if query_frames.shape != (points.shape[1],) or query_frames.dtype.kind not in 'iu':
        raise ValueError('Tracks require integer query_frame[N]')
    native = np.asarray(source_frame_indices if source_frame_indices is not None else np.arange(len(times)))
    query_indexing = str(values.get('query_frame_indexing', 'original_source'))
    if query_indexing == 'geometry_rows':
        if (query_frames < 0).any() or (query_frames >= len(native)).any():
            raise ValueError('Query frame row outside original geometry sequence')
        query_frames = native[query_frames]
    elif query_indexing != 'original_source':
        raise ValueError('query_frame_indexing must be original_source or geometry_rows')
    columns = np.flatnonzero(np.isin(query_frames, native[ids]))
    if not len(columns):
        raise ValueError('No point queries originate in the selected input frames')
    result = dict(tracks3d=points[ids][:, columns], timestamps=times[ids],
        query_frame=query_frames[columns], query_column_ids=columns,
        coordinate_frame_name=np.asarray(coordinate_name), native_frame_ids=native[ids], geometry_row_ids=ids,
        query_frame_indexing=np.asarray('original_source'))
    for name in ['tracks2d','visible','trackvisibility','confidence']:
        if name in values:
            if values[name].shape[:2] != points.shape[:2]:
                raise ValueError('Track field must share [T,N]: ' + name)
            result[name] = values[name][ids][:, columns]
    for name in ['query_pixels','query_flat_indices']:
        if name in values:
            if values[name].shape[0] != points.shape[1]:
                raise ValueError('Track query metadata must share N: ' + name)
            result[name] = values[name][columns]
    return result


def default_config(model, python):
    return dict(schema_version=1, model=model, python=str(Path(python).absolute()), reasoning_effort='high',
        seed=20261119, time_limit_seconds=600, max_agent_turns=6, max_tool_calls=40,
        max_reported_tokens=24000, max_scene_checkpoints=30,
        timing_scope='fresh_agent_only; model frontend, selection and packet construction separate',
        token_limit_enforcement='reported completed-turn usage; in-flight requests may overshoot',
        repetitions=1, source_checks='predicted silhouette/depth agreement; no GT or simulator',
        minimum_valid_scene_objects=1)


def build_packet(*, observations, manifest, output, condition, budget, config, object_id='task_object',
                 masks=None, mask_manifest=None, object_prompt=None, coordinate_name='predicted_world',
                 asset=None, asset_manifest=None, initializer=None, initializer_receipt=None,
                 weak_hull_initializer=False, frontend_context_frame_ids=None, tracks=None, tracks_manifest=None):
    from PIL import Image
    observations, manifest, output = map(lambda x: Path(x).resolve(), (observations, manifest, output))
    if output.exists():
        raise FileExistsError(output)
    if condition not in {'A_raw', 'B_object'}:
        raise ValueError('condition must be A_raw or B_object')
    report = load_manifest(manifest, [('depth', observations), ('cameras', observations)])
    if report['condition'] != 'rgb_only' or report['task_mode'] != 'reconstruction':
        raise ValueError('Comparison accepts only authenticated source RGB reconstruction predictions')
    with np.load(observations, allow_pickle=False) as z:
        data = normalize_observations({k: z[k] for k in z.files})
        if frontend_context_frame_ids is None and 'source_frame_indices' in z.files:
            frontend_context_frame_ids = data['source_frame_indices'].tolist()
    parents = dict(observations_sha256=file_sha256(observations), manifest_sha256=file_sha256(manifest))
    if masks is not None:
        if not mask_manifest or not object_prompt:
            raise ValueError('External SAM3 masks require their manifest and object prompt')
        mask_report = load_manifest(mask_manifest, [('masks', masks)])
        if mask_report['condition'] != 'rgb_only':
            raise ValueError('Masks must be RGB-only predictions')
        source = json.loads(manifest.read_text())
        mask_source = json.loads(Path(mask_manifest).read_text())
        def rgb_roots(value, validation):
            return {a['sha256'] for a in value['artifacts'] if a['id'] in validation['method_artifact_ids']
                    and a['origin'] == 'observed_rgb'}
        if rgb_roots(source, report) != rgb_roots(mask_source, mask_report):
            raise ValueError('Mask and geometry source RGB roots differ')
        add_sam3_masks(data, masks, object_prompt)
        parents.update(masks_sha256=file_sha256(masks), mask_manifest_sha256=file_sha256(mask_manifest))
    ids = select_frames(len(data['timestamps']), budget)
    native_ids = data['source_frame_indices'][ids]
    if frontend_context_frame_ids is not None:
        if (not isinstance(frontend_context_frame_ids, list) or
                any(type(x) is not int or x < 0 for x in frontend_context_frame_ids) or
                len(set(frontend_context_frame_ids)) != len(frontend_context_frame_ids)):
            raise ValueError('frontend_context_frame_ids must be explicit unique nonnegative frame IDs')
    selected = {k: v[ids].copy() for k, v in data.items()}
    selected['native_frame_ids'] = native_ids
    selected['geometry_row_ids'] = ids
    coordinate = dict(name=coordinate_name, handedness='right', up_axis='unknown', pose_convention='T_world_local')
    track_data = None
    if tracks is not None:
        if not tracks_manifest:
            raise ValueError('Model point tracks require an RGB-only manifest')
        tracks_report = load_manifest(tracks_manifest)
        tracks_source = json.loads(Path(tracks_manifest).read_text())
        matches = [a for a in tracks_source['artifacts'] if a['id'] in tracks_report['method_artifact_ids']
                   and 'path' in a and (Path(tracks_manifest).resolve().parent / a['path']).resolve() == Path(tracks).resolve()
                   and a['kind'] in {'depth','object_geometry'}]
        if tracks_report['condition'] != 'rgb_only' or not matches:
            raise ValueError('Track file is not authenticated source-only model geometry')
        source = json.loads(manifest.read_text())
        roots = lambda m,r: {a['sha256'] for a in m['artifacts'] if a['origin']=='observed_rgb'
                             and a['id'] in r['method_artifact_ids']}
        if roots(source,report) != roots(tracks_source,tracks_report):
            raise ValueError('Tracks and geometry source RGB roots differ')
        track_data = selected_tracks(tracks,ids,data['timestamps'],coordinate_name,data['source_frame_indices'])
        parents['tracks_sha256'] = file_sha256(tracks)
    metric = report['metric_scale']['status']
    # Strict RGB model predictions never become measured/known metres here.
    metric = 'estimated' if metric in {'known', 'calibrated', 'estimated'} else 'unresolved'
    if asset is not None:
        if not asset_manifest:
            raise ValueError('Canonical model asset requires its RGB-only input manifest')
        asset_report = load_manifest(asset_manifest, [('object_geometry', asset)])
        if asset_report['condition'] != 'rgb_only':
            raise ValueError('Canonical asset must be derived from source RGB')
        source = json.loads(manifest.read_text())
        asset_source = json.loads(Path(asset_manifest).read_text())
        rgb_hashes = lambda m, r: {a['sha256'] for a in m['artifacts'] if a['id'] in r['method_artifact_ids']
                                  and a['origin'] == 'observed_rgb'}
        if rgb_hashes(source, report) != rgb_hashes(asset_source, asset_report):
            raise ValueError('Asset and geometry source RGB roots differ')
        parents['shared_asset_sha256'] = file_sha256(asset)
    scene = None
    if condition == 'B_object':
        if initializer is not None:
            if not initializer_receipt:
                raise ValueError('Structured initializer requires a selected-input receipt')
            receipt = json.loads(Path(initializer_receipt).read_text())
            for key in ['shared_asset_sha256','masks_sha256','tracks_sha256']:
                if receipt.get(key) is not None and key not in parents:
                    raise ValueError('Initializer consumed raw evidence absent from both packets: '+key)
            expected = dict(source_observations_sha256=parents['observations_sha256'],
                            native_frame_ids=native_ids.tolist(), input_scope='selected_frames_only',
                            frontend_context_frame_ids=frontend_context_frame_ids)
            if 'frontend_context_frame_ids' not in receipt:
                raise ValueError('Initializer receipt must declare upstream frontend context, even when unknown/null')
            if asset is not None:
                expected['shared_asset_sha256'] = parents['shared_asset_sha256']
            if masks is not None:
                expected['masks_sha256'] = parents['masks_sha256']
            if tracks is not None:
                expected['tracks_sha256'] = parents['tracks_sha256']
            for key, value in expected.items():
                if receipt.get(key) != value:
                    raise ValueError('Initializer receipt differs from packet source/selection: ' + key)
            if (not isinstance(receipt.get('producer_sha256'), str) or len(receipt['producer_sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in receipt['producer_sha256'])):
                raise ValueError('Initializer receipt requires producer SHA256')
            if receipt.get('initializer_sha256') != file_sha256(initializer):
                raise ValueError('Initializer receipt must bind the actual scene bytes')
            scene = json.loads(Path(initializer).read_text())
            validate_scene(scene)
            if not np.array_equal(scene['timestamps'], selected['timestamps']):
                raise ValueError('Initializer timestamps must equal only selected timestamps')
            if scene['coordinate_frame'] != coordinate or scene['metric_scale']['status'] != metric:
                raise ValueError('Initializer coordinate/scale status differs from packet')
            if any(o['geometry'].get('uri') for o in scene['objects']):
                raise ValueError('Initializer mesh geometry must be inline/self-contained')
            parents['initializer_sha256'] = file_sha256(initializer)
            parents['initializer_receipt_sha256'] = file_sha256(initializer_receipt)
        elif weak_hull_initializer:
            scene = compact_initializer(selected, object_id=object_id, metric_status=metric, coordinate_frame=coordinate)
        else:
            raise ValueError('B requires selected-input structured initializer; weak hull is an explicit separate ablation')
    output.mkdir(parents=True)
    np.savez_compressed(output / 'observations.npz', **selected)
    if track_data is not None:
        np.savez_compressed(output / 'tracks.npz', **track_data)
    (output / 'images').mkdir()
    for t, native in enumerate(native_ids):
        Image.fromarray(selected['rgb'][t, 0]).save(output / 'images' / f'{t:03d}-source-{native:05d}.png')
    if asset is not None:
        shutil.copy2(asset, output / ('canonical-asset' + Path(asset).suffix))
    if scene is not None:
        write_json(output / 'initializer-scene.json', scene)
        np.savez_compressed(output / 'representation-targets.npz', **feature_targets(scene))
    # Ship a narrowly scoped source-only tool and schema, never repository docs/results.
    shutil.copy2(HERE / 'representation_agent_tools.py', output / 'scene_tools.py')
    shutil.copy2(HERE.parent / 'artirecon/scene_sequence.py', output / 'scene_sequence.py')
    shutil.copy2(HERE / 'representation_agent_prompt.txt', output / 'PROMPT.txt')
    write_json(output / 'config.json', config)
    packet = dict(schema_version=1, condition=condition, source_parents=parents,
        native_frame_ids=native_ids.tolist(), geometry_row_ids=ids.tolist(), timestamps=selected['timestamps'].tolist(),
        frontend_context_frame_ids=frontend_context_frame_ids,
        requested_frame_budget=budget, selected_frame_count=len(ids),
        sampling='agent input budget only; selection precedes structured initialization, not necessarily frontend inference',
        coordinate_frame=coordinate, metric_scale_status=metric, object_id=object_id,
        initializer='initializer-scene.json' if scene is not None else None,
        canonical_asset=('canonical-asset' + Path(asset).suffix) if asset is not None else None,
        point_tracks='tracks.npz' if track_data is not None else None,
        initializer_kind=('weak_visible_hull' if weak_hull_initializer and initializer is None else 'source_model_mesh_rigid_track') if scene else None,
        evidence='observations.npz', images='images',
        evidence_semantics=dict(depth='model-predicted camera Z, not sensor depth',
            points='native predicted world points when supplied; otherwise predicted depth unprojected with K/c2w',
            camera_projection='K may approximate predicted dense rays; K_pinhole_rmse_px is retained when supplied',
            object_masks='predicted instance masks, association unknown unless object_mask_known',
            occlusion_mask='unknown object space; excluded from source silhouette/depth comparison',
            hidden_geometry='unknown; visible hull is only an initial hypothesis',
            contact='unknown; do not attach hand/object or assume resting/grasp'),
        feature_target_semantics=dict(source='model-derived pseudo-targets, not GT',
            canonical_extent='visible hull XYZ extent in packet gauge; complete shape_known=false',
            pose_rotation_6d='first two rotation matrix columns, column-major; rotation_known required',
            motion='pose_translation/pose_rotation_6d are one track per shared object',
            unknown='masked by observed, shape_known, rotation_known, contact_known'),
        measurement_template=dict(status='not_started', source_mask_iou=None, predicted_depth_agreement_m=None,
            gt_geometry_accuracy=None, contact_accuracy=None, execution_success=None,
            agent_turns=None, tool_calls=None, input_tokens=None, output_tokens=None,
            elapsed_seconds=None, scene_edits=None, monetary_cost=None),
        preprocessing_timing_seconds=dict(frontend=None, masks=None, canonical_asset=None,
            structured_initializer=(receipt.get('elapsed_seconds') if scene is not None and initializer is not None else None)),
        forbidden_inputs=['evaluation outputs', 'ground truth', 'sensor depth', 'other agent runs',
                          'prior reconstruction metrics', 'internet', 'simulator feedback'])
    write_json(output / 'packet.json', packet)
    hashes = {str(p.relative_to(output)): file_sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'frozen-inputs.json', dict(schema_version=1, files=hashes,
        common_evidence_sha256=hashes['observations.npz'], prompt_sha256=hashes['PROMPT.txt'],
        config_sha256=hashes['config.json'], packet_builder_sha256=file_sha256(__file__)))
    return packet


def verify_packet(packet):
    packet = Path(packet)
    frozen = json.loads((packet / 'frozen-inputs.json').read_text())
    for name, expected in frozen['files'].items():
        path = packet / name
        if path.is_symlink() or not path.resolve().is_relative_to(packet.resolve()):
            raise ValueError('Frozen packet paths must be local ordinary files: ' + name)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError('Frozen packet input changed: ' + name)
    allowed = set(frozen['files']) | {'frozen-inputs.json','execution-receipt.json'}
    for path in packet.rglob('*'):
        name = str(path.relative_to(packet))
        if path.is_symlink():
            raise ValueError('Packet contains a symlink: ' + name)
        if path.is_file() and name not in allowed and not name.startswith('output/'):
            raise ValueError('Unlisted file in source-only packet: ' + name)
    return frozen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('config')
    c.add_argument('--model', required=True)
    c.add_argument('--python', required=True)
    c.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('build')
    for name in ['observations', 'manifest', 'config', 'output']:
        b.add_argument('--' + name, type=Path, required=True)
    b.add_argument('--condition', choices=['A_raw', 'B_object'], required=True)
    b.add_argument('--budget', type=int, required=True)
    b.add_argument('--masks', type=Path)
    b.add_argument('--mask-manifest', type=Path)
    b.add_argument('--object-prompt')
    b.add_argument('--object-id', default='task_object')
    b.add_argument('--coordinate-name', default='predicted_world')
    for name in ['asset', 'asset-manifest', 'initializer', 'initializer-receipt', 'tracks', 'tracks-manifest']:
        b.add_argument('--' + name, type=Path)
    b.add_argument('--weak-hull-initializer', action='store_true')
    b.add_argument('--frontend-context-frame-count', type=int)
    b.add_argument('--frontend-context-frame-ids', type=Path,
                   help='JSON list of original source indices used by the model (supports strided inference)')
    v = sub.add_parser('verify')
    v.add_argument('packet', type=Path)
    a = p.parse_args()
    if a.command == 'config':
        if a.output.exists():
            raise FileExistsError(a.output)
        write_json(a.output, default_config(a.model, a.python))
    elif a.command == 'verify':
        print(json.dumps(verify_packet(a.packet), indent=2))
    else:
        args = vars(a).copy()
        del args['command']
        args['config'] = json.loads(args['config'].read_text())
        context_count = args.pop('frontend_context_frame_count')
        context_ids = args.pop('frontend_context_frame_ids')
        if context_count is not None and context_ids is not None:
            raise ValueError('Provide context frame count or explicit IDs, not both')
        args['frontend_context_frame_ids'] = (json.loads(context_ids.read_text()) if context_ids is not None
            else list(range(context_count)) if context_count is not None else None)
        print(json.dumps(build_packet(**args), indent=2))


if __name__ == '__main__':
    main()
