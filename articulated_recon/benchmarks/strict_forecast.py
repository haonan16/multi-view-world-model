"""First-image-only preparation and provenance checks for the RGB Wan adapter.

These checks authenticate declared lineage and array contents, not the truth of
an estimator's provenance or the quality of generated camera views/motion.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from artirecon.input_protocol import file_sha256, load_manifest, validate_manifest
from artirecon.scene_sequence import load_scene


TEMPORAL_FIELDS = {'depth_m', 'depth', 'points_world', 'local_points_m', 'raw_uncertainty',
                   'uncertainty', 'confidence', 'confidence_rank', 'valid', 'focal_scale',
                   'per_frame_output_intrinsics', 'source_frame_indices'}


def _path(record, manifest_path):
    return (Path(manifest_path).resolve().parent / record['path']).resolve()


def _first_image_manifest(path):
    """Reject observed future ancestry, including mislabeled multi-frame files."""
    path = Path(path).resolve()
    report = load_manifest(path)
    manifest = json.loads(path.read_text())
    if report['condition'] != 'rgb_only' or report['task_mode'] != 'forecast':
        raise ValueError('Strict forecast requires RGB-only forecast ancestry throughout')
    active = set(report['method_artifact_ids'])
    records = [a for a in manifest['artifacts'] if a['id'] in active]
    roots = [a for a in records if a['origin'] == 'observed_rgb']
    if len(roots) != 1:
        raise ValueError('Strict forecast requires exactly one observed image artifact')
    root = roots[0]
    cutoff = manifest['observed_until_s']
    if root['time_range_s'] != [cutoff, cutoff]:
        raise ValueError('Observed RGB must declare only the initial timestamp')
    if _path(root, path).suffix != '.npz':
        raise ValueError('Use an RGB-only numeric NPZ exported by rgb_frontend')
    with np.load(_path(root, path), allow_pickle=False) as data:
        if set(data.files) != {'rgb', 'timestamps'}:
            raise ValueError('Observed source packet may contain only RGB and timestamps')
        rgb, times = data['rgb'].copy(), data['timestamps'].copy()
    if (rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] != 3
            or min(rgb.shape[1:3]) < 2 or times.shape != (1,) or times[0] != cutoff):
        raise ValueError('Observed source must contain exactly one RGB image and its timestamp')
    for record in records:
        if record['kind'] == 'model_prior':
            continue
        if 'time_range_s' in record and record['time_range_s'] != [cutoff, cutoff]:
            raise ValueError('Initial geometry/camera ancestry may not contain future time ranges')
        material = _path(record, path)
        if record['kind'] in {'depth', 'cameras'} and material.suffix != '.npz':
            raise ValueError('Initial depth/camera ancestry requires numeric singleton-timestamp NPZs')
        if material.suffix == '.npz':
            with np.load(material, allow_pickle=False) as arrays:
                if record['kind'] in {'depth', 'cameras'} and 'timestamps' not in arrays:
                    raise ValueError('Initial depth/camera ancestry requires a singleton timestamp')
                for name in TEMPORAL_FIELDS.intersection(arrays.files):
                    if arrays[name].ndim < 1 or arrays[name].shape[0] != 1:
                        raise ValueError('Initial ancestry temporal array must contain one frame: ' + name)
                if 'timestamps' in arrays and (arrays['timestamps'].shape != (1,)
                                               or arrays['timestamps'][0] != cutoff):
                    raise ValueError('Initial geometry/camera ancestry contains later timestamps')
                if 'rgb' in arrays and (arrays['rgb'].shape[0] != 1
                                       or arrays['rgb'].ndim == 5 and arrays['rgb'].shape[1] != 1):
                    raise ValueError('Initial geometry/camera ancestry contains extra RGB frames/views')
                if record['kind'] == 'cameras' and 'c2w' in arrays and arrays['c2w'].shape not in {(4, 4), (1, 4, 4)}:
                    raise ValueError('Initial camera ancestry must contain only the source camera')
                if record['kind'] == 'depth':
                    names = [name for name in ['depth_m', 'depth'] if name in arrays]
                    if len(names) != 1 or arrays[names[0]].ndim != 4 or arrays[names[0]].shape[:2] != (1, 1):
                        raise ValueError('Initial depth ancestry needs depth_m (or legacy depth) [1,1,H,W]')
        elif material.suffix == '.json':
            payload = json.loads(material.read_text())
            if isinstance(payload, dict) and 'timestamps' in payload and payload['timestamps'] != [cutoff]:
                raise ValueError('Initial scene/camera ancestry contains later timestamps')
    return manifest, report, root, rgb, float(cutoff)


def initial_inputs(geometry_bundle, scene_path, scene_manifest, object_id):
    geometry_bundle, scene_path = Path(geometry_bundle).resolve(), Path(scene_path).resolve()
    gm = geometry_bundle / 'input-manifest.json'
    geometry = geometry_bundle / 'geometry.npz'
    g, gr, source, rgb, t0 = _first_image_manifest(gm)
    s, sr, scene_source, _, scene_t0 = _first_image_manifest(scene_manifest)
    load_manifest(gm, [('depth', geometry), ('cameras', geometry)])
    load_manifest(scene_manifest, [('object_geometry', scene_path)])
    if source['sha256'] != scene_source['sha256'] or scene_t0 != t0:
        raise ValueError('Initial geometry and scene must share the identical first-image source')
    scene = load_scene(scene_path, require_metric_scale=True)
    if scene['timestamps'] != [t0] or scene['coordinate_frame']['name'] != 'source_camera':
        raise ValueError('Initial scene must contain one timestamp in source_camera coordinates')
    obj = next((o for o in scene['objects'] if o['id'] == object_id), None)
    if obj is None or obj['observed'] != [True]:
        raise ValueError('Camera target needs an explicitly supported initial object pose')
    with np.load(geometry, allow_pickle=False) as z:
        if z['rgb'].shape != (1, 1, *rgb.shape[1:]) or not np.array_equal(z['rgb'][:, 0], rgb):
            raise ValueError('Predicted geometry RGB must exactly match the sole observed image')
        if z['timestamps'].shape != (1,) or z['timestamps'][0] != t0:
            raise ValueError('Predicted geometry must contain only the initial timestamp')
        depth_name = 'depth_m' if 'depth_m' in z else 'depth'
        if z[depth_name].shape != (1, 1, *rgb.shape[1:3]):
            raise ValueError('Predicted depth must cover exactly one source image')
        k, c = z['K'].copy(), z['c2w'].copy()
        if k.shape == (1, 3, 3): k = k[0]
        if c.shape == (1, 4, 4): c = c[0]
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1]) or c.shape != (4, 4)
                or not np.allclose(c, np.eye(4))):
            raise ValueError('Use one finite estimated K and identity source-camera gauge')
    merged = deepcopy(s)
    merged['artifacts'] = []
    by_id = {}
    for manifest, report, path in [(s, sr, scene_manifest), (g, gr, gm)]:
        active = set(report['method_artifact_ids'])
        for original in manifest['artifacts']:
            if original['id'] not in active: continue
            record = deepcopy(original)
            if 'path' in record: record['path'] = str(_path(record, path))
            if record['id'] in by_id:
                if ({k: v for k, v in by_id[record['id']].items() if k != 'path'}
                        != {k: v for k, v in record.items() if k != 'path'}):
                    raise ValueError('Conflicting initial artifact lineage: ' + record['id'])
            else: by_id[record['id']] = record
    merged['artifacts'] = list(by_id.values())
    merged['method_inputs'] = list(dict.fromkeys(sr['method_inputs'] + gr['method_inputs']))
    target = np.asarray(obj['T_world_object'])[0, :3, 3]
    return merged, rgb, k, t0, target


def condition_arrays(rgb, k, t0, target, *, frames, dt_s, angle_degrees):
    import cv2
    from benchmarks.world_model_views import virtual_cameras
    if type(frames) is not int or frames < 5 or (frames - 1) % 4:
        raise ValueError('Declare a 4k+1 prediction horizon of at least five frames')
    if not isinstance(dt_s, (int, float)) or isinstance(dt_s, bool) or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError('Declare a finite positive prediction timestep independent of future RGB')
    h, w = rgb.shape[1:3]
    ch, cw = min(h, w // 2), min(w, h * 2)
    y, x = (h - ch) // 2, (w - cw) // 2
    initial = cv2.resize(rgb[0, y:y+ch, x:x+cw], (320, 160), interpolation=cv2.INTER_AREA)[None]
    k = k.copy(); k[0, 2] -= x; k[1, 2] -= y
    k[0] *= 320 / cw; k[1] *= 160 / ch
    indices = np.full(frames, -1, dtype=np.int64); indices[0] = 0
    times = t0 + np.arange(frames) * dt_s
    if not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('Declared prediction times must remain finite and increase strictly')
    return dict(rgb=initial, K=np.repeat(k[None], 4, axis=0),
                c2w=virtual_cameras(target, angle_degrees),
                timestamps=times, source_frame_indices=indices), [x, y, cw, ch]


def prepare_forecast(*, geometry_bundle, scene_path, scene_manifest, output, object_id,
                     task_text, prompt_key, dt_s, frames=17, angle_degrees=20.):
    """Design fixed virtual requests using only authenticated initial predictions."""
    if not isinstance(task_text, str) or not task_text.strip() or not isinstance(prompt_key, str) or not prompt_key.strip():
        raise ValueError('Declare the exact task text and its embedding key')
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    manifest, rgb, k, t0, target = initial_inputs(geometry_bundle, scene_path, scene_manifest, object_id)
    arrays, crop = condition_arrays(rgb, k, t0, target, frames=frames, dt_s=dt_s, angle_degrees=angle_degrees)
    output.mkdir(parents=True)
    np.savez_compressed(output / 'condition.npz', **arrays)
    request_id = 'forecast_camera_requests'
    manifest['artifacts'].append(dict(id=request_id, kind='cameras', origin='model_prediction',
        parent_ids=manifest['method_inputs'].copy(), path='condition.npz', sha256=file_sha256(output/'condition.npz'),
        camera_design_horizon='Initial image only', prediction_frames=frames, prediction_dt_s=dt_s,
        task_text=task_text, prompt_key=prompt_key,
        assumption='Designed nonzero virtual baselines; generated camera consistency is unverified'))
    manifest['method_inputs'] = [request_id]
    validate_manifest(manifest, output, verify_files=True)
    (output/'input-manifest.json').write_text(json.dumps(manifest, indent=2))
    geometry_manifest = Path(geometry_bundle).resolve() / 'input-manifest.json'
    prep = dict(mode='forecast', conditioning_artifact_id=request_id, observed_source_frames=1,
        extra_observed_views=0, condition_sha256=file_sha256(output/'condition.npz'),
        geometry_manifest=str(geometry_manifest), geometry_manifest_sha256=file_sha256(geometry_manifest),
        scene_manifest=str(Path(scene_manifest).resolve()), scene_manifest_sha256=file_sha256(scene_manifest),
        scene_path=str(Path(scene_path).resolve()), scene_sha256=file_sha256(scene_path), object_id=object_id,
        task_text=task_text, prompt_key=prompt_key, frames=frames, dt_s=float(dt_s), angle_degrees=angle_degrees,
        camera_design_horizon='Initial image only', conditioning_horizon='One observed source image; declared future times are prediction labels',
        source_timestamps=[t0], prediction_timestamps=arrays['timestamps'].tolist(), source_crop_xywh=crop,
        camera_interpretation='Requested poses, not measured calibration of generated pixels',
        supplied_dataset_cameras=False, evaluator_geometry_used=False,
        output_modality='RGB only; this adapter does not export native predicted depth',
        training_caveat='The documented robot+human checkpoint includes single-view first-frame training. Estimated virtual cameras and RGB-only scene estimates may differ from its training distribution; generated-view and task-motion accuracy remain unmeasured')
    (output/'preparation.json').write_text(json.dumps(prep, indent=2))
    return prep


def verify_condition_arrays(actual, expected):
    """Recreate inputs exactly, allowing only float64 camera-design roundoff.

    NumPy/BLAS builds can differ by a few ULPs when normalizing virtual-camera
    bases. This is not a geometric tolerance: stored arrays remain hash-bound,
    and RGB, K, timestamps and source IDs must still match exactly. At zero the
    bound is eight float64 subnormal steps, not an absolute camera tolerance.
    """
    if set(actual)!=set(expected):
        raise ValueError('Forecast conditioning arrays have extra or missing fields')
    camera_report=None
    for name,wanted in expected.items():
        value=np.asarray(actual[name]);wanted=np.asarray(wanted)
        if name=='c2w':
            if value.shape!=wanted.shape or value.dtype!=np.dtype('float64') or wanted.dtype!=np.dtype('float64') or not np.isfinite(value).all():
                raise ValueError('Recreated forecast cameras require finite float64 arrays with identical shape')
            spacing=np.spacing(np.abs(wanted));difference=np.abs(value-wanted)
            if not np.all(difference<=8*spacing):
                raise ValueError('Forecast camera requests differ beyond float64 recreation roundoff')
            camera_report=dict(exact=bool(np.array_equal(value,wanted)),max_abs_difference=float(difference.max()),
                max_float64_spacing_units=float(np.max(difference/spacing)),allowed_float64_spacing_units=8)
        elif not np.array_equal(value,wanted):
            raise ValueError('Forecast arrays differ from first-image-only camera/horizon preparation: '+name)
    return dict(non_camera_arrays_exact=True,camera_recreation=camera_report,stored_artifact_hash_verification_unchanged=True)


def validate_forecast(condition):
    condition = Path(condition).resolve()
    prep = json.loads((condition/'preparation.json').read_text())
    if prep.get('mode') != 'forecast': raise ValueError('Strict forecast requires its dedicated first-image preparation')
    for key in ['geometry_manifest', 'scene_manifest', 'scene_path']:
        expected = prep['scene_sha256'] if key == 'scene_path' else prep[key+'_sha256']
        if file_sha256(prep[key]) != expected: raise ValueError('First-image preparation ancestry changed')
    manifest, rgb, k, t0, target = initial_inputs(Path(prep['geometry_manifest']).parent,
        prep['scene_path'], prep['scene_manifest'], prep['object_id'])
    expected, crop = condition_arrays(rgb, k, t0, target, frames=prep['frames'], dt_s=prep['dt_s'], angle_degrees=prep['angle_degrees'])
    report = load_manifest(condition/'input-manifest.json', [('cameras', condition/'condition.npz')])
    actual_manifest = json.loads((condition/'input-manifest.json').read_text())
    if report['condition'] != 'rgb_only' or report['task_mode'] != 'forecast' or actual_manifest['observed_until_s'] != t0:
        raise ValueError('Strict forecast input boundary changed')
    if file_sha256(condition/'condition.npz') != prep['condition_sha256']:
        raise ValueError('Forecast condition changed')
    with np.load(condition/'condition.npz', allow_pickle=False) as z:
        report['condition_recreation']=verify_condition_arrays(z,expected)
    expected_records = {a['id']: a for a in manifest['artifacts']}
    actual_records = {a['id']: a for a in actual_manifest['artifacts']}
    if set(actual_records) != set(expected_records) | {'forecast_camera_requests'}:
        raise ValueError('Forecast condition has extra or missing ancestry')
    for name, record in expected_records.items():
        if actual_records[name] != record: raise ValueError('Forecast initial ancestry changed')
    request = actual_records['forecast_camera_requests']
    if (actual_manifest['method_inputs'] != ['forecast_camera_requests']
            or request['parent_ids'] != manifest['method_inputs']
            or request.get('task_text') != prep['task_text'] or request.get('prompt_key') != prep['prompt_key']
            or prep.get('conditioning_artifact_id') != 'forecast_camera_requests'):
        raise ValueError('Forecast task or camera ancestry binding changed')
    return prep, report


def bind_task(prep, embedding_path, prompt_key, null_text=False, custom=None):
    """Verify the exact task, including existing author cached task embeddings."""
    import torch
    if null_text or prompt_key != prep['prompt_key']:
        raise ValueError('Strict task forecast must use its declared task key; null text is unsupported')
    path = Path(embedding_path).resolve()
    data = torch.load(path, map_location='cpu', weights_only=True)
    row = data.get(prompt_key) if isinstance(data, dict) else None
    if not isinstance(row, dict) or row.get('text') != prep['task_text']:
        raise ValueError('Forecast task text does not match the actual cached/custom embedding')
    if custom and custom['text'] != prep['task_text']:
        raise ValueError('Custom task receipt disagrees with forecast preparation')
    return dict(mode='custom_task_prompt' if custom else 'cached_task_prompt', task_text_used=True,
                prompt_key=prompt_key, text=prep['task_text'], embedding_path=str(path),
                embedding_sha256=file_sha256(path))


def generated_records(output, times):
    """One mixed packet, with explicit stream selectors and generated future."""
    shared = dict(kind='rgb', origin='model_prediction', generated=True, path='observations.npz',
                  sha256=file_sha256(Path(output)/'observations.npz'),
                  parent_ids=['forecast_camera_requests', 'world_model_weights'],
                  mixed_origin_rule='Only observed_mask[0,0] is observed RGB; all other pixels are generated')
    return [dict(shared, id='generated_source_forecast', view_role='source_forecast', view_indices=[0],
                 frame_indices=list(range(1, len(times))), time_range_s=[float(times[1]), float(times[-1])]),
            dict(shared, id='generated_helper_rgb', view_role='helper', view_indices=[1, 2, 3],
                 frame_indices=list(range(len(times))), time_range_s=[float(times[0]), float(times[-1])])]
