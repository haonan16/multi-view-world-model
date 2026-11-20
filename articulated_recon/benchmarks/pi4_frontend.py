"""Run released Pi4 on an isolated RGB bundle, preserving its native predictions.

No depth, calibration, reference geometry, masks, or evaluation files are read.
The output uses a time-varying camera schema; it is deliberately not the legacy
static-camera UniDepth archive. Metric units remain model estimates.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
from rgb_frontend import load_source_bundle, sha256
from artirecon.input_protocol import validate_manifest


GEOMETRY_KEYS = {'local_points', 'camera_poses', 'metric_scale', 'confidence'}
MOTION_KEYS = {'tracks_2d', 'tracks_3d', 'query_frame', 'dynamic_mask',
               'visibility', 'tracking_confidence'}


def select_frames(count, stride=1, max_frames=None):
    if type(stride) is not int or stride < 1:
        raise ValueError('stride must be a positive integer')
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError('max_frames must be a positive integer')
    return np.arange(count, dtype=np.int64)[::stride][:max_frames]


def resize_rgb(rgb, max_size=512):
    """Same shared-size LANCZOS/16-pixel rounding as released pi4.io.load_images."""
    from PIL import Image
    if type(max_size) is not int or max_size < 16:
        raise ValueError('max_size must be an integer >=16')
    height, width = rgb.shape[1:3]
    factor = max_size / max(height, width)
    target = (max(16, round(width * factor / 16) * 16),
              max(16, round(height * factor / 16) * 16))
    return np.stack([np.asarray(Image.fromarray(x).resize(target, Image.Resampling.LANCZOS))
                     for x in rgb])


def validate_predictions(raw, shape, query_frame):
    """Reject malformed geometry without replacing NaN track observations."""
    frames, height, width = shape[:3]
    expected = GEOMETRY_KEYS | (MOTION_KEYS if query_frame is not None else set())
    if set(raw) != expected:
        raise ValueError(f'Unexpected Pi4 output keys: {set(raw) ^ expected}')
    grid = (1, frames, height, width)
    shapes = {'local_points': (*grid, 3), 'camera_poses': (1, frames, 4, 4),
              'metric_scale': (1,), 'confidence': grid,
              'tracks_2d': (*grid, 2), 'tracks_3d': (*grid, 3), 'query_frame': (1,),
              'dynamic_mask': grid, 'visibility': grid, 'tracking_confidence': grid}
    for key, value in raw.items():
        if value.shape != shapes[key] or value.dtype.kind not in 'fiu':
            raise ValueError(f'Invalid Pi4 {key} shape/dtype: {value.shape}/{value.dtype}')
    scale = raw['metric_scale']
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Pi4 must predict a finite positive metric scale')
    poses = raw['camera_poses']
    rotation = poses[..., :3, :3]
    if (not np.isfinite(poses).all() or
            not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-4) or
            not np.allclose(rotation @ rotation.swapaxes(-1, -2), np.eye(3), atol=2e-3) or
            not np.allclose(np.linalg.det(rotation), 1, atol=2e-3)):
        raise ValueError('Pi4 returned malformed camera-to-world poses')
    for key in ['confidence'] + (['dynamic_mask', 'visibility', 'tracking_confidence']
                                if query_frame is not None else []):
        values = raw[key]
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f'Invalid native probability values: {key}')
    points = raw['local_points']
    if not np.all(np.any(np.isfinite(points).all(-1) & (points[..., 2] > 0), axis=(-2, -1))):
        raise ValueError('Pi4 produced a frame without finite positive-depth geometry')
    if query_frame is not None and (raw['query_frame'].dtype.kind not in 'iu' or
                                   raw['query_frame'].tolist() != [query_frame]):
        raise ValueError('Returned query index differs from requested sampled-frame index')


def metric_geometry(raw, rgb, timestamps, source_frame_indices):
    """Apply predicted scale once to local points and camera translations.

    Tracks are already in metric world units and must not be multiplied again.
    No rigid alignment, recentering, static-camera constraint, or interpolation.
    """
    scale = float(raw['metric_scale'][0])
    local = raw['local_points'][0] * scale
    cameras = raw['camera_poses'][0].copy()
    cameras[:, :3, 3] *= scale
    world = np.einsum('tij,thwj->thwi', cameras[:, :3, :3], local)
    world += cameras[:, None, None, :3, 3]
    result = dict(rgb=rgb[:, None], timestamps=timestamps, source_frame_indices=source_frame_indices,
                  local_points_m=local[:, None], points_world=world[:, None], depth_m=local[..., 2][:, None],
                  c2w=cameras[:, None], metric_scale=np.asarray(scale),
                  confidence=raw['confidence'][0, :, None],
                  depth_valid=(np.isfinite(local).all(-1) & (local[..., 2] > 0))[:, None])
    if 'tracks_3d' in raw:
        for key in MOTION_KEYS:
            result[key] = raw[key][0] if key == 'query_frame' else raw[key][0, :, None]
        result['track_finite'] = np.isfinite(result['tracks_3d']).all(-1) & np.isfinite(result['tracks_2d']).all(-1)
    return result


def extract_tracks(raw, query_flat_indices):
    """Select fixed native query pixels for a downstream identity-aware fit.

    The caller selects indices from an RGB-derived query-frame object mask.
    Visibility remains a probability; no interpolation or threshold is applied.
    """
    if not MOTION_KEYS.issubset(raw):
        raise ValueError('Native motion outputs are required')
    values = raw['tracks_3d']
    if values.ndim != 5 or values.shape[0] != 1 or values.shape[-1] != 3:
        raise ValueError('Expected native tracks_3d [1,T,H,W,3]')
    _, count, height, width, _ = values.shape
    ids = np.asarray(query_flat_indices)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids)
            or np.any(ids < 0) or np.any(ids >= height * width)):
        raise ValueError('Query indices must be unique valid flattened query pixels')
    def selected(key, channels=()):
        return raw[key][0].reshape(count, height * width, *channels)[:, ids].copy()
    return dict(tracks3d=selected('tracks_3d', (3,)),
        tracks2d=selected('tracks_2d', (2,)),
        query_pixels=np.stack([ids % width, ids // width], axis=-1),
        query_frame=np.full(len(ids), int(raw['query_frame'][0]), dtype=np.int64),
        query_flat_indices=ids.copy(), trackvisibility=selected('visibility'),
        confidence=selected('tracking_confidence'))


def pinhole_diagnostic(local_points, max_samples=8192):
    """Approximate model rays by per-frame pinhole K; never used for point lifting.

    Dense model rays need not be pinhole-consistent. The residual quantifies
    this approximation, and an impossible fit stays NaN rather than invented K.
    """
    height, width = local_points.shape[1:3]
    yy, xx = np.mgrid[:height, :width]
    pixel = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    matrices, residuals = [], []
    for xyz in local_points:
        xyz = xyz.reshape(-1, 3)
        valid = np.isfinite(xyz).all(-1) & (xyz[:, 2] > 0)
        ids = np.flatnonzero(valid)
        ids = ids[np.linspace(0, len(ids) - 1, min(max_samples, len(ids)), dtype=int)] if len(ids) else ids
        matrix, residual = np.full((3, 3), np.nan), np.nan
        if len(ids) >= 4:
            rays = xyz[ids, :2] / xyz[ids, 2:3]
            fits = [np.linalg.lstsq(np.c_[rays[:, axis], np.ones(len(ids))],
                                    pixel[ids, axis], rcond=None) for axis in range(2)]
            if all(fit[2] == 2 and np.isfinite(fit[0]).all() and fit[0][0] > 0 for fit in fits):
                fx, cx = fits[0][0]
                fy, cy = fits[1][0]
                matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
                projected = rays * [fx, fy] + [cx, cy]
                residual = float(np.sqrt(np.mean(np.sum((projected - pixel[ids]) ** 2, axis=1))))
        matrices.append(matrix)
        residuals.append(residual)
    return np.stack(matrices), np.asarray(residuals)


def make_manifest(source_manifest, source, out, checkpoint_record, tracking):
    manifest = copy.deepcopy(source_manifest)
    source_record = next(x for x in manifest['artifacts'] if x['id'] == source['id'])
    source_record['path'] = 'rgb.npz'
    manifest['artifacts'].append(dict(id='pi4_weights', kind='model_prior', origin='prior',
        prior_type='learned_model_weights', scope='global', parent_ids=[], **checkpoint_record))
    parents = ['source_rgb', 'pi4_weights']
    for artifact_id, kind, filename in [
            ('pi4_raw_predictions', 'depth', 'raw-predictions.npz'),
            ('pi4_metric_geometry', 'depth', 'geometry.npz'),
            ('pi4_predicted_cameras', 'cameras', 'geometry.npz')]:
        manifest['artifacts'].append(dict(id=artifact_id, kind=kind, origin='model_prediction',
            parent_ids=parents, path=filename, sha256=sha256(out / filename)))
        manifest['method_inputs'].append(artifact_id)
    if tracking:
        manifest['artifacts'].append(dict(id='pi4_motion', kind='masks', origin='model_prediction',
            parent_ids=parents, path='raw-predictions.npz', sha256=sha256(out / 'raw-predictions.npz'),
            description='Native dynamic probabilities, visibility and dense correspondences'))
        manifest['method_inputs'].append('pi4_motion')
    manifest['metric_scale'] = dict(status='estimated', artifact_ids=['pi4_metric_geometry'],
        description='Pi4 learned sequence scale; no reference alignment or external scale fitting')
    manifest['robot_registration'] = dict(status='unresolved', artifact_ids=[],
        description='Pi4 world coordinates have no independently observed target-robot registration')
    return manifest


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def git_revision(repo):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    revision = git('rev-parse', 'HEAD')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Pi4 tracked code is modified; use a clean official revision')
    return revision


def infer(bundle, out, checkpoint, revision, repo, code_revision, stride=1, max_frames=None,
          max_size=512, query_frame=0, warm_repeats=1, viewer=False, seed=0,
          checkpoint_sha256=None):
    start = time.monotonic()
    bundle, out, checkpoint, repo = map(lambda p: Path(p).resolve(), (bundle, out, checkpoint, repo))
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite predictions: {out}')
    if not checkpoint.is_file() or checkpoint.suffix != '.safetensors':
        raise ValueError('Use an existing local .safetensors checkpoint; no inference-time downloads')
    if not revision.strip() or git_revision(repo) != code_revision:
        raise ValueError('Declare checkpoint revision and exact clean Pi4 code commit')
    if type(warm_repeats) is not int or not 0 <= warm_repeats <= 3:
        raise ValueError('warm_repeats must be in [0,3]')
    manifest, source, source_rgb, source_times = load_source_bundle(bundle)
    ids = select_frames(len(source_rgb), stride, max_frames)
    if query_frame is not None and (type(query_frame) is not int or len(ids) < 2 or not 0 <= query_frame < len(ids)):
        raise ValueError('Tracking needs >=2 sampled frames and a valid query_frame')
    frames = resize_rgb(source_rgb[ids], max_size)
    timestamps = source_times[ids]
    metadata = dict(source=str(bundle / source['path']), source_frame_indices=ids.tolist(),
        source_timestamps_s=timestamps.tolist(), image_size=[frames.shape[2], frames.shape[1]],
        resize='shared first-image aspect ratio; patch-aligned LANCZOS resize')
    if len(timestamps) > 1 and np.allclose(np.diff(timestamps), np.diff(timestamps)[0]):
        metadata['source_fps'] = float(1 / np.diff(timestamps)[0])
    sys.path.insert(0, str(repo))
    import torch
    import pi4
    from pi4 import Pi4
    if not Path(pi4.__file__).resolve().is_relative_to(repo):
        raise ValueError('Imported Pi4 differs from the declared code repository')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Released Pi4 requires a CUDA GPU with BF16 support')
    torch.manual_seed(seed)
    np.random.seed(seed)
    out.mkdir(parents=True)
    checkpoint_record = dict(model_name='Pi4', model_url='https://huggingface.co/yyfz233/Pi4',
        revision=revision, checkpoint_sha256=sha256(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
        code_url='https://github.com/yyfz/Pi4', code_revision=code_revision,
        license='CC-BY-NC-4.0', revision_source='caller-declared checkpoint revision; file SHA256 bound')
    if checkpoint_sha256 is not None and checkpoint_record['checkpoint_sha256'] != checkpoint_sha256:
        raise ValueError('Checkpoint differs from the preflight-pinned SHA256')
    run = dict(status='running', checkpoint=str(checkpoint), **checkpoint_record,
        source_manifest_sha256=sha256(bundle / 'input-manifest.json'), frames=len(frames),
        source_frame_indices=ids.tolist(), timestamps_s=timestamps.tolist(),
        input_resolution_hw=list(source_rgb.shape[1:3]), model_resolution_hw=list(frames.shape[1:3]),
        stride=stride, max_frames=max_frames, max_size=max_size, query_frame=query_frame,
        seed=seed, warm_repeats=warm_repeats, native_joint_sequence=True, temporal_windowing=False,
        measured_depth_used=False, known_intrinsics_supplied=False, fixed_camera_assumption=False,
        reference_alignment=False, geometry_schema='pi4_dynamic_camera_v1',
        coordinate_frame='Pi4 predicted world frame; no robot registration',
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), adapter_sha256=sha256(Path(__file__)),
        packages={key: importlib.metadata.version(key) for key in ['triton', 'safetensors', 'flex_gemm']})
    write_json(out / 'inference.json', run)
    try:
        t0 = time.monotonic()
        images = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div_(255).unsqueeze(0).cuda()
        torch.cuda.synchronize()
        run['input_transfer_seconds'] = time.monotonic() - t0
        t0 = time.monotonic()
        model = Pi4.from_pretrained(str(checkpoint)).cuda().eval()
        torch.cuda.synchronize()
        run['model_load_seconds'] = time.monotonic() - t0
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        prediction = model(images, query_frame=query_frame)
        torch.cuda.synchronize()
        run['cold_forward_seconds'] = time.monotonic() - t0
        run['cold_frames_per_second'] = len(frames) / run['cold_forward_seconds']
        run['cold_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        run['cold_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        run['native_tensor_dtypes'] = {key: str(value.dtype) for key, value in prediction.items()}
        t0 = time.monotonic()
        cpu_prediction = {key: value.detach().cpu() for key, value in prediction.items()}
        del prediction
        raw = {key: (value.float() if value.dtype == torch.bfloat16 else value).numpy()
               for key, value in cpu_prediction.items()}
        run['raw_cpu_transfer_seconds'] = time.monotonic() - t0
        validate_predictions(raw, frames.shape, query_frame)
        run['warm_forward_seconds'] = []
        for _ in range(warm_repeats):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.monotonic()
            warm = model(images, query_frame=query_frame)
            torch.cuda.synchronize()
            run['warm_forward_seconds'].append(time.monotonic() - t0)
            run['warm_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            run['warm_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            del warm
        run['warm_frames_per_second'] = [len(frames) / value for value in run['warm_forward_seconds']]
        run['timing_note'] = ('Cold forward includes first-call compilation/tuning in this process; persistent '
            'kernel caches may already be warm. Warm repeats use identical full inputs. FPS excludes loading, '
            'CPU transfer and export. No cross-method speed claim is implied.')
        t0 = time.monotonic()
        np.savez_compressed(out / 'raw-predictions.npz', **raw)
        geometry = metric_geometry(raw, frames, timestamps, ids)
        k, residual = pinhole_diagnostic(geometry['local_points_m'][:, 0])
        geometry['K'], geometry['K_pinhole_rmse_px'] = k[:, None], residual[:, None]
        geometry['K_is_pinhole_approximation'] = np.asarray(True)
        np.savez_compressed(out / 'geometry.npz', **geometry)
        source_bytes = (bundle / source['path']).read_bytes()
        import hashlib
        if hashlib.sha256(source_bytes).hexdigest() != source['sha256']:
            raise ValueError('Source RGB changed during inference')
        (out / 'rgb.npz').write_bytes(source_bytes)
        manifest = make_manifest(manifest, source, out, checkpoint_record, query_frame is not None)
        validation = validate_manifest(manifest, out, verify_files=True)
        write_json(out / 'input-manifest.json', manifest)
        run['artifact_export_seconds'] = time.monotonic() - t0
        if viewer:
            from pi4.visualization import save_visualization
            t0 = time.monotonic()
            save_visualization(out / 'viewer', frames, cpu_prediction, metadata, viewer=True)
            run['native_viewer_export_seconds'] = time.monotonic() - t0
        run.update(status='complete', total_seconds=time.monotonic() - start, validation=validation,
                   raw_arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in raw.items()})
        write_json(out / 'inference.json', run)
        return run
    except Exception as error:
        run.update(status='failed', error_type=type(error).__name__, error=str(error),
                   total_seconds=time.monotonic() - start)
        write_json(out / 'inference.json', run)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'out', 'checkpoint', 'repo']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--revision', required=True, help='Checkpoint snapshot revision')
    parser.add_argument('--checkpoint-sha256', help='Require this preflight-pinned checkpoint content hash')
    parser.add_argument('--code-revision', required=True, help='Exact clean official Git commit')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--max-size', type=int, default=512)
    parser.add_argument('--query-frame', type=int, default=0)
    parser.add_argument('--geometry-only', action='store_true')
    parser.add_argument('--warm-repeats', type=int, default=1)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = vars(parser.parse_args())
    if args.pop('geometry_only'):
        args['query_frame'] = None
    print(json.dumps(infer(**args), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
