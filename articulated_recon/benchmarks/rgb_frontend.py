"""Extract only RGB, then estimate metric depth and cameras without dataset K.

This is a source-camera geometry baseline, not an object reconstruction or a
robot policy. Static-camera poses are an explicit capture assumption. Unknown
robot registration blocks execution in the input protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import validate_manifest


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_rgb(rgb, timestamps):
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected one source camera: uint8 RGB [T,H,W,3]')
    if any(size == 0 for size in rgb.shape) or timestamps.shape != (len(rgb),):
        raise ValueError('Timestamp/image count mismatch')
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Timestamps must be finite and strictly increasing')


def read_rgb_source(source, task_mode, camera_index=None):
    """Read native-resolution pixels, never dataset depth/calibration fields.

    Video timestamps use frame index / advertised constant FPS. Forecast mode
    decodes only its first frame; hashing the container does not expose future
    pixels to inference. NPZ loading necessarily decompresses its RGB array,
    but only selected observations are subsequently exported to the method.
    """
    suffix = source.suffix.lower()
    if suffix == '.npz':
        with np.load(source, allow_pickle=False) as data:
            rgb = data['rgb'].copy()
            if rgb.ndim == 5:
                if camera_index is None:
                    raise ValueError('Select exactly one source camera with --camera-index')
                if not isinstance(camera_index, int) or not 0 <= camera_index < rgb.shape[1]:
                    raise ValueError('Source camera index is outside the available views')
                rgb = rgb[:, camera_index]
            elif camera_index is not None:
                raise ValueError('Camera index is only valid for multiview input arrays')
            ts = data['timestamps'].astype(float).copy()
        return rgb, ts, {'source_format': 'npz', 'timestamp_source': 'timestamps array',
                         'decoded_frames': len(rgb)}
    if camera_index is not None:
        raise ValueError('Camera index is only valid for multiview input arrays')
    if suffix in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}:
        from PIL import Image
        with Image.open(source) as image:
            rgb = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()[None]
        return rgb, np.array([0.]), {'source_format': 'image', 'timestamp_source': 'single image t=0',
                                     'decoded_frames': 1}
    if suffix not in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}:
        raise ValueError('Source must be an RGB NPZ, supported image, or video')
    import cv2
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open source video: {source}')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Video must provide a finite positive frame rate')
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if task_mode == 'forecast':
                break
        if not frames:
            raise ValueError('Source video contains no decodable RGB frames')
        rgb = np.stack(frames)
        ts = np.arange(len(rgb), dtype=float) / fps
        return rgb, ts, {'source_format': 'video', 'source_fps': fps,
                         'timestamp_source': 'frame index / container FPS (constant-rate assumption)',
                         'decoded_frames': len(rgb)}
    finally:
        capture.release()


def prepare(source, out, task_mode='reconstruction', camera_index=None, frames=None):
    source, out = Path(source).resolve(), Path(out).resolve()
    if task_mode not in {'reconstruction', 'forecast'}:
        raise ValueError('task_mode must be reconstruction or forecast')
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite input bundle: {out}')
    rgb, ts, source_info = read_rgb_source(source, task_mode, camera_index)
    validate_rgb(rgb, ts)
    ids = np.arange(len(rgb))
    if task_mode == 'forecast':
        ids = ids[:1]
    elif frames is not None:
        if type(frames) is not int or frames < 2:
            raise ValueError('Use at least two reconstruction frames')
        ids = np.unique(np.rint(np.linspace(0, len(rgb) - 1, min(frames, len(rgb)))).astype(int))
    rgb, ts = rgb[ids], ts[ids]
    out.mkdir(parents=True)
    np.savez_compressed(out / 'rgb.npz', rgb=rgb, timestamps=ts)
    from PIL import Image
    Image.fromarray(rgb[0]).save(out / 'first-frame.png')
    manifest = {
        'schema_version': 1, 'condition': 'rgb_only', 'task_mode': task_mode,
        'method_inputs': ['source_rgb'],
        'artifacts': [{'id': 'source_rgb', 'kind': 'rgb', 'origin': 'observed_rgb',
                       'generated': False, 'parent_ids': [], 'path': 'rgb.npz',
                       'sha256': sha256(out / 'rgb.npz'), 'time_range_s': [float(ts[0]), float(ts[-1])]}],
        'metric_scale': {'status': 'unresolved', 'artifact_ids': [], 'description': 'RGB input alone'},
        'robot_registration': {'status': 'unresolved', 'artifact_ids': [],
                               'description': 'No target robot base observed in the human demonstration'},
    }
    if task_mode == 'forecast':
        manifest['observed_until_s'] = float(ts[0])
    report = validate_manifest(manifest, out, verify_files=True)
    (out / 'input-manifest.json').write_text(json.dumps(manifest, indent=2))
    (out / 'extraction-audit.json').write_text(json.dumps({
        'container_sha256': sha256(source), 'source_camera_index': camera_index,
        **source_info, 'exported_resolution_hw': list(rgb.shape[1:3]),
        'selected_frame_indices': ids.tolist(), 'exported_arrays': ['rgb', 'timestamps'],
        'copied_depth': False, 'copied_camera_calibration': False,
        'copied_scene_geometry': False, 'validation': report,
        'note': 'Only RGB/timing are read. No resizing, calibration, depth, or geometry extraction. '
                'The caller identifies the source as recorded observation; filenames do not establish this provenance.',
    }, indent=2))
    return manifest


def rank_confidence(uncertainty, valid):
    """Larger is better; rank only, never a calibrated probability."""
    from scipy.stats import rankdata
    result = np.zeros(uncertainty.shape, dtype=np.float32)
    valid = valid & np.isfinite(uncertainty) & (uncertainty >= 0)
    if valid.any():
        result[valid] = rankdata(-uncertainty[valid], method='average') / valid.sum()
    return result


def load_source_bundle(bundle):
    """Bind actual consumed pixels and timestamps to the active RGB declaration."""
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'input-manifest.json').read_text())
    validate_manifest(manifest, bundle, verify_files=True)
    if manifest['condition'] != 'rgb_only' or manifest['method_inputs'] != ['source_rgb']:
        raise ValueError('This frontend requires an isolated active source_rgb input only')
    source = next((a for a in manifest['artifacts'] if a['id'] == 'source_rgb'), None)
    if source is None or source['kind'] != 'rgb' or source['origin'] != 'observed_rgb':
        raise ValueError('source_rgb must be the observed RGB root, not a generated view')
    with np.load(bundle / source['path'], allow_pickle=False) as data:
        if set(data.files) != {'rgb', 'timestamps'}:
            raise ValueError('Unexpected fields in isolated RGB bundle')
        rgb, timestamps = data['rgb'].copy(), data['timestamps'].copy()
    validate_rgb(rgb, timestamps)
    if source['time_range_s'] != [float(timestamps[0]), float(timestamps[-1])]:
        raise ValueError('Declared source time range differs from actual RGB timestamps')
    if manifest['task_mode'] == 'forecast' and timestamps[-1] > manifest['observed_until_s']:
        raise ValueError('RGB bundle contains observations after the forecast cutoff')
    return manifest, source, rgb, timestamps


def validate_intrinsics(k):
    if (k.shape != (3, 3) or not np.isfinite(k).all() or
            k[0, 0] <= 0 or k[1, 1] <= 0 or
            not np.allclose(k[2], [0, 0, 1], atol=1e-6)):
        raise ValueError('UniDepth returned invalid pinhole intrinsics')


def infer_unidepth(bundle, out, checkpoint, revision, device='cuda', resolution_level=9):
    bundle, out = Path(bundle).resolve(), Path(out).resolve()
    manifest, source, rgb, timestamps = load_source_bundle(bundle)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir() or not (checkpoint / 'config.json').is_file():
        raise ValueError('An existing local checkpoint directory with config.json is required')
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError('Declare the local checkpoint revision')
    if type(resolution_level) is not int or not 0 <= resolution_level < 10:
        raise ValueError('resolution_level must be an integer in [0, 9]')
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite predictions: {out}')
    import torch
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole

    out.mkdir(parents=True)
    start = time.monotonic()
    model = UniDepthV2.from_pretrained(str(checkpoint), local_files_only=True).to(device).eval()
    model.resolution_level = resolution_level
    model.interpolation_mode = 'bilinear'
    load_seconds = time.monotonic() - start
    depths, uncertainties, predicted_intrinsics = [], [], []
    frame_seconds = []
    predicted_k = None
    with torch.inference_mode():
        for i, frame in enumerate(rgb):
            t0 = time.monotonic()
            image = torch.from_numpy(frame.copy()).permute(2, 0, 1)
            # Frame zero predicts its camera; later frames reuse that estimated
            # intrinsic matrix under the explicitly declared fixed-camera setup.
            # UniDepth's BatchCamera retains and mutates the passed camera while
            # padding/resizing. Never reuse it or pass a tensor sharing our K.
            if predicted_k is None:
                pred = model.infer(image)
                predicted_k = pred['intrinsics'][0].detach().float().cpu().numpy().copy()
                validate_intrinsics(predicted_k)
            else:
                camera = Pinhole(K=torch.from_numpy(predicted_k.copy())[None].to(device))
                pred = model.infer(image, camera)
            depth = pred['depth'][0, 0].detach().float().cpu().numpy()
            uncertainty = pred['confidence'][0, 0].detach().float().cpu().numpy()
            if depth.shape != frame.shape[:2] or uncertainty.shape != depth.shape:
                raise ValueError('UniDepth output grid differs from the RGB input')
            if not (np.isfinite(depth) & (depth > 0)).any():
                raise ValueError(f'UniDepth produced no usable depth for source frame {i}')
            depths.append(depth)
            uncertainties.append(uncertainty)
            # Upstream still returns the camera-head prediction when supplied
            # camera rays drive depth. This is diagnostic, not the lifting K.
            frame_k = pred['intrinsics'][0].detach().float().cpu().numpy().copy()
            validate_intrinsics(frame_k)
            predicted_intrinsics.append(frame_k)
            frame_seconds.append(time.monotonic() - t0)
            print(f'UniDepth source frame {i + 1}/{len(rgb)}', flush=True)
    depth, uncertainty = np.stack(depths), np.stack(uncertainties)
    valid = np.isfinite(depth) & (depth > 0)
    ranks = np.stack([rank_confidence(u, v) for u, v in zip(uncertainty, valid)])
    np.savez_compressed(out / 'geometry.npz', rgb=rgb[:, None], depth_m=depth[:, None],
                        K=predicted_k[None], c2w=np.eye(4)[None], timestamps=timestamps,
                        confidence_rank=ranks[:, None], raw_uncertainty=uncertainty[:, None],
                        valid=valid[:, None], per_frame_output_intrinsics=np.stack(predicted_intrinsics))
    # Keep source bytes separate, including when the caller moves this output.
    source_bytes = (bundle / source['path']).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != source['sha256']:
        raise ValueError('Source RGB changed during inference; refusing completion')
    (out / 'rgb.npz').write_bytes(source_bytes)
    source['path'] = 'rgb.npz'
    manifest['artifacts'].append({
        'id': 'unidepth_weights', 'kind': 'model_prior', 'origin': 'prior',
        'prior_type': 'learned_model_weights', 'scope': 'global', 'parent_ids': [],
        'model_name': 'UniDepthV2', 'revision': revision,
        'revision_source': 'caller-declared local snapshot',
        'config_sha256': sha256(checkpoint / 'config.json')})
    manifest['artifacts'].append({
        'id': 'predicted_geometry', 'kind': 'depth', 'origin': 'model_prediction',
        'parent_ids': ['source_rgb', 'unidepth_weights'], 'path': 'geometry.npz',
        'sha256': sha256(out / 'geometry.npz')})
    manifest['artifacts'].append({
        'id': 'predicted_cameras', 'kind': 'cameras', 'origin': 'model_prediction',
        'parent_ids': ['source_rgb', 'unidepth_weights'], 'path': 'geometry.npz',
        'sha256': sha256(out / 'geometry.npz'),
        'assumption': 'Fixed source camera; first-frame estimated intrinsics; source camera is coordinate-frame origin'})
    manifest['method_inputs'] += ['predicted_geometry', 'predicted_cameras']
    manifest['metric_scale'] = {'status': 'estimated', 'artifact_ids': ['predicted_geometry'],
                                'description': 'Learned monocular metric scale; no reference alignment or scale fitting'}
    validation = validate_manifest(manifest, out, verify_files=True)
    (out / 'input-manifest.json').write_text(json.dumps(manifest, indent=2))
    (out / 'inference.json').write_text(json.dumps({
        'status': 'complete', 'checkpoint': str(checkpoint), 'checkpoint_revision': revision,
        'resolution_level': resolution_level, 'device': device,
        'model_load_seconds': load_seconds, 'frame_inference_seconds': frame_seconds,
        'total_seconds': time.monotonic() - start, 'frames': len(rgb),
        'confidence': 'API confidence is predicted error; confidence_rank is inverse error ordering, not probability',
        'known_intrinsics_supplied': False, 'measured_depth_used': False,
        'camera_conditioning': 'First frame unconditioned; later frames use a fresh copy of its predicted K',
        'per_frame_output_intrinsics': 'Diagnostic camera-head predictions, not the rays used after frame zero',
        'fixed_camera_assumption': True,
        'camera_pose': 'Identity by coordinate convention, not measured world/robot extrinsics',
        'validation': validation,
    }, indent=2))
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--source', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--task-mode', choices=['reconstruction', 'forecast'], default='reconstruction')
    prep.add_argument('--camera-index', type=int)
    prep.add_argument('--frames', type=int)
    infer = subs.add_parser('infer-unidepth')
    infer.add_argument('--bundle', type=Path, required=True)
    infer.add_argument('--out', type=Path, required=True)
    infer.add_argument('--checkpoint', type=Path, required=True, help='Existing local checkpoint, no automatic download')
    infer.add_argument('--revision', required=True, help='Checkpoint snapshot/commit identifier')
    infer.add_argument('--device', default='cuda')
    infer.add_argument('--resolution-level', type=int, default=9)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.source, args.out, args.task_mode, args.camera_index, args.frames)
    else:
        result = infer_unidepth(args.bundle, args.out, args.checkpoint, args.revision, args.device, args.resolution_level)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
