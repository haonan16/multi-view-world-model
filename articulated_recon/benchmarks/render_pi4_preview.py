"""Render source RGB, Pi4 depth and native predicted points from one frozen run.

Only geometry.npz and inference.json are read. The third panel is a virtual
camera rendering of observed colored points, not generated RGB or a completed
surface. Point identity, sequence-wide clipping and framing stay fixed. Video
frames hold the nearest preceding prediction; geometry is never interpolated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_predictions(run_dir):
    run_dir = Path(run_dir).resolve()
    paths = [run_dir / 'geometry.npz', run_dir / 'inference.json']
    bindings = {str(path): sha256(path) for path in paths}
    inference = json.loads(paths[1].read_text())
    if inference.get('status') != 'complete' or inference.get('geometry_schema') != 'pi4_dynamic_camera_v1':
        raise ValueError('Require a completed native Pi4 geometry export')
    for key in ['measured_depth_used', 'known_intrinsics_supplied', 'reference_alignment']:
        if inference.get(key) is not False:
            raise ValueError('Preview requires declared source-only predictions: ' + key)
    with np.load(paths[0], allow_pickle=False) as data:
        rgb = data['rgb'].copy()
        points = data['points_world'].copy()
        depth = data['depth_m'].copy()
        cameras = data['c2w'].copy()
        timestamps = data['timestamps'].copy()
        source_ids = data['source_frame_indices'].copy()
    if rgb.ndim != 5 or rgb.shape[1] != 1 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected source RGB [T,1,H,W,3] uint8')
    count, _, height, width, _ = rgb.shape
    if (count < 2 or height < 2 or width < 2 or points.shape != rgb.shape
            or depth.shape != rgb.shape[:-1] or cameras.shape != (count, 1, 4, 4)):
        raise ValueError('Malformed or mismatched native Pi4 geometry arrays')
    if (timestamps.shape != (count,) or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0) or source_ids.shape != (count,)
            or source_ids.dtype.kind not in 'iu' or np.any(source_ids < 0)
            or np.any(np.diff(source_ids.astype(np.float64)) <= 0)):
        raise ValueError('Invalid source timestamps or frame indices')
    if (inference.get('source_frame_indices') != source_ids.tolist()
            or inference.get('timestamps_s') != timestamps.tolist()
            or inference.get('frames') != count):
        raise ValueError('Inference record does not bind the selected frame/time sequence')
    cameras = cameras[:, 0]
    rotation = cameras[:, :3, :3]
    if (not np.isfinite(cameras).all()
            or not np.allclose(cameras[:, 3], [0, 0, 0, 1], atol=1e-4)
            or not np.allclose(rotation @ rotation.swapaxes(-1, -2), np.eye(3), atol=2e-3)
            or not np.allclose(np.linalg.det(rotation), 1, atol=2e-3)):
        raise ValueError('Malformed predicted camera-to-world transforms')
    return dict(rgb=rgb[:, 0], points=points[:, 0], depth=depth[:, 0], cameras=cameras,
                timestamps=timestamps, source_ids=source_ids, bindings=bindings)


def sample_indices(height, width, maximum=60000):
    """One deterministic image-grid sampling, reused for every prediction."""
    if type(maximum) is not int or maximum < 1:
        raise ValueError('maximum must be a positive integer')
    return np.linspace(0, height * width - 1, min(maximum, height * width), dtype=np.int64)


def display_schedule(timestamps, fps=30.):
    """Zero-order hold on actual timestamp offsets; infer only final hold length."""
    timestamps = np.asarray(timestamps, dtype=float)
    if (len(timestamps) < 2 or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0) or not np.isfinite(fps) or fps <= 0):
        raise ValueError('Require increasing finite timestamps and positive display FPS')
    relative = timestamps - timestamps[0]
    final_hold = float(np.median(np.diff(timestamps)))
    duration = float(relative[-1] + final_hold)
    times = np.arange(int(np.ceil(duration * fps - 1e-8)), dtype=float) / fps
    indices = np.searchsorted(relative, times + 1e-9, side='right') - 1
    return times, np.clip(indices, 0, len(timestamps) - 1), final_hold


def fixed_display_geometry(points, cameras, indices, percentile=0.5):
    """Choose display-only bounds in the first predicted camera's orientation."""
    if not 0 <= percentile < 10:
        raise ValueError('Clipping percentile must be in [0,10)')
    origin, basis = cameras[0, :3, 3], cameras[0, :3, :3]
    samples = points.reshape(len(points), -1, 3)[:, indices]
    reference = (samples.astype(np.float64) - origin) @ basis
    finite = np.isfinite(reference).all(-1)
    if not finite.any():
        raise ValueError('No finite sampled native world points')
    lower, upper = np.percentile(reference[finite], [percentile, 100 - percentile], axis=0)
    center = (lower + upper) / 2
    radius = max(float(np.linalg.norm(upper - lower) / 2), 1e-6)
    padding = max(radius * 1e-6, 1e-9)
    lower, upper = lower - padding, upper + padding
    valid = finite & (reference >= lower).all(-1) & (reference <= upper).all(-1)
    return dict(origin=origin, basis=basis, lower=lower, upper=upper,
                center=center, radius=radius, valid=valid, finite=finite)


def virtual_camera(display, angle_degrees, elevation_degrees=12., fov_degrees=55.):
    """Return an explicitly virtual c2w; the model geometry is unchanged."""
    angle, elevation, fov = np.radians([angle_degrees, elevation_degrees, fov_degrees])
    if not 10 <= fov_degrees <= 100 or not -80 <= elevation_degrees <= 80:
        raise ValueError('Unsupported orbit field of view or elevation')
    radius = display['radius']
    distance = radius / np.sin(fov / 2) * 1.2
    offset = distance * np.array([np.sin(angle) * np.cos(elevation),
                                  -np.sin(elevation), -np.cos(angle) * np.cos(elevation)])
    eye_reference = display['center'] + offset
    forward = -offset / np.linalg.norm(offset)
    right = np.cross(forward, [0., -1., 0.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, :3] = display['basis'] @ np.stack([right, down, forward], axis=1)
    c2w[:3, 3] = display['origin'] + display['basis'] @ eye_reference
    return c2w, max(1e-8, distance - radius * 1.5), distance + radius * 1.5


def project_points(points, c2w, width, height, fov_degrees=55., near=1e-6, far=np.inf):
    camera = (np.asarray(points, dtype=np.float64) - c2w[:3, 3]) @ c2w[:3, :3]
    valid = np.isfinite(camera).all(-1) & (camera[:, 2] > near) & (camera[:, 2] < far)
    ids = np.flatnonzero(valid)
    focal = 0.5 * height / np.tan(np.radians(fov_degrees) / 2)
    uv = camera[ids, :2] / camera[ids, 2:3] * focal + [(width - 1) / 2, (height - 1) / 2]
    in_image = ((uv >= [-0.5, -0.5]).all(-1) & (uv < [width - 0.5, height - 0.5]).all(-1))
    ids, uv = ids[in_image], uv[in_image]
    return ids, np.rint(uv).astype(np.int64), camera[ids, 2]


def rasterize_points(points, colors, c2w, width, height, fov_degrees=55., near=1e-6, far=np.inf):
    """One-pixel nearest-depth splats; no mesh, filling, smoothing or extrapolation."""
    image = np.full((height, width, 3), [15, 20, 28], dtype=np.uint8)
    ids, uv, depth = project_points(points, c2w, width, height, fov_degrees, near, far)
    if not len(ids):
        return image, 0
    pixels = uv[:, 1] * width + uv[:, 0]
    order = np.lexsort((ids, depth, pixels))
    sorted_pixels = pixels[order]
    front = order[np.r_[True, np.diff(sorted_pixels) != 0]]
    image.reshape(-1, 3)[pixels[front]] = colors[ids[front]]
    return image, len(front)


def fixed_depth_limits(depth):
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError('No finite positive predicted depth')
    lower, upper = np.percentile(depth[valid], [2, 98])
    return float(lower), float(max(upper, lower + max(abs(lower) * 1e-6, 1e-8)))


def depth_colors(depth, limits):
    valid = np.isfinite(depth) & (depth > 0)
    image = np.zeros((*depth.shape, 3), dtype=np.uint8)
    value = np.clip((depth[valid] - limits[0]) / (limits[1] - limits[0]), 0, 1)
    anchors = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]])
    xp = np.linspace(0, 1, len(anchors))
    image[valid] = np.stack([np.interp(value, xp, anchors[:, i]) for i in range(3)], axis=-1).astype(np.uint8)
    return image


def render(run_dir, output, fps=30., max_points=60000, orbit_degrees=100.,
           clip_percentile=0.5, show_source_indices=False):
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg

    if not np.isfinite(orbit_degrees) or not 0 <= orbit_degrees <= 360:
        raise ValueError('Orbit sweep must be in [0,360] degrees')
    data = load_predictions(run_dir)
    rgb, points, depth = data['rgb'], data['points'], data['depth']
    count, height, width, _ = rgb.shape
    ids = sample_indices(height, width, max_points)
    display = fixed_display_geometry(points, data['cameras'], ids, clip_percentile)
    limits = fixed_depth_limits(depth)
    output_times, frame_map, final_hold = display_schedule(data['timestamps'], fps)
    angles = np.linspace(-orbit_degrees / 2, orbit_degrees / 2, count)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'point-indices.npy').open('xb') as stream:
        np.save(stream, ids, allow_pickle=False)
    header = 36
    canvas_height = height + header + (height + header) % 2
    canvas_width = 3 * width + (3 * width) % 2
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 17)
    except OSError:
        font = ImageFont.load_default()
    writer = imageio_ffmpeg.write_frames(str(output / 'preview.mp4'), (canvas_width, canvas_height),
        fps=fps, codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', macro_block_size=1,
        output_params=['-crf', '18', '-movflags', '+faststart'])
    writer.send(None)
    rendered, statistics, virtual_cameras = {}, [], []
    try:
        for i in range(count):
            c2w, near, far = virtual_camera(display, angles[i])
            virtual_cameras.append(c2w.tolist())
            valid = display['valid'][i]
            sampled_points = points[i].reshape(-1, 3)[ids][valid]
            sampled_colors = rgb[i].reshape(-1, 3)[ids][valid]
            cloud, pixels = rasterize_points(sampled_points, sampled_colors, c2w, width, height,
                                            near=near, far=far)
            image = Image.new('RGB', (canvas_width, canvas_height), (15, 20, 28))
            draw = ImageDraw.Draw(image)
            labels = ['SOURCE RGB', 'PREDICTED DEPTH', 'POINTS / VIRTUAL ORBIT']
            if show_source_indices:
                labels[0] += f"  [{int(data['source_ids'][i])}]"
            for column, (panel, label) in enumerate(zip([rgb[i], depth_colors(depth[i], limits), cloud], labels)):
                image.paste(Image.fromarray(panel), (column * width, header))
                draw.text((column * width + 8, 9), label, font=font, fill=(240, 245, 250))
            rendered[i] = np.asarray(image).copy()
            statistics.append(dict(prediction_index=i, source_frame_index=int(data['source_ids'][i]),
                finite_sampled_points=int(display['finite'][i].sum()),
                inside_fixed_bounds=int(valid.sum()), colored_visible_pixels=int(pixels)))
            if i == count // 2:
                image.save(output / 'poster.jpg')
        for i in frame_map:
            writer.send(np.ascontiguousarray(rendered[int(i)]))
    finally:
        writer.close()
    for path, expected in data['bindings'].items():
        if sha256(path) != expected:
            raise ValueError('Prediction artifact changed while rendering: ' + path)
    intervals = np.diff(data['timestamps'])
    report = dict(schema='pi4-source-preview-v1', status='complete', input_hashes=data['bindings'],
        renderer_sha256=sha256(__file__), metric_scale='estimated by Pi4; no reference scale or alignment',
        panels=['recorded source RGB at model resolution', 'predicted local Z depth',
                'native predicted world points with source RGB colors, rendered from virtual orbit cameras'],
        reference_data_read=False, generated_rgb=False, geometry_interpolated=False,
        hidden_surfaces_completed=False, confidence_filter_applied=False,
        world_geometry_modified=False, point_size_pixels=1,
        source_timestamps_s=data['timestamps'].tolist(), source_frame_indices=data['source_ids'].tolist(),
        source_sampling_fps=float(1 / intervals[0]) if np.allclose(intervals, intervals[0]) else None,
        display_fps=float(fps), display_frame_count=len(frame_map), display_prediction_indices=frame_map.tolist(),
        display_times_relative_s=output_times.tolist(), final_frame_hold_seconds=final_hold,
        temporal_display='Hold preceding source prediction; repeat entire panel; no geometry or RGB interpolation',
        final_hold_policy='Median positive input timestamp interval',
        point_sampling=dict(rule='fixed evenly spaced flattened source-image indices for all frames',
            points_per_frame=len(ids), indices_file='point-indices.npy', indices_sha256=sha256(output / 'point-indices.npy')),
        clipping=dict(coordinate_frame='First predicted camera orientation; display-only',
            origin_world=display['origin'].tolist(), basis_world=display['basis'].tolist(),
            bounds_low=display['lower'].tolist(), bounds_high=display['upper'].tolist(),
            percentile_per_axis=clip_percentile, depth_color_limits_m=list(limits),
            depth_color_percentiles=[2,98], temporal_policy='One fixed range for the entire sequence'),
        virtual_orbit=dict(sweep_degrees=orbit_degrees, elevation_degrees=12., fov_degrees=55.,
            cameras_c2w=virtual_cameras, angles_degrees=angles.tolist(),
            camera_motion='Display-only orbit; held fixed within repeats of each predicted source frame'),
        per_frame=statistics, resolution=[canvas_width, canvas_height],
        video_sha256=sha256(output / 'preview.mp4'))
    with (output / 'render.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fps', type=float, default=30.)
    parser.add_argument('--max-points', type=int, default=60000)
    parser.add_argument('--orbit-degrees', type=float, default=100.)
    parser.add_argument('--clip-percentile', type=float, default=0.5)
    parser.add_argument('--show-source-indices', action='store_true')
    result = render(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'], display_frames=result['display_frame_count'],
                          source_fps=result['source_sampling_fps'], display_fps=result['display_fps'])))


if __name__ == '__main__':
    main()
