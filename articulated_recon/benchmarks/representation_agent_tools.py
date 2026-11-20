"""Small offline scene interface copied into each frozen agent workspace.

Only source/model prediction consistency is measured. Unknown/occluded pixels
are excluded, never treated as free space. Rendering assumes pinhole K and is
approximate when the frontend reports a nonzero dense-ray pinhole residual.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image

try:
    from scene_sequence import validate_scene
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from artirecon.scene_sequence import validate_scene


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(root):
    root = Path(root).resolve()
    frozen = json.loads((root / 'frozen-inputs.json').read_text())
    for name, expected in frozen['files'].items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Frozen input is not a local ordinary file: ' + name)
        if digest(path) != expected:
            raise ValueError('Frozen input changed: ' + name)
    allowed = set(frozen['files']) | {'frozen-inputs.json','execution-receipt.json'}
    for path in root.rglob('*'):
        name = str(path.relative_to(root))
        if path.is_symlink() or (path.is_file() and name not in allowed and not name.startswith('output/')):
            raise ValueError('Unlisted/nonlocal packet file: ' + name)
    packet = json.loads((root / 'packet.json').read_text())
    with np.load(root / 'observations.npz', allow_pickle=False) as z:
        obs = {k: z[k].copy() for k in z.files}
    return packet, obs


def validate_export(scene, packet):
    counts = validate_scene(scene)
    if not np.array_equal(scene['timestamps'], packet['timestamps']):
        raise ValueError('Export timestamps must equal selected packet timestamps exactly')
    if scene['coordinate_frame'] != packet['coordinate_frame']:
        raise ValueError('Export must retain the packet coordinate frame')
    if scene['metric_scale']['status'] != packet['metric_scale_status']:
        raise ValueError('Export must retain the predicted/unresolved metric-scale status')
    if not scene['objects'] or packet['object_id'] not in {o['id'] for o in scene['objects']}:
        raise ValueError('Export must include the packet task object')
    for obj in scene['objects']:
        if obj['geometry'].get('uri') is not None:
            raise ValueError('Export meshes must use inline vertices_m/faces, not external URIs')
    # Contact is a source hypothesis, never verified contact truth in this task.
    if any(c['status'] == 'observed' for c in scene['contacts']):
        raise ValueError('Source-only contact must remain unknown or explicitly hypothesized')
    return counts


def geometry_arrays(geometry):
    kind = geometry['type']
    if kind == 'mesh':
        v, f = np.asarray(geometry['vertices_m'], float), np.asarray(geometry['faces'], int)
        return v @ np.asarray(geometry.get('local_linear', np.eye(3))).T, f
    if kind == 'box':
        v = np.asarray([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                        [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], float)
        f = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
             [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
        return v * geometry['half_extents_m'], np.asarray(f)
    if kind == 'cylinder':
        n, radius, height = 32, geometry['radius_m'], geometry['height_m']
        theta = np.arange(n) * 2*np.pi/n
        ring = np.column_stack([radius*np.cos(theta), radius*np.sin(theta)])
        v = np.concatenate([np.column_stack([ring, np.full(n, -height/2)]),
                            np.column_stack([ring, np.full(n, height/2)]), [[0,0,-height/2],[0,0,height/2]]])
        f = []
        for i in range(n):
            j = (i+1) % n
            f.extend([[i,j,n+j], [i,n+j,n+i], [2*n,j,i], [2*n+1,n+i,n+j]])
        return v, np.asarray(f)
    if kind == 'sphere':
        n, rings = 24, 12
        theta, phi = np.meshgrid(np.arange(n)*2*np.pi/n, np.linspace(0, np.pi, rings+1))
        v = geometry['radius_m'] * np.stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)], axis=-1).reshape(-1,3)
        f = []
        for r in range(rings):
            for i in range(n):
                a, b = r*n+i, r*n+(i+1)%n
                f.extend([[a,b,b+n],[a,b+n,a+n]])
        return v, np.asarray(f)
    raise ValueError('Unsupported geometry')


def raster_depth(vertices, faces, k, c2w, size):
    """Deterministic triangle Z buffer; pixel-centre samples, no hidden fill."""
    height, width = size
    camera = (np.asarray(vertices) - c2w[:3, 3]) @ c2w[:3, :3]
    projected = camera @ k.T
    uv = projected[:, :2] / np.where(projected[:, 2:] != 0, projected[:, 2:], 1)
    zbuffer = np.full((height, width), np.inf)
    for face in faces:
        xyz, tri = camera[face], uv[face]
        if (xyz[:, 2] <= 1e-6).any():
            continue
        x0, y0 = np.maximum(np.floor(tri.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(tri.max(axis=0)).astype(int), [width-1,height-1])
        if x1 < x0 or y1 < y0:
            continue
        a, b, c = tri
        denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(denominator) < 1e-12:
            continue
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
        u = ((b[1]-c[1])*(xx-c[0]) + (c[0]-b[0])*(yy-c[1]))/denominator
        v = ((c[1]-a[1])*(xx-c[0]) + (a[0]-c[0])*(yy-c[1]))/denominator
        w = 1-u-v
        inside = (u >= -1e-8) & (v >= -1e-8) & (w >= -1e-8)
        inverse = u/xyz[0,2] + v/xyz[1,2] + w/xyz[2,2]
        depth = np.divide(1., inverse, out=np.full_like(inverse,np.inf), where=inverse > 0)
        patch = zbuffer[y0:y1+1, x0:x1+1]
        np.minimum(patch, np.where(inside, depth, np.inf), out=patch)
    return np.where(np.isfinite(zbuffer), zbuffer, 0).astype(np.float32)


def agreement(rendered, reference_depth, reference_mask, mask_known, occlusion, valid):
    """Score only evidence that is available; null != empty/zero/error-free."""
    if not mask_known:
        return dict(status='unknown_object_mask', visible_mask_iou=None,
                    predicted_depth_agreement_median_m=None, depth_overlap_pixels=0)
    known_space = ~occlusion
    reference = reference_mask & known_space
    prediction = (rendered > 0) & known_space
    if not reference.any():
        return dict(status='no_visible_object_reference', visible_mask_iou=None,
                    predicted_depth_agreement_median_m=None, depth_overlap_pixels=0)
    overlap = prediction & reference & valid
    union = prediction | reference
    return dict(status='measured_source_prediction_agreement',
        visible_mask_iou=float((prediction & reference).sum()/union.sum()),
        predicted_depth_agreement_median_m=float(np.median(np.abs(rendered[overlap]-reference_depth[overlap]))) if overlap.any() else None,
        depth_overlap_pixels=int(overlap.sum()), visible_reference_pixels=int(reference.sum()),
        excluded_occluded_pixels=int(occlusion.sum()))


def check_scene(root, scene_path, output):
    root, scene_path, output = map(lambda p: Path(p).resolve(), (root, scene_path, output))
    packet, obs = load_inputs(root)
    scene = json.loads(scene_path.read_text())
    counts = validate_export(scene, packet)
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to(root / 'output'):
        raise ValueError('Source tool outputs must be under workspace/output')
    output.mkdir(parents=True)
    objects = [(obj, *geometry_arrays(obj['geometry'])) for obj in scene['objects']]
    target = packet['object_id']
    rows = []
    for t, rgb in enumerate(obs['rgb'][:, 0]):
        k, c = obs['K'][t, 0], obs['c2w'][t, 0]
        depths = {}
        for obj, vertices, faces in objects:
            pose = np.asarray(obj['T_world_object'][t])
            world = vertices @ pose[:3, :3].T + pose[:3, 3]
            depths[obj['id']] = (raster_depth(world, faces, k, c, rgb.shape[:2]) if obs['camera_projection_valid'][t,0]
                                 else np.zeros(rgb.shape[:2],np.float32))
        depth = depths[target]
        row = agreement(depth, obs['depth_m'][t,0], obs['object_masks'][t,0],
                        bool(obs['object_mask_known'][t,0]), obs['occlusion_mask'][t,0], obs['depth_valid'][t,0])
        if not obs['camera_projection_valid'][t,0]:
            row.update(status='camera_projection_unavailable',visible_mask_iou=None,
                       predicted_depth_agreement_median_m=None,depth_overlap_pixels=0)
        row.update(selected_frame=t, native_frame=int(obs['native_frame_ids'][t]),
                   timestamp_s=float(obs['timestamps'][t]),
                   exported_observed=next(o for o in scene['objects'] if o['id']==target)['observed'][t])
        if 'K_pinhole_rmse_px' in obs:
            residual=float(obs['K_pinhole_rmse_px'][t,0])
            row['predicted_camera_pinhole_rmse_px'] = residual if np.isfinite(residual) else None
        rows.append(row)
        overlay = rgb.copy()
        visible = (depth > 0) & ~obs['occlusion_mask'][t,0]
        overlay[visible] = (.45*rgb[visible] + .55*np.array([240,160,50])).astype(np.uint8)
        unknown = obs['occlusion_mask'][t,0]
        overlay[unknown] = (.5*rgb[unknown] + .5*np.array([130,50,180])).astype(np.uint8)
        Image.fromarray(np.concatenate([rgb, overlay], axis=1)).save(output/f'{t:03d}-rgb-overlay.png')
        np.save(output/f'{t:03d}-rendered-depth.npy', depth)
    ious = [x['visible_mask_iou'] for x in rows if x['visible_mask_iou'] is not None]
    depths = [x['predicted_depth_agreement_median_m'] for x in rows if x['predicted_depth_agreement_median_m'] is not None]
    report = dict(status='source_check_complete', scene_sha256=digest(scene_path), counts=counts,
        source_mask_iou_mean=float(np.mean(ious)) if ious else None,
        predicted_depth_agreement_median_m=float(np.median(depths)) if depths else None,
        gt_geometry_accuracy=None, contact_accuracy=None, execution_success=None,
        GT_used=False, simulator_used=False, per_frame=rows,
        interpretation='Agreement with source model predictions, not GT. Occlusion is unknown. '
                       'Pinhole projection may approximate frontend dense rays. No actor mesh/physical check is performed.')
    (output/'source-check.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect')
    validate = sub.add_parser('validate')
    validate.add_argument('--scene', type=Path, required=True)
    check = sub.add_parser('check')
    check.add_argument('--scene', type=Path, required=True)
    check.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    started = time.monotonic()
    packet, obs = load_inputs(ROOT)
    if a.command == 'inspect':
        result = dict(packet=packet, arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in obs.items()})
    elif a.command == 'validate':
        result = validate_export(json.loads(a.scene.read_text()), packet)
    else:
        result = check_scene(ROOT, a.scene, a.output)
    log = ROOT/'output'/'source-tool-events.jsonl'
    log.parent.mkdir(exist_ok=True)
    with log.open('a') as stream:
        stream.write(json.dumps(dict(command=a.command, elapsed_seconds=time.monotonic()-started,
                                    scene_sha256=digest(a.scene) if hasattr(a,'scene') else None))+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
