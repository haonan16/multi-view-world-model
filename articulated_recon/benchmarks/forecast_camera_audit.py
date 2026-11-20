"""Check requested-camera response at the sole observed forecast timestamp.

Uses only the initial predicted depth and masks, never a recorded future frame.
These are internal consistency measurements, not ground-truth camera accuracy.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import file_sha256, load_manifest
from benchmarks.da3_helper_views import read_inputs
from benchmarks.helper_camera_audit import (compare_displacements, dynamic_prompt_ids,
    feature_matches, requested_pixels, transform_grid)


def run(helper, condition, output):
    helper, condition, output = map(lambda p: Path(p).resolve(), (helper, condition, output))
    if output.exists():
        raise FileExistsError(output)
    job = read_inputs(helper, condition)
    if job['mode'] != 'forecast':
        raise ValueError('This diagnostic requires the strict first-image forecast')
    prep = json.loads((condition/'preparation.json').read_text())
    geometry = Path(prep['geometry_manifest']).parent/'geometry.npz'
    scene_manifest_path = Path(prep['scene_manifest'])
    scene_manifest = json.loads(scene_manifest_path.read_text())
    report = load_manifest(scene_manifest_path)
    masks = [a for a in scene_manifest['artifacts'] if a['kind'] == 'masks'
             and a['id'] in report['method_artifact_ids']]
    if len(masks) != 1:
        raise ValueError('Require the sole initial scene mask ancestor')
    mask_path = (scene_manifest_path.parent/masks[0]['path']).resolve()
    with np.load(mask_path, allow_pickle=False) as z:
        shape = tuple(z['shape']); prompts = z['prompts'].tolist()
        dynamic = np.zeros(shape, bool)
        categories = dynamic_prompt_ids(prompts)
        for i in np.flatnonzero((z['t'] == 0) & np.isin(z['p'], categories)):
            dynamic |= np.unpackbits(z['mask'][i])[:np.prod(shape)].reshape(shape).astype(bool)
    with np.load(geometry, allow_pickle=False) as z:
        depth, valid = z['depth_m'][0, 0], z['valid'][0, 0]
    generation = json.loads((helper/'generation.json').read_text())
    if (generation.get('status') != 'complete' or
            generation.get('generated_sha256') != file_sha256(helper/'generated.npz') or
            generation.get('observations_sha256') != job['helper_sha256']):
        raise ValueError('Raw decoder packet does not match its completed generation receipt')
    with np.load(helper/'generated.npz', allow_pickle=False) as z:
        raw = z['rgb'].copy()
        if (raw.shape != job['rgb'].shape or raw.dtype != np.uint8 or
                any(not np.array_equal(z[name], job[key]) for name, key in
                    [('timestamps', 'timestamps'), ('K', 'K'), ('c2w', 'c2w'), ('observed_mask', 'observed_mask')]) or
                not np.array_equal(raw[:, 1:], job['rgb'][:, 1:]) or
                not np.array_equal(raw[1:, 0], job['rgb'][1:, 0])):
            raise ValueError('Raw generated packet differs from the authenticated forecast streams')
    rgb = job['rgb']; crop = prep['source_crop_xywh']; size = (rgb.shape[3], rgb.shape[2])
    depth = transform_grid(depth, crop, size)
    excluded = transform_grid(dynamic, crop, size, mask=True) > 0
    excluded = cv2.dilate(excluded.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    allowed = (transform_grid(valid, crop, size, mask=True) > 0) & ~excluded & np.isfinite(depth) & (depth > 0)
    rows, panels = [], []
    for v in range(4):
        target = raw[0, 0] if v == 0 else rgb[0, v]
        a, b, counts = feature_matches(rgb[0, 0], target, allowed)
        row = dict(view=v, time_s=float(job['timestamps'][0]), **counts,
                   role='source_decoder_control' if v == 0 else 'generated_helper')
        panel = cv2.resize(target, (640, 320), interpolation=cv2.INTER_LINEAR)
        if len(a) >= 8:
            d = cv2.remap(depth, a[:, 0].astype(np.float32)[None],
                          a[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
            expected, good = requested_pixels(a, d, job['K'][0], job['K'][v], job['c2w'][0], job['c2w'][v])
            a, b, expected = a[good], b[good], expected[good]
            if len(a) >= 8:
                row.update(status='measured', **compare_displacements(a, b, expected))
                for observed, predicted in zip(b[::max(1, len(b)//60)], expected[::max(1, len(b)//60)]):
                    if np.max(abs(predicted)) < 2000:
                        cv2.line(panel, tuple(np.rint(2*observed).astype(int)),
                                 tuple(np.rint(2*predicted).astype(int)), (230, 90, 190), 1, cv2.LINE_AA)
                        cv2.circle(panel, tuple(np.rint(2*observed).astype(int)), 3, (80, 240, 100), -1)
        row.setdefault('status', 'insufficient_static_matches')
        panel[:28] = 20
        cv2.putText(panel, 'Source decoder' if v == 0 else f'Generated view {v}',
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .6, (245, 245, 245), 1, cv2.LINE_AA)
        rows.append(row); panels.append(panel)
    output.mkdir(parents=True)
    grid = np.concatenate([np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)])
    cv2.imwrite(str(output/'initial-camera-response.png'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    result = dict(schema='strict-forecast-camera-audit-v1', frame_index=0, per_view=rows,
        source_sha256=job['observed']['sha256'], helper_sha256=job['helper_sha256'],
        initial_geometry_sha256=file_sha256(geometry), initial_masks_sha256=file_sha256(mask_path),
        future_recorded_rgb_used=False, sensor_depth_used=False, evaluator_used=False,
        interpretation='Flow fraction near zero is source-like; near one follows requested parallax under initial predicted depth/K.',
        limitations=['Initial predicted depth/K and masks may be wrong.',
                     'Homography-filtered features emphasize planar background.',
                     'No claim of accurate future cameras or object motion from this first-frame measurement.'])
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['helper', 'condition', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))
