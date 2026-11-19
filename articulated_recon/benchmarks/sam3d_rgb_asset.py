"""Fresh author SAM3D assets from recorded RGB and learned SAM3 candidates.

No depth or camera arguments are supplied to SAM3D. Its internal MoGe pose/scale
are predictions, not measured calibration or metric ground truth. Ambiguous
source instance selection stops before inference and saves candidate overlays.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import load_manifest, file_sha256, validate_manifest


def select_candidate(masks, scores, table_masks, score_margin=.15):
    """Fixed RGB-only rule: table convex-footprint gate then clear score winner.

    Table hull fills holes occluded by objects. Raw mask overlap is reported too;
    it is not required because a correct tabletop mask may exclude the object.
    This is a task-object selection heuristic, not measured relevance accuracy.
    """
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull, QhullError
    masks = np.asarray(masks, bool)
    scores = np.asarray(scores, float)
    if masks.ndim != 3 or scores.shape != (len(masks),):
        raise ValueError('Expected candidate masks [N,H,W] and one score each')
    if not np.isfinite(scores).all() or not 0 <= score_margin <= 1:
        raise ValueError('Invalid scores or declared score margin')
    table_masks = np.asarray(table_masks, bool)
    table = table_masks.any(0) if len(table_masks) else np.zeros(masks.shape[1:], bool)
    hull_mask = np.zeros_like(table)
    ys, xs = np.nonzero(table)
    if len(xs) >= 3:
        xy = np.column_stack([xs, ys])
        try:
            hull = ConvexHull(xy)
            image = Image.new('1', (table.shape[1], table.shape[0]))
            ImageDraw.Draw(image).polygon([tuple(p) for p in xy[hull.vertices]], fill=1)
            hull_mask = np.asarray(image, bool)
        except QhullError:
            pass
    candidates = []
    for index, (mask, score) in enumerate(zip(masks, scores)):
        y, x = np.nonzero(mask)
        inside = bool(len(x) and hull_mask[int(round(y.mean())), int(round(x.mean()))])
        candidates.append(dict(index=index, score=float(score), area_pixels=int(mask.sum()),
                               centroid_inside_table_hull=inside,
                               raw_table_overlap_fraction=float((mask & table).sum() / max(mask.sum(), 1))))
    eligible = [c for c in candidates if c['area_pixels'] > 0]
    gate_used = bool(hull_mask.any())
    if gate_used:
        eligible = [c for c in eligible if c['centroid_inside_table_hull']]
    eligible.sort(key=lambda c: (-c['score'], c['index']))
    selected = None
    reason = 'No eligible source object candidate'
    if len(eligible) == 1:
        selected, reason = eligible[0]['index'], 'Only eligible text-prompted candidate'
    elif len(eligible) > 1:
        if eligible[0]['score'] - eligible[1]['score'] >= score_margin:
            selected, reason = eligible[0]['index'], 'Top candidate exceeds the predeclared score margin'
        else:
            reason = 'Multiple plausible candidates have insufficient score separation'
    return dict(selected_index=selected, reason=reason, candidates=candidates,
                table_hull_gate_used=gate_used, score_margin=score_margin,
                rule='Centroid inside predicted table convex footprint when available; unique candidate or score margin')


def prepare_asset(mask_bundle, output, object_prompt, frame_index=0, score_margin=.15):
    from PIL import Image, ImageDraw
    mask_bundle, output = Path(mask_bundle).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = load_manifest(mask_bundle / 'input-manifest.json', [('masks', mask_bundle / 'masks.npz')])
    if report['condition'] != 'rgb_only':
        raise ValueError('SAM3D source asset requires RGB-only ancestry')
    manifest = json.loads((mask_bundle / 'input-manifest.json').read_text())
    source = next(a for a in manifest['artifacts'] if a['id'] == 'source_rgb')
    if source['origin'] != 'observed_rgb' or source['id'] not in report['method_artifact_ids']:
        raise ValueError('Source must be active recorded RGB')
    rgb_path = (mask_bundle / source['path']).resolve()
    with np.load(rgb_path, allow_pickle=False) as z:
        if set(z.files) != {'rgb', 'timestamps'}:
            raise ValueError('Source must be an isolated RGB/timestamps bundle')
        source_rgb = z['rgb']
        if not 0 <= frame_index < len(source_rgb):
            raise ValueError('Source frame index is outside the available RGB')
        rgb, timestamp = source_rgb[frame_index].copy(), float(z['timestamps'][frame_index])
    with np.load(mask_bundle / 'masks.npz', allow_pickle=False) as z:
        prompts = z['prompts'].tolist()
        p = prompts.index(object_prompt)
        table_p = prompts.index('tabletop')
        height, width = z['shape']
        if rgb.shape != (height, width, 3):
            raise ValueError('SAM3 candidates and source RGB differ in resolution')
        rows = np.flatnonzero((z['t'] == frame_index) & (z['p'] == p))
        table_rows = np.flatnonzero((z['t'] == frame_index) & (z['p'] == table_p))
        decode = lambda r: np.unpackbits(z['mask'][r], axis=1)[:, :height * width].reshape(len(r), height, width).astype(bool)
        masks, table_masks, scores = decode(rows), decode(table_rows), z['score'][rows].copy()
    selection = select_candidate(masks, scores, table_masks, score_margin)
    selection.update(frame_index=frame_index, timestamp_s=timestamp, object_prompt=object_prompt,
                     candidate_row_ids=rows.tolist(), source_rgb_sha256=source['sha256'])
    output.mkdir(parents=True)
    for index, mask in enumerate(masks):
        pixels = rgb.copy()
        pixels[mask] = (.5 * pixels[mask] + .5 * np.array([30, 230, 90])).astype(np.uint8)
        overlay = Image.fromarray(pixels)
        ImageDraw.Draw(overlay).text((8, 8), f'Candidate {int(rows[index])}; score {scores[index]:.3f}', fill='white', stroke_width=2, stroke_fill='black')
        overlay.save(output / f'candidate-{int(rows[index])}.png')
    (output / 'selection.json').write_text(json.dumps(selection, indent=2))
    chosen = selection['selected_index']
    if chosen is None:
        (output / 'execution.json').write_text(json.dumps(dict(status='ambiguous_input', selection=selection), indent=2))
        return None
    Image.fromarray(rgb).save(output / 'source.png')
    Image.fromarray(masks[chosen].astype(np.uint8) * 255).save(output / 'mask.png')
    for artifact in manifest['artifacts']:
        if 'path' in artifact:
            artifact['path'] = str((mask_bundle / artifact['path']).resolve())
    # Selection is evidence derived from the source masks, never an oracle mask.
    manifest['artifacts'].append(dict(id='selected_object_mask', kind='masks', origin='model_prediction',
        parent_ids=['source_rgb', 'sam3_masks'], path='mask.png', sha256=file_sha256(output / 'mask.png'),
        selection=selection))
    manifest['method_inputs'].append('selected_object_mask')
    validate_manifest(manifest, output, verify_files=True)
    (output / 'input-manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def generate_asset(folder, wrapper, wrapper_sha256, seed=0):
    folder, wrapper = Path(folder).resolve(), Path(wrapper).resolve()
    if (folder / 'object.ply').exists():
        raise FileExistsError('Refusing to reuse a previous generated asset')
    report = load_manifest(folder / 'input-manifest.json', [('masks', folder / 'mask.png')])
    if report['condition'] != 'rgb_only' or file_sha256(wrapper) != wrapper_sha256:
        raise ValueError('SAM3D wrapper or input condition changed')
    manifest = json.loads((folder / 'input-manifest.json').read_text())
    selection = json.loads((folder / 'selection.json').read_text())
    mask_record = next(a for a in manifest['artifacts'] if a['id'] == 'selected_object_mask')
    if selection != mask_record['selection']:
        raise ValueError('Source candidate selection changed after preparation')
    source_record = next(a for a in manifest['artifacts'] if a['id'] == 'source_rgb')
    from PIL import Image
    with np.load(source_record['path'], allow_pickle=False) as z:
        expected = z['rgb'][selection['frame_index']]
    if not np.array_equal(np.asarray(Image.open(folder / 'source.png').convert('RGB')), expected):
        raise ValueError('SAM3D RGB is not the selected original source frame')
    module_name = '_author_sam3d_rgb_asset'
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, wrapper)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    module = sys.modules[module_name]
    argv = [str(wrapper), '--image', str(folder / 'source.png'), '--mask', str(folder / 'mask.png'),
            '--out', str(folder / 'object.ply'), '--seed', str(seed), '--no_preview']
    receipt = dict(status='running', argv=argv, wrapper_sha256=wrapper_sha256,
                   supplied_depth=False, supplied_intrinsics=False, supplied_extrinsics=False,
                   scale_status='Internal MoGe estimate; not reference-calibrated metric geometry', seed=seed)
    start = time.monotonic(); previous_argv = sys.argv
    try:
        sys.argv = argv
        module.main()
        pose = json.loads((folder / 'object.json').read_text())
        if pose['depth_source'] == 'user_depth' or pose['n_faces'] <= 0 or pose['n_vertices'] <= 0:
            raise ValueError('SAM3D did not produce a fresh RGB-only mesh')
        if not np.isfinite(np.asarray(pose['T_obj2cam_opencv'])).all():
            raise ValueError('SAM3D predicted a nonfinite object pose')
        manifest['artifacts'].append(dict(id='sam3d_weights', kind='model_prior', origin='prior',
            prior_type='learned_model_weights', scope='global', parent_ids=[],
            model_name='SAM3D Objects with internal MoGe', wrapper_sha256=wrapper_sha256))
        for aid, name in [('sam3d_canonical_mesh', 'object.ply'), ('sam3d_camera_mesh', 'object_cam.ply'), ('sam3d_pose', 'object.json')]:
            path = folder / name
            if path.stat().st_size == 0:
                raise ValueError(f'Empty SAM3D output: {name}')
            manifest['artifacts'].append(dict(id=aid, kind='object_geometry', origin='model_prediction',
                parent_ids=['source_rgb', 'selected_object_mask', 'sam3d_weights'], path=name, sha256=file_sha256(path)))
            manifest['method_inputs'].append(aid)
        validation = validate_manifest(manifest, folder, verify_files=True)
        (folder / 'input-manifest.json').write_text(json.dumps(manifest, indent=2))
        receipt.update(status='complete', pose_summary=pose, validation=validation,
                       completion_meaning='Generated mesh and predicted pose; geometry/contact accuracy not yet evaluated')
    except Exception as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        sys.argv = previous_argv
        receipt['seconds'] = time.monotonic() - start
        (folder / 'execution.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--mask-bundle', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--object-prompt', required=True)
    prep.add_argument('--frame-index', type=int, default=0)
    prep.add_argument('--score-margin', type=float, default=.15)
    run = sub.add_parser('generate')
    run.add_argument('--folders', type=Path, nargs='+', required=True)
    run.add_argument('--wrapper', type=Path, required=True)
    run.add_argument('--wrapper-sha256', required=True)
    run.add_argument('--seed', type=int, default=0)
    args = vars(parser.parse_args()); command = args.pop('command')
    if command == 'prepare':
        prepare_asset(**args)
    else:
        folders = args.pop('folders')
        for folder in folders:
            generate_asset(folder, **args)


if __name__ == '__main__':
    main()
