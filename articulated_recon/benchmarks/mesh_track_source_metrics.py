"""Common source-mask/depth agreement for frozen source mesh reconstructions.

These compare model outputs with model estimates, not GT geometry/contact.
Missing-association pose fallbacks are scored too and explicitly identified.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/'mano')]
from artirecon.input_protocol import file_sha256, load_manifest
from mano_recon.run_integrity import artifact_record


def measure(track, geometry, masks, prompt='pump bottle'):
    import trimesh
    from mano_recon.render_fast import render
    track, geometry, masks = map(lambda p: Path(p).resolve(), [track, geometry, masks])
    load_manifest(track/'input-manifest.json', [('depth', geometry/'geometry.npz'), ('masks', masks/'masks.npz'),
                                               ('object_geometry', track/'scene-sequence.json')])
    audit = json.loads((track/'execution.json').read_text())
    if audit['status'] != 'complete':
        raise ValueError('Track is not complete')
    for name, saved in audit['outputs'].items():
        if artifact_record(track/name) != saved:
            raise ValueError('Track output changed: '+name)
    with np.load(geometry/'geometry.npz', allow_pickle=False) as z:
        depth, k, c, times = [z[n].copy() for n in ['depth_m', 'K', 'c2w', 'timestamps']]
    with np.load(masks/'masks.npz', allow_pickle=False) as z:
        mask_data = {n:z[n].copy() for n in z.files}
    object_p = list(mask_data['prompts']).index(prompt)
    rows = []
    for t in range(len(times)):
        candidates = np.flatnonzero((mask_data['t'] == t) & (mask_data['p'] == object_p) & (mask_data['score'] >= .3))
        rows.append(int(candidates[0]) if len(candidates) == 1 else None)
    with np.load(track/'mesh-track.npz', allow_pickle=False) as z:
        ids, poses, fit_times, observed = [z[n].copy() for n in ['frame_ids', 'poses', 'timestamps', 'observed']]
    if not np.array_equal(times, mask_data['timestamps']) or not np.array_equal(times[ids], fit_times):
        raise ValueError('Native timestamps differ')
    mesh = trimesh.load(track/'fitted-mesh.ply', force='mesh', process=False)
    h, w = depth.shape[-2:]
    decode = lambda row: np.unpackbits(mask_data['mask'][row])[:h*w].reshape(h, w).astype(bool)
    hand_p = list(mask_data['prompts']).index('hand')
    values = []
    for j, native in enumerate(ids):
        row = rows[native]
        result = dict(source_frame=int(native), timestamp_s=float(times[native]),
                      fit_had_association=bool(observed[j]), reference_candidate=row)
        if row is None:
            result['status'] = 'ambiguous_source_reference'
            values.append(result)
            continue
        excluded = np.zeros((h, w), bool)
        for hand_row in np.flatnonzero((mask_data['t'] == native) & (mask_data['p'] == hand_p) & (mask_data['score'] >= .3)):
            excluded |= decode(hand_row)
        reference = decode(row) & ~excluded
        vertices = np.asarray(mesh.vertices) @ poses[j, :3, :3].T + poses[j, :3, 3]
        _, rendered_depth = render(vertices, mesh.faces, k[0], c[0], size=(h, w))
        prediction = (rendered_depth > 0) & ~excluded
        valid_depth = reference & prediction & np.isfinite(depth[native, 0]) & (depth[native, 0] > 0)
        union = (reference | prediction).sum()
        result.update(status='measured' if reference.any() else 'no_visible_reference',
            visible_reference_pixels=int(reference.sum()),
            visible_mask_iou=float((reference & prediction).sum()/union) if union else None,
            depth_overlap_pixels=int(valid_depth.sum()),
            predicted_depth_agreement_median_mm=float(np.median(np.abs(rendered_depth[valid_depth]-depth[native, 0][valid_depth]))*1000)
                if valid_depth.any() else None)
        values.append(result)
    good = [x for x in values if x['status'] == 'measured']
    ious = [x['visible_mask_iou'] for x in good]
    supported = [x['visible_mask_iou'] for x in good if x['fit_had_association']]
    agreement = [x['predicted_depth_agreement_median_mm'] for x in good if x['predicted_depth_agreement_median_mm'] is not None]
    return dict(track=str(track), association=audit.get('association_policy', 'author_depth'),
        source_frames_evaluated=len(good), source_frames_with_fit_association=int(observed.sum()),
        all_pose_mask_iou_mean=float(np.mean(ious)) if ious else None,
        supported_pose_mask_iou_mean=float(np.mean(supported)) if supported else None,
        predicted_depth_agreement_median_mm=float(np.median(agreement)) if agreement else None,
        per_frame=values, GT_used=False, simulator_used=False,
        interpretation='Agreement with SAM3 and UniDepth predictions. Missing-association fallback poses are included in all-pose scores; not GT geometry/contact accuracy.',
        shared_source_geometry_sha256=file_sha256(geometry/'geometry.npz'), shared_masks_sha256=file_sha256(masks/'masks.npz'),
        mesh_track_sha256=file_sha256(track/'mesh-track.npz'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track', action='append', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    reports = [measure(t, a.geometry, a.masks) for t in a.track]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(producer_sha256=file_sha256(__file__), comparisons=reports), indent=2, allow_nan=False))
    print(json.dumps(reports, indent=2), flush=True)
