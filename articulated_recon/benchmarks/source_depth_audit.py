"""Compare source-predicted depth structure without reference geometry or physics.

Identical RGB masks/pixels and timestamps are used. Neither a lower plane residual
nor a tighter object cloud proves more accurate depth; no cross-method alignment
or metric rescaling is performed by this diagnostic.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import file_sha256
from benchmarks import source_geometry as base
from benchmarks import static_depth_calibration as calibration


def object_statistics(geometry, masks):
    depth, k, times = base.validate_geometry(geometry)
    candidates = base.MaskCandidates(masks, len(times), depth.shape[-2:], times)
    selected, association = base.associate_masks(candidates, len(times), base.default_config())
    inverse_k = np.linalg.inv(k); rows = []
    for frame, chosen in enumerate(selected):
        mask = candidates.mask(chosen) if chosen is not None else np.zeros(depth.shape[-2:], bool)
        mask &= ~candidates.union(candidates.rows(frame, 'hand', .3))
        mask &= distance_transform_edt(mask) > 1
        mask &= np.asarray(geometry['valid'])[frame, 0] & np.isfinite(depth[frame]) & (depth[frame] > 0)
        yy, xx = np.where(mask); values = depth[frame, yy, xx]
        row = dict(frame=frame, timestamp_s=float(times[frame]), selected_candidate_row=chosen, visible_valid_points=len(xx),
                   camera_z_quantiles_10_50_90_m=None, camera_point_span_05_95_m=None, centered_cloud_rms_m=None)
        if len(xx) >= base.default_config()['min_object_points']:
            points = (np.c_[xx, yy, np.ones(len(xx))] @ inverse_k.T) * values[:, None]
            row.update(camera_z_quantiles_10_50_90_m=np.quantile(values, [.1, .5, .9]).tolist(),
                       camera_point_span_05_95_m=np.diff(np.quantile(points, [.05, .95], axis=0), axis=0)[0].tolist(),
                       centered_cloud_rms_m=float(np.sqrt(np.mean(np.sum((points - np.median(points, axis=0))**2, axis=1)))))
        rows.append(row)
    medians = [r['camera_z_quantiles_10_50_90_m'][1] for r in rows if r['camera_z_quantiles_10_50_90_m'] is not None]
    widths = [r['camera_z_quantiles_10_50_90_m'][2] - r['camera_z_quantiles_10_50_90_m'][0] for r in rows if r['camera_z_quantiles_10_50_90_m'] is not None]
    return dict(rows=rows, confidence_available=base.confidence_available(geometry), confidence_selection_used=False,
                observed_frames=len(medians), frames=len(times), median_foreground_camera_z_m=float(np.median(medians)) if medians else None,
                mean_foreground_depth_10_90_span_m=float(np.mean(widths)) if widths else None,
                association=association, GT_accuracy_measured=False, simulator_accessed=False)


def compare(*, unidepth_bundle, da3_bundle, mask_bundle, config_path, output):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    da3_bundle = Path(da3_bundle).resolve(); receipt = json.loads((da3_bundle / 'inference.json').read_text())
    if receipt.get('status') != 'complete' or receipt.get('confidence_available') is not False:
        raise ValueError('Wait for validated complete DA3 Metric inference; its head has no confidence')
    configs = json.loads(Path(config_path).read_text()); calibration.validate_config(configs)
    mask_manifest = base._materialized_manifest(mask_bundle); masks_record = next(a for a in mask_manifest['artifacts'] if a['id'] == 'sam3_masks')
    mask_source = next(a for a in mask_manifest['artifacts'] if a['id'] == 'source_rgb')
    manifests = {name: base._materialized_manifest(folder) for name, folder in [('unidepth', unidepth_bundle), ('da3_metric', da3_bundle)]}
    output.mkdir(parents=True); reports = {}; camera = None; timestamps = None
    with np.load(masks_record['path'], allow_pickle=False) as masks:
        for name, manifest in manifests.items():
            records = {a['id']: a for a in manifest['artifacts']}
            if records['source_rgb']['sha256'] != mask_source['sha256']: raise ValueError('Depth/masks consumed different source RGB')
            if name == 'da3_metric' and records['predicted_geometry']['sha256'] != receipt.get('geometry_sha256'):
                raise ValueError('DA3 receipt does not bind the actual geometry')
            with np.load(records['predicted_geometry']['path'], allow_pickle=False) as data, np.load(records['source_rgb']['path'], allow_pickle=False) as rgb:
                if camera is None: camera, timestamps = data['K'].copy(), data['timestamps'].copy()
                elif not np.array_equal(camera, data['K']) or not np.array_equal(timestamps, data['timestamps']):
                    raise ValueError('This comparison requires identical estimated source K and actual timestamps')
                if name == 'da3_metric' and base.confidence_available(data): raise ValueError('Metric head must not advertise confidence')
                raw = object_statistics(data, masks)
                corrected, _, table = calibration.calibrate(data, masks, rgb['rgb'], configs)
                corrected_statistics = object_statistics(corrected, masks) if table['status'] == 'accepted' else None
            reports[name] = dict(source_rgb_sha256=records['source_rgb']['sha256'], geometry_sha256=records['predicted_geometry']['sha256'],
                raw_objects=raw, static_calibration=table, calibrated_objects=corrected_statistics)
            (output / (name + '.json')).write_text(json.dumps(reports[name], indent=2, allow_nan=False))
    summary = dict(status='complete', estimated_K_identical=True, timestamps_identical=True, source_RGB_identical=True,
        no_reference_alignment=True, GT_accuracy_measured=False, simulator_accessed=False, raw_predictions_changed=False,
        producer_sha256=file_sha256(__file__), config_sha256=file_sha256(config_path), depth_frontends={})
    for name, report in reports.items():
        table = report['static_calibration']; rows = table['rows']
        summary['depth_frontends'][name] = dict(confidence_available=report['raw_objects']['confidence_available'],
            raw_foreground_camera_z_median_m=report['raw_objects']['median_foreground_camera_z_m'],
            raw_foreground_depth_span_mean_m=report['raw_objects']['mean_foreground_depth_10_90_span_m'],
            raw_observed_frames=report['raw_objects']['observed_frames'], static_calibration_status=table['status'],
            raw_background_plane_absolute_median_mean_m=float(np.mean([r['raw_heldout_absolute_median_m'] for r in rows])),
            calibrated_background_plane_absolute_median_mean_m=float(np.mean([r['corrected_heldout_absolute_median_m'] for r in rows])),
            frame_scale_range=[min(r['scale'] for r in rows), max(r['scale'] for r in rows)], rejected_frames=table['rejected_frames'])
    summary['limitations'] = ['Both use the same UniDepth-estimated focal anchor; neither is an independently calibrated metric-depth oracle.',
        'Object cloud dispersion includes real shape, visibility and prediction errors; smaller is not necessarily better.',
        'Background consistency is measured against each method\'s own frame-zero plane, not the physical table.']
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('unidepth-bundle', 'da3-bundle', 'mask-bundle', 'config-path', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    print(json.dumps(compare(**vars(parser.parse_args())), indent=2))
