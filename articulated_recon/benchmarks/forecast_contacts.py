"""Conservative contact hypotheses from generated source RGB and fitted motion.

No simulator, GT contact, geometry correction, or first/last grasp fallback.
Source-view SAM3 candidates are framewise detections, not tracked hand identities.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import file_sha256, load_manifest, validate_manifest
from artirecon.scene_sequence import load_scene, validate_scene
from benchmarks.strict_forecast import validate_forecast
from benchmarks.source_geometry import intervals
from benchmarks.forecast_geometry import merge_inputs


def default_config():
    return dict(schema_version=1, object_id='block', object_prompt='wooden block on patterned mat',
        hand_prompt='hand', source_view=0, minimum_score=.3, duplicate_iou=.85, minimum_mask_pixels=12,
        proximity_object_diagonals=.15, hand_ambiguity_object_diagonals=.05,
        minimum_duration_s=.25, minimum_motion_object_diagonals=.5,
        moving_step_object_diagonals=.04, minimum_motion_cosine=.3, minimum_coupled_steps=3,
        minimum_lift_m=.01, maximum_before_support_gap_s=.5,
        support_window_s_min=.4, support_minimum_observations=3,
        support_stationarity_object_diagonals=.1, support_distant_hand_object_diagonals=.75,
        support_clearance_m=.012, allow_declared_initial_support_prior=True, allow_terminal_hold=True)


def validate_config(config):
    if set(config) != set(default_config()) or config['schema_version'] != 1 or config['source_view'] != 0:
        raise ValueError('Use the complete saved source-view forecast contact configuration')
    for name in ['object_id', 'object_prompt', 'hand_prompt']:
        if not isinstance(config[name], str) or not config[name].strip(): raise ValueError('Missing contact prompt/ID')
    for name in ['minimum_mask_pixels', 'minimum_coupled_steps', 'support_minimum_observations']:
        if type(config[name]) is not int or config[name] < 1: raise ValueError('Invalid positive integer: '+name)
    if config['support_minimum_observations'] < 2: raise ValueError('Stationarity requires multiple images')
    for name in ['allow_declared_initial_support_prior', 'allow_terminal_hold']:
        if type(config[name]) is not bool: raise ValueError('Contact policy flags must be boolean: '+name)
    excluded = {'schema_version', 'source_view', 'object_id', 'object_prompt', 'hand_prompt',
                'minimum_mask_pixels', 'minimum_coupled_steps', 'support_minimum_observations',
                'allow_declared_initial_support_prior', 'allow_terminal_hold'}
    for name in set(config)-excluded:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError('Invalid positive contact threshold: '+name)
    for name in ['minimum_score', 'duplicate_iou', 'minimum_motion_cosine']:
        if config[name] > 1: raise ValueError('Threshold above one: '+name)


def mask_evidence(masks, times, config):
    """Unique object + nearest unambiguous hand; no missing-frame propagation."""
    shape = tuple(int(x) for x in masks['shape']); prompts = list(masks['prompts'])
    if len(shape) != 2 or min(shape) <= 0 or not np.array_equal(masks['timestamps'], times):
        raise ValueError('Masks must match forecast timestamps and image grid')
    if any(p not in prompts for p in [config['object_prompt'], config['hand_prompt']]):
        raise ValueError('Missing object/hand prompt in generated mask evidence')
    t, v, p, score, packed = [np.asarray(masks[n]) for n in ['t', 'v', 'p', 'score', 'mask']]
    n = len(t)
    if (any(a.shape != (n,) for a in [v, p, score]) or packed.shape != (n, (np.prod(shape)+7)//8)
            or packed.dtype != np.uint8 or any(a.dtype.kind not in 'iu' for a in [t, v, p])
            or not np.isfinite(score).all() or np.any((score < 0) | (score > 1))
            or np.any((t < 0) | (t >= len(times))) or np.any((p < 0) | (p >= len(prompts)))
            or np.any(v != 0)):
        raise ValueError('Use the separately exported source-only forecast mask packet')
    expected_known = np.zeros((len(times), 1), bool); expected_known[0, 0] = True
    if not np.array_equal(masks['observed_mask'], expected_known) or masks['observed_mask'].dtype.kind != 'b':
        raise ValueError('Only initial source image may be marked observed')
    if not np.array_equal(masks['source_frame_indices'], np.r_[0, np.full(len(times)-1, -1)]):
        raise ValueError('Generated future must retain source frame index -1')
    def unique(rows):
        selected = []
        for row in sorted(rows, key=lambda r: (-float(score[r]), int(r))):
            mask = np.unpackbits(packed[row])[:np.prod(shape)].reshape(shape).astype(bool)
            if mask.sum() < config['minimum_mask_pixels']: continue
            if any((mask & old).sum()/max(1, (mask | old).sum()) >= config['duplicate_iou'] for _, old in selected): continue
            selected.append((int(row), mask))
        return selected
    frames = len(times)
    centers, hands = np.zeros((frames, 2)), np.zeros((frames, 2))
    gaps, diagonals = np.full(frames, np.nan), np.zeros(frames)
    object_known, hand_known, reports = np.zeros(frames, bool), np.zeros(frames, bool), []
    for frame in range(frames):
        eligible = (t == frame) & (score >= config['minimum_score'])
        objects = unique(np.flatnonzero(eligible & (p == prompts.index(config['object_prompt']))))
        hand_candidates = unique(np.flatnonzero(eligible & (p == prompts.index(config['hand_prompt']))))
        row = dict(frame=frame, object_candidates=[r for r, _ in objects], hand_candidates=[r for r, _ in hand_candidates],
                   selected_object_row=None, selected_hand_row=None, status='missing_or_ambiguous_object')
        if len(objects) == 1:
            object_row, mask = objects[0]; xy = np.argwhere(mask)[:, ::-1]
            centers[frame] = xy.mean(0); diagonals[frame] = max(1., float(np.linalg.norm(np.ptp(xy, axis=0))))
            object_known[frame] = True; row['selected_object_row'] = object_row
            alternatives = sorted((float(distance_transform_edt(~hand)[mask].min()), hand_row,
                                   np.argwhere(hand)[:, ::-1].mean(0)) for hand_row, hand in hand_candidates)
            if alternatives and (len(alternatives) == 1 or alternatives[1][0]-alternatives[0][0]
                                 >= config['hand_ambiguity_object_diagonals']*diagonals[frame]):
                gaps[frame], hand_row, hands[frame] = alternatives[0]
                hand_known[frame] = True; row.update(selected_hand_row=hand_row, status='unique_object_and_nearest_hand')
            else: row['status'] = 'missing_or_ambiguous_hand'
        reports.append(row)
    return dict(object_centers=centers, hand_centers=hands, gaps=gaps, diagonals=diagonals,
                object_known=object_known, hand_known=hand_known, valid=object_known & hand_known, association=reports)


def infer(scene, evidence, config, *, initial_support_prior=False, artifact_ids=None):
    validate_config(config); validate_scene(scene, require_metric_scale=True)
    times = np.asarray(scene['timestamps'], float); count = len(times)
    if count < 2 or scene.get('forecast', {}).get('observed_until_s') != times[0]:
        raise ValueError('Contact inference requires a strict forecast shared scene')
    objects = [o for o in scene['objects'] if o['id'] == config['object_id']]
    if len(objects) != 1 or objects[0]['geometry']['type'] != 'box' or scene['support'] is None:
        raise ValueError('Current forecast contacts require the fitted block and estimated table')
    obj = objects[0]; observed = np.asarray(obj['observed'], bool); predicted = np.asarray(obj.get('prediction_supported'))
    if (predicted.shape != (count,) or predicted.dtype.kind != 'b' or predicted[0]
            or not observed[0] or observed[1:].any()):
        raise ValueError('Keep sole observed initial pose separate from supported future predictions')
    for name in ['gaps', 'diagonals', 'valid', 'object_known', 'hand_known']:
        if np.asarray(evidence[name]).shape != (count,): raise ValueError('Evidence length mismatch: '+name)
    for name in ['object_centers', 'hand_centers']:
        if np.asarray(evidence[name]).shape != (count, 2): raise ValueError('Evidence center shape mismatch')
    supported = observed | predicted
    known = supported & evidence['valid'] & np.isfinite(evidence['gaps']) & (evidence['diagonals'] > 0)
    near = known & (evidence['gaps'] <= config['proximity_object_diagonals']*evidence['diagonals'])
    far = known & (evidence['gaps'] >= config['support_distant_hand_object_diagonals']*evidence['diagonals'])
    poses = np.asarray(obj['T_world_object']); normal = np.asarray(scene['support']['normal_world'])
    half = np.asarray(obj['geometry']['half_extents_m']); heights = poses[:, :3, 3] @ normal
    extent = np.sum(np.abs(np.einsum('i,tij->tj', normal, poses[:, :3, :3]))*half, axis=1)
    clearance = heights + scene['support']['offset_m'] - extent
    cadence = float(np.median(np.diff(times)))
    window = max(config['support_window_s_min'], (config['support_minimum_observations']-1)*cadence)
    before_gap = max(config['maximum_before_support_gap_s'], 2*cadence)
    # Any contiguous stationary window may support its endpoints. This avoids
    # incorrectly requiring observations beyond the end of the forecast.
    rgb_support = np.zeros(count, bool); samples = config['support_minimum_observations']
    for first in range(count-samples+1):
        ids = np.arange(first, first+samples)
        if not far[ids].all() or times[ids[-1]]-times[ids[0]] > window+1e-9: continue
        xy = evidence['object_centers'][ids]
        spread = float(np.max(np.linalg.norm(xy-xy.mean(0), axis=1)))
        if spread <= config['support_stationarity_object_diagonals']*float(np.median(evidence['diagonals'][ids])):
            rgb_support[ids] = True
    support = rgb_support & (np.abs(clearance) <= config['support_clearance_m']) & supported
    initial_prior_used = bool(initial_support_prior and config['allow_declared_initial_support_prior']
        and supported[0] and evidence['object_known'][0] and abs(clearance[0]) <= config['support_clearance_m'])
    if initial_prior_used: support[0] = True
    provenance = dict(method='Generated source RGB proximity/co-motion + fitted lift/support; uncertain contact hypothesis',
        artifact_ids=list(artifact_ids or ['forecast_scene', 'forecast_sam3_masks']),
        confidence_status='uncalibrated', origin='model_prediction', generated=True)
    contacts, candidates = [], []
    for first, last in intervals(near):
        recent = np.flatnonzero(support & (times <= times[first]) & (times[first]-times <= before_gap+1e-9))
        after_support = bool(last+1 < count and support[last+1] and known[last+1])
        terminal_hold = bool(config['allow_terminal_hold'] and last == count-1 and known[last])
        diagonal = float(np.median(evidence['diagonals'][first:last+1]))
        path = evidence['object_centers'][first:last+1]; hand = evidence['hand_centers'][first:last+1]
        delta, hand_delta = np.diff(path, axis=0), np.diff(hand, axis=0)
        motion = np.linalg.norm(delta, axis=1); hand_motion = np.linalg.norm(hand_delta, axis=1)
        moving = (motion >= config['moving_step_object_diagonals']*diagonal) & (hand_motion > 1e-6)
        cosine = np.sum(delta[moving]*hand_delta[moving], axis=1)/(motion[moving]*hand_motion[moving])
        co_motion = float(np.median(cosine)) if len(cosine) else None
        displacement = float(np.linalg.norm(np.ptp(path, axis=0)))
        lift = float(clearance[first:last+1].max()-np.median(clearance[recent])) if len(recent) else None
        reasons = []
        if times[last]-times[first] < config['minimum_duration_s']: reasons.append('proximity too brief')
        if not len(recent): reasons.append('no recent support evidence')
        if not after_support and not terminal_hold: reasons.append('first released frame lacks stationary separated supported evidence')
        if displacement < config['minimum_motion_object_diagonals']*diagonal: reasons.append('insufficient object image motion')
        if len(cosine) < config['minimum_coupled_steps'] or co_motion is None or co_motion < config['minimum_motion_cosine']:
            reasons.append('insufficient hand/object co-motion')
        if lift is None or lift < config['minimum_lift_m']: reasons.append('estimated lift not established; could be pushing')
        candidates.append(dict(start_frame=first, end_frame=last, start_s=float(times[first]), end_s=float(times[last]),
            duration_s=float(times[last]-times[first]), object_displacement_px=displacement,
            coupled_steps=len(cosine), median_motion_cosine=co_motion, estimated_lift_m=lift,
            first_release_supported=after_support, terminal_hold=terminal_hold,
            release_status='release_unobserved_beyond_horizon' if terminal_hold else ('supported_release_hypothesis' if after_support else 'unsupported_or_missing_release'),
            accepted=not reasons, reasons=reasons))
        if not reasons:
            contacts.append(dict(id=f'forecast-grasp-{len(contacts)}', a=obj['id'], b='world', kind='grasp',
                start_frame=first, end_frame=last, status='hypothesized', confidence=None, provenance=provenance,
                terminal_hold=terminal_hold,
                release_status='release_unobserved_beyond_horizon' if terminal_hold else 'supported_release_hypothesis'))
    for first, last in intervals(support):
        contacts.append(dict(id=f'forecast-support-{first}', a=obj['id'], b=scene['support']['id'], kind='support',
            start_frame=first, end_frame=last, status='hypothesized', confidence=None,
            provenance={**provenance, 'method':'Stationary source masks + separated hand + estimated bottom near plane; t0 may use declared initial support prior',
                        'includes_initial_support_prior':initial_prior_used and first == 0}))
    for first, last in intervals(~known):
        contacts.append(dict(id=f'forecast-contact-unknown-{first}', a=obj['id'], b='world', kind='touch',
            start_frame=first, end_frame=last, status='unknown', confidence=None,
            provenance={**provenance, 'method':'Missing, ambiguous or unsupported predicted hand/object evidence'}))
    output = deepcopy(scene)
    old = deepcopy(output['contacts'])
    output['contacts'] = [c for c in old if obj['id'] not in {c['a'], c['b']}] + contacts
    output['forecast']['contact_state'] = 'Numerical hypotheses from generated source masks and estimated rigid motion; not force closure'
    report = dict(status='estimated', config=config, candidates=candidates,
        accepted_intervals=sum(c['kind']=='grasp' for c in contacts),
        terminal_hold_hypotheses=sum(c['kind']=='grasp' and c.get('terminal_hold',False) for c in contacts),
        missing_evidence_frames=np.flatnonzero(~known).tolist(),
        effective_support_window_s=window, effective_before_support_gap_s=before_gap, median_prediction_dt_s=cadence,
        rgb_stationary_support=rgb_support.tolist(), final_support=support.tolist(), bottom_clearance_m=clearance.tolist(),
        initial_support_prior_used=initial_prior_used, initial_support_prior_supplied=bool(initial_support_prior),
        raw_contacts=old, geometry_changed=False, missing_intervals_merged=False, forced_grasp=False,
        simulator_used=False, GT_used=False, confidence_status='uncalibrated',
        limitations=['Generated views are predictions, not independent observations.',
            'Nearest hand candidates are not temporal hand identities; mask centroids change with articulation.',
            'Estimated depth/plane/shape can bias lift and clearance.', 'A contact hypothesis does not establish force closure.'])
    output['forecast_contact_estimation'] = dict(config=config, filled_missing_frames=False, simulator_feedback_used=False)
    validate_scene(output, require_metric_scale=True)
    return output, report


def merge_contact_inputs(scene_manifest, mask_manifest, condition_manifest, mask_path):
    """Alias a distinct source-only terminal mask without modifying its producer.

    All-view scene fitting and the independently computed source-view subset
    historically use the same artifact ID. Their differing files must remain
    separate evidence. Shared RGB/weights are deduplicated without renaming.
    """
    merged = merge_inputs([scene_manifest, condition_manifest])
    mask_manifest = Path(mask_manifest).resolve()
    source = json.loads(mask_manifest.read_text())
    report = load_manifest(mask_manifest, [('masks', mask_path)])
    if source['condition']!='rgb_only' or source['task_mode']!='forecast' or source['observed_until_s']!=merged['observed_until_s']:
        raise ValueError('Source mask input must share the strict forecast condition/cutoff')
    active = set(report['method_artifact_ids'])
    records = []
    for original in source['artifacts']:
        if original['id'] not in active: continue
        record = deepcopy(original)
        if 'path' in record: record['path'] = str((mask_manifest.parent/record['path']).resolve())
        records.append(record)
    before = {a['id']:a for a in merged['artifacts']}
    aliases = {}
    for record in records:
        old = before.get(record['id'])
        if old is None or {k:v for k,v in old.items() if k!='path'} == {k:v for k,v in record.items() if k!='path'}: continue
        if record['kind']!='masks' or record.get('path')!=str(Path(mask_path).resolve()):
            raise ValueError('Conflicting nonterminal forecast mask ancestor: '+record['id'])
        alias = 'forecast_contact_source_masks'
        if alias in before or alias in {a['id'] for a in records}:
            raise ValueError('Source mask namespace already exists; use an unmodified fitted scene')
        aliases[record['id']] = alias
    for record in records:
        record['id'] = aliases.get(record['id'],record['id'])
        record['parent_ids'] = [aliases.get(n,n) for n in record['parent_ids']]
        old = before.get(record['id'])
        if old is not None:
            if {k:v for k,v in old.items() if k!='path'} != {k:v for k,v in record.items() if k!='path'}:
                raise ValueError('Conflicting renamed forecast mask ancestor: '+record['id'])
        else:
            merged['artifacts'].append(record); before[record['id']] = record
    merged['method_inputs'] = list(dict.fromkeys(merged['method_inputs']+[aliases.get(n,n) for n in source['method_inputs']]))
    root_hashes = {a['sha256'] for a in merged['artifacts'] if a['origin']=='observed_rgb'}
    if len(root_hashes)!=1: raise ValueError('Contact masks and fitted scene must share the same initial RGB')
    validate_manifest(merged, verify_files=True)
    return merged, {aliases.get(n,n) for n in active}, dict(artifact_id_aliases=aliases,
        source_manifest=str(mask_manifest), source_manifest_sha256=file_sha256(mask_manifest),
        interpretation='Only conflicting terminal source-mask IDs are aliased; file bytes, parents and shared observation ancestry are unchanged')


def produce(*, scene_bundle, masks, condition, config_path, output):
    scene_bundle, masks, condition, output = map(lambda p: Path(p).resolve(), (scene_bundle, masks, condition, output))
    if output.exists(): raise FileExistsError(output)
    prep, _ = validate_forecast(condition)
    scene_path = scene_bundle/'scene-sequence.json'; mask_path = masks/'masks.npz'
    scene_report = load_manifest(scene_bundle/'input-manifest.json', [('object_geometry', scene_path)])
    mask_report = load_manifest(masks/'input-manifest.json', [('masks', mask_path)])
    manifest, active_masks, alias_report = merge_contact_inputs(scene_bundle/'input-manifest.json',
        masks/'input-manifest.json', condition/'input-manifest.json', mask_path)
    active_scene = set(scene_report['method_artifact_ids'])
    records = {a['id']: a for a in manifest['artifacts']}
    scene_ids = [n for n in sorted(active_scene) if records[n]['kind']=='object_geometry' and records[n].get('path')==str(scene_path)]
    mask_ids = [n for n in sorted(active_masks) if records[n]['kind']=='masks' and records[n].get('path')==str(mask_path)]
    if not scene_ids or len(mask_ids)!=1 or any(records[n].get('view_role')=='helper' for n in active_masks):
        raise ValueError('Use a bound fitted scene and independently exported source-only forecast masks')
    config = json.loads(Path(config_path).read_text()); validate_config(config)
    scene = load_scene(scene_path, require_metric_scale=True)
    with np.load(condition/'condition.npz', allow_pickle=False) as z:
        if not np.array_equal(scene['timestamps'], z['timestamps']): raise ValueError('Scene and prediction horizon differ')
    with np.load(mask_path, allow_pickle=False) as z: evidence = mask_evidence(z, np.asarray(scene['timestamps']), config)
    initial = load_scene(prep['scene_path'], require_metric_scale=True)
    declared = initial.get('initial_support_prior', {}).get('used') is True
    result, report = infer(scene, evidence, config, initial_support_prior=declared, artifact_ids=scene_ids+mask_ids)
    report['association'] = evidence['association']
    report['source_mask_merge'] = alias_report
    output.mkdir(parents=True)
    result_path = output/'scene-sequence.json'; result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    artifact = dict(id='forecast_contact_phases', kind='object_geometry', origin='model_prediction',
        parent_ids=list(dict.fromkeys(scene_ids+mask_ids+['forecast_camera_requests'])), path='scene-sequence.json',
        sha256=file_sha256(result_path), method='Generated source RGB + fitted motion contact hypotheses; no geometry change')
    if artifact['id'] in records: raise ValueError('Existing forecast contact output must not be overwritten')
    manifest['artifacts'].append(artifact); manifest['method_inputs'].append(artifact['id'])
    report['input_protocol'] = validate_manifest(manifest, output, verify_files=True)
    report['producer_sha256'] = file_sha256(__file__); report['config_sha256'] = file_sha256(config_path)
    (output/'input-manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    (output/'contact-config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
    (output/'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='mode', required=True)
    config = sub.add_parser('config'); config.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('run')
    for name in ['scene-bundle', 'masks', 'condition', 'config-path', 'output']:
        run.add_argument('--'+name, type=Path, required=True)
    args = vars(parser.parse_args()); mode = args.pop('mode')
    if mode == 'config':
        with args['output'].open('x') as stream: json.dump(default_config(), stream, indent=2)
    else: print(json.dumps(produce(**args), indent=2))


if __name__ == '__main__': main()
