from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artirecon.scene_sequence import scene_from_arrays, object_from_arrays
from benchmarks import forecast_contacts as phase


def inputs():
    count = 11; times = np.arange(count)/3
    poses = np.repeat(np.eye(4)[None], count, axis=0); poses[:, 2, 3] = .02
    poses[3:8, 2, 3] = [.04, .08, .1, .08, .04]
    provenance = dict(method='synthetic generated evidence', artifact_ids=['generated_source_forecast'], confidence_status='uncalibrated')
    obj = object_from_arrays(object_id='block', poses=poses, observed=[True]+[False]*(count-1),
        primitive=dict(type='box', half_extents_m=[.02]*3), provenance=provenance)
    obj['prediction_supported'] = [False]+[True]*(count-1)
    obj['prediction_provenance'] = dict(origin='model_prediction', generated=True,
        method='synthetic forecast fixture', artifact_ids=['generated_source_forecast'])
    scene = scene_from_arrays(timestamps=times,
        coordinate_frame=dict(name='source_camera', handedness='right', up_axis='+z', pose_convention='T_world_local'),
        metric_scale=dict(status='estimated', provenance=provenance), actors=[], objects=[obj],
        support=dict(id='table', normal_world=[0, 0, 1], offset_m=0, confidence=None, provenance=provenance),
        contacts=[], uncertainty=dict(status='unestimated'), task=dict(description='Pick and place block', target=None))
    scene['forecast'] = dict(observed_until_s=0., future_recorded_RGB_used=False)
    centers = np.c_[[0, 0, 0, 1, 6, 11, 16, 21, 21, 21, 21], np.zeros(count)]
    evidence = dict(object_centers=centers, hand_centers=centers+[0, 2], gaps=np.array([30, 30, 30, 1, 1, 1, 1, 1, 30, 30, 30]),
        diagonals=np.ones(count)*20, object_known=np.ones(count, bool), hand_known=np.ones(count, bool),
        valid=np.ones(count, bool), association=[])
    return scene, evidence


def test_cadence_aware_grasp_hypothesis_preserves_forecast_geometry_and_flags():
    scene, evidence = inputs(); before = deepcopy(scene)
    result, report = phase.infer(scene, evidence, phase.default_config())
    assert scene == before and result['objects'] == scene['objects'] and result['support'] == scene['support']
    grasp = [c for c in result['contacts'] if c['kind']=='grasp']
    assert [(c['start_frame'], c['end_frame']) for c in grasp] == [(3, 7)]
    assert grasp[0]['status']=='hypothesized' and grasp[0]['confidence'] is None
    assert report['effective_support_window_s'] >= 2/3-1e-9
    assert report['final_support'] == [True, True, True, False, False, False, False, False, True, True, True]
    assert not report['missing_intervals_merged'] and not report['forced_grasp']
    assert result['objects'][0]['observed'] == [True]+[False]*10


@pytest.mark.parametrize('kind', ['mask_gap', 'unsupported_pose'])
def test_missing_predicted_evidence_splits_grasp_instead_of_bridging(kind):
    scene, evidence = inputs()
    if kind=='mask_gap': evidence['valid'][5] = False
    else: scene['objects'][0]['prediction_supported'][5] = False
    result, report = phase.infer(scene, evidence, phase.default_config())
    assert report['accepted_intervals'] == 0 and report['missing_evidence_frames'] == [5]
    assert not any(c['kind']=='grasp' for c in result['contacts'])
    assert any(c['status']=='unknown' and c['start_frame']==5 and c['end_frame']==5 for c in result['contacts'])


def test_push_and_countermoving_hand_are_not_grasps():
    scene, evidence = inputs()
    for pose in scene['objects'][0]['T_world_object']: pose[2][3] = .02
    _, report = phase.infer(scene, evidence, phase.default_config())
    assert report['accepted_intervals']==0 and 'estimated lift not established; could be pushing' in report['candidates'][0]['reasons']
    scene, evidence = inputs(); evidence['hand_centers'] = -evidence['object_centers']
    _, report = phase.infer(scene, evidence, phase.default_config())
    assert report['accepted_intervals']==0 and 'insufficient hand/object co-motion' in report['candidates'][0]['reasons']


def test_release_needs_separation_stationarity_and_near_table_geometry():
    scene, evidence = inputs()
    for pose in scene['objects'][0]['T_world_object'][8:]: pose[2][3] = .08
    _, report = phase.infer(scene, evidence, phase.default_config())
    assert report['rgb_stationary_support'][8:] == [True]*3
    assert report['final_support'][8:] == [False]*3 and report['accepted_intervals']==0
    assert 'first released frame lacks stationary separated supported evidence' in report['candidates'][0]['reasons']


def test_explicit_initial_support_prior_cannot_invent_grasp_or_override_clearance():
    scene, evidence = inputs(); evidence['gaps'][:3] = 5.  # Neither near nor distant: no early RGB support.
    _, baseline = phase.infer(scene, evidence, phase.default_config())
    assert not baseline['initial_support_prior_used'] and baseline['accepted_intervals']==0
    result, prior = phase.infer(scene, evidence, phase.default_config(), initial_support_prior=True)
    assert prior['initial_support_prior_used']
    # t0 support is too old for a grasp starting at1second; it is not propagated.
    assert prior['accepted_intervals']==0 and prior['final_support'][:3] == [True, False, False]
    scene['objects'][0]['T_world_object'][0][2][3] = .1
    _, wrong_height = phase.infer(scene, evidence, phase.default_config(), initial_support_prior=True)
    assert not wrong_height['initial_support_prior_used']


def test_terminal_hold_preserves_unknown_release_without_extending_grasp():
    scene, evidence = inputs(); evidence['gaps'][8:] = 1
    result, report = phase.infer(scene, evidence, phase.default_config())
    assert report['accepted_intervals']==1 and report['terminal_hold_hypotheses']==1
    assert report['candidates'][0]['end_frame']==10
    assert not report['candidates'][0]['first_release_supported']
    grasp = next(c for c in result['contacts'] if c['kind']=='grasp')
    assert (grasp['start_frame'],grasp['end_frame'])==(3,10)
    assert grasp['terminal_hold'] and grasp['release_status']=='release_unobserved_beyond_horizon'
    assert not any(c['kind']=='support' and c['end_frame']>=8 for c in result['contacts'])
    disabled = dict(phase.default_config(), allow_terminal_hold=False)
    _, report = phase.infer(scene, evidence, disabled)
    assert report['accepted_intervals']==0


@pytest.mark.parametrize('reason', ['missing_last_mask', 'unsupported_last_pose', 'no_lift'])
def test_terminal_hold_does_not_bridge_unknown_endpoint_or_relax_grasp_evidence(reason):
    scene, evidence = inputs(); evidence['gaps'][8:] = 1
    if reason=='missing_last_mask': evidence['valid'][-1] = False
    elif reason=='unsupported_last_pose': scene['objects'][0]['prediction_supported'][-1] = False
    else:
        for pose in scene['objects'][0]['T_world_object']: pose[2][3] = .02
    result, report = phase.infer(scene, evidence, phase.default_config())
    assert report['accepted_intervals']==0 and report['terminal_hold_hypotheses']==0
    assert not any(c['kind']=='grasp' for c in result['contacts'])


def test_two_independently_supported_grasps_remain_two_intervals():
    scene, evidence = inputs()
    # Repeat the sequence with a continuous time axis; do not merge grasps.
    count = 22
    scene['timestamps'] = (np.arange(count)/3).tolist()
    obj = scene['objects'][0]; obj['T_world_object'] *= 2
    obj['observed'] = [True]+[False]*21; obj['prediction_supported'] = [False]+[True]*21
    for key, value in evidence.items():
        if isinstance(value, np.ndarray): evidence[key] = np.concatenate([value, value])
    result, report = phase.infer(scene, evidence, phase.default_config())
    assert report['accepted_intervals']==2
    assert [(c['start_frame'], c['end_frame']) for c in result['contacts'] if c['kind']=='grasp'] == [(3, 7), (14, 18)]


@pytest.mark.parametrize('change', ['observed_future', 'predicted_initial', 'missing_prediction_flags'])
def test_forecast_observed_prediction_boundary_is_not_rewritten(change):
    scene, evidence = inputs()
    if change=='observed_future': scene['objects'][0]['observed'][3] = True
    elif change=='predicted_initial': scene['objects'][0]['prediction_supported'][0] = True
    else: scene['objects'][0].pop('prediction_supported')
    with pytest.raises(ValueError, match='sole observed'): phase.infer(scene, evidence, phase.default_config())


def masks_fixture():
    shape = (40, 60); masks, t, p = [], [], []
    for frame in range(3):
        block = np.zeros(shape, bool); block[15:20, 10+frame:15+frame] = True
        hand = np.zeros(shape, bool); hand[15:20, 2+frame:7+frame] = True
        for prompt, mask in [(0, block), (0, block.copy()), (1, hand)]:
            masks.append(mask); t.append(frame); p.append(prompt)
    known = np.zeros((3, 1), bool); known[0, 0] = True
    return dict(mask=np.packbits(np.stack(masks).reshape(9, -1), axis=1), t=np.array(t), p=np.array(p),
        v=np.zeros(9, dtype=int), score=np.ones(9)*.8, shape=np.array(shape), timestamps=np.arange(3)/3,
        prompts=np.array(['wooden block on patterned mat','hand','tabletop']), observed_mask=known,
        source_frame_indices=np.array([0,-1,-1]))


def test_duplicate_masks_collapse_but_distinct_objects_remain_ambiguous():
    masks = masks_fixture(); config = phase.default_config()
    evidence = phase.mask_evidence(masks, masks['timestamps'], config)
    assert evidence['valid'].all() and np.all(evidence['gaps']==4)
    distinct = np.zeros((40,60), bool); distinct[25:30,30:35] = True
    masks['mask'][1] = np.packbits(distinct.ravel())
    evidence = phase.mask_evidence(masks, masks['timestamps'], config)
    assert not evidence['valid'][0] and not evidence['object_known'][0]


def test_contact_masks_cannot_silently_filter_out_helper_ancestry():
    masks = masks_fixture(); masks['v'][0] = 1
    with pytest.raises(ValueError, match='source-only'): phase.mask_evidence(masks, masks['timestamps'], phase.default_config())


@pytest.mark.parametrize('helper_fit', [False, True])
def test_producer_preserves_scene_bytes_and_generated_parent_lineage(tmp_path, helper_fit):
    path = Path(__file__).with_name('test_forecast_masks.py')
    spec = importlib.util.spec_from_file_location('contact_forecast_fixture', path)
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    helper, condition, checkpoint, _ = fixture.generated_fixture(tmp_path)
    masks = tmp_path/'masks'
    fixture.fm.run(helper, condition, masks, checkpoint, fixture.detections)
    initial = json.loads((tmp_path/'initial/scene.json').read_text())
    times = np.arange(5)*.2 + 2.
    obj = initial['objects'][0]
    obj['T_world_object'] = obj['T_world_object']*5
    obj['observed'] = [True]+[False]*4; obj['prediction_supported'] = [False]+[True]*4
    obj['prediction_provenance'] = dict(origin='model_prediction', generated=True,
        method='synthetic generated scene', artifact_ids=['generated_source_forecast'])
    initial['timestamps'] = times.tolist()
    initial['support'] = dict(id='table', normal_world=[0,0,1], offset_m=-.78, confidence=None,
                             provenance=dict(method='synthetic estimated plane', confidence_status='uncalibrated'))
    initial['forecast'] = dict(observed_until_s=2., future_recorded_RGB_used=False)
    scene_dir = tmp_path/'fit'; scene_dir.mkdir()
    scene_path = scene_dir/'scene-sequence.json'; scene_path.write_text(json.dumps(initial))
    before = phase.file_sha256(scene_path)
    parent_bundle = masks if helper_fit else helper
    manifest = json.loads((parent_bundle/'input-manifest.json').read_text())
    for record in manifest['artifacts']:
        if 'path' in record: record['path'] = str((parent_bundle/record['path']).resolve())
    manifest['artifacts'].append(dict(id='forecast_scene', kind='object_geometry', origin='model_prediction',
        parent_ids=['forecast_sam3_masks'] if helper_fit else ['generated_source_forecast'],
        path='scene-sequence.json', sha256=before))
    manifest['method_inputs'] = ['forecast_scene']
    (scene_dir/'input-manifest.json').write_text(json.dumps(manifest))
    config_path = tmp_path/'contacts.json'; config_path.write_text(json.dumps(phase.default_config()))
    out = tmp_path/'contacts'
    report = phase.produce(scene_bundle=scene_dir, masks=masks/'source-only', condition=condition,
                           config_path=config_path, output=out)
    assert report['accepted_intervals']==0  # Fake predictor supplies no hands.
    assert phase.file_sha256(scene_path)==before
    result = json.loads((out/'scene-sequence.json').read_text())
    assert result['objects']==initial['objects'] and result['support']==initial['support']
    active = set(report['input_protocol']['method_artifact_ids'])
    assert {'forecast_contact_phases','forecast_scene','forecast_sam3_masks','generated_source_forecast'} <= active
    if helper_fit:
        assert 'generated_helper_rgb' in active and 'forecast_contact_source_masks' in active
        assert report['source_mask_merge']['artifact_id_aliases'] == {'forecast_sam3_masks':'forecast_contact_source_masks'}
        output_manifest = json.loads((out/'input-manifest.json').read_text())
        records = {a['id']:a for a in output_manifest['artifacts']}
        assert records['forecast_sam3_masks']['path'] == str(masks/'masks.npz')
        assert records['forecast_contact_source_masks']['path'] == str(masks/'source-only/masks.npz')
        assert records['forecast_contact_source_masks']['sha256'] == phase.file_sha256(masks/'source-only/masks.npz')
        assert 'generated_helper_rgb' not in records['forecast_contact_source_masks']['parent_ids']
        assert 'forecast_contact_source_masks' in records['forecast_contact_phases']['parent_ids']
    else:
        assert 'generated_helper_rgb' not in active
        assert report['source_mask_merge']['artifact_id_aliases'] == {}
    assert report['input_protocol']['task_mode']=='forecast'
    # All-view masks cannot be silently filtered to claim the matched baseline.
    with pytest.raises(ValueError, match='source-only'):
        phase.produce(scene_bundle=scene_dir, masks=masks, condition=condition,
                      config_path=config_path, output=tmp_path/'bad-contacts')
