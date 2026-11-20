"""Forecast tracker adapter tests use a synthetic callback, never real SAM3."""
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from artirecon.input_protocol import file_sha256,load_manifest
from benchmarks import forecast_tracker as ft
from benchmarks.forecast_geometry import merge_inputs
from benchmarks import forecast_masks as fm
from test_forecast_masks import generated_fixture,detections


def fixture(tmp_path,predict=detections):
    helper,condition,checkpoint,rgb=generated_fixture(tmp_path)
    baseline=tmp_path/'framewise';fm.run(helper,condition,baseline,checkpoint,predict)
    initial=tmp_path/'initial-masks';initial.mkdir()
    native=np.zeros((12,16),bool);native[4:8,4:8]=True
    np.savez(initial/'masks.npz',mask=np.packbits(native.reshape(-1))[None],score=np.array([.83],np.float32),
        box=np.array([[4,4,7,7]],np.float32),t=np.array([0]),p=np.array([0]),
        prompts=np.asarray([ft.OBJECT_PROMPT]),shape=np.array([12,16]),timestamps=np.array([2.]))
    manifest=json.loads((tmp_path/'initial/input-manifest.json').read_text())
    manifest['artifacts'].append(dict(id='initial_sam3_masks',kind='masks',origin='model_prediction',
        parent_ids=['source'],path=str(initial/'masks.npz'),sha256=file_sha256(initial/'masks.npz')))
    manifest['metric_scale']=dict(status='unresolved',artifact_ids=[],description='RGB masks have no metric scale')
    manifest['method_inputs']=['initial_sam3_masks'];(initial/'input-manifest.json').write_text(json.dumps(manifest))
    return dict(helper=helper,condition=condition,baseline_masks=baseline,initial_masks=initial,
                output=tmp_path/'tracked',checkpoint=checkpoint),rgb,native


def fake_track(video,seed):
    for t in range(len(video)):
        mask=np.roll(seed,t,axis=1)
        yield dict(frame_index=t,mask_logits=np.where(mask,2.,-2.),object_score_logit=-4. if t==2 else 1.2)


def test_source_seed_uses_exact_native_crop_and_not_resized_detector(tmp_path):
    args,rgb,native=fixture(tmp_path);inputs=ft.load_inputs(**{k:args[k] for k in ['helper','condition','baseline_masks','initial_masks']})
    prep=json.loads((args['condition']/'preparation.json').read_text())
    expected=ft.source_seed(native,prep['source_crop_xywh'],rgb.shape[2:4])
    np.testing.assert_array_equal(inputs['seeds'][0],expected)
    assert inputs['seeds'][0].sum()>100
    assert inputs['seeds'][1].sum()==30
    assert inputs['seed_reports'][0]['source']=='native_first_observed_image'


def test_all_views_track_exact_generated_video_and_retain_visibility_logits(tmp_path):
    args,rgb,native=fixture(tmp_path);seen=[]
    def callback(video,seed):
        seen.append(video.copy());yield from fake_track(video,seed)
    result=ft.run(**args,track=callback)
    assert result['status']=='complete' and result['processed_frames']==20
    assert result['seeded_views']==[0,1,2,3] and len(seen)==4
    progress=json.loads((args['output']/'progress.json').read_text())
    assert progress['status']=='complete' and progress['processed_frames']==20
    for v,video in enumerate(seen):np.testing.assert_array_equal(video,rgb[:,v])
    with np.load(args['output']/'masks.npz') as z:
        obj=z['p']==list(z['prompts']).index(ft.OBJECT_PROMPT)
        assert obj.sum()==16 and not np.any(z['t'][obj]==2)
        assert np.all(z['score_kind'][obj]==1) and np.all(z['track_id'][obj]==1)
        np.testing.assert_allclose(z['score'][obj],1/(1+np.exp(-1.2)),atol=1e-7)
        np.testing.assert_allclose(z['object_score_logit'][obj],1.2)
        assert z['observed_mask'].sum()==1 and z['source_frame_indices'].tolist()==[0,-1,-1,-1,-1]
    with np.load(args['output']/'tracker-logits.npz') as z:
        assert z['mask_logits'].shape==(5,4,160,320)
        assert z['processed'].all() and np.all(z['object_score_logits'][2]==-4)


def test_nonobject_masks_and_scores_are_not_changed(tmp_path):
    args,_,_=fixture(tmp_path);ft.run(**args,track=fake_track)
    for suffix in ['', 'source-only']:
        before=ft.arrays(args['baseline_masks']/suffix/'masks.npz');after=ft.arrays(args['output']/suffix/'masks.npz')
        a=before['p']!=0;b=after['p']!=0
        for key in ['mask','box','score','t','v','p']:np.testing.assert_array_equal(before[key][a],after[key][b])
        assert np.all(after['score_kind'][b]==0)


def test_source_product_excludes_helpers_and_merges_with_all_view_geometry(tmp_path):
    args,_,_=fixture(tmp_path);ft.run(**args,track=fake_track)
    source=args['output']/'source-only';report=load_manifest(source/'input-manifest.json')
    active=set(report['method_artifact_ids'])
    assert 'generated_source_forecast' in active and 'generated_helper_rgb' not in active
    assert 'initial_sam3_masks' in active and 'forecast_tracker_source_framewise_masks' in active
    # Separate raw-logit and framewise-mask IDs avoid hidden collisions when
    # helper geometry later consumes independently tracked source contacts.
    merged=merge_inputs([args['output']/'input-manifest.json',source/'input-manifest.json'])
    ids={a['id'] for a in merged['artifacts']}
    assert {'forecast_video_tracker_masks','forecast_source_video_tracker_masks'}<=ids
    with np.load(source/'tracker-logits.npz') as z:assert z['mask_logits'].shape==(5,1,160,320)


def test_missing_helper_seed_remains_unknown_not_forward_filled(tmp_path):
    def predict(image,prompt):
        if prompt==ft.OBJECT_PROMPT and np.all(image==53):
            return np.zeros((0,160,320),bool),np.zeros(0),np.zeros((0,4))
        return detections(image,prompt)
    args,_,_=fixture(tmp_path,predict=predict);calls=[]
    def track(video,seed):calls.append(1);yield from fake_track(video,seed)
    result=ft.run(**args,track=track)
    assert result['unsupported_views']==[3] and len(calls)==3
    with np.load(args['output']/'masks.npz') as z:assert not np.any((z['p']==0)&(z['v']==3))
    with np.load(args['output']/'tracker-logits.npz') as z:
        assert not z['processed'][:,3].any() and np.isnan(z['mask_logits'][:,3]).all()


def test_ambiguous_seed_is_not_selected_by_score():
    masks=np.zeros((2,10,10),bool);masks[0,:4,:4]=True;masks[1,6:,6:]=True
    data=dict(prompts=np.array([ft.OBJECT_PROMPT]),shape=np.array([10,10]),t=np.zeros(2,int),p=np.zeros(2,int),
        score=np.array([.99,.6]),mask=np.packbits(masks.reshape(2,-1),axis=1))
    seed,report=ft.select_seed(data,0,ft.OBJECT_PROMPT)
    assert seed is None and report['status']=='ambiguous'
    data['mask'][1]=data['mask'][0]
    seed,report=ft.select_seed(data,0,ft.OBJECT_PROMPT)
    assert seed.sum()==16 and report['selected_row']==0


def association_fixture(tmp_path,scores=(.9,.4)):
    args,rgb,_=fixture(tmp_path)
    masks=np.zeros((2,12,16),bool);masks[0,2:6,2:6]=True;masks[1,7:11,10:14]=True
    path=args['initial_masks']/'masks.npz'
    data=ft.arrays(path)
    data.update(mask=np.packbits(masks.reshape(2,-1),axis=1),score=np.asarray(scores,np.float32),
                box=np.array([[2,2,5,5],[10,7,13,10]],np.float32),t=np.zeros(2,int),p=np.zeros(2,int),
                prompts=np.asarray([ft.OBJECT_PROMPT,'hand','tabletop']))
    np.savez(path,**data)
    manifest_path=args['initial_masks']/'input-manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['artifacts'][-1]['sha256']=file_sha256(path);manifest_path.write_text(json.dumps(manifest))
    from benchmarks.source_geometry_v2 import default_config
    config=tmp_path/'initializer-config.json';config.write_text(json.dumps(default_config()))
    return args,rgb,masks,config


def test_optional_source_association_recomputes_initial_mask_only_and_records_margin(tmp_path):
    args,rgb,masks,config=association_fixture(tmp_path)
    base={k:args[k] for k in ['helper','condition','baseline_masks','initial_masks']}
    assert ft.load_inputs(**base)['seeds'][0] is None
    inputs=ft.load_inputs(**base,source_association_config=config)
    report=inputs['seed_reports'][0]
    assert report['status']=='heuristic_selected' and report['selected_row']==0
    assert report['selection_margin']>.19 and report['candidate_margin_required']==.05
    assert report['heuristic_not_ground_truth'] and not report['depth_used']
    assert report['frames_used']==[0] and report['config_sha256']==file_sha256(config)
    crop=json.loads((args['condition']/'preparation.json').read_text())['source_crop_xywh']
    np.testing.assert_array_equal(inputs['seeds'][0],ft.source_seed(masks[0],crop,rgb.shape[2:4]))
    result=ft.run(**args,source_association_config=config,track=fake_track)
    assert result['status']=='complete' and result['seeded_views']==[0,1,2,3]
    assert result['input_files'][str(config.resolve())]==file_sha256(config)


def test_optional_source_association_keeps_equal_hypotheses_unsupported(tmp_path):
    args,_,_,config=association_fixture(tmp_path,scores=(.8,.8))
    result=ft.run(**args,source_association_config=config,track=fake_track)
    assert result['unsupported_views']==[0]
    assert result['seeds'][0]['status']=='ambiguous'
    assert result['seeds'][0]['selection_margin']==0
    with np.load(args['output']/'source-only/tracker-logits.npz') as z:
        assert not z['processed'].any() and np.isnan(z['mask_logits']).all()


def test_source_association_config_tampering_rejects_output(tmp_path):
    args,_,_,config=association_fixture(tmp_path)
    def altered(video,seed):
        config.write_text(config.read_text()+' ')
        yield from fake_track(video,seed)
    with pytest.raises(ValueError,match='inputs changed'):
        ft.run(**args,source_association_config=config,track=altered)
    assert json.loads((args['output']/'execution.json').read_text())['status']=='failed'


def test_source_association_rejects_extra_native_time_even_without_later_candidates(tmp_path):
    _,_,_,config=association_fixture(tmp_path)
    native=ft.arrays(tmp_path/'initial-masks/masks.npz');native['timestamps']=np.array([2.,3.])
    with pytest.raises(ValueError,match='exactly one source image'):
        ft.associate_source_seed(native,config)


def transport_fixture():
    image=np.zeros((160,320,3),np.uint8);seed=np.zeros((160,320),bool);seed[75:85,150:160]=True
    xx,yy=np.meshgrid(np.linspace(20,300,7),np.linspace(20,140,5))
    a=np.c_[xx.ravel(),yy.ravel()];b=a+np.array([5.,-3.])
    counts=dict(source_features=80,target_features=80,mutual_matches=len(a),homography_inliers=len(a))
    return image,seed,a,b,counts


def test_background_transport_moves_mask_without_camera_or_object_detector_input():
    image,seed,a,b,counts=transport_fixture()
    mask,report=ft.transport_seed(image,image,seed,~seed,match_fn=lambda *_:(a,b,counts))
    expected=np.zeros_like(seed);expected[72:82,155:165]=True
    np.testing.assert_array_equal(mask,expected)
    assert report['status']=='transported_hypothesis' and report['local_inliers']>=3
    assert not report['ground_truth_used'] and not report['requested_cameras_used']
    assert report['median_error_px']<1e-5 and report['area_ratio']==1


@pytest.mark.parametrize('bad',['few_matches','clustered','far_from_target'])
def test_unreliable_background_transport_stays_unknown(bad):
    image,seed,a,b,counts=transport_fixture()
    if bad=='few_matches':a=a[:5];b=b[:5]
    elif bad=='clustered':a=a*.05+100;b=a+5
    else:a[:,0]=a[:,0]*.3;b=a+5
    counts.update(mutual_matches=len(a),homography_inliers=len(a))
    mask,report=ft.transport_seed(image,image,seed,~seed,match_fn=lambda *_:(a,b,counts))
    assert mask is None and report['status']=='unsupported'


def test_helper_transport_overrides_unrelated_detector_seed_but_never_source_evidence(tmp_path,monkeypatch):
    args,rgb,_,config=association_fixture(tmp_path);calls=[]
    def transport(source,target,seed,allowed):
        np.testing.assert_array_equal(source,rgb[0,0]);assert not allowed[seed].any()
        calls.append(target.copy())
        return seed.copy(),dict(status='transported_hypothesis')
    monkeypatch.setattr(ft,'transport_seed',transport)
    result=ft.run(**args,source_association_config=config,helper_seed_transport=True,track=fake_track)
    assert result['cross_view_identity_estimated'] and len(calls)==6 # before and after immutable-input check
    assert all(x['source']=='source_mask_transported_to_generated_helper_t0' for x in result['seeds'][1:])
    report=load_manifest(args['output']/'source-only/input-manifest.json')
    assert 'generated_helper_rgb' not in report['method_artifact_ids']


def test_helper_transport_never_falls_back_to_unrelated_detection_when_source_ambiguous(tmp_path,monkeypatch):
    args,_,_,config=association_fixture(tmp_path,scores=(.8,.8))
    def forbidden(*_):raise AssertionError('Ambiguous source cannot seed any helper')
    monkeypatch.setattr(ft,'transport_seed',forbidden)
    result=ft.run(**args,source_association_config=config,helper_seed_transport=True,track=forbidden)
    assert not result['tracker_executed'] and result['unsupported_views']==[0,1,2,3]
    assert all(x['reason']=='source_seed_unavailable' for x in result['seeds'][1:])


def test_visibility_and_area_gates_do_not_invent_mask_confidence():
    logits=np.ones((4,4));mask,score,valid=ft.candidate(logits,-1.)
    assert mask.sum()==16 and score<.5 and not valid
    _,score,valid=ft.candidate(-logits,8.)
    assert score>.99 and not valid
    with pytest.raises(ValueError):ft.candidate(logits,float('nan'))


@pytest.mark.parametrize('failure',['duplicate','missing'])
def test_incomplete_native_propagation_is_a_failed_run(tmp_path,failure):
    args,_,_=fixture(tmp_path)
    def bad(video,seed):
        rows=list(fake_track(video,seed))
        yield from rows[:-1] if failure=='missing' else rows+[rows[-1]]
    with pytest.raises(ValueError):ft.run(**args,track=bad)
    assert json.loads((args['output']/'execution.json').read_text())['status']=='failed'
    assert not (args['output']/'input-manifest.json').exists()


def test_changed_seed_file_cannot_be_published_as_verified_track(tmp_path):
    args,_,_=fixture(tmp_path);changed=False
    def bad(video,seed):
        nonlocal changed
        if not changed:
            with (args['initial_masks']/'masks.npz').open('ab') as stream:stream.write(b'tampered')
            changed=True
        yield from fake_track(video,seed)
    with pytest.raises(ValueError):ft.run(**args,track=bad)
    assert json.loads((args['output']/'execution.json').read_text())['status']=='failed'


def test_source_only_cannot_silently_mix_detector_candidates_from_other_run(tmp_path):
    args,_,_=fixture(tmp_path);folder=args['baseline_masks']/'source-only'
    data=ft.arrays(folder/'masks.npz');data['score'][0]=.72
    np.savez_compressed(folder/'masks.npz',**data)
    manifest=json.loads((folder/'input-manifest.json').read_text())
    for record in manifest['artifacts']:
        if record['id']=='forecast_sam3_masks':record['sha256']=file_sha256(folder/'masks.npz')
    (folder/'input-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='exact independent view-0 candidate subset'):
        ft.load_inputs(**{k:args[k] for k in ['helper','condition','baseline_masks','initial_masks']})


def test_native_hf_wrapper_preserves_raw_mask_and_presence_logits(monkeypatch):
    class Tensor:
        def __init__(self,value):self.value=np.asarray(value)
        def detach(self):return self
        def float(self):return self
        def cpu(self):return self
        def numpy(self):return self.value
    video=np.arange(2*4*5*3,dtype=np.uint8).reshape(2,4,5,3)
    seed=np.ones((4,5),bool);session=object();calls=[]
    class Processor:
        def init_video_session(self,**kw):
            np.testing.assert_array_equal(kw['video'],video)
            assert kw['inference_device']=='cuda' and kw['dtype']=='bf16'
            calls.append('init');return session
        def add_inputs_to_inference_session(self,**kw):
            assert kw['inference_session'] is session and kw['frame_idx']==0 and kw['obj_ids']==[1]
            np.testing.assert_array_equal(kw['input_masks'],seed.astype(np.float32));calls.append('seed')
        def post_process_masks(self,masks,**kw):
            assert kw==dict(original_sizes=[[4,5]],binarize=False)
            return masks
    class Model:
        def propagate_in_video_iterator(self,sess,**kw):
            assert sess is session and kw==dict(start_frame_idx=0,reverse=False)
            for t in range(2):
                yield SimpleNamespace(object_ids=[1],frame_idx=t,pred_masks=Tensor(np.full((1,1,4,5),t-.4)),
                    object_score_logits=Tensor(np.array([-.8+t])))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(bfloat16='bf16',inference_mode=nullcontext))
    results=list(ft.native_track(Model(),Processor())(video,seed))
    assert calls==['init','seed']
    assert [r['frame_index'] for r in results]==[0,1]
    np.testing.assert_allclose(results[0]['mask_logits'],-.4)
    assert results[0]['mask_logits'].shape==(4,5) and results[0]['object_score_logit']==-.8
