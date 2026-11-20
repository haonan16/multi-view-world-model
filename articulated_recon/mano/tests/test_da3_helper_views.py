import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def module():
    path=Path(__file__).resolve().parents[2]/'benchmarks/da3_helper_views.py'
    spec=importlib.util.spec_from_file_location('da3_helper_views',path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_matched_baseline_rejects_generated_source_or_timing():
    m=module()
    rgb=np.arange(3*4*6*3,dtype=np.uint8).reshape(3,4,6,3)
    ids=np.array([0,2]); times=np.array([0.,.2,.4])
    helper=np.repeat(rgb[ids,None],4,axis=1)
    assert np.array_equal(m.verify_matched_pixels(helper,rgb,times[ids],times,ids,[0,0,6,4]),rgb[ids])
    helper[1,0,0,0,0]^=1
    with pytest.raises(ValueError,match='exact'):
        m.verify_matched_pixels(helper,rgb,times[ids],times,ids,[0,0,6,4])
    with pytest.raises(ValueError,match='timing'):
        m.verify_matched_pixels(helper,rgb,np.array([0.,.3]),times,ids,[0,0,6,4])


def test_prediction_grid_intrinsics_and_camera_gauge():
    m=module()
    ex=np.repeat(np.eye(4)[None],2,axis=0)
    ex[0,0,3]=2.;ex[1,0,3]=3.
    k=np.repeat(np.eye(3)[None],2,axis=0); k[:,0,0]=6.;k[:,1,1]=4.
    pred=SimpleNamespace(depth=np.ones((2,2,3)),intrinsics=k,extrinsics=ex,
        conf=np.ones((2,2,3))*.8,is_metric=0)
    result=m.prediction_arrays(pred,(4,6))
    assert result['depth'].shape==(2,4,6)
    assert result['K'][0,0,0]==12.
    assert result['K'][0,1,1]==8.
    np.testing.assert_allclose(result['c2w'][0],np.eye(4))
    assert result['c2w'][1,0,3]==-1.
    assert result['is_metric'] is False


def test_source_call_never_passes_helper_pixels_or_requested_cameras(tmp_path):
    # Stop after recording the model call; no GPU, checkpoints, or assets needed.
    m=module();calls=[]
    class StopAfterInputs:
        def inference(self,images,**kwargs):
            calls.append((images,kwargs))
            raise RuntimeError('captured')
    source=np.full((2,4,6,3),42,np.uint8)
    helper=np.full((2,4,4,6,3),180,np.uint8);helper[:,0]=source
    job=dict(source=source,rgb=helper)
    with pytest.raises(RuntimeError,match='captured'):
        m.run_condition(job,StopAfterInputs(),tmp_path/'checkpoint',tmp_path,
            'source_unposed',320,0.)
    assert len(calls[0][0])==1
    np.testing.assert_array_equal(calls[0][0][0],source[0])
    assert calls[0][1]['extrinsics'] is None
    assert calls[0][1]['intrinsics'] is None
    assert calls[0][1]['align_to_input_ext_scale'] is False


def test_source_completion_manifest_has_only_observed_pixels_and_model_prior(tmp_path):
    m=module()
    source=np.full((2,4,6,3),42,np.uint8)
    rgb_file=tmp_path/'rgb.npz'
    np.savez_compressed(rgb_file,rgb=source,timestamps=np.array([0.,1.]))
    helper=tmp_path/'helper';helper.mkdir()
    np.savez_compressed(helper/'observations.npz',rgb=np.repeat(source[:,None],4,axis=1))
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    (checkpoint/'config.json').write_text('{}')
    class FakeModel:
        def inference(self,images,**kwargs):
            assert len(images)==1 and kwargs['extrinsics'] is None
            return SimpleNamespace(depth=np.ones((1,2,3)),intrinsics=np.eye(3)[None],
                extrinsics=np.eye(4)[None],conf=np.ones((1,2,3)),is_metric=0)
    job=dict(source=source,rgb=np.repeat(source[:,None],4,axis=1),helper=helper,
        timestamps=np.array([0.,1.]),ids=np.array([0,1]),
        helper_sha256=m.file_sha256(helper/'observations.npz'),
        observed=dict(id='source_rgb',kind='rgb',origin='observed_rgb',generated=False,
            parent_ids=[],path=str(rgb_file),sha256=m.file_sha256(rgb_file),time_range_s=[0.,1.]))
    m.run_condition(job,FakeModel(),checkpoint,tmp_path,'source_unposed',320,0.)
    receipt=json.loads((tmp_path/'source_unposed/execution.json').read_text())
    assert receipt['status']=='complete'
    assert receipt['validation']['execution_ready'] is False
    assert set(receipt['validation']['method_artifact_ids'])=={'source_rgb','da3_weights','da3_cameras','da3_depth'}


def forecast_pixels():
    source = np.full((1, 4, 6, 3), 42, np.uint8)
    generated = np.full((3, 4, 4, 6, 3), 180, np.uint8)
    generated[0, 0] = source[0]
    generated[1, 0] = 95
    known = np.zeros((3, 4), bool); known[0, 0] = True
    return source, generated, known


def test_forecast_preserves_future_pixels_and_only_checks_observed_anchor():
    m=module(); source, rgb, known = forecast_pixels()
    result = m.verify_forecast_pixels(rgb, source, np.array([0., .1, .2]), np.array([0.]),
                                     np.array([0, -1, -1]), known, [0, 0, 6, 4])
    np.testing.assert_array_equal(result, rgb[:, 0])
    assert (result[1] == 95).all() and (result[2] == 180).all()
    rgb[0, 0, 0, 0, 0] ^= 1
    with pytest.raises(ValueError, match='exact'):
        m.verify_forecast_pixels(rgb, source, np.array([0., .1, .2]), np.array([0.]),
                                 np.array([0, -1, -1]), known, [0, 0, 6, 4])


def test_forecast_rejects_hidden_recorded_future_and_false_observed_flags():
    m=module(); source, rgb, known = forecast_pixels()
    with pytest.raises(ValueError, match='exactly one'):
        m.verify_forecast_pixels(rgb, np.repeat(source, 3, axis=0), np.array([0., .1, .2]),
            np.array([0., .1, .2]), np.array([0, -1, -1]), known, [0, 0, 6, 4])
    known[1, 0] = True
    with pytest.raises(ValueError, match='Only the initial'):
        m.verify_forecast_pixels(rgb, source, np.array([0., .1, .2]), np.array([0.]),
            np.array([0, -1, -1]), known, [0, 0, 6, 4])
    known[1, 0] = False
    with pytest.raises(ValueError, match='no recorded source index'):
        m.verify_forecast_pixels(rgb, source, np.array([0., .1, .2]), np.array([0.]),
            np.array([0, 1, 2]), known, [0, 0, 6, 4])


def test_forecast_source_only_output_keeps_generated_ancestry(tmp_path):
    m=module(); source, rgb, known = forecast_pixels()
    times=np.array([0., .1, .2]); ids=np.array([0,-1,-1])
    original=tmp_path/'original.npz'; np.savez(original,rgb=source,timestamps=np.array([0.]))
    helper=tmp_path/'helper'; helper.mkdir()
    np.savez(helper/'observations.npz',rgb=rgb)
    observed=dict(id='source_rgb',kind='rgb',origin='observed_rgb',generated=False,
        parent_ids=[],path=str(original),sha256=m.file_sha256(original),time_range_s=[0.,0.])
    artifacts=[observed,dict(id='world_model',kind='model_prior',origin='prior',
        prior_type='learned_model_weights',scope='global',parent_ids=[])]
    for name, role in [('generated_source_forecast','source_forecast'),('generated_helper_rgb','helper')]:
        artifacts.append(dict(id=name,kind='rgb',origin='model_prediction',generated=True,view_role=role,
            parent_ids=['source_rgb','world_model'],path=str(helper/'observations.npz'),
            sha256=m.file_sha256(helper/'observations.npz'),time_range_s=[0.,.2]))
    manifest=dict(schema_version=1,condition='rgb_only',task_mode='forecast',observed_until_s=0.,
        artifacts=artifacts,method_inputs=['generated_source_forecast','generated_helper_rgb'])
    checkpoint=tmp_path/'checkpoint'; checkpoint.mkdir(); (checkpoint/'config.json').write_text('{}')
    calls=[]
    class FakeModel:
        def inference(self,images,**kwargs):
            assert len(images)==1
            calls.append(images[0].copy())
            return SimpleNamespace(depth=np.ones((1,2,3)),intrinsics=np.eye(3)[None],
                extrinsics=np.eye(4)[None],conf=np.ones((1,2,3)),is_metric=0)
    job=dict(source=rgb[:,0].copy(),rgb=rgb,helper=helper,timestamps=times,ids=ids,
        observed=observed,mode='forecast',observed_mask=known,manifest=manifest,
        active_ids={a['id'] for a in artifacts},helper_sha256=m.file_sha256(helper/'observations.npz'))
    m.run_condition(job,FakeModel(),checkpoint,tmp_path,'source_unposed',320,0.)
    np.testing.assert_array_equal(np.stack(calls),rgb[:,0])
    result=json.loads((tmp_path/'source_unposed/input-manifest.json').read_text())
    receipt=json.loads((tmp_path/'source_unposed/execution.json').read_text())
    assert result['task_mode']=='forecast' and result['observed_until_s']==0.
    active=set(receipt['validation']['method_artifact_ids'])
    assert 'generated_source_forecast' in active and 'generated_helper_rgb' not in active
    assert receipt['generated_views_used'] and not receipt['generated_helper_views_used']
    assert receipt['source_stream']=='observed_initial_plus_generated_future'
    with np.load(tmp_path/'source_unposed/predictions.npz') as z:
        np.testing.assert_array_equal(z['rgb'][:,0],rgb[:,0])
        np.testing.assert_array_equal(z['observed_mask'],known[:,:1])
        np.testing.assert_array_equal(z['source_frame_indices'],ids)


def test_strict_preparation_export_is_consumable_by_depth_adapter(tmp_path):
    # Exercise the real producer and consumer contracts together, without a GPU.
    fixture_path=Path(__file__).with_name('test_world_model_forecast.py')
    spec=importlib.util.spec_from_file_location('strict_forecast_test_fixture',fixture_path)
    fixture_module=importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture_module)
    sf, wm = fixture_module.sf, fixture_module.wm
    args,_=fixture_module.fixture_inputs(tmp_path)
    sf.prepare_forecast(**args)
    condition=args['output']; helper=tmp_path/'generated'; helper.mkdir()
    with np.load(condition/'condition.npz') as z:
        arrays={k:z[k].copy() for k in z.files}
    _,known=wm.source_tensor(arrays['rgb'],forecast=True,total_frames=len(arrays['timestamps']))
    predicted=np.full((len(known),4,160,320,3),77,np.uint8)
    rgb=wm.restore_observed_source(predicted,arrays['rgb'],known)
    np.savez_compressed(helper/'observations.npz',rgb=rgb,observed_mask=known,
        **{k:v for k,v in arrays.items() if k!='rgb'})
    manifest=json.loads((condition/'input-manifest.json').read_text())
    for record in manifest['artifacts']:
        if 'path' in record: record['path']=str((condition/record['path']).resolve())
    manifest['artifacts'].append(dict(id='world_model_weights',kind='model_prior',origin='prior',
        prior_type='learned_model_weights',scope='global',parent_ids=[]))
    manifest['artifacts'].extend(sf.generated_records(helper,arrays['timestamps']))
    manifest['method_inputs'].extend(['generated_source_forecast','generated_helper_rgb'])
    (helper/'input-manifest.json').write_text(json.dumps(manifest))
    job=module().read_inputs(helper,condition)
    assert job['mode']=='forecast' and job['observed_mask'].sum()==1
    assert job['camera_request_ids']==['forecast_camera_requests']
    assert (job['source'][1:]==77).all()
    np.testing.assert_array_equal(job['source'][0],arrays['rgb'][0])
    # The depth consumer must revalidate strict preparation, not accept the
    # forecast label merely because the observed RGB file contains one image.
    prep=json.loads((condition/'preparation.json').read_text())
    prep['task_text']='A different task'
    (condition/'preparation.json').write_text(json.dumps(prep))
    with pytest.raises(ValueError,match='task or camera ancestry'):
        module().read_inputs(helper,condition)
