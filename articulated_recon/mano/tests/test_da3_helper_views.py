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
