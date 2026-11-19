import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def module():
    p=Path(__file__).resolve().parents[2]/'benchmarks/world_model_views.py'
    spec=importlib.util.spec_from_file_location('wm_views',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_virtual_cameras_are_rigid_source_relative_and_aim_at_estimated_target():
    m=module();target=np.array([.1,.03,.8]);cams=m.virtual_cameras(target)
    np.testing.assert_allclose(cams[0],np.eye(4))
    r=cams[:,:3,:3]
    np.testing.assert_allclose(r.transpose(0,2,1)@r,np.broadcast_to(np.eye(3),(4,3,3)),atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(r),1.,atol=1e-12)
    for c in cams[1:]:
        pc=(target-c[:3,3])@c[:3,:3]
        np.testing.assert_allclose(pc[:2],0,atol=1e-12)
        assert pc[2]>0


def test_full_source_video_cannot_fill_unknown_views():
    m=module();rgb=np.full((9,4,8,3),255,np.uint8)
    pack,known=m.source_tensor(rgb)
    assert pack.shape==(1,4,9,3,4,8)
    assert known[:,0].all() and not known[:,1:].any()
    assert np.all(pack[0,0]==1) and not pack[:,1:].any()


def test_forecast_encodes_only_one_source_frame():
    m=module();rgb=np.full((1,4,8,3),255,np.uint8)
    pack,known=m.source_tensor(rgb,forecast=True,total_frames=9)
    assert known.sum()==1
    assert not pack[:,1:].any() and not pack[:,:,1:].any()
    with pytest.raises(ValueError,match='exactly one'):
        m.source_tensor(np.repeat(rgb,9,axis=0),forecast=True,total_frames=9)


def test_first_frame_diagnostic_does_not_encode_or_restore_later_recorded_frames():
    m=module();rgb=np.arange(17*4*8*3,dtype=np.uint8).reshape(17,4,8,3)
    pixels,known=m.reconstruction_condition(rgb,'first_frame_diagnostic')
    assert known.sum()==1 and known[0,0]
    assert not pixels[:,1:].any() and not pixels[:,:,1:].any()
    raw=np.full((17,4,4,8,3),73,np.uint8)
    restored=m.restore_observed_source(raw,rgb,known)
    assert np.array_equal(restored[0,0],rgb[0])
    assert np.array_equal(restored[1:,0],raw[1:,0])
    assert np.array_equal(restored[:,1:],raw[:,1:])
    full,full_known=m.reconstruction_condition(rgb,'full_source')
    assert full_known[:,0].all() and not full_known[:,1:].any()
    assert np.array_equal(m.restore_observed_source(raw,rgb,full_known)[:,0],rgb)


@pytest.mark.parametrize('target',[[0,0,-1],[0,0,0],[0,float('nan'),1]])
def test_invalid_camera_target_rejected(target):
    with pytest.raises(ValueError):module().virtual_cameras(target)


def test_guidance_changes_clamp_check_count_without_counting_unconditioned_branch():
    m=module()
    assert m.checked_guidance(2.,1.,30)==60
    assert m.checked_guidance(2.,0.,30)==60
    assert m.checked_guidance(0.,0.,30)==30
    assert m.checked_guidance(0.,1.,30)==30
    for bad in [-1.,float('nan'),float('inf')]:
        with pytest.raises(ValueError):m.checked_guidance(bad,0.,30)


def test_zero_guidance_is_not_silently_treated_as_null_text():
    from types import SimpleNamespace
    m=module();algo=SimpleNamespace(use_text=True,text_embeds={'task_0004':'cached'})
    result=m.configure_text_conditioning(algo,'task_0004',null_text=False,text_guidance=0.)
    assert algo.use_text and result['task_text_used'] and result['prompt_key']=='task_0004'
    result=m.configure_text_conditioning(algo,'uncached_bottle_task',null_text=True,text_guidance=0.)
    assert not algo.use_text and result=={'mode':'null_context','prompt_key':None,'task_text_used':False}
    with pytest.raises(ValueError,match='Unknown fixed task'):
        m.configure_text_conditioning(algo,'uncached_bottle_task',null_text=False,text_guidance=0.)
    with pytest.raises(ValueError,match='requires'):
        m.configure_text_conditioning(algo,'anything',null_text=True,text_guidance=2.)


def test_custom_text_embedding_binds_correct_task_text_and_completed_receipt(tmp_path):
    import torch
    m=module();key='hocap_pump_bottle';text='Pick up the pump bottle'
    tensor=torch.zeros((6,4096),dtype=torch.bfloat16);tensor[0,0]=-1.  # Embedding values need not be positive.
    data={key:{'text':text,'embed':tensor},'':{'text':'','embed':torch.ones((1,4096),dtype=torch.bfloat16)}}
    path=tmp_path/'text.pt';torch.save(data,path)
    receipt={'status':'complete','key':key,'text':text,'dtype':'bfloat16','embedding_sha256':m.file_sha256(path),
        **{name:'0'*64 for name in ['encoder_sha256','encoder_source_sha256','tokenizer_source_sha256','tokenizer_config_sha256']},
        'schema':{name:{'text':row['text'],'shape':list(row['embed'].shape)} for name,row in data.items()}}
    receipt_path=tmp_path/'receipt.json';receipt_path.write_text(json.dumps(receipt))
    result=m.validate_text_embeddings(path,receipt_path,key)
    assert result['text']==text and result['mode']=='custom_task_prompt'
    with pytest.raises(ValueError,match='matching-key'):
        m.validate_text_embeddings(path,receipt_path,'task_0004')
    receipt['status']='running';receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='completed'):
        m.validate_text_embeddings(path,receipt_path,key)
    receipt['status']='complete';receipt['text']='Press a button';receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='matching text'):
        m.validate_text_embeddings(path,receipt_path,key)


def test_prefix_preserves_camera_requests_and_does_not_resample_whole_clip(tmp_path):
    m=module();original=tmp_path/'original';original.mkdir()
    observed=original/'rgb.dat';observed.write_bytes(b'observed RGB ancestry')
    rgb=np.arange(25*2*4*3,dtype=np.uint8).reshape(25,2,4,3)
    times=np.arange(25)/3.;ids=np.arange(25)*3
    k=np.repeat(np.eye(3)[None],4,axis=0);c=m.virtual_cameras([.1,0.,.8])
    np.savez(original/'condition.npz',rgb=rgb,K=k,c2w=c,timestamps=times,source_frame_indices=ids)
    manifest={'schema_version':1,'condition':'rgb_only','task_mode':'reconstruction','method_inputs':['virtual_camera_requests'],
        'artifacts':[{'id':'rgb','kind':'rgb','origin':'observed_rgb','parent_ids':[],'path':'rgb.dat','sha256':m.file_sha256(observed),
                      'generated':False,'time_range_s':[0,8]},
                     {'id':'virtual_camera_requests','kind':'cameras','origin':'model_prediction','parent_ids':['rgb'],
                      'path':'condition.npz','sha256':m.file_sha256(original/'condition.npz')}],
        'metric_scale':{'status':'unresolved','artifact_ids':[],'description':'Synthetic fixture'},
        'robot_registration':{'status':'unresolved','artifact_ids':[],'description':'Synthetic fixture'}}
    (original/'input-manifest.json').write_text(json.dumps(manifest))
    (original/'preparation.json').write_text(json.dumps({'mode':'reconstruction','condition_sha256':m.file_sha256(original/'condition.npz')}))
    out=tmp_path/'prefix';report=m.prepare_prefix(condition=original,output=out,frames=17)
    with np.load(out/'condition.npz') as z:
        assert np.array_equal(z['rgb'],rgb[:17]) and np.array_equal(z['timestamps'],times[:17])
        assert np.array_equal(z['K'],k) and np.array_equal(z['c2w'],c)
    assert report['conditioned_source_time_range_s']==[0.,16/3.]
    assert report['original_source_time_range_s']==[0.,8.]
    assert report['conditioning_artifact_id']=='prefix_camera_requests'
    child=json.loads((out/'input-manifest.json').read_text())['artifacts'][-1]
    assert child['parent_ids']==['virtual_camera_requests']
