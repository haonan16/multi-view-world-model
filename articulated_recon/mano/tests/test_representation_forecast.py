"""Synthetic protocol and copied-tool smoke tests, no model/agent requests."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
import representation_forecast as rf
import representation_agent_tools as checks
import representation_agent_harness as harness
from test_forecast_geometry import full_input_fixture


def paired(tmp_path,branch):
    args=full_input_fixture(tmp_path,branch);rf.fg.run(**args)
    cfg=rf.default_config('gpt-6.2-sol',sys.executable);cfg['max_reported_tokens']=1000000
    out=tmp_path/'pair'
    result=rf.pair(**{k:args[k] for k in ['condition','helper','predictions','masks']},initializer=args['output'],output=out,config=cfg)
    return args,out,result


@pytest.mark.parametrize('branch',['source_unposed','helper_unposed'])
def test_forecast_pair_raw_scale_independence_and_export(tmp_path,branch):
    args,out,result=paired(tmp_path,branch)
    a,b=out/'A_raw',out/'B_object'
    assert result['status']=='prepared'
    for filename in ['observations.npz','raw-predictions.npz','raw-masks.npz','first-image-scene.json','optimizer-config.json','PROMPT.txt']:
        assert (a/filename).read_bytes()==(b/filename).read_bytes()
    assert not (a/'initializer-scene.json').exists()
    meta,obs=checks.load_inputs(a);native=rf.read_npz(args['predictions']/'predictions.npz')
    assert obs['source_frame_indices'].tolist()==[0,-1,-1,-1,-1]
    np.testing.assert_allclose(obs['depth_m'],native['depth']*meta['forecast']['common_depth_scale'])
    assert obs['observed_mask'].sum()==1
    suite=harness.prepare_suite([a,b],tmp_path/'suite');assert len(suite['runs'])==2
    report=checks.check_scene(b,b/'initializer-scene.json',b/'output/check')
    assert len(report['per_frame'])==len(native['timestamps'])*native['rgb'].shape[1]
    assert all('candidate_scale_diagnostics' in r for r in report['per_frame'])
    scene=json.loads((b/'initializer-scene.json').read_text())
    scene['objects'][0]['observed'][1]=True
    with pytest.raises(ValueError,match='cannot be labeled observed'):checks.validate_export(scene,meta)
    exported=tmp_path/'export';rf.export_bundle(packet=b,scene=b/'initializer-scene.json',output=exported)
    import render_forecast_reconstruction as renderer
    renderer.load_inputs(out/'controller/helper',exported)


def test_copied_source_optimizer_runs_without_repository_imports(tmp_path):
    args,out,_=paired(tmp_path,'source_unposed');a=out/'A_raw'
    command=[sys.executable,'-I',str(a/'scene_tools.py'),'optimize','--output',str(a/'output/fit')]
    result=subprocess.run(command,cwd=a,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    scene=json.loads((a/'output/fit/scene-sequence.json').read_text())
    checks.validate_export(scene,json.loads((a/'packet.json').read_text()))
    assert scene['objects'][0]['prediction_supported'][0] is False


def test_relocation_preserves_source_manifest_bytes(tmp_path):
    args=full_input_fixture(tmp_path,'source_unposed');rf.fg.run(**args)
    old=tmp_path.resolve();remote=Path('/remote/strict-forecast-fixture')
    # Simulate byte-identical downloaded remote manifests and bound preparation.
    manifests=list(tmp_path.rglob('input-manifest.json'))
    for path in manifests:
        value=json.loads(path.read_text())
        for a in value['artifacts']:
            if 'path' in a and Path(a['path']).is_absolute():a['path']=str(remote/Path(a['path']).relative_to(old))
        rf.write_json(path,value)
    prep_path=args['condition']/'preparation.json';prep=json.loads(prep_path.read_text())
    for key in ['geometry_manifest','scene_manifest','scene_path']:
        local=Path(prep[key]);prep[key]=str(remote/local.relative_to(old))
        prep['scene_sha256' if key=='scene_path' else key+'_sha256']=rf.file_sha256(local)
    rf.write_json(prep_path,prep)
    hashes={p:rf.file_sha256(p) for p in manifests+[prep_path]}
    cfg=rf.default_config('gpt-6.2-sol',sys.executable)
    rf.pair(**{k:args[k] for k in ['condition','helper','predictions','masks']},initializer=args['output'],
        output=tmp_path/'relocated',config=cfg,remote_prefix=remote,local_prefix=old)
    assert all(rf.file_sha256(p)==sha for p,sha in hashes.items())
