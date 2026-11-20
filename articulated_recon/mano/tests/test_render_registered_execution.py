"""Offline log/coordinate checks. FakePlant below is not a physics result."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from artirecon.input_protocol import file_sha256
from benchmarks import render_registered_execution as renderer
from benchmarks import execute_registered as executor
from test_execute_registered import sealed, FakePlant
from test_freeze_submission import case


@pytest.mark.parametrize('normal,offset', [([0,0,1],.07),([.6,0,.8],-.12),([0,1,0],.4),([0,0,-2],-.3)])
def test_visual_support_preserves_native_plane_tilt_and_offset(normal,offset):
    vertices,faces=renderer.support_mesh(normal,offset,cells=3)
    np.testing.assert_allclose(vertices@np.asarray(normal)+offset,0,atol=1e-12)
    assert vertices.shape==(16,3) and faces.shape==(18,3)
    assert faces.min()==0 and faces.max()==15


def log_rows():
    return [dict(step=i,time_s=(i+1)/20,qpos=[0.]*9,object_pose=np.eye(4).tolist(),
                 tcp=np.eye(4).tolist(),phase='transport') for i in range(3)]


@pytest.mark.parametrize('mutation',[
    lambda rows:rows.pop(1),
    lambda rows:rows[1].update(time_s=.11),
    lambda rows:rows[1].update(step=3),
    lambda rows:rows[1].update(qpos=[0.]*8),
    lambda rows:rows[1]['object_pose'][0].__setitem__(0,1.2),
])
def test_cannot_present_missing_or_retimed_states_as_native_20hz(mutation):
    rows=log_rows();mutation(rows)
    with pytest.raises(ValueError):renderer.validate_records(rows)


def test_log_validation_does_not_modify_recorded_states():
    rows=log_rows();before=deepcopy(rows)
    assert renderer.validate_records(rows).tolist()==[.05,.1,.15]
    assert rows==before


def fake_execution(case,monkeypatch):
    root=case['output'].parent
    urdf=root/'synthetic-robot.urdf';urdf.write_text('<robot name="synthetic-test"/>')
    original=executor.default_config
    def synthetic_config(target_frame='panda_base'):
        config=original(target_frame);config['robot_urdf_sha256']=file_sha256(urdf)
        return config
    monkeypatch.setattr(executor,'default_config',synthetic_config)
    payload,config,receipt=sealed(case)
    monkeypatch.setattr(executor,'_runtime_versions',lambda config:{'test':'fake plant only'})
    monkeypatch.setattr(executor,'make_environment',FakePlant)
    output=root/'fake-native-logs'
    executor.run(submission=case['output'],evaluation_receipt=receipt,output=output)
    monkeypatch.setattr(executor,'make_environment',lambda *a:pytest.fail('Renderer must never construct a simulator'))
    return dict(submission=case['output'],evaluation_receipt=receipt,execution=output,urdf=urdf)


def test_completed_log_reader_uses_bound_recorded_states_without_simulator(case,monkeypatch):
    args=fake_execution(case,monkeypatch)
    frozen,_,_,summary,rows=renderer.load_completed(**args)
    assert frozen['input_protocol']['condition']=='rgb_only'
    assert summary['steps']==len(rows)==47
    assert rows[0]['time_s']==.05 and rows[-1]['time_s']==2.35


@pytest.mark.parametrize('part',['records','seal_binding','claim','incomplete'])
def test_completed_reader_rejects_tampering_or_incomplete_execution(case,monkeypatch,part):
    args=fake_execution(case,monkeypatch)
    root=args['execution']
    if part=='records':
        path=root/'records.json';rows=json.loads(path.read_text());rows[0]['object_pose'][0][3]+=.1;path.write_text(json.dumps(rows))
    elif part in {'seal_binding','incomplete'}:
        path=root/'summary.json';value=json.loads(path.read_text())
        value['submission_sha256' if part=='seal_binding' else 'status']='wrong';path.write_text(json.dumps(value))
    else:
        path=root/'execution-inputs.json';value=json.loads(path.read_text());value['claim']['output']=str(root/'another-run');path.write_text(json.dumps(value))
    with pytest.raises(ValueError):renderer.load_completed(**args)
