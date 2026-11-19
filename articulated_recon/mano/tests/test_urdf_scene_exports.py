"""Native URDF exports must not invent timing, observations or registration."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artirecon.scene_sequence import SceneValidationError, validate_scene
from artirecon.urdf_scene_exports import urdf_fit_scene, native_timestamps, export_native_urdf


def arrays():
    base = np.repeat(np.eye(4)[None], 3, axis=0)
    base[:, :3, 3] = [[.1, .2, .3], [.1, .2, .31], [.1, .2, .33]]
    return dict(q=np.array([[.4, .02], [.5, .01], [.6, .03]]), base=base,
                joint_names=['shoulder_revolute', 'finger_prismatic'], model_uri='/local/robot.urdf',
                timestamps=[3., 3.1, 3.37], observed=np.array([True, False, True]),
                coordinate_frame=dict(name='estimated_source_camera', handedness='right', up_axis='unknown', pose_convention='T_world_local'))


def test_preserves_native_joint_order_base_motion_and_nonuniform_timestamps():
    data = arrays(); scene = urdf_fit_scene(**data)
    assert validate_scene(scene)['actors'] == 1
    actor = scene['actors'][0]
    np.testing.assert_array_equal(actor['q'], data['q'])
    np.testing.assert_array_equal(actor['T_world_base'], data['base'])
    assert actor['joint_names'] == data['joint_names']
    assert scene['timestamps'] == data['timestamps']
    assert actor['observed'] == [True, False, True]
    assert scene['coordinate_frame'] == data['coordinate_frame']
    assert scene['metric_scale']['status'] == 'unresolved'
    assert scene['input_protocol']['status'] == 'unverified'
    assert scene['contacts'] == []
    assert not scene['export_scope']['robot_registration_established_by_export']


def test_absent_observation_flags_do_not_become_all_true():
    data = arrays(); data.pop('observed')
    scene = urdf_fit_scene(**data)
    assert scene['actors'][0]['observed'] == [False] * 3
    assert scene['export_scope']['unknown_original_observation_evidence'] is True


def test_only_materialized_validated_input_report_can_carry_metric_status():
    data = arrays()
    report = dict(status='validated', files_verified=True, condition='rgb_only', method_artifact_ids=['predicted_depth'],
                  metric_scale=dict(status='estimated', artifact_ids=['predicted_depth'], description='Learned metric gauge'),
                  robot_registration=dict(status='unresolved'))
    scene = urdf_fit_scene(**data, input_protocol=report)
    assert scene['metric_scale']['status'] == 'estimated'
    assert scene['input_protocol']['robot_registration']['status'] == 'unresolved'
    report['files_verified'] = False
    assert urdf_fit_scene(**data, input_protocol=report)['metric_scale']['status'] == 'unresolved'


@pytest.mark.parametrize('change', [lambda d: d.update(joint_names=['one']),
                                    lambda d: d['base'].__setitem__((1, 0, 0), 2.),
                                    lambda d: d.update(timestamps=[0, 0, 1]),
                                    lambda d: d.update(observed=[1, 0, 1])])
def test_malformed_state_contract_is_rejected(change):
    data = arrays(); change(data)
    with pytest.raises(SceneValidationError): urdf_fit_scene(**data)


def test_native_control_steps_need_an_explicit_frequency():
    result = dict(robot=dict(q=np.zeros((3, 2))), steps=[4, 7, 19])
    with pytest.raises(SceneValidationError, match='No timestamps'):
        native_timestamps(result, {})
    times, method = native_timestamps(result, {'control_hz':20})
    np.testing.assert_array_equal(times, [.2, .35, .95])
    assert 'explicitly declared' in method
    result['timestamps'] = [11., 11.1, 11.8]
    times, _ = native_timestamps(result, {'control_hz':20})
    np.testing.assert_array_equal(times, result['timestamps'])
    with pytest.raises(SceneValidationError, match='disagree'):
        native_timestamps(result, {'timestamps':[0, 1, 2]})


def test_runner_hook_preserves_native_fit_and_uses_actual_robot_joint_names(tmp_path):
    data = arrays()
    result = dict(robot=dict(q=data['q'], base=data['base'], observed=data['observed']), timestamps=data['timestamps'],
                  coordinate_frame=data['coordinate_frame'], label='native', objects=[{'old':'retained'}],
                  plane=(np.array([0., 0., 1.]), -.01))
    before = deepcopy(result)
    robot = SimpleNamespace(active=[{'name':name} for name in data['joint_names']], urdf=Path(data['model_uri']))
    receipt = export_native_urdf(result, robot, tmp_path)
    assert receipt['status'] == 'exported'
    scene = json.loads((tmp_path/'scene-sequence.json').read_text())
    assert scene['actors'][0]['joint_names'] == data['joint_names']
    assert scene['native_source']['native_object_count'] == 1
    assert scene['support']['offset_m'] == -.01
    np.testing.assert_array_equal(result['robot']['q'], before['robot']['q'])
    assert result['objects'] == before['objects']


def test_metadata_cannot_upgrade_legacy_protocol_or_invent_a_timebase(tmp_path):
    data = arrays(); robot = SimpleNamespace(active=[{'name':n} for n in data['joint_names']], urdf=data['model_uri'])
    result = dict(robot=dict(q=data['q'], base=data['base']), steps=[2, 4, 9])
    receipt = export_native_urdf(result, robot, tmp_path)
    assert receipt['status'] == 'not_exported'
    assert not (tmp_path/'scene-sequence.json').exists()
    (tmp_path/'scene-export-metadata.json').write_text(json.dumps(dict(control_hz=10, coordinate_frame=data['coordinate_frame'],
        input_protocol=dict(status='validated', files_verified=True, metric_scale=dict(status='known')))))
    receipt = export_native_urdf(result, robot, tmp_path)
    assert receipt['status'] == 'exported'
    scene = json.loads((tmp_path/'scene-sequence.json').read_text())
    assert scene['timestamps'] == [.2, .4, .9]
    assert scene['input_protocol']['status'] == 'unverified'
    assert scene['metric_scale']['status'] == 'unresolved'


def test_malformed_optional_sidecar_does_not_abort_native_runner(tmp_path):
    (tmp_path/'scene-export-metadata.json').write_text('{bad json')
    native = tmp_path/'fit.pkl'; native.write_bytes(b'native fit preserved')
    receipt = export_native_urdf(dict(robot=dict(q=np.zeros((1, 1)))), SimpleNamespace(), tmp_path)
    assert receipt['status'] == 'not_exported'
    assert native.read_bytes() == b'native fit preserved'
