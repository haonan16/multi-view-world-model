"""Registered TCP targets retain estimated object motion without simulator input."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artirecon.input_protocol import ProtocolError, file_sha256
from artirecon.scene_sequence import scene_from_arrays, object_from_arrays
from benchmarks import registered_plan as planner
from benchmarks.freeze_submission import freeze_submission, verify_submission


def provenance():
    return {"method": "Synthetic model-estimate fixture", "artifact_ids": ["depth"], "confidence_status": "uncalibrated"}


def scene():
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    poses[:, :3, :3] = Rotation.from_rotvec(np.c_[np.zeros((5, 2)), [0, 0, .1, .2, .2]]).as_matrix()
    poses[:, :3, 3] = [[0, 0, .025], [0, 0, .025], [.05, 0, .08], [.1, 0, .025], [.1, 0, .025]]
    return scene_from_arrays(
        timestamps=[7., 7.13, 7.6, 8.1, 8.23],
        coordinate_frame={"name": "source_world", "handedness": "right", "up_axis": "+z", "pose_convention": "T_world_local"},
        metric_scale={"status": "estimated", "provenance": provenance()}, actors=[],
        objects=[object_from_arrays(object_id="block", poses=poses, observed=[True] * 5,
                                   primitive={"type": "box", "half_extents_m": [.015, .02, .025]}, provenance=provenance())],
        support={"id": "table", "normal_world": [0, 0, 1], "offset_m": 0, "confidence": None, "provenance": provenance()},
        contacts=[{"id": "grasp", "a": "block", "b": "world", "kind": "grasp", "start_frame": 1, "end_frame": 3,
                   "status": "hypothesized", "confidence": None, "provenance": provenance()},
                  {"id": "rest", "a": "block", "b": "table", "kind": "support", "start_frame": 4, "end_frame": 4,
                   "status": "hypothesized", "confidence": None, "provenance": provenance()}],
        uncertainty={"status": "unestimated"}, task={"description": "Synthetic block transfer", "target": None})


def registration():
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", .4).as_matrix()
    transform[:3, 3] = [.3, .2, .01]
    return {"schema": "deployment-registration-v1", "source_frame": "source_world", "target_frame": "robot_base",
            "T_robot_scene": transform.tolist(), "fixed_across_methods": True,
            "derived_from_evaluation_geometry": False, "deployment_type": "virtual",
            "procedure": "Synthetic fixed virtual deployment; no measured physical robot is claimed"}


def test_preserves_object_se3_geometry_and_actual_timestamps():
    source = scene()
    original = deepcopy(source)
    calibrated = registration()
    arrays, report = planner.build_plan(source, calibrated, planner.default_config("block"))
    expected = np.asarray(calibrated["T_robot_scene"]) @ np.asarray(source["objects"][0]["T_world_object"])
    np.testing.assert_allclose(arrays["T_robot_object"], expected)
    np.testing.assert_array_equal(arrays["object_times_s"], source["timestamps"])
    assert source == original
    assert report["object_pose_corrections_m"] == 0
    assert report["object_geometry_replaced"] is False
    assert report["release"] is True
    assert report["feasibility"]["panda_ik"] == "unmeasured"
    assert report["feasibility"]["robot_collision"] == "unknown"
    assert report["inverse_dynamics_used"] is False
    assert report["simulator_accessed"] is False
    np.testing.assert_allclose([r["source_duration_s"] for r in report["retiming"]], [.47, .5, .13])
    for index in [2, 3, 4]:
        matches = np.flatnonzero(arrays["source_time_valid"] & np.isclose(arrays["source_time_s"], source["timestamps"][index]))
        target = arrays["T_robot_tcp"][matches[0]]
        np.testing.assert_allclose(target, expected[index] @ arrays["T_object_tcp"], atol=1e-7)
    assert np.all(np.diff(arrays["command_time_s"]) > 0)


@pytest.mark.parametrize("mutation, message", [
    (lambda s: s.__setitem__("contacts", []), "Exactly one"),
    (lambda s: s["contacts"][0].__setitem__("status", "unknown"), "Exactly one"),
    (lambda s: s["contacts"].append(dict(s["contacts"][0], id="other")), "Exactly one"),
    (lambda s: s["contacts"][0].__setitem__("end_frame", 1), "at least two"),
    (lambda s: s["contacts"].pop(), "supported release"),
    (lambda s: s["objects"][0]["observed"].__setitem__(2, False), "Unobserved"),
    (lambda s: s["objects"][0]["T_world_object"][1][0].__setitem__(3, .2), "moves before"),
    (lambda s: s["objects"][0]["T_world_object"][3][2].__setitem__(3, -.01), "penetrates"),
    (lambda s: s["objects"][0]["geometry"].__setitem__("half_extents_m", [.1, .1, .025]), "gripper envelope"),
])
def test_rejects_unsupported_estimates_instead_of_repairing(mutation, message):
    source = scene()
    mutation(source)
    with pytest.raises(planner.PlanError, match=message):
        planner.build_plan(source, registration(), planner.default_config("block"))


def test_final_held_state_holds_instead_of_inventing_release():
    source = scene()
    source["contacts"][0]["end_frame"] = 4
    source["contacts"] = source["contacts"][:1]
    arrays, report = planner.build_plan(source, registration(), planner.default_config("block"))
    assert report["release"] is False
    assert arrays["phase"][-1] == "hold"
    assert arrays["gripper_command"][-1] == -1


@pytest.mark.parametrize("change", [
    {"schema": "shared-table-registration-v1"}, {"derived_from_evaluation_geometry": True},
    {"fixed_across_methods": False}, {"source_frame": "another_camera"},
    {"deployment_type": "unspecified"}, {"procedure": ""},
])
def test_rejects_legacy_or_undeclared_deployment(change):
    calibrated = registration()
    calibrated.update(change)
    with pytest.raises(planner.PlanError):
        planner.build_plan(scene(), calibrated, planner.default_config("block"))


def test_rejects_nonrigid_calibration():
    calibrated = registration()
    calibrated["T_robot_scene"][0][0] *= 2
    with pytest.raises(planner.PlanError, match="rigid"):
        planner.build_plan(scene(), calibrated, planner.default_config("block"))


def test_workspace_failure_does_not_recenter():
    config = planner.default_config("block")
    config["workspace_bounds_robot_m"] = [[-1, -1, 0], [.01, .01, 1]]
    with pytest.raises(planner.PlanError, match="no recentering"):
        planner.build_plan(scene(), registration(), config)


def test_negative_source_times_keep_explicit_validity():
    source = scene()
    source["timestamps"] = [-4, -3.87, -3.4, -2.9, -2.77]
    arrays, _ = planner.build_plan(source, registration(), planner.default_config("block"))
    assert np.all(arrays["source_time_s"][arrays["source_time_valid"]] < 0)
    assert arrays["source_time_valid"].sum() > 3
    assert not arrays["source_time_valid"][0]


def write(path, data):
    path.write_text(json.dumps(data))
    return path


def materialized_inputs(tmp_path):
    source = write(tmp_path / "scene.json", scene())
    calibration = write(tmp_path / "registration.json", registration())
    rgb = tmp_path / "rgb.dat"
    rgb.write_text("synthetic RGB bytes")
    depth = tmp_path / "depth.dat"
    depth.write_text("synthetic predicted depth bytes")
    manifest = {
        "schema_version": 1, "condition": "rgb_only", "task_mode": "reconstruction",
        "method_inputs": ["rgb", "depth", "registration"],
        "artifacts": [
            {"id": "rgb", "kind": "rgb", "origin": "observed_rgb", "generated": False, "time_range_s": [7, 8.23],
             "parent_ids": [], "path": rgb.name, "sha256": file_sha256(rgb)},
            {"id": "weights", "kind": "model_prior", "origin": "prior", "scope": "global", "parent_ids": [],
             "prior_type": "learned_model_weights", "revision": "synthetic"},
            {"id": "depth", "kind": "depth", "origin": "model_prediction", "parent_ids": ["rgb", "weights"],
             "path": depth.name, "sha256": file_sha256(depth)},
            {"id": "registration", "kind": "robot_registration", "origin": "model_prediction", "parent_ids": ["rgb", "depth"],
             "path": calibration.name, "sha256": file_sha256(calibration)},
        ],
        "metric_scale": {"status": "estimated", "artifact_ids": ["depth"], "description": "Synthetic model metric scale"},
        "robot_registration": {"status": "estimated", "artifact_ids": ["registration"], "description": "Fixed virtual deployment for tests"},
    }
    return source, write(tmp_path / "inputs.json", manifest), write(tmp_path / "config.json", planner.default_config("block"))


def test_actual_producer_metadata_can_be_frozen_before_evaluator_attachment(tmp_path):
    source, manifest, config = materialized_inputs(tmp_path)
    output = tmp_path / "plan"
    metadata = planner.produce_plan(scene_path=source, input_manifest=manifest, registration_id="registration",
                                    config_path=config, output=output)
    assert metadata["plan_sha256"] == file_sha256(output / "plan.npz")
    assert metadata["planner_sha256"] == file_sha256(planner.__file__)
    assert metadata["input_condition"] == "rgb_only"
    assert metadata["deployment_registration"]["type"] == "virtual"
    assert metadata["deployment_registration"]["status"] == "estimated"
    assert metadata["deployment_registration"]["origin"] == "model_prediction"
    executor = tmp_path / "executor-placeholder.py"
    executor.write_text("# Synthetic sealing test; no simulator execution")
    controller = write(tmp_path / "controller.json", {"target_frame": "robot_base"})
    sealed = freeze_submission(scene=source, input_manifest=manifest, plan=output / "plan.npz",
                             plan_metadata=output / "plan-metadata.json", planner=planner.__file__,
                             executor=executor, controller_config=controller, output=tmp_path / "seal.json")
    assert sealed["evaluation_attached"] is False
    assert verify_submission(tmp_path / "seal.json") == sealed
    with pytest.raises(FileExistsError):
        planner.produce_plan(scene_path=source, input_manifest=manifest, registration_id="registration", config_path=config, output=output)


def test_tampered_calibration_rejected_before_plan(tmp_path):
    source, manifest, config = materialized_inputs(tmp_path)
    (tmp_path / "registration.json").write_text("tampered")
    with pytest.raises(ProtocolError):
        planner.produce_plan(scene_path=source, input_manifest=manifest, registration_id="registration", config_path=config, output=tmp_path / "plan")
    assert not (tmp_path / "plan").exists()


def forecast_scene():
    source=scene()
    obj=source['objects'][0]
    obj['observed']=[True,False,False,False,False]
    obj['prediction_supported']=[False,True,True,True,True]
    obj['prediction_provenance']={'origin':'model_prediction','generated':True,
        'artifact_ids':['depth'],'method':'Synthetic depth fitted to generated future RGB'}
    return source


def test_supported_forecast_preserves_recorded_vs_predicted_flags():
    source=forecast_scene();original=deepcopy(source)
    arrays,report=planner.build_plan(source,registration(),planner.default_config('block'),
                                     task_mode='forecast',observed_until_s=7.)
    assert source==original
    assert arrays['object_observed'].tolist()==[True,False,False,False,False]
    assert arrays['object_prediction_supported'].tolist()==[False,True,True,True,True]
    assert report['task_mode']=='forecast'
    assert report['object_pose_evidence']['observed_until_s']==7.
    expected=np.asarray(registration()['T_robot_scene'])@np.asarray(source['objects'][0]['T_world_object'])
    np.testing.assert_allclose(arrays['T_robot_object'],expected)
    # The robot commands are unchanged by evidence labels; observations are not
    # relabeled to make the existing physical guards accept generated futures.
    recorded,_=planner.build_plan(scene(),registration(),planner.default_config('block'))
    np.testing.assert_allclose(arrays['T_robot_tcp'],recorded['T_robot_tcp'])


@pytest.mark.parametrize('mutation,message',[
    (lambda o:o.pop('prediction_supported'),'complete boolean'),
    (lambda o:o.__setitem__('prediction_supported',[0,1,1,1,1]),'complete boolean'),
    (lambda o:o.__setitem__('prediction_supported',[False,True]),'complete boolean'),
    (lambda o:o['prediction_supported'].__setitem__(0,True),'only after'),
    (lambda o:o['prediction_supported'].__setitem__(2,False),'unsupported predicted'),
    (lambda o:o['observed'].__setitem__(2,True),'relabeled'),
    (lambda o:o['observed'].__setitem__(0,False),'initial object pose'),
    (lambda o:o.pop('prediction_provenance'),'explicit generated'),
    (lambda o:o['prediction_provenance'].__setitem__('generated',False),'explicit generated'),
    (lambda o:o['prediction_provenance'].__setitem__('artifact_ids',[]),'explicit generated'),
])
def test_forecast_missing_or_misdeclared_evidence_is_rejected(mutation,message):
    source=forecast_scene();mutation(source['objects'][0])
    with pytest.raises(planner.PlanError,match=message):
        planner.build_plan(source,registration(),planner.default_config('block'),
                           task_mode='forecast',observed_until_s=7.)


def test_reconstruction_cannot_use_prediction_flags_to_fill_observation_gap():
    with pytest.raises(planner.PlanError,match='Unobserved'):
        planner.build_plan(forecast_scene(),registration(),planner.default_config('block'))


@pytest.mark.parametrize('cutoff',[None,True,float('nan'),6.,7.13])
def test_forecast_requires_actual_single_image_cutoff(cutoff):
    with pytest.raises(planner.PlanError):
        planner.build_plan(forecast_scene(),registration(),planner.default_config('block'),
                           task_mode='forecast',observed_until_s=cutoff)


@pytest.mark.parametrize('mutation,message',[
    (lambda s:s['contacts'].pop(),'supported release'),
    (lambda s:s['objects'][0]['T_world_object'][3][2].__setitem__(3,-.01),'penetrates'),
    (lambda s:s['objects'][0]['T_world_object'][1][0].__setitem__(3,.2),'moves before'),
])
def test_forecast_keeps_contact_support_and_stationary_prefix_guards(mutation,message):
    source=forecast_scene();mutation(source)
    with pytest.raises(planner.PlanError,match=message):
        planner.build_plan(source,registration(),planner.default_config('block'),
                           task_mode='forecast',observed_until_s=7.)


def forecast_inputs(tmp_path):
    source,manifest_path,config=materialized_inputs(tmp_path)
    write(source,forecast_scene())
    manifest=json.loads(manifest_path.read_text())
    manifest.update(task_mode='forecast',observed_until_s=7.)
    manifest['artifacts'][0]['time_range_s']=[7.,7.]
    future=tmp_path/'future.dat';future.write_text('Synthetic generated future RGB')
    manifest['artifacts'].append(dict(id='future_rgb',kind='rgb',origin='model_prediction',
        generated=True,parent_ids=['rgb','weights'],path=future.name,sha256=file_sha256(future),
        time_range_s=[7.13,8.23]))
    next(a for a in manifest['artifacts'] if a['id']=='depth')['parent_ids'].append('future_rgb')
    # Registration stays anchored to initial RGB, never the future depth track.
    next(a for a in manifest['artifacts'] if a['id']=='registration')['parent_ids']=['rgb']
    write(manifest_path,manifest)
    return source,manifest_path,config


def test_compile_uses_manifest_forecast_mode_and_authenticates_future_lineage(tmp_path):
    source,manifest,config=forecast_inputs(tmp_path)
    report=planner.produce_plan(scene_path=source,input_manifest=manifest,registration_id='registration',
                                config_path=config,output=tmp_path/'plan')
    assert report['task_mode']=='forecast'
    assert report['object_pose_evidence']['observed']==[True,False,False,False,False]
    assert report['object_pose_evidence']['prediction_supported']==[False,True,True,True,True]
    executor=tmp_path/'executor-placeholder.py';executor.write_text('# Synthetic seal only')
    controller=write(tmp_path/'controller.json',{'target_frame':'robot_base'})
    sealed=freeze_submission(scene=source,input_manifest=manifest,plan=tmp_path/'plan/plan.npz',
        plan_metadata=tmp_path/'plan/plan-metadata.json',planner=planner.__file__,executor=executor,
        controller_config=controller,output=tmp_path/'forecast-seal.json')
    assert sealed['input_protocol']['task_mode']=='forecast'
    assert verify_submission(tmp_path/'forecast-seal.json')==sealed


@pytest.mark.parametrize('part',['missing_id','initial_depth_only','short_horizon'])
def test_compile_rejects_prediction_claims_without_consumed_future_rgb(tmp_path,part):
    source,manifest_path,config=forecast_inputs(tmp_path)
    if part=='missing_id':
        value=json.loads(source.read_text());value['objects'][0]['prediction_provenance']['artifact_ids']=['unknown']
        write(source,value)
    else:
        manifest=json.loads(manifest_path.read_text())
        if part=='initial_depth_only':
            next(a for a in manifest['artifacts'] if a['id']=='depth')['parent_ids'].remove('future_rgb')
        else:
            next(a for a in manifest['artifacts'] if a['id']=='future_rgb')['time_range_s'][1]=8.1
        write(manifest_path,manifest)
    with pytest.raises(ProtocolError,match='Forecast pose'):
        planner.produce_plan(scene_path=source,input_manifest=manifest_path,registration_id='registration',
                             config_path=config,output=tmp_path/'plan')
    assert not (tmp_path/'plan').exists()
