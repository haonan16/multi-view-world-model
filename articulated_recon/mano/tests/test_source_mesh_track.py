import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def module():
    p = Path(__file__).resolve().parents[2] / 'benchmarks/source_mesh_track.py'
    spec = importlib.util.spec_from_file_location('source_mesh_track', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def observations():
    return dict(rgb=np.zeros((4, 1, 2, 2, 3), np.uint8), depth_m=np.ones((4, 1, 2, 2)),
                K=np.eye(3)[None], c2w=np.eye(4)[None], timestamps=np.array([0., .1, .35, .8]))


@pytest.mark.parametrize('field', ['rgb', 'depth_m', 'K', 'c2w', 'timestamps'])
def test_rejects_changed_hand_to_object_observations(field):
    m = module()
    hand = observations()
    geometry = {k: v.copy() for k, v in hand.items()}
    geometry[field].flat[0] += 1
    with pytest.raises(ValueError, match=field):
        m.equal_observations(hand, geometry)


def test_native_timing_and_missing_evidence_are_preserved():
    m = module()
    hand = observations()
    m.equal_observations(hand, hand)
    ids = m.sample_frames(4, 3)
    assert ids.tolist() == [0, 2, 3]
    assert hand['timestamps'][ids].tolist() == [0., .35, .8]
    assert m.frame_observation_flags([{0: {'P': np.zeros((0, 3))}}, {}, {0: {}}]).tolist() == [True, False, True]
    broken = observations()
    del broken['timestamps']
    with pytest.raises(ValueError, match='timestamps'):
        m.equal_observations(broken, broken)


def test_asset_local_selection_maps_to_global_mask_candidate():
    m = module()
    assert m.global_candidate_row({'selected_index': 1, 'candidate_row_ids': [8, 23, 91]}) == 23
    with pytest.raises(ValueError, match='global row mapping'):
        m.global_candidate_row({'selected_index': 1})


def test_unknown_grasp_does_not_merge_moving_frames_into_shared_shape():
    m = module()
    selected = [{0: {'P': np.zeros((4, 3))}}, {0: {'P': np.ones((4, 3))}}, {0: {'P': np.ones((4, 3)) * 2}}]
    shape = m.initial_shape_evidence(selected)
    assert shape[0] is selected[0]
    assert shape[1:] == [{}, {}]
    with pytest.raises(ValueError, match='First-frame'):
        m.initial_shape_evidence([{}, selected[1]])


def hand_fixture(tmp_path):
    m = module()
    from artirecon.input_protocol import file_sha256, load_manifest
    from mano_recon.run_integrity import output_records
    case, results = tmp_path / 'case', tmp_path / 'results'
    case.mkdir()
    results.mkdir()
    for p in [case / 'rgb.npz', case / 'obs.npz', case / 'protocol.json',
              case / 'frozen-hand.npz', results / 'fit.npz', results / 'scene-sequence.json']:
        p.write_bytes(b'fixture')
    artifacts = [dict(id='rgb', kind='rgb', origin='observed_rgb', generated=False,
                      parent_ids=[], time_range_s=[0, 1], path='rgb.npz', sha256=file_sha256(case / 'rgb.npz'))]
    for aid, kind, path in [('depth', 'depth', 'obs.npz'), ('K', 'cameras', 'obs.npz'), ('table', 'support', 'protocol.json')]:
        artifacts.append(dict(id=aid, kind=kind, origin='model_prediction', parent_ids=['rgb'],
                              path=path, sha256=file_sha256(case / path)))
    manifest = dict(schema_version=1, condition='rgb_only', task_mode='reconstruction',
                    artifacts=artifacts, method_inputs=['depth', 'K', 'table'])
    (case / 'input-manifest.json').write_text(json.dumps(manifest))
    audit = dict(status='complete', input_protocol=load_manifest(case / 'input-manifest.json'),
                 output_artifacts=output_records({'fit': results / 'fit.npz', 'frozen_hand': case / 'frozen-hand.npz',
                                                 'scene_sequence': results / 'scene-sequence.json'}))
    (results / 'execution.json').write_text(json.dumps(audit))
    return m, case, results


def test_hand_stage_authenticates_original_manifest_and_actual_outputs(tmp_path):
    m, case, results = hand_fixture(tmp_path)
    report = m.verify_hand_stage(case, results)
    assert report['condition'] == 'rgb_only'
    (case / 'frozen-hand.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        m.verify_hand_stage(case, results)


def test_hand_stage_cannot_be_rebound_to_a_new_downstream_manifest(tmp_path):
    m, case, results = hand_fixture(tmp_path)
    path = case / 'input-manifest.json'
    manifest = json.loads(path.read_text())
    manifest['experiment_note'] = 'new downstream metadata changes identity'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='original manifest'):
        m.verify_hand_stage(case, results)
