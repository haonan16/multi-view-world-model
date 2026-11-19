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


def test_rgb_identity_preserves_unique_candidates_but_marks_ambiguity_unknown():
    m = module()
    mask = np.zeros((16, 16), bool)
    mask[4:12, 4:12] = True
    data = dict(prompts=['pump bottle', 'hand'], p=np.array([0, 1, 0, 0, 0]), t=np.array([0, 0, 1, 1, 2]),
                score=np.ones(5)*.9, mask=np.stack([np.packbits(mask)]*5), shape=[16, 16], timestamps=[0, .1, .2])
    rows, audit = m.rgb_identity_track(data, 'pump bottle', 0)
    assert rows == [0, None, 4]
    assert audit[1]['status'] == 'ambiguous_or_missing_candidate'
    assert all('depth' not in k for record in audit for k in record)


def test_mask_first_association_keeps_2d_evidence_when_depth_is_missing():
    m = module()
    class Fitter:
        def instances(self, prompt, t, view):
            return [dict(i=t, m=np.ones((4, 4), bool), P=np.ones((20, 3)) if t == 0 else np.zeros((0, 3)),
                         c=np.array([0., 0, 1.]) if t == 0 else None)]
    selected, centres, held, evidence = m.mask_first_associations(Fitter(), [3, 7], [0, 1], [3, 7], 'pump bottle')
    assert all(selected) and not held.any()
    np.testing.assert_array_equal(centres, [[0, 0, 1], [0, 0, 1]])
    assert evidence[1]['usable_2d_mask'] is True
    assert evidence[1]['depth_points'] == 0 and evidence[1]['depth_centroid_available'] is False


def test_hand_occlusion_is_unknown_even_when_predicted_depth_says_free_space():
    import torch
    from types import SimpleNamespace
    from mano_recon.amani_objects.geom import DepthViews, depth_term
    m = module()
    obs = SimpleNamespace(depth=np.ones((1, 1, 5, 5))*2, K=np.array([[[1, 0, 2], [0, 1, 2], [0, 0, 1]]]),
                          c2w=np.eye(4)[None], T=1)
    mask = np.zeros((5, 5), bool)
    mask[2, 2] = True
    views = m.make_occlusion_views(obs, {0: mask})
    plain = DepthViews(obs.depth, obs.K, obs.c2w)
    points = torch.tensor([[[0., 0, 1.]]])
    normals = torch.tensor([[[0., 0, -1.]]])
    assert depth_term(plain, points, normals, [0]).item() > .9
    assert depth_term(views, points, normals, [0]).item() == 0
    u, v, z = views.project(points)
    # This positive-depth pixel would not be z-test occluded: D=2, z=1.
    d, _ = views.sample(u, v, [0])
    assert not ((d > 0) & (d < z-.015)).any()
    assert views.occluded(u, v, [0], channel='silhouette').all()
    assert views.ignored_queries['silhouette'] == 1


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


@pytest.mark.parametrize('association', ['author_depth', 'rgb_identity'])
def test_actual_author_pose_objective_runs_and_keeps_mesh_without_asserting_grasp(tmp_path, association):
    """Exercise the real fitter on tiny synthetic data, not a mocked success."""
    import trimesh
    from PIL import Image
    from artirecon.input_protocol import file_sha256, load_manifest
    from artirecon.scene_exports import mano_fit_scene
    from mano_recon.run_integrity import output_records
    m = module()
    geometry, masks, asset, hand_run = [tmp_path / n for n in ['geometry', 'masks', 'asset', 'hand']]
    case, results = hand_run / 'inputs/bottle', hand_run / 'hand-results/bottle'
    for p in [geometry, masks, asset, case, results]:
        p.mkdir(parents=True)
    rgb = np.zeros((2, 1, 32, 32, 3), np.uint8)
    times = np.array([0., .1])
    mask = np.zeros((32, 32), bool)
    mask[8:24, 8:24] = True
    depth = np.full((2, 1, 32, 32), 1.5, np.float32)
    depth[:, 0, mask] = 1.3
    k = np.array([[[100., 0, 16], [0, 100., 16], [0, 0, 1]]])
    data = dict(rgb=rgb, depth_m=depth, K=k, c2w=np.eye(4)[None], timestamps=times)
    np.savez(geometry / 'rgb.npz', rgb=rgb[:, 0], timestamps=times)
    np.savez(geometry / 'geometry.npz', **data)
    np.savez(case / 'obs.npz', **data)
    np.savez(masks / 'masks.npz', p=[0, 0], t=[0, 1], score=[.9, .9], box=[[8, 8, 24, 24]]*2,
             mask=np.stack([np.packbits(mask)]*2), prompts=['pump bottle', 'hand'], shape=[32, 32], timestamps=times)
    Image.fromarray(rgb[0, 0]).save(asset / 'source.png')
    Image.fromarray(mask.astype(np.uint8)*255).save(asset / 'mask.png')
    mesh = trimesh.creation.box(extents=[.2, .2, .2])
    mesh.export(asset / 'object.ply')
    transform = np.eye(4)
    transform[2, 3] = 1.3
    (asset / 'object.json').write_text(json.dumps({'T_obj2cam_opencv': transform.tolist()}))
    (asset / 'selection.json').write_text(json.dumps(dict(selected_index=0, candidate_row_ids=[0], frame_index=0,
        source_rgb_sha256=file_sha256(geometry/'rgb.npz'), object_prompt='pump bottle')))
    (case / 'protocol.json').write_text(json.dumps({'table': {'normal': [0, 0, -1], 'offset': 1.5}}))
    roots = [dict(id='source_rgb', kind='rgb', origin='observed_rgb', generated=False,
                  parent_ids=[], time_range_s=[0, .1], path=str(geometry/'rgb.npz'), sha256=file_sha256(geometry/'rgb.npz'))]
    def derived(aid, kind, path, parents):
        return dict(id=aid, kind=kind, origin='model_prediction', parent_ids=parents,
                    path=str(path), sha256=file_sha256(path))
    depth_arts = [derived('predicted_geometry', 'depth', geometry/'geometry.npz', ['source_rgb']),
                  derived('predicted_cameras', 'cameras', geometry/'geometry.npz', ['source_rgb'])]
    mask_arts = [derived('sam3_masks', 'masks', masks/'masks.npz', ['source_rgb'])]
    asset_arts = [derived('sam3d_canonical_mesh', 'object_geometry', asset/'object.ply', ['source_rgb', 'sam3_masks']),
                  derived('sam3d_pose', 'object_geometry', asset/'object.json', ['source_rgb', 'sam3_masks']),
                  derived('selected_object_mask', 'masks', asset/'mask.png', ['sam3_masks'])]
    hand_arts = [derived('hand_depth', 'depth', case/'obs.npz', ['predicted_geometry']),
                 derived('hand_camera', 'cameras', case/'obs.npz', ['predicted_cameras']),
                 derived('hand_support', 'support', case/'protocol.json', ['predicted_geometry'])]
    for folder, arts, inputs in [(geometry, roots+depth_arts, depth_arts), (masks, roots+mask_arts, mask_arts),
                                 (asset, roots+mask_arts+asset_arts, asset_arts),
                                 (case, roots+depth_arts+hand_arts, hand_arts)]:
        manifest = dict(schema_version=1, condition='rgb_only', task_mode='reconstruction', artifacts=arts,
                        method_inputs=[a['id'] for a in inputs])
        (folder/'input-manifest.json').write_text(json.dumps(manifest))
    report = load_manifest(case/'input-manifest.json')
    q, base = np.zeros((2, 15)), np.repeat(np.eye(4)[None], 2, axis=0)
    np.savez(case/'frozen-hand.npz', q=q, base=base, points=np.full((2, 4, 3), 1000.),
             normals=np.ones((2, 4, 3)), hand_frames=base, pinch_points=np.full((2, 3), 1000.),
             pinch_width=np.full(2, .15), visible=np.zeros(2, bool))
    np.savez(results/'fit.npz', q=q)
    scene = mano_fit_scene(q=q, base=base, betas=np.zeros((1, 10)), timestamps=times,
        point_counts=[0, 0], visible=[False, False], model_uri='/fixture/MANO_RIGHT.npz',
        input_protocol=report, support_normal=[0, 0, -1], support_offset=1.5,
        coordinate_frame=dict(name='source_camera', handedness='right', up_axis='unknown', pose_convention='T_world_local'))
    (results/'scene-sequence.json').write_text(json.dumps(scene))
    audit = dict(status='complete', input_protocol=report,
                 output_artifacts=output_records({'fit': results/'fit.npz', 'frozen_hand': case/'frozen-hand.npz',
                                                 'scene_sequence': results/'scene-sequence.json'}))
    (results/'execution.json').write_text(json.dumps(audit))
    output = tmp_path/'output'
    result = m.run(geometry=geometry, masks=masks, asset=asset, hand_run=hand_run, case='bottle', output=output,
                   keyframes=2, shape_iters=1, pose_iters=1, samples=30, association=association)
    assert result['status'] == 'complete'
    assert result['retained_sam3d_faces'] == len(mesh.faces)
    assert result['shape_evidence_native_frames'] == [0]
    assert result['planning_ready'] is False and result['resting_attraction_used'] is False
    assert result['shared_pose_extra_frames'] == []
    assert result['association_policy'] == association
    load_manifest(output/'input-manifest.json', [('object_geometry', output/'scene-sequence.json')])
