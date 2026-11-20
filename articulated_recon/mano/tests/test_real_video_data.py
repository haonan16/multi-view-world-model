import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import pytest


def load_module(name):
    path = Path(__file__).resolve().parents[2] / 'benchmarks' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_archive(tmp_path):
    module = load_module('real_video_data')
    selection = dict(schema='real-video-selection-v1', condition='rgb_only',
        selection_frozen_before_new_fits=True, archive_prefix='shortlist', limitations=[], clips=[])
    files = {}
    for n, role in enumerate(['train', 'heldout']):
        clip = dict(id=f'clip{n}', split=role, sequence=f'subject_{n}/seq',
            task_label='Pick up object', source_camera='123', evaluator_cameras=['456'],
            frame_start=10, frame_end=11, fps=30, image_size=[8, 6], development_reused=False)
        selection['clips'].append(clip)
        base = f"shortlist/subject_{n}-seq/inputs/{clip['sequence']}"
        for frame in [10, 11]:
            buffer = io.BytesIO()
            Image.new('RGB', (8, 6), color=(10 + n, frame, 30)).save(buffer, format='JPEG')
            files[f'{base}/123/color_{frame:06d}.jpg'] = buffer.getvalue()
            files[f'{base}/123/depth_{frame:06d}.png'] = b'privileged measured depth'
        files[f'{base}/456/color_000010.jpg'] = b'privileged heldout view'
        files[f'{base}/meta.yaml'] = b'privileged sequence geometry metadata'
    records = [dict(path=p, size=len(data), kind='file', sha256=hashlib.sha256(data).hexdigest())
               for p, data in files.items()]
    selection_path, index_path, archive = [tmp_path / p for p in ['selection.json', 'index.jsonl', 'archive.tar']]
    selection_path.write_text(json.dumps(selection))
    index_path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    with tarfile.open(archive, 'w') as tar:
        for name, data in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    plan_path = tmp_path / 'plan.json'
    plan = module.build_plan(selection_path, index_path, plan_path)
    return module, selection, index_path, archive, plan_path, plan


def test_extract_and_pack_retains_rgb_only_boundary_and_original_times(tmp_path):
    module, _, _, archive, plan_path, plan = fixture_archive(tmp_path)
    output = tmp_path / 'data'
    receipt = module.extract_plan(plan_path, archive, output, module.sha256(plan_path))
    assert receipt['status'] == 'verified'
    assert plan['missing_optional_evaluator_members']
    assert not any(p.name.startswith(('depth_', 'label_')) for p in (output / 'method').rglob('*'))
    assert (output / 'evaluator/clip0/123/depth_000010.png').is_file()
    public = (output / 'method/clip0/source-manifest.json').read_text()
    assert '456' not in public and 'evaluator/' not in public
    module.pack_method_bundles(output / 'method')
    frontend = load_module('rgb_frontend')
    manifest, _, rgb, times = frontend.load_source_bundle(output / 'method/clip0')
    assert rgb.shape == (2, 6, 8, 3)
    assert manifest['method_inputs'] == ['source_rgb']
    np.testing.assert_array_equal(times, np.array([10 / 30, 11 / 30]))
    with np.load(output / 'method/clip0/rgb.npz') as data:
        assert set(data.files) == {'rgb', 'timestamps'}
    with pytest.raises(FileExistsError):
        module.pack_method_bundles(output / 'method')


def test_missing_rgb_blocks_plan_instead_of_changing_selection(tmp_path):
    module, _, index_path, _, _, _ = fixture_archive(tmp_path)
    lines = index_path.read_text().splitlines()
    index_path.write_text('\n'.join(lines[1:]))
    with pytest.raises(ValueError, match='required RGB'):
        module.build_plan(tmp_path / 'selection.json', index_path, tmp_path / 'missing.json')


def test_reference_cannot_be_redirected_into_method_directory(tmp_path):
    module, _, _, _, _, plan = fixture_archive(tmp_path)
    reference = next(r for r in plan['files'] if r['role'] == 'evaluator_only')
    reference['destination'] = 'method/clip0/sensor.npz'
    with pytest.raises(ValueError, match='Evaluator data'):
        module.validate_plan(plan)
    reference['role'] = 'observed_rgb'
    reference['frame_id'] = 10
    with pytest.raises(ValueError, match='exact selected source RGB'):
        module.validate_plan(plan)


def test_sequence_overlap_and_development_heldout_are_rejected(tmp_path):
    module, selection, *_ = fixture_archive(tmp_path)
    changed = copy.deepcopy(selection)
    changed['clips'][1]['sequence'] = changed['clips'][0]['sequence']
    with pytest.raises(ValueError, match='Duplicate clip or sequence'):
        module.validate_selection(changed)
    selection['clips'][1]['development_reused'] = True
    with pytest.raises(ValueError, match='Development clips'):
        module.validate_selection(selection)


def test_plan_and_archive_hash_changes_block_extraction(tmp_path):
    module, _, _, archive, plan_path, plan = fixture_archive(tmp_path)
    digest = module.sha256(plan_path)
    plan_path.write_text(plan_path.read_text() + ' ')
    with pytest.raises(ValueError, match='plan changed'):
        module.extract_plan(plan_path, archive, tmp_path / 'bad-plan', digest)
    plan['files'][0]['sha256'] = '0' * 64
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='hash differs'):
        module.extract_plan(plan_path, archive, tmp_path / 'bad-archive', module.sha256(plan_path))


def test_changed_rgb_blocks_frontend_packing(tmp_path):
    module, _, _, archive, plan_path, _ = fixture_archive(tmp_path)
    output = tmp_path / 'data'
    module.extract_plan(plan_path, archive, output, module.sha256(plan_path))
    (output / 'method/clip0/rgb/color_000010.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        module.pack_method_bundles(output / 'method')
