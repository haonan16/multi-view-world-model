"""Prepare hash-verified real RGB clips without exposing evaluator references.

The selection is frozen independently of model results. `plan` reads only an
archive file index, and `extract` copies an exact allowlist from a local archive.
No download, model inference, training, or geometry evaluation is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def safe_relative(path):
    p = PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or str(p) in ('', '.'):
        raise ValueError('Unsafe archive or destination path: ' + str(path))
    return str(p)


def index_records(path):
    """Read the existing archive's JSON/JSONL file-integrity index."""
    with Path(path).open() as stream:
        if Path(path).suffix == '.jsonl':
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        else:
            data = json.load(stream)
            if isinstance(data, dict) and 'files' in data:
                data = data['files']
            yield from data.values() if isinstance(data, dict) else data


def validate_selection(selection):
    if selection.get('schema') != 'real-video-selection-v1':
        raise ValueError('Unknown selection schema')
    if selection.get('condition') != 'rgb_only' or not selection.get('selection_frozen_before_new_fits'):
        raise ValueError('Require a frozen RGB-only selection')
    identifiers, sequences = set(), set()
    for clip in selection['clips']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', clip['id']):
            raise ValueError('Unsafe clip identifier')
        if clip['id'] in identifiers or clip['sequence'] in sequences:
            raise ValueError('Duplicate clip or sequence across the split')
        identifiers.add(clip['id'])
        sequences.add(clip['sequence'])
        safe_relative(clip['sequence'])
        if clip['split'] not in ('train', 'heldout'):
            raise ValueError('Unknown split role')
        if clip['split'] == 'heldout' and clip.get('development_reused'):
            raise ValueError('Development clips cannot be fresh heldout data')
        if clip['fps'] <= 0 or clip['frame_start'] < 0 or clip['frame_end'] < clip['frame_start']:
            raise ValueError('Invalid frame range or FPS')
        cameras = [clip['source_camera'], *clip['evaluator_cameras']]
        if len(cameras) != len(set(cameras)) or not all(re.fullmatch(r'\d+', c) for c in cameras):
            raise ValueError('Expected distinct numeric camera identifiers')
    if not any(c['split'] == 'train' for c in selection['clips']) or not any(c['split'] == 'heldout' for c in selection['clips']):
        raise ValueError('Require both training and heldout sequences')


def frame_ids(clip):
    return list(range(clip['frame_start'], clip['frame_end'] + 1))


def evaluator_frames(clip):
    frames = frame_ids(clip)
    count = min(13, len(frames))
    if count == 1:
        return frames
    return [frames[round(i * (len(frames) - 1) / (count - 1))] for i in range(count)]


def build_plan(selection_path, index_path, output):
    selection = json.loads(Path(selection_path).read_text())
    validate_selection(selection)
    prefix = safe_relative(selection['archive_prefix'])
    wanted, calibration_prefixes = {}, {}
    for clip in selection['clips']:
        key = clip['sequence'].replace('/', '-', 1)
        base = f'{prefix}/{key}/inputs'
        sequence = f"{base}/{clip['sequence']}"
        def add(member, destination, role, required=False, original_frame=None):
            if member in wanted:
                raise ValueError('Duplicate archive selection')
            wanted[safe_relative(member)] = dict(destination=safe_relative(destination),
                role=role, clip_id=clip['id'], required=required, frame_id=original_frame)
        for frame in frame_ids(clip):
            name = f'color_{frame:06d}.jpg'
            add(f"{sequence}/{clip['source_camera']}/{name}",
                f"method/{clip['id']}/rgb/{name}", 'observed_rgb', True, frame)
            for kind, extension in [('depth', 'png'), ('label', 'npz')]:
                name = f'{kind}_{frame:06d}.{extension}'
                add(f"{sequence}/{clip['source_camera']}/{name}",
                    f"evaluator/{clip['id']}/{clip['source_camera']}/{name}", 'evaluator_only', original_frame=frame)
        for camera in clip['evaluator_cameras']:
            for frame in evaluator_frames(clip):
                for kind, extension in [('color', 'jpg'), ('depth', 'png'), ('label', 'npz')]:
                    name = f'{kind}_{frame:06d}.{extension}'
                    add(f'{sequence}/{camera}/{name}', f"evaluator/{clip['id']}/{camera}/{name}",
                        'evaluator_only', original_frame=frame)
        for name in ['meta.yaml', 'poses_m.npy', 'poses_o.npy']:
            add(f'{sequence}/{name}', f"evaluator/{clip['id']}/{name}", 'evaluator_only')
        for camera in [clip['source_camera'], *clip['evaluator_cameras']]:
            name = f'calibration/intrinsics/{camera}.yaml'
            add(f'{base}/{name}', f"evaluator/{clip['id']}/{name}", 'evaluator_only')
        subject = clip['sequence'].split('/')[0]
        name = f'calibration/mano/{subject}.yaml'
        add(f'{base}/{name}', f"evaluator/{clip['id']}/{name}", 'evaluator_only')
        calibration_prefixes[f'{base}/calibration/extrinsics/'] = clip['id']

    files, seen = [], set()
    for record in index_records(index_path):
        member = record.get('path', '').removeprefix('./')
        match = wanted.get(member)
        if match is None:
            for calibration, clip_id in calibration_prefixes.items():
                if member.startswith(calibration) and member.endswith('.yaml') and '/' not in member[len(calibration):]:
                    match = dict(destination=f"evaluator/{clip_id}/calibration/extrinsics/{Path(member).name}",
                                 role='evaluator_only', clip_id=clip_id, required=False, frame_id=None)
                    break
        if match is None or record.get('kind', 'file') != 'file':
            continue
        if member in seen:
            raise ValueError('Duplicate selected member in archive index: ' + member)
        seen.add(member)
        if not re.fullmatch(r'[a-f0-9]{64}', record.get('sha256', '')) or int(record['size']) < 0:
            raise ValueError('Missing valid file hash or size: ' + member)
        files.append(dict(archive_member=safe_relative(member), sha256=record['sha256'],
                          bytes=int(record['size']), **match))
    missing_required = [p for p, r in wanted.items() if r['required'] and p not in seen]
    if missing_required:
        raise ValueError(f'Missing {len(missing_required)} required RGB frames; first: {missing_required[0]}')
    destinations = [r['destination'] for r in files]
    if len(set(destinations)) != len(destinations):
        raise ValueError('Multiple archive members target the same output file')
    plan = dict(schema='real-video-extraction-plan-v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
        selection_sha256=sha256(selection_path), selection=selection,
        archive_index_sha256=sha256(index_path), files=sorted(files, key=lambda r: r['archive_member']),
        missing_optional_evaluator_members=sorted(set(wanted) - seen),
        selected_bytes=sum(r['bytes'] for r in files))
    write_json(output, plan)
    return plan


def validate_plan(plan):
    if plan.get('schema') != 'real-video-extraction-plan-v1':
        raise ValueError('Unknown extraction plan')
    validate_selection(plan['selection'])
    by_id = {c['id']: c for c in plan['selection']['clips']}
    members, destinations = set(), set()
    for record in plan['files']:
        member, destination = safe_relative(record['archive_member']), safe_relative(record['destination'])
        clip = by_id[record['clip_id']]
        if member in members or destination in destinations:
            raise ValueError('Duplicate extraction member or destination')
        members.add(member)
        destinations.add(destination)
        if record['role'] == 'observed_rgb':
            frame = record['frame_id']
            name = f'color_{frame:06d}.jpg'
            key = clip['sequence'].replace('/', '-', 1)
            expected_member = f"{plan['selection']['archive_prefix']}/{key}/inputs/{clip['sequence']}/{clip['source_camera']}/{name}"
            if (destination != f"method/{clip['id']}/rgb/{name}" or member != expected_member
                    or frame not in frame_ids(clip)):
                raise ValueError('Method inputs must be exact selected source RGB frames')
        elif record['role'] != 'evaluator_only' or not destination.startswith(f"evaluator/{clip['id']}/"):
            raise ValueError('Evaluator data cannot enter the method area')
    for clip in by_id.values():
        actual = sorted(r['frame_id'] for r in plan['files'] if r['clip_id'] == clip['id'] and r['role'] == 'observed_rgb')
        if actual != frame_ids(clip):
            raise ValueError('Source RGB coverage differs from the frozen selection')


def extract_plan(plan_path, archive, output, expected_plan_sha256):
    if sha256(plan_path) != expected_plan_sha256:
        raise ValueError('Extraction plan changed after freezing')
    plan = json.loads(Path(plan_path).read_text())
    validate_plan(plan)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    wanted = {r['archive_member']: r for r in plan['files']}
    remaining, records = set(wanted), []
    start = time.monotonic()
    process = None
    archive = Path(archive).resolve()
    if archive.suffix == '.zst':
        process = subprocess.Popen(['zstd', '-qdc', str(archive)], stdout=subprocess.PIPE)
        stream = process.stdout
    else:
        stream = archive.open('rb')
    try:
        with tarfile.open(fileobj=stream, mode='r|*') as tar:
            for member in tar:
                name = member.name.removeprefix('./')
                if name not in remaining:
                    continue
                expected = wanted[name]
                if not member.isfile() or member.size != expected['bytes']:
                    raise ValueError('Archive member type or size differs: ' + name)
                data = tar.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != expected['sha256']:
                    raise ValueError('Archive member hash differs: ' + name)
                target = output / expected['destination']
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as destination:
                    destination.write(data)
                records.append(expected)
                remaining.remove(name)
                if len(records) % 120 == 0:
                    print(f'Verified {len(records)}/{len(wanted)} selected files', flush=True)
                if not remaining:
                    break
    finally:
        stream.close()
        if process is not None:
            process.terminate()
            process.wait()
    if remaining:
        raise ValueError(f'Archive did not contain {len(remaining)} selected members')
    public_clips = []
    for clip in plan['selection']['clips']:
        source_files = sorted((r for r in records if r['clip_id'] == clip['id'] and r['role'] == 'observed_rgb'),
                              key=lambda r: r['frame_id'])
        public = dict(schema='real-video-rgb-source-v1', condition='rgb_only', dataset='HO-Cap',
            clip_id=clip['id'], split=clip['split'], sequence=clip['sequence'],
            task_label=clip['task_label'], source_camera=clip['source_camera'],
            image_size=clip['image_size'], crop=None, frame_ids=frame_ids(clip), fps=clip['fps'],
            timestamps_s=[i / clip['fps'] for i in frame_ids(clip)],
            timestamp_provenance='Nominal original frame ID divided by published 30 fps; not measured acquisition clock',
            development_reused=clip['development_reused'], selection_sha256=plan['selection_sha256'],
            extraction_plan_sha256=expected_plan_sha256,
            observed_inputs=['recorded_source_rgb', 'nominal_timestamps', 'task_label', 'source_provenance'],
            training_target_policy='Only model-derived pseudo targets; no sensor depth, calibration or annotation targets',
            source_provenance=dict(official_source='https://irvlutd.github.io/HOCap/',
                cached_mirror='https://huggingface.co/datasets/pablovela5620/hocap-original',
                archive_index_sha256=plan['archive_index_sha256']),
            files=[dict(path='rgb/' + Path(r['destination']).name, frame_id=r['frame_id'],
                        timestamp_s=r['frame_id'] / clip['fps'], sha256=r['sha256'], bytes=r['bytes'],
                        archive_member=r['archive_member']) for r in source_files])
        manifest = output / 'method' / clip['id'] / 'source-manifest.json'
        write_json(manifest, public)
        public_clips.append(dict(clip_id=clip['id'], split=clip['split'],
                                manifest=f"{clip['id']}/source-manifest.json", manifest_sha256=sha256(manifest)))
        evaluator = output / 'evaluator' / clip['id']
        evaluator.mkdir(parents=True, exist_ok=True)
        write_json(evaluator / 'reference-manifest.json', dict(schema='real-video-evaluator-v1',
            clip=clip, extraction_plan_sha256=expected_plan_sha256, source_manifest_sha256=sha256(manifest),
            evaluator_frame_ids=evaluator_frames(clip),
            files=[r for r in records if r['clip_id'] == clip['id'] and r['role'] == 'evaluator_only'],
            scores=None, status='references_prepared_not_evaluated'))
    write_json(output / 'method' / 'split.json', dict(schema='real-video-method-split-v1',
        selection_sha256=plan['selection_sha256'], clips=public_clips,
        limitations=plan['selection']['limitations']))
    receipt = dict(schema='real-video-extraction-receipt-v1', status='verified',
        archive=str(archive), archive_index_sha256=plan['archive_index_sha256'],
        extraction_plan_sha256=expected_plan_sha256, selection_sha256=plan['selection_sha256'],
        seconds=time.monotonic() - start, files=records, selected_bytes=sum(r['bytes'] for r in records),
        missing_optional_evaluator_members=plan['missing_optional_evaluator_members'])
    write_json(output / 'extraction-receipt.json', receipt)
    return receipt


def pack_method_bundles(method_root):
    """Read only the isolated RGB directory; emit the existing frontend format."""
    import numpy as np
    from PIL import Image

    method_root = Path(method_root).resolve()
    split = json.loads((method_root / 'split.json').read_text())
    results = []
    for entry in split['clips']:
        manifest_path = method_root / safe_relative(entry['manifest'])
        if sha256(manifest_path) != entry['manifest_sha256']:
            raise ValueError('Source manifest changed')
        source = json.loads(manifest_path.read_text())
        directory = manifest_path.parent
        if any((directory / name).exists() for name in ['rgb.npz', 'input-manifest.json', 'extraction-audit.json', 'first-frame.png']):
            raise FileExistsError('Refusing to overwrite an existing frontend bundle')
        frames = []
        if source['frame_ids'] != [r['frame_id'] for r in source['files']]:
            raise ValueError('Source frame order changed')
        for record in source['files']:
            path = directory / safe_relative(record['path'])
            if path.parent != directory / 'rgb' or sha256(path) != record['sha256']:
                raise ValueError('RGB provenance or hash mismatch')
            with Image.open(path) as image:
                if list(image.size) != source['image_size']:
                    raise ValueError('RGB size differs from frozen selection')
                frames.append(np.asarray(image.convert('RGB'), dtype=np.uint8).copy())
        rgb = np.stack(frames)
        timestamps = np.asarray(source['timestamps_s'], dtype=np.float64)
        if timestamps.shape != (len(rgb),) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError('Invalid source timestamps')
        with (directory / 'rgb.npz').open('xb') as destination:
            np.savez_compressed(destination, rgb=rgb, timestamps=timestamps)
        with (directory / 'first-frame.png').open('xb') as destination:
            Image.fromarray(rgb[0]).save(destination, format='PNG')
        manifest = dict(schema_version=1, condition='rgb_only', task_mode='reconstruction',
            method_inputs=['source_rgb'], artifacts=[dict(id='source_rgb', kind='rgb',
            origin='observed_rgb', generated=False, parent_ids=[], path='rgb.npz',
            sha256=sha256(directory / 'rgb.npz'), time_range_s=[float(timestamps[0]), float(timestamps[-1])])],
            metric_scale=dict(status='unresolved', artifact_ids=[], description='RGB input alone'),
            robot_registration=dict(status='unresolved', artifact_ids=[],
                                    description='No target robot base observed in the human demonstration'))
        write_json(directory / 'input-manifest.json', manifest)
        audit = dict(schema='real-video-rgb-packing-v1', source_manifest_sha256=entry['manifest_sha256'],
            selection_sha256=source['selection_sha256'], original_frame_ids=source['frame_ids'],
            source_camera=source['source_camera'], timestamp_source=source['timestamp_provenance'],
            exported_arrays=['rgb', 'timestamps'], exported_resolution_hw=list(rgb.shape[1:3]),
            copied_depth=False, copied_camera_calibration=False, copied_scene_geometry=False,
            rgb_bundle_sha256=manifest['artifacts'][0]['sha256'])
        write_json(directory / 'extraction-audit.json', audit)
        results.append(dict(clip_id=source['clip_id'], split=source['split'], frames=len(rgb),
                            rgb_bundle_sha256=audit['rgb_bundle_sha256']))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--selection', required=True, type=Path)
    p.add_argument('--archive-index', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p = sub.add_parser('extract')
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--archive', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p = sub.add_parser('pack')
    p.add_argument('--method-root', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        result = build_plan(args.selection, args.archive_index, args.output)
        print(json.dumps(dict(plan_sha256=sha256(args.output), selected_files=len(result['files']),
            selected_bytes=result['selected_bytes'], missing_optional_evaluator_files=len(result['missing_optional_evaluator_members']))))
    elif args.command == 'extract':
        result = extract_plan(args.plan, args.archive, args.output, args.plan_sha256)
        print(json.dumps(dict(status=result['status'], selected_files=len(result['files']),
                             selected_bytes=result['selected_bytes'], seconds=result['seconds'])))
    else:
        print(json.dumps(pack_method_bundles(args.method_root)))


if __name__ == '__main__':
    main()
