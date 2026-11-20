"""One explicitly bounded Pi4 fallback; only enumerated RGB/code files upload.

Run once with Modal profile surajmvr14. Dependencies and weights build on CPU.
The single L40S call has no retries and tries box, bottle, then spatula only if
each preceding stage succeeds and the fixed runtime budget permits another.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE / 'outputs/real-video-20261119/mirror-rgb/method'
CASES = ('hocap_box', 'hocap_bottle', 'hocap_spatula')
CODE_REVISION = '32dfbea9ce6268ea17446208a804710df9475c32'
WEIGHTS_REVISION = '59f307317fb630c9c5a5aa635038d98ed106aaf5'
WEIGHTS_SHA = 'fefacac1fb3d23d33c628c41f40d3ce3be98ca064c271948371805db7b5a0e69'
WEIGHTS_BYTES = 1995767812
VOLUME_NAME = 'pi4-rgb-fallback-20261119'
RUN_ID = 'real-rgb-stride5-once-v1'
LOCAL_ROOT = HERE / 'outputs/real-video-20261119/pi4-modal' / RUN_ID
MAX_LOCAL_BYTES = 950_000_000
RUNTIME_BUDGET_SECONDS = 840


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


DOWNLOAD_CODE = f'''
import hashlib, json, urllib.request
from pathlib import Path
target = Path('/opt/pi4-weights/model.safetensors')
target.parent.mkdir(parents=True, exist_ok=True)
sha, count = hashlib.sha256(), 0
url = 'https://huggingface.co/yyfz233/Pi4/resolve/{WEIGHTS_REVISION}/model.safetensors'
with urllib.request.urlopen(url, timeout=60) as response, target.open('wb') as stream:
    for block in iter(lambda: response.read(8 * 1024**2), b''):
        count += len(block)
        if count > {WEIGHTS_BYTES}: raise RuntimeError('Checkpoint exceeds pinned byte limit')
        sha.update(block)
        stream.write(block)
assert count == {WEIGHTS_BYTES}, count
assert sha.hexdigest() == '{WEIGHTS_SHA}', sha.hexdigest()
print(json.dumps(dict(checkpoint_bytes=count, checkpoint_sha256=sha.hexdigest())))
'''


image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('git', 'build-essential')
    .run_commands(
        'git init /opt/Pi4',
        'git -C /opt/Pi4 remote add origin https://github.com/yyfz/Pi4.git',
        f'timeout 120 git -C /opt/Pi4 fetch --depth 1 origin {CODE_REVISION}',
        f'git -C /opt/Pi4 checkout --detach {CODE_REVISION}',
    )
    .workdir('/opt/Pi4')
    .run_commands('timeout 900 python -m pip install --no-cache-dir -r requirements-cu126.txt')
    .run_commands(
        'python -m pip check',
        'python -m pip freeze > /opt/pi4-environment.txt',
        'timeout 360 python -c ' + shlex.quote(DOWNLOAD_CODE),
    )
    .env({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
          'PYTORCH_ALLOC_CONF': 'expandable_segments:True', 'OMP_NUM_THREADS': '8',
          'PYTHONPATH': '/workspace:/workspace/articulated_recon/benchmarks',
          'TRITON_CACHE_DIR': '/tmp/pi4-triton', 'TORCHINDUCTOR_CACHE_DIR': '/tmp/pi4-inductor'})
)

# No directory mounts, repository automount, evaluator files, credentials, or
# assets. The list below is the complete local upload boundary.
UPLOAD_FILES = [
    (HERE / 'pi4_modal.py', '/workspace/pi4_modal.py'),
    (HERE / 'pi4_frontend.py', '/workspace/articulated_recon/benchmarks/pi4_frontend.py'),
    (HERE / 'rgb_frontend.py', '/workspace/articulated_recon/benchmarks/rgb_frontend.py'),
    (HERE.parent / 'artirecon/input_protocol.py', '/workspace/articulated_recon/artirecon/input_protocol.py'),
]
for case in CASES:
    UPLOAD_FILES.extend((SOURCE_ROOT / case / filename, f'/inputs/{case}/{filename}')
                        for filename in ('rgb.npz', 'input-manifest.json'))
if modal.is_local():
    for local, remote in UPLOAD_FILES:
        image = image.add_local_file(local, remote)

app = modal.App('pi4-rgb-bounded-fallback')
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


@app.function(image=image, gpu='L40S', cpu=8, memory=32768,
              timeout=900, startup_timeout=45, max_containers=1, min_containers=0,
              scaledown_window=2, retries=0, single_use_containers=True,
              volumes={'/results': volume}, include_source=False)
def run_once(upload_receipt):
    import os
    import subprocess
    import traceback

    start = time.monotonic()
    root = Path('/results') / RUN_ID
    if root.exists():
        raise FileExistsError(f'Exclusive run already exists: {root}; no automatic retry')
    root.mkdir(parents=True)
    receipt = dict(status='running', run_id=RUN_ID, volume=VOLUME_NAME,
        function_call_id=modal.current_function_call_id(), gpu='L40S', max_containers=1,
        timeout_seconds=900, internal_budget_seconds=RUNTIME_BUDGET_SECONDS,
        startup_timeout_seconds=45, retries=0, cases=[], upload_receipt=upload_receipt,
        source_frames_per_clip=120, inference_indices=list(range(0, 120, 5)),
        inference_frames_per_clip=24, query_frame=0, warm_repeats=0,
        full_120_frame_inference=False, helper_views_used=False, stage_errors=[],
        code_revision=CODE_REVISION, checkpoint_revision=WEIGHTS_REVISION,
        checkpoint_sha256=WEIGHTS_SHA, checkpoint_bytes=WEIGHTS_BYTES,
        cost_limit_note='One GPU invocation authorized within a conservative $3 allowance; this is not a billing receipt')
    def save():
        receipt['elapsed_seconds'] = time.monotonic() - start
        (root / 'run-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        volume.commit()
    save()
    print(json.dumps({'stage': 'gpu_started', 'call_id': receipt['function_call_id'],
                      'volume': VOLUME_NAME, 'path': str(root)}), flush=True)
    try:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('L40S CUDA/BF16 preflight failed')
        receipt['hardware'] = dict(name=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda)
        if digest('/opt/pi4-weights/model.safetensors') != WEIGHTS_SHA:
            raise ValueError('Image checkpoint digest mismatch')
        for local_record in upload_receipt['files']:
            if digest(local_record['remote']) != local_record['sha256']:
                raise ValueError(f"Uploaded file hash mismatch: {local_record['remote']}")
        for index, case in enumerate(CASES):
            remaining = RUNTIME_BUDGET_SECONDS - (time.monotonic() - start)
            if remaining < 155:
                receipt['cases'].append(dict(case=case, status='not_attempted', reason='Fixed remaining-time gate'))
                continue
            case_start = time.monotonic()
            stage = dict(case=case, status='running', frames=24, stride=5,
                         timeout_seconds=min(550, int(remaining - 60)))
            receipt['cases'].append(stage)
            save()
            command = [sys.executable, '/workspace/articulated_recon/benchmarks/pi4_frontend.py',
                '--bundle', f'/inputs/{case}', '--out', str(root / case),
                '--checkpoint', '/opt/pi4-weights/model.safetensors', '--checkpoint-sha256', WEIGHTS_SHA,
                '--revision', WEIGHTS_REVISION, '--repo', '/opt/Pi4', '--code-revision', CODE_REVISION,
                '--stride', '5', '--max-frames', '25', '--max-size', '512', '--query-frame', '0',
                '--warm-repeats', '0']
            stage['command'] = command
            print(json.dumps({'stage': 'clip_started', 'case': case, 'timeout_seconds': stage['timeout_seconds']}), flush=True)
            try:
                with (root / f'{case}.log').open('w') as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=stage['timeout_seconds'], check=False)
                stage['exit_code'] = process.returncode
                stage['status'] = 'complete' if process.returncode == 0 else 'failed'
                if process.returncode:
                    stage['log_tail'] = (root / f'{case}.log').read_text()[-10000:]
            except subprocess.TimeoutExpired:
                stage.update(status='failed', error='Fixed per-clip timeout', exit_code=124)
            stage['elapsed_seconds'] = time.monotonic() - case_start
            save()
            print(json.dumps({'stage': 'clip_finished', **stage}), flush=True)
            if stage['status'] != 'complete':
                for other in CASES[index + 1:]:
                    receipt['cases'].append(dict(case=other, status='not_attempted', reason='Previous clip failed; no retries'))
                break
        receipt['status'] = 'complete' if all(x['status'] == 'complete' for x in receipt['cases']) else 'partial_or_failed'
    except Exception as error:
        receipt['status'] = 'failed'
        receipt['stage_errors'].append(dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    receipt['artifacts'] = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'run-receipt.json':
            receipt['artifacts'].append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                                              sha256=digest(path)))
    save()
    print(json.dumps({'stage': 'gpu_finished', 'status': receipt['status'],
                      'elapsed_seconds': receipt['elapsed_seconds']}), flush=True)
    return receipt


def preflight():
    sys.path.insert(0, str(HERE))
    from rgb_frontend import load_source_bundle
    for case in CASES:
        _, _, rgb, timestamps = load_source_bundle(SOURCE_ROOT / case)
        if rgb.shape != (120, 480, 640, 3):
            raise ValueError(f'Unexpected source shape for {case}: {rgb.shape}')
    result = dict(profile='surajmvr14', cases=list(CASES), run_id=RUN_ID, files=[])
    for local, remote in UPLOAD_FILES:
        if not local.is_file():
            raise FileNotFoundError(local)
        result['files'].append(dict(local=str(local), remote=remote, bytes=local.stat().st_size, sha256=digest(local)))
    result['upload_bytes'] = sum(x['bytes'] for x in result['files'])
    if result['upload_bytes'] > 400_000_000:
        raise ValueError('Unexpected upload size; only explicit source RGB and adapter files are authorized')
    return result


def download_results(receipt):
    """Download full-precision geometry only, with a fixed aggregate byte cap."""
    artifacts = {x['path']: x for x in receipt['artifacts']}
    download_receipt = dict(downloaded=[], omitted=[], max_bytes=MAX_LOCAL_BYTES, bytes=0)
    for case in CASES:
        wanted = [f'{case}/geometry.npz', f'{case}/inference.json', f'{case}/input-manifest.json', f'{case}.log']
        for relative in wanted:
            entry = artifacts.get(relative)
            if entry is None:
                continue
            if download_receipt['bytes'] + entry['bytes'] > MAX_LOCAL_BYTES:
                download_receipt['omitted'].append(dict(**entry, reason='Aggregate local download cap'))
                continue
            target = LOCAL_ROOT / relative
            if target.name == 'input-manifest.json':
                target = target.with_name('input-manifest.remote.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            sha = hashlib.sha256()
            with target.open('xb') as stream:
                for block in volume.read_file(f'/{RUN_ID}/{relative}'):
                    sha.update(block)
                    stream.write(block)
            if target.stat().st_size != entry['bytes'] or sha.hexdigest() != entry['sha256']:
                raise ValueError(f'Download integrity failure for {relative}')
            download_receipt['bytes'] += entry['bytes']
            download_receipt['downloaded'].append(dict(**entry, local=str(target)))
        # Bind only actual local method inputs. Original raw archive stays remote
        # and is explicitly excluded from this narrower consumer manifest.
        out = LOCAL_ROOT / case
        if (out / 'geometry.npz').is_file() and (out / 'input-manifest.remote.json').is_file():
            manifest = json.loads((out / 'input-manifest.remote.json').read_text())
            manifest['method_inputs'] = [x for x in manifest['method_inputs'] if x != 'pi4_raw_predictions']
            manifest['artifacts'] = [x for x in manifest['artifacts'] if x['id'] != 'pi4_raw_predictions']
            for artifact in manifest['artifacts']:
                if artifact['id'] == 'source_rgb':
                    artifact['path'] = str(SOURCE_ROOT / case / 'rgb.npz')
                elif artifact['id'] == 'pi4_motion':
                    artifact['path'] = 'geometry.npz'
                    artifact['sha256'] = artifacts[f'{case}/geometry.npz']['sha256']
                    artifact['description'] = 'Native motion values copied into geometry archive, with singleton view axis'
            native = artifacts[f'{case}/raw-predictions.npz']
            manifest['native_raw_archive'] = dict(**native, storage_uri=f'modal-volume://{VOLUME_NAME}/{RUN_ID}/{native["path"]}',
                                                   locally_present=False)
            from artirecon.input_protocol import validate_manifest
            validate_manifest(manifest, out, verify_files=True)
            (out / 'input-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for relative, entry in artifacts.items():
        if relative.endswith('raw-predictions.npz'):
            download_receipt['omitted'].append(dict(**entry, reason='Native raw archive preserved remotely'))
    (LOCAL_ROOT / 'download-receipt.json').write_text(json.dumps(download_receipt, indent=2) + '\n')
    return download_receipt


@app.local_entrypoint()
def main():
    if LOCAL_ROOT.exists():
        raise FileExistsError(f'Exclusive local run already exists: {LOCAL_ROOT}; no automatic resubmission')
    uploads = preflight()
    LOCAL_ROOT.mkdir(parents=True)
    (LOCAL_ROOT / 'upload-receipt.json').write_text(json.dumps(uploads, indent=2) + '\n')
    call = run_once.spawn(uploads)
    launch = dict(call_id=call.object_id, run_id=RUN_ID, volume=VOLUME_NAME,
                  remote_output=f'/{RUN_ID}', local_output=str(LOCAL_ROOT))
    (LOCAL_ROOT / 'launch-receipt.json').write_text(json.dumps(launch, indent=2) + '\n')
    print(json.dumps({'stage': 'launched', **launch}), flush=True)
    try:
        receipt = call.get()
        (LOCAL_ROOT / 'run-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        downloaded = download_results(receipt)
        print(json.dumps(dict(status=receipt['status'], downloaded_bytes=downloaded['bytes'], **launch), indent=2))
    except Exception as error:
        (LOCAL_ROOT / 'client-error.json').write_text(json.dumps(dict(type=type(error).__name__, message=str(error)), indent=2) + '\n')
        raise


if __name__ == '__main__':
    print(json.dumps(preflight(), indent=2))
