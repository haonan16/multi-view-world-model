#!/usr/bin/env bash
# Cluster-only preparation. This script never schedules a GPU job.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: pi4_setup.sh preflight|setup --root ABSOLUTE_NEW_PI4_ROOT --python ABSOLUTE_PYTHON_3_11

preflight prints disk/runtime requirements and changes nothing.
setup creates an isolated venv, pins official code, installs its released stack,
and downloads only the 1,995,767,812-byte public model on the Linux cluster.
It does not modify test2, train a model, or submit a job. Reserve >=25 GiB free.
EOF
}

if [[ $# -lt 1 ]]; then usage; exit 2; fi
if [[ $1 == --help || $1 == -h ]]; then usage; exit 0; fi
mode=$1
shift
pi4_root=''
base_python=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) pi4_root=${2:?missing --root value}; shift 2 ;;
    --python) base_python=${2:?missing --python value}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[[ "$mode" == preflight || "$mode" == setup ]] || { usage; exit 2; }
[[ "$pi4_root" == /* && "$base_python" == /* && -x "$base_python" ]] || { usage; exit 2; }
command -v git >/dev/null
command -v cc >/dev/null

code_revision=32dfbea9ce6268ea17446208a804710df9475c32
weights_revision=59f307317fb630c9c5a5aa635038d98ed106aaf5
weights_sha256=fefacac1fb3d23d33c628c41f40d3ce3be98ca064c271948371805db7b5a0e69
weights_bytes=1995767812

"$base_python" - "$pi4_root" "$mode" <<'PY'
import json, platform, shutil, sys
from pathlib import Path
root = Path(sys.argv[1])
if sys.version_info[:2] != (3, 11):
    raise SystemExit('Use an existing Python 3.11 interpreter for the released pinned stack')
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise SystemExit('Run on the Linux x86_64 cluster; no local model download')
probe = root
while not probe.exists():
    probe = probe.parent
free = shutil.disk_usage(probe).free
minimum = 25 * 1024**3
if free < minimum:
    raise SystemExit(f'Need >=25 GiB free before setup; found {free / 1024**3:.2f} GiB')
if root.exists() and any(root.iterdir()) and not (root / '.pi4-install-root').is_file():
    raise SystemExit('Choose a new dedicated root; refusing an unrelated nonempty directory')
print(json.dumps(dict(mode=sys.argv[2], root=str(root), python=sys.executable,
    free_bytes=free, minimum_free_bytes=minimum, new_weight_bytes=1995767812,
    weight_budget_bytes=5_000_000_000, estimated_environment_budget_bytes=15 * 1024**3,
    initial_output_budget_bytes=2 * 1024**3, gpu_job_submitted=False), indent=2))
PY
[[ "$mode" == setup ]] || exit 0

mkdir -p "$pi4_root"
touch "$pi4_root/.pi4-install-root"
mkdir -p "$pi4_root/logs" "$pi4_root/tmp" "$pi4_root/cache" "$pi4_root/weights"
export TMPDIR="$pi4_root/tmp"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export GIT_TERMINAL_PROMPT=0

if [[ ! -d "$pi4_root/Pi4/.git" ]]; then
  git init "$pi4_root/Pi4"
  git -C "$pi4_root/Pi4" remote add origin https://github.com/yyfz/Pi4.git
  git -C "$pi4_root/Pi4" fetch --depth 1 origin "$code_revision"
  git -C "$pi4_root/Pi4" checkout --detach "$code_revision"
fi
[[ $(git -C "$pi4_root/Pi4" rev-parse HEAD) == "$code_revision" ]] || { echo 'Wrong Pi4 code revision' >&2; exit 1; }
[[ -z $(git -C "$pi4_root/Pi4" status --porcelain --untracked-files=no) ]] || { echo 'Pi4 code is modified' >&2; exit 1; }
if [[ ! -x "$pi4_root/env/bin/python" ]]; then
  "$base_python" -m venv "$pi4_root/env"
fi
pi4_python="$pi4_root/env/bin/python"
# No system-site-packages and no changes to the caller's existing environment.
"$pi4_python" -m pip install --no-cache-dir --upgrade pip 2>&1 | tee "$pi4_root/logs/pip-bootstrap.log"
(
  cd "$pi4_root/Pi4"
  "$pi4_python" -m pip install --no-cache-dir -r requirements-cu126.txt
) 2>&1 | tee "$pi4_root/logs/pip-install.log"
"$pi4_python" -m pip check 2>&1 | tee "$pi4_root/logs/pip-check.log"
"$pi4_python" -m pip freeze > "$pi4_root/logs/pip-freeze.txt"
"$pi4_python" - <<'PY'
import torch, triton
from pi4 import Pi4
assert torch.__version__ == '2.12.1+cu126', torch.__version__
assert triton.__version__ == '3.7.1', triton.__version__
print('CPU import preflight complete; CUDA/BF16 check occurs inside the bounded GPU job')
PY

"$pi4_python" - "$pi4_root" "$weights_revision" "$weights_sha256" "$weights_bytes" "$code_revision" <<'PY'
import hashlib, json, os, shutil, sys, urllib.request
from pathlib import Path
root, revision, expected_sha, expected_bytes, code_revision = sys.argv[1:]
root, expected_bytes = Path(root), int(expected_bytes)
target = root / 'weights/model.safetensors'
if expected_bytes > 5_000_000_000:
    raise SystemExit('Checkpoint exceeds the total new-weight budget')
def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            sha.update(block)
    return sha.hexdigest()
if target.exists():
    if target.stat().st_size != expected_bytes or digest(target) != expected_sha:
        raise SystemExit('Existing checkpoint is wrong; refusing to overwrite it')
else:
    if shutil.disk_usage(root).free < expected_bytes + 5 * 1024**3:
        raise SystemExit('Insufficient remaining disk for checkpoint and bounded output reserve')
    url = f'https://huggingface.co/yyfz233/Pi4/resolve/{revision}/model.safetensors'
    partial = target.with_suffix('.safetensors.partial')
    sha, count = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as stream:
        for block in iter(lambda: response.read(8 * 1024**2), b''):
            count += len(block)
            if count > expected_bytes:
                raise SystemExit('Download exceeds pinned size; refusing extra weight bytes')
            sha.update(block)
            stream.write(block)
    if count != expected_bytes or sha.hexdigest() != expected_sha:
        raise SystemExit('Checkpoint size/hash verification failed')
    os.replace(partial, target)
record = dict(status='setup_complete', code_revision=code_revision,
    checkpoint_revision=revision, checkpoint_sha256=expected_sha,
    checkpoint_bytes=expected_bytes, checkpoint_path=str(target), python=sys.executable,
    free_bytes_after_setup=shutil.disk_usage(root).free, gpu_forward_completed=False)
(root / 'setup-complete.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
PY
