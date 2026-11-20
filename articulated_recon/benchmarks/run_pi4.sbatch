#!/usr/bin/env bash
#SBATCH --job-name=pi4-rgb-smoke
#SBATCH --partition=gpu_requeue
#SBATCH --account=ydu_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=pi4-smoke-%j.log

# Submit exactly one smoke job, with explicit paths:
# sbatch run_pi4.sbatch PI4_ROOT RGB_BUNDLE ADAPTER_REPOSITORY NEW_OUTPUT
# Use --constraint only after inspecting actual partition features with sinfo.
set -euo pipefail
[[ $# == 4 ]] || { echo 'Expected PI4_ROOT RGB_BUNDLE ADAPTER_REPOSITORY NEW_OUTPUT' >&2; exit 2; }
pi4_root=$1
source_bundle=$2
adapter_root=$3
output_dir=$4
for path in "$pi4_root" "$source_bundle" "$adapter_root" "$output_dir"; do
  [[ "$path" == /* ]] || { echo 'All paths must be absolute' >&2; exit 2; }
done
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Run through sbatch inside a bounded GPU allocation' >&2; exit 2; }
[[ -f "$pi4_root/setup-complete.json" && ! -e "$output_dir" ]] || { echo 'Setup missing or output already exists' >&2; exit 2; }
pi4_python="$pi4_root/env/bin/python"
adapter="$adapter_root/articulated_recon/benchmarks/pi4_frontend.py"
[[ -x "$pi4_python" && -f "$adapter" ]] || { echo 'Python or adapter missing' >&2; exit 2; }

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$pi4_root/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$pi4_root/cache/torchinductor"
export TMPDIR="$pi4_root/tmp/job-$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Source-only preflight: the loader verifies declared hashes and rejects any
# array other than RGB/timestamps. Nothing under evaluator/reference is read.
"$pi4_python" - "$adapter_root" "$source_bundle" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'articulated_recon/benchmarks'))
from rgb_frontend import load_source_bundle
import torch
_, _, rgb, timestamps = load_source_bundle(sys.argv[2])
if len(rgb) < 25:
    raise SystemExit('This frozen smoke test requires at least 25 source frames')
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit('Allocated GPU lacks required CUDA/BF16 support')
print(json.dumps(dict(source_shape=list(rgb.shape), smoke_frames=25,
    smoke_time_range_s=[float(timestamps[0]), float(timestamps[24])],
    gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda)))
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

set +e
timeout --signal=TERM --kill-after=30s 18m "$pi4_python" "$adapter" \
  --bundle "$source_bundle" --out "$output_dir" \
  --checkpoint "$pi4_root/weights/model.safetensors" \
  --checkpoint-sha256 fefacac1fb3d23d33c628c41f40d3ce3be98ca064c271948371805db7b5a0e69 \
  --revision 59f307317fb630c9c5a5aa635038d98ed106aaf5 \
  --repo "$pi4_root/Pi4" --code-revision 32dfbea9ce6268ea17446208a804710df9475c32 \
  --stride 1 --max-frames 25 --max-size 512 --query-frame 0 --warm-repeats 1 --viewer
job_exit=$?
set -e
"$pi4_python" - "$pi4_root" "$output_dir" "$job_exit" <<'PY'
import json, os, sys
from pathlib import Path
root, output, code = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
status = dict(job_id=os.environ['SLURM_JOB_ID'], exit_code=code,
    output=str(output), status='finished' if code == 0 else 'failed',
    smoke_test_only=True, full_sequence_completed=False)
(root / 'logs' / f"smoke-job-{status['job_id']}.json").write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status, indent=2))
PY
exit "$job_exit"
