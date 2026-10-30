#!/bin/bash
#SBATCH -J mvwm_env
#SBATCH -o /n/lab_storage/ydu_lab/Lab/haonan/venvs/logs/%x_%j.out
#SBATCH --account=ydu_lab
#SBATCH --partition=sapphire,shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
# Editable FastVideo install (CUDA 12.6 wheels) into a uv venv for the RH20T multiview port.
set -euo pipefail
REPO=/n/lab_storage/ydu_lab/Lab/haonan/projects/multi-view-world-model
VENV=${VENV:-/n/lab_storage/ydu_lab/Lab/haonan/venvs/mvwm}
export UV_CACHE_DIR=/n/lab_storage/ydu_lab/Lab/haonan/.state/cache/uv UV_LINK_MODE=copy
cd "$REPO"
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 --seed "$VENV"
source "$VENV/bin/activate"
# fastvideo-kernel (sparse-attention kernels: VSA/VMoBA/SLA) only ships a glibc>=2.34 x86 wheel and
# its sdist needs nvcc; the cluster is RHEL8 (glibc 2.28) and the RH20T configs use dense attention,
# so drop it with an override whose marker never matches.
OVERRIDES=$(mktemp)
echo "fastvideo-kernel; sys_platform == 'never'" > "$OVERRIDES"
UV_TORCH_BACKEND=cu126 uv pip install -e ".[dev]" --override "$OVERRIDES"
python -c "import fastvideo, torch, av, torchdata; print('fastvideo ok, torch', torch.__version__, torch.version.cuda)"
