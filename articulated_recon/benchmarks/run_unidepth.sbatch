#!/usr/bin/env bash
# Submit from the repository root after choosing the cluster account/partition.
# This reuses an installed environment and checkpoint; it downloads nothing.
#SBATCH --job-name=rgb-depth
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail

: "${RGB_INPUT_BUNDLE:?Set isolated RGB bundle directory}"
: "${RGB_DEPTH_OUT:?Set a fresh prediction directory}"
: "${RGB_DEPTH_PYTHON:?Set installed UniDepth Python executable}"
: "${RGB_UNIDEPTH_CODE:?Set existing UniDepth checkout directory}"
: "${RGB_UNIDEPTH_CHECKPOINT:?Set existing local checkpoint directory}"
: "${RGB_UNIDEPTH_REVISION:?Set checkpoint revision identifier}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$RGB_UNIDEPTH_CODE"

"$RGB_DEPTH_PYTHON" \
  "$SLURM_SUBMIT_DIR/articulated_recon/benchmarks/rgb_frontend.py" infer-unidepth \
  --bundle "$RGB_INPUT_BUNDLE" --out "$RGB_DEPTH_OUT" \
  --checkpoint "$RGB_UNIDEPTH_CHECKPOINT" --revision "$RGB_UNIDEPTH_REVISION" \
  --device cuda
