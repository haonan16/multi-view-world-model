#!/bin/bash
#SBATCH -J mv_render
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH --account=ydu_lab
#SBATCH --partition=shared,sapphire
#SBATCH --array=0-44
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
# Point-cloud presentation videos, one (scene, source) per array task
# (scripts/rh20t/render_cloud_videos.py).   ROOT=<infer dir>  OUT=<video dir>  [CLOUDS=clouds_native]
set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
export OMP_NUM_THREADS=12 PYTHONUNBUFFERED=1
SRCS=(gt pred_tokens pred_tiled)
/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin/python scripts/rh20t/render_cloud_videos.py \
  --root "${ROOT:?}" --out "${OUT:?}" --clouds "${CLOUDS:-clouds}" \
  --scenes $((SLURM_ARRAY_TASK_ID / 3)) --sources ${SRCS[$((SLURM_ARRAY_TASK_ID % 3))]}
echo "RENDER_EXIT=$?"
