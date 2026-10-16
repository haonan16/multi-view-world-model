#!/bin/bash
#SBATCH -J rh20t_check
#SBATCH -o slurm_logs/rh20t_check_%j.out
#SBATCH --account=ydu_lab
#SBATCH --partition=sapphire,shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
# Logs RH20T multiview samples + loader stats to wandb (scripts/rh20t/check_dataset_wandb.py).
# Extra args are passed through, e.g.: sbatch scripts/rh20t/check_dataset_wandb.sbatch 'dataset.cfgs=[7]'
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[ -d "$REPO/datasets" ] || REPO=$SLURM_SUBMIT_DIR
cd "$REPO" || exit 1
PY=/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin/python
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
$PY scripts/rh20t/check_dataset_wandb.py +check.num_workers=7 "$@"
