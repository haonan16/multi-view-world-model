#!/bin/bash
#SBATCH -J rh20t_mv
#SBATCH -o /n/lab_storage/ydu_lab/Lab/haonan/projects/multi-view-world-model/logs/slurm/%x_%j.out
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
# torch cu126 has no sm_120 kernels (RTX Pro 6000 Blackwell); a100-mig nodes are MIG slices
#SBATCH --constraint=h100|h200|a100
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
# RH20T multi-view training (single node, 4 GPUs).
#   sbatch scripts/rh20t/train_multiview.sbatch <config.yaml> [--dotted.key value ...]
# --requeue: kempner_requeue can preempt; the configs set
# training.checkpoint.resume_from_checkpoint=latest, so a restart resumes from the newest
# DCP checkpoint in training.checkpoint.output_dir.
set -euo pipefail
REPO=/n/lab_storage/ydu_lab/Lab/haonan/projects/multi-view-world-model
VENV=${VENV:-/n/lab_storage/ydu_lab/Lab/haonan/venvs/mvwm}
CONFIG=${1:?usage: sbatch $0 <config.yaml> [overrides...]}
shift
cd "$REPO"
mkdir -p logs/slurm
source "$VENV/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-online}
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "[$(date)] $SLURM_JOB_ID on $(hostname): $NUM_GPUS GPUs, config $CONFIG ${*:-}"
torchrun --nnodes 1 --nproc_per_node "$NUM_GPUS" --master_port $((29500 + SLURM_JOB_ID % 1000)) \
  -m fastvideo.train.entrypoint.train --config "$CONFIG" "$@"
echo "TRAIN_EXIT=$?"
