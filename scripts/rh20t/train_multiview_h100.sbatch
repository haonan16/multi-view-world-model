#!/bin/bash
#SBATCH -J rh20t_mv
#SBATCH -o /n/lab_storage/ydu_lab/Lab/haonan/projects/multi-view-world-model/logs/slurm/%x_%j.out
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
# RH20T multi-view training on kempner_h100 (non-preemptible), N nodes x 4 H100.
#   sbatch [--nodes N] scripts/rh20t/train_multiview_h100.sbatch <config.yaml> [--dotted.key value ...]
# World size is derived from the allocation and passed to the trainer as
# num_gpus = hsdp_replicate_dim (pure data parallel; the 1.3B model fits per GPU).
# The configs set resume_from_checkpoint=latest, so resubmitting the same command
# continues from the newest checkpoint after the 2-day limit.
set -euo pipefail
REPO=/n/lab_storage/ydu_lab/Lab/haonan/projects/multi-view-world-model
VENV=${VENV:-/n/lab_storage/ydu_lab/Lab/haonan/venvs/mvwm}
CONFIG=${1:?usage: sbatch $0 <config.yaml> [overrides...]}
shift
cd "$REPO"
mkdir -p logs/slurm
GPUS_PER_NODE=${GPUS_PER_NODE:-4}  # override (with --gres) for single-GPU smoke runs
WORLD=$((SLURM_NNODES * GPUS_PER_NODE))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[$(date)] $SLURM_JOB_ID: $SLURM_NNODES nodes ($SLURM_JOB_NODELIST), world $WORLD, config $CONFIG ${*:-}"
export REPO VENV CONFIG MASTER_ADDR MASTER_PORT GPUS_PER_NODE WORLD
srun --kill-on-bad-exit=1 bash -c '
  source "$VENV/bin/activate"
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
  export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export WANDB_MODE=${WANDB_MODE:-online}
  exec torchrun --nnodes "$SLURM_NNODES" --node_rank "$SLURM_NODEID" --nproc_per_node "$GPUS_PER_NODE" \
    --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
    -m fastvideo.train.entrypoint.train --config "$CONFIG" \
    --training.distributed.num_gpus "$WORLD" \
    --training.distributed.hsdp_replicate_dim "$WORLD" \
    --training.distributed.hsdp_shard_dim 1 "$@"
' _ "$@"
echo "TRAIN_EXIT=$?"
