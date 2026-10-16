#!/bin/bash
#SBATCH -J mv_ablation
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --constraint=h100|h200|a100|rtx6000pro   # >=80 GB cards (exclude A100-40GB nodes with
#                                                 # --exclude); never MIG slices or A40/A6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
# Multiview ablation on RH20T (algorithms/wan/wan_multiview.py, datasets/rh20t.py).
#   DESIGN=tokens|tiled  NAME=<run name>  STEPS=<max steps>  [extra hydra overrides as args]
#   RUN_DIR=<existing run dir>  continue that run from its last.ckpt (also lets a chain of
#                               jobs, e.g. --dependency=afterany, share one run dir)
#   WANDB_RUN_ID=<id>           log into that wandb run instead of starting a new one
#   RACE=1                      the same run is also queued in another partition under the
#                               same job name (kempner_requeue can't be in a multi-partition
#                               submission): the first copy to start cancels the others
# --requeue: kempner_requeue can preempt; the restart (same job id -> same RUN_DIR) resumes
# from last.ckpt with full optimizer state (save_weights_only=false).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p slurm_logs
if [ "${RACE:-0}" = 1 ]; then
  # A copy that is already running always wins (elapsed >= 60 s, or the lower job id if
  # both started together); this copy then cancels itself. Otherwise cancel the others.
  others=$(squeue -u "$USER" -n "$SLURM_JOB_NAME" -h -o "%i %t %M" | awk -v me="$SLURM_JOB_ID" '$1!=me')
  winner=$(echo "$others" | awk -v me="$SLURM_JOB_ID" '$2=="R"||$2=="CF" {
      n=split($3,a,":"); old=($3 ~ /-/ || n>2 || a[1]+0>=1);
      if (old || $1+0 < me+0) print $1 }' | head -1)
  if [ -n "$winner" ]; then
    echo "RACE: copy $winner already running; cancelling $SLURM_JOB_ID"; scancel "$SLURM_JOB_ID"; exit 0
  fi
  losers=$(echo "$others" | awk '{print $1}' | paste -sd' ')
  [ -n "$losers" ] && echo "RACE: won; cancelling $losers" && scancel $losers
fi
module load cuda/12.4.1-fasrc01 2>/dev/null || true
module load gcc/9.5.0-fasrc01 2>/dev/null || true
# Unpack the conda env onto node-local NVMe (scripts/rh20t/pack_env.sbatch). On the
# lab_storage NFS copy, `import diffusers` alone can take >5 min per rank.
ENV_TAR=/n/netscratch/ydu_lab/Lab/akiruga/envs/test2.tar.zst
LOCAL_ENV=/tmp/${USER}_envs/test2
if [ -f "$ENV_TAR" ]; then
  srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES bash -c \
    "[ -x $LOCAL_ENV/bin/python ] || { mkdir -p /tmp/${USER}_envs && zstd -dc $ENV_TAR | tar -C /tmp/${USER}_envs -xf - ; }"
  export PATH=$LOCAL_ENV/bin:$PATH
else
  echo "WARN: $ENV_TAR missing, using the NFS env (slow imports)"
  export PATH=/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin:$PATH
fi
echo "python: $(which python)"
export HF_HOME=/n/netscratch/ydu_lab/Lab/akiruga HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True

DESIGN=${DESIGN:?tokens or tiled}
NAME=${NAME:-mv_${DESIGN}}
RUN_DIR=${RUN_DIR:-/n/netscratch/ydu_lab/Lab/akiruga/mv_runs/${NAME}_${SLURM_JOB_ID}}
[ -n "${WANDB_RUN_ID:-}" ] && export WANDB_RUN_ID WANDB_RESUME=allow
RESUME=()
[ -f "$RUN_DIR/checkpoints/last.ckpt" ] && RESUME=(load=$RUN_DIR/checkpoints/last.ckpt) && echo "REQUEUE: resuming $RUN_DIR/checkpoints/last.ckpt"

srun python -m main \
  hydra.run.dir=$RUN_DIR \
  +name=$NAME \
  experiment=exp_video \
  algorithm=wan_multiview \
  algorithm.design=$DESIGN \
  dataset=rh20t_multiview \
  experiment.num_nodes=${SLURM_NNODES} \
  experiment.strategy=ddp \
  experiment.training.lr=${LR:-2e-5} \
  experiment.training.batch_size=${BS:-8} \
  experiment.training.max_steps=${STEPS:-10000} \
  experiment.training.optim.accumulate_grad_batches=1 \
  experiment.training.optim.gradient_clip_val=1.0 \
  experiment.training.data.num_workers=14 \
  experiment.training.checkpointing.every_n_train_steps=${CKPT_EVERY:-1000} \
  experiment.training.checkpointing.save_weights_only=false \
  experiment.training.checkpointing.save_top_k=1 \
  experiment.training.async_checkpointing=false \
  experiment.validation.val_every_n_step=${VAL_EVERY:-1000} \
  experiment.validation.limit_batch=${VAL_BATCHES:-8} \
  experiment.validation.batch_size=4 \
  experiment.validation.data.num_workers=4 \
  experiment.validation.num_sanity_val_steps=0 \
  wandb.mode=online \
  "${RESUME[@]}" "$@"
echo "TRAIN_EXIT=$?"
