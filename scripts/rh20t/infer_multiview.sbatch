#!/bin/bash
#SBATCH -J mv_infer
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH --account=ydu_lab
#SBATCH --partition=gpu_requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200|a100|rtx6000pro
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
# Held-out inference for both multiview designs (scripts/rh20t/infer_multiview.py).
#   OUT=<dir> [N=15] [VGGT=1: then build VGGT-Omega dynamic clouds from OUT]
#   [RACE=1: same job name queued in several partitions, first to start cancels the rest]
set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
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
ENV_TAR=/n/netscratch/ydu_lab/Lab/akiruga/envs/test2.tar.zst
LOCAL_ENV=/tmp/${USER}_envs/test2
[ -x $LOCAL_ENV/bin/python ] || { mkdir -p /tmp/${USER}_envs && zstd -dc $ENV_TAR | tar -C /tmp/${USER}_envs -xf - ; }
export PATH=$LOCAL_ENV/bin:$PATH
export HF_HOME=/n/netscratch/ydu_lab/Lab/akiruga HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
[ -f "${OUT:?}/metrics.json" ] || python scripts/rh20t/infer_multiview.py --out "$OUT" --n "${N:-15}" "$@"
echo "INFER_EXIT=$?"
if [ "${VGGT:-0}" = 1 ]; then
  python scripts/rh20t/vggt_dynamic_clouds.py --root "$OUT"
  echo "VGGT_EXIT=$?"
fi
