#!/bin/bash
#SBATCH -J rh20t_eval_unseen
#SBATCH -o /n/lab_storage/ydu_lab/Lab/haonan/projects/multi-view-world-model/logs/slurm/%x_%j.out
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=0-08:00:00
# Matched unseen-view evaluation of one checkpoint: one shard of the val clips per GPU.
#   sbatch scripts/rh20t/eval_unseen_matched.sbatch <checkpoint-dir> <out-dir> [extra eval args...]
# Writes <out-dir>/shard{0..N-1}.npz; summarize with scripts/rh20t/summarize_unseen_eval.py.
set -euo pipefail
REPO=/n/lab_storage/ydu_lab/Lab/haonan/projects/multi-view-world-model
CKPT=${1:?usage: sbatch $0 <checkpoint-dir> <out-dir> [eval args...]}
OUT=${2:?usage: sbatch $0 <checkpoint-dir> <out-dir> [eval args...]}
shift 2
NGPU=${NGPU:-4}
cd "$REPO"
source /n/lab_storage/ydu_lab/Lab/haonan/venvs/mvwm/bin/activate
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export WANDB_MODE=disabled
mkdir -p "$OUT"
echo "[$(date)] $SLURM_JOB_ID: $CKPT -> $OUT ($NGPU shards) ${*:-}"
pids=()
for s in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$s MASTER_PORT=$((29500 + SLURM_JOB_ID % 500 * 4 + s)) \
    python scripts/rh20t/eval_unseen_matched.py --checkpoint "$CKPT" --out "$OUT/shard$s.npz" \
    --shard "$s" --num-shards "$NGPU" "$@" > "$OUT/shard$s.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
tail -n 2 "$OUT"/shard*.log
echo "[$(date)] EVAL_EXIT=$status"
exit $status
