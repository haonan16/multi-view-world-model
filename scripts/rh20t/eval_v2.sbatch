#!/bin/bash
# Final v2 camera-conditioning comparison on held-out RH20T clips (1-frame history), once
# without guidance and once with text + history guidance. Submit to both requeue partitions;
# the first copy to start cancels the other (same name).
#   MODELS=concat=concat:CKPT,... OUT=DIR sbatch -p gpu_requeue -A ydu_lab scripts/rh20t/eval_v2.sbatch
#SBATCH -J mvv2_eval
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200|a100|rtx6000pro
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
set -euo pipefail
other=$(squeue -h -u $USER -n $SLURM_JOB_NAME -o %i | grep -v "^$SLURM_JOB_ID$" || true)
[ -n "$other" ] && scancel $other
export PATH=/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin:$PATH
export HF_HOME=/n/netscratch/ydu_lab/Lab/akiruga HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
python scripts/rh20t/infer_multiview.py --out "$OUT/noguide" --n ${N:-15} --models "$MODELS"
python scripts/rh20t/infer_multiview.py --out "$OUT/guide_t2_h1" --n ${N:-15} --models "$MODELS" \
  --text_guidance 2 --hist_guidance 1
echo EVAL_DONE
