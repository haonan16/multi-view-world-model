#!/bin/bash
# v2 camera-conditioning ablation (tokens design): 5 runs that differ ONLY in cam_mode.
# Shared: 17 frames @ 1.5 fps, scale-normalised view-0-relative cameras, cached task text
# (10% dropout), diffusion-forcing noise mixture, full gradient checkpointing.
# Each run is queued in gpu_requeue AND kempner_requeue (RACE=1: first to start wins);
# run dirs + checkpoints live on lab_storage (netscratch ydu_lab quota is full).
#   bash scripts/rh20t/launch_v2.sh [mode ...]        (default: all five)
#   resubmitting a mode resumes its run dir from checkpoints/last.ckpt
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=/n/lab_storage/ydu_lab/Lab/akiruga/mv_runs_v2
EXCL=$(cat /n/netscratch/ydu_lab/Lab/akiruga/mv_runs/a100_40gb_nodes.txt)
MODES=("$@")
[ $# -eq 0 ] && MODES=(concat cat4d prope gta camtoken)
COMMON=(
  algorithm.text.enabled=true
  algorithm.gradient_checkpointing_rate=1
  "algorithm.noise.mix={first_frame:0.4,rand_history:0.15,independent:0.2,view_history:0.15,none:0.1}"
  algorithm.noise.hist_noise_prob=0.2
)
mkdir -p "$ROOT"
declare -A IDS
[ -f "$ROOT/wandb_ids.txt" ] && while read -r m i; do IDS[$m]=$i; done < "$ROOT/wandb_ids.txt"
for m in "${MODES[@]}"; do
  id=${IDS[$m]:-$(python3 -c "import secrets,string;print(''.join(secrets.choice(string.ascii_lowercase+string.digits) for _ in range(8)))")}
  grep -q "^$m " "$ROOT/wandb_ids.txt" 2>/dev/null || echo "$m $id" >> "$ROOT/wandb_ids.txt"
  export RACE=1 DESIGN=tokens DATASET=rh20t_multiview_v2 NAME=mvv2_$m STEPS=${STEPS:-15000} \
         CKPT_EVERY=250 BS=${BS:-4} RUN_DIR=$ROOT/mvv2_$m WANDB_RUN_ID=$id
  a=$(sbatch --parsable --exclude=$EXCL -p kempner_requeue -A kempner_ydu_lab -J mvv2_$m \
        scripts/rh20t/train_multiview.sbatch algorithm.cam_mode=$m "${COMMON[@]}")
  b=$(sbatch --parsable --exclude=$EXCL -p gpu_requeue -A ydu_lab -J mvv2_$m \
        scripts/rh20t/train_multiview.sbatch algorithm.cam_mode=$m "${COMMON[@]}")
  echo "$m  wandb=$id  jobs: $a (kempner_requeue) $b (gpu_requeue)"
done
