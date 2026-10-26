#!/bin/bash
# Robot + human-demo fine-tune: continue a single-view (sv) model from its step-10000
# weights (fresh optimiser) on RH20T robot episodes mixed 50/50 with RH20T human
# demonstrations (dataset=rh20t_multiview_v2_human), same single_view-heavy noise mixture,
# single-view validation on robot episodes. Each run queued in both requeue partitions.
#   bash scripts/rh20t/launch_v2_human.sh [mode ...]        (default: prope)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=/n/lab_storage/ydu_lab/Lab/akiruga/mv_runs_v2
EXCL=$(cat /n/netscratch/ydu_lab/Lab/akiruga/mv_runs/a100_40gb_nodes.txt)
MODES=("$@")
[ $# -eq 0 ] && MODES=(prope)
COMMON=(
  algorithm.text.enabled=true
  algorithm.gradient_checkpointing_rate=1
  "algorithm.noise.mix={single_view:0.5,first_frame:0.15,rand_history:0.1,independent:0.1,view_history:0.05,none:0.1}"
  algorithm.noise.hist_noise_prob=0.2
  algorithm.val_hist_views=1
)
for m in "${MODES[@]}"; do
  init=$(ls $ROOT/mvv2_${m}_sv/checkpoints/*-step=10000.ckpt 2>/dev/null | head -1)
  [ -n "$init" ] || { echo "$m: no ${m}_sv step-10000 checkpoint, skipped"; continue; }
  key=${m}_svh
  id=$(awk -v k=$key '$1==k{print $2}' $ROOT/wandb_ids.txt)
  [ -n "$id" ] || { id=$(python3 -c "import secrets,string;print(''.join(secrets.choice(string.ascii_lowercase+string.digits) for _ in range(8)))"); echo "$key $id" >> $ROOT/wandb_ids.txt; }
  export RACE=1 DESIGN=tokens DATASET=rh20t_multiview_v2_human NAME=mvv2_$key STEPS=${STEPS:-5000} \
         CKPT_EVERY=250 BS=${BS:-8} RUN_DIR=$ROOT/mvv2_$key WANDB_RUN_ID=$id
  args=(algorithm.cam_mode=$m "algorithm.model.tuned_ckpt_path='$init'" "${COMMON[@]}")
  a=$(sbatch --parsable --exclude=$EXCL -p kempner_requeue -A kempner_ydu_lab -J mvv2_$key \
        scripts/rh20t/train_multiview.sbatch "${args[@]}")
  b=$(sbatch --parsable --exclude=$EXCL -p gpu_requeue -A ydu_lab -J mvv2_$key \
        scripts/rh20t/train_multiview.sbatch "${args[@]}")
  echo "$key  init=$(basename $init)  wandb=$id  jobs: $a (kempner_requeue) $b (gpu_requeue)"
done
