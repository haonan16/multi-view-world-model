#!/bin/bash
# Best RH20T model (tokens design, PRoPE camera conditioning): chained 10k-step stages,
# 4 GPUs each (scripts/rh20t/train_multiview.sbatch), each stage initialised from the last.
#   1. base         rh20t_multiview_v2, history mostly frame 0 of every view
#   2. single-view  same data, history mostly frame 0 of ONE view (the test-time setting)
#   3. (human only) RH20T human demos mixed 50/50 into training
#   bash scripts/rh20t/train_best.sh robot|human     [ROOT=<run dir root>]
# Partition/account: SBATCH_PARTITION / SBATCH_ACCOUNT env vars (or edit the .sbatch header).
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA=${1:?robot or human}
ROOT=${ROOT:-$PWD/runs/prope}
COMMON=(algorithm.cam_mode=prope algorithm.text.enabled=true
        algorithm.gradient_checkpointing_rate=1 algorithm.noise.hist_noise_prob=0.2)
MIX_BASE="algorithm.noise.mix={first_frame:0.4,rand_history:0.15,independent:0.2,view_history:0.15,none:0.1}"
MIX_SV="algorithm.noise.mix={single_view:0.5,first_frame:0.15,rand_history:0.1,independent:0.1,view_history:0.05,none:0.1}"
export DESIGN=tokens STEPS=10000 BS=8 CKPT_EVERY=1000

stage() {   # name dataset init_ckpt dependency_job [overrides...]
  local name=$1 ds=$2 init=$3 dep=$4; shift 4
  local a=("${COMMON[@]}" "$@")
  [ -n "$init" ] && a+=("algorithm.model.tuned_ckpt_path='$init'")
  NAME=prope_$name DATASET=$ds RUN_DIR=$ROOT/$name sbatch --parsable -J prope_$name \
    ${dep:+--dependency=afterok:$dep} scripts/rh20t/train_multiview.sbatch "${a[@]}"
}

j1=$(stage base rh20t_multiview_v2 "" "" "$MIX_BASE")
j2=$(stage sv rh20t_multiview_v2 "$ROOT/base/checkpoints/last.ckpt" "$j1" "$MIX_SV" algorithm.val_hist_views=1)
echo "base: $j1   single-view: $j2"
if [ "$DATA" = human ]; then
  j3=$(stage svh rh20t_multiview_v2_human "$ROOT/sv/checkpoints/last.ckpt" "$j2" "$MIX_SV" algorithm.val_hist_views=1)
  echo "robot+human: $j3"
fi
