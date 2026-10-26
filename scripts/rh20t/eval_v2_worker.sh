#!/bin/bash
# Work-pool worker for the v2 eval: tasks = (pass, model); each task runs
# infer_multiview.py for ONE model into $OUT/<pass>/<model>/ (raw uint8 frames + latents).
# Any number of workers (local GPUs, requeue jobs) can run at once: a task is claimed by an
# atomic mkdir of $OUT/claims/<pass>__<model>; a claim older than STALE_MIN without a result
# (e.g. a preempted job) is taken over. Done = <pass>/<model>/metrics.json exists.
#   OUT=DIR [N=25] [EXTRA="--hist_views 1"] [PASSES=guide_t2_h1] bash scripts/rh20t/eval_v2_worker.sh             (local; set CUDA_VISIBLE_DEVICES)
#   OUT=DIR sbatch -p gpu_requeue -A ydu_lab scripts/rh20t/eval_v2_worker.sh
# Models: $OUT/models.txt = label=cam_mode:ckpt,...
#SBATCH -J mvv2_evalw
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200|a100|rtx6000pro
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --requeue
set -uo pipefail
cd /n/lab_storage/ydu_lab/Lab/akiruga/ego_world_model/ego_world_model_clean
export PATH=/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin:$PATH
export HF_HOME=/n/netscratch/ydu_lab/Lab/akiruga HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
OUT=${OUT:?set OUT}; N=${N:-25}; STALE_MIN=${STALE_MIN:-40}
# a requeued batch job keeps its job id, so it reclaims its own task; local workers are host:pid
[ "${SLURM_JOB_NAME:-}" = mvv2_evalw ] && ME=$SLURM_JOB_ID || ME=$(hostname):$$
declare -A ARGS=([noguide]="" [guide_t2_h1]="--text_guidance 2 --hist_guidance 1")
mkdir -p $OUT/claims
for pass in ${PASSES:-noguide guide_t2_h1}; do
  for spec in $(tr ',' ' ' < $OUT/models.txt); do
    m=${spec%%=*}; dst=$OUT/$pass/$m; c=$OUT/claims/${pass}__$m
    [ -f $dst/metrics.json ] && continue
    if ! mkdir $c 2>/dev/null; then
      owner=$(cat $c/owner 2>/dev/null)
      age=$(( ($(date +%s) - $(stat -c %Y $c/owner 2>/dev/null || date +%s)) / 60 ))
      [ "$owner" = "$ME" ] || [ $age -ge $STALE_MIN ] || continue
    fi
    echo $ME > $c/owner
    echo "[$ME] $(date +%T) $pass / $m"
    python scripts/rh20t/infer_multiview.py --out $dst --n $N --models "$spec" \
      --save_latents ${ARGS[$pass]} ${EXTRA:-} > $dst.log 2>&1 \
      && echo "[$ME] $(date +%T) done $pass / $m" || echo "[$ME] FAILED $pass / $m (see $dst.log)"
  done
done
echo "[$ME] no tasks left"
