#!/bin/bash
#SBATCH -J rh20t_depth
#SBATCH -o /n/lab_storage/ydu_lab/Lab/haonan/datasets/rh20t/logs/%x_%A_%a.out
#SBATCH --account=ydu_lab
#SBATCH --partition=sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-7
# RH20T per-episode depth (original release, 640x360, byte-split video coding; see
# fastvideo/dataset/rh20t_depth.py) from the HaojunChen/RH20T HF mirror. The official
# tonyfanggalaxies/RH20T repo has no depth split (RH20T_depth 404s as of 2026-10-21).
# Each cfg is a multi-part tar.gz; the parts are streamed in order into one tar process.
# Kept: robot episodes, static cameras only (human demos and in-hand cameras skipped).
#   depth/RH20T_cfgN/<episode>/cam_<serial>/depth.mp4   (frame i <-> color frame i)
# --touch: tar would otherwise keep the 2023 mtimes.
set -o pipefail
ROOT=${ROOT:-/n/lab_storage/ydu_lab/Lab/haonan/datasets/rh20t}
C=${SLURM_ARRAY_TASK_ID:?run as an array job (1-7)}
BASE=https://huggingface.co/datasets/HaojunChen/RH20T/resolve/main/depth
declare -A PARTS=([1]=12 [2]=6 [3]=2 [4]=5 [5]=4 [6]=5 [7]=3)
# In-hand cameras per cfg (same as IN_HAND in fastvideo/dataset/rh20t_multiview.py)
declare -A IN_HAND=([1]="043322070878" [2]="104422070042" [3]="045322071843" [4]="045322071843"
  [5]="104422070042 135122079702" [6]="135122075425 135122070361" [7]="135122075425 135122070361")

excludes=(--exclude='*_human' --exclude='*_human/*')
for s in ${IN_HAND[$C]}; do excludes+=(--exclude="*/cam_$s" --exclude="*/cam_$s/*"); done

stream () {
  for ((i = 0; i < PARTS[$C]; i++)); do
    curl -sSfL --retry 5 --retry-delay 30 "$BASE/RH20T_cfg${C}_depth.tar.gz.part-$(printf %03d $i)" || return 1
  done
}

mkdir -p "$ROOT/depth" "$ROOT/logs"
[ -e "$ROOT/logs/depth_cfg$C.ok" ] && { echo "cfg$C already done"; exit 0; }
for attempt in 1 2 3; do
  echo "[$(date +%T)] cfg$C depth attempt $attempt (${PARTS[$C]} parts)"
  if stream | tar xzf - -C "$ROOT/depth" --touch "${excludes[@]}"; then
    touch "$ROOT/logs/depth_cfg$C.ok" "$ROOT/depth/RH20T_cfg$C/.complete"  # .complete gates the dataset record cache
    echo "[$(date +%T)] cfg$C depth OK"
    du -sh "$ROOT/depth/RH20T_cfg$C"
    exit 0
  fi
done
echo "[$(date +%T)] cfg$C depth FAILED"
exit 1
