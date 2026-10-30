#!/bin/bash
#SBATCH -J rh20t_lowdim
#SBATCH -o /n/lab_storage/ydu_lab/Lab/haonan/datasets/rh20t/logs/%x_%A_%a.out
#SBATCH --account=ydu_lab
#SBATCH --partition=sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-12:00:00
#SBATCH --array=1-7
# RH20T robot low-dim signals + calibration from the authors' official HF release
# (tonyfanggalaxies/RH20T), streamed straight into an extracted tree on lab_storage
# (NOT netscratch: tar keeps the 2021-2022 mtimes, which the 90-day netscratch purge
# deletes). Only */transformed/* and */metadata.json are kept (~6 MB per episode);
# audio and the raw command logs are skipped.
#   RH20T_lowdim/RH20T_cfgN.tar.gz -> lowdim/RH20T_cfgN/<episode>/{metadata.json,transformed/*.npy}
#   RH20T_calib/RH20T_cfgN.tar.gz  -> calib/RH20T_cfgN/calib/<ts>/...
# Color video comes from akiruga's 320x180 copy (rh20t_lowres/data), keyed by episode name.
set -o pipefail
ROOT=${ROOT:-/n/lab_storage/ydu_lab/Lab/haonan/datasets/rh20t}
C=${SLURM_ARRAY_TASK_ID:?run as an array job (1-7)}
BASE=https://huggingface.co/datasets/tonyfanggalaxies/RH20T/resolve/main

fetch () {  # $1=split  $2=dest dir  $3..=tar member filters
  local split=$1 dest=$2; shift 2
  mkdir -p "$dest"
  for attempt in 1 2 3; do
    echo "[$(date +%T)] cfg$C $split attempt $attempt"
    if curl -sSfL --retry 5 --retry-delay 30 "$BASE/$split/RH20T_cfg$C.tar.gz" \
        | tar xzf - -C "$dest" --wildcards "$@"; then
      echo "[$(date +%T)] cfg$C $split OK"; return 0
    fi
  done
  echo "[$(date +%T)] cfg$C $split FAILED"; return 1
}

if [ ! -e "$ROOT/logs/calib_cfg$C.ok" ]; then
  fetch RH20T_calib "$ROOT/calib" '*' && touch "$ROOT/logs/calib_cfg$C.ok"
fi
if [ ! -e "$ROOT/logs/lowdim_cfg$C.ok" ]; then
  fetch RH20T_lowdim "$ROOT/lowdim" '*/transformed/*' '*/metadata.json' \
    && touch "$ROOT/logs/lowdim_cfg$C.ok"
fi
du -sh "$ROOT/lowdim/RH20T_cfg$C" "$ROOT/calib/RH20T_cfg$C" 2>/dev/null
