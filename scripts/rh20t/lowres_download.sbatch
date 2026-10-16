#!/bin/bash
#SBATCH -J rh20t_lowres
#SBATCH -o /n/netscratch/ydu_lab/Lab/akiruga/rh20t_lowres/logs/%x_%A_%a.out
#SBATCH --account=ydu_lab
#SBATCH --partition=sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-12:00:00
#SBATCH --array=1-7
# RH20T official 320x180 release (tonyfanggalaxies/RH20T), streamed straight into the
# extracted tree -- no archive copy is kept on disk (project storage cap: 3 TB total).
#   RH20T_color/RH20T_cfgN.tar.gz  -> data/RH20T_cfgN/<episode>/cam_<serial>/color/{color.mp4,timestamps.npy}
#   RH20T_lowdim/RH20T_cfgN.tar.gz -> only */metadata.json (calib timestamp + rating per episode)
# Calibration (RH20T_calib) is small and was fetched separately into calib/.
# gzip verifies its CRC at end of stream; a nonzero tar/curl exit marks the cfg FAILED.
set -o pipefail
ROOT=/n/netscratch/ydu_lab/Lab/akiruga/rh20t_lowres
C=${SLURM_ARRAY_TASK_ID}
BASE=https://huggingface.co/datasets/tonyfanggalaxies/RH20T/resolve/main
mkdir -p "$ROOT/data" "$ROOT/logs"
cd "$ROOT/data" || exit 1

fetch () {  # $1=split  $2..=extra tar args
  local split=$1; shift
  for attempt in 1 2 3; do
    echo "[$(date +%T)] cfg$C $split attempt $attempt"
    if curl -sSfL --retry 5 --retry-delay 30 "$BASE/$split/RH20T_cfg$C.tar.gz" | tar xzf - "$@"; then
      echo "[$(date +%T)] cfg$C $split OK"; return 0
    fi
  done
  echo "[$(date +%T)] cfg$C $split FAILED"; return 1
}

# STAGES=color,meta (default both); a stage whose .ok marker exists is skipped.
STAGES=${STAGES:-color,meta}
if [[ $STAGES == *color* && ! -e $ROOT/logs/color_cfg$C.ok ]]; then
  fetch RH20T_color && touch "$ROOT/logs/color_cfg$C.ok"
fi
if [[ $STAGES == *meta* && ! -e $ROOT/logs/meta_cfg$C.ok ]]; then
  fetch RH20T_lowdim --wildcards '*/metadata.json' && touch "$ROOT/logs/meta_cfg$C.ok"
fi
du -s --apparent-size "$ROOT/data/RH20T_cfg$C" 2>/dev/null
