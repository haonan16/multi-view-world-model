#!/bin/bash
# Usage: run_sam3d.sh --image img.png --mask mask.png --out out.ply [--seed 0] [--depth d.npy --K fx,fy,cx,cy]
# Runs sam3d_mesh.py in the sam3d env. On this login node (no GPU) it goes through the holder job:
#   GPUJOB=59472673 run_sam3d.sh ...   (uses srun --overlap with a clean env)
S=/n/lab_storage/ydu_lab/Lab/akiruga/articulated_recon/sam3d
ARGS=$(printf '%q ' "$@")
CMD="source $S/env.sh; cd $S; python $S/sam3d_mesh.py $ARGS"
if [ -n "$GPUJOB" ]; then
  exec env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin HF_TOKEN=$HF_TOKEN \
    srun --jobid=$GPUJOB --overlap bash -c "$CMD"
else
  exec bash -c "$CMD"
fi
