#!/bin/bash
# object stage once per episode for run id $RUN (marker results/real_v2/<ep><tag>/.done_$RUN); waits for inputs
# usage (on the GPU job): RUN=final1 bash obj_worker.sh <list> [extra real_v2 args...]
L=$1; shift; EXTRA="$@"
PY=${PY:-/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin/python}
TAG=$(echo "$EXTRA" | grep -o -- "--tag [^ ]*" | cut -d" " -f2)
for pass in $(seq 1 60); do
  left=0
  for e in $(cat $L); do
    d=results/real_v2/$e$TAG
    [ -f $d/.done_$RUN ] && continue
    if [ -f results/real_v2/$e/robot.pkl ] && [ -f real/v2/$e/sam3.npz ] && [ -f real/v2/$e/ann.json ]; then
      $PY real_v2.py $e --stage objects $EXTRA > logs/obj_$e$TAG.log 2>&1
      mkdir -p $d; touch $d/.done_$RUN
      echo "$e$TAG $(grep -ao '"obj/[a-z_]*/iou_mean": [0-9.]*' logs/obj_$e$TAG.log | tr '\n' ' ')"
    else left=$((left+1)); fi
  done
  [ $left -eq 0 ] && break; sleep 60
done
echo ALLDONE
