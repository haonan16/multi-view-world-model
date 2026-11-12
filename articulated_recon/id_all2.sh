#!/bin/bash
# object-centric plan from the reconstruction (contact snap, static unless held, grasp prior, no robot-fit approach
# waypoints); the same rules for every goal source; world-model goals anchored to the current observation (frame 0)
P=/n/lab_storage/ydu_lab/akiruga/.conda/envs/s4d/bin/python
F="--act_h 32 --reps 1 --final_reps 3 --snap 1 --static 1 --grasp_prior 1 --approach 0"
run() { $P id_run.py "$@" $F 2>&1 | grep -E "SUMMARY|Traceback|Error" ; }
for mode in retarget learned; do
  run --mode $mode --goal oracle --list results/id/gt_list.txt --out results/id/v2_${mode}_oracle | sed "s/^/$mode oracle /" &
  run --mode $mode --goal recon --fit_tag _gt_raw_idgt --list results/id/gt_list.txt --out results/id/v2_${mode}_reconGT | sed "s/^/$mode recon-GT /" &
  run --mode $mode --goal recon --fit_tag _da3_final --anchor _gt_raw_idgt --list results/wm/selected.txt --out results/id/v2_${mode}_reconWM | sed "s/^/$mode recon-WM /" &
done
wait; echo ALLDONE2
