#!/bin/bash
P=/n/lab_storage/ydu_lab/akiruga/.conda/envs/s4d/bin/python
F="--act_h 32 --reps 1 --final_reps 3 --snap 1 --static 1 --grasp_prior 1 --approach 0"
run() { $P id_run.py "$@" $F 2>&1 | grep -E "SUMMARY|Traceback|Error" ; }
run --mode retarget --goal oracle --list results/id/gt_list.txt --out results/id/v3_retarget_oracle | sed "s/^/retarget oracle /" &
run --mode learned --goal oracle --list results/id/gt_list.txt --out results/id/v3_learned_oracle | sed "s/^/learned oracle /" &
run --mode learned --goal recon --fit_tag _gt_raw_idgt --list results/id/gt_list.txt --out results/id/v3_learned_reconGT | sed "s/^/learned recon-GT /" &
run --mode learned --goal recon --fit_tag _da3_final --anchor _gt_raw_idgt --list results/wm/selected.txt --out results/id/v3_learned_reconWM | sed "s/^/learned recon-WM /" &
wait; echo ALLDONE3
