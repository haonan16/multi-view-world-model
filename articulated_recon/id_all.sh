#!/bin/bash
# inverse dynamics from 3D, executed in ManiSkill: oracle subgoals vs our reconstruction (GT frames), then world-model predictions
P=/n/lab_storage/ydu_lab/akiruga/.conda/envs/s4d/bin/python
L="--act_h 32 --reps 1 --final_reps 3"
for mode in learned retarget; do
  $P id_run.py --mode $mode --goal oracle --list results/id/gt_list.txt $L 2>&1 | grep -E "SUMMARY|Traceback" | sed "s/^/$mode oracle /"
  $P id_run.py --mode $mode --goal recon --fit_tag _gt_raw_idgt --list results/id/gt_list.txt $L 2>&1 | grep -E "SUMMARY|Traceback" | sed "s/^/$mode recon-GT /"
  $P id_run.py --mode $mode --goal recon --fit_tag _da3_final --list results/wm/selected.txt $L 2>&1 | grep -E "SUMMARY|Traceback" | sed "s/^/$mode recon-WM /"
done
echo ALLDONE
