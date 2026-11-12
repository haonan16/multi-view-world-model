#!/bin/bash
# our 3D reconstruction on the simulator's own frames (GT RGB + sim depth) for the inverse-dynamics test
PY=${PY:-/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin/python}
for n in $(cat results/id/gt_list.txt); do
  o=results/wm/${n}_gt_raw_idgt
  [ -f $o/fit.pkl ] && continue
  HOLD_LAST=1 $PY wm_fit.py gt:results/id/gtsrc/$n --depth raw --stage all --tag _idgt 2>&1 | grep "frame-0\|robot vs GT\|silhouette IoU\|Error\|Traceback\|task" | sed "s#^#$n #"
done
