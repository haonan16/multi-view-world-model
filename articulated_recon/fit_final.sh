#!/bin/bash
# final 3D fit of the selected predictions: preprocess (predicted RGB -> colour masks, SAM 3, DA3), fit, overlay
# fit_final.sh <list>
R=/n/lab_storage/ydu_lab/Lab/akiruga/geo_runs/rollouts_A7000
PY=${PY:-/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin/python}
export HF_HOME=/n/netscratch/ydu_lab/Lab/akiruga HF_HUB_OFFLINE=1
for n in $(cat $1); do
  d=$R/$n; o=results/wm/${n}_da3_final
  [ -f $o/overlay.jpg ] && continue
  [ -f $d/obs.npz ] || $PY prep_roll.py $d
  [ -f $d/color.npz ] || $PY color_masks.py $d
  [ -f $d/da3.npz ] || $PY da3_wm.py roll $d
  [ -f $d/sam3.npz ] || { echo $d > logs/one_$(basename $1).txt; $PY seg_sam3.py --many logs/one_$(basename $1).txt; }
  HOLD_LAST=1 OBJ_WDEPTH=0.05 $PY wm_fit.py roll:$d --depth da3 --stage all --tag _final 2>&1 | grep "frame-0\|silhouette IoU\|Error\|Traceback" | sed "s#^#$n #"
  $PY wm_overlay.py $o $d
done
