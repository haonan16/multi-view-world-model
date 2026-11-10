#!/bin/bash
# prep every episode of episodes25.txt whose colour + depth have been streamed (T=20, 2 s padding)
S=/n/netscratch/ydu_lab/Lab/akiruga/rh20t_cfg5_stream; R=/n/lab_storage/ydu_lab/Lab/akiruga/articulated_recon
for E in $(cat $R/real/episodes25.txt); do
  [ -f $R/real/v2/$E/obs.npz ] && continue
  [ -d $S/color/RH20T_cfg5/$E/transformed ] && [ -d $S/depth/RH20T_cfg5/$E ] || { echo "skip $E (not streamed)"; continue; }
  PYTHONNOUSERSITE=1 /n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin/python $R/real/prep_rh20t_episode.py \
     --color_dir $S/color/RH20T_cfg5/$E --depth_dir $S/depth/RH20T_cfg5/$E --cfg 5 --out $R/real/v2/$E --T 20 --pad_s 2 > $R/logs/prep_$E.log 2>&1 \
     && echo "ok $E" || echo "FAIL $E"
done
