#!/bin/bash
# stream RH20T cfg5 colour (Malak-Mansour) + depth (introvoyz043) for the episodes in episodes25.txt; no archive stored
D=/n/netscratch/ydu_lab/Lab/akiruga/rh20t_cfg5_stream
PAT=$(sed 's|.*|*&/*|' /n/lab_storage/ydu_lab/Lab/akiruga/articulated_recon/real/episodes25.txt | tr '\n' ' ')
which=$1
if [ "$which" = color ]; then
  curl -sL --retry 5 https://huggingface.co/datasets/Malak-Mansour/RH20T/resolve/main/RH20T_cfg5.tar.gz | tar -xz --wildcards -C $D/color $PAT
else
  ( for p in 000 001 002 003; do curl -sL --retry 5 https://huggingface.co/datasets/introvoyz043/RH20T/resolve/main/depth/RH20T_cfg5_depth.tar.gz.part-$p; done ) | tar -xz --wildcards -C $D/depth $PAT
fi
echo "done $which $? $(date)"
