#!/bin/bash
# usage: run_lyra.sh <gpu> <tag> <extra args...> -- cases...
cd /tmp/akiruga_lyra/src/Lyra-2
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SITE=/tmp/akiruga_envs/test2/lib/python3.12/site-packages
export LD_LIBRARY_PATH=/tmp/akiruga_envs/test2/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export CUDNN_PATH=$SITE/nvidia/cudnn
export PYTHONPATH=.:lyra_2/_src/inference/depth_anything_3/src
g=$1; shift
CUDA_VISIBLE_DEVICES=$g /tmp/akiruga_envs/test2/bin/python /tmp/akiruga_lyra/$SCRIPT "$@"
