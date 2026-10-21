#!/bin/bash
# One-time setup of a GPU node for the light-stage pipeline (see README.md). Run from the repo root:
#   bash lightstage/setup_node.sh
# Everything goes to node-local /tmp (the scripts hardcode /tmp/akiruga_lyra and /tmp/akiruga_lightstage).
set -e
REPO=$(pwd)
ENV=/tmp/akiruga_envs/test2; PY=$ENV/bin/python
export PYTHONNOUSERSITE=1

# 1. python env: packed conda env "test2" (python 3.12, torch 2.9 cu129, transformer_engine, flash-attn, megatron-core, MoGe)
if [ ! -x $PY ]; then
  mkdir -p /tmp/akiruga_envs && zstd -dc /n/netscratch/ydu_lab/Lab/akiruga/envs/test2.tar.zst | tar -C /tmp/akiruga_envs -xf -
fi
$PY -m pip install -q --no-deps pyquaternion webdataset ffmpegcv braceexpand lru-dict portalocker fasteners ninja

# 2. pipeline code at the paths the scripts expect
mkdir -p /tmp/akiruga_lyra /tmp/akiruga_lightstage/cases
cp $REPO/lightstage/lyra/* /tmp/akiruga_lyra/
cp $REPO/lightstage/rig/* /tmp/akiruga_lightstage/
chmod +x /tmp/akiruga_lyra/*.sh

# 3. Lyra 2.0 code (+ Depth Anything 3 submodule) and our seed-view patch
if [ ! -d /tmp/akiruga_lyra/src ]; then
  git clone https://github.com/nv-tlabs/lyra.git /tmp/akiruga_lyra/src
  git -C /tmp/akiruga_lyra/src checkout 9fffc9adc37004091ecf26ef03abfb3abdf4d59a   # commit the results were made with
  git -C /tmp/akiruga_lyra/src submodule update --init --depth 1 Lyra-2/lyra_2/_src/inference/depth_anything_3
fi
(cd /tmp/akiruga_lyra/src/Lyra-2 && $PY $REPO/lightstage/patch_lyra_multiview_seeds.py)

# 4. Lyra 2.0 checkpoints (~97 GB) and Depth Anything 3 weights (HF cache)
[ -f /tmp/akiruga_lyra/src/Lyra-2/checkpoints/vae/vae.pth ] || bash /tmp/akiruga_lyra/dl.sh
$PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('depth-anything/DA3NESTED-GIANT-LARGE-1.1'))"
echo "setup done"
