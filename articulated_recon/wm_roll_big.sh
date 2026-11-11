#!/bin/bash
# run world-model rollouts on a GPU holder job: GPUJOB=<id> wm_roll.sh <shard> <seeds> <n>
J=${GPUJOB:?}
exec env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin srun --jobid=$J --overlap bash -c "
cd /n/lab_storage/ydu_lab/Lab/akiruga/ego_world_model/ego_world_model_clean
export PATH=/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin:\$PATH PYTHONNOUSERSITE=1 PYTHONPATH=\$PWD
export NVTE_CUDA_INCLUDE_DIR=/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/lib/python3.12/site-packages/nvidia/cuda_runtime/include HF_HOME=/n/netscratch/ydu_lab/Lab/akiruga HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/geo/rollout_batch.py --out /n/lab_storage/ydu_lab/Lab/akiruga/geo_runs/rollouts_A7000 \
  --run A=depth:/n/lab_storage/ydu_lab/Lab/akiruga/geo_runs/best/geo_A_depth.ckpt --n ${3:-50} --seeds ${2:-0} --shard ${1:-0/1} --tasks ${4:-StackCube-v1,PullCubeTool-v1,PegInsertionSide-v1}
"
