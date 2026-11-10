#!/bin/bash
# run a command on the GPU holder job (clean env: this login session lives inside another Slurm job)
J=${GPUJOB:-59530030}
exec env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin HF_TOKEN=$HF_TOKEN \
  srun --jobid=$J --overlap bash -c "cd /n/lab_storage/ydu_lab/Lab/akiruga/articulated_recon; export PYTHONNOUSERSITE=1 HF_HOME=/n/netscratch/ydu_lab/Lab/akiruga/hf_cache; PY=/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin/python; $*"
