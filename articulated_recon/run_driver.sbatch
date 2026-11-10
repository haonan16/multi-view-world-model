#!/bin/bash
#SBATCH -p gpu_requeue,kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=cc9.0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/slurm_%j.out
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1 HF_HOME=/n/netscratch/ydu_lab/Lab/akiruga/hf_cache
nvidia-smi --query-gpu=name,memory.total --format=csv
/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin/python -u driver.py "$@"
