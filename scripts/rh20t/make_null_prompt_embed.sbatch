#!/bin/bash
#SBATCH -J null_embed
#SBATCH -o slurm_logs/null_embed_%j.out
#SBATCH --account=ydu_lab
#SBATCH --partition=sapphire,shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-00:30:00
cd "$SLURM_SUBMIT_DIR" || exit 1
export HF_HOME=/n/netscratch/ydu_lab/Lab/akiruga HF_HUB_OFFLINE=1
/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin/python scripts/rh20t/make_null_prompt_embed.py \
  /n/netscratch/ydu_lab/Lab/akiruga/rh20t_lowres/null_prompt_embed_wan21.pt
