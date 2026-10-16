#!/bin/bash
#SBATCH -J pack_env
#SBATCH -o slurm_logs/pack_env_%j.out
#SBATCH --account=ydu_lab
#SBATCH --partition=sapphire,shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-06:00:00
# Snapshot the test2 conda env (on slow lab_storage NFS) into one archive on netscratch.
# Training jobs unpack it onto node-local /tmp (scripts/rh20t/train_multiview.sbatch), so
# Python imports read local NVMe instead of stat-ing ~100k files over NFS per rank.
set -eo pipefail
OUT=/n/netscratch/ydu_lab/Lab/akiruga/envs/test2.tar.zst
SRC=/n/lab_storage/ydu_lab/akiruga/.conda/envs
date; tar -C "$SRC" --exclude='*/__pycache__' -cf - test2 | zstd -T4 -3 -q -o "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
ls -la "$OUT"; date; echo PACK_OK
