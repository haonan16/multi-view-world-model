#!/bin/bash
#SBATCH -J s13B_tta_F
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.out
#SBATCH --mail-user=amaniedu71@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_ydu_lab
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append

# 1.3B (Wan2.1-T2V-1.3B) concat MoT, TARTANAIR V1 ONLY, MODEL-F data encoding.
# Rebuttal variant using the 5B model-F scale scheme (launch_5b_F_Aresume_14ds_dpose_16node):
#   - PER-CLIP scale (dataset.scale_preserving=false): s = max(max|moment|, 1.0) divides
#     moments; depth = disparity 2/(1+d/s)-1 (datasets/tartanair.py _apply_mutual_scale).
#   - ALL diffusion settings: subset_settings.tartanair default [0,1,2,3,4] (not F's
#     depth+pose-only [1]) -> the model trains on RGB generation/NVS AND rays+depth.
# Differences from 5B F: algorithm=wan_t2v_ray_depth_mot_concat (1.3B, 16-ch Wan2.1 VAE),
# starts from pretrained Wan2.1-1.3B RGB weights (no 5B Model-A ckpt to warm-start from),
# lr stays at the 1e-5 default (F's 4e-5 was tuned for global batch 256; here gb=16), and
# 1.3B RESOLUTION 240x320 / raymap 30x40 (as all prior 1.3B runs: dl3dv_only,
# mixture_tartanair) — mix_only_tartanair.yaml hardcodes the 5B's 288x384, so we override.
# kempner_requeue can preempt: --requeue restarts this script with the SAME job id, so the
# run dir is stable and we warm-start from our own last.ckpt when it exists.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 1
mkdir -p slurm_logs
module load Mambaforge 2>/dev/null || true
mamba deactivate 2>/dev/null || true
mamba activate test2
module load cuda/12.4.1-fasrc01 2>/dev/null || true
module load gcc/9.5.0-fasrc01 2>/dev/null || true

export HF_HOME=/n/netscratch/ydu_lab/Lab/akiruga
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

RUN_DIR=/n/netscratch/ydu_lab/Lab/akiruga/vwm_train_runs/s13B_tta_F_4g_${SLURM_JOB_ID}
RESUME_ARG=""
if [ -f "$RUN_DIR/checkpoints/last.ckpt" ]; then
  RESUME_ARG="algorithm.model.tuned_ckpt_path=$RUN_DIR/checkpoints/last.ckpt"
  echo "REQUEUE: warm-starting from $RUN_DIR/checkpoints/last.ckpt"
fi

srun python -m main \
  hydra.run.dir=$RUN_DIR \
  +name=rebuttal_1-3b_ttav1_F \
  experiment=exp_video \
  algorithm=wan_t2v_ray_depth_mot_concat \
  ${RESUME_ARG:+$RESUME_ARG} \
  dataset=mix_only_tartanair \
  dataset.scale_preserving=false \
  dataset.height=240 \
  dataset.width=320 \
  dataset.raymap_height=30 \
  dataset.raymap_width=40 \
  dataset.n_frames=50 \
  algorithm.model.model_type=t2v \
  algorithm.diffusion_forcing.enabled=True \
  algorithm.diffusion_forcing.mode=ray_depth_prediction_firstlast \
  algorithm.diffusion_forcing.train_rgb=true \
  algorithm.diffusion_forcing.mask_clean_token_loss=false \
  experiment.num_nodes=1 \
  experiment.training.batch_size=4 \
  experiment.training.data.num_workers=10 \
  experiment.validation.data.num_workers=2 \
  algorithm.gradient_checkpointing_rate=1.0 \
  experiment.training.optim.accumulate_grad_batches=1 \
  algorithm.model.compile=false \
  algorithm.vae.compile=false \
  algorithm.text_encoder.compile=false \
  experiment.training.checkpointing.every_n_train_steps=500 \
  experiment.training.checkpointing.save_weights_only=True \
  experiment.validation.val_every_n_step=100000 \
  experiment.validation.limit_batch=0 \
  experiment.validation.num_sanity_val_steps=0 \
  wandb.mode=online \
  cluster=fas_high \
  cluster.params.memory=1000G \
  +_on_compute_node=True
echo "TRAIN_EXIT=$?"
