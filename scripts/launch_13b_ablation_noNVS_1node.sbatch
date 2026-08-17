#!/bin/bash
#SBATCH -J s13B_ablNoNVS
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

# === R1 ABLATION: M_noNVS ===
# Implemented as mode-1-only (all RGB clean history -> predict rays+depth) for BOTH datasets.
# NOTE: "no NVS" here means dropping ALL generative modes, not just the camera-controlled
# ones (2,3) -- mode-1-only also removes plain gen (0) and depth-conditioned gen (4). So it
# is effectively a pure discriminative geometry predictor (the single-task specialist R1
# asks for). Describe it as such in the writeup. It never generates RGB and never sees
# sparse (first/first+last) context, so comparing M_full vs this shows what the generative
# modes contribute to (a) depth prediction and (b) SPARSE-view pose (the model here has only
# ever seen full-video context -> sparse pose should collapse, the H4 result).
# REBUTTAL_RGB_CONTEXT_ONLY=1 makes this a TRUE geometry-only predictor: RGB is forced clean
# (t=0 always, overriding the 50/50 clean_hist noising) and its generative loss is excluded
# (w_rgb=0). So the model only ever learns clean-RGB-video -> rays+depth; it never denoises or
# generates RGB. rays+depth are supervised normally in mode 1. Everything else identical to
# M_full (launch_13b_tta_re10k_F_1node): 50/50 sampling, per-clip F scale, 240x320, bs4,
# gckpt 1.0, no grad-clip, lr 1e-5.
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
export REBUTTAL_RGB_CONTEXT_ONLY=1   # RGB = clean observed input only: forced t=0 + no RGB loss
                                     # -> pure clean-RGB-video -> rays+depth geometry predictor

RUN_DIR=/n/netscratch/ydu_lab/Lab/akiruga/vwm_train_runs/s13B_ablNoNVS_4g_${SLURM_JOB_ID}
RESUME_ARG=""
if [ -f "$RUN_DIR/checkpoints/last.ckpt" ]; then
  RESUME_ARG="algorithm.model.tuned_ckpt_path=$RUN_DIR/checkpoints/last.ckpt"
  echo "REQUEUE: warm-starting from $RUN_DIR/checkpoints/last.ckpt"
fi

srun python -m main \
  hydra.run.dir=$RUN_DIR \
  +name=rebuttal_ablation_noNVS \
  experiment=exp_video \
  algorithm=wan_t2v_ray_depth_mot_concat_rebuttal \
  ${RESUME_ARG:+$RESUME_ARG} \
  dataset=mixture_tartanair_re10k \
  dataset.training.weight_type=absolute \
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
  'algorithm.diffusion_forcing.subset_settings.tartanair=[1]' \
  'algorithm.diffusion_forcing.subset_settings.re10k=[1]' \
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
