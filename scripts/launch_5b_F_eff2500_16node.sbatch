#!/bin/bash
#SBATCH -J s5bF_eff2500
#SBATCH -o slurm_logs/s5bF_eff2500_%j.out
#SBATCH -e slurm_logs/s5bF_eff2500_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --partition=kempner_h200_priority
#SBATCH --account=kempner_ydu_lab
#SBATCH --qos=kemp_gpu64_id36
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=1400G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

# ============================================================================
# MODEL F -- the recipe that produced F-best (cumulative step 2500, "eff-2500").
#
# Reconstructed from the AUTHORITATIVE record: the resolved Hydra overrides of
# job 33283037, i.e.
#   /n/netscratch/ydu_lab/Lab/akiruga/vwm_train_runs/s5bF_Aresume14_64g_33283037/
#       .hydra/overrides.yaml
# NOT from the later launch script, which had drifted to batch_size=4 /
# gradient_checkpointing_rate=0.50 (global batch 256) for the 33987485 resume.
# The numbers below are the ones that trained the checkpoint we report:
#
#   batch_size                     = 16   -> GLOBAL BATCH 1024 (16 nodes x 4 GPU x 16)
#   gradient_checkpointing_rate    = 0.925
#   lr                             = 4e-5, grad-clip 1.0
#
# WHAT MODEL F IS
#   5B Wan2.2 MoT-concat (rgb | ray_d+ray_m fused | depth), per-clip scale
#   normalisation (`scale_preserving=false`: s = max(max|Plucker moment|, 1.0),
#   depth = disparity 2/(1+d/s)-1). `dataset=mixture_parallax14` supplies only the
#   14-subset ROSTER -- parallax/scheme-C is NOT active in this run.
#   Task split: NVS (settings [2,3]) for re10k + dl3dv_colmap only; the 12
#   synthetic subsets are depth+pose only (setting [1]).
#
# WARM-START CHAIN (weights-only at each hop)
#   Model A best  s5bA_perclip_32g_30557574/checkpoints/epoch=0-step=2000.ckpt
#     -> F 32852495  (cum step 1000)
#     -> F 33023613  (cum step 2000)
#     -> F 33283037  (cum step 2500)  <-- F-best, eff-2500
#   Set TUNED_CKPT below to the appropriate link in that chain. To reproduce
#   eff-2500 exactly you must replay all three hops; a single 2500-step run from
#   A-best is NOT equivalent because each hop resets the AdamW state and re-runs
#   the 1000-step LR warmup.
#
# Only 7 of the 12 synthetic subsets are overridden to [1] below. The other five
# (tartanair_v2, ase, matrixcity, omniworld_game, scenenet) already default to [1]
# in configurations/algorithm/wan_t2v_ray_depth_mot.yaml.
# ============================================================================

set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 1
mkdir -p slurm_logs

module load Mambaforge 2>/dev/null || true
mamba deactivate 2>/dev/null || true
mamba activate test2
module load cuda/12.4.1-fasrc01 2>/dev/null || true
module load gcc/9.5.0-fasrc01 2>/dev/null || true

# --- data roots: point these at your copies -------------------------------
export HF_HOME=${HF_HOME:?set HF_HOME to a writable HuggingFace cache}
export ASE_DATA_ROOT=${ASE_DATA_ROOT:?set ASE_DATA_ROOT to the ASE scenes dir}
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- the checkpoint this segment warm-starts from -------------------------
TUNED_CKPT=${TUNED_CKPT:?set TUNED_CKPT (see WARM-START CHAIN above)}

srun python -m main \
  hydra.run.dir=${RUN_DIR:-runs/s5bF_eff2500_${SLURM_JOB_ID}} \
  +name=s5bF_Aresume14_64g \
  experiment=exp_video \
  algorithm=wan_t2v_ray_depth_mot_concat_5b \
  algorithm.model.tuned_ckpt_path=${TUNED_CKPT} \
  experiment.training.lr=4e-5 \
  experiment.training.optim.gradient_clip_val=1.0 \
  dataset=mixture_parallax14 \
  dataset.scale_preserving=false \
  dataset.height=288 \
  dataset.width=384 \
  dataset.n_frames=50 \
  algorithm.model.model_type=t2v \
  algorithm.diffusion_forcing.enabled=True \
  algorithm.diffusion_forcing.mode=ray_depth_prediction_firstlast \
  algorithm.diffusion_forcing.train_rgb=true \
  algorithm.diffusion_forcing.mask_clean_token_loss=false \
  'algorithm.diffusion_forcing.subset_settings.tartanair=[1]' \
  'algorithm.diffusion_forcing.subset_settings.spring=[1]' \
  'algorithm.diffusion_forcing.subset_settings.bedlam=[1]' \
  'algorithm.diffusion_forcing.subset_settings.bedlam2=[1]' \
  'algorithm.diffusion_forcing.subset_settings.vkitti2=[1]' \
  'algorithm.diffusion_forcing.subset_settings.mid_air=[1]' \
  'algorithm.diffusion_forcing.subset_settings.mvs_synth=[1]' \
  experiment.num_nodes=16 \
  +experiment.fsdp_shard_within_node=true \
  +experiment.pg_timeout_min=90 \
  experiment.training.batch_size=16 \
  experiment.training.data.num_workers=10 \
  experiment.validation.data.num_workers=2 \
  algorithm.gradient_checkpointing_rate=0.925 \
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
  cluster=seas_h200 \
  cluster.params.memory=1400G \
  +_on_compute_node=True
echo "TRAIN_EXIT=$?"
