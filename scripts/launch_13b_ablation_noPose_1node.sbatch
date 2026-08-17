#!/bin/bash
#SBATCH -J s13B_ablNoPose
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

# === R1 ABLATION: M_noPose (no camera/pose modality at all) ===
# Camera/pose is treated as ABSENT for the whole run via REBUTTAL_DROP_RAYMAP=1: has_raymap
# forced False, so every ray token is pure noise (t=1, zero info) AND both ray losses are
# masked -> pose never leaks into depth/RGB and is never predicted. We also drop the
# camera-conditioned modes 2,3 (their ray-history would just be noise) -> tartanair = [0,1,4].
# Depth is still predicted (modes 0,1). Comparing depth(M_full) vs depth(M_noPose) tests
# H2: pose -> depth.
# NOTE re10k = [0] ONLY: re10k has no depth and (pose now dropped) its modes 1/2/3 would
# supervise NOTHING -> wasted compute. Mode 0 (first RGB -> rest RGB) keeps re10k as pure
# RGB-gen data. This model CANNOT do NVS/pose eval (no camera modality at all).
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
export REBUTTAL_DROP_RAYMAP=1   # camera/pose treated as ABSENT: ray tokens forced to pure
                                # noise (t=1) AND loss masked -> no pose info leaks into depth/RGB.

RUN_DIR=/n/netscratch/ydu_lab/Lab/akiruga/vwm_train_runs/s13B_ablNoPose_4g_${SLURM_JOB_ID}
RESUME_ARG=""
if [ -f "$RUN_DIR/checkpoints/last.ckpt" ]; then
  RESUME_ARG="algorithm.model.tuned_ckpt_path=$RUN_DIR/checkpoints/last.ckpt"
  echo "REQUEUE: warm-starting from $RUN_DIR/checkpoints/last.ckpt"
fi

srun python -m main \
  hydra.run.dir=$RUN_DIR \
  +name=rebuttal_ablation_noPose \
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
  'algorithm.diffusion_forcing.subset_settings.tartanair=[0,1,4]' \
  'algorithm.diffusion_forcing.subset_settings.re10k=[0]' \
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
