#!/bin/bash
#SBATCH -J s13B_ablNoDepth
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

# === R1 ABLATION: M_noDepth (no depth modality at all) ===
# Depth is treated as ABSENT for the whole run via REBUTTAL_DROP_DEPTH=1: the has_depth flag
# is forced False, so the missing-modality path forces every depth token to pure noise (t=1,
# zero info) AND masks the depth loss. Depth therefore never leaks into pose/RGB as input and
# is never predicted. We also drop setting 4 (its depth-history would just be noise anyway)
# -> settings [0,1,2,3]. Camera stays (modes 2,3 do NVS; mode 1 predicts pose).
# Comparing pose(M_full) vs pose(M_noDepth) tests H3: depth -> pose.
# Everything else identical to M_full: 50/50 sampling, F scale, 240x320, bs4, gckpt 1.0.
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
export REBUTTAL_DROP_DEPTH=1   # depth treated as ABSENT: tokens forced to pure noise (t=1)
                               # AND loss masked -> no depth info leaks into pose/RGB.

RUN_DIR=/n/netscratch/ydu_lab/Lab/akiruga/vwm_train_runs/s13B_ablNoDepth_4g_${SLURM_JOB_ID}
RESUME_ARG=""
if [ -f "$RUN_DIR/checkpoints/last.ckpt" ]; then
  RESUME_ARG="algorithm.model.tuned_ckpt_path=$RUN_DIR/checkpoints/last.ckpt"
  echo "REQUEUE: warm-starting from $RUN_DIR/checkpoints/last.ckpt"
fi

srun python -m main \
  hydra.run.dir=$RUN_DIR \
  +name=rebuttal_ablation_noDepth \
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
  'algorithm.diffusion_forcing.subset_settings.tartanair=[0,1,2,3]' \
  'algorithm.diffusion_forcing.subset_settings.re10k=[0,1,2,3]' \
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
