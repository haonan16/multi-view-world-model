#!/bin/bash
#SBATCH -J rh20t_prep_actions
#SBATCH -o /n/lab_storage/ydu_lab/Lab/haonan/projects/multi-view-world-model/logs/slurm/%x_%j.out
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
# Init checkpoint for rh20t_unseen_pc_actions.yaml: export a point-cloud DCP checkpoint and
# append ACTION_CHANNELS (16) zero input channels after the 6 raymap + 17 point-cloud ones.
#   sbatch scripts/rh20t/prepare_actions_checkpoint.sbatch [<dcp checkpoint dir>]
set -euo pipefail
REPO=/n/lab_storage/ydu_lab/Lab/haonan/projects/multi-view-world-model
MODELS=/n/lab_storage/ydu_lab/Lab/haonan/datasets/models
CKPT=${1:-$REPO/outputs/eval_unseen_matched/ckpts/pc_78500}
NAME=Wan2.1-T2V-1.3B-Diffusers
PC="$MODELS/rh20t_pc_78500/$NAME"
ACT="$MODELS/rh20t_pc_78500_act16/$NAME"
cd "$REPO"
source /n/lab_storage/ydu_lab/Lab/haonan/venvs/mvwm/bin/activate
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 WANDB_MODE=disabled
EXTRA=$(python -c "from fastvideo.train.models.wan.multiview import ACTION_CHANNELS; print(ACTION_CHANNELS)")
if [ ! -e "$PC/transformer/config.json" ]; then
  mkdir -p "$(dirname "$PC")"
  python -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint "$CKPT" --output-dir "$PC" --verify
fi
python scripts/rh20t/widen_checkpoint.py --src "$PC" --out "$ACT" --extra-channels "$EXTRA"
python - "$PC" "$ACT" <<'PY'
import json, sys
from pathlib import Path
from safetensors import safe_open
for d in sys.argv[1:]:
    cfg = json.loads((Path(d) / "transformer" / "config.json").read_text())
    for f in sorted((Path(d) / "transformer").glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            if "patch_embedding.weight" in h.keys():
                w = h.get_tensor("patch_embedding.weight")
                print(d, "in", cfg["in_channels"], "patch", tuple(w.shape),
                      "|w| of last 16 inputs", float(w[:, -16:].abs().sum()))
    assert (Path(d) / "null_prompt_embed.pt").exists(), d
PY
