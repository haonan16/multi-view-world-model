#!/bin/bash
#SBATCH -J rh20t_prep_ckpt
#SBATCH -o /n/lab_storage/ydu_lab/Lab/haonan/projects/multi-view-world-model/logs/slurm/%x_%j.out
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
# Init checkpoints for the unseen-view runs, both from the RGB tokens baseline:
#   rh20t_mv_tokens_10k       exported DCP checkpoint-10000 (22 inputs)  -> rh20t_unseen_rgb.yaml
#   rh20t_mv_tokens_10k_pc17  same + 17 zero point-cloud inputs (39)     -> rh20t_unseen_pc.yaml
set -euo pipefail
REPO=/n/lab_storage/ydu_lab/Lab/haonan/projects/multi-view-world-model
MODELS=/n/lab_storage/ydu_lab/Lab/haonan/datasets/models
CKPT=${CKPT:-$REPO/outputs/rh20t_mv_tokens/checkpoint-10000}
NAME=Wan2.1-T2V-1.3B-Diffusers
cd "$REPO"
source /n/lab_storage/ydu_lab/Lab/haonan/venvs/mvwm/bin/activate
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
RGB="$MODELS/rh20t_mv_tokens_10k/$NAME"
PC="$MODELS/rh20t_mv_tokens_10k_pc17/$NAME"
if [ ! -e "$RGB/transformer/config.json" ]; then
  mkdir -p "$(dirname "$RGB")"
  python -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint "$CKPT" --output-dir "$RGB" --verify
fi
ls -la "$RGB" "$RGB/transformer"
python scripts/rh20t/widen_checkpoint.py --src "$RGB" --out "$PC" --extra-channels 17
python - "$RGB" "$PC" <<'EOF'
import json, sys
from pathlib import Path
from safetensors import safe_open
for d in sys.argv[1:]:
    cfg = json.loads((Path(d) / "transformer" / "config.json").read_text())
    for f in sorted((Path(d) / "transformer").glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            if "patch_embedding.weight" in h.keys():
                w = h.get_tensor("patch_embedding.weight")
                print(d, cfg["_class_name"], "in", cfg["in_channels"], "patch", tuple(w.shape),
                      "new-input |w|", float(w[:, 22:].abs().sum()) if w.shape[1] > 22 else None)
    assert (Path(d) / "null_prompt_embed.pt").exists(), d
EOF
