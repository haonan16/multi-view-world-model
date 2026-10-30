# SPDX-License-Identifier: Apache-2.0
"""Build the multi-view Wan2.1-T2V-1.3B diffusers checkpoint used by RH20T training.

The transformer's patch embedding is widened from z_dim (16) to z_dim + 6 input channels
for the per-view Plucker raymaps. The new input weights are ZERO, so at step 0 the model
is exactly the pretrained T2V model. ``transformer/config.json`` gets
``_class_name = MultiViewWanTransformer3DModel`` and ``in_channels = 22``; every other
component (VAE, text encoder, tokenizer, scheduler) is symlinked from the HF snapshot.

It also saves null_prompt_embed.pt: the UMT5 embedding of the empty prompt, post-processed
exactly like FastVideo's Wan text stage (trimmed to the prompt length, zero-padded to 512).
RH20T has no captions, so training reads this file instead of loading the ~22 GB fp32 text
encoder on every rank.

The output directory keeps the base name ``Wan2.1-T2V-1.3B-Diffusers`` so FastVideo's
registry resolves it to the stock Wan2.1 T2V 1.3B pipeline config.

Usage:
    python scripts/rh20t/make_multiview_checkpoint.py \
        --out /n/lab_storage/ydu_lab/Lab/haonan/datasets/models/wan_multiview_ray6/Wan2.1-T2V-1.3B-Diffusers
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

BASE_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
CLASS_NAME = "MultiViewWanTransformer3DModel"
PATCH_KEY = "patch_embedding.weight"
RAYMAP_CHANNELS = 6
NULL_EMBED_FILE = "null_prompt_embed.pt"
TEXT_LEN = 512


def widen_patch_embedding(weight: torch.Tensor, extra: int) -> torch.Tensor:
    """[D, C, kt, kh, kw] -> [D, C + extra, kt, kh, kw], new input channels zero."""
    pad = weight.new_zeros(weight.shape[0], extra, *weight.shape[2:])
    return torch.cat([weight, pad], dim=1)


@torch.no_grad()
def save_null_prompt_embed(model_dir: Path, prompt: str = "") -> Path:
    """UMT5 embedding of prompt -> {"embeds": [1, 512, D] fp32, "mask": [1, n]}."""
    from transformers import AutoTokenizer, UMT5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(model_dir / "text_encoder", torch_dtype=torch.float32).eval()
    tokens = tokenizer(prompt,
                       truncation=True,
                       max_length=TEXT_LEN,
                       add_special_tokens=True,
                       return_attention_mask=True,
                       return_tensors="pt")
    hidden = encoder(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask).last_hidden_state
    n = int(tokens.attention_mask.gt(0).sum())
    embeds = torch.cat([hidden[0, :n], hidden.new_zeros(TEXT_LEN - n, hidden.shape[-1])])[None]
    path = model_dir / NULL_EMBED_FILE
    torch.save({"embeds": embeds.float(), "mask": tokens.attention_mask.float(), "prompt": prompt}, path)
    return path


def build(out: Path, base_repo: str, extra: int) -> None:
    src = Path(snapshot_download(base_repo))
    if out.name != Path(base_repo).name:
        raise ValueError(f"output dir must be named {Path(base_repo).name!r} for pipeline-config resolution")
    out.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.name in ("transformer", ".cache"):
            continue
        link = out / entry.name
        if not link.exists():
            os.symlink(entry.resolve(), link)

    t_src, t_out = src / "transformer", out / "transformer"
    t_out.mkdir(exist_ok=True)
    config = json.loads((t_src / "config.json").read_text())
    z_dim = int(config["in_channels"])
    config.update({"_class_name": CLASS_NAME, "in_channels": z_dim + extra, "out_channels": z_dim})
    (t_out / "config.json").write_text(json.dumps(config, indent=2))

    widened = False
    for f in sorted(t_src.iterdir()):
        if f.suffix == ".safetensors":
            tensors = load_file(str(f))
            if PATCH_KEY in tensors:
                w = tensors[PATCH_KEY]
                if w.shape[1] != z_dim:
                    raise ValueError(f"{PATCH_KEY} has {w.shape[1]} input channels, config says {z_dim}")
                tensors[PATCH_KEY] = widen_patch_embedding(w, extra)
                widened = True
            save_file(tensors, str(t_out / f.name), metadata={"format": "pt"})
        elif f.name != "config.json":
            link = t_out / f.name
            if not link.exists():
                os.symlink(f.resolve(), link)
    if not widened:
        raise RuntimeError(f"{PATCH_KEY} not found in {t_src}")
    print(f"wrote {out} ({CLASS_NAME}, in_channels {z_dim} -> {z_dim + extra})")
    print(f"wrote {save_null_prompt_embed(out)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-repo", default=BASE_REPO)
    parser.add_argument("--extra-channels", type=int, default=RAYMAP_CHANNELS)
    args = parser.parse_args()
    build(args.out, args.base_repo, args.extra_channels)


if __name__ == "__main__":
    main()
