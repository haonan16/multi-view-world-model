# SPDX-License-Identifier: Apache-2.0
"""Append zero-initialised input channels to a multi-view Wan diffusers checkpoint.

Used to warm-start the point-cloud model (in_channels 22 -> 39: + VAE latent of the
rendered cloud (16) + coverage mask (1)) from a trained RGB baseline exported with
``python -m fastvideo.train.entrypoint.dcp_to_diffusers``. The new weights are zero,
so the widened model reproduces the source model exactly at step 0. Every component
other than the transformer is symlinked; null_prompt_embed.pt is copied over.

    python scripts/rh20t/widen_checkpoint.py --src <exported>/Wan2.1-T2V-1.3B-Diffusers \
        --out <dir>/Wan2.1-T2V-1.3B-Diffusers --extra-channels 17
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

from make_multiview_checkpoint import NULL_EMBED_FILE, PATCH_KEY, widen_patch_embedding


def widen(src: Path, out: Path, extra: int, null_embed: Path | None) -> None:
    if out.name != src.name:
        raise ValueError(f"output dir must keep the name {src.name!r} for pipeline-config resolution")
    if extra < 1:
        raise ValueError(f"extra channels must be >= 1, got {extra}")
    out.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.name in ("transformer", NULL_EMBED_FILE):
            continue
        link = out / entry.name
        if not link.exists():
            os.symlink(entry.resolve(), link)
    embed = null_embed or src / NULL_EMBED_FILE
    if not embed.exists():
        raise FileNotFoundError(f"{embed} missing; pass --null-embed from the base multi-view checkpoint")
    shutil.copy2(embed, out / NULL_EMBED_FILE)

    t_src, t_out = src / "transformer", out / "transformer"
    t_out.mkdir(exist_ok=True)
    config = json.loads((t_src / "config.json").read_text())
    c_in = int(config["in_channels"])
    config["in_channels"] = c_in + extra
    (t_out / "config.json").write_text(json.dumps(config, indent=2))
    widened = False
    for f in sorted(t_src.iterdir()):
        if f.suffix == ".safetensors":
            tensors = load_file(str(f))
            if PATCH_KEY in tensors:
                if tensors[PATCH_KEY].shape[1] != c_in:
                    raise ValueError(f"{PATCH_KEY} has {tensors[PATCH_KEY].shape[1]} inputs, config says {c_in}")
                tensors[PATCH_KEY] = widen_patch_embedding(tensors[PATCH_KEY], extra)
                widened = True
            save_file(tensors, str(t_out / f.name), metadata={"format": "pt"})
        elif f.name != "config.json":
            link = t_out / f.name
            if not link.exists():
                os.symlink(f.resolve(), link)
    if not widened:
        raise RuntimeError(f"{PATCH_KEY} not found in {t_src}")
    print(f"wrote {out} (in_channels {c_in} -> {c_in + extra})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra-channels", type=int, default=17)
    ap.add_argument("--null-embed", type=Path, default=None)
    args = ap.parse_args()
    widen(args.src, args.out, args.extra_channels, args.null_embed)


if __name__ == "__main__":
    main()
