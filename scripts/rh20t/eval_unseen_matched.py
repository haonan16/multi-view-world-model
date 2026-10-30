# SPDX-License-Identifier: Apache-2.0
"""Matched evaluation of unseen-view checkpoints on a fixed set of RH20T val clips.

Clip i is val index i: the dataset makes the episode, time window, camera order and
context views a function of i only, and the sampling noise is seeded from (seed, i), so
every checkpoint sees identical inputs and noise. Writes per-clip, per-view, per-frame
PSNR/SSIM of the prediction, the copy-frame-0 baseline and (point-cloud runs) the static
render, so arms can be compared clip by clip (scripts/rh20t/summarize_unseen_eval.py).
Clips are always loaded with the point-cloud render, also for RGB-only models (their
``pack`` ignores it), so both arms see the same clips and are also scored separately on the
pixels the context cloud covers and the pixels it does not.

    python scripts/rh20t/eval_unseen_matched.py --checkpoint <checkpoint-N> --out <shard.npz> \
        [--num-clips 254 --shard 0 --num-shards 4 --batch-size 4 --steps 30 --seed 1000]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fastvideo.dataset.rh20t_multiview import RH20TMultiViewDataset
from fastvideo.train.entrypoint.dcp_to_diffusers import _ensure_distributed, _run_config_from_raw
from fastvideo.train.methods.fine_tuning.multiview_flow import MultiViewFlowMethod, ssim


def load_method(checkpoint: Path) -> MultiViewFlowMethod:
    """Build the run's method from the checkpoint metadata and load its DCP weights (1 GPU)."""
    import torch.distributed.checkpoint as dcp

    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.utils.builder import build_from_config

    _ensure_distributed()
    meta = json.loads((checkpoint / "metadata.json").read_text())
    cfg = _run_config_from_raw(meta["config"])
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    dist = cfg.training.distributed
    dist.tp_size = dist.sp_size = dist.num_gpus = dist.hsdp_replicate_dim = dist.hsdp_shard_dim = 1
    _, method, _, _ = build_from_config(cfg)
    dcp.load(method.checkpoint_state(), checkpoint_id=str(checkpoint / "dcp"))
    if not isinstance(method, MultiViewFlowMethod):
        raise TypeError(f"{checkpoint} is not a MultiViewFlowMethod run: {type(method).__name__}")
    return method


def per_image(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """[B, V, T, 3, H, W] in [-1, 1]; mask [B, V, 1, H, W] (1 = covered by the context cloud).
    -> [B, V, T, 4]: PSNR, SSIM, PSNR on covered pixels, PSNR on uncovered pixels (NaN if none)."""
    p = ((pred.float() + 1) / 2).clamp(0, 1)
    g = ((gt.float() + 1) / 2).clamp(0, 1)
    se = ((p - g)**2).mean(dim=3)  # B, V, T, H, W
    m = mask.float()  # B, V, 1, H, W -> broadcasts over T

    def psnr(weight: torch.Tensor) -> torch.Tensor:
        n = weight.expand_as(se).sum(dim=(-2, -1))
        mse = (se * weight).sum(dim=(-2, -1)) / n.clamp(min=1)
        return torch.where(n > 0, -10 * torch.log10(mse.clamp(min=1e-10)), torch.nan)

    s = ssim(p.flatten(0, 2), g.flatten(0, 2)).reshape(gt.shape[:3])
    return torch.stack([psnr(torch.ones_like(m)), s, psnr(m), psnr(1 - m)], dim=-1).cpu().numpy()


@torch.no_grad()
def evaluate(method: MultiViewFlowMethod, indices: list[int], batch_size: int, steps: int,
             seed: int) -> dict[str, np.ndarray]:
    student = method.student
    student.ensure_null_context()
    dataset = RH20TMultiViewDataset(dataclasses.replace(student.dataset_config, point_cloud=True), split="val")
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=4)
    out: dict[str, list] = {k: [] for k in ("index", "episode", "context", "gt_mean", "pred", "copy0", "render")}
    generator = torch.Generator(device=student.device)
    done = 0
    for batch in loader:
        batch_idx = indices[done:done + len(batch["videos"])]
        done += len(batch_idx)
        generator.manual_seed(seed * 100003 + batch_idx[0])
        gt = batch["videos"].to(student.device).float()
        latents, cond = student.pack(batch)
        ctx = student.context_mask(batch, latents.shape[1])
        sampled = student.sample(latents, cond, n_hist=student.n_hist_latent, steps=steps, generator=generator,
                                 context=ctx)
        pred = method._to_full(student.decode_views(sampled).float(), *gt.shape[-2:])
        mask = batch["pc_mask"].to(student.device)  # B, V, 1, H, W
        render = batch["pc_render"].to(student.device).float()[:, :, None].expand_as(gt)
        out["pred"].append(per_image(pred, gt, mask))
        out["copy0"].append(per_image(gt[:, :, :1].expand_as(gt), gt, mask))
        out["render"].append(per_image(render, gt, mask))
        out["gt_mean"].append(gt.mean(dim=(2, 3, 4, 5)).cpu().numpy())  # identical inputs across arms
        out["index"].extend(batch_idx)
        out["episode"].extend(batch["episode"])
        out["context"].append(ctx.cpu().numpy())
        print(f"[eval] {done}/{len(indices)} clips", flush=True)
    arrays = {k: np.concatenate(v) for k, v in out.items() if k not in ("index", "episode")}
    return {**arrays, "index": np.asarray(out["index"]), "episode": np.asarray(out["episode"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-clips", type=int, default=254, help="val indices 0..N-1 (254 = every val episode)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError(f"--shard must be in [0, {args.num_shards}), got {args.shard}")
    if not (args.checkpoint / "dcp").is_dir():
        raise FileNotFoundError(f"no dcp/ under {args.checkpoint}")

    t0 = time.time()
    method = load_method(args.checkpoint)
    indices = list(range(args.shard, args.num_clips, args.num_shards))
    result = evaluate(method, indices, args.batch_size, args.steps, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    step = json.loads((args.checkpoint / "metadata.json").read_text())["step"]
    np.savez(args.out, **result, step=step, steps=args.steps, seed=args.seed, checkpoint=str(args.checkpoint))
    print(f"[eval] wrote {args.out} ({len(indices)} clips, step {step}) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
