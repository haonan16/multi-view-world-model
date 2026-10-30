# SPDX-License-Identifier: Apache-2.0
"""Matched evaluation of unseen-view checkpoints on a fixed set of RH20T val clips.

Clip i is val index i: the dataset makes the episode, time window, camera order and
context views a function of i only, and the sampling noise is seeded from (seed, i), so
every checkpoint sees identical inputs and noise. Writes per-clip, per-view, per-frame
image metrics (scripts/rh20t/unseen_metrics.py: PSNR, SSIM, PSNR on covered / uncovered /
moving pixels, LPIPS) of the prediction, the copy-frame-0 baseline (raw pixels, and after the
VAE round trip, which has the same reconstruction ceiling as the model) and the static render,
plus cross-view reprojection PSNR of the prediction and of the ground truth, so arms can be
compared clip by clip (scripts/rh20t/summarize_unseen_eval.py).
Clips are always loaded with the point-cloud render and every frame's depth, also for
RGB-only models (their ``pack`` ignores both), so both arms see the same clips.

With ``--num-samples K > 1`` each clip is sampled K times with different noise; ``pred``
is the first sample (comparable across runs) and ``pred_mean`` scores the average of the K
samples, which separates sample-to-sample randomness from systematic error.

    python scripts/rh20t/eval_unseen_matched.py --checkpoint <checkpoint-N> --out <shard.npz> \
        [--num-clips 254 --shard 0 --num-shards 4 --batch-size 4 --steps 30 --seed 1000 --num-samples 1 \
         --history-guidance 1.0]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unseen_metrics import FIELDS, cross_view_psnr, image_metrics  # noqa: E402

from fastvideo.dataset.rh20t_multiview import RH20TMultiViewDataset  # noqa: E402
from fastvideo.train.entrypoint.dcp_to_diffusers import _ensure_distributed, _run_config_from_raw  # noqa: E402
from fastvideo.train.methods.fine_tuning.multiview_flow import MultiViewFlowMethod  # noqa: E402

SAMPLE_SEED_STRIDE = 7919  # seed offset between the K samples of one clip


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


def vae_copy0(method: MultiViewFlowMethod, batch: dict, gt: torch.Tensor) -> torch.Tensor:
    """Frame 0 after the VAE round trip, repeated over time -> [B, V, T, 3, H, W]. The causal
    VAE codes frame 0 alone in latent frame 0, so this is the model's best static output."""
    latents, _ = method.student.pack(batch)
    first = method._to_full(method.student.decode_views(latents[:, :, :, :1]).float(), *gt.shape[-2:])
    return first[:, :, :1].expand_as(gt)


def sample_views(method: MultiViewFlowMethod, batch: dict, gt: torch.Tensor, steps: int, generator: torch.Generator,
                 history_guidance: float) -> torch.Tensor:
    """One sample of every view -> [B, V, T, 3, H, W] at the ground-truth resolution."""
    student = method.student
    latents, cond = student.pack(batch)
    ctx = student.context_mask(batch, latents.shape[1])
    sampled = student.sample(latents,
                             cond,
                             n_hist=student.n_hist_latent,
                             steps=steps,
                             generator=generator,
                             context=ctx,
                             history_guidance=history_guidance)
    return method._to_full(student.decode_views(sampled).float(), *gt.shape[-2:])


@torch.no_grad()
def evaluate(method: MultiViewFlowMethod, indices: list[int], batch_size: int, steps: int, seed: int,
             num_samples: int, history_guidance: float) -> dict[str, np.ndarray]:
    import lpips

    student = method.student
    student.ensure_null_context()
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(student.device).eval()
    cfg = dataclasses.replace(student.dataset_config, point_cloud=True, return_depth=True)
    loader = DataLoader(Subset(RH20TMultiViewDataset(cfg, split="val"), indices),
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4)
    keys = ("context", "gt_mean", "pred", "pred_mean", "copy0", "copy0_vae", "render", "xview_pred", "xview_gt")
    out: dict[str, list] = {k: [] for k in ("index", "episode", *keys)}
    generator = torch.Generator(device=student.device)
    done = 0
    for batch in loader:
        batch_idx = indices[done:done + len(batch["videos"])]
        done += len(batch_idx)
        gt = batch["videos"].to(student.device).float()
        samples = []
        for k in range(num_samples):
            generator.manual_seed(seed * 100003 + batch_idx[0] + k * SAMPLE_SEED_STRIDE)
            samples.append(sample_views(method, batch, gt, steps, generator, history_guidance))
        pred = samples[0]
        cover = batch["pc_mask"].to(student.device)  # B, V, 1, H, W
        render = batch["pc_render"].to(student.device).float()[:, :, None].expand_as(gt)
        out["pred"].append(image_metrics(pred, gt, cover, lpips_fn))
        if num_samples > 1:
            out["pred_mean"].append(image_metrics(torch.stack(samples).mean(0), gt, cover, lpips_fn))
        out["copy0"].append(image_metrics(gt[:, :, :1].expand_as(gt), gt, cover, lpips_fn))
        out["copy0_vae"].append(image_metrics(vae_copy0(method, batch, gt), gt, cover, lpips_fn))
        out["render"].append(image_metrics(render, gt, cover, lpips_fn))
        geometry = [batch[k].to(student.device) for k in ("depth", "intrinsics", "c2w")]
        out["xview_pred"].append(cross_view_psnr(pred, *geometry))
        out["xview_gt"].append(cross_view_psnr(gt, *geometry))
        out["gt_mean"].append(gt.mean(dim=(2, 3, 4, 5)).cpu().numpy())  # identical inputs across arms
        out["index"].extend(batch_idx)
        out["episode"].extend(batch["episode"])
        out["context"].append(student.context_mask(batch, gt.shape[1]).cpu().numpy())
        print(f"[eval] {done}/{len(indices)} clips", flush=True)
    arrays = {k: np.concatenate(out[k]) for k in keys if out[k]}
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
    ap.add_argument("--num-samples", type=int, default=1, help="samples per clip; >1 also scores their mean")
    ap.add_argument("--history-guidance", type=float, default=1.0, help="CFG weight on the history (1 = off)")
    args = ap.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError(f"--shard must be in [0, {args.num_shards}), got {args.shard}")
    if args.num_samples < 1 or args.steps < 1:
        raise ValueError("--num-samples and --steps must be >= 1")
    if not (args.checkpoint / "dcp").is_dir():
        raise FileNotFoundError(f"no dcp/ under {args.checkpoint}")

    t0 = time.time()
    method = load_method(args.checkpoint)
    indices = list(range(args.shard, args.num_clips, args.num_shards))
    result = evaluate(method, indices, args.batch_size, args.steps, args.seed, args.num_samples, args.history_guidance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    step = json.loads((args.checkpoint / "metadata.json").read_text())["step"]
    np.savez(args.out,
             **result,
             fields=np.asarray(FIELDS),
             step=step,
             steps=args.steps,
             seed=args.seed,
             num_samples=args.num_samples,
             history_guidance=args.history_guidance,
             checkpoint=str(args.checkpoint))
    print(f"[eval] wrote {args.out} ({len(indices)} clips, step {step}) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
