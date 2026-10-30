# SPDX-License-Identifier: Apache-2.0
"""Multi-view future-prediction finetuning (RH20T views-as-tokens / views-tiled).

Pairs with ``fastvideo.train.models.wan.multiview.WanMultiViewModel``. Training loss is
the flow-matching MSE averaged over the noised latent frames only (clean history
frames carry no loss). Validation conditions on frame 0 of every view, samples the
rest, and reports PSNR / SSIM on the future frames at full per-view resolution
(the tiled design is upsampled from its tiles), alongside two references: the VAE
round trip through the design's packing (its reconstruction ceiling) and copying
frame 0 forward.

Unseen-view runs (``dataset.n_context_views`` < n_views) also report the context and
target views separately (``*_ctx`` frames 1.., ``*_tgt`` every frame, since a target view
has no observation) and, with point clouds, the rendered cloud held fixed over time
(``*_render_tgt``) as the geometry-only reference for target views.

Method config keys (``method:`` in YAML):
    validation_every_steps   0 disables validation (default 1000)
    validation_batches       per-rank batches per validation (default 2)
    validation_batch_size    default 2
    validation_sampling_steps  Euler steps (default 30)
    validation_num_workers   default 2
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from fastvideo.dataset.rh20t_multiview import RH20TMultiViewDataset
from fastvideo.logger import init_logger
from fastvideo.train.methods.base import LogScalar
from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.config import get_optional_int

logger = init_logger(__name__)

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5


def masked_flow_loss(pred: torch.Tensor, latents: torch.Tensor, noise: torch.Tensor,
                     loss_mask: torch.Tensor) -> torch.Tensor:
    """MSE between predicted flow and (noise - x0), averaged over frames with loss_mask = 1.

    pred / latents / noise: [B, V, C, F, h, w]; loss_mask: [B, F] (shared by views) or [B, V, F]."""
    se = ((pred.float() - (noise - latents).float())**2).mean(dim=(2, 4, 5))  # B, V, F
    if loss_mask.dim() == 2:
        loss_mask = loss_mask[:, None].expand_as(se)
    return (se * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gaussian-window SSIM per image. x, y: [N, 3, H, W] in [0, 1] -> [N]."""
    coords = torch.arange(SSIM_WINDOW, device=x.device, dtype=torch.float32) - SSIM_WINDOW // 2
    g = torch.exp(-coords**2 / (2 * SSIM_SIGMA**2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).expand(3, 1, SSIM_WINDOW, SSIM_WINDOW).to(x.dtype)

    def blur(z: torch.Tensor) -> torch.Tensor:
        return F.conv2d(z, window, padding=SSIM_WINDOW // 2, groups=3)

    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx**2, blur(y * y) - my**2, blur(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return s.mean(dim=(1, 2, 3))


def future_frame_metrics(pred: torch.Tensor,
                         gt: torch.Tensor,
                         frames: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of per-image PSNR and SSIM of [B, V, T, 3, H, W] videos in [-1, 1] over the
    (b, v, t) images selected by ``frames`` ([B, V, T] bool; default frames 1.. of every view)."""
    if frames is None:
        frames = torch.ones(gt.shape[:3], dtype=torch.bool, device=gt.device)
        frames[:, :, 0] = False
    p = ((pred[frames].float() + 1) / 2).clamp(0, 1)
    g = ((gt[frames].float() + 1) / 2).clamp(0, 1)
    if not len(g):
        return g.new_zeros(()), g.new_zeros(())
    mse = ((p - g)**2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
    return (-10 * torch.log10(mse)).sum(), ssim(p, g).sum()


class MultiViewFlowMethod(FineTuneMethod):
    """Flow-matching finetune with clean-history frames and multi-view validation."""

    def __init__(self, *, cfg: Any, role_models: dict[str, ModelBase]) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        mcfg = self.method_config
        where = "method"

        def opt_int(key: str, default: int, minimum: int) -> int:
            value = get_optional_int(mcfg, key, where=f"{where}.{key}")
            value = default if value is None else value
            if value < minimum:
                raise ValueError(f"{where}.{key} must be >= {minimum}, got {value}")
            return value

        self.val_every = opt_int("validation_every_steps", 1000, 0)  # 0 disables validation
        self.val_batches = opt_int("validation_batches", 2, 1)
        self.val_batch_size = opt_int("validation_batch_size", 2, 1)
        self.val_steps = opt_int("validation_sampling_steps", 30, 1)
        self.val_workers = opt_int("validation_num_workers", 2, 0)
        self._val_loader: DataLoader | None = None

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del iteration
        tb = self.student.prepare_batch(batch, generator=self.cuda_generator, latents_source="data")
        pred = self.student.predict_noise(tb.noisy_model_input, tb.timesteps, tb, conditional=True)
        loss = masked_flow_loss(pred, tb.latents, tb.noise, tb.loss_mask)
        outputs: dict[str, Any] = {"_fv_backward": (tb.timesteps, None)}
        metrics: dict[str, LogScalar] = {
            "train/noised_frame_fraction": float(tb.loss_mask.mean()),
            "train/tokens_per_sample": float(np.prod(tb.latents.shape[1:2]) * np.prod(tb.latents.shape[3:]) / 4),
        }
        return {"total_loss": loss, "finetune_loss": loss}, outputs, metrics

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validation_loader(self) -> DataLoader:
        if self._val_loader is None:
            student = self.student
            dataset = RH20TMultiViewDataset(student.dataset_config, split="val")
            world = student.world_group
            per_rank = self.val_batches * self.val_batch_size
            start = int(world.rank) * per_rank
            indices = [(start + i) % len(dataset) for i in range(per_rank)]
            self._val_loader = DataLoader(Subset(dataset, indices),
                                          batch_size=self.val_batch_size,
                                          shuffle=False,
                                          num_workers=self.val_workers,
                                          drop_last=False)
        return self._val_loader

    @torch.no_grad()
    def on_validation_begin(self, iteration: int = 0) -> dict[str, LogScalar]:
        if self.val_every <= 0 or iteration <= 0 or iteration % self.val_every:
            return {}
        student = self.student
        student.ensure_null_context()
        generator = torch.Generator(device=student.device).manual_seed(int(self.training_config.data.seed))
        names = ["pred", "vae", "copy0", "pred_ctx", "pred_tgt", "vae_tgt", "render_tgt"]
        totals = torch.zeros(len(names), 3, device=student.device, dtype=torch.float64)  # (n, psnr, ssim)
        first_video = None
        for batch in self._validation_loader():
            gt = batch["videos"].to(student.device).float()  # B, V, T, 3, H, W
            H, W = gt.shape[-2:]
            latents, cond = student.pack(batch)
            ctx = student.context_mask(batch, latents.shape[1])
            pred = self._to_full(
                student.decode_views(
                    student.sample(latents,
                                   cond,
                                   n_hist=student.n_hist_latent,
                                   steps=self.val_steps,
                                   generator=generator,
                                   context=ctx)).float(), H, W)
            vae = self._to_full(student.decode_views(latents).float(), H, W)
            copy0 = gt[:, :, :1].expand_as(gt)
            ctx_v = ctx if ctx.shape[1] == gt.shape[1] else ctx[:, :1].expand(-1, gt.shape[1])
            future = torch.ones(gt.shape[:3], dtype=torch.bool, device=gt.device)
            future[:, :, :1 + 4 * (student.n_hist_latent - 1)] = False  # pixel frames of the history latents
            ctx_f = future & ctx_v[:, :, None]
            tgt_f = (~ctx_v)[:, :, None].expand(gt.shape[:3])
            evaluated = ctx_f | tgt_f
            refs = {
                "pred": (pred, evaluated),
                "vae": (vae, evaluated),
                "copy0": (copy0, ctx_f),
                "pred_ctx": (pred, ctx_f),
                "pred_tgt": (pred, tgt_f),
                "vae_tgt": (vae, tgt_f),
            }
            if "pc_render" in batch:
                render = batch["pc_render"].to(student.device).float()[:, :, None].expand_as(gt)
                refs["render_tgt"] = (render, tgt_f)
            for k, name in enumerate(names):
                if name in refs:
                    ref, frames = refs[name]
                    psnr_sum, ssim_sum = future_frame_metrics(ref, gt, frames)
                    totals[k] += torch.stack([frames.sum().double(), psnr_sum.double(), ssim_sum.double()])
            if first_video is None:
                extra = batch["pc_render"][0] if "pc_render" in batch else None
                first_video = (gt[0].cpu(), pred[0].cpu(), batch["episode"][0], ctx_v[0].cpu(), extra)
        totals = student.world_group.all_reduce(totals)
        metrics: dict[str, LogScalar] = {}
        for k, name in enumerate(names):
            if totals[k, 0] > 0:
                metrics[f"val/psnr_{name}"] = float(totals[k, 1] / totals[k, 0])
                metrics[f"val/ssim_{name}"] = float(totals[k, 2] / totals[k, 0])
        if first_video is not None and int(student.world_group.rank) == 0:
            self._log_video(*first_video, iteration)
        return metrics

    @staticmethod
    def _to_full(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """[B, V, T, 3, h, w] -> [B, V, T, 3, H, W] (bilinear; identity at full size)."""
        if x.shape[-2:] == (H, W):
            return x
        y = F.interpolate(x.flatten(0, 2), size=(H, W), mode="bilinear", align_corners=False)
        return y.unflatten(0, x.shape[:3])

    def _log_video(self,
                   gt: torch.Tensor,
                   pred: torch.Tensor,
                   episode: str,
                   context: torch.Tensor,
                   render: torch.Tensor | None,
                   iteration: int) -> None:
        """Views side by side: ground truth, prediction, then (point cloud) the rendered cloud.
        gt/pred: [V, T, 3, H, W]; render: [V, 3, H, W]. Target (unobserved) views get a red
        top border."""
        if self.tracker is None:
            return
        gt = gt.clone()
        gt[~context, :, :, :3] = torch.tensor([1.0, -1.0, -1.0])[:, None, None]
        rows = [torch.cat(list(gt), dim=-1), torch.cat(list(pred), dim=-1)]
        if render is not None:
            rows.append(torch.cat(list(render[:, None].expand_as(gt).float()), dim=-1))
        grid = torch.cat(rows, dim=-2)  # T, 3, rows*H, VW
        frames = ((grid.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
        fps = max(1, int(round(self.student.dataset_config.fps)))
        video = self.tracker.video(frames, caption=f"{self.student.design} | {episode}", fps=fps)
        if video is not None:
            self.tracker.log_artifacts({"val/gt_top_pred_bottom": video}, iteration)  # + render row (pc)
