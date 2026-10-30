# SPDX-License-Identifier: Apache-2.0
"""Multi-view Wan model plugin for RH20T (views-as-tokens vs views-tiled).

Ported from akiruga's ego_world_model ``algorithms/wan/wan_multiview.py``. Both designs
consume the same RH20T sample (``fastvideo/dataset/rh20t_multiview.py``: V views x T
frames + per-view Plucker raymaps) with the same backbone; only the view packing differs:

  tokens  each view is VAE-encoded on its own at (H, W); the V views' tokens share one
          sequence (same RoPE per view, camera from the raymap channels).
  tiled   the V views are downscaled into ONE (H, W) canvas, VAE-encoded as one video,
          raymaps tiled the same way -> V x fewer tokens.

Task: future prediction. The first ``n_hist_latent`` latent frames (latent frame 0 =
pixel frame 0 of every view) are clean history with probability ``1 - p_no_hist``; the
remaining frames are noised at one shared sigma per sample. The method computes the
flow-matching MSE on the noised frames only (``TrainingBatch.loss_mask``).

Unseen views (``dataset.n_context_views``, tokens design only): only the context views
get a clean history frame; the other views are generated from frame 0 on. With
``point_cloud: true`` every view also receives the VAE latent of the context views'
frame-0 point cloud rendered into it (+ a coverage mask) in latent frame 0, as 17 extra
input channels (zero-initialised; ``scripts/rh20t/widen_checkpoint.py``).

Video is decoded and VAE-encoded online (no parquet preprocessing); there is no text,
every sample uses the UMT5 embedding of the empty prompt.
"""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import os

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.dataset.rh20t_multiview import (RH20TMultiViewConfig, RH20TMultiViewDataset, tile_views, untile_views)
from fastvideo.distributed import get_sp_group, get_world_group
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.training_utils import get_sigmas, normalize_dit_input

from fastvideo.train.models.wan.wan import WanModel
from fastvideo.train.utils.moduleloader import load_module_from_path
from fastvideo.train.utils.negative_prompt import encode_negative_prompt

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import TrainingConfig

logger = init_logger(__name__)

DESIGNS = ("tokens", "tiled")
RAYMAP_CHANNELS = 6
PC_CHANNELS = 16 + 1  # VAE latent of the rendered point cloud + coverage mask
NULL_EMBED_FILE = "null_prompt_embed.pt"  # written by scripts/rh20t/make_multiview_checkpoint.py


class InfiniteRandomSampler(Sampler[int]):
    """Fresh random permutation on every ``iter()``; per-rank seed so DP ranks differ.

    Stateful (epoch + position) so a requeued run continues the stream instead of
    replaying epoch 0 (StatefulDataLoader checkpoints it via the trainer)."""

    def __init__(self, n: int, seed: int) -> None:
        self.n = int(n)
        self.seed = int(seed)
        self.epoch = 0
        self._resume_offset = 0
        self._yielded = 0

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch)
        order = torch.randperm(self.n, generator=g).tolist()
        start, self._resume_offset = self._resume_offset, 0
        self.epoch += 1
        self._yielded = start
        for i in order[start:]:
            self._yielded += 1
            yield i

    def state_dict(self) -> dict[str, int]:
        # epoch was already advanced when the current pass started
        return {"epoch": max(self.epoch - 1, 0), "offset": self._yielded}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state["epoch"])
        self._resume_offset = int(state["offset"]) % max(self.n, 1)


class WanMultiViewModel(WanModel):
    """Wan2.1 student over V camera views of RH20T; owns the RH20T dataloader."""

    _transformer_cls_name: str = "MultiViewWanTransformer3DModel"

    def __init__(
        self,
        *,
        dataset: dict[str, Any],
        design: Literal["tokens", "tiled"] = "tokens",
        n_hist_latent: int = 1,
        p_no_hist: float = 0.1,
        vae_encode_chunk: int = 8,
        null_prompt_embed_path: str | None = None,
        point_cloud: bool = False,
        **kwargs: Any,
    ) -> None:
        if design not in DESIGNS:
            raise ValueError(f"design must be one of {DESIGNS}, got {design!r}")
        if design == "tiled" and (dataset.get("n_context_views") or point_cloud):
            raise ValueError("unseen views / point_cloud need per-view timesteps: use design=tokens")
        if bool(point_cloud) != bool(dataset.get("point_cloud", False)):
            raise ValueError("model point_cloud and dataset.point_cloud must agree")
        if not 0.0 <= float(p_no_hist) <= 1.0:
            raise ValueError(f"p_no_hist must be in [0, 1], got {p_no_hist}")
        super().__init__(**kwargs)
        self.design = design
        self.dataset_config = RH20TMultiViewConfig(**{**dataset, "cfgs": tuple(dataset.get("cfgs") or range(1, 8))})
        self.n_views = self.dataset_config.n_views
        self.n_hist_latent = int(n_hist_latent)
        self.p_no_hist = float(p_no_hist)
        self.vae_encode_chunk = int(vae_encode_chunk)
        self.point_cloud = bool(point_cloud)
        self.null_prompt_embed_path = null_prompt_embed_path or os.path.join(self._init_from, NULL_EMBED_FILE)
        # no captions and no CFG: the null context replaces the negative prompt
        self.set_requires_negative_conditioning(False)
        self.null_context: torch.Tensor | None = None
        self.null_context_mask: torch.Tensor | None = None
        expected_in = self.transformer.in_channels if hasattr(self.transformer, "in_channels") else None
        extra = RAYMAP_CHANNELS + (PC_CHANNELS if self.point_cloud else 0)
        if expected_in is not None and expected_in != self.transformer.out_channels + extra:
            raise ValueError(f"transformer in_channels={expected_in}, expected out_channels + {extra}; build the "
                             "checkpoint with scripts/rh20t/make_multiview_checkpoint.py (+ widen_checkpoint.py)")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        if int(training_config.distributed.sp_size or 1) != 1:
            raise ValueError("WanMultiViewModel feeds every rank its own batch; set training.distributed.sp_size=1")
        # Wan T2V configs ship the VAE decoder only; online encoding needs the encoder.
        training_config.pipeline_config.vae_config.load_encoder = True  # type: ignore[union-attr]
        training_config.pipeline_config.vae_config.load_decoder = True  # type: ignore[union-attr]
        self.vae = load_module_from_path(
            model_path=str(training_config.model_path),
            module_type="vae",
            training_config=training_config,
        ).eval().requires_grad_(False)

        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self._init_timestep_mechanics()
        scheduler_shift = float(self.noise_scheduler.config.shift)
        if abs(scheduler_shift - self.timestep_shift) > 1e-6:
            raise ValueError(f"model flow_shift ({scheduler_shift}) must equal pipeline.flow_shift "
                             f"({self.timestep_shift}): training noise and sampling would disagree")

        data = training_config.data
        dataset = RH20TMultiViewDataset(self.dataset_config, split="train")
        self.dataloader = StatefulDataLoader(
            dataset,
            batch_size=int(data.train_batch_size),
            sampler=InfiniteRandomSampler(len(dataset), seed=int(data.seed) + int(self.world_group.rank)),
            num_workers=int(data.dataloader_num_workers),
            pin_memory=True,
            drop_last=True,
            persistent_workers=int(data.dataloader_num_workers) > 0,
        )
        self.start_step = 0

    def on_train_start(self) -> None:
        self.ensure_null_context()

    def ensure_null_context(self) -> None:
        """Empty-prompt UMT5 context: precomputed file if present (no 22 GB encoder on
        every rank), else encoded on the fly."""
        if self.null_context is not None:
            return
        dtype = self._get_training_dtype()
        if os.path.exists(self.null_prompt_embed_path):
            saved = torch.load(self.null_prompt_embed_path, map_location="cpu", weights_only=True)
            embeds, mask = saved["embeds"], saved["mask"]
        else:
            logger.warning("%s not found; encoding the empty prompt with the text encoder", self.null_prompt_embed_path)
            embeds, mask = encode_negative_prompt(self.training_config, prompt="", device=self.device, dtype=dtype)
        self.null_context = embeds.to(self.device, dtype)
        self.null_context_mask = mask.to(self.device, dtype)

    # ------------------------------------------------------------------
    # Data -> latents
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_videos(self, videos: torch.Tensor) -> torch.Tensor:
        """[N, T, 3, H, W] in [-1, 1] -> normalized latents [N, C, F, h, w]."""
        vae_dtype = next(self.vae.parameters()).dtype
        out = []
        for chunk in videos.split(self.vae_encode_chunk):
            x = chunk.permute(0, 2, 1, 3, 4).to(self.device, vae_dtype)
            out.append(self.vae.encode(x).mode())
        latents = torch.cat(out)
        return normalize_dit_input("wan", latents.float(), self.vae).to(self._get_training_dtype())

    @torch.no_grad()
    def decode_views(self, latents: torch.Tensor) -> torch.Tensor:
        """latents [B, V', C, F, h, w] -> pixels [B, V, T, 3, H_v, W_v] in [-1, 1]."""
        B, Vp = latents.shape[:2]
        frames = [self.decode_latents(z[None].permute(0, 2, 1, 3, 4)) for z in latents.flatten(0, 1)]
        x = torch.cat(frames).permute(0, 2, 1, 3, 4).unflatten(0, (B, Vp)) * 2 - 1  # B, V', T, 3, H, W
        if self.design == "tiled":
            x = untile_views(x[:, 0], self.n_views)
        return x

    @torch.no_grad()
    def pack(self, raw_batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """-> latents [B, V', C, F, h, w], cond [B, V', 6 (+17), F, h, w] (V' = V tokens / 1 tiled).

        cond = raymaps on every frame, then (point_cloud) the rendered-cloud latent and its
        coverage mask on latent frame 0 only (zeros elsewhere)."""
        videos = raw_batch["videos"]  # B, V, T, 3, H, W
        rays = raw_batch["raymaps"]  # B, V, 6, Hr, Wr
        if videos.shape[1] != self.n_views:
            raise ValueError(f"batch has {videos.shape[1]} views, model expects {self.n_views}")
        B = videos.shape[0]
        if self.design == "tiled":
            videos = tile_views(videos)[:, None]
            rays = tile_views(rays[:, :, None])[:, None, 0]
        Vp = videos.shape[1]
        latents = self.encode_videos(videos.flatten(0, 1)).unflatten(0, (B, Vp))
        h, w = latents.shape[-2:]
        rays = rays.to(self.device, torch.float32)
        if rays.shape[-2:] != (h, w):
            rays = F.interpolate(rays.flatten(0, 1), size=(h, w), mode="area").unflatten(0, (B, Vp))
        n_lat = latents.shape[3]
        cond = [rays.to(latents.dtype)[:, :, :, None].expand(-1, -1, -1, n_lat, -1, -1)]
        if self.point_cloud:
            render = raw_batch["pc_render"]  # B, V, 3, H, W
            pc = self.encode_videos(render.flatten(0, 1)[:, None]).unflatten(0, (B, Vp))  # B, V, 16, 1, h, w
            mask = F.interpolate(raw_batch["pc_mask"].flatten(0, 1).to(self.device, torch.float32),
                                 size=(h, w),
                                 mode="area").unflatten(0, (B, Vp))[:, :, :, None]
            first = torch.cat([pc, mask.to(pc.dtype)], dim=2)  # B, V, 17, 1, h, w
            cond.append(F.pad(first, (0, 0, 0, 0, 0, n_lat - 1)))
        return latents, torch.cat(cond, dim=2)

    def context_mask(self, raw_batch: dict[str, Any], n_views: int) -> torch.Tensor:
        """[B, V'] bool: views whose history frames are observed (all, unless unseen-view task)."""
        ctx = raw_batch.get("context")
        if ctx is None or self.design == "tiled":
            return torch.ones(len(raw_batch["videos"]), n_views, dtype=torch.bool, device=self.device)
        return ctx.to(self.device, torch.bool)

    # ------------------------------------------------------------------
    # Training primitives
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        if latents_source != "data":
            raise ValueError("WanMultiViewModel only supports latents_source='data'")
        self.ensure_null_context()
        latents, cond = self.pack(raw_batch)
        B, Vp, _, n_lat = latents.shape[:4]
        device = latents.device

        noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
        timesteps = self._sample_timesteps(B, device, generator)
        sigma = get_sigmas(self.noise_scheduler, device, timesteps, n_dim=2, dtype=torch.float32)  # B, 1
        sigma = sigma[:, :, None].expand(B, Vp, n_lat).clone()
        hist = torch.rand(B, generator=generator, device=device) >= self.p_no_hist
        hist_views = hist[:, None] & self.context_mask(raw_batch, Vp)  # B, V'
        sigma[:, :, :self.n_hist_latent] *= (~hist_views)[:, :, None].float()
        if self.point_cloud:  # no-history samples are unconditional: the render is built from frame 0
            cond = cond.clone()
            cond[~hist, :, RAYMAP_CHANNELS:] = 0
        s = sigma[:, :, None, :, None, None].to(latents.dtype)

        batch = TrainingBatch()
        batch.latents = latents
        batch.noise = noise
        batch.noisy_model_input = (1.0 - s) * latents + s * noise
        batch.sigmas = sigma
        batch.timesteps = sigma * self.num_train_timestep  # B, V', F (0 on clean history frames)
        batch.raw_latent_shape = latents.shape
        batch.raymaps = cond
        batch.loss_mask = (sigma > 0).float()
        batch.attn_metadata = None
        batch.attn_metadata_vsa = None
        batch.conditional_dict = self._null_text_dict(B)
        batch.infos = raw_batch.get("episode")
        return batch

    def _null_text_dict(self, batch_size: int) -> dict[str, torch.Tensor]:
        assert self.null_context is not None and self.null_context_mask is not None
        return {
            "encoder_hidden_states": self.null_context.expand(batch_size, *self.null_context.shape[1:]),
            "encoder_attention_mask": self.null_context_mask.expand(batch_size, *self.null_context_mask.shape[1:]),
        }

    def predict_flow(self, noisy_latents: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor,
                     text_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """noisy [B, V', C, F, h, w], timesteps [B, F] or [B, V', F] (0..1000), cond from `pack`
        ([B, V', Cc, F, h, w], or static [B, V', Cc, h, w]) -> flow (noise - x0) [B, V', C, F, h, w]."""
        dtype = self._get_training_dtype()
        if cond.dim() == 5:
            cond = cond[:, :, :, None].expand(-1, -1, -1, noisy_latents.shape[3], -1, -1)
        hidden = torch.cat([noisy_latents.to(dtype), cond.to(dtype)], dim=2)
        with torch.autocast(self.device.type, dtype=dtype), set_forward_context(current_timestep=timesteps,
                                                                                attn_metadata=None):
            return self.transformer(hidden_states=hidden,
                                    encoder_hidden_states=text_dict["encoder_hidden_states"],
                                    encoder_attention_mask=text_dict["encoder_attention_mask"],
                                    timestep=timesteps,
                                    return_dict=False)

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del conditional, cfg_uncond, attn_kind, clean_x, aug_t  # single null-text condition
        return self.predict_flow(noisy_latents, timestep, batch.raymaps, batch.conditional_dict)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self,
               latents: torch.Tensor,
               cond: torch.Tensor,
               *,
               n_hist: int,
               steps: int,
               generator: torch.Generator,
               context: torch.Tensor | None = None) -> torch.Tensor:
        """Euler flow-ODE sampling; the first ``n_hist`` latent frames of the ``context``
        views ([B, V'] bool, default all) are kept clean, everything else is generated."""
        B, Vp, _, n_lat = latents.shape[:4]
        shift = self.timestep_shift
        s = torch.linspace(1.0, 0.0, steps + 1, device=latents.device)
        sigmas = shift * s / (1 + (shift - 1) * s)
        if context is None:
            context = torch.ones(B, Vp, dtype=torch.bool, device=latents.device)
        hist = context.to(latents.device)[:, :, None] & (torch.arange(n_lat, device=latents.device) < n_hist)
        keep = hist[:, :, None, :, None, None]  # B, V', 1, F, 1, 1
        clean = latents.float()
        x = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=torch.float32)
        text = self._null_text_dict(B)
        for i in range(steps):  # state kept in fp32; only the model input is cast
            x = torch.where(keep, clean, x)
            t = (~hist).float() * float(sigmas[i]) * self.num_train_timestep  # B, V', F
            flow = self.predict_flow(x, t, cond, text)
            x = x + (sigmas[i + 1] - sigmas[i]) * flow.float()
        x = torch.where(keep, clean, x)
        return x.to(latents.dtype)
