"""Multiview video dynamics: "views as tokens" vs "views tiled into one frame".

Both designs train on the same RH20T samples (datasets/rh20t.py: V views x T frames,
per-view Plucker raymaps) with the same backbone (Wan2.1-T2V-1.3B, MultiViewWanModel),
objective, optimiser and schedule. Only the view packing differs (cfg.design):

  tokens  each view is VAE-encoded on its own at (H, W); the V views' tokens share
          one sequence (same RoPE per view, camera from the raymap channels).
          Tokens per sample = V x F x (H/16) x (W/16).
  tiled   the V views are downscaled into ONE (H, W) canvas (fixed canvas), VAE-encoded
          as one video, raymaps tiled the same way. Tokens = F x (H/16) x (W/16), i.e. V x
          fewer.

Task: future prediction. The first latent frame (= pixel frame 0 of every view) is clean
history with prob 1 - p_no_hist; the remaining latent frames are noised at one shared t
(flow matching, shifted-uniform t). Loss = MSE on the flow of the noised frames only.

Validation: condition on frame 0 of every view, sample the rest, and report per-view
PSNR / SSIM / LPIPS on frames 1.. at the full per-view resolution (the tiled design is
upsampled from its tiles) AND at tile resolution (both designs area-downsampled to the
tile size, isolating what tiling costs beyond resolution). Also logged: the VAE round
trip through the design's packing (its reconstruction ceiling) and a copy-frame-0
baseline.
"""
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from transformers import get_scheduler

from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from datasets._multiview import tile_views, untile_views
from .modules.model import WanAttentionBlock
from .modules.model_multiview import MultiViewWanModel
from .modules.vae import video_vae_factory
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from huggingface_hub import hf_hub_download


def _ssim(x, y):
    """x, y: [N, 3, H, W] in [0, 1]. Gaussian-window SSIM (11x11, sigma 1.5), mean over N."""
    g = torch.exp(-(torch.arange(11, device=x.device) - 5.0) ** 2 / (2 * 1.5 ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).to(x.dtype)
    f = lambda z: F.conv2d(z, w, padding=5, groups=3)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.mean(dim=(1, 2, 3))


class WanMultiView(BasePytorchAlgo):

    def __init__(self, cfg):
        self.design = cfg.design
        assert self.design in ("tokens", "tiled"), self.design
        self.n_views = cfg.n_views
        self.n_frames = cfg.n_frames
        self.height, self.width = cfg.height, cfg.width
        self.sample_shift = cfg.sample_shift
        self.num_train_timesteps = cfg.num_train_timesteps
        self.gradient_checkpointing_rate = cfg.gradient_checkpointing_rate
        super().__init__(cfg)
        # checkpoints hold only the DiT (on_save_checkpoint); the frozen VAE and the null
        # context are rebuilt in configure_model, so resuming must not demand them
        self.strict_loading = False

    @staticmethod
    def classes_to_shard():
        return {WanAttentionBlock}

    # ---- setup ----

    def configure_model(self):
        if getattr(self, "model", None) is not None:
            return
        vae_path = hf_hub_download(os.path.dirname(self.cfg.vae.ckpt_path),
                                   os.path.basename(self.cfg.vae.ckpt_path))
        self.vae = video_vae_factory(pretrained_path=vae_path, z_dim=self.cfg.vae.z_dim
                                     ).eval().requires_grad_(False).to(self.dtype)
        self.register_buffer("vae_mean", torch.tensor(self.cfg.vae.mean, dtype=self.dtype))
        self.register_buffer("vae_inv_std", 1.0 / torch.tensor(self.cfg.vae.std, dtype=self.dtype))
        # no captions: every sample uses the UMT5 embedding of the empty prompt
        self.register_buffer("null_context", torch.load(self.cfg.null_prompt_embed_path,
                                                        map_location="cpu").to(self.dtype))
        self.model = MultiViewWanModel.from_t2v_pretrained(self.cfg.model.ckpt_path,
                                                           local_files_only=True)
        if self.cfg.model.tuned_ckpt_path:
            sd = torch.load(self.cfg.model.tuned_ckpt_path, map_location="cpu", mmap=True,
                            weights_only=False)["state_dict"]
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
            self.model.load_state_dict(sd)
        self.model.to(self.dtype).train()
        if self.gradient_checkpointing_rate > 0:
            self.model.gradient_checkpointing_enable(p=self.gradient_checkpointing_rate)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr,
                                betas=tuple(self.cfg.betas), weight_decay=self.cfg.weight_decay)
        sched = get_scheduler(optimizer=opt, **self.cfg.lr_scheduler)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_save_checkpoint(self, checkpoint):
        # VAE / null context are rebuilt from pretrained every run; keep only the DiT.
        sd = checkpoint.get("state_dict")
        if sd is not None:
            for k in list(sd):
                if not k.startswith("model."):
                    del sd[k]

    # ---- data -> latents ----

    @torch.no_grad()
    def encode(self, x):
        """x: [N, T, 3, H, W] in [-1, 1] -> [N, C, F, h, w]"""
        return self.vae.encode(rearrange(x, "n t c h w -> n c t h w").to(self.dtype),
                               [self.vae_mean, self.vae_inv_std])

    @torch.no_grad()
    def decode(self, z):
        """z: [N, C, F, h, w] -> [N, T, 3, H, W] in [-1, 1]"""
        x = self.vae.decode(z.to(self.dtype), [self.vae_mean, self.vae_inv_std]).clamp_(-1, 1)
        return rearrange(x, "n c t h w -> n t c h w")

    @torch.no_grad()
    def pack(self, batch):
        """-> latents [B, V', C, F, h, w], cond [B, V', 6, h, w] (V' = V tokens / 1 tiled)."""
        videos = batch["videos"].to(self.device, self.dtype)            # B, V, T, 3, H, W
        rays = batch["raymaps"].to(self.device, self.dtype)              # B, V, 6, Hr, Wr
        B, V = videos.shape[:2]
        if self.design == "tiled":
            videos = tile_views(videos)[:, None]                           # B, 1, T, 3, H, W
            rays = tile_views(rays[:, :, None])[:, None, 0]                # B, 1, 6, Hr, Wr
        Vp = videos.shape[1]
        lat = self.encode(videos.flatten(0, 1))
        lat = lat.unflatten(0, (B, Vp))
        h, w = lat.shape[-2:]
        if rays.shape[-2:] != (h, w):
            rays = F.interpolate(rays.flatten(0, 1), size=(h, w), mode="area").unflatten(0, (B, Vp))
        return lat, rays

    def context(self, B):
        return [self.null_context] * B

    # ---- training ----

    def _shift(self, t):
        s = self.sample_shift
        return t * s / (1 + (s - 1) * t)

    def training_step(self, batch, batch_idx=None):
        lat, cond = self.pack(batch)
        B, Vp, C, Fl = lat.shape[:4]
        noise = torch.randn_like(lat)
        t = self._shift(torch.rand(B, 1, device=self.device)).expand(B, Fl).clone()
        hist = torch.rand(B, device=self.device) >= self.cfg.p_no_hist
        t[hist, : self.cfg.n_hist_latent] = 0.0
        te = t[:, None, None, :, None, None]
        noisy = lat * (1 - te) + noise * te
        pred = self.model(noisy, cond, t * self.num_train_timesteps, self.context(B))
        se = ((pred.float() - (noise - lat).float()) ** 2).mean(dim=(1, 2, 4, 5))   # B, F
        w = (t > 0).float()
        loss = (se * w).sum() / w.sum().clamp(min=1)
        self.log("train/loss", loss, sync_dist=True, prog_bar=True)
        self.log("global_step", float(self.global_step))   # ModelCheckpoint monitor key
        self.log("train/tokens_per_sample", float(Vp * Fl * lat.shape[-2] * lat.shape[-1] / 4))
        return loss

    # ---- sampling / validation ----

    @torch.no_grad()
    def sample(self, lat, cond, n_hist):
        B, Vp, C, Fl = lat.shape[:4]
        sched = FlowUniPCMultistepScheduler(num_train_timesteps=self.num_train_timesteps,
                                            shift=self.sample_shift, use_dynamic_shifting=False)
        sched.set_timesteps(self.cfg.sample_steps, device=self.device, shift=self.sample_shift)
        x = torch.randn_like(lat)
        for ts in sched.timesteps:
            x[:, :, :, :n_hist] = lat[:, :, :, :n_hist]
            t = torch.full((B, Fl), float(ts), device=self.device)
            t[:, :n_hist] = 0
            flow = self.model(x, cond, t, self.context(B))
            x = sched.step(flow.flatten(0, 1), ts, x.flatten(0, 1), return_dict=False)[0].unflatten(0, (B, Vp))
        x[:, :, :, :n_hist] = lat[:, :, :, :n_hist]
        return x

    def _views(self, z, B):
        """latents [B, V', C, F, h, w] -> pixels [B, V, T, 3, H_v, W_v]"""
        x = self.decode(z.flatten(0, 1)).unflatten(0, (B, z.shape[1]))
        if self.design == "tiled":
            x = untile_views(x[:, 0], self.n_views)
        return x

    def validation_step(self, batch, batch_idx=None):
        gt = batch["videos"].to(self.device).float()                       # B, V, T, 3, H, W
        B, V, T, _, H, W = gt.shape
        lat, cond = self.pack(batch)
        pred = self._views(self.sample(lat, cond, self.cfg.n_hist_latent), B).float()
        th, tw = H // 2, W // 2                                          # 2x2 tile size
        down = lambda x, s: F.interpolate(x.flatten(0, 2), size=s, mode="area").unflatten(0, x.shape[:3])
        up = lambda x: F.interpolate(x.flatten(0, 2), size=(H, W), mode="bilinear",
                                     align_corners=False).unflatten(0, x.shape[:3])
        to_full = lambda x: x if x.shape[-2:] == (H, W) else up(x)
        pred_full = to_full(pred)
        # ceilings / baselines: VAE round trip through this design's packing (for "tiled"
        # this is what the canvas resolution alone costs), and copying frame 0 forward
        vae_full = to_full(self._views(lat, B).float())
        copy = gt[:, :, :1].expand_as(gt)
        gt_tile = down(gt, (th, tw))
        out = {}
        for name, p, g in [("full", pred_full, gt), ("tile", down(pred_full, (th, tw)), gt_tile),
                           ("vae_full", vae_full, gt), ("vae_tile", down(vae_full, (th, tw)), gt_tile),
                           ("copy0_full", copy, gt)]:
            p, g = p[:, :, 1:], g[:, :, 1:]                                   # future frames only
            p01, g01 = (p.flatten(0, 2) + 1) / 2, (g.flatten(0, 2) + 1) / 2
            mse = ((p01 - g01) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
            out[f"val/psnr_{name}"] = (-10 * torch.log10(mse)).mean()
            out[f"val/ssim_{name}"] = _ssim(p01.clamp(0, 1), g01).mean()
            if self._lpips is not None:
                out[f"val/lpips_{name}"] = self._lpips(p01 * 2 - 1, g01 * 2 - 1).mean()
        for k, v in out.items():
            self.log(k, v, sync_dist=True, batch_size=B)
        if batch_idx == 0 and self.trainer.is_global_zero:
            vid = torch.cat([torch.cat(list(gt[0]), dim=-1), torch.cat(list(pred_full[0]), dim=-1)], dim=-2)
            self.logger.experiment.log({"val/gt_top_pred_bottom": wandb.Video(
                ((vid.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy(), fps=10, format="mp4",
                caption=f"{self.design} | {batch['episode'][0]}")}, step=self.global_step)

    def on_validation_epoch_start(self):
        if not hasattr(self, "_lpips"):
            try:
                import lpips
                self._lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
            except ImportError:
                logging.warning("lpips not installed; skipping LPIPS")
                self._lpips = None
