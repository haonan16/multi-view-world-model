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

Task: future prediction. Every (view, latent frame) gets its own noise level, drawn from a
diffusion-forcing mixture (cfg.noise, see sample_noise_levels); the test-time setting --
latent frame 0 (= pixel frame 0 of every view) clean, everything else noised -- is always
part of the mixture. Flow matching with shifted-uniform t; loss = MSE on the flow of every
position with t > 0. Camera conditioning: cfg.cam_mode (model_multiview.CAM_MODES). Text:
cached UMT5 embeddings of the RH20T task description (cfg.text), dropped to "" with
cfg.text.dropout so classifier-free text guidance works at sampling.

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
        self.cam_mode = cfg.get("cam_mode", "concat")
        assert self.design == "tokens" or self.cam_mode in ("concat", "cat4d"), \
            f"cam_mode {self.cam_mode} needs per-view tokens"
        # diffusion-forcing mixture over noise-level patterns (weights, normalised); the v1
        # default is "frame 0 clean, rest one t" with 10% unconditional
        self.noise_mix = {k: float(v) for k, v in cfg.get("noise", {}).get("mix", {"first_frame": 0.9, "none": 0.1}).items() if float(v) > 0}
        self.hist_noise_prob = float(cfg.get("noise", {}).get("hist_noise_prob", 0.0))
        self.hist_noise_max = float(cfg.get("noise", {}).get("hist_noise_max", 0.15))
        text = cfg.get("text", {})
        self.use_text = bool(text.get("enabled", False))
        self.text_dropout = float(text.get("dropout", 0.1))
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
        # text: cached UMT5 embeddings of the task descriptions (make_task_text_embeds.py),
        # or only the embedding of "" when text is disabled
        if self.use_text:
            emb = torch.load(self.cfg.text.embeds_path, map_location="cpu", weights_only=False)
            self.text_embeds = {k: v["embed"] for k, v in emb.items()}
            null = self.text_embeds[""]
        else:
            self.text_embeds = {}
            null = torch.load(self.cfg.null_prompt_embed_path, map_location="cpu")
        self.register_buffer("null_context", null.to(self.dtype), persistent=False)
        self.model = MultiViewWanModel.from_t2v_pretrained(self.cfg.model.ckpt_path,
                                                           cam_mode=self.cam_mode,
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
        if self.cam_mode == "cat4d":
            # CAT3D/CAT4D raymap: ray origin (camera centre, view-0 frame) + unit direction
            o = batch["c2w"].to(self.device, self.dtype)[:, :, :3, 3, None, None]
            rays = torch.cat([o.expand_as(rays[:, :, :3]), rays[:, :, :3]], dim=2)
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

    def context(self, batch, drop_prob=0.0, force_null=False):
        """Per-sample text embeddings of the task description ("" when text is off, when
        force_null, or with prob drop_prob -- for classifier-free text guidance)."""
        out = []
        for task in batch["prompts"]:
            e = self.text_embeds.get(task) if (self.use_text and not force_null) else None
            if e is None or (drop_prob > 0 and torch.rand(()) < drop_prob):
                out.append(self.null_context)
            else:
                out.append(e.to(self.device, self.dtype))
        return out

    def cam_mats(self, batch):
        """[B, V, 4, 4] relative camera matrices for prope (K~ [R|t]) / gta ([R|t]), from the
        view-0-relative c2w and the intrinsics (pixels at height x width)."""
        if self.cam_mode not in ("prope", "gta"):
            return None
        w2c = torch.linalg.inv(batch["c2w"].to(self.device).float())
        if self.cam_mode == "gta":
            return w2c
        K = batch["intrinsics"].to(self.device).float()
        Kn = torch.zeros(*K.shape[:2], 4, 4, device=self.device)
        # intrinsics in units of the image size, principal point at the image centre ~ 0
        Kn[..., 0, 0] = K[..., 0, 0] / self.width
        Kn[..., 1, 1] = K[..., 1, 1] / self.height
        Kn[..., 0, 2] = K[..., 0, 2] / self.width - 0.5
        Kn[..., 1, 2] = K[..., 1, 2] / self.height - 0.5
        Kn[..., 2, 2] = Kn[..., 3, 3] = 1
        return Kn @ w2c

    def sample_noise_levels(self, B, Vp, Fl):
        """Diffusion-forcing noise levels t in [0, 1] (shifted-uniform) per (sample, view,
        latent frame), from the mixture self.noise_mix:
          first_frame   latent frame 0 of every view is history, the rest one shared t
                        -- the test-time setting (1 frame of history)
          rand_history  a history prefix of k = 1..F-1 latent frames, the rest one shared t
          independent   an independent t per latent frame (shared across views)
          view_history  frame 0 of every view + ALL frames of a random non-empty proper
                        subset of views are history (other views of the dynamics as context)
          none          no history, one shared t (unconditional; enables history guidance)
        History is clean, or with prob hist_noise_prob slightly noised (t <= hist_noise_max)
        so the model tolerates imperfect history. Returns t [B, Vp, Fl], hist [B, Vp, Fl]."""
        names = list(self.noise_mix)
        probs = torch.tensor([self.noise_mix[n] for n in names], dtype=torch.float)
        modes = torch.multinomial(probs / probs.sum(), B, replacement=True)
        t = torch.empty(B, Vp, Fl, device=self.device)
        hist = torch.zeros(B, Vp, Fl, dtype=torch.bool, device=self.device)
        for i, m in enumerate(modes.tolist()):
            mode = names[m]
            t[i] = self._shift(torch.rand((), device=self.device))
            if mode == "first_frame":
                hist[i, :, 0] = True
            elif mode == "rand_history" and Fl > 1:
                hist[i, :, :int(torch.randint(1, Fl, ()))] = True
            elif mode == "independent":
                t[i] = self._shift(torch.rand(1, Fl, device=self.device)).expand(Vp, Fl)
            elif mode == "view_history" and Vp > 1:
                hist[i, :, 0] = True
                k = int(torch.randint(1, Vp, ()))
                hist[i, torch.randperm(Vp)[:k]] = True
            if hist[i].any():
                h_t = 0.0
                if torch.rand(()) < self.hist_noise_prob:
                    h_t = float(torch.rand(())) * self.hist_noise_max
                t[i][hist[i]] = h_t
        return t, hist

    # ---- training ----

    def _shift(self, t):
        s = self.sample_shift
        return t * s / (1 + (s - 1) * t)

    def training_step(self, batch, batch_idx=None):
        lat, cond = self.pack(batch)
        B, Vp, C, Fl = lat.shape[:4]
        noise = torch.randn_like(lat)
        t, hist = self.sample_noise_levels(B, Vp, Fl)                        # B, Vp, Fl
        te = t[:, :, None, :, None, None]
        noisy = lat * (1 - te) + noise * te
        pred = self.model(noisy, cond, t * self.num_train_timesteps,
                          self.context(batch, drop_prob=self.text_dropout),
                          cam_mats=self.cam_mats(batch), hist_mask=hist)
        se = ((pred.float() - (noise - lat).float()) ** 2).mean(dim=(2, 4, 5))   # B, Vp, F
        w = (t > 0).float()
        loss = (se * w).sum() / w.sum().clamp(min=1)
        self.log("train/loss", loss, sync_dist=True, prog_bar=True)
        self.log("global_step", float(self.global_step))   # ModelCheckpoint monitor key
        self.log("train/tokens_per_sample", float(Vp * Fl * lat.shape[-2] * lat.shape[-1] / 4))
        return loss

    # ---- sampling / validation ----

    @torch.no_grad()
    def sample(self, lat, cond, n_hist=1, batch=None, hist_mask=None,
               text_guidance=0.0, hist_guidance=0.0):
        """UniPC sampling with clean history. hist_mask [B, Vp, Fl] (default: the first
        n_hist latent frames of every view). Guidance (classifier-free):
          text_guidance  extrapolate away from the "" prompt
          hist_guidance  extrapolate away from the prediction without history (history
                         frames replaced by pure noise at t = 1, as in the DFoT codebase)"""
        B, Vp, C, Fl = lat.shape[:4]
        if hist_mask is None:
            hist_mask = torch.zeros(B, Vp, Fl, dtype=torch.bool, device=lat.device)
            hist_mask[:, :, :n_hist] = True
        hm = hist_mask[:, :, None, :, None, None]
        ctx = self.context(batch) if batch is not None else [self.null_context] * B
        null_ctx = [self.null_context] * B
        cams = self.cam_mats(batch) if batch is not None else None
        sched = FlowUniPCMultistepScheduler(num_train_timesteps=self.num_train_timesteps,
                                            shift=self.sample_shift, use_dynamic_shifting=False)
        sched.set_timesteps(self.cfg.sample_steps, device=self.device, shift=self.sample_shift)
        x = torch.randn_like(lat)
        for ts in sched.timesteps:
            x = torch.where(hm, lat, x)
            t = torch.full((B, Vp, Fl), float(ts), device=self.device).masked_fill(hist_mask, 0)
            flow = self.model(x, cond, t, ctx, cam_mats=cams, hist_mask=hist_mask)
            g = flow
            if text_guidance:
                f0 = self.model(x, cond, t, null_ctx, cam_mats=cams, hist_mask=hist_mask)
                g = g + text_guidance * (flow - f0)
            if hist_guidance:
                xn = torch.where(hm, torch.randn_like(x), x)
                tn = t.masked_fill(hist_mask, float(self.num_train_timesteps))
                f0 = self.model(xn, cond, tn, ctx, cam_mats=cams,
                                hist_mask=torch.zeros_like(hist_mask))
                g = g + hist_guidance * (flow - f0)
            x = sched.step(g.flatten(0, 1), ts, x.flatten(0, 1), return_dict=False)[0].unflatten(0, (B, Vp))
        return torch.where(hm, lat, x)

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
        pred = self._views(self.sample(lat, cond, self.cfg.n_hist_latent, batch=batch,
                                       text_guidance=self.cfg.get("val_text_guidance", 0.0),
                                       hist_guidance=self.cfg.get("val_hist_guidance", 0.0)), B).float()
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
