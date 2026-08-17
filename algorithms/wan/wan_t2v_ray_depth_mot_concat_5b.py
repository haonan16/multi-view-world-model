import logging
import gc
import torch
import bitsandbytes as bnb
import torch.nn.functional as F
import wandb
import numpy as np
import torch.distributed as dist
from einops import rearrange, repeat
from tqdm import tqdm
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from transformers import get_scheduler
import zmq
import msgpack
import io
import os
from huggingface_hub import hf_hub_download
from PIL import Image
import torchvision.transforms as transforms
from utils.video_utils import numpy_to_mp4_bytes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .modules.model_ray_depth_mot_concat_5b import WanModel, WanAttentionBlock
from .modules.model import WanModel as BaseWanModel
from .modules.t5 import umt5_xxl, T5CrossAttention, T5SelfAttention
from .modules.tokenizers import HuggingfaceTokenizer
from .modules.vae import video_vae_factory
from .modules.vae_v2 import (
    video_vae_factory as video_vae_factory_v2,
    video_vae_flf_factory_v2,
)
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.distributed_utils import is_rank_zero
from datasets.re10k import raymap_to_camera


def _depth_to_colormap(depth: torch.Tensor, ref: torch.Tensor = None) -> torch.Tensor:
    """Apply turbo colormap to depth/disparity. depth: [B, T, C, H, W] -> [B, T, 3, H, W] in [0, 1].
    Per-sample min-max normalized over the whole clip so the colormap stretches the actual range
    (raw disparity values cluster in a narrow band and are visually flat without this). When `ref`
    is given, its per-sample min/max is used to normalize `depth` — pass GT as `ref` for both pred
    and GT to keep them on the same scale side-by-side."""
    cmap = plt.get_cmap("turbo")
    B, T = depth.shape[:2]
    # collapse channels to 1 (handles both 1-ch GT and 3-ch VAE-decoded pred)
    d = depth.mean(dim=2, keepdim=True)  # [B, T, 1, H, W]
    src = ref.mean(dim=2, keepdim=True) if ref is not None else d
    src_flat = src.reshape(B, -1).float()
    d_min = src_flat.amin(dim=1).view(B, 1, 1, 1, 1)
    d_max = src_flat.amax(dim=1).view(B, 1, 1, 1, 1)
    rng = (d_max - d_min).clamp(min=1e-6)
    depth_01 = ((d.float() - d_min) / rng).clamp(0, 1).squeeze(2).cpu().numpy()  # [B, T, H, W]
    colored = cmap(depth_01.reshape(-1, *depth_01.shape[2:]))[:, :, :, :3]  # [B*T, H, W, 3]
    return torch.from_numpy(colored).permute(0, 3, 1, 2).float().view(B, T, 3, *depth.shape[-2:])


def _psnr_batch(pred: torch.Tensor, gt: torch.Tensor, skip_first_last: bool = True) -> float:
    """pred, gt: [B, T, 3, H, W] in [-1,1]. Returns mean PSNR over interior generated frames."""
    import math
    pred_u = (pred * 0.5 + 0.5).clamp(0, 1).float()
    gt_u   = (gt   * 0.5 + 0.5).clamp(0, 1).float()
    if skip_first_last and pred_u.shape[1] > 2:
        pred_u = pred_u[:, 1:-1]
        gt_u   = gt_u[:, 1:-1]
    mse  = ((pred_u - gt_u) ** 2).mean(dim=(2, 3, 4)).clamp(min=1e-10)  # [B, T]
    psnr = -10.0 * torch.log10(mse)
    return float(psnr.mean().item())


def _depth_delta_batch(pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> dict:
    """
    pred_depth: [B, T, 3, H, W] in [-1,1] (3-ch VAE decoded)
    gt_depth:   [B, T, 1, H, W] in [-1,1]
    Returns delta1/2/3 after per-sample least-squares scale+shift alignment.
    """
    B = pred_depth.shape[0]
    pred_u = (pred_depth.mean(dim=2) * 0.5 + 0.5).clamp(0, 1).float()  # [B, T, H, W]
    gt_u   = (gt_depth.squeeze(2)   * 0.5 + 0.5).clamp(0, 1).float()   # [B, T, H, W]
    d1, d2, d3 = [], [], []
    for b in range(B):
        pv = pred_u[b].reshape(-1).cpu()
        gv = gt_u[b].reshape(-1).cpu()
        valid = (gv > 1e-4) & pv.isfinite() & gv.isfinite()
        if valid.sum() < 10:
            continue
        pv, gv = pv[valid], gv[valid]
        if pv.shape[0] > 50000:  # subsample for speed
            idx = torch.randperm(pv.shape[0])[:50000]
            pv, gv = pv[idx], gv[idx]
        A   = torch.stack([pv, torch.ones_like(pv)], dim=1)
        sol = torch.linalg.lstsq(A, gv.unsqueeze(1)).solution
        scale, shift = float(sol[0].item()), float(sol[1].item())
        if scale <= 0:
            continue
        aligned = (scale * pred_u[b] + shift).clamp(min=1e-4)
        gt_b    = gt_u[b].clamp(min=1e-4)
        ratio   = torch.max(aligned / gt_b, gt_b / aligned)
        d1.append(float((ratio < 1.25).float().mean().item()))
        d2.append(float((ratio < 1.25 ** 2).float().mean().item()))
        d3.append(float((ratio < 1.25 ** 3).float().mean().item()))
    if not d1:
        return {"delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
    return {"delta1": float(np.mean(d1)), "delta2": float(np.mean(d2)), "delta3": float(np.mean(d3))}


def _compute_relative_pose_errors(w2c_pred: np.ndarray, w2c_gt: np.ndarray) -> tuple:
    """Returns (combined_errors, rot_errors, trans_errors) arrays."""
    N = len(w2c_pred)
    combined, rots, trans = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            rel_pred = w2c_pred[j] @ np.linalg.inv(w2c_pred[i])
            rel_gt   = w2c_gt[j]   @ np.linalg.inv(w2c_gt[i])
            R_err  = rel_pred[:3, :3] @ rel_gt[:3, :3].T
            cos_a  = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
            rot_e  = float(np.degrees(np.arccos(cos_a)))
            tp, tg = rel_pred[:3, 3], rel_gt[:3, 3]
            np_n, ng_n = np.linalg.norm(tp), np.linalg.norm(tg)
            if np_n < 1e-8 or ng_n < 1e-8:
                trans_e = 0.0
            else:
                cos_t   = np.clip(np.dot(tp / np_n, tg / ng_n), -1.0, 1.0)
                trans_e = float(np.degrees(np.arccos(cos_t)))
            combined.append(max(rot_e, trans_e))
            rots.append(rot_e)
            trans.append(trans_e)
    return np.array(combined), np.array(rots), np.array(trans)


def _compute_auc(errors: np.ndarray, max_threshold: float, n_points: int = 1000) -> float:
    thresholds = np.linspace(0.0, max_threshold, n_points)
    return float(np.array([(errors <= t).mean() for t in thresholds]).mean()) * 100.0


def _pose_metrics_batch(pred_ray_d: torch.Tensor, pred_ray_m: torch.Tensor,
                        gt_raymaps: torch.Tensor) -> dict:
    """
    pred_ray_d: [B, T, 3, H, W]
    pred_ray_m: [B, T, 3, H, W]
    gt_raymaps: [B, T, 6, H, W]
    Returns auc3, auc30, rot_err, trans_err (mean over all pairs × batch).
    """
    all_combined, all_rot, all_trans = [], [], []
    for b in range(pred_ray_d.shape[0]):
        pred_rm = torch.cat([pred_ray_d[b], pred_ray_m[b]], dim=1)  # [T, 6, H, W]
        gt_rm   = gt_raymaps[b]                                       # [T, 6, H, W]
        try:
            _, pred_ext = raymap_to_camera(pred_rm.float())
            _, gt_ext   = raymap_to_camera(gt_rm.float())
        except Exception:
            continue
        pred_w2c = np.linalg.inv(pred_ext.cpu().numpy().astype(np.float64))
        gt_w2c   = np.linalg.inv(gt_ext.cpu().numpy().astype(np.float64))
        combined, rots, trans = _compute_relative_pose_errors(pred_w2c, gt_w2c)
        all_combined.append(combined)
        all_rot.append(rots)
        all_trans.append(trans)
    if not all_combined:
        return {"auc3": 0.0, "auc30": 0.0, "rot_err": 0.0, "trans_err": 0.0}
    cat_c = np.concatenate(all_combined)
    cat_r = np.concatenate(all_rot)
    cat_t = np.concatenate(all_trans)
    return {
        "auc3":      _compute_auc(cat_c, max_threshold=3.0),
        "auc30":     _compute_auc(cat_c, max_threshold=30.0),
        "rot_err":   float(cat_r.mean()),
        "trans_err": float(cat_t.mean()),
    }


def _procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align source camera centers to target via optimal rotation + scale (no reflection)."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    s_c, t_c = source - mu_s, target - mu_t
    U, _, Vt = np.linalg.svd(t_c.T @ s_c)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    ss = (s_c ** 2).sum()
    scale = (t_c * (s_c @ R.T)).sum() / ss if ss > 0 else 1.0
    return scale * (source - mu_s) @ R.T + mu_t


def _make_trajectory_figure(pred_raymaps: torch.Tensor, gt_raymaps: torch.Tensor) -> Image.Image:
    """
    Return a PIL image of the 3-D trajectory plot (pred + Procrustes-aligned GT).

    pred_raymaps: [T, 6, H, W]
    gt_raymaps:   [T, 6, H, W]
    """
    try:
        _, pred_ext = raymap_to_camera(pred_raymaps.float())  # [T, 4, 4]
        _, gt_ext   = raymap_to_camera(gt_raymaps.float())    # [T, 4, 4]
    except Exception as e:
        logging.warning(f"raymap_to_camera failed, skipping trajectory viz: {e}")
        return None

    pred_c = pred_ext[:, :3, 3].cpu().numpy()   # [T, 3]
    gt_c   = gt_ext[:, :3, 3].cpu().numpy()     # [T, 3]
    pred_aligned = _procrustes_align(pred_c, gt_c)

    T_val = len(pred_c)
    cmap = plt.get_cmap("hsv")
    colors = [cmap(i / max(T_val - 1, 1)) for i in range(T_val)]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(T_val - 1):
        ax.plot(pred_aligned[i:i+2, 0], pred_aligned[i:i+2, 1], pred_aligned[i:i+2, 2],
                color=colors[i], linewidth=2)
    ax.scatter(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2],
               c=colors, s=30, zorder=5, label="pred (aligned)")
    ax.plot(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2],
            color="black", linewidth=1.5, linestyle="--")
    ax.scatter(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2],
               color="black", s=20, zorder=4, label="GT")
    ax.legend()
    ax.set_title("Camera Trajectories (pred Procrustes-aligned to GT)")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    pil_img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return pil_img


class WanTextToVideoRayDepthMoTConcat5B(BasePytorchAlgo):
    """
    5B variant of WanTextToVideoRayDepthMoTConcat. Identical model/behavior, but
    `configure_optimizers` uses bitsandbytes 8-bit AdamW (uint8 optimizer states) to
    cut optimizer-state GPU memory ~4x (8 bytes/param -> ~2 bytes/param), so the 5B
    model (Wan2.2-TI2V-5B + v2 VAE) fits. Routed via config name
    `wan_t2v_ray_depth_mot_concat_5b` in experiments/exp_video.py:compatible_algorithms.

    Wan T2V with Mixture-of-Transformers for RGB + Plücker raymaps + depth.

    Token layout: [rgb | ray_d | ray_m | depth]
    Stream 0 (RGB) / Stream 1 (Ray, shared for ray_d and ray_m) / Stream 2 (Depth).
    """

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        """Lightning's FSDPPrecision raises MisconfigurationException on trainer-level
        norm clipping (jobs 29375725/26 died at the first optimizer step). Bypass the
        precision plugin: FSDP's own clip_grad_norm_ computes the TOTAL grad norm
        across shards (incl. the hybrid-mesh process groups) and is the supported way
        to clip under FSDP. Fallback covers non-FSDP (single-GPU debug) runs."""
        if gradient_clip_val is None:
            return
        model = getattr(self.trainer.strategy, "model", None)
        if model is not None and hasattr(model, "clip_grad_norm_"):
            norm = model.clip_grad_norm_(gradient_clip_val)
        else:
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)
        # clip_grad_norm_ returns the TOTAL pre-clip norm (identical on every rank —
        # it's the global cross-shard norm), so logging it is free. Stashed here and
        # emitted from training_step's logging block (self.log inside this hook would
        # depend on Lightning's fx whitelist; the stash avoids that entirely).
        self._grad_norm_last = float(norm)

    def __init__(self, cfg):
        self.num_train_timesteps = cfg.num_train_timesteps
        self.height = cfg.height
        self.width = cfg.width
        self.n_frames = cfg.n_frames
        self.gradient_checkpointing_rate = cfg.gradient_checkpointing_rate
        self.sample_solver = cfg.sample_solver
        self.sample_steps = cfg.sample_steps
        self.sample_shift = cfg.sample_shift
        self.lang_guidance = cfg.lang_guidance
        self.neg_prompt = cfg.neg_prompt
        self.hist_guidance = cfg.hist_guidance
        self.sliding_hist = cfg.sliding_hist
        self.diffusion_forcing = cfg.diffusion_forcing
        self.vae_stride = cfg.vae.stride
        self.patch_size = cfg.model.patch_size
        self.diffusion_type = cfg.diffusion_type

        self.lat_h = self.height // self.vae_stride[1]
        self.lat_w = self.width // self.vae_stride[2]
        
        if self.diffusion_forcing.enabled and self.diffusion_forcing.mode == "ray_depth_prediction_firstlast":
            if self.n_frames <= 1:
                per_modality_t = self.n_frames
            else:
                interior = self.n_frames - 2
                per_modality_t = 2 + (interior + self.vae_stride[0] - 1) // self.vae_stride[0]
            self.lat_t = 4 * per_modality_t
        else:
            self.lat_t = 4 * (1 + (self.n_frames - 1) // self.vae_stride[0])  # rgb + ray_d + ray_m + depth

        self.lat_c = cfg.vae.z_dim
        self.max_area = self.height * self.width
        # The model fuses ray_d + ray_m into one block, so its internal sequence
        # spans 3 modality blocks (rgb | ray | depth), not 4 frames-per-modality.
        # Using 4*T here would zero-pad the seq by exactly one block; the padding
        # tokens get modality_mask=0 and are then miscounted as RGB tokens.
        self.max_tokens = (
            (3 * self.lat_t // 4)
            * (self.lat_h // self.patch_size[1])
            * (self.lat_w // self.patch_size[2])
        )

        self.load_prompt_embed = cfg.load_prompt_embed
        self.load_video_latent = cfg.load_video_latent
        self.socket = None
        if (self.sliding_hist - 1) % self.vae_stride[0] != 0:
            raise ValueError(
                "sliding_hist - 1 must be a multiple of vae_stride[0]. "
                f"Got {self.sliding_hist} and vae stride {self.vae_stride[0]}"
            )
        if self.load_video_latent:
            raise NotImplementedError("Loading video latent is not implemented yet")
        super().__init__(cfg)

    @staticmethod
    def classes_to_shard():
        return {WanAttentionBlock, T5CrossAttention, T5SelfAttention}

    @property
    def is_inference(self) -> bool:
        return self._trainer is None or not self.trainer.training

    def configure_model(self):
        logging.info("Building model...")
        if not self.cfg.load_prompt_embed:
            text_encoder = (
                umt5_xxl(
                    encoder_only=True,
                    return_tokenizer=False,
                    dtype=torch.bfloat16 if self.is_inference else self.dtype,
                    device=torch.device("cpu"),
                )
                .eval()
                .requires_grad_(False)
            )
            if self.cfg.text_encoder.ckpt_path is not None:
                text_encoder_repo_id = os.path.dirname(self.cfg.text_encoder.ckpt_path)
                text_encoder_filename = os.path.basename(self.cfg.text_encoder.ckpt_path)
                text_encoder_model_path = hf_hub_download(
                    repo_id=text_encoder_repo_id, filename=text_encoder_filename
                )
                text_encoder.load_state_dict(
                    torch.load(text_encoder_model_path, map_location="cpu", weights_only=True)
                )
            if self.cfg.text_encoder.compile:
                text_encoder = torch.compile(text_encoder)
        else:
            text_encoder = None
        self.text_encoder = text_encoder

        self.tokenizer = HuggingfaceTokenizer(
            name=self.cfg.text_encoder.name,
            seq_len=self.cfg.text_encoder.text_len,
            clean="whitespace",
        )
        vae_repo_id = os.path.dirname(self.cfg.vae.ckpt_path)
        vae_filename = os.path.basename(self.cfg.vae.ckpt_path)
        vae_model_path = hf_hub_download(repo_id=vae_repo_id, filename=vae_filename)
        
        is_v2 = getattr(self.cfg.vae, "version", "wan2_1") == "wan2_2"
        is_flf = (self.diffusion_forcing.enabled and
                  self.diffusion_forcing.mode in ("ray_prediction_firstlast", "ray_depth_prediction_firstlast"))
        if is_v2 and is_flf:
            vae_factory = video_vae_flf_factory_v2
        elif is_v2:
            vae_factory = video_vae_factory_v2
        elif is_flf:
            from .modules.vae import video_vae_flf_factory as vae_factory
        else:
            vae_factory = video_vae_factory

        self.vae = (
            vae_factory(pretrained_path=vae_model_path, z_dim=self.cfg.vae.z_dim)
            .eval()
            .requires_grad_(False)
        ).to(self.dtype)
        self.register_buffer("vae_mean", torch.tensor(self.cfg.vae.mean, dtype=self.dtype))
        self.register_buffer("vae_inv_std", 1.0 / torch.tensor(self.cfg.vae.std, dtype=self.dtype))
        self.vae_scale = [self.vae_mean, self.vae_inv_std]
        if self.cfg.vae.compile:
            self.vae = torch.compile(self.vae)

        if self.cfg.model.tuned_ckpt_path is not None and is_rank_zero:
            # tuned_ckpt_path is a WEIGHTS-ONLY warm start, not a resume. It restores
            # parameters and nothing else: the AdamW state is fresh, the LR scheduler
            # restarts (so the 1000-step warmup re-runs) and global_step goes back to 0.
            # For a true resume use `load=<ckpt>` / `resume=<wandb id>`, which reaches
            # trainer.fit(ckpt_path=...) -- but that needs checkpoints saved with
            # experiment.training.checkpointing.save_weights_only=false, otherwise there
            # is no optimizer state in the file to restore.
            print(
                "[WARM START] algorithm.model.tuned_ckpt_path is set -> weights only. "
                "AdamW state reset, LR warmup will re-run, global_step restarts at 0. "
                "This is NOT a resume; use load=<ckpt> with save_weights_only=false "
                "for that.",
                flush=True,
            )

        if self.cfg.model.tuned_ckpt_path is None:
            # local_files_only=True: the 5B checkpoint is SHARDED (3 safetensors + index),
            # and diffusers' shard resolver calls model_info() (a hub API call) unless this
            # is set — which fails under HF_HUB_OFFLINE=1. The shards are fully cached, so
            # force offline resolution. (The 1.3B single-file ckpt never hit this path.)
            self.model = WanModel.from_t2v_pretrained(
                self.cfg.model.ckpt_path, local_files_only=True
            )
        else:
            model_repo_id = self.cfg.model.ckpt_path
            config_path = hf_hub_download(repo_id=model_repo_id, filename="config.json")
            loaded_state_dict = self._load_tuned_state_dict()

            is_mot_checkpoint = any(
                k.startswith("blocks.0.self_attn.q.0.") for k in loaded_state_dict
            )

            if is_mot_checkpoint:
                # Construct on the meta device so from_config (full fp32 allocation +
                # init_weights) costs ~0 host RAM; load_state_dict(assign=True) then
                # materializes each parameter exactly once from the mmap'd checkpoint.
                # Without this, every rank allocates a full ~34GB fp32 model AND loads the
                # ~58GB ckpt on top -> the 4 ranks/node OOM'd host RAM at load (job
                # 5355524). In a self-resume all MoT params are present, so the
                # init_*_embedding fallbacks below don't fire; only `freqs` (a plain
                # attribute, not in state_dict / not moved by .to()) is left on meta and
                # is rebuilt on a real device for forward's self.freqs.to(device).
                with torch.device("meta"):
                    self.model = WanModel.from_config(
                        WanModel._dict_from_json_file(config_path)
                    )
                # Ordering preserved exactly as the real-device path: the bf16 cast on a
                # meta model only sets param dtype metadata (no data), and is overwritten
                # for every loaded key by assign=True anyway — kept here so behavior is
                # identical for is_inference.
                if self.is_inference:
                    self.model.to(torch.bfloat16)
                self.model.load_state_dict(loaded_state_dict, assign=True, strict=False)
                if "patch_embedding_ray.weight" not in loaded_state_dict:
                    self.model.init_ray_embedding_from_scratch()
                if "patch_embedding_depth.weight" not in loaded_state_dict:
                    self.model.init_depth_embedding_from_rgb()
                # freqs is a plain attribute (not a Parameter/registered buffer): not in
                # state_dict and not moved by .to(), so meta-init leaves it on meta.
                # Rebuild it on a real device — same code/values as __init__ produced.
                self.model.freqs = self.model._build_freqs()
            else:
                base_model = BaseWanModel.from_config(WanModel._dict_from_json_file(config_path))
                base_model.load_state_dict(loaded_state_dict, assign=True, strict=False)
                self.model = WanModel.from_config(base_model.config)
                self.model.load_from_t2v_pretrained(base_model)

        expected_ray_ch = 2 * self.vae_stride[0] * 3
        assert self.model.patch_embedding_ray.in_channels == expected_ray_ch, (
            f"patch_embedding_ray expects {self.model.patch_embedding_ray.in_channels} "
            f"in-channels but the fused ray block has {expected_ray_ch} "
            f"(= 2 * vae_stride[0]({self.vae_stride[0]}) * 3). Set model.ray_in_dim accordingly."
        )

        if not self.is_inference:
            self.model.to(self.dtype).train()
        if self.gradient_checkpointing_rate > 0:
            self.model.gradient_checkpointing_enable(p=self.gradient_checkpointing_rate)
        if self.cfg.model.compile:
            self.model = torch.compile(self.model)

        self.training_scheduler, self.training_timesteps = self.build_scheduler(True)

    def configure_optimizers(self):
        # 8-bit AdamW (bitsandbytes): uint8 m/v optimizer states instead of fp32, ~4x
        # smaller optimizer memory. Same hyperparameters as the fp32 AdamW used by the
        # 1.3B concat algorithm; the 8-bit states are dequantized on-the-fly per step.
        optimizer = bnb.optim.AdamW8bit(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
        )
        lr_scheduler_config = {
            "scheduler": get_scheduler(optimizer=optimizer, **self.cfg.lr_scheduler),
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def _load_tuned_state_dict(self, prefix="model."):
        ckpt = torch.load(
            self.cfg.model.tuned_ckpt_path,
            mmap=True,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = {
            k[len(prefix):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith(prefix)
        }
        del ckpt
        gc.collect()
        return state_dict

    def build_scheduler(self, is_training=True):
        if self.sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=self.sample_shift,
                use_dynamic_shifting=False,
            )
            if not is_training:
                scheduler.set_timesteps(self.sample_steps, device=self.device, shift=self.sample_shift)
            timesteps = scheduler.timesteps
        elif self.sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=self.sample_shift,
                use_dynamic_shifting=False,
            )
            if not is_training:
                sampling_sigmas = get_sampling_sigmas(self.sample_steps, self.sample_shift)
                timesteps, _ = retrieve_timesteps(scheduler, device=self.device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return scheduler, timesteps

    def encode_text(self, texts):
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]

    def encode_video(self, videos):
        """videos: [B, C, T, H, W]"""
        return self.vae.encode(videos, self.vae_scale)

    def decode_video(self, zs):
        return self.vae.decode(zs, self.vae_scale).clamp_(-1, 1)

    def clone_batch(self, batch):
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _pack_ray_latent(self, rays):
        "Bypass-VAE encode: spatially downsample"
        B, T_in, C, H, W = rays.shape
        # Rays bypass the VAE, so (unlike RGB/depth) they aren't auto-resampled to the
        # latent grid. The dataset emits raymaps on the 1.3B 8x grid (e.g. 30x40); the
        # v2 (5B) VAE latent grid is 16x (e.g. 15x20). Resample to (lat_h, lat_w) so the
        # packed ray latent aligns spatially with the RGB latent / zero-pad block. Both
        # the history input and the flow-matching target come from here, so this stays
        # self-consistent; _unpack_ray_latent upsamples back to full res for viz.
        if H != self.lat_h or W != self.lat_w:
            rays = F.interpolate(
                rays.reshape(B * T_in, C, H, W),
                size=(self.lat_h, self.lat_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, T_in, C, self.lat_h, self.lat_w)
        T_lat = self.lat_t // 4
        stride = self.vae_stride[0]  # typically 4
        r = rays

        is_flf = (self.diffusion_forcing.enabled and self.diffusion_forcing.mode in ("ray_prediction_firstlast", "ray_depth_prediction_firstlast"))
        slots = [] 

        if is_flf: 
            # this is the first frame repeated stride times. stride is 4. 
            slots.append(torch.cat([r[:, 0]] * stride, dim=1))
            # interior slots: stride consecutive frames each 
            for i in range(1, T_lat - 1): 
                # all the middle frames 
                start = 1 + (i - 1) * stride
                group = [r[:, min(start + j, T_in - 1)] for j in range(stride)]
                slots.append(torch.cat(group, dim=1))

            slots.append(torch.cat([r[:, -1]] * stride, dim=1))
        else: 
            for i in range(T_lat): 
                group = [r[:, min(i * stride + j, T_in - 1)] for j in range(stride)]
                slots.append(torch.cat(group, dim=1))

        packed = torch.stack(slots, dim=2)
        pad = packed.new_zeros(B, self.lat_c - stride * C, T_lat, self.lat_h, self.lat_w)
        return torch.cat([packed, pad], dim=1)

    def _unpack_ray_latent(self, ray_lat, upsample_spatial=True): 
        """Bypass-VAE decode: 
        ray_lat: [B, lat_c, T_lat, lat_h, lat_w] eg. [B, 16, 8, 30, 40]
        8 is the compressed temporal dimension from 26
        returns: [B, 3, n_frames, H, W]
        """
        B = ray_lat.shape[0]
        stride = self.vae_stride[0]
        C = 3
        real = ray_lat[:, :stride * C]

        # [B, 4, 3, 8, 30, 40]
        per = real.view(B, stride, C, -1, self.lat_h, self.lat_w) 
        r = per.mean(dim=1) # [B, 3, 8, 30, 40]
        # 
        if upsample_spatial: 
            r = rearrange(r, "b c t h w -> (b t) c h w")
            r = F.interpolate(r, size=(self.height, self.width), mode="bilinear", align_corners=False)
            r = rearrange(r, "(b t) c h w -> b c t h w ", b = B)
        
        # temporaly upsample to n_frames 
        r = F.interpolate(
            r, 
            size=(self.n_frames, r.shape[-2], r.shape[-1]),
            mode="trilinear", 
            align_corners=False,
        )
        return r

    @torch.no_grad()
    def prepare_embeds(self, batch):
        videos  = batch["videos"].to(device=self.device, dtype=self.dtype, non_blocking=True)
        raymaps = batch["raymaps"].to(device=self.device, dtype=self.dtype, non_blocking=True)
        depths  = batch["depths"].to(device=self.device, dtype=self.dtype, non_blocking=True)
        prompts = batch["prompts"]

        # Per-sample modality presence flags from MixtureDataset._standardize_sample.
        # Default to all-True if absent (e.g. legacy single-dataset runs).
        B = videos.shape[0]
        has_raymap = batch.get("has_raymap")
        has_depth  = batch.get("has_depth")
        if has_raymap is None:
            has_raymap = torch.ones(B, dtype=torch.bool, device=self.device)
        else:
            has_raymap = has_raymap.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if has_depth is None:
            has_depth = torch.ones(B, dtype=torch.bool, device=self.device)
        else:
            has_depth = has_depth.to(device=self.device, dtype=torch.bool, non_blocking=True)
        batch["has_raymap"] = has_raymap
        batch["has_depth"]  = has_depth

        batch["videos"]  = videos
        batch["raymaps"] = raymaps
        batch["depths"]  = depths

        batch_size, t, _, h, w = videos.shape
        if t != self.n_frames:
            raise ValueError(f"Expected {self.n_frames} frames, got {t}")
        if h != self.height or w != self.width:
            raise ValueError(f"Expected {self.height}x{self.width}, got {h}x{w}")

        if not self.cfg.load_prompt_embed:
            prompt_embeds = self.encode_text(prompts)
        elif isinstance(batch["prompt_embeds"], list):
            prompt_embeds = batch["prompt_embeds"]  # already processed
        else:
            prompt_embeds = batch["prompt_embeds"].to(self.dtype)
            prompt_embed_len = batch["prompt_embed_len"]
            prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, prompt_embed_len)]

        # Encode RGB
        video_lat = self.encode_video(rearrange(videos, "b t c h w -> b c t h w"))

        # Split raymaps into ray_d (ch 0-2) and ray_m (ch 3-5)
        rays_d = raymaps[:, :, :3, :, :]  # [B, T, 3, H, W]
        rays_m = raymaps[:, :, 3:, :, :]  # [B, T, 3, H, W]

        ray_d_lat = self._pack_ray_latent(rays_d)
        ray_m_lat = self._pack_ray_latent(rays_m)

        depths_3ch = depths.repeat(1,1,3,1,1) 
        depth_lat = self.encode_video(rearrange(depths_3ch, "b t c h w -> b c t h w"))
        # done 

        T_lat = video_lat.shape[2]
        # modality_mask: 0=RGB, 1=fused ray, 2=depth. Internal layout the model
        # sees is 3*T_lat frames (ray_d and ray_m are channel-concat fused into
        # one block right before model.forward). video_lat below keeps the 4*T
        # layout because diffusion-forcing noise scheduling is per ray_d/ray_m.
        modality_mask = torch.cat([
            torch.zeros (batch_size, T_lat,     device=self.device, dtype=torch.long),
            torch.ones  (batch_size, T_lat,     device=self.device, dtype=torch.long),
            torch.full  ((batch_size, T_lat), 2, device=self.device, dtype=torch.long),
        ], dim=1)  # [B, 3*T_lat]

        # Concatenate all latents along time: [rgb | ray_d | ray_m | depth]
        video_lat = torch.cat([video_lat, ray_d_lat, ray_m_lat, depth_lat], dim=2)  # [B, C, 4*T_lat, H, W]
        # done 
        assert video_lat.shape[2] == self.lat_t, \
            f"Expected lat_t={self.lat_t}, got {video_lat.shape[2]}"

        batch["prompt_embeds"]  = prompt_embeds
        batch["video_lat"]      = video_lat
        batch["modality_mask"]  = modality_mask
        batch["image_embeds"]   = None
        batch["clip_embeds"]    = None
        return batch

    def add_training_noise(self, video_lat, batch=None):
        b, _, f = video_lat.shape[:3]
        device = video_lat.device
        if self.diffusion_type == "discrete":
            video_lat = rearrange(video_lat, "b c f h w -> (b f) c h w")
            noise = torch.randn_like(video_lat)
            timesteps = self.num_train_timesteps
            if self.diffusion_forcing.enabled:
                match self.diffusion_forcing.mode:
                    case "independent":
                        t = np.random.randint(timesteps, size=(b, f))
                        if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                            t[:, 0] = timesteps - 1
                    case "rand_history":
                        # currently we aim to support two history lengths, 1 and 6
                        possible_hist_lengths = [1, 2, 3, 4, 5, 6]
                        hist_length_probs = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
                        t = np.zeros((b, f), dtype=np.int64)
                        for i in range(b):
                            hist_len_idx = np.random.choice(
                                len(possible_hist_lengths), p=hist_length_probs
                            )
                            hist_len = possible_hist_lengths[hist_len_idx]
                            history_t = np.random.randint(timesteps)
                            future_t = np.random.randint(timesteps)
                            t[i, :hist_len] = history_t
                            t[i, hist_len:] = future_t
                            if (
                                np.random.rand()
                                < self.diffusion_forcing.clean_hist_prob
                            ):
                                t[i, :hist_len] = timesteps - 1
                    case "depth_prediction": 
                        # currently we aim to support two history lengths, 1 and 6
                        possible_hist_lengths = [1, 4]
                        hist_length_probs = [0.5, 0.5]
                        t = np.zeros((b, f), dtype=np.int64)
                        for i in range(b):
                            hist_len_idx = np.random.choice(
                                len(possible_hist_lengths), p=hist_length_probs
                            )
                            hist_len = possible_hist_lengths[hist_len_idx]
                            history_t = np.random.randint(timesteps)
                            future_t = np.random.randint(timesteps)
                            t[i, :hist_len] = history_t
                            t[i, hist_len:] = future_t
                            if (
                                np.random.rand()
                                < self.diffusion_forcing.clean_hist_prob
                            ):
                                t[i, :hist_len] = timesteps - 1
                t = self.training_timesteps[t.flatten()].reshape(b, f)
                t_expanded = t.flatten()
            else:
                t = np.random.randint(timesteps, size=(b,))
                t_expanded = repeat(t, "b -> (b f)", f=f)
                t = self.training_timesteps[t]
                t_expanded = self.training_timesteps[t_expanded]

            noisy_lat = self.training_scheduler.add_noise(video_lat, noise, t_expanded)
            noisy_lat = rearrange(noisy_lat, "(b f) c h w -> b c f h w", b=b, f=f)
            noise = rearrange(noise, "(b f) c h w -> b c f h w", b=b, f=f)
        elif self.diffusion_type == "continuous":
            # continious time steps.
            # 1. first sample t ~ U[0, 1]
            # 2. shift t with equation: t = t * self.sample_shift / (1 + (self.sample_shift - 1) * t)
            # 3. expand t to [b, 1/f, 1, 1, 1]
            # 4. compute noisy_lat = video_lat * (1.0 - t_expanded) + noise * t_expanded
            # 5. scale t to [0, num_train_timesteps]
            # returns:
            #  t is in [0, num_train_timesteps] of shape [b, f] or [b,], of dtype torch.float32
            # video_lat is shape [b, c, f, h, w]
            # noise is shape [b, c, f, h, w]
            dist = torch.distributions.uniform.Uniform(0, 1)
            noise = torch.randn_like(video_lat)  # [b, c, f, h, w]

            if self.diffusion_forcing.enabled:
                match self.diffusion_forcing.mode:
                    case "independent":
                        t = dist.sample((b, f)).to(device)
                        if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                            t[:, 0] = 0.0
                    case "ray_prediction":
                        # currently we aim to support two history lengths, 1 and 6
                        possible_hist_lengths = [1, 4]
                        hist_length_probs = [0.5, 0.5]
                        t = np.zeros((b, f), dtype=np.float32)
                        for i in range(b):
                            hist_len_idx = np.random.choice(
                                len(possible_hist_lengths), p=hist_length_probs
                            )
                            hist_len = possible_hist_lengths[hist_len_idx]
                            history_t = np.random.uniform(0, 1)
                            future_t = np.random.uniform(0, 1)
                            t[i, :hist_len] = history_t
                            t[i, hist_len:] = future_t
                            if (
                                np.random.rand()
                                < self.diffusion_forcing.clean_hist_prob
                            ):
                                t[i, :hist_len] = 0


                        # cast dtype of t
                        t = torch.from_numpy(t).to(device)
                        t = t.float()
                    case "ray_prediction_firstlast":
                        # 4 settings sampled uniformly:
                        #   0  first RGB history, rest (remaining RGB + rays) future
                        #   1  all RGB history, rays future
                        #   2  first + last RGB history, all rays history, middle RGB future
                        #        (matches inference_single_gpu_ray2rgb_firstlast_eval)
                        #   3  first RGB history, all rays history, remaining RGB future
                        #        (matches inference_single_gpu_ray2rgb_eval)
                        T = f // 3  # per-modality latent frames
                        t = np.zeros((b, f), dtype=np.float32)
                        for i in range(b):
                            # setting = np.random.choice(4, p=[0.01, 0.13, 0.13, 0.73])
                            setting = np.random.choice(4, p=[0.22, 0.22, 0.35, 0.21])
                            future_t = np.random.uniform(0, 1)

                            if setting == 0:
                                # first RGB frame as history, everything else future
                                history_t = np.random.uniform(0, 1)
                                t[i, :] = future_t
                                t[i, 0] = history_t
                                if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                                    t[i, 0] = 0

                            elif setting == 1:
                                # all RGB frames as history, rays future
                                history_t = np.random.uniform(0, 1)
                                t[i, :] = future_t
                                t[i, :T] = history_t
                                if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                                    t[i, :T] = 0

                            elif setting == 2:
                                # first + last RGB history, all rays history, middle RGB future
                                history_t = np.random.uniform(0, 1)
                                t[i, :] = future_t
                                t[i, 0] = history_t      # first RGB
                                t[i, T - 1] = history_t  # last RGB
                                t[i, T:] = history_t     # all rays
                                if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                                    t[i, 0] = 0
                                    t[i, T - 1] = 0
                                    t[i, T:] = 0

                            else:  # setting == 3
                                # first RGB history, all rays history, remaining RGB future
                                history_t = np.random.uniform(0, 1)
                                t[i, :] = future_t
                                t[i, 0] = history_t   # first RGB
                                t[i, T:] = history_t  # all rays
                                if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                                    t[i, 0] = 0
                                    t[i, T:] = 0

                        t = torch.from_numpy(t).to(device)
                        t = t.float()
                    case "ray_depth_prediction_firstlast":
                        # 5 settings; per-batch eligibility restricted by subset_id
                        # (looked up in self.diffusion_forcing.subset_settings).
                        #   0  first RGB history, rest (remaining RGB + rays + depth) future
                        #   1  all RGB history, rays + depth future
                        #   2  first + last RGB history, all rays history, middle RGB+depth future
                        #   3  first RGB history, all rays history, remaining RGB+depth future
                        #   4  first RGB history, all depth history, remaining RGB+pose future
                        if batch is None or "subset_id" not in batch:
                            raise KeyError(
                                "ray_depth_prediction_firstlast requires batch['subset_id']"
                            )
                        subset_settings = self.diffusion_forcing.subset_settings
                        DEFAULT_P = np.array([0.10, 0.37, 0.38, 0.10, 0.05])
                        T = f // 4  # per-modality latent frames
                        t = np.zeros((b, f), dtype=np.float32)
                        for i in range(b):
                            subset_name = batch["subset_id"][i]
                            if subset_name not in subset_settings:
                                raise KeyError(
                                    f"diffusion_forcing.subset_settings has no entry for "
                                    f"subset_id={subset_name!r}; add it to the algorithm yaml"
                                )
                            allowed = list(subset_settings[subset_name])
                            p = DEFAULT_P[allowed]
                            p = p / p.sum()
                            setting = int(np.random.choice(allowed, p=p))
                            future_t = np.random.uniform(0, 1)

                            if setting == 0:
                                # first RGB frame as history, everything else future
                                history_t = np.random.uniform(0, 1)
                                t[i, :] = future_t
                                t[i, 0] = history_t
                                if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                                    t[i, 0] = 0

                            elif setting == 1:
                                # all RGB frames as history, rays and depth future
                                history_t = np.random.uniform(0, 1)
                                t[i, :] = future_t
                                t[i, :T] = history_t
                                if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                                    t[i, :T] = 0

                            elif setting == 2:
                                # first + last RGB history, all rays history, middle RGB and depth future
                                history_t = np.random.uniform(0, 1)
                                t[i, :] = future_t
                                t[i, 0] = history_t       # first RGB
                                t[i, T - 1] = history_t   # last RGB
                                t[i, T:3*T] = history_t   # all rays (ray_d + ray_m)
                                if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                                    t[i, 0] = 0
                                    t[i, T - 1] = 0
                                    t[i, T:3*T] = 0

                            elif setting == 3:  # setting == 3
                                # first RGB history, all rays history, remaining RGB and depth future
                                history_t = np.random.uniform(0, 1)
                                t[i, :] = future_t
                                t[i, 0] = history_t        # first RGB
                                t[i, T:3*T] = history_t    # all rays (ray_d + ray_m)
                                if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                                    t[i, 0] = 0
                                    t[i, T:3*T] = 0
                            else: 
                                # first RGB history, all depth history, remaining RGB and pose future
                                history_t = np.random.uniform(0, 1)
                                t[i, :] = future_t
                                t[i, 0] = history_t        # first RGB
                                t[i, 3*T:] = history_t     # all depth
                                if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                                    t[i, 0] = 0
                                    t[i, 3*T:] = 0

                            # Depth tokens always predicted — never conditioned
                            if setting != 4:
                                t[i, 3*T:] = future_t

                        t = torch.from_numpy(t).to(device)
                        t = t.float()

                        # Missing-modality enforcement: clamp t to 1.0 (full
                        # noise) for raymap/depth blocks of samples whose source
                        # subset doesn't provide that modality. This guarantees
                        # the noisy input at those tokens is pure Gaussian
                        # (noise * 1 + placeholder * 0), regardless of which
                        # setting was drawn or whether the setting put that
                        # modality in "history" or "future". Prevents the
                        # placeholder pattern from leaking into the other
                        # modalities' attention. The training_step's loss
                        # masking takes care of suppressing the spurious
                        # gradient on these tokens.
                        if batch is not None:
                            has_raymap = batch.get("has_raymap")
                            has_depth  = batch.get("has_depth")
                            if has_raymap is not None:
                                missing_ray = (~has_raymap.bool()).to(device).view(b, 1).float()
                                t[:, T:3*T] = t[:, T:3*T] * (1.0 - missing_ray) + missing_ray
                            if has_depth is not None:
                                missing_depth = (~has_depth.bool()).to(device).view(b, 1).float()
                                t[:, 3*T:]   = t[:, 3*T:]   * (1.0 - missing_depth) + missing_depth
                    case "rand_history":
                        # currently we aim to support two history lengths, 1 and 6
                        possible_hist_lengths = [1, 2, 3, 4, 5, 6]
                        hist_length_probs = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
                        t = np.zeros((b, f), dtype=np.float32)
                        for i in range(b):
                            hist_len_idx = np.random.choice(
                                len(possible_hist_lengths), p=hist_length_probs
                            )
                            hist_len = possible_hist_lengths[hist_len_idx]
                            history_t = np.random.uniform(0, 1)
                            future_t = np.random.uniform(0, 1)
                            t[i, :hist_len] = history_t
                            t[i, hist_len:] = future_t
                            if (
                                np.random.rand()
                                < self.diffusion_forcing.clean_hist_prob
                            ):
                                t[i, :hist_len] = 0

                        # cast dtype of t
                        t = torch.from_numpy(t).to(device)
                        t = t.float()
                # t is [b, f] in range [0, 1] or dtype torch.float32  0 indicates clean.
                t = t * self.sample_shift / (1 + (self.sample_shift - 1) * t)
                t_expanded = (
                    t.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # [b, f] -> [b, 1, f, 1, 1]

                # [b, c, f, h, w] * [b, 1, f, 1, 1] + [b, c, f, h, w] * [b, 1, f, 1, 1]
                noisy_lat = video_lat * (1.0 - t_expanded) + noise * t_expanded
                t = t * self.num_train_timesteps  # [b, f] -> [b, f]
                # now t is in [0, num_train_timesteps] of shape [b, f]
            else:
                t = dist.sample((b,)).to(device)
                t = t * self.sample_shift / (1 + (self.sample_shift - 1) * t)
                t_expanded = t.view(-1, 1, 1, 1, 1)

                noisy_lat = video_lat * (1.0 - t_expanded) + noise * t_expanded
                t = t * self.num_train_timesteps  # [b,]
                # now t is in [0, num_train_timesteps] of shape [b,]
        else:
            raise NotImplementedError("Unsupported time step type.")

        return noisy_lat, noise, t

    def remove_noise(self, flow_pred, t, video_pred_lat):
        b, _, f = video_pred_lat.shape[:3]
        video_pred_lat = rearrange(video_pred_lat, "b c f h w -> (b f) c h w")
        flow_pred = rearrange(flow_pred, "b c f h w -> (b f) c h w")
        if t.ndim == 1:
            t = repeat(t, "b -> (b f)", f=f)
        elif t.ndim == 2:
            t = t.flatten()
        video_pred_lat = self.inference_scheduler.step(
            flow_pred, t, video_pred_lat, return_dict=False
        )[0]
        video_pred_lat = rearrange(video_pred_lat, "(b f) c h w -> b c f h w", b=b)
        return video_pred_lat

    def training_step(self, batch, batch_idx=None):
        batch = self.prepare_embeds(batch)
        prompt_embeds   = batch["prompt_embeds"]
        modality_mask   = batch["modality_mask"]
        video_lat       = batch["video_lat"]
        clip_embeds     = batch["clip_embeds"]
        image_embeds    = batch["image_embeds"]

        noisy_lat, noise, t = self.add_training_noise(video_lat, batch=batch)
        flow = noise - video_lat

        assert noisy_lat.shape[2] == self.lat_t, \
            f"noisy_lat temporal dim should be {self.lat_t}, got {noisy_lat.shape[2]}"

        T = self.lat_t // 4
        video_noise_lat = noisy_lat[:, :, :T]        # RGB [B, C, T, H, W]
        # Channel-concat ray_d and ray_m into one fused block, keeping only the real
        # packed channels (strip the zero padding that exists in the shared latent so
        # RGB/ray/depth share a uniform channel dim). Fused block: [B, 2*ray_ch, T, H, W]
        ray_ch = self.vae_stride[0] * 3              # stride frames × 3 Plücker channels
        ray_noise_lat = torch.cat([
            noisy_lat[:, :ray_ch, T:2*T],
            noisy_lat[:, :ray_ch, 2*T:3*T],
        ], dim=1)
        depth_noise_lat = noisy_lat[:, :, 3*T:]      # depth [B, C, T, H, W]

        # Slice t from 4T to 3T (rgb | ray_d (== ray_m in firstlast modes) | depth)
        t_model = torch.cat([t[:, :T], t[:, T:2*T], t[:, 3*T:]], dim=1) if t.ndim == 2 else t

        flow_pred = self.model(
            video_noise_lat,
            ray_noise_lat,
            depth_noise_lat,
            modality_mask,
            t=t_model,
            context=prompt_embeds,
            clip_fea=clip_embeds,
            seq_len=self.max_tokens,
            y=image_embeds,
        )

        # Split predictions and targets. Ray modalities are masked to the real
        # packed channels [:ray_ch]; channels [ray_ch:] are the zero-pad slots
        # (carried only so rays fit the shared lat_c-channel latent) whose flow
        # target is pure noise — supervising them wastes capacity and dilutes the
        # ray loss. RGB/depth are real VAE latents and use all channels.
        flow_rgb_pred   = flow_pred[:, :, :T]
        flow_ray_d_pred = flow_pred[:, :ray_ch, T:2*T]
        flow_ray_m_pred = flow_pred[:, :ray_ch, 2*T:3*T]
        flow_depth_pred = flow_pred[:, :, 3*T:]
        flow_rgb_target   = flow[:, :, :T]
        flow_ray_d_target = flow[:, :ray_ch, T:2*T]
        flow_ray_m_target = flow[:, :ray_ch, 2*T:3*T]
        flow_depth_target = flow[:, :, 3*T:]

        # Per-sample MSE so we can mask out modalities missing from each sample.
        # `mean(dim=(1,2,3,4))` matches `mse_loss`'s overall scale on a
        # full-modality batch — keeps logged loss curves comparable across the
        # missing-modality refactor.
        def _per_sample_mse(pred, target):
            return ((pred - target) ** 2).mean(dim=(1, 2, 3, 4))

        def _masked_mean(per_sample_loss, mask):
            mask_f = mask.to(per_sample_loss.dtype)
            denom = mask_f.sum().clamp(min=1.0)
            return (per_sample_loss * mask_f).sum() / denom

        # Clean tokens are CONTEXT, not targets. Diffusion forcing gives every token
        # its own noise level; a token at t == 0 has noisy_lat == video_lat exactly, so
        # its flow target (noise - video_lat) still carries the noise draw, which is
        # unobservable from the input. Supervising it adds irreducible gradient variance
        # -- the same reason the ray zero-pad channels are excluded just above.
        #
        # This is not a small effect. In setting 1 (all-RGB history -> rays+depth), a
        # `clean_hist_prob` hit puts EVERY RGB frame at t == 0, so that sample's whole
        # RGB term is noise, yet it still contributed 1/4 of total_loss.
        #
        # `t` is [B, 4T] over the packed blocks [rgb | ray_d | ray_m | depth]. Averaging
        # over (c,h,w) then taking a frame-weighted mean keeps the same scale as the old
        # mean(dim=(1,2,3,4)) whenever every frame is noised, so loss curves stay
        # comparable. Samples with no noised frame in a modality drop out of that term.
        #
        # Gated so the historical recipes stay bit-reproducible; see
        # diffusion_forcing.mask_clean_token_loss in the algorithm yaml.
        # The `t.ndim == 2` clause mirrors the same guard on t_model above: the 1-D
        # timestep path carries no per-frame information to mask with, so it falls back
        # to the unmasked loss instead of raising on `t[:, :T]`.
        mask_clean = (
            bool(self.cfg.diffusion_forcing.get("mask_clean_token_loss", True))
            and t.ndim == 2
        )

        def _mse_and_valid(pred, target, t_block):
            if not mask_clean:
                ones = torch.ones(
                    pred.shape[0], dtype=torch.bool, device=pred.device
                )
                return _per_sample_mse(pred, target), ones
            se = ((pred - target) ** 2).mean(dim=(1, 3, 4))      # [B, T_block]
            w = (t_block > 0).to(se.dtype)                        # [B, T_block]
            n = w.sum(dim=1)
            return (se * w).sum(dim=1) / n.clamp(min=1.0), n > 0

        has_raymap = batch["has_raymap"]
        has_depth  = batch["has_depth"]
        _t2 = t if t.ndim == 2 else None
        rgb_ps,  rgb_v  = _mse_and_valid(flow_rgb_pred,   flow_rgb_target,   None if _t2 is None else _t2[:, :T])
        rayd_ps, rayd_v = _mse_and_valid(flow_ray_d_pred, flow_ray_d_target, None if _t2 is None else _t2[:, T:2*T])
        raym_ps, raym_v = _mse_and_valid(flow_ray_m_pred, flow_ray_m_target, None if _t2 is None else _t2[:, 2*T:3*T])
        dep_ps,  dep_v  = _mse_and_valid(flow_depth_pred, flow_depth_target, None if _t2 is None else _t2[:, 3*T:])
        rgb_loss   = _masked_mean(rgb_ps,  rgb_v)
        ray_d_loss = _masked_mean(rayd_ps, has_raymap.bool() & rayd_v)
        ray_m_loss = _masked_mean(raym_ps, has_raymap.bool() & raym_v)
        depth_loss = _masked_mean(dep_ps,  has_depth.bool()  & dep_v)
        # train_rgb (default True): the RGB stream is a supervised generative modality, so its
        # flow loss drives learning together with the geometry modalities (equal per-modality
        # weight). Required whenever subset_settings include RGB-future settings (0/2/3/4) —
        # otherwise the model is fed noised future RGB it is never taught to denoise.
        # Set train_rgb=False to reproduce the old RGB->geometry-only run (RGB = clean
        # conditioning, subset_settings=[1], no RGB gradient).
        train_rgb = bool(self.cfg.diffusion_forcing.get("train_rgb", True))
        if train_rgb:
            total_loss = (rgb_loss + ray_d_loss + ray_m_loss + depth_loss) / 4
        else:
            assert False, "not train_rgb is not supported"
            total_loss = (ray_d_loss + ray_m_loss + depth_loss) / 3

        if self.global_step % self.cfg.logging.loss_freq == 0:
            self.log("train/loss_total", total_loss, sync_dist=True)
            self.log("train/loss_rgb",   rgb_loss,   sync_dist=True)
            self.log("train/loss_ray_d", ray_d_loss, sync_dist=True)
            self.log("train/loss_ray_m", ray_m_loss, sync_dist=True)
            self.log("train/loss_depth", depth_loss, sync_dist=True)
            self.log("global_step", self.global_step, sync_dist=True)
            # pre-clip global grad norm from configure_gradient_clipping (previous
            # step's — the hook runs after this forward). Already the cross-shard
            # total on every rank, so no sync needed; costs nothing extra.
            gn = getattr(self, "_grad_norm_last", None)
            if gn is not None:
                self.log("train/grad_norm", gn, sync_dist=False)

        return total_loss

    def on_save_checkpoint(self, checkpoint) -> None:
        # The default 62GB rank-0 full-state write takes >10 min on the network FS, which stalls the
        # other ranks' next collective past the NCCL watchdog timeout -> job crash (killed job 24613168
        # at the step-1000 save). Shrink the write two ways: (1) drop the frozen text_encoder+VAE (~3GB,
        # reconstructed from pretrained every run); (2) cast the trained model weights fp32->bf16
        # (14.8B params: 59GB -> 30GB), halving the write to ~5 min. bf16 is fine: the model already
        # trains in bf16-mixed and the resume path (_load_tuned_state_dict, model.* only) casts back.
        sd = checkpoint.get("state_dict")
        if sd is not None:
            for k in list(sd.keys()):
                if k.startswith("text_encoder.") or k.startswith("vae."):
                    del sd[k]
                elif torch.is_tensor(sd[k]) and sd[k].is_floating_point() and sd[k].dtype != torch.bfloat16:
                    sd[k] = sd[k].to(torch.bfloat16)

    def on_train_start(self) -> None:
        # Reclaim GPU memory left fragmented by the checkpoint-load + FSDP shard. The
        # resume path builds on meta, assigns mmap'd fp32 weights, casts to compute dtype,
        # then FSDP flattens/shards them — leaving reserved-but-unallocated blocks that
        # otherwise sit on top of the steady-state footprint and shrink headroom for the
        # 24h run (and the checkpoint-save full-gather spike). Cheap and math-neutral.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from lightning.pytorch.callbacks import ModelCheckpoint
        checkpoint_dir = None
        if hasattr(self.trainer, "checkpoint_callbacks"):
            for cb in self.trainer.checkpoint_callbacks:
                if isinstance(cb, ModelCheckpoint) and getattr(cb, "dirpath", None):
                    checkpoint_dir = cb.dirpath
                    break
        if not checkpoint_dir:
            if getattr(self.logger, "save_dir", None):
                checkpoint_dir = os.path.join(self.logger.save_dir, "checkpoints")
            else:
                checkpoint_dir = os.path.join(self.trainer.default_root_dir, "checkpoints")
        try:
            global_rank = int(os.environ.get("RANK", "0"))
        except ValueError:
            global_rank = 0
        if global_rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.trainer.save_checkpoint(os.path.join(checkpoint_dir, "startup.ckpt"))

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        try:
            global_rank = int(os.environ.get("RANK", "0"))
        except ValueError:
            global_rank = 0
        self.log("epoch", float(self.current_epoch), sync_dist=True)
        if global_rank != 0:
            return
        from lightning.pytorch.callbacks import ModelCheckpoint
        checkpoint_dir = None
        if hasattr(self.trainer, "checkpoint_callbacks"):
            for cb in self.trainer.checkpoint_callbacks:
                if isinstance(cb, ModelCheckpoint) and getattr(cb, "dirpath", None):
                    checkpoint_dir = cb.dirpath
                    break
        if not checkpoint_dir:
            if getattr(self.logger, "save_dir", None):
                checkpoint_dir = os.path.join(self.logger.save_dir, "checkpoints")
            else:
                checkpoint_dir = os.path.join(self.trainer.default_root_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.trainer.save_checkpoint(
            os.path.join(checkpoint_dir, f"epoch={self.current_epoch:04d}.ckpt")
        )

    @torch.no_grad()
    def sample_seq(self, batch, hist_len=26, pbar=None, is_ray_hist=False, hist_indexes=None):
        """
        Denoising sampling loop for [rgb | ray_d | ray_m | depth].

        hist_len: number of pixel-space frames used as history conditioning.
        """
        hist_len = min(hist_len, self.n_frames)
        is_flf_mode = (self.diffusion_forcing.enabled and
                       self.diffusion_forcing.mode == "ray_depth_prediction_firstlast")
        one_frame_tokens = 2 if is_flf_mode else 1
        if (hist_len - one_frame_tokens) % self.vae_stride[0] != 0:
            raise ValueError(
                f"hist_len - {one_frame_tokens} must be a multiple of vae_stride[0]. "
                f"Got {hist_len} and vae stride {self.vae_stride[0]}"
            )
        hist_len = (hist_len - one_frame_tokens) // self.vae_stride[0] + one_frame_tokens  # latent-space history length

        self.inference_scheduler, self.inference_timesteps = self.build_scheduler(False)
        lang_guidance = self.lang_guidance if self.lang_guidance else 0
        hist_guidance = self.hist_guidance if self.hist_guidance else 0

        batch = self.prepare_embeds(batch)
        clip_embeds   = batch["clip_embeds"]
        image_embeds  = batch["image_embeds"]
        prompt_embeds = batch["prompt_embeds"]
        video_lat     = batch["video_lat"]
        modality_mask = batch["modality_mask"]

        batch_size = video_lat.shape[0]
        T = self.lat_t // 4  # frames per modality in latent space

        video_pred_lat = torch.randn_like(video_lat)

        if lang_guidance:
            neg_prompt_embeds = self.encode_text([self.neg_prompt] * len(batch["prompts"]))

        if pbar is None:
            pbar = tqdm(range(len(self.inference_timesteps)), desc="Sampling")

        if hist_indexes is not None:
            hist_indexes = hist_indexes  # caller-supplied
        elif is_flf_mode:
            # Condition on first + last frames of RGB and both ray streams; depth never conditioned
            hist_indexes = torch.tensor([0, T - 1, T, 2*T - 1, 2*T, 3*T - 1])
        elif not is_ray_hist:
            hist_indexes = torch.arange(hist_len)
        else:
            hist_indexes = torch.cat([
                torch.arange(T, T + hist_len),      # ray_d history frames
                torch.arange(2*T, 2*T + hist_len),  # ray_m history frames
            ])

        ray_ch = self.vae_stride[0] * 3              # stride frames × 3 Plücker channels
        def _call_model(pred_lat, ctx):
            ray_lat = torch.cat([
                pred_lat[:, :ray_ch, T:2*T],
                pred_lat[:, :ray_ch, 2*T:3*T],
            ], dim=1)  # fused ray block, real channels only: [B, 2*ray_ch, T, H, W]
            t_m = (torch.cat([t_expanded[:, :T], t_expanded[:, T:2*T], t_expanded[:, 3*T:]], dim=1)
                   if t_expanded.ndim == 2 else t_expanded)
            return self.model(
                pred_lat[:, :, :T],
                ray_lat,
                pred_lat[:, :, 3*T:],    # depth
                modality_mask,
                t=t_m,
                context=ctx,
                seq_len=self.max_tokens,
                clip_fea=clip_embeds,
                y=image_embeds,
            )

        for t in self.inference_timesteps:
            if self.diffusion_forcing.enabled:
                video_pred_lat[:, :, hist_indexes] = video_lat[:, :, hist_indexes]
                t_expanded = torch.full((batch_size, self.lat_t), t, device=self.device)
                t_expanded[:, hist_indexes] = self.inference_timesteps[-1]
            else:
                t_expanded = torch.full((batch_size,), t, device=self.device)

            flow_pred = _call_model(video_pred_lat, prompt_embeds)

            if lang_guidance:
                no_lang_flow_pred = _call_model(video_pred_lat, neg_prompt_embeds)
            else:
                no_lang_flow_pred = torch.zeros_like(flow_pred)

            if hist_guidance and self.diffusion_forcing.enabled:
                no_hist_lat = video_pred_lat.clone()
                no_hist_lat[:, :, hist_indexes] = torch.randn_like(no_hist_lat[:, :, hist_indexes])
                t_expanded[:, hist_indexes] = self.inference_timesteps[0]
                no_hist_flow_pred = _call_model(no_hist_lat, prompt_embeds)
            else:
                no_hist_flow_pred = torch.zeros_like(flow_pred)

            flow_pred = flow_pred * (1 + lang_guidance + hist_guidance)
            flow_pred = flow_pred - lang_guidance * no_lang_flow_pred - hist_guidance * no_hist_flow_pred

            video_pred_lat = self.remove_noise(flow_pred, t, video_pred_lat)
            pbar.update(1)

        if self.diffusion_forcing.enabled:
            video_pred_lat[:, :, hist_indexes] = video_lat[:, :, hist_indexes]

        # Decode all 4 modalities
        rgb_lat   = video_pred_lat[:, :, :T]
        ray_d_lat = video_pred_lat[:, :, T:2*T]
        ray_m_lat = video_pred_lat[:, :, 2*T:3*T]
        depth_lat = video_pred_lat[:, :, 3*T:]

        video_rgb   = self.decode_video(rgb_lat)
        video_ray_d = self._unpack_ray_latent(ray_d_lat)
        video_ray_m = self._unpack_ray_latent(ray_m_lat)
        video_depth = self.decode_video(depth_lat)

        # Concatenate along temporal dimension: [rgb | ray_d | ray_m | depth]
        video_pred = torch.cat([video_rgb, video_ray_d, video_ray_m, video_depth], dim=2)
        video_pred = rearrange(video_pred, "b c t h w -> b t c h w")
        return video_pred

    def validation_step(self, batch, batch_idx=None):
        T = self.lat_t // 4

        # Alternate between the two sampling tasks each validation call so
        # total validation time is halved. All ranks do the same task so
        # all_gather works correctly.
        val_every = self.trainer.val_check_interval if self.trainer is not None else 100
        do_rgb_to_raymap = (self.global_step // max(1, val_every)) % 2 == 0

        # Reroute: raymap_to_rgb_depth requires clean ray GT to feed in as
        # history. If the batch contains a sample without raymaps (e.g.
        # scenenet_depth from the mixture), the raymap "history" tokens would
        # be the randn placeholder — meaningless conditioning. Fall back to
        # rgb_to_raymap_depth, which is always feasible (every subset has RGB).
        has_raymap_all = bool(batch["has_raymap"].all().item())
        has_depth_all  = bool(batch["has_depth"].all().item())
        if not do_rgb_to_raymap and not has_raymap_all:
            do_rgb_to_raymap = True

        if do_rgb_to_raymap:
            hist_rgb_to_raymap = torch.arange(T)
            video_pred = self.sample_seq(batch, hist_indexes=hist_rgb_to_raymap)
            self.visualize(video_pred, batch, prefix="rgb_to_raymap_depth/",
                           compute_rgb_metrics=False,
                           compute_pose_metrics=has_raymap_all,
                           compute_depth_metrics=has_depth_all)
        else:
            hist_raymap_to_rgb = torch.cat([
                torch.tensor([0, T - 1]),       # first + last RGB
                torch.arange(T, 3 * T),         # all ray_d + ray_m
            ])
            video_pred = self.sample_seq(batch, hist_indexes=hist_raymap_to_rgb)
            self.visualize(video_pred, batch, prefix="raymap_to_rgb_depth/",
                           compute_rgb_metrics=True,
                           compute_pose_metrics=False,
                           compute_depth_metrics=has_depth_all)

    def visualize(self, video_pred, batch, prefix="", compute_rgb_metrics=False,
                  compute_pose_metrics=False, compute_depth_metrics=False):
        """
        video_pred: [B, 4T, 3, H, W] — [rgb | ray_d | ray_m | depth] frames
        """
        video_gt  = batch["videos"]              # [B, T, 3, H, W]
        raymaps   = batch.get("raymaps", None)   # [B, T, 6, H, W]
        depth_gt  = batch.get("depths", None)    # [B, T, 1, H, W]

        T = video_pred.shape[1] // 4

        pred_rgb   = video_pred[:, :T]        # [B, T, 3, H, W]
        pred_ray_d = video_pred[:, T:2*T]
        pred_ray_m = video_pred[:, 2*T:3*T]
        pred_depth = video_pred[:, 3*T:]

        if raymaps is not None:
            B_r, T_r = raymaps.shape[:2]
            H_pred, W_pred = pred_ray_d.shape[-2:]
            rays_d_gt = rearrange(raymaps[:, :, :3], "b t c h w -> (b t) c h w")
            rays_m_gt = rearrange(raymaps[:, :, 3:], "b t c h w -> (b t) c h w")
            if rays_d_gt.shape[-2] != H_pred or rays_d_gt.shape[-1] != W_pred:
                rays_d_gt = F.interpolate(rays_d_gt, size=(H_pred, W_pred), mode="bilinear", align_corners=False)
                rays_m_gt = F.interpolate(rays_m_gt, size=(H_pred, W_pred), mode="bilinear", align_corners=False)
            rays_d_gt = rearrange(rays_d_gt, "(b t) c h w -> b t c h w", b=B_r, t=T_r)
            rays_m_gt = rearrange(rays_m_gt, "(b t) c h w -> b t c h w", b=B_r, t=T_r)
            vis_rgb   = torch.cat([pred_rgb,   video_gt],  dim=-1)
            vis_ray_d = torch.cat([pred_ray_d, rays_d_gt], dim=-1)
            vis_ray_m = torch.cat([pred_ray_m, rays_m_gt], dim=-1)
        else:
            vis_rgb   = pred_rgb
            vis_ray_d = pred_ray_d
            vis_ray_m = pred_ray_m

        # When GT is valid, use its per-sample min/max as the shared normalization reference so
        # pred and GT are on the same scale. Otherwise (placeholder GT or no GT) fall back to each
        # tensor's own range so the prediction at least has a visible dynamic range.
        has_depth_t = batch.get("has_depth")
        gt_is_valid = (depth_gt is not None and has_depth_t is not None
                       and bool(has_depth_t.all().item()))
        if gt_is_valid:
            pred_depth_color = _depth_to_colormap(pred_depth, ref=depth_gt) * 2 - 1
            vis_depth = torch.cat([pred_depth_color, _depth_to_colormap(depth_gt) * 2 - 1], dim=-1)
        elif depth_gt is not None:
            pred_depth_color = _depth_to_colormap(pred_depth) * 2 - 1
            vis_depth = torch.cat([pred_depth_color, _depth_to_colormap(depth_gt) * 2 - 1], dim=-1)
        else:
            pred_depth_color = _depth_to_colormap(pred_depth) * 2 - 1
            vis_depth = pred_depth_color

        vis_depth = vis_depth.to(vis_rgb.device)
        vis = torch.cat([vis_rgb, vis_ray_d, vis_ray_m, vis_depth], dim=-1) * 0.5 + 0.5
        vis = vis.cpu()

        # gather for metrics and trajectory (keep cpu copies before gather for metrics)
        pred_rgb_cpu   = pred_rgb.cpu()
        pred_depth_cpu = pred_depth.cpu()
        video_gt_cpu   = video_gt.cpu() if isinstance(video_gt, torch.Tensor) else None
        depth_gt_cpu   = depth_gt.cpu() if depth_gt is not None else None

        if dist.is_available() and dist.is_initialized():
            vis = rearrange(self.all_gather(vis), "p b ... -> (p b) ...")
            pred_ray_d = rearrange(self.all_gather(pred_ray_d.cpu()), "p b ... -> (p b) ...")
            pred_ray_m = rearrange(self.all_gather(pred_ray_m.cpu()), "p b ... -> (p b) ...")
            pred_rgb_cpu   = rearrange(self.all_gather(pred_rgb_cpu),   "p b ... -> (p b) ...")
            pred_depth_cpu = rearrange(self.all_gather(pred_depth_cpu), "p b ... -> (p b) ...")
            if video_gt_cpu is not None:
                video_gt_cpu = rearrange(self.all_gather(video_gt_cpu), "p b ... -> (p b) ...")
            if depth_gt_cpu is not None:
                depth_gt_cpu = rearrange(self.all_gather(depth_gt_cpu), "p b ... -> (p b) ...")
            if raymaps is not None:
                raymaps = rearrange(self.all_gather(raymaps.cpu()), "p b ... -> (p b) ...")

        if is_rank_zero:
            # Per-subset binning: with val_bs=1, every batch contains samples
            # from a single subset (or a list of subset_ids if val_bs>1). We
            # log each metric twice: under `val_metrics/global/...` and under
            # `val_metrics/<subset_id>/...`. If multiple subsets coexist in
            # the batch, we only emit the per-subset variant when the entire
            # batch shares one subset_id (the common val_bs=1 case).
            subset_ids = batch.get("subset_id")
            if isinstance(subset_ids, str):
                subset_ids = [subset_ids]
            elif subset_ids is None:
                subset_ids = []
            unique_sid = subset_ids[0] if (subset_ids and all(s == subset_ids[0] for s in subset_ids)) else None

            def _log_metric(name: str, value):
                logs = {f"val_metrics/global/{prefix}{name}": value}
                if unique_sid is not None:
                    logs[f"val_metrics/{unique_sid}/{prefix}{name}"] = value
                self.logger.experiment.log(logs, step=self.global_step)

            # ── Validation panel video (per-subset + global) ─────────────────────
            self.log_video(
                f"validation_vis/global/{prefix}rgb_rayd_raym_depth",
                vis[:16],
                fps=self.cfg.logging.fps,
                step=self.global_step,
            )
            if unique_sid is not None:
                self.log_video(
                    f"validation_vis/{unique_sid}/{prefix}rgb_rayd_raym_depth",
                    vis[:16],
                    fps=self.cfg.logging.fps,
                    step=self.global_step,
                )

            # ── RGB quality metrics (PSNR) ─────────────────────────────────────
            if compute_rgb_metrics and video_gt_cpu is not None:
                psnr = _psnr_batch(pred_rgb_cpu, video_gt_cpu, skip_first_last=True)
                _log_metric("psnr", psnr)

            # ── Depth metrics ──────────────────────────────────────────────────
            # Gated on explicit flag (NOT on `depth_gt is not None`) because
            # MixtureDataset emits randn placeholders for missing depth GT —
            # the placeholder is not None and would silently contaminate the
            # delta metrics. Caller passes compute_depth_metrics=has_depth.all().
            if compute_depth_metrics and depth_gt_cpu is not None:
                dm = _depth_delta_batch(pred_depth_cpu, depth_gt_cpu)
                for k, v in dm.items():
                    _log_metric(k, v)

            # ── Camera pose metrics ─────────────────────────────────────────────
            if compute_pose_metrics and raymaps is not None:
                pm = _pose_metrics_batch(pred_ray_d, pred_ray_m, raymaps)
                for k, v in pm.items():
                    _log_metric(k, v)

            # ── Trajectory figure ───────────────────────────────────────────────
            if compute_pose_metrics and raymaps is not None:
                B_traj = min(pred_ray_d.shape[0], 16)
                traj_imgs = []
                for i in range(B_traj):
                    pred_raymaps_i = torch.cat([pred_ray_d[i], pred_ray_m[i]], dim=1)
                    traj_img = _make_trajectory_figure(pred_raymaps_i, raymaps[i])
                    if traj_img is not None:
                        traj_imgs.append(traj_img)
                if traj_imgs:
                    n = len(traj_imgs)
                    ncols = min(n, 4)
                    nrows = int(np.ceil(n / ncols))
                    w, h = traj_imgs[0].size
                    grid_img = Image.new("RGB", (ncols * w, nrows * h), color=(255, 255, 255))
                    for idx, img in enumerate(traj_imgs):
                        row, col = divmod(idx, ncols)
                        grid_img.paste(img, (col * w, row * h))
                    traj_logs = {f"validation_viz/global/{prefix}trajectory": wandb.Image(grid_img)}
                    if unique_sid is not None:
                        traj_logs[f"validation_viz/{unique_sid}/{prefix}trajectory"] = wandb.Image(grid_img)
                    self.logger.experiment.log(traj_logs, step=self.global_step)

    def maybe_reset_socket(self):
        if not self.socket:
            ctx = zmq.Context()
            socket = ctx.socket(zmq.ROUTER)
            socket.setsockopt(zmq.ROUTER_HANDOVER, 1)
            socket.bind(f"tcp://*:{self.cfg.serving.port}")
            self.socket = socket
            print(f"Server ready on port {self.cfg.serving.port}...")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if is_rank_zero:
            self.maybe_reset_socket()

        print(f"Waiting for request on local rank: {dist.get_rank()}")
        if is_rank_zero:
            ident, payload = self.socket.recv_multipart()
            request = msgpack.unpackb(payload, raw=False)
            print(f"Received request with prompt: {request['prompt']}")
            image_bytes = request["image"]
            prompt = request["prompt"]
            data_to_broadcast = [image_bytes, prompt]
        else:
            data_to_broadcast = [None, None]

        dist.broadcast_object_list(data_to_broadcast, src=0)
        image_bytes, prompt = data_to_broadcast
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.RandomResizedCrop(
                    size=(self.height, self.width),
                    scale=(1.0, 1.0),
                    ratio=(self.width / self.height, self.width / self.height),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            ]
        )
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(pil_image)
        batch["videos"][:, 0] = image[None]

        prompt_segments = prompt.split("<sep>")
        hist_len = 1
        videos = batch["videos"][:, :hist_len]
        for i, prompt in enumerate(prompt_segments):
            print(f"Generating task {i+1} out of {len(prompt_segments)} sub-tasks")
            batch["prompts"] = [prompt] * batch["videos"].shape[0]
            batch["videos"][:, :hist_len] = videos[:, -hist_len:]
            videos = torch.cat([videos, self.sample_seq(batch, hist_len)], dim=1)
            videos = torch.clamp(videos, -1, 1)
            hist_len = self.sliding_hist
        videos = rearrange(self.all_gather(videos), "p b t c h w -> (p b) t h w c")
        videos = videos.float().cpu().numpy()

        if is_rank_zero:
            videos = np.clip(videos * 0.5 + 0.5, 0, 1)
            videos = (videos * 255).astype(np.uint8)
            video_bytes_list = [
                numpy_to_mp4_bytes(video, fps=self.cfg.logging.fps) for video in videos
            ]
            reply = {"videos": video_bytes_list}
            self.socket.send_multipart([ident, msgpack.packb(reply)])
            print(f"Sent reply to {ident}")
            self.log_video(
                "test_vis/video_pred",
                rearrange(videos, "b t h w c -> b t c h w"),
                fps=self.cfg.logging.fps,
                caption="<sep>\n".join(prompt_segments),
            )
