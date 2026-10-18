"""Multiview Wan backbone for the RH20T multiview ablations.

One token sequence holds V views laid out view-major: [view][frame][h][w].
  - Every view gets the SAME (frame, h, w) RoPE positions, so attention is
    permutation-equivariant over views; views are told apart only by their camera.
  - Timestep modulation is per (view, frame), so diffusion forcing can give every
    latent frame of every view its own noise level (clean history, noisy future, ...).
The "tiled" design is the same model with V = 1 on the packed canvas.

Camera conditioning (cam_mode), all relative to view 0 (= identity):
  concat    Plucker raymap (6 ch) concatenated to the latent channels before the patch
            embedding; extra input weights zero-initialised (v1 baseline).
  cat4d     CAT3D/CAT4D input conditioning: a 6-ch raymap of ray ORIGIN + DIRECTION (not
            Plucker; built by the algorithm) + a binary "clean conditioning" mask channel
            concatenated to the latent, and CAT4D's time conditioning: each frame's
            normalised timestamp (0..1) -> sinusoidal embedding -> 2-layer MLP (last layer
            zero-init), added to the diffusion-timestep embedding.
  prope     PRoPE: per view a projective matrix P = K~ [R|t] (4x4); in every self-attention
            layer q <- D^T q, k <- D^-1 k, v <- D^-1 v, out <- D out on CAM_DIMS of every
            head, D = blockdiag(P, ..., P). Attention then depends on P_i P_j^-1 only.
  gta       GTA: same mechanism with the SE(3) world-to-camera matrix (no intrinsics).
  camtoken  each view's raymap is patch-embedded into its own h x w "camera tokens" that
            are appended to that view's frames and kept in the sequence through every
            block (attended to like a clean extra frame, never decoded).
Weights load 1:1 from Wan2.1-T2V via `MultiViewWanModel.from_t2v_pretrained`.
"""
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import flash_attention
from .model import WanModel, WanSelfAttention, sinusoidal_embedding_1d

CAM_MODES = ("concat", "cat4d", "prope", "gta", "camtoken")
# Head dims carrying the camera transform for prope / gta: complex pairs 6..21 of the
# temporal RoPE band (dims 12..43, 8 blocks of 4). These are its lowest frequencies, which
# barely rotate over our <= 8 latent frames; temporal pairs 0..5 and all spatial pairs keep
# RoPE, so every head still encodes (frame, h, w).
CAM_PAIRS = slice(6, 22)
CAM_DIMS = slice(12, 44)


def rope_freqs_for_views(freqs, n_views, f, h, w, extra_frames=0):
    """Per-token complex RoPE multipliers [V*(f+extra)*h*w, 1, C/2] for a view-major
    layout where every view repeats the same (f, h, w) grid. `extra_frames` camera-token
    frames per view reuse frame position 0 (they are aligned with the view's pixels)."""
    c = freqs.shape[1]
    fq = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    fi = torch.cat([torch.arange(f), torch.zeros(extra_frames, dtype=torch.long)]).to(freqs.device)
    ff = f + extra_frames
    one = torch.cat([
        fq[0][fi].view(ff, 1, 1, -1).expand(ff, h, w, -1),
        fq[1][:h].view(1, h, 1, -1).expand(ff, h, w, -1),
        fq[2][:w].view(1, 1, w, -1).expand(ff, h, w, -1),
    ], dim=-1).reshape(ff * h * w, 1, -1)
    return one.repeat(n_views, 1, 1)


def apply_rope(x, tok_freqs):
    """x: [B, L, n, d]; tok_freqs: [L, 1, d/2] complex. Same rotation as model.rope_apply,
    but in complex64 instead of complex128: positions here are < 64, where fp32 angles are
    exact to ~1e-6, and complex128 at V*F*h*w tokens per sample OOMs the tokens design."""
    b, L, n, d = x.shape
    xc = torch.view_as_complex(x.float().reshape(b, L, n, -1, 2))
    return torch.view_as_real(xc * tok_freqs.to(torch.complex64).unsqueeze(0)).flatten(3).type_as(x)


def cam_apply(x, D):
    """Apply per-token 4x4 matrices to CAM_DIMS of every head: x [B, L, n, d], D [B, L, 4, 4]
    acting on column 4-vectors (y = D x for each 4-block)."""
    xc = x[..., CAM_DIMS].float().unflatten(-1, (-1, 4))                 # B, L, n, 8, 4
    y = torch.einsum("blij,blnkj->blnki", D.float(), xc).flatten(-2)
    out = x.clone()
    out[..., CAM_DIMS] = y.type_as(x)
    return out


class MultiViewSelfAttention(WanSelfAttention):
    """WanSelfAttention with RoPE precomputed per token and an optional relative camera
    transform (prope / gta). `freqs` = (tok_freqs [L, 1, d/2], cam) where cam is None or
    (D^T, D^-1, D) each [B, L, 4, 4]; passed as a block argument (not module state) so
    gradient-checkpoint recomputation sees exactly the same inputs.
    Parameter names are unchanged, so pretrained weights load directly."""

    def forward(self, x, seq_lens, grid_sizes, freqs):
        freqs, cam = freqs
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = apply_rope(self.norm_q(self.q(x)).view(b, s, n, d), freqs)
        k = apply_rope(self.norm_k(self.k(x)).view(b, s, n, d), freqs)
        v = self.v(x).view(b, s, n, d)
        if cam is not None:
            DT, Dinv, Dfwd = cam
            q, k, v = cam_apply(q, DT), cam_apply(k, Dinv), cam_apply(v, Dinv)
        x = flash_attention(q=q, k=k, v=v, k_lens=seq_lens, window_size=self.window_size)
        if cam is not None:
            x = cam_apply(x, Dfwd)
        return self.o(x.flatten(2))


class MultiViewWanModel(WanModel):

    def __init__(self, *args, cam_mode="concat", cond_dim=6, **kwargs):
        super().__init__(*args, **kwargs)
        assert cam_mode in CAM_MODES, cam_mode
        self.cam_mode = cam_mode
        self.cond_dim = cond_dim
        self.extra_in = {"concat": cond_dim, "cat4d": cond_dim + 1}.get(cam_mode, 0)
        # swap self-attention for the per-token-RoPE variant (same parameters)
        for blk in self.blocks:
            old = blk.self_attn
            new = MultiViewSelfAttention(old.dim, old.num_heads, old.window_size, old.qk_norm, old.eps)
            new.load_state_dict(old.state_dict())
            blk.self_attn = new
        if self.extra_in:
            # widen the patch embedding for the conditioning channels (zero-init: step 0 =
            # the pretrained model)
            pe = self.patch_embedding
            self.patch_embedding = nn.Conv3d(self.in_dim + self.extra_in, self.dim,
                                             kernel_size=self.patch_size, stride=self.patch_size)
            with torch.no_grad():
                self.patch_embedding.weight.zero_()
                self.patch_embedding.weight[:, :self.in_dim] = pe.weight
                self.patch_embedding.bias.copy_(pe.bias)
        if cam_mode == "cat4d":
            self.frame_time_embedding = nn.Sequential(
                nn.Linear(self.freq_dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim))
            nn.init.zeros_(self.frame_time_embedding[2].weight)
            nn.init.zeros_(self.frame_time_embedding[2].bias)
        if cam_mode == "camtoken":
            # raymap (h x w at latent resolution) -> one token per 2x2 patch, like the latent
            self.cam_embedding = nn.Sequential(
                nn.Conv2d(cond_dim, self.dim, kernel_size=self.patch_size[1:], stride=self.patch_size[1:]),
                nn.GELU(approximate="tanh"), nn.Conv2d(self.dim, self.dim, 1))
            nn.init.normal_(self.cam_embedding[2].weight, std=0.02)
            nn.init.zeros_(self.cam_embedding[2].bias)

    @classmethod
    def from_t2v_pretrained(cls, path, cam_mode="concat", **kwargs):
        base = WanModel.from_pretrained(path, **kwargs)
        model = cls(**{k: v for k, v in base.config.items() if not k.startswith("_")}, cam_mode=cam_mode)
        sd = base.state_dict()
        w = sd.pop("patch_embedding.weight")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        new = [k for k in missing if not k.startswith(("cam_embedding.", "frame_time_embedding."))]
        assert new == ["patch_embedding.weight"] and not unexpected, (missing, unexpected)
        with torch.no_grad():
            if model.extra_in:
                model.patch_embedding.weight.zero_()
                model.patch_embedding.weight[:, :model.in_dim] = w
            else:
                model.patch_embedding.weight.copy_(w)
        return model

    def forward(self, x, cond, t, context, cam_mats=None, hist_mask=None):
        """
        x:        [B, V, C, F, h, w]  noisy latents
        cond:     [B, V, cond_dim, h, w]  raymap per view (static in time): Plucker for
                  concat / camtoken, origin + direction for cat4d
        t:        [B, F] or [B, V, F]  timesteps in [0, num_train_timesteps]
        context:  list of B text embeddings [L_i, text_dim]
        cam_mats: [B, V, 4, 4] relative camera matrices (prope: K~[R|t], gta: [R|t])
        hist_mask:[B, V, F] 1 where the latent frame is clean history (cat4d mask channel)
        returns   [B, V, C_out, F, h, w]
        """
        B, V, C, F_, h, w = x.shape
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if t.ndim == 2:
            t = t[:, None].expand(-1, V, -1)

        u = x
        if self.cam_mode in ("concat", "cat4d"):
            ch = [x, cond[:, :, :, None].expand(-1, -1, -1, F_, -1, -1).to(x.dtype)]
            if self.cam_mode == "cat4d":
                m = hist_mask if hist_mask is not None else (t == 0)
                ch.append(m[:, :, None, :, None, None].expand(-1, -1, 1, -1, h, w).to(x.dtype))
            u = torch.cat(ch, dim=2)
        u = self.patch_embedding(u.flatten(0, 1))                           # B*V, D, F, h', w'
        f, hp, wp = u.shape[2:]
        u = u.flatten(2).transpose(1, 2).unflatten(0, (B, V))               # B, V, f*h'*w', D
        extra = 0
        if self.cam_mode == "camtoken":
            ct = self.cam_embedding(cond.flatten(0, 1).to(u.dtype))          # B*V, D, h', w'
            ct = ct.flatten(2).transpose(1, 2).unflatten(0, (B, V))          # B, V, h'*w', D
            u = torch.cat([u, ct], dim=2)                                    # camera tokens last
            t = torch.cat([t, t.new_zeros(B, V, 1)], dim=2)                  # ... and clean
            extra = 1
        u = u.flatten(1, 2)                                                  # B, L, D
        L = u.shape[1]
        seq_lens = torch.full((B,), L, dtype=torch.long)
        tok_freqs = rope_freqs_for_views(self.freqs, V, f, hp, wp, extra)

        cam = None
        if self.cam_mode in ("prope", "gta"):
            tok_freqs = tok_freqs.clone()
            tok_freqs[:, :, CAM_PAIRS] = 1                                   # camera dims: no RoPE
            per_tok = lambda M: M[:, :, None].expand(-1, -1, (f + extra) * hp * wp, -1, -1).flatten(1, 2)
            Mf = cam_mats.float()
            cam = (per_tok(Mf.transpose(-1, -2)), per_tok(torch.linalg.inv(Mf)), per_tok(Mf))

        Ft = F_ + extra
        tt = t.reshape(B, V * Ft)                                            # per (view, frame)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, tt.flatten()).type_as(u))
        e = e.unflatten(0, (B, V * Ft))
        if self.cam_mode == "cat4d":
            # CAT4D: frame timestamp in [0, 1] (scaled like diffusion t) added to the t embedding
            ts = torch.linspace(0, 1, F_, device=device) * 1000
            et = self.frame_time_embedding(sinusoidal_embedding_1d(self.freq_dim, ts).type_as(u))
            e = e + et[None, None].expand(B, V, -1, -1).reshape(B, V * Ft, -1)
        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))

        context = self.text_embedding(torch.stack([
            torch.cat([c, c.new_zeros(self.text_len - c.size(0), c.size(1))]) for c in context]))

        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=None, freqs=(tok_freqs, cam),
                      context=context, context_lens=None)
        for i, block in enumerate(self.blocks):
            block = partial(block, **kwargs)
            if i in self.gradient_checkpointing_indices:
                u = checkpoint(block, u, use_reentrant=False)
            else:
                u = block(u)

        u = self.head(u, e)                                                  # B, L, prod(p)*c
        c = self.out_dim
        pt, ph, pw = self.patch_size
        u = u.view(B, V, Ft, hp, wp, pt, ph, pw, c)[:, :, :f]                # drop camera tokens
        u = torch.einsum("bvfhwpqrc->bvcfphqwr", u)
        return u.reshape(B, V, c, f * pt, hp * ph, wp * pw)
