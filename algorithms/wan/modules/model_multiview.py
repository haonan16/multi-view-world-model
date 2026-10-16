"""Multiview Wan backbone for the "views as tokens" vs "views tiled" ablation.

One token sequence holds V views laid out view-major: [view][frame][h][w].
  - Every view gets the SAME (frame, h, w) RoPE positions, so attention is
    permutation-equivariant over views; views are told apart only by their camera,
    which enters as Plucker raymap channels concatenated to the latent before the
    patch embedding (in_dim = z_dim + 6). The extra input weights start at zero, so
    step 0 is exactly the pretrained T2V model.
  - Timestep modulation is per (view, frame), so diffusion forcing can keep some
    frames clean as history.
The "tiled" design is the same model with V = 1 on the packed canvas.

Weights load 1:1 from Wan2.1-T2V (patch_embedding is widened; everything else is
unchanged), via `MultiViewWanModel.from_t2v_pretrained`.
"""
import math
from functools import partial

import torch
import torch.nn as nn
from einops import repeat
from torch.utils.checkpoint import checkpoint

from .attention import flash_attention
from .model import WanModel, WanSelfAttention, sinusoidal_embedding_1d


def rope_freqs_for_views(freqs, n_views, f, h, w):
    """Per-token complex RoPE multipliers [V*f*h*w, 1, C/2] for a view-major layout
    where every view repeats the same (f, h, w) grid."""
    c = freqs.shape[1]
    fq = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    one = torch.cat([
        fq[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        fq[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        fq[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1)
    return one.repeat(n_views, 1, 1)


def apply_rope(x, tok_freqs):
    """x: [B, L, n, d]; tok_freqs: [L, 1, d/2] complex. Same rotation as model.rope_apply,
    but in complex64 instead of complex128: positions here are < 64, where fp32 angles are
    exact to ~1e-6, and complex128 at V*F*h*w tokens per sample OOMs the tokens design."""
    b, L, n, d = x.shape
    xc = torch.view_as_complex(x.float().reshape(b, L, n, -1, 2))
    return torch.view_as_real(xc * tok_freqs.to(torch.complex64).unsqueeze(0)).flatten(3).type_as(x)


class MultiViewSelfAttention(WanSelfAttention):
    """WanSelfAttention with RoPE precomputed per token (`freqs` = [L, 1, d/2]).
    Parameter names are unchanged, so pretrained weights load directly."""

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        x = flash_attention(q=apply_rope(q, freqs), k=apply_rope(k, freqs), v=v,
                            k_lens=seq_lens, window_size=self.window_size)
        return self.o(x.flatten(2))


class MultiViewWanModel(WanModel):

    def __init__(self, *args, cond_dim=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_dim = cond_dim
        # swap self-attention for the per-token-RoPE variant (same parameters)
        for blk in self.blocks:
            old = blk.self_attn
            new = MultiViewSelfAttention(old.dim, old.num_heads, old.window_size, old.qk_norm, old.eps)
            new.load_state_dict(old.state_dict())
            blk.self_attn = new
        # widen the patch embedding for the raymap channels
        pe = self.patch_embedding
        self.patch_embedding = nn.Conv3d(self.in_dim + cond_dim, self.dim,
                                         kernel_size=self.patch_size, stride=self.patch_size)
        with torch.no_grad():
            self.patch_embedding.weight.zero_()
            self.patch_embedding.weight[:, :self.in_dim] = pe.weight
            self.patch_embedding.bias.copy_(pe.bias)

    @classmethod
    def from_t2v_pretrained(cls, path, **kwargs):
        base = WanModel.from_pretrained(path, **kwargs)
        model = cls(**{k: v for k, v in base.config.items() if not k.startswith("_")})
        sd = base.state_dict()
        w = sd.pop("patch_embedding.weight")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert missing == ["patch_embedding.weight"] and not unexpected, (missing, unexpected)
        with torch.no_grad():
            model.patch_embedding.weight.zero_()
            model.patch_embedding.weight[:, :model.in_dim] = w
        return model

    def forward(self, x, cond, t, context):
        """
        x:       [B, V, C, F, h, w]  noisy latents
        cond:    [B, V, cond_dim, h, w]  per-view camera conditioning (static in time)
        t:       [B, F] or [B, V, F]  timesteps in [0, num_train_timesteps]
        context: list of B text embeddings [L_i, text_dim]
        returns  [B, V, C_out, F, h, w]
        """
        B, V, C, F_, h, w = x.shape
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        cond = cond[:, :, :, None].expand(-1, -1, -1, F_, -1, -1)
        u = torch.cat([x, cond.to(x.dtype)], dim=2).flatten(0, 1)          # B*V, C+6, F, h, w
        u = self.patch_embedding(u)                                         # B*V, D, F, h', w'
        f, hp, wp = u.shape[2:]
        u = u.flatten(2).transpose(1, 2).reshape(B, V * f * hp * wp, self.dim)
        L = u.shape[1]
        seq_lens = torch.full((B,), L, dtype=torch.long)
        tok_freqs = rope_freqs_for_views(self.freqs, V, f, hp, wp)

        if t.ndim == 2:
            t = t[:, None].expand(-1, V, -1)
        t = t.reshape(B, V * F_)                                            # per (view, frame)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(u))
        e = e.unflatten(0, (B, V * F_))
        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))

        context = self.text_embedding(torch.stack([
            torch.cat([c, c.new_zeros(self.text_len - c.size(0), c.size(1))]) for c in context]))

        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=None, freqs=tok_freqs,
                      context=context, context_lens=None)
        for i, block in enumerate(self.blocks):
            block = partial(block, **kwargs)
            if i in self.gradient_checkpointing_indices:
                u = checkpoint(block, u, use_reentrant=False)
            else:
                u = block(u)

        u = self.head(u, e)                                                 # B, L, prod(p)*c
        c = self.out_dim
        pt, ph, pw = self.patch_size
        u = u.view(B, V, f, hp, wp, pt, ph, pw, c)
        u = torch.einsum("bvfhwpqrc->bvcfphqwr", u)
        return u.reshape(B, V, c, f * pt, hp * ph, wp * pw)
