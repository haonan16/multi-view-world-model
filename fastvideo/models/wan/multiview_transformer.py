# SPDX-License-Identifier: Apache-2.0
"""Multi-view Wan DiT: V camera views share one token sequence.

Ported from the "views as tokens" backbone in akiruga's ego_world_model
(``algorithms/wan/modules/model_multiview.py``) onto FastVideo's dense Wan
transformer:

- Tokens are laid out view-major, ``[view][frame][h][w]``. Every view gets the
  SAME (frame, h, w) RoPE positions, so attention is permutation-equivariant
  over views; views are told apart only by their camera, which enters as
  per-view conditioning channels (Plucker raymaps) concatenated to the latent
  before the patch embedding (``in_channels = z_dim + cond_dim``).
- Timestep modulation is per (view, frame), so some frames can be kept clean
  as history (diffusion-forcing style conditioning).
- The "tiled" design is the same model with V = 1 on a packed canvas.

Weights load 1:1 from a Wan2.1-T2V diffusers checkpoint whose patch embedding
was zero-widened by ``scripts/rh20t/make_multiview_checkpoint.py``, so step 0
reproduces the pretrained T2V model exactly.
"""

import torch

from fastvideo.distributed.communication_op import (sequence_model_parallel_all_gather_with_unpad,
                                                    sequence_model_parallel_shard)
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.models.wan.transformer import WanTransformer3DModel


class MultiViewWanTransformer3DModel(WanTransformer3DModel):
    """``forward`` takes ``[B, V, C_in, F, H, W]`` latents (+ conditioning
    channels) and returns ``[B, V, C_out, F, H, W]``. A 5-D input falls back
    to the single-view parent forward."""

    def _view_rope(self, n_views: int, f: int, h: int, w: int, device: torch.device):
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        cos, sin = get_rotary_pos_embed((f, h, w),
                                        self.hidden_size,
                                        self.num_attention_heads,
                                        rope_dim_list,
                                        dtype=torch.float64,
                                        rope_theta=10000)
        # identical positions for every view: [f*h*w, d] -> [V*f*h*w, d]
        return (cos.to(device).float().repeat(n_views, 1), sin.to(device).float().repeat(n_views, 1))

    def _per_token_modulation(self, timestep: torch.Tensor, batch_size: int, n_views: int, n_frames: int,
                              tokens_per_frame: int):
        """timestep [B], [B, F] or [B, V, F] -> temb [B, L, D], timestep_proj [B, L, 6, D].

        The embedding MLP runs once per (view, frame) and is then expanded over that
        frame's spatial tokens (calling the parent's ``timestep_seq_len`` path would
        embed every token separately, and it only supports batch size 1)."""
        if timestep.dim() == 1:
            timestep = timestep[:, None, None].expand(batch_size, n_views, n_frames)
        elif timestep.dim() == 2:
            timestep = timestep[:, None].expand(batch_size, n_views, n_frames)
        if timestep.shape != (batch_size, n_views, n_frames):
            raise ValueError(f"timestep shape {tuple(timestep.shape)} does not match "
                             f"(B, V, F) = {(batch_size, n_views, n_frames)}")
        embedder = self.condition_embedder
        temb = embedder.time_embedder(timestep.reshape(-1))  # [B*V*F, D]
        timestep_proj = embedder.time_modulation(temb)  # [B*V*F, 6*D]
        temb = temb.view(batch_size, n_views * n_frames, 1, -1)
        timestep_proj = timestep_proj.view(batch_size, n_views * n_frames, 1, 6, -1)
        temb = temb.expand(-1, -1, tokens_per_frame, -1).flatten(1, 2)
        timestep_proj = timestep_proj.expand(-1, -1, tokens_per_frame, -1, -1).flatten(1, 2)
        return temb, timestep_proj

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.Tensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
                guidance=None,
                r_timestep: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        if hidden_states.dim() == 5:
            return super().forward(hidden_states, encoder_hidden_states, timestep, encoder_hidden_states_image,
                                   guidance, r_timestep, **kwargs)
        if encoder_hidden_states_image is not None or r_timestep is not None:
            raise NotImplementedError("multi-view Wan supports text (or null-text) conditioning only")

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]

        batch_size, n_views, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        f, h, w = num_frames // p_t, height // p_h, width // p_w
        freqs_cis = self._view_rope(n_views, f, h, w, hidden_states.device)

        # patch-embed each view separately, then concatenate views along the sequence
        hidden_states = self.patch_embedding(hidden_states.flatten(0, 1))  # [B*V, D, f, h, w]
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # [B*V, f*h*w, D]
        hidden_states = hidden_states.reshape(batch_size, n_views * f * h * w, -1)

        if timestep.dim() >= 2 and timestep.shape[-1] != f:
            raise ValueError(f"per-frame timestep has {timestep.shape[-1]} frames, latents have {f}")
        temb, timestep_proj = self._per_token_modulation(timestep, batch_size, n_views, f, h * w)

        # Shard tokens and their per-token modulation identically (identity when SP = 1).
        hidden_states, original_seq_len = sequence_model_parallel_shard(hidden_states, dim=1)
        timestep_proj, _ = sequence_model_parallel_shard(timestep_proj, dim=1)
        temb, _ = sequence_model_parallel_shard(temb, dim=1)

        encoder_hidden_states = self.condition_embedder.text_embedder(encoder_hidden_states)
        assert encoder_hidden_states.dtype == orig_dtype

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, encoder_hidden_states,
                                                                  timestep_proj, freqs_cis, original_seq_len)
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, freqs_cis, original_seq_len)

        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        hidden_states = self.norm_out(hidden_states, shift.squeeze(2), scale.squeeze(2))

        hidden_states = sequence_model_parallel_all_gather_with_unpad(hidden_states, original_seq_len, dim=1)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, n_views, f, h, w, p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 1, 8, 2, 5, 3, 6, 4, 7)  # B, V, C, f, pt, h, ph, w, pw
        return hidden_states.reshape(batch_size, n_views, -1, f * p_t, h * p_h, w * p_w)


EntryClass = MultiViewWanTransformer3DModel
