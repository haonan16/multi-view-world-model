"""Helpers shared by the multiview datasets and the multiview algorithms.

The two multiview designs under ablation consume the SAME dataset sample
(videos [V, T, 3, H, W] + per-view cameras); only how the views reach the model
differs:
  - "tokens": every view is encoded separately at full (H, W) and its tokens are
    concatenated in the sequence (V x tokens-per-view).
  - "tiled":  the V views are downscaled and packed into ONE canvas of size (H, W)
    (fixed canvas => ~V x fewer tokens). `tile_views` does the packing and returns
    the intrinsics of each tile so per-view raymaps can be tiled identically.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def grid_for(n_views: int) -> Tuple[int, int]:
    """Near-square (rows, cols) grid for n_views tiles, rows <= cols."""
    rows = int(math.floor(math.sqrt(n_views)))
    while n_views % rows:
        rows -= 1
    return rows, n_views // rows


def tile_views(x: torch.Tensor, grid: Tuple[int, int] = None) -> torch.Tensor:
    """Pack views into one canvas of the SAME spatial size as a single view.

    x: [..., V, T, C, H, W] -> [..., T, C, H, W]. Each view is area-downsampled to
    (H / rows, W / cols); H and W must be divisible by rows / cols.
    """
    *lead, V, T, C, H, W = x.shape
    rows, cols = grid or grid_for(V)
    assert rows * cols == V, f"grid {rows}x{cols} does not hold {V} views"
    assert H % rows == 0 and W % cols == 0, f"{H}x{W} not divisible by grid {rows}x{cols}"
    h, w = H // rows, W // cols
    y = x.reshape(-1, C, H, W)
    y = F.interpolate(y.float(), size=(h, w), mode="area").to(x.dtype)
    y = y.reshape(-1, rows, cols, T, C, h, w)
    y = y.permute(0, 3, 4, 1, 5, 2, 6).reshape(*lead, T, C, H, W)
    return y


def untile_views(y: torch.Tensor, n_views: int, grid: Tuple[int, int] = None) -> torch.Tensor:
    """Inverse layout of `tile_views` (without upsampling): [..., T, C, H, W] ->
    [..., V, T, C, H / rows, W / cols]."""
    *lead, T, C, H, W = y.shape
    rows, cols = grid or grid_for(n_views)
    h, w = H // rows, W // cols
    z = y.reshape(-1, T, C, rows, h, cols, w).permute(0, 3, 5, 1, 2, 4, 6)
    return z.reshape(*lead, n_views, T, C, h, w)


def tile_intrinsics(K: torch.Tensor, height: int, width: int,
                    grid: Tuple[int, int] = None) -> torch.Tensor:
    """Intrinsics of each view after `tile_views`, expressed in the tile's OWN pixel
    frame (each tile is a downscaled copy of the full view). K: [..., V, 3, 3]."""
    V = K.shape[-3]
    rows, cols = grid or grid_for(V)
    K = K.clone()
    K[..., 0, :] *= 1.0 / cols
    K[..., 1, :] *= 1.0 / rows
    return K
