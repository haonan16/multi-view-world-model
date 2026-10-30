# SPDX-License-Identifier: Apache-2.0
"""RH20T depth decoding and point-cloud rendering.

Depth source: ``HaojunChen/RH20T`` ``depth/RH20T_cfgN_depth.tar.gz.part-*`` (the original
full-resolution release; verified 2026-10-21)::

    RH20T_cfgN/<episode>/cam_<serial>/depth.mp4

Each depth.mp4 frame is 640x720 grey: the top 640x360 half is the low byte and the bottom
half the high byte of a uint16 depth map aligned to the colour camera (rh20t_api
``extract.convert_depth``). Frame i pairs with colour frame i (same count as
``color/timestamps.npy``). Units: millimetres for D400 cameras; L515 cameras
(serial ``f*``) store quarter-millimetres scaled by 1/4 again by the API, so both come out
as ``raw / 1000`` metres (rh20t_api ``utils/point_cloud.py``). The high byte is lossy-video
coded, so values near byte boundaries can jump by 256 mm; ``clean_depth`` removes them.
"""
from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

DEPTH_H, DEPTH_W = 360, 640
MIN_DEPTH_M, MAX_DEPTH_M = 0.1, 3.0  # RH20T workspaces are within ~2 m of every camera


def decode_depth_frame(gray: np.ndarray) -> np.ndarray:
    """[2H, W] uint8 stacked low/high byte frame -> [H, W] float32 metres (0 = invalid)."""
    if gray.ndim != 2 or gray.shape[0] % 2:
        raise ValueError(f"expected a [2H, W] stacked depth frame, got {gray.shape}")
    h = gray.shape[0] // 2
    raw = gray[h:].astype(np.int32) * 256 + gray[:h].astype(np.int32)
    depth = raw.astype(np.float32) / 1000.0
    depth[(depth < MIN_DEPTH_M) | (depth > MAX_DEPTH_M)] = 0.0
    return depth


def read_depth_frames(path: str | Path, indices: np.ndarray) -> np.ndarray:
    """Decode the requested frame indices of a depth.mp4 -> [T, H, W] float32 metres."""
    wanted = {int(i) for i in indices}
    last = max(wanted)
    frames: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                # yuvj444p: luma = the grey value the encoder was given
                frames[i] = decode_depth_frame(frame.to_ndarray(format="gray"))
            if i >= last:
                break
    missing = wanted - frames.keys()
    if missing:
        raise ValueError(f"{path}: depth video has fewer frames than requested (missing {sorted(missing)[:5]})")
    return np.stack([frames[int(i)] for i in indices])


def clean_depth(depth: torch.Tensor, max_jump_m: float = 0.05) -> torch.Tensor:
    """Zero pixels that differ from their 3x3 median by more than max_jump_m.

    depth: [..., H, W] metres. Removes isolated high-byte coding errors (+-0.256 m spikes)
    and flying pixels at depth edges."""
    lead, (H, W) = depth.shape[:-2], depth.shape[-2:]
    d = depth.reshape(-1, 1, H, W)
    patches = F.unfold(F.pad(d, (1, 1, 1, 1), mode="replicate"), 3).reshape(d.shape[0], 9, H, W)
    med = patches.median(dim=1, keepdim=True).values
    bad = (d - med).abs() > max_jump_m
    return d.masked_fill(bad | (med <= 0), 0.0).reshape(*lead, H, W)


def unproject(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """depth [H, W] metres, K [3, 3] at (H, W), c2w [4, 4] -> (world points [N, 3], valid mask [H, W])."""
    H, W = depth.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=depth.dtype, device=depth.device) + 0.5,
                            torch.arange(W, dtype=depth.dtype, device=depth.device) + 0.5,
                            indexing="ij")
    valid = depth > 0
    z = depth[valid]
    x = (xs[valid] - K[0, 2]) / K[0, 0] * z
    y = (ys[valid] - K[1, 2]) / K[1, 1] * z
    cam = torch.stack([x, y, z], dim=-1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3], valid


def render_points(points: torch.Tensor, colors: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor, height: int,
                  width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-buffered single-pixel splat of a coloured point cloud into one camera.

    points [N, 3] world, colors [N, C], K [3, 3] at (height, width), w2c [4, 4].
    Returns (image [C, H, W], depth [H, W] metres (0 = empty), mask [H, W] bool)."""
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    front = z > 1e-3
    cam, z, colors = cam[front], z[front], colors[front]
    u = torch.floor(cam[:, 0] / z * K[0, 0] + K[0, 2]).long()
    v = torch.floor(cam[:, 1] / z * K[1, 1] + K[1, 2]).long()
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z, colors = u[inside], v[inside], z[inside], colors[inside]
    pix = v * width + u
    # z-buffer: keep only the nearest point of each pixel (index_put with duplicate
    # indices keeps an arbitrary writer, so filter to the per-pixel minimum first)
    nearest = torch.full((height * width, ), float("inf"), dtype=points.dtype, device=points.device)
    nearest.scatter_reduce_(0, pix, z, reduce="amin")
    keep = z <= nearest[pix]
    zbuf = torch.zeros(height * width, dtype=points.dtype, device=points.device)
    img = torch.zeros(height * width, colors.shape[1], dtype=colors.dtype, device=colors.device)
    zbuf[pix[keep]] = z[keep]
    img[pix[keep]] = colors[keep]
    mask = zbuf > 0
    return img.T.reshape(-1, height, width), zbuf.reshape(height, width), mask.reshape(height, width)
