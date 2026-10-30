# SPDX-License-Identifier: Apache-2.0
"""Per-image and cross-view metrics for RH20T unseen-view evaluation.

Videos are [B, V, T, 3, H, W] in [-1, 1]. Image metrics come back as [B, V, T, len(FIELDS)]:

    psnr, ssim            whole image
    psnr_covered/_uncov.  pixels the context point cloud covers / does not cover
    psnr_moving           pixels whose ground truth changed since frame 0 (NaN on frame 0 or
                          when nothing moved), so copying frame 0 earns nothing there
    lpips                 AlexNet LPIPS (lower is better)

Cross-view consistency warps one generated view into another with the sensor depth of the
source view and scores agreement on pixels that pass a depth (occlusion) test. Running the
same warp on the ground truth gives the floor set by depth noise, lighting and specularities.
"""

from __future__ import annotations

import numpy as np
import torch

from fastvideo.dataset.rh20t_depth import render_points, unproject
from fastvideo.train.methods.fine_tuning.multiview_flow import ssim

FIELDS = ("psnr", "ssim", "psnr_covered", "psnr_uncovered", "psnr_moving", "lpips")
MOTION_THRESHOLD = 0.05  # mean abs RGB change in [0, 1] (about 13/255)
OCCLUSION_TOL_M = 0.05  # warped depth must match the target view's depth within this
MIN_OVERLAP_PX = 200  # fewer overlapping pixels -> NaN
LPIPS_CHUNK = 64


def masked_psnr(se: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """se [..., H, W] squared error, weight broadcastable to it -> [...] PSNR (NaN if empty)."""
    weight = weight.expand_as(se).float()
    n = weight.sum(dim=(-2, -1))
    mse = (se * weight).sum(dim=(-2, -1)) / n.clamp(min=1)
    return torch.where(n > 0, -10 * torch.log10(mse.clamp(min=1e-10)), torch.nan)


def lpips_distance(lpips_fn, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """[N, 3, H, W] in [-1, 1] -> [N]."""
    out = [lpips_fn(p, g).flatten() for p, g in zip(pred.split(LPIPS_CHUNK), gt.split(LPIPS_CHUNK))]
    return torch.cat(out)


def image_metrics(pred: torch.Tensor, gt: torch.Tensor, cover: torch.Tensor, lpips_fn) -> np.ndarray:
    """pred/gt [B, V, T, 3, H, W] in [-1, 1]; cover [B, V, 1, H, W] (1 = covered by the context
    cloud) -> [B, V, T, len(FIELDS)]."""
    p = ((pred.float() + 1) / 2).clamp(0, 1)
    g = ((gt.float() + 1) / 2).clamp(0, 1)
    se = ((p - g)**2).mean(dim=3)  # B, V, T, H, W
    m = cover.float()
    moving = ((g - g[:, :, :1]).abs().mean(dim=3) > MOTION_THRESHOLD).float()
    shape = gt.shape[:3]
    s = ssim(p.flatten(0, 2), g.flatten(0, 2)).reshape(shape)
    lp = lpips_distance(lpips_fn, pred.float().flatten(0, 2).clamp(-1, 1), gt.float().flatten(0, 2)).reshape(shape)
    ones = torch.ones_like(m)
    cols = [masked_psnr(se, ones), s, masked_psnr(se, m), masked_psnr(se, 1 - m), masked_psnr(se, moving), lp]
    return torch.stack(cols, dim=-1).cpu().numpy()


def warp_view(img: torch.Tensor, depth: torch.Tensor, K_src: torch.Tensor, c2w_src: torch.Tensor,
              K_dst: torch.Tensor, c2w_dst: torch.Tensor, depth_dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-warp img [3, H, W] of the source camera into the destination camera.
    -> (warped [3, H, W], valid [H, W]): valid where a source point lands, the destination has
    depth, and the two depths agree (the point is not occluded in the destination)."""
    H, W = depth.shape
    pts, valid = unproject(depth, K_src, c2w_src)
    colors = img.permute(1, 2, 0)[valid]
    warped, zbuf, hit = render_points(pts, colors, K_dst, torch.linalg.inv(c2w_dst), H, W)
    ok = hit & (depth_dst > 0) & ((zbuf - depth_dst).abs() < OCCLUSION_TOL_M)
    return warped, ok


def cross_view_psnr(video: torch.Tensor, depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> np.ndarray:
    """video [B, V, T, 3, H, W] in [-1, 1]; depth [B, V, T, H, W] metres; K [B, V, 3, 3] at (H, W);
    c2w [B, V, 4, 4] -> [B, T, V, V] PSNR of view a warped into view b against view b (NaN on the
    diagonal or with too little overlap)."""
    B, V, T = video.shape[:3]
    x = ((video.float() + 1) / 2).clamp(0, 1)
    d, K, c2w = depth.float(), K.float(), c2w.float()
    out = torch.full((B, T, V, V), torch.nan)
    for b in range(B):
        for t in range(T):
            for a in range(V):
                for v in range(V):
                    if a == v:
                        continue
                    warped, ok = warp_view(x[b, a, t], d[b, a, t], K[b, a], c2w[b, a], K[b, v], c2w[b, v], d[b, v, t])
                    if ok.sum() < MIN_OVERLAP_PX:
                        continue
                    se = ((warped - x[b, v, t])**2).mean(dim=0)
                    out[b, t, a, v] = masked_psnr(se, ok).cpu()
    return out.numpy()
