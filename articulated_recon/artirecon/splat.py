"""GPU point-splat renderer (z-buffer) for overlays and silhouette metrics, no rasteriser needed.
Parts are given as dense oriented surface samples; each sample is splatted to a 2x2 pixel footprint."""
import numpy as np
import torch


def render_points(parts, K, w2c, H, W, dev="cuda", light=True, footprint=None):
    """parts: list of (P [N,3] world, Nrm [N,3], rgb (3,)) -> rgb [H,W,3] float, alpha, depth, part id.
    footprint: per part splat size (1 = single pixel, needs dense samples; 2 = 2x2), default 2"""
    P = torch.cat([torch.as_tensor(p, dtype=torch.float32, device=dev) for p, _, _ in parts])
    Nn = torch.cat([torch.as_tensor(n, dtype=torch.float32, device=dev) for _, n, _ in parts])
    pid = torch.cat([torch.full((len(p),), i, dtype=torch.long, device=dev) for i, (p, _, _) in enumerate(parts)])
    fp = torch.as_tensor(footprint or [2] * len(parts), device=dev)[pid]
    col = torch.stack([torch.as_tensor(c, dtype=torch.float32, device=dev) for _, _, c in parts])[pid]
    w2c = torch.as_tensor(w2c, dtype=torch.float32, device=dev)
    K = torch.as_tensor(K, dtype=torch.float32, device=dev)
    Pc = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = Pc[:, 2]
    ok = z > 0.05
    u = K[0, 0] * Pc[:, 0] / z.clamp(min=1e-4) + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / z.clamp(min=1e-4) + K[1, 2]
    if light:
        vd = -Pc / Pc.norm(dim=1, keepdim=True)
        nc = Nn @ w2c[:3, :3].T
        sh = 0.45 + 0.55 * (nc * vd).sum(1).abs()
        col = col * sh[:, None]
    zbuf = torch.full((H * W,), float("inf"), device=dev)
    idx_all = []
    for du in (0, 1):
        for dv in (0, 1):
            ui = torch.where(fp == 2, (u - 0.5 + du).floor(), u.floor()).long()
            vi = torch.where(fp == 2, (v - 0.5 + dv).floor(), v.floor()).long()
            m = ok & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & ((fp == 2) | ((du == 0) & (dv == 0)))
            flat = vi.clamp(0, H - 1) * W + ui.clamp(0, W - 1)
            zbuf.scatter_reduce_(0, flat[m], z[m], reduce="amin")
            idx_all.append((flat, m))
    img = torch.zeros(H * W, 3, device=dev)
    idm = torch.full((H * W,), -1, dtype=torch.long, device=dev)
    for flat, m in idx_all:
        win = m & (z <= zbuf[flat.clamp(0, H * W - 1)] + 1e-6)
        img[flat[win]] = col[win]
        idm[flat[win]] = pid[win]
    alpha = torch.isfinite(zbuf)
    depth = torch.where(alpha, zbuf, torch.zeros_like(zbuf))
    return (img.view(H, W, 3).cpu().numpy(), alpha.view(H, W).cpu().numpy(), depth.view(H, W).cpu().numpy(),
            idm.view(H, W).cpu().numpy())
