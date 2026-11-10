"""Geometric observation model shared by every fit: multiview depth 'rendering' by point projection.

Instead of rasterising and comparing pixels, the candidate articulated scene is represented by
oriented surface samples. Each sample is projected into every observed camera (cheap) and its
camera-z is compared with the observed (or predicted) depth at that pixel:
  |z - D| small      -> on the observed surface (robust attraction)
  z < D - tau        -> the sample floats in space that camera saw as empty (free-space violation)
  z > D + tau        -> occluded in that view, no evidence (constant cost, so hiding gains nothing)
  D == 0 / sky       -> nothing there: free-space violation
Plus a data->model term: observed foreground points must be explained by some model surface.
This is z-buffer reasoning over all views at once (multiview 'rendering' of the same 3D state)
without a rasteriser, and it is exactly differentiable in the part poses / joint angles.
"""
import numpy as np
import torch
from pytorch3d.ops import knn_points


def rho(x, s):
    x2 = x * x
    return x2 / (x2 + s * s)


class DepthViews:
    """Depth maps of T frames x V views on the GPU. depth [T,V,H,W] metres (0 = empty)."""

    def __init__(self, depth, K, c2w, device="cuda"):
        self.D = torch.as_tensor(np.ascontiguousarray(depth), dtype=torch.float32, device=device)
        self.T, self.V, self.H, self.W = self.D.shape
        self.K = torch.as_tensor(K, dtype=torch.float32, device=device)
        c2w = torch.as_tensor(c2w, dtype=torch.float64, device=device)
        self.w2c = torch.linalg.inv(c2w).float()
        self.C = c2w[:, :3, 3].float()

    def project(self, P, frames=None):
        """P [B,N,3] world (B frames, aligned with `frames` index list) -> u,v,z [B,V,N]"""
        Pc = torch.einsum("vij,bnj->bvni", self.w2c[:, :3, :3], P) + self.w2c[None, :, None, :3, 3]
        z = Pc[..., 2]
        zc = z.clamp(min=1e-4)
        u = self.K[None, :, None, 0, 0] * Pc[..., 0] / zc + self.K[None, :, None, 0, 2]
        v = self.K[None, :, None, 1, 1] * Pc[..., 1] / zc + self.K[None, :, None, 1, 2]
        return u, v, z

    def sample(self, u, v, frames):
        """nearest depth at (u,v) [B,V,N] for frames [B] -> D, inside"""
        ui, vi = u.round().long(), v.round().long()
        inside = (ui >= 0) & (ui < self.W) & (vi >= 0) & (vi < self.H)
        ui, vi = ui.clamp(0, self.W - 1), vi.clamp(0, self.H - 1)
        B, V, N = u.shape
        f = torch.as_tensor(frames, device=u.device).view(B, 1, 1).expand(B, V, N)
        vv = torch.arange(V, device=u.device).view(1, V, 1).expand(B, V, N)
        return self.D[f, vv, vi, ui], inside


def depth_term(views, P, Nrm, frames, tau=0.02, s=0.01, empty_cost=1.0, per_point=False, dscale=None):
    """mean robust cost of surface samples P [B,N,3] (normals Nrm) against the depth views.
    dscale [B,V]: optional per-view multiplicative correction of the observed depth."""
    u, v, z = views.project(P)
    D, inside = views.sample(u, v, frames)
    if dscale is not None:
        D = D * dscale[:, :, None]
    facing = ((views.C[None, :, None] - P[:, None]) * Nrm[:, None]).sum(-1) > 0
    ok = inside & facing & (z > 0.05)
    r = z - D
    c = torch.where(r > tau, rho(torch.full_like(r, tau), s), rho(r, s))
    c = torch.where(D <= 0, torch.full_like(c, empty_cost), c)
    w = ok.float()
    if per_point:
        return (c * w).sum(1) / w.sum(1).clamp(min=1)          # B,N
    return (c * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1)   # B


def chamfer_obs_to_model(F, Fn, M, s=0.02):
    """F [B,K,3] observed points (Fn valid counts [B]), M [B,N,3] model -> mean rho(dist)"""
    d2 = knn_points(F, M, lengths1=Fn, K=1).dists[..., 0]
    d = d2.clamp(min=1e-12).sqrt()
    mask = torch.arange(F.shape[1], device=F.device)[None] < Fn[:, None]
    return (rho(d, s) * mask).sum(1) / Fn.clamp(min=1), d


def fit_plane_ransac(P, iters=400, thr=0.006, rng=0):
    g = np.random.default_rng(rng)
    best, bn = None, -1
    for _ in range(iters):
        a, b, c = P[g.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        k = (np.abs(P @ n + d) < thr).sum()
        if k > bn:
            best, bn = (n, d), k
    n, d = best
    inl = np.abs(P @ n + d) < thr
    c = P[inl].mean(0)
    _, _, vt = np.linalg.svd(P[inl] - c, full_matrices=False)
    n = vt[2]
    d = -n @ c
    if n[2] < 0:          # orient 'up' toward the cameras' side: pick the side most cameras are on
        n, d = -n, -d
    return n, d, inl


def table_calibrate(obs, iters=3, band=0.05, log=print):
    """GT-free per-(frame, view) depth scale from the static table: scale each depth map about its camera
    centre so its table pixels land on the shared table plane. Returns (scales [T,V], plane (n,d), noise_m)."""
    from .data import backproject
    T, V = obs.T, obs.V
    S = np.ones((T, V))
    P0 = np.concatenate([backproject(obs.depth[t, v], obs.K[v], obs.c2w[v], 4)[0] for t in range(T) for v in range(V)])
    sub = P0[np.random.default_rng(0).choice(len(P0), min(len(P0), 80000), replace=False)]
    n, d, _ = fit_plane_ransac(sub, thr=0.015)
    if np.mean(obs.c2w[:, :3, 3] @ n + d) < 0:
        n, d = -n, -d
    for _ in range(iters):
        res_all = []
        for t in range(T):
            for v in range(V):
                P, _ = backproject(obs.depth[t, v] * S[t, v], obs.K[v], obs.c2w[v], 4)
                C = obs.c2w[v][:3, 3]
                h = P @ n + d
                m = np.abs(h) < band
                if m.sum() < 200:
                    continue
                s = -(n @ C + d) / ((P[m] - C) @ n)                 # per-pixel scale putting it on the plane
                S[t, v] *= float(np.median(s))
                res_all.append(h[m])
        # re-fit the plane on the corrected points
        Pc = np.concatenate([backproject(obs.depth[t, v] * S[t, v], obs.K[v], obs.c2w[v], 4)[0] for t in range(T) for v in range(V)])
        Pc = Pc[np.abs(Pc @ n + d) < band]
        c = Pc.mean(0)
        _, _, vt = np.linalg.svd(Pc[np.random.default_rng(1).choice(len(Pc), min(len(Pc), 50000), replace=False)] - c,
                                 full_matrices=False)
        n2 = vt[2] * np.sign(vt[2] @ n)
        n, d = n2, -n2 @ c
    # per-camera offset along the table normal (extrinsic drift): median table height seen by each camera
    offs = np.zeros(V)
    for v in range(V):
        hs = []
        for t in range(T):
            P, _ = backproject(obs.depth[t, v] * S[t, v], obs.K[v], obs.c2w[v], 4)
            h = P @ n + d
            hs.append(h[np.abs(h) < band])
        hv = np.concatenate(hs)
        offs[v] = float(np.median(hv)) if len(hv) > 200 else 0.0
    Pc = np.concatenate([backproject(obs.depth[t, v] * S[t, v], obs.K[v], obs.c2w[v], 4)[0] - offs[v] * n
                         for t in range(T) for v in range(V)])
    Pc = Pc[np.abs(Pc @ n + d) < band]
    h = Pc @ n + d
    noise = float(1.4826 * np.median(np.abs(h - np.median(h))))
    log(f"  table self-calibration: per-camera offsets along the table normal {np.round(offs * 1000, 1)} mm")
    table_calibrate.offsets = (offs, n)
    log(f"  table self-calibration: depth scales {np.round(S.mean(0), 3)} (per-frame spread {np.round(S.std(0), 3)}), "
        f"table noise {noise * 1000:.1f} mm")
    return S, (n, d), noise
