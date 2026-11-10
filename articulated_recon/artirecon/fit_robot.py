"""Fit an articulated URDF robot (base pose + joint angles per frame) to multiview depth.

Coarse-to-fine, all batched on the GPU:
  1. table plane (RANSAC) -> foreground points (above the table)
  2. base search: the URDF's base column (links 0-1, independent of the arm's joints) slid over
     the table plane, scored by distance to the observed foreground (all frames pooled)
  3. frame 0: multi-hypothesis (base candidates x yaw) joint optimisation of yaw, base xy and q
  4. every later frame: multi-start (previous q, ready pose, perturbations), base fixed
  5. joint refinement of all frames + base with a weak temporal prior
"""
import time

import numpy as np
import torch

from .data import backproject
from .geom import DepthViews, chamfer_obs_to_model, depth_term, fit_plane_ransac
from .robot import PANDA_READY, URDFRobot, axis_angle


def prepare(obs, n_fg=6000, stride=2, seed=0, max_h=1.4, fg_thr=0.008):
    rng = np.random.default_rng(seed)
    T, V = obs.T, obs.V
    pts = [[backproject(obs.depth[t, v], obs.K[v], obs.c2w[v], stride)[0] for v in range(V)] for t in range(T)]
    P0 = np.concatenate(pts[0])
    sub = P0[rng.choice(len(P0), min(len(P0), 60000), replace=False)]
    n, d, _ = fit_plane_ransac(sub)
    cam_side = np.mean(obs.c2w[:, :3, 3] @ n + d)
    if cam_side < 0:
        n, d = -n, -d
    # table extent: inliers of the plane, robust bounds in the plane basis
    e1 = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    h0 = sub @ n + d
    tab = sub[np.abs(h0) < 0.006]
    a, b = tab @ e1, tab @ e2
    box = (np.quantile(a, 0.005), np.quantile(a, 0.995), np.quantile(b, 0.005), np.quantile(b, 0.995))
    fg, fg_all, fgv = [], [], []
    for t in range(T):
        P = np.concatenate(pts[t])
        Vid = np.concatenate([np.full(len(p), v) for v, p in enumerate(pts[t])])
        h = P @ n + d
        a, b = P @ e1, P @ e2
        m = (h > fg_thr) & (h < max_h) & (a > box[0] - 0.05) & (a < box[1] + 0.05) & (b > box[2] - 0.05) & (b < box[3] + 0.05)
        F, Fv = P[m], Vid[m]
        fg_all.append(F)
        sel = rng.choice(len(F), min(len(F), n_fg), replace=False) if len(F) else np.zeros(0, int)
        fg.append(F[sel])
        fgv.append(Fv[sel])
    return dict(n=n, d=d, e1=e1, e2=e2, box=box, fg=fg, fg_all=fg_all, fg_view=fgv,
                origin=-d * n)


def pad(list_of_pts, device="cuda"):
    K = max(len(p) for p in list_of_pts)
    F = torch.zeros(len(list_of_pts), K, 3, device=device)
    for i, p in enumerate(list_of_pts):
        F[i, :len(p)] = torch.as_tensor(p, dtype=torch.float32)
    return F, torch.tensor([len(p) for p in list_of_pts], device=device)


class RobotFitter:
    def __init__(self, obs, prep, robot=None, device="cuda", tie_fingers=True, log=print, calib=False, known_base=None):
        self.calib = calib
        self.known_base = known_base
        self.obs, self.prep, self.dev, self.log = obs, prep, device, log
        self.robot = robot or URDFRobot(device=device)
        self.views = DepthViews(obs.depth, obs.K, obs.c2w, device) if obs.depth is not None else None
        self.n = torch.tensor(prep["n"], dtype=torch.float32, device=device)
        self.e1 = torch.tensor(prep["e1"], dtype=torch.float32, device=device)
        self.e2 = torch.tensor(prep["e2"], dtype=torch.float32, device=device)
        self.o = torch.tensor(prep["origin"], dtype=torch.float32, device=device)
        Rp = torch.stack([self.e1, self.e2, self.n], 1)               # plane basis -> world
        self.Rp = Rp
        self.F, self.Fn = pad(prep["fg"], device)
        Kmax = self.F.shape[1]
        self.FV = torch.zeros(len(prep["fg"]), Kmax, dtype=torch.long, device=device)
        for i, fv in enumerate(prep["fg_view"]):
            self.FV[i, :len(fv)] = torch.as_tensor(fv, device=device)
        self.lo, self.hi = self.robot.lo, self.robot.hi
        self.logs = None                 # per (frame, view) log depth scale (kinematic self-calibration)

    # ---- parametrisation -------------------------------------------------------------------
    def base_T(self, yaw, xy, dz=None):
        """yaw [B], xy [B,2] -> [B,4,4]; base z-axis = table normal, origin on the table plane."""
        B = yaw.shape[0]
        if getattr(self, "known_base", None) is not None:      # calibrated rig: exact base pose (any tilt / height)
            return torch.as_tensor(self.known_base, dtype=torch.float32, device=self.dev)[None].expand(B, 4, 4).clone()
        Rz = axis_angle(torch.tensor([0., 0, 1], device=self.dev), yaw)
        T = torch.zeros(B, 4, 4, device=self.dev)
        T[:, :3, :3] = self.Rp @ Rz
        T[:, :3, 3] = self.o + xy[:, :1] * self.e1 + xy[:, 1:2] * self.e2 + (0 if dz is None else dz[:, None] * self.n)
        T[:, 3, 3] = 1
        return T

    def full_q(self, q8):
        return torch.cat([q8, q8[:, -1:]], 1)                         # tie the two fingers

    def loss(self, q8, base, frames, w_depth=1.0, s_ch=0.02, model_extra=None):
        P, N = self.robot.surface(self.full_q(q8), base)
        if model_extra is not None:
            P = torch.cat([P, model_extra[0]], 1)
            N = torch.cat([N, model_extra[1]], 1)
        fr = torch.as_tensor(frames, device=self.dev)
        F = self.F[fr]
        sc = None
        if self.logs is not None:
            sc = self.logs[fr].exp()                                   # B,V
            fv = self.FV[fr]
            C = self.views.C[fv]                                       # B,K,3
            F = C + sc.gather(1, fv)[..., None] * (F - C)
        ch, _ = chamfer_obs_to_model(F, self.Fn[fr], P, s_ch)
        dl = depth_term(self.views, P, N, frames, dscale=sc) if self.views is not None and w_depth > 0 else torch.zeros_like(ch)
        lim = (torch.relu(self.lo[:8] - q8) + torch.relu(q8 - self.hi[:8])).pow(2).sum(1) * 100
        return ch + w_depth * dl + lim, ch, dl

    # ---- stages ---------------------------------------------------------------------------
    def search_base(self, step=0.03, topk=4):
        """slide links 0/1 (q-independent) over the plane; score pooled foreground distance."""
        allF = np.concatenate(self.prep["fg_all"])
        allF = allF[np.random.default_rng(0).choice(len(allF), min(len(allF), 40000), replace=False)]
        box = self.prep["box"]
        a = np.arange(box[0] - 0.3, box[1] + 0.05, step)
        b = np.arange(box[2] - 0.3, box[3] + 0.3, step)
        A, Bv = np.meshgrid(a, b, indexing="ij")
        xy = torch.tensor(np.stack([A.ravel(), Bv.ravel()], 1), dtype=torch.float32, device=self.dev)
        oa = float(self.o @ self.e1)
        ob = float(self.o @ self.e2)
        xy = xy - torch.tensor([oa, ob], device=self.dev)
        rob = self.robot
        idx = torch.where(rob.link_of < 2)[0]                          # mesh links 0,1 = link0, link1
        pts = rob.pts[idx]                                             # local coords of link0 frame (link1 ~ z-sym)
        q0 = torch.tensor(PANDA_READY, device=self.dev)[None]
        T = rob.fk(q0)
        Ts = torch.stack([T[l] for l in rob.mesh_links], 1)[0][rob.link_of[idx]]
        loc = (Ts[:, :3, :3] @ pts[:, :, None])[..., 0] + Ts[:, :3, 3]
        Fw = torch.as_tensor(allF, dtype=torch.float32, device=self.dev)[None]
        scores = []
        from pytorch3d.ops import knn_points
        for yaw in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            for chunk in torch.split(torch.arange(len(xy), device=self.dev), 512):
                B = len(chunk)
                bT = self.base_T(torch.full((B,), yaw, device=self.dev), xy[chunk])
                Pm = torch.einsum("bij,nj->bni", bT[:, :3, :3], loc) + bT[:, None, :3, 3]
                d = knn_points(Pm, Fw.expand(B, -1, -1), K=1).dists[..., 0].sqrt()
                vis = (Pm - self.o) @ self.n > 0.012                   # parts above the table cut
                sc = (d.clamp(max=0.03) * vis).sum(1) / vis.sum(1).clamp(min=1)
                scores.append((sc, chunk, yaw))
        best = {}
        for sc, chunk, yaw in scores:
            for s_, i in zip(sc.tolist(), chunk.tolist()):
                best[i] = min(best.get(i, 9), s_)
        order = sorted(best, key=best.get)
        picks = []
        for i in order:
            p = xy[i]
            if all((p - xy[j]).norm() > 0.08 for j in picks):
                picks.append(i)
            if len(picks) >= topk:
                break
        self.log(f"  base search: top scores {[round(best[i] * 1000, 1) for i in picks]} mm")
        return xy[picks]

    def optimise(self, yaw, xy, q8, frames, iters=300, lr=0.03, opt_base=True, s_sched=(0.03, 0.01),
                 smooth=0.0, w_depth=1.0, model_extra=None):
        yaw = yaw.clone().requires_grad_(opt_base)
        xy = xy.clone().requires_grad_(opt_base)
        q8 = q8.clone().requires_grad_(True)
        params = [{"params": [q8], "lr": lr}]
        if self.logs is not None and self.logs.requires_grad:
            params.append({"params": [self.logs], "lr": 0.003})
        if opt_base:
            params.append({"params": [yaw, xy], "lr": lr * 0.3})
        opt = torch.optim.Adam(params)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
        for it in range(iters):
            s = s_sched[0] * (s_sched[1] / s_sched[0]) ** (it / max(1, iters - 1))
            base = self.base_T(yaw.expand(len(q8)), xy.expand(len(q8), -1))
            L, ch, dl = self.loss(q8, base, frames, w_depth, s, model_extra)
            tot = L.sum()
            if self.logs is not None and self.logs.requires_grad:
                tot = tot + 0.5 * self.logs.pow(2).sum()               # weak prior: scale ~ 1
            if smooth > 0 and len(frames) > 1:
                tot = tot + smooth * (q8[1:, :7] - q8[:-1, :7]).pow(2).sum()
            opt.zero_grad()
            tot.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            L, ch, dl = self.loss(q8, self.base_T(yaw.expand(len(q8)), xy.expand(len(q8), -1)), frames, w_depth,
                                  s_sched[1], model_extra)
        return yaw.detach(), xy.detach(), q8.detach(), L.detach(), ch.detach(), dl.detach()

    def run(self, n_yaw=8, iters=300, n_starts=10, seed=int(__import__("os").environ.get("FIT_SEED", 0))):
        t0 = time.time()
        g = torch.Generator(device="cpu").manual_seed(seed)
        T = self.obs.T
        single = self.views is None or self.views.V == 1
        if single:                       # one view: weaker evidence, search wider
            n_yaw = 12
        ready = torch.tensor(PANDA_READY[:8], device=self.dev)
        if self.known_base is not None:
            # calibrated rig: base pose known (camera-to-base extrinsics); express it in the plane parametrisation
            Tb = torch.as_tensor(self.known_base, dtype=torch.float32, device=self.dev)
            Rl = self.Rp.T @ Tb[:3, :3]
            yaw0 = torch.atan2(Rl[1, 0], Rl[0, 0])
            dpos = Tb[:3, 3] - self.o
            xy0 = torch.stack([dpos @ self.e1, dpos @ self.e2])
            H = 24
            yaw = yaw0.repeat(H)
            xy = xy0[None].repeat(H, 1)
            q8 = ready[None].repeat(H, 1) + torch.randn(H, 8, generator=g).to(self.dev) * 0.4 * (torch.arange(H, device=self.dev) > 0)[:, None]
            q8 = torch.max(torch.min(q8, self.hi[:8]), self.lo[:8])
            yaw, xy, q8, L, ch, dl = self.optimise(yaw, xy, q8, [0] * H, iters=iters, opt_base=False)
            self.log(f"  known base (calibrated rig): frame-0 {H} joint hypotheses")
        else:
            cands = self.search_base(topk=6 if single else 4)
            # frame 0: hypotheses = base candidates x yaws
            Hs = []
            for c in cands:
                for k in range(n_yaw):
                    Hs.append((2 * np.pi * k / n_yaw, c))
            yaw = torch.tensor([h[0] for h in Hs], device=self.dev)
            xy = torch.stack([h[1] for h in Hs])
            q8 = ready[None].repeat(len(Hs), 1)
            yaw, xy, q8, L, ch, dl = self.optimise(yaw, xy, q8, [0] * len(Hs), iters=iters)
        b = int(L.argmin())
        self.log(f"  frame 0: {len(L)} hypotheses, best loss {L[b]:.4f} (chamfer {ch[b]:.4f}, depth {dl[b]:.4f}); "
                 f"median {L.median():.4f}  [{time.time() - t0:.1f}s]")
        yaw, xy = yaw[b:b + 1], xy[b:b + 1]
        qs = [q8[b]]
        for t in range(1, T):
            inits = [qs[-1], ready]
            for _ in range(n_starts - 2):
                z = torch.randn(8, generator=g).to(self.dev) * 0.35
                z[7] = torch.rand(1, generator=g).item() * 0.04 - qs[-1][7]
                inits.append(qs[-1] + z)
            Q = torch.stack(inits)
            Q = torch.max(torch.min(Q, self.hi[:8]), self.lo[:8])
            n = len(Q)
            _, _, Q, L, ch, dl = self.optimise(yaw.repeat(n), xy.repeat(n, 1), Q, [t] * n, iters=iters,
                                               opt_base=False)
            b = int(L.argmin())
            qs.append(Q[b])
            self.log(f"  frame {t}: best loss {L[b]:.4f} (chamfer {ch[b]:.4f}, depth {dl[b]:.4f})  [{time.time() - t0:.1f}s]")
        # backward pass: re-initialise each frame from its successor's solution, keep the better fit
        Lf = []
        with torch.no_grad():
            for t in range(T):
                Lt, _, _ = self.loss(qs[t][None], self.base_T(yaw, xy), [t], 1.0, 0.01)
                Lf.append(float(Lt))
        for t in range(T - 2, -1, -1):
            inits = [qs[t], qs[t + 1]]
            for _ in range(4):
                inits.append(qs[t + 1] + torch.randn(8, generator=g).to(self.dev) * 0.15)
            Qi = torch.max(torch.min(torch.stack(inits), self.hi[:8]), self.lo[:8])
            n = len(Qi)
            _, _, Qi, Li, _, _ = self.optimise(yaw.repeat(n), xy.repeat(n, 1), Qi, [t] * n, iters=iters // 2,
                                               opt_base=False, s_sched=(0.02, 0.01))
            b = int(Li.argmin())
            if float(Li[b]) < Lf[t] - 1e-4:
                if (Qi[b] - qs[t]).abs().max() > 0.05:
                    self.log(f"  backward pass fixed frame {t}: loss {Lf[t]:.4f} -> {float(Li[b]):.4f}")
                qs[t], Lf[t] = Qi[b], float(Li[b])
        Q = torch.stack(qs)
        yaw_m, xy_m, Q, L, ch, dl = self.optimise(yaw, xy, Q, list(range(T)), iters=iters, opt_base=self.known_base is None,
                                                  lr=0.01, s_sched=(0.015, 0.008), smooth=0.0)
        # gauge: base yaw and joint 1 rotate about the same vertical axis; only link0's asymmetry
        # separates them. 1-D search over (yaw + delta, q1 - delta), scored by the full loss.
        with torch.no_grad():
            best = None
            for dlt in (np.linspace(-np.pi, np.pi, 73)[:-1] if self.known_base is None else [0.0]):
                q1 = Q[:, 0] - dlt
                if (q1 < self.lo[0]).any() or (q1 > self.hi[0]).any():
                    continue
                Qd = Q.clone(); Qd[:, 0] = q1
                yd = (yaw_m + dlt).expand(T)
                Ld, _, _ = self.loss(Qd, self.base_T(yd, xy_m.expand(T, -1)), list(range(T)), 1.0, 0.008)
                if best is None or Ld.mean() < best[0]:
                    best = (float(Ld.mean()), dlt)
        if best is not None and abs(best[1]) > 1e-6:
                self.log(f"  yaw/joint1 gauge: shifted by {np.degrees(best[1]):.0f} deg (loss {L.mean():.4f} -> {best[0]:.4f})")
                yaw_m = yaw_m + best[1]
                Q = Q.clone(); Q[:, 0] -= best[1]
                yaw_m, xy_m, Q, L, ch, dl = self.optimise(yaw_m, xy_m, Q, list(range(T)), iters=iters // 2,
                                                          lr=0.005, s_sched=(0.01, 0.008))
        if self.calib:
            self.logs = torch.zeros(T, self.views.V, device=self.dev, requires_grad=True)
            yaw_m, xy_m, Q, L, ch, dl = self.optimise(yaw_m, xy_m, Q, list(range(T)), iters=iters, lr=0.005,
                                                      s_sched=(0.015, 0.008), opt_base=self.known_base is None)
            self.logs = self.logs.detach()
            self.log(f"  depth self-calibration: scale per view (mean over frames) "
                     f"{np.round(self.logs.exp().mean(0).cpu().numpy(), 3)}, per frame spread "
                     f"{np.round(self.logs.exp().std(0).cpu().numpy(), 3)}")
        self.log(f"  joint refine: mean loss {L.mean():.4f} (chamfer {ch.mean():.4f}, depth {dl.mean():.4f})  [{time.time() - t0:.1f}s]")
        self.yaw, self.xy, self.Q = yaw_m[:1], xy_m[:1], Q
        return dict(base=self.base_T(yaw_m[:1], xy_m[:1])[0].cpu().numpy(), q=self.full_q(Q).cpu().numpy(),
                    depth_scale=None if self.logs is None else self.logs.exp().cpu().numpy(),
                    loss=L.cpu().numpy(), chamfer=ch.cpu().numpy(), depth=dl.cpu().numpy(), seconds=time.time() - t0)
