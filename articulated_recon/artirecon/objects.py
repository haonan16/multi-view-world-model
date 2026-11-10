"""Manipulated objects: discovery by motion, primitive shape, per-frame pose, articulation.

After the robot is explained, the remaining foreground ('residual') is clustered at frame 0.
Each frame-0 point is then tested in every later frame against the multiview depth:
  present  some camera sees a surface at that depth (within 8 mm)
  absent   some camera sees *through* it (observed surface >= 1.5 cm behind it): it moved
Clusters with no absent points are static scenery and are dropped unless the moving parts touch
them (e.g. the target box of an insertion). Moving clusters are either
  free rigid bodies (all points absent at some frame)   -> 6-DoF pose per frame
  articulated (only a part goes absent)                  -> static parent + moving child, joint
                                                            type/axis/range fitted from the motion
Shapes are oriented boxes (gravity-aligned at rest), extents refined from the points of all
frames mapped into the part frame. Poses are fitted with the same multiview depth / chamfer
objective as the robot (geom.py).
"""
import numpy as np
import torch
from pytorch3d.ops import knn_points
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

from .geom import chamfer_obs_to_model, depth_term, rho
from .robot import so3_exp


def knn_d(A, B):
    """numpy A [n,3], B [m,3] -> nearest distance from each A to B"""
    if len(B) == 0:
        return np.full(len(A), np.inf)
    a = torch.as_tensor(A, dtype=torch.float32, device="cuda")[None]
    b = torch.as_tensor(B, dtype=torch.float32, device="cuda")[None]
    return knn_points(a, b, K=1).dists[0, :, 0].sqrt().cpu().numpy()


def voxel_down(P, s=0.004):
    if len(P) == 0:
        return P
    k = np.floor(P / s).astype(np.int64)
    _, i = np.unique(k, axis=0, return_index=True)
    return P[i]


def residuals(prep, robot, q, base, thr=0.015):
    """foreground points not explained by the fitted robot, per frame."""
    out = []
    with torch.no_grad():
        qt = torch.as_tensor(q, dtype=torch.float32, device="cuda")
        bt = torch.as_tensor(base, dtype=torch.float32, device="cuda")[None].expand(len(qt), 4, 4)
        P, _ = robot.surface(qt, bt)
    for t, F in enumerate(prep["fg_all"]):
        F = voxel_down(F, 0.004)
        d = knn_d(F, P[t].cpu().numpy())
        out.append(F[d > thr])
    return out, P.cpu().numpy()


def presence(views, P, t, tau_in=0.008, tau_out=0.015):
    """frame-0 points P [n,3] at frame t -> +1 present, -1 absent (seen through), 0 unknown"""
    Pt = torch.as_tensor(P, dtype=torch.float32, device="cuda")[None]
    u, v, z = views.project(Pt)
    D, inside = views.sample(u, v, [t])
    ok = inside & (z > 0.05)
    present = (ok & (D > 0) & ((z - D).abs() < tau_in)).any(1)[0]
    absent = (ok & ((D - z > tau_out) | (D <= 0))).any(1)[0]
    s = torch.zeros(len(P), dtype=torch.int8, device="cuda")
    s[absent] = -1
    s[present] = 1                       # any camera confirming the surface wins
    return s.cpu().numpy()


def cluster(P, eps=0.012, min_pts=40):
    if len(P) < min_pts:
        return []
    lab = DBSCAN(eps=eps, min_samples=8).fit_predict(P)
    cl = [P[lab == k] for k in range(lab.max() + 1)]
    return [c for c in cl if len(c) >= min_pts]


# ---------------------------------------------------------------------------------------------
# oriented boxes
# ---------------------------------------------------------------------------------------------
def gravity_box(P, n, d, rest_on_table=True):
    """oriented box with z = table normal; yaw from the min-area rectangle of the footprint.
    -> (T 4x4 box frame at centre, half extents [3])"""
    e1 = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    a, b, h = P @ e1, P @ e2, P @ n + d
    best = None
    for th in np.linspace(0, np.pi / 2, 91):
        c, s = np.cos(th), np.sin(th)
        x, y = c * a + s * b, -s * a + c * b
        lo = np.quantile(x, 0.01), np.quantile(y, 0.01)
        hi = np.quantile(x, 0.99), np.quantile(y, 0.99)
        area = (hi[0] - lo[0]) * (hi[1] - lo[1])
        if best is None or area < best[0]:
            best = (area, th, lo, hi)
    _, th, lo, hi = best
    c, s = np.cos(th), np.sin(th)
    ax = c * e1 + s * e2
    ay = -s * e1 + c * e2
    h0 = 0.0 if rest_on_table else np.quantile(h, 0.01)
    h1 = np.quantile(h, 0.995)
    cx, cy, cz = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (h0 + h1) / 2
    T = np.eye(4)
    T[:3, :3] = np.stack([ax, ay, n], 1)
    T[:3, 3] = cx * ax + cy * ay + (cz - d) * n
    half = np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (h1 - h0) / 2])
    return T, half


def box_surface(half, n=1500, seed=0):
    """surface samples + normals of a centred box"""
    g = np.random.default_rng(seed)
    areas = np.array([half[1] * half[2], half[1] * half[2], half[0] * half[2], half[0] * half[2],
                      half[0] * half[1], half[0] * half[1]])
    k = np.maximum(10, (areas / areas.sum() * n).astype(int))
    P, N = [], []
    for f, m in enumerate(k):
        ax, sg = f // 2, (1 if f % 2 == 0 else -1)
        uv = g.uniform(-1, 1, (m, 3)) * half
        uv[:, ax] = sg * half[ax]
        nn = np.zeros((m, 3))
        nn[:, ax] = sg
        P.append(uv)
        N.append(nn)
    return np.concatenate(P), np.concatenate(N)


def refine_box_extents(pts_local, half, q=0.01):
    """robust extents of points expressed in the box frame; returns (centre shift, half)"""
    lo = np.quantile(pts_local, q, axis=0)
    hi = np.quantile(pts_local, 1 - q, axis=0)
    # only grow/shrink along axes with real support on both sides; keep prior otherwise
    newh = (hi - lo) / 2
    c = (hi + lo) / 2
    return c, newh


# ---------------------------------------------------------------------------------------------
# rigid pose fitting of a primitive against residual points + multiview depth
# ---------------------------------------------------------------------------------------------
def fit_pose(views, frame, R_pts, Ploc, Nloc, T_inits, iters=200, lr=0.01, s=(0.02, 0.006),
             w_depth=1.0, extra=None):
    """optimise a rigid transform for each init; returns best T, loss, all losses."""
    dev = "cuda"
    B = len(T_inits)
    Ti = torch.as_tensor(np.stack(T_inits), dtype=torch.float32, device=dev)
    R0, t0 = Ti[:, :3, :3], Ti[:, :3, 3]
    w = torch.zeros(B, 3, device=dev, requires_grad=True)
    dt = torch.zeros(B, 3, device=dev, requires_grad=True)
    Pl = torch.as_tensor(Ploc, dtype=torch.float32, device=dev)
    Nl = torch.as_tensor(Nloc, dtype=torch.float32, device=dev)
    F = torch.as_tensor(R_pts, dtype=torch.float32, device=dev)[None].expand(B, -1, -1)
    Fn = torch.full((B,), F.shape[1], device=dev)
    opt = torch.optim.Adam([w, dt], lr=lr)

    def ev(sv):
        R = so3_exp(w) @ R0
        t = t0 + dt
        P = torch.einsum("bij,nj->bni", R, Pl) + t[:, None]
        N = torch.einsum("bij,nj->bni", R, Nl)
        # model->data: object samples near some observed point (one-sided, robust)
        dmd = knn_points(P, F, K=1).dists[..., 0].clamp(min=1e-12).sqrt()
        if views is not None:
            with torch.no_grad():       # only samples some camera can actually see need a nearby point
                u, v, z = views.project(P)
                D, ins = views.sample(u, v, [frame] * B)
                facing = ((views.C[None, :, None] - P[:, None]) * N[:, None]).sum(-1) > 0
                vis = (ins & facing & ((z - D) < 0.01) & (D > 0)).any(1).float()
            md = (rho(dmd, sv) * vis).sum(1) / vis.sum(1).clamp(min=1)
        else:
            md = rho(dmd, sv).mean(1)
        # data->model only for observed points near this object (within 3 cm of its surface)
        dd = knn_points(F, P, K=1).dists[..., 0].clamp(min=1e-12).sqrt()
        near = (dd < 0.03).float()
        dm = (rho(dd, sv) * near).sum(1) / near.sum(1).clamp(min=1)
        dl = depth_term(views, P, N, [frame] * B) if views is not None and w_depth > 0 else torch.zeros_like(md)
        cov = 1 - near.sum(1) / F.shape[1]          # not used in loss; diagnostic
        return md + 0.5 * dm + w_depth * dl, R, t

    for it in range(iters):
        sv = s[0] * (s[1] / s[0]) ** (it / max(1, iters - 1))
        L, _, _ = ev(sv)
        opt.zero_grad()
        L.sum().backward()
        opt.step()
    with torch.no_grad():
        L, R, t = ev(s[1])
    b = int(L.argmin())
    T = np.eye(4)
    T[:3, :3] = R[b].cpu().numpy()
    T[:3, 3] = t[b].cpu().numpy()
    return T, float(L[b]), L.cpu().numpy()


def box_symmetries():
    """the 4 proper rotations mapping a box onto itself (180 deg about each axis + identity)"""
    out = [np.eye(3)]
    for ax in range(3):
        R = -np.eye(3)
        R[ax, ax] = 1
        out.append(R)
    return out


def infer_hole(static_ob, free_ob, clearance=0.003, min_frac=0.05):
    """Non-penetration: if a tracked free body ends up inside a static box, the static part must have a
    through-hole along the body's long axis. Returns hole dict (in the static box frame) or None."""
    from .evaluate import box_sdf
    Ts, hs = static_ob["poses"][0], np.asarray(static_ob["half"])
    hf = np.asarray(free_ob["half"])
    best = None
    for t in range(len(free_ob["poses"])):
        Tf = free_ob["poses"][t]
        Pl, _ = box_surface(hf, 2000)
        Pw = (Tf[:3, :3] @ Pl.T).T + Tf[:3, 3]
        frac = float((box_sdf(Pw, Ts, hs) < -0.002).mean())
        if best is None or frac > best[0]:
            best = (frac, t)
    frac, t = best
    if frac < min_frac:
        return None
    if np.any(np.sort(hf)[:2] + clearance >= np.sort(hs)[:2]):     # body as wide as the box: not an insertion
        return None
    Tf = free_ob["poses"][t]
    la = int(np.argmax(hf))                                   # long axis of the inserted body
    dw = Ts[:3, :3].T @ Tf[:3, la]                            # in static frame
    a = int(np.argmax(np.abs(dw)))                            # hole axis = static axis most aligned
    c = Ts[:3, :3].T @ (Tf[:3, 3] - Ts[:3, 3])                # body centre in static frame
    others = [i for i in range(3) if i != a]
    # cross-section half sizes of the body seen along the hole axis (its two short extents)
    Rf = Ts[:3, :3].T @ Tf[:3, :3]
    ext = np.abs(Rf) @ hf                                     # body half extents along static axes
    hole_half = {i: float(ext[i] + clearance) for i in others}
    return dict(axis=a, centre={i: float(c[i]) for i in others}, half=hole_half, frame=int(t), penetration_frac=frac,
                note=f"inferred from non-penetration at frame {t} ({frac:.0%} of the body inside the solid box); "
                     f"clearance {clearance * 1000:.0f} mm assumed")


def static_boxes(ob):
    """geometry of a static part as a list of (T, half): 1 box, or 4 slabs around an inferred hole."""
    T, H = ob["poses"][0], np.asarray(ob["half"], float)
    hole = ob.get("hole")
    if not hole:
        return [(T, H)]
    a = hole["axis"]
    b, c = [i for i in range(3) if i != a]
    hb, hc = hole["half"][b], hole["half"][c]
    cb, cc = np.clip(hole["centre"][b], -H[b] + hb, H[b] - hb), np.clip(hole["centre"][c], -H[c] + hc, H[c] - hc)
    out = []

    def slab(lo, hi):
        lo, hi = np.array(lo, float), np.array(hi, float)
        if np.any(hi - lo <= 1e-4):
            return
        Tl = T.copy()
        Tl[:3, 3] = T[:3, 3] + T[:3, :3] @ ((lo + hi) / 2)
        out.append((Tl, (hi - lo) / 2))
    lo, hi = -H.copy(), H.copy()
    l1, h1 = lo.copy(), hi.copy(); h1[b] = cb - hb; slab(l1, h1)
    l2, h2 = lo.copy(), hi.copy(); l2[b] = cb + hb; slab(l2, h2)
    l3, h3 = lo.copy(), hi.copy(); l3[b], h3[b], h3[c] = cb - hb, cb + hb, cc - hc; slab(l3, h3)
    l4, h4 = lo.copy(), hi.copy(); l4[b], h4[b], l4[c] = cb - hb, cb + hb, cc + hc; slab(l4, h4)
    return out if out else [(T, H)]


def tray_slabs(half, up, open_sign=1, floor_top=None, th=0.008):
    """open container: floor + 4 walls in the part frame (open on the `up` axis at +open_sign side)."""
    H = np.asarray(half, float)
    out = []
    o = [i for i in range(3) if i != up]
    ft = -H[up] + th if floor_top is None else max(floor_top, -H[up] + th)
    # floor
    lo, hi = -H.copy(), H.copy()
    if open_sign > 0:
        hi[up] = ft
    else:
        lo[up] = -ft
    out.append((lo, hi))
    for ax in o:
        for s in (-1, 1):
            lo, hi = -H.copy(), H.copy()
            if s < 0:
                hi[ax] = -H[ax] + th
            else:
                lo[ax] = H[ax] - th
            out.append((lo, hi))
    res = []
    for lo, hi in out:
        T = np.eye(4); T[:3, 3] = (lo + hi) / 2
        res.append((T, (hi - lo) / 2))
    return res


def shell_slabs(half, open_axis, open_sign, th=0.008):
    """hollow box open on one face (cabinet carcass)."""
    H = np.asarray(half, float)
    res = []
    for ax in range(3):
        for s in (-1, 1):
            if ax == open_axis and s == open_sign:
                continue
            lo, hi = -H.copy(), H.copy()
            if s < 0:
                hi[ax] = -H[ax] + th
            else:
                lo[ax] = H[ax] - th
            T = np.eye(4); T[:3, 3] = (lo + hi) / 2
            res.append((T, (hi - lo) / 2))
    return res


def part_boxes(T, half, slabs=None):
    """world boxes of a part: one box, or its slabs (local T, half) composed with the part pose"""
    if not slabs:
        return [(np.asarray(T), np.asarray(half))]
    return [(np.asarray(T) @ Tl, np.asarray(hl)) for Tl, hl in slabs]
