"""Objects v2: annotation + SAM 3 masks + multiview depth.

What moves and where is not discovered from point-cloud residuals any more. It comes from a per-episode
annotation (object name, SAM 3 text prompt, role moved / static, one click on the object in one view/frame),
written by looking at the frames. SAM 3 gives instance masks for the prompt in every view and frame; the
clicked instance is followed through time in 3D:
  - each mask is lifted with its view's depth (robot surface and table removed) -> 3D points + centre
  - per frame, in every view, the instance whose 3D centre is closest to the predicted position is taken;
    predicted position = previous one, or carried rigidly by the gripper when it was held
  - the selected masks of all views are fused into one point set per frame
Shape: from the frames where the object rests at its start pose (all views fused): gravity-aligned box
(bottom on the table) or a 2.5D hull (footprint polygon extruded to the observed top), or a given mesh
(e.g. SAM 3D) rescaled to the fused points.
Pose per frame (6-DoF): fused-mask points -> model chamfer, multiview depth z-test (free space seen through
by any camera is penalised), silhouette (visible model points inside the instance mask), table
non-penetration, resting contact when not held, and rigid-in-hand consistency while held.
"""
import cv2
import numpy as np
import torch
import trimesh
from pytorch3d.ops import knn_points

from .geom import DepthViews, depth_term, rho
from .scene_fit import hand_T, tcp_points


# ------------------------------------------------------------------ SAM 3 instances
class Sam3Masks:
    def __init__(self, path, H, W):
        z = np.load(path)
        self.prompts = list(z["prompts"])
        self.p, self.t, self.v, self.score, self.box = z["p"], z["t"], z["v"], z["score"], z["box"]
        self._packed = z["mask"]
        self.H, self.W = H, W

    def mask(self, i):
        return np.unpackbits(self._packed[i])[: self.H * self.W].reshape(self.H, self.W).astype(bool)

    def ids(self, prompt, t, v):
        ps = prompt if isinstance(prompt, (list, tuple)) else [prompt]
        pis = [self.prompts.index(x) for x in ps if x in self.prompts]
        return np.nonzero(np.isin(self.p, pis) & (self.t == t) & (self.v == v))[0]


def lift(obs, t, v, m, erode=2):
    if erode:
        m = cv2.erode(m.astype(np.uint8), np.ones((2 * erode + 1,) * 2, np.uint8)) > 0
    D = obs.depth[t, v]
    ys, xs = np.nonzero(m & (D > 0))
    if len(xs) == 0:
        return np.zeros((0, 3))
    z = D[ys, xs]
    K = obs.K[v]
    Pc = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z, np.ones_like(z)], 1)
    return (Pc @ obs.c2w[v].T)[:, :3]


def _knn_d(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.full(len(A), np.inf)
    a = torch.as_tensor(A, dtype=torch.float32, device="cuda")[None]
    b = torch.as_tensor(B, dtype=torch.float32, device="cuda")[None]
    return knn_points(a, b, K=1).dists[0, :, 0].sqrt().cpu().numpy()


# ------------------------------------------------------------------ shapes
def box_shape(P, n, d, base_h=0.0):
    """gravity-aligned box resting on a support at height base_h above the table (n.x + d = 0) -> T, half"""
    e1 = np.cross(n, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 0.1 else np.cross(n, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    h = P @ n + d
    top = np.quantile(h, 0.97)
    up0 = h > base_h + 0.6 * (top - base_h)
    Pa = P[up0] if up0.sum() > 30 else P
    xy = np.stack([Pa @ e1, Pa @ e2], 1).astype(np.float32)
    (cx, cy), (w, l), ang = cv2.minAreaRect(xy)
    a = np.deg2rad(ang)
    r1 = np.cos(a) * e1 + np.sin(a) * e2
    r2 = np.cross(n, r1)
    R = np.stack([r1, r2, n], 1)
    # robust extents along the rectangle axes from the upper part of the object (mask edges bleed depth
    # onto the support, which inflates the footprint at the bottom)
    up = h > base_h + 0.6 * (top - base_h)
    Pu = P[up] if up.sum() > 30 else P
    u1, u2 = Pu @ r1, Pu @ r2
    lo1, hi1 = np.quantile(u1, [0.03, 0.97]); lo2, hi2 = np.quantile(u2, [0.03, 0.97])
    w, l = hi1 - lo1, hi2 - lo2
    ctr = (lo1 + hi1) / 2 * r1 + (lo2 + hi2) / 2 * r2
    o = ctr - (ctr @ n) * n - d * n               # point on the table under the centre
    hgt = max(top - base_h, 0.004)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = o + n * (base_h + hgt / 2)
    return T, np.array([w / 2, l / 2, hgt / 2])


def container_mesh(half, th=0.004):
    hx, hy, hz = half
    parts = [((0, 0, -hz + th / 2), (2 * hx, 2 * hy, th)),
             ((hx - th / 2, 0, 0), (th, 2 * hy, 2 * hz)), ((-hx + th / 2, 0, 0), (th, 2 * hy, 2 * hz)),
             ((0, hy - th / 2, 0), (2 * hx, th, 2 * hz)), ((0, -hy + th / 2, 0), (2 * hx, th, 2 * hz))]
    ms = []
    for c, e in parts:
        b = trimesh.creation.box(extents=e); b.apply_translation(c); ms.append(b)
    return trimesh.util.concatenate(ms)


def hull_shape(P, n, d, base_h=0.0):
    """2.5D hull: footprint polygon (convex) extruded from the support to the observed top"""
    T, half = box_shape(P, n, d, base_h)
    Pl = (P - T[:3, 3]) @ T[:3, :3]
    hull = cv2.convexHull(Pl[:, :2].astype(np.float32))[:, 0]
    poly = trimesh.path.polygons.Polygon(hull)
    m = trimesh.creation.extrude_polygon(poly, 2 * half[2])
    m.apply_translation([0, 0, -half[2]])
    return T, half, m


def mesh_samples(m, n=1500, seed=0):
    P, fi = trimesh.sample.sample_surface_even(m, n, seed=seed) if hasattr(trimesh.sample, "sample_surface_even") \
        else trimesh.sample.sample_surface(m, n)
    return np.asarray(P), np.asarray(m.face_normals[fi])


# ------------------------------------------------------------------ rotations
def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = w / th
    K = torch.zeros(w.shape[:-1] + (3, 3), device=w.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    I = torch.eye(3, device=w.device).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


class ObjectsV2:
    def __init__(self, obs, prep, robot, rob, sam, ann, log=print, views=None, exclude_views=()):
        self.obs, self.prep, self.robot, self.rob, self.sam, self.ann, self.log = obs, prep, robot, rob, sam, ann, log
        self.views = views or DepthViews(obs.depth, obs.K, obs.c2w)
        self.n, self.d = np.asarray(prep["n"], float), float(prep["d"])
        self.use_views = [v for v in range(obs.V) if v not in exclude_views]
        self.sc_range = (np.log(0.55), np.log(1.5))
        q, base = rob["q"], rob["base"]
        with torch.no_grad():
            P, _ = robot.surface(torch.as_tensor(q, dtype=torch.float32, device="cuda"),
                                 torch.as_tensor(base, dtype=torch.float32, device="cuda")[None].expand(len(q), 4, 4)
                                 if np.ndim(base) == 2 else torch.as_tensor(base, dtype=torch.float32, device="cuda"))
        self.robot_pts = P.cpu().numpy()                                  # T,N,3
        self.Th = hand_T(robot, q, base)[0]                               # T,4,4 hand frame
        self.tcp = tcp_points(robot, q, base)                             # T,3 point between the finger tips
        self.width = np.asarray(q)[:, 7:9].sum(1) if np.asarray(q).shape[1] >= 9 else np.zeros(len(q))
        # masks as torch for the silhouette term: distance transform outside the selected instance (pixels)
        self.H, self.W = obs.depth.shape[2:]

    # -------------------------------------------------------------- instance selection
    def instances(self, prompt, t, v):
        out = []
        for i in self.sam.ids(prompt, t, v):
            m = self.sam.mask(i)
            if m.sum() > 0.25 * self.H * self.W:          # table / background false positives
                continue
            P = lift(self.obs, t, v, m)
            if len(P):
                P = P[(P @ self.n + self.d) > 0.004]
                if len(P):
                    P = P[_knn_d(P, self.robot_pts[t][::2]) > 0.008]
            if len(P) < 15:
                out.append(dict(i=i, m=m, P=P, c=None, score=float(self.sam.score[i])))
                continue
            mu = m.astype(np.uint8)
            ring = (cv2.dilate(mu, np.ones((19, 19), np.uint8)) > 0) & ~(cv2.dilate(mu, np.ones((7, 7), np.uint8)) > 0)
            Q = lift(self.obs, t, v, ring, erode=0)
            if len(Q):
                Q = Q[_knn_d(Q, self.robot_pts[t][::2]) > 0.01]
            out.append(dict(i=i, m=m, P=P, c=np.median(P, 0), score=float(self.sam.score[i]),
                            ring_h=(Q @ self.n + self.d) if len(Q) else np.array([])))
        return out

    def track(self, spec):
        """follow the annotated instance through all frames -> per frame: {v: inst}, centre, held flag"""
        obs, T = self.obs, self.obs.T
        pr = spec["prompt"]
        th, vh, (uh, vvh) = spec["hint"]["t"], spec["hint"]["v"], spec["hint"]["uv"]
        cands = self.instances(pr, th, vh)
        hit = sorted([c for c in cands if c["m"][int(vvh), int(uh)] and c["c"] is not None],
                     key=lambda c: c["m"].sum())          # most specific (smallest) mask under the click
        if not hit:      # nearest mask centroid to the click
            hit = sorted([c for c in cands if c["c"] is not None],
                         key=lambda c: np.hypot(*(np.argwhere(c["m"]).mean(0) - [vvh, uh])))[:1]
        if not hit:
            raise RuntimeError(f"{spec['name']}: no SAM 3 instance for '{pr}' at the hint")
        c0 = hit[0]["c"]
        size = lambda c: float(np.quantile(np.linalg.norm(c["P"] - c["c"], axis=1), 0.9))
        ref_size = size(hit[0])
        sel = [None] * T
        cen = [None] * T
        held = np.zeros(T, bool)
        gate = spec.get("gate", 0.05)

        allow_global = [True]

        def pick(t, pred_list, g=None, min_views=1):
            g = g or gate
            chosen, pts = {}, []
            for v in self.use_views:
                best = None
                for c in self.instances(pr, t, v):
                    if c["c"] is None:
                        continue
                    dmin = min(np.linalg.norm(c["c"] - p) for p in pred_list)
                    r = size(c) / max(ref_size, 1e-3)
                    if r > 1.7 or r < 0.4:       # a container / neighbour with a similar centre, not this object
                        continue
                    if dmin < g and (best is None or dmin < best[0]):
                        best = (dmin, c)
                if best:
                    chosen[v] = best[1]; pts.append(best[1]["P"])
            if len(pts) < min_views:
                if g >= 2.5 * gate and min_views >= 2 and allow_global[0]:     # global re-acquisition: size-consistent instance seen by >= 3 views
                    cand = []
                    for v in self.use_views:
                        for c in self.instances(pr, t, v):
                            if c["c"] is not None and 0.5 < size(c) / max(ref_size, 1e-3) < 1.6:
                                cand.append((v, c))
                    bestg = None
                    for v, c in cand:
                        agree = {}
                        for v2, c2 in cand:
                            dd = np.linalg.norm(c2["c"] - c["c"])
                            if dd < 0.03 and (v2 not in agree or dd < agree[v2][0]):
                                agree[v2] = (dd, c2)
                        if len(agree) >= 3 and (bestg is None or len(agree) > len(bestg)):
                            bestg = agree
                    if bestg:
                        ch = {v2: x[1] for v2, x in bestg.items()}
                        return ch, np.median(np.concatenate([x["P"] for x in ch.values()]), 0)
                    return {}, None
                if g < 2.5 * gate:          # re-detection with a wider gate, but only if >= 2 views agree
                    ch2, c2 = pick(t, pred_list, 2.5 * gate, 2)
                    if c2 is not None:
                        cs = [x["c"] for x in ch2.values()]
                        if np.max([np.linalg.norm(a - c2) for a in cs]) < 0.03:
                            return ch2, c2
                return {}, None
            P = np.concatenate(pts)
            return chosen, np.median(P, 0)

        def is_held(t, c):
            if c is None:
                return False
            dt = np.linalg.norm(self.tcp[t] - c)
            return dt < spec.get("hold_dist", 0.05)

        # grasp state machine on the fitted robot: rest -> held (gripper closed with its real fingertip
        # point within 3 cm of the object's observed points) -> carried rigidly with the hand -> released
        # (gripper opens 1.5 cm beyond its grasp width) -> rest.  'push' objects are never carried: they
        # are predicted to follow the hand's motion in the table plane.
        w = self.width
        w_open = float(np.max(w))
        static = spec.get("role") == "static"
        push = spec.get("mode") == "push"
        gpt = np.einsum("tij,j->ti", self.Th, np.array([0, 0, spec.get("finger_reach", 0.15), 1.0]))[:, :3]
        w_g = None
        last_pts = None
        for t in range(th, T):
            if t == th:
                chosen, c = pick(t, [c0]); c = c if c is not None else c0
                held[t] = False
                if chosen:            # reference size from all views (one view may see the object partially)
                    Pf = np.concatenate([x["P"] for x in chosen.values()])
                    ref_size = max(ref_size, float(np.quantile(np.linalg.norm(Pf - np.median(Pf, 0), axis=1), 0.9)))
            else:
                tp = t - 1
                if held[tp]:
                    Tm = self.Th[t] @ np.linalg.inv(self.Th[tp])
                    carried = Tm[:3, :3] @ cen[tp] + Tm[:3, 3]
                    preds = [carried, cen[tp]]
                    held[t] = not (w[t] > max(w_g + 0.025, w_open - 0.015) or w[t] < min(0.005, 0.5 * w_g) - 0.0005)
                    allow_global[0] = False
                    chosen, c = pick(t, preds)
                    allow_global[0] = True
                    # evidence of release: the object stays where it was while the hand moves away
                    if held[t] and c is not None and np.linalg.norm(c - cen[tp]) < 0.02 and \
                            np.linalg.norm(c - carried) > 0.03 and np.linalg.norm(gpt[t] - c) > 0.05:
                        held[t] = False
                    if c is None or (held[t] and np.linalg.norm(c - carried) > 0.03):
                        c = carried if held[t] else (c if c is not None else cen[tp])
                else:
                    preds = [cen[tp]]
                    if push:
                        dlt = gpt[t] - gpt[tp]; dlt -= (dlt @ self.n) * self.n
                        preds.append(cen[tp] + dlt)
                    near = last_pts is not None and len(last_pts) and \
                        np.min(np.linalg.norm(last_pts - gpt[t], axis=1)) < spec.get("hold_dist", 0.035 + 0.4 * ref_size)
                    closing = np.max(w[max(0, t - 3):t + 1]) > w_open - 0.012     # it was open just before
                    held[t] = (not static) and (not push) and w[t] < w_open - 0.004 and w[t] > 0.0005 \
                        and w[t] <= w[tp] + 0.003 and closing and near
                    if held[t]:
                        w_g = w[t]
                    chosen, c = pick(t, preds)
                    if c is None:
                        c = preds[0]
            sel[t], cen[t] = chosen, c
            if chosen:
                last_pts = np.concatenate([x["P"] for x in chosen.values()])[::5]
            elif last_pts is not None and t > th:
                last_pts = last_pts + (cen[t] - cen[t - 1])
        for t in range(th - 1, -1, -1):        # before the hint frame: assume at rest
            chosen, c = pick(t, [cen[t + 1]])
            sel[t], cen[t] = chosen, (c if c is not None else cen[t + 1])
        return sel, np.array(cen), held

    def support_h(self, t, insts):
        """height of what the object stands on: low quantile of the depth just around its masks"""
        hs = [c.get("ring_h", np.array([])) for c in insts.values()]
        hs = np.concatenate(hs) if hs else np.array([])
        return float(np.quantile(hs, 0.3)) if len(hs) > 20 else 0.0

    # -------------------------------------------------------------- shape
    def shape(self, spec, sel, cen, held):
        T = self.obs.T
        rest = [t for t in range(T) if not held[t] and np.linalg.norm(cen[t] - cen[0]) < 0.01 and sel[t]]
        if not rest:
            rest = [t for t in range(T) if sel[t]][:1]
        P = np.concatenate([np.concatenate([c["P"] for c in sel[t].values()]) for t in rest])
        # statistical clean-up: drop points far from the bulk
        med = np.median(P, 0)
        r = np.linalg.norm(P - med, axis=1)
        P = P[r < max(0.03, np.quantile(r, 0.9) * 1.5)]
        # density filter: keep points with >= 8 neighbours within 6 mm (removes flying pixels)
        if len(P) > 50:
            sub = P[np.random.default_rng(0).permutation(len(P))[:4000]]
            D2 = torch.cdist(torch.as_tensor(sub, device="cuda").float(), torch.as_tensor(sub, device="cuda").float())
            dens = (D2 < 0.006).sum(1).cpu().numpy()
            P = sub[dens >= 8] if (dens >= 8).sum() > 30 else sub
        kind = spec.get("shape", "box")
        base = float(np.median([self.support_h(t, sel[t]) for t in rest]))
        h_lo = float(np.quantile(P @ self.n + self.d, 0.05))
        if h_lo > 0.015 and kind != "sphere":   # lowest visible points well above the table: it rests on something
            base = max(base, h_lo - 0.003)
        base = 0.0 if base < 0.006 else base
        self.log(f"  {spec['name']}: support height {base * 1000:.0f} mm")
        if kind == "box":
            T0, half = box_shape(P, self.n, self.d, base)
            m = trimesh.creation.box(extents=2 * half)
        elif kind == "container":            # open-top box (tray / bin): floor + 4 walls, 4 mm thick
            T0, half = box_shape(P, self.n, self.d, base)
            m = container_mesh(half)
        elif kind == "sphere":
            T0, half = box_shape(P, self.n, self.d, base)
            r = float(np.mean([half[0], half[1], half[2]]))
            T0[:3, 3] += self.n * (base + r - (T0[:3, 3] @ self.n + self.d))
            half = np.array([r, r, r])
            m = trimesh.creation.icosphere(subdivisions=3, radius=r)
        elif kind == "mesh" and spec.get("mesh"):
            # SAM 3D initial asset: orientation from its predicted camera pose (hint view), size/position
            # from the fused multiview points; refined below with silhouettes (+ per-axis scale)
            import json as _json
            m = trimesh.load(spec["mesh"], force="mesh")
            m.apply_translation(-m.bounds.mean(0))
            js = _json.load(open(spec["mesh_json"]))
            Tc = np.asarray(js["T_obj2cam_opencv"], float)
            Rc = Tc[:3, :3] / np.cbrt(abs(np.linalg.det(Tc[:3, :3])))
            R0 = self.obs.c2w[spec["hint"]["v"]][:3, :3] @ Rc
            ctr = np.median(P, 0)
            Q = (P - ctr) @ R0
            ext_o = np.quantile(Q, 0.97, 0) - np.quantile(Q, 0.03, 0)
            ext_m = np.maximum(m.extents, 1e-6)
            sc = float(np.median(np.sort(ext_o / ext_m)[-2:]))       # two best-observed axes
            m.apply_scale(sc)
            T0 = np.eye(4); T0[:3, :3] = R0
            Vw = np.asarray(m.vertices) @ R0.T
            hmin = (Vw @ self.n).min()
            c_xy = ctr - (ctr @ self.n + self.d) * self.n
            T0[:3, 3] = c_xy + self.n * (base - hmin)
            half = m.extents / 2
        else:
            T0, half, m = hull_shape(P, self.n, self.d, base)
        self.log(f"  {spec['name']}: shape {kind} from {len(rest)} rest frames ({len(P)} pts), "
                 f"extents {np.round(2 * half * 1000, 1)} mm")
        return T0, half, m, rest

    # -------------------------------------------------------------- pose fit
    def fit_pose(self, t, T_init_list, Ps, Nm, sel_t, held_t, T_hold=None, iters=150, opt_scale=False,
                 extra=(), w_sil=1.0, w_depth=None):
        """6-DoF (+ optional per-axis scale) fit at frame t (and `extra` frames [(t, sel)] with the same pose).
        Silhouettes are the main signal (SAM 3 masks are sharper than the depth at object edges):
          out-of-mask: projected model points that are not occluded must fall inside the instance mask
          coverage   : the instance mask must be covered by the projected model (2D chamfer)
        plus robust 3D chamfer to the fused mask points and the multiview depth z-test (weaker)."""
        dev = "cuda"
        w_depth = getattr(self, "w_depth_obj", 0.3) if w_depth is None else w_depth
        frames = [(t, sel_t)] + list(extra)
        self.cur_support = self.support_h(t, sel_t) if sel_t else 0.0
        if sel_t:
            hl = float(np.quantile(np.concatenate([c["P"] for c in sel_t.values()]) @ self.n + self.d, 0.05))
            if hl > 0.015 and not getattr(self, "cur_sphere", False):
                self.cur_support = max(self.cur_support, hl - 0.003)
        self.cur_support = 0.0 if self.cur_support < 0.006 else self.cur_support
        rng = np.random.default_rng(0)
        obsP = [np.concatenate([c["P"] for c in st.values()]) for _, st in frames if st]
        obsP = np.concatenate(obsP) if obsP else np.zeros((0, 3))
        F = torch.as_tensor(obsP[rng.permutation(len(obsP))[:3000]], dtype=torch.float32, device=dev)
        Pm = torch.as_tensor(Ps, dtype=torch.float32, device=dev)
        Nm_ = torch.as_tensor(Nm, dtype=torch.float32, device=dev)
        n = torch.as_tensor(self.n, dtype=torch.float32, device=dev)
        sil = []          # (frame, view, distance map, mask pixel samples [K,2])
        for tf, st in frames:
            for v, c in st.items():
                m = c["m"]
                dt = torch.as_tensor(cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3), dtype=torch.float32, device=dev)
                ys, xs = np.nonzero(m)
                k = rng.permutation(len(xs))[:300]
                sil.append((tf, v, dt, torch.as_tensor(np.stack([xs[k], ys[k]], 1), dtype=torch.float32, device=dev)))
        T0 = torch.as_tensor(np.stack(T_init_list), dtype=torch.float32, device=dev)          # B,4,4
        B = len(T0)
        w = torch.zeros(B, 3, device=dev, requires_grad=True)
        tr = torch.zeros(B, 3, device=dev, requires_grad=True)
        ls = torch.zeros(B, 3, device=dev, requires_grad=True)
        groups = [{"params": [w], "lr": 0.02}, {"params": [tr], "lr": 0.003}]
        if opt_scale:
            groups.append({"params": [ls], "lr": 0.02})
        opt = torch.optim.Adam(groups)
        Th = torch.as_tensor(T_hold, dtype=torch.float32, device=dev) if T_hold is not None else None
        for it in range(iters):
            R = so3_exp(w) @ T0[:, :3, :3]
            ctr = T0[:, :3, 3] + tr
            P = torch.einsum("bij,bnj->bni", R, Pm[None] * ls.clamp(*self.sc_range).exp()[:, None]) + ctr[:, None]
            N = torch.einsum("bij,nj->bni", R, Nm_)
            s = 0.01 if it < iters // 2 else 0.005
            L = torch.zeros(B, device=dev)
            if len(F):
                dd = knn_points(F[None].expand(B, -1, -1).contiguous(), P, K=1).dists[..., 0].clamp(min=1e-12).sqrt()
                L = L + w_depth * rho(dd, s).mean(1)
            for tf in sorted({f for f, _ in frames}):
                L = L + w_depth * depth_term(self.views, P, N, [tf] * B, tau=0.015, s=s, empty_cost=1.0) / len(frames)
            if sil:
                u, vv, z = self.views.project(P)
                Ls = 0
                for tf, v, dt, mp in sil:
                    ui = u[:, v]; vi = vv[:, v]
                    inside = (ui >= 0) & (ui <= self.W - 1) & (vi >= 0) & (vi <= self.H - 1)
                    g = torch.stack([ui.clamp(0, self.W - 1) / (self.W - 1) * 2 - 1,
                                     vi.clamp(0, self.H - 1) / (self.H - 1) * 2 - 1], -1)[:, None]
                    dpx = torch.nn.functional.grid_sample(dt[None, None].expand(B, 1, -1, -1), g, align_corners=True)[:, 0, 0]
                    D, _ = self.views.sample(u[:, v:v + 1], vv[:, v:v + 1], [tf] * B)
                    occl = (D[:, 0] > 0) & (D[:, 0] < z[:, v] - 0.015)          # something in front of the model
                    wv = (inside & ~occl).float()
                    out = (rho(dpx / 4.0, 1.0) * wv).sum(1) / wv.sum(1).clamp(min=1)
                    uvm = torch.stack([ui, vi], -1)                             # B,N,2
                    dc = torch.cdist(mp[None].expand(B, -1, -1), uvm).min(-1).values   # B,K
                    cov = rho(dc / 4.0, 1.0).mean(1)
                    Ls = Ls + out + cov
                L = L + w_sil * Ls / len(sil)
            h = P @ n + self.d
            L = L + 50 * torch.relu(-h - 0.002).mean(1) + 200 * torch.relu(-h - 0.001).max(1).values  # no table penetration
            if not held_t:
                L = L + 5 * (h.min(1).values - self.cur_support).abs()        # resting on its support
            if Th is not None:
                L = L + 2 * ((ctr - Th[:3, 3]).norm(dim=-1) + 0.05 * (R - Th[:3, :3]).norm(dim=(1, 2)))
            if opt_scale:
                L = L + 0.05 * (ls ** 2).sum(1)
            opt.zero_grad(); L.sum().backward(); opt.step()
        with torch.no_grad():
            b = int(L.argmin())
            Tout = np.eye(4)
            Tout[:3, :3] = (so3_exp(w) @ T0[:, :3, :3])[b].cpu().numpy()
            Tout[:3, 3] = (T0[:, :3, 3] + tr)[b].cpu().numpy()
            self.last_scale = ls.clamp(*self.sc_range).exp()[b].cpu().numpy()
        return Tout, float(L[b].detach())

    def run(self):
        out = []
        T = self.obs.T
        for spec in self.ann["objects"]:
            try:
                sel, cen, held = self.track(spec)
                self.cur_sphere = spec.get("shape") == "sphere"
                thin = bool(spec.get("thin"))
                self.sc_range = (np.log(0.1), np.log(2.5)) if thin else (np.log(0.55), np.log(1.5))
                self.w_depth_obj = 0.05 if thin else 0.3
            except RuntimeError as ex:
                self.log(f"  SKIPPED {spec['name']}: {ex}")
                continue
            poses = np.zeros((T, 4, 4)); loss = np.zeros(T)
            static = spec.get("role") == "static"
            yaw = lambda a: np.array([[np.cos(a), -np.sin(a), 0, 0], [np.sin(a), np.cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            # candidate shapes: a SAM 3D asset competes with the 2.5D hull (its orientation can be wrong);
            # keep the one that explains the rest-frame silhouettes + depth best after refinement
            variants = [spec] + ([dict(spec, shape="hull")] if spec.get("shape") == "mesh" else [])
            best = None
            for sv in variants:
                T0, half, mesh, rest = self.shape(sv, sel, cen, held)
                Ps, Nm = mesh_samples(mesh, 1500)
                Ps = Ps.astype(np.float32); Nm = Nm.astype(np.float32)
                t_first = rest[0]
                init = [T0 @ yaw(a) for a in (0, np.pi / 2)] if sv.get("shape", "box") in ("box", "container") else \
                    [T0 @ yaw(a) for a in (0, np.pi / 2, np.pi, -np.pi / 2)] if sv.get("shape") == "hull" else [T0]
                extra = [(tt, sel[tt]) for tt in rest[1:] if sel[tt]][:: max(1, len(rest) // 3)][:3]
                Tf, _ = self.fit_pose(t_first, init, Ps, Nm, sel[t_first], held[t_first], iters=250, opt_scale=True,
                                      extra=extra)
                sc = self.last_scale
                if sv.get("shape", "box") == "sphere":
                    sc = np.full(3, float(np.mean(sc)))
                half = np.asarray(half) * sc
                if sv.get("shape") == "container":
                    mesh = container_mesh(half)
                    Ps, Nm = mesh_samples(mesh, 1500); Ps = Ps.astype(np.float32); Nm = Nm.astype(np.float32)
                else:
                    mesh = mesh.copy(); mesh.vertices = np.asarray(mesh.vertices) * sc
                    Ps = (Ps * sc).astype(np.float32)
                Tp, lp = self.fit_pose(t_first, [Tf], Ps, Nm, sel[t_first], held[t_first], extra=extra)
                self.log(f"  {spec['name']}: [{sv.get('shape', 'box')}] silhouette shape refinement scale {np.round(sc, 3)} "
                         f"-> extents {np.round(2 * half * 1000, 1)} mm, loss {lp:.4f}")
                if best is None or lp < best[0]:
                    best = (lp, Tp, half, mesh, Ps, Nm, rest, t_first, sv.get("shape", "box"))
            lp, Tp, half, mesh, Ps, Nm, rest, t_first, kind_used = best
            if len(variants) > 1:
                self.log(f"  {spec['name']}: using {kind_used}")
            poses[t_first], loss[t_first] = Tp, lp
            if static:
                for t in range(T):
                    poses[t] = poses[t_first]
            else:
                for direction in (range(t_first + 1, T), range(t_first - 1, -1, -1)):
                    for t in direction:
                        tp = t - 1 if t > t_first else t + 1
                        inits = [poses[tp]]
                        T_hold = None
                        if held[tp] or held[t]:
                            Tm = self.Th[t] @ np.linalg.inv(self.Th[tp])
                            inits.append(Tm @ poses[tp])
                            if held[tp] and held[t]:
                                T_hold = Tm @ poses[tp]
                        # re-detection: object re-appears somewhere new (centre of the fused masks)
                        if sel[t]:
                            c = np.median(np.concatenate([x["P"] for x in sel[t].values()]), 0)
                            Tc = poses[tp].copy(); Tc[:3, 3] += c - (poses[tp][:3, :3] @ Ps.mean(0) + poses[tp][:3, 3])
                            inits.append(Tc)
                        inits += [p @ yaw(a) for p in inits[:1] for a in (-0.3, 0.3)]
                        if not sel[t] and T_hold is not None:       # fully occluded in the hand: rigid with it
                            poses[t] = T_hold
                            continue
                        poses[t], loss[t] = self.fit_pose(t, inits, Ps, Nm, sel[t], held[t], T_hold)
            self.log(f"  {spec['name']}: role {spec.get('role')}, held frames {np.nonzero(held)[0].tolist()}, "
                     f"views/frame {[len(s) for s in sel]}, travel {np.linalg.norm(poses[-1][:3, 3] - poses[0][:3, 3]) * 100:.1f} cm")
            sup = np.zeros(T)
            for t in range(T):
                if sel[t]:
                    sp = self.support_h(t, sel[t])
                    hl = float(np.quantile(np.concatenate([c["P"] for c in sel[t].values()]) @ self.n + self.d, 0.05))
                    sp = max(sp, hl - 0.003) if (hl > 0.015 and not self.cur_sphere) else sp
                    sup[t] = 0.0 if sp < 0.006 else sp
                else:
                    sup[t] = sup[t - 1] if t else 0.0
            out.append(dict(name=spec["name"], type="static" if static else "free", half=half, poses=poses, support=sup,
                            mesh=(np.asarray(mesh.vertices), np.asarray(mesh.faces)), held=held, loss=loss,
                            sel={t: {v: int(c["i"]) for v, c in sel[t].items()} for t in range(T)},
                            world_samples=np.concatenate([(poses[t][:3, :3] @ Ps.T).T + poses[t][:3, 3] for t in range(T)])))
        return out
