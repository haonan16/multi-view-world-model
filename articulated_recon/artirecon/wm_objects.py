"""Object tracking for world-model predictions (ManiSkill): small, saturated objects whose predicted depth is
unreliable.  The object's position in each frame comes from multiview TRIANGULATION of its mask centroids
(rays from the calibrated cameras, no depth), chosen by a RANSAC-style vote over cross-view pairs and gated by
the previous position / the gripper.  Shape and pose then reuse ObjectsV2 (silhouette-driven fit)."""
import numpy as np
from .objects_v2 import ObjectsV2


def _ray(K, c2w, uv):
    d = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
    d = c2w[:3, :3] @ d
    return c2w[:3, 3], d / np.linalg.norm(d)


def _tri(rays):
    """least-squares point closest to all rays"""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in rays:
        M = np.eye(3) - np.outer(d, d)
        A += M; b += M @ o
    return np.linalg.lstsq(A, b, rcond=None)[0]


class WMObjects(ObjectsV2):
    def _cands(self, pr, t):
        out = {}
        for v in self.use_views:
            cs = []
            for c in self.instances(pr, t, v):
                ys, xs = np.nonzero(c["m"])
                if len(xs) < 15:                 # specks of colour in generated frames (a cube is ~100+ px)
                    continue
                c = dict(c, uv=np.array([xs.mean(), ys.mean()]), area=len(xs))
                dup = [k for k, e in enumerate(cs) if np.linalg.norm(e["uv"] - c["uv"]) < 4]     # same blob from 2 mask sources
                if dup:
                    if c["area"] > cs[dup[0]]["area"]:
                        cs[dup[0]] = c
                    continue
                cs.append(c)
            out[v] = cs
        return out

    def _proj(self, v, X):
        pc = np.linalg.inv(self.obs.c2w[v]) @ np.r_[X, 1]
        if pc[2] < 0.05:
            return None
        return (self.obs.K[v] @ pc[:3])[:2] / pc[2]

    def _locate(self, cands, preds, gate, ray0=None):
        """-> (chosen {v: cand}, X) best triangulated hypothesis near a prediction"""
        hyps = []
        vs = [v for v in cands if cands[v]]
        if vs and all(len(cands[v]) == 1 for v in vs) and len(vs) >= 2:
            gate = max(gate, 0.30)            # one blob of that colour per view: trust the multiview consensus
        for i, v1 in enumerate(vs):
            for v2 in vs[i + 1:]:
                for c1 in cands[v1]:
                    for c2 in cands[v2]:
                        X = _tri([_ray(self.obs.K[v1], self.obs.c2w[v1], c1["uv"]), _ray(self.obs.K[v2], self.obs.c2w[v2], c2["uv"])])
                        hyps.append(X)
        if ray0 is not None:                                   # hint click ray at the hint frame
            hyps = [X for X in hyps if np.linalg.norm(np.cross(X - ray0[0], ray0[1])) < 0.03]
        best = None
        for X in hyps:
            if (X @ self.n + self.d) < -0.01:
                continue
            dpred = min(np.linalg.norm(X - p) for p in preds) if preds else 0.0
            if dpred > gate:
                continue
            chosen, err = {}, 0.0
            for v in vs:
                uv = self._proj(v, X)
                if uv is None:
                    continue
                cbest = min(cands[v], key=lambda c: np.linalg.norm(c["uv"] - uv))
                e = np.linalg.norm(cbest["uv"] - uv)
                if e < (15.0 if len(cands[v]) == 1 else max(8.0, 0.9 * np.sqrt(cbest["area"]))):   # a unique blob of that colour
                    chosen[v] = cbest; err += e
            if len(chosen) < 2:
                continue
            Xr = _tri([_ray(self.obs.K[v], self.obs.c2w[v], c["uv"]) for v, c in chosen.items()])
            score = len(chosen) - (err / len(chosen)) / 6.0 - 25.0 * dpred      # views, px error, distance to the prediction (m)
            if best is None or score > best[0]:
                best = (score, chosen, Xr)
        if best is None:                                       # single view: ray at the predicted height
            for v in vs:
                for c in cands[v]:
                    o, d = _ray(self.obs.K[v], self.obs.c2w[v], c["uv"])
                    for p in preds:
                        h = p @ self.n + self.d
                        s = (h - (o @ self.n + self.d)) / (d @ self.n)
                        X = o + s * d
                        dp = np.linalg.norm(X - p)
                        if dp < 0.5 * gate and (best is None or dp < best[0]):
                            best = (dp, {v: c}, X)
            if best is None:
                return {}, None
        return best[1], best[2]

    def track(self, spec):
        T = self.obs.T
        self.cube_side = None
        pr = spec["prompt"]
        th, vh, (uh, vvh) = spec["hint"]["t"], spec["hint"]["v"], spec["hint"]["uv"]
        static = spec.get("role") == "static"
        w = self.width; w_open = float(np.max(w))
        sel, cen, held = [None] * T, [None] * T, np.zeros(T, bool)
        ray0 = _ray(self.obs.K[vh], self.obs.c2w[vh], (uh, vvh))
        cands0 = self._cands(pr, th)
        unique = all(len(c) <= 1 for c in cands0.values())
        # unique colour blobs: the prediction's multiview consensus (no anchoring to the input view); else the click
        ch, X = self._locate(cands0, [], 1e9, ray0=None if unique else ray0)
        if X is None:      # rest prior: the clicked instance's ray meets its resting height (depth of its points, else 2 cm)
            hit = [c for c in cands0.get(vh, []) if c["m"][int(vvh), int(uh)]] or \
                sorted(cands0.get(vh, []), key=lambda c: np.linalg.norm(c["uv"] - [uh, vvh]))[:1]
            if not hit:
                raise RuntimeError(f"{spec['name']}: no instance at the hint")
            c = hit[0]
            h = float(np.median(c["P"] @ self.n + self.d)) if len(c["P"]) > 10 else 0.02
            o, dd = _ray(self.obs.K[vh], self.obs.c2w[vh], c["uv"])
            X = o + (h - (o @ self.n + self.d)) / (dd @ self.n) * dd
            ch = {vh: c}
            self.log(f"  {spec['name']}: hint by single-view ray at height {h * 1000:.0f} mm")
        sel[th], cen[th] = ch, X
        for t in range(th + 1, T):
            tp = t - 1
            Tm = self.Th[t] @ np.linalg.inv(self.Th[tp])
            carried = Tm[:3, :3] @ cen[tp] + Tm[:3, 3]
            near = np.linalg.norm(self.tcp[tp] - cen[tp]) < spec.get("hold_dist", 0.04)
            preds = [cen[tp]] + ([carried] if (near or held[tp]) and not static else []) + \
                ([self.tcp[t]] if not static and np.linalg.norm(self.tcp[t] - cen[tp]) < 0.25 else [])   # carried by the gripper
            ch, X = self._locate(self._cands(pr, t), preds, 0.05 if static else 0.10)
            if static:
                X = cen[tp] if X is None else X
            if not static:
                closed = w[t] < w_open - 0.004
                moved_with = X is not None and np.linalg.norm(X - carried) < 0.025 and np.linalg.norm(carried - cen[tp]) > 0.02
                at_tcp = np.linalg.norm(self.tcp[t] - (X if X is not None else carried)) < spec.get("hold_dist", 0.04)
                held[t] = closed and at_tcp and (held[tp] or moved_with or (X is not None and (X @ self.n + self.d) > 0.04))
                if held[t] and (X is None or np.linalg.norm(X - carried) > 0.03):
                    X = carried
                if X is None:
                    X = cen[tp]
            sel[t], cen[t] = ch, X
        for t in range(th - 1, -1, -1):
            ch, X = self._locate(self._cands(pr, t), [cen[t + 1]], 0.03)
            sel[t], cen[t] = ch, (X if X is not None else cen[t + 1])
        if spec.get("shape") == "cube":
            sides = []
            for t in range(T):
                if held[t]:
                    continue
                for v, c in sel[t].items():
                    pc = np.linalg.inv(self.obs.c2w[v]) @ np.r_[cen[t], 1]
                    if pc[2] > 0.05 and c["area"] >= 9:
                        sides.append(np.sqrt(c["area"]) * pc[2] / self.obs.K[v][0, 0] / 1.35)   # sqrt(blob area)*z/f = 1.35 x side (measured on real renders, n=160)
            if sides:
                self.cube_side = float(np.median(sides))
                self.log(f"  {spec['name']}: cube side from {len(sides)} silhouettes {self.cube_side * 1000:.1f} mm")
        self.log(f"  {spec['name']}: triangulated, views/frame {[len(s) for s in sel]}, held {np.nonzero(held)[0].tolist()}")
        return sel, np.array(cen), held
