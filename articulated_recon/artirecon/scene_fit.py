"""Full reconstruction of the dynamic parts: robot (URDF) + manipulated objects (+ articulation)."""
import time

import numpy as np
import torch

from .fit_robot import RobotFitter, prepare
from .geom import DepthViews
from .objects import (box_surface, box_symmetries, cluster, fit_pose, gravity_box, knn_d, presence,
                      residuals, voxel_down)
from .robot import URDFRobot


def hand_T(robot, q, base):
    with torch.no_grad():
        qt = torch.as_tensor(q, dtype=torch.float32, device="cuda")
        bt = torch.as_tensor(base, dtype=torch.float32, device="cuda")[None].expand(len(qt), 4, 4)
        T = robot.fk(qt, bt)
    return T["panda_hand"].cpu().numpy(), T["panda_leftfinger"].cpu().numpy(), T["panda_rightfinger"].cpu().numpy()


def tcp_points(robot, q, base):
    Th, Tl, Tr = hand_T(robot, q, base)
    off = np.array([0, 0, 0.1034, 1.0])
    return (Th @ off)[:, :3]


class SceneFitter:
    def __init__(self, obs, log=print, robot=None, calib=False, known_base=None):
        self.obs, self.log, self.calib, self.known_base = obs, log, calib, known_base
        self.robot = robot or URDFRobot()
        self.timing = {}

    def run(self):
        t0 = time.time()
        obs, log = self.obs, self.log
        self.noise = 0.0
        if self.calib:
            # GT-free: static table -> per (frame, view) depth scale, and the depth noise level
            from .geom import table_calibrate
            import copy
            S, _, self.noise = table_calibrate(obs, log=log)
            obs = self.obs = copy.copy(obs)
            obs.depth = obs.depth * S[:, :, None, None].astype(np.float32)
            offs, nrm = table_calibrate.offsets
            obs.c2w = obs.c2w.copy()
            for v in range(obs.V):                      # shift each camera so its table matches the shared plane
                obs.c2w[v][:3, 3] -= offs[v] * nrm
        self.prep = prep = prepare(obs, fg_thr=max(0.008, 3 * self.noise))
        self.views = DepthViews(obs.depth, obs.K, obs.c2w)
        log(f"[{obs.label}] T={obs.T} V={obs.V}  plane n={np.round(prep['n'], 3)}")
        rf = RobotFitter(obs, prep, self.robot, log=log, calib=self.calib, known_base=self.known_base)
        self.rob = rob = rf.run()
        if self.calib and rob.get("depth_scale") is not None:
            # correct the observed depth with the robot-calibrated scales, re-lift for the object stage
            import copy
            obs = self.obs = copy.copy(obs)
            obs.depth = obs.depth * rob["depth_scale"][:, :, None, None].astype(np.float32)
            n_, d_ = prep["n"], prep["d"]
            self.prep = prep = prepare(obs, fg_thr=max(0.008, 3 * self.noise))
            self.views = DepthViews(obs.depth, obs.K, obs.c2w)
        self.timing["robot_s"] = time.time() - t0
        t1 = time.time()
        R, Prob = residuals(prep, self.robot, rob["q"], rob["base"], thr=max(0.015, 2.5 * self.noise))
        self.res = R
        tcp = tcp_points(self.robot, rob["q"], rob["base"])
        self.tcp = tcp
        n, d = prep["n"], prep["d"]
        C0 = cluster(voxel_down(R[0], 0.005), eps=0.025, min_pts=15)
        # a manipulable / articulated object fits in ~0.6 m; larger blobs are scenery (or depth artefacts)
        C0 = [c for c in C0 if np.linalg.norm(np.quantile(c, 0.98, 0) - np.quantile(c, 0.02, 0)) < 0.8]
        log(f"  frame-0 residual clusters: {[len(c) for c in C0]}")
        T = obs.T
        objs = []
        # clusters touching the robot at frame 0 are robot hardware the URDF lacks (cables, mounts, wrist camera)
        Lk = self.robot.link_of.cpu().numpy()
        hand_links = {self.robot.mesh_links.index(l) for l in ("panda_hand", "panda_leftfinger", "panda_rightfinger")
                      if l in self.robot.mesh_links}
        Th_all, _, _ = hand_T(self.robot, rob["q"], rob["base"])
        keep, self.attached = [], []
        for k, c in enumerate(C0):
            dd = torch.as_tensor(c, dtype=torch.float32, device="cuda")[None]
            kn = __import__("pytorch3d.ops", fromlist=["knn_points"]).knn_points(
                dd, torch.as_tensor(Prob[0], dtype=torch.float32, device="cuda")[None], K=1)
            dist = kn.dists[0, :, 0].sqrt().cpu().numpy()
            lk = np.bincount(Lk[kn.idx[0, :, 0].cpu().numpy()]).argmax()
            near_hand = lk in hand_links and np.median(dist) < 0.05 and np.quantile(c @ prep["n"] + prep["d"], 0.02) > 0.03
            if near_hand or (np.median(dist) < max(0.03, 3 * self.noise) and (dist < 0.02).mean() > 0.3):
                if lk in hand_links:
                    T0b, hb = gravity_box(c, prep["n"], prep["d"], rest_on_table=False)
                    poses = np.stack([Th_all[t] @ np.linalg.inv(Th_all[0]) @ T0b for t in range(obs.T)])
                    self.attached.append(dict(name=f"robot_attachment_{k}", type="attachment", half=hb, poses=poses,
                                              rel_hand=np.stack([np.linalg.inv(Th_all[0]) @ T0b] * obs.T),
                                              world_samples=np.zeros((0, 3))))
                    log(f"  cluster {k}: touches the hand at frame 0 -> rigid robot attachment (e.g. wrist camera)")
                else:
                    log(f"  cluster {k}: touches robot link {self.robot.mesh_links[lk]} at frame 0 -> unmodelled robot hardware, ignored")
                continue
            keep.append((k, c))
        for k, c in keep:
            st = np.stack([presence(self.views, c, t, tau_in=max(0.008, 2.5 * self.noise),
                                    tau_out=max(0.015, 4 * self.noise)) for t in range(T)])          # T, n
            absent = (st == -1).mean(1)
            present = (st == 1).mean(1)
            near_tcp = min(np.min(knn_d(tcp[t:t + 1], c)) for t in range(T))
            ctr = c.mean(0)
            objs.append(dict(id=k, pts0=c, absent=absent, present=present, near_tcp=near_tcp, centre=ctr,
                             status=st))
            log(f"  cluster {k}: n={len(c)} centre={np.round(ctr, 3)} absent/frame={np.round(absent, 2)} "
                f"tcp_min={near_tcp * 100:.1f}cm")
        moving = [o for o in objs if o["absent"].max() > 0.05 and (o["status"] == -1).sum(1).max() > 25
                  and o["near_tcp"] < 0.06]       # relevant = touched by the gripper at some frame
        for o in objs:
            if o["absent"].max() > 0.05 and o["near_tcp"] >= 0.06 and (o["status"] == -1).sum(1).max() > 25:
                log(f"  cluster {o['id']}: moves but never within 6 cm of the gripper -> not manipulated, ignored")
        self.objects = list(self.attached)
        for o in moving:
            frac_moved = o["absent"].max()
            tstar = int(o["absent"].argmax())
            moved_mask = o["status"][tstar] == -1
            # articulated if a substantial part stays (present) while another part leaves
            stay = (o["status"] == 1).all(0) | ((o["status"] >= 0).all(0) & (o["status"] == 1).any(0))
            if stay.mean() > 0.3 and moved_mask.mean() < 0.7:
                self.objects.append(self.fit_articulated(o, tstar))
            else:
                self.objects.append(self.fit_free(o))
        # static clusters touched by a moving part / the gripper -> relevant context (e.g. target)
        moved_paths = [ob["world_samples"] for ob in self.objects]
        for o in objs:
            if o in moving:
                continue
            dmin = min([np.min(knn_d(o["pts0"][::3], p)) for p in moved_paths] + [o["near_tcp"]])
            if dmin < 0.03:
                self.objects.append(self.fit_static(o, dmin))
        from .objects import infer_hole
        for so in [x for x in self.objects if x["type"] == "static"]:
            for fo in [x for x in self.objects if x["type"] == "free"]:
                h = infer_hole(so, fo)
                if h:
                    so["hole"] = h
                    log(f"  {so['name']}: {h['note']}; hole half {h['half']} m along axis {h['axis']}")
        self.timing["objects_s"] = time.time() - t1
        self.timing["total_s"] = time.time() - t0
        log(f"  objects: {[(ob['name'], ob['type']) for ob in self.objects]}  [{self.timing['total_s']:.1f}s]")
        return self

    # ------------------------------------------------------------------------------------------
    def new_points(self, t, static_pts):
        """residual points of frame t not explained by frame-0 static geometry (moved stuff)"""
        Rt = self.res[t]
        return Rt[knn_d(Rt, static_pts) > 0.01]

    def fit_free(self, o):
        """free rigid body: box shape from frame 0, pose per frame (prev / hand-carried / re-detected)"""
        n, d = self.prep["n"], self.prep["d"]
        rest = float(np.quantile(o["pts0"] @ n + d, 0.02)) < 0.015      # resting on the table at frame 0?
        T0, half = gravity_box(o["pts0"], n, d, rest_on_table=rest)
        Pl, Nl = box_surface(half)
        T = self.obs.T
        Th, _, _ = hand_T(self.robot, self.rob["q"], self.rob["base"])
        poses = [T0]
        static0 = np.concatenate(self.res[:1])
        losses = [0.0]
        R0 = self.res[0]
        newpts = [None] + [self.res[t][knn_d(self.res[t], R0) > 0.01] for t in range(1, T)]
        for t in range(1, T):
            prev = poses[-1]
            if o["absent"][t] < 0.05:              # nothing seen through: the object has not moved
                poses.append(prev.copy())
                losses.append(losses[-1])
                continue
            inits = []
            for S in box_symmetries():
                A = prev.copy(); A[:3, :3] = prev[:3, :3] @ S
                inits.append(A)
                B = Th[t] @ np.linalg.inv(Th[t - 1]) @ A          # carried by the hand
                inits.append(B)
            newp = newpts[t]
            if len(newp) > 30:
                # nearest moved blob to the gripper: PCA-aligned box hypothesis
                cl = [c for c in [newp]]
                c = newp[knn_d(newp, self.tcp[t:t + 1]) < 0.25] if (knn_d(newp, self.tcp[t:t + 1]) < 0.25).sum() > 30 else newp
                mu = c.mean(0)
                _, _, vt = np.linalg.svd(c - mu, full_matrices=False)
                order = np.argsort(-half)                         # longest box axis <-> first PC
                for S in box_symmetries():
                    R = np.zeros((3, 3))
                    for i, ax in enumerate(order):
                        R[:, ax] = vt[i]
                    if np.linalg.det(R) < 0:
                        R[:, order[2]] *= -1
                    A = np.eye(4); A[:3, :3] = R @ S; A[:3, 3] = mu
                    inits.append(A)
            Rt = newpts[t] if len(newpts[t]) > 30 else self.res[t]   # only what moved (static scenery removed)
            # observed points within reach of any hypothesis
            cand_centres = np.stack([a[:3, 3] for a in inits])
            sel = np.min(np.linalg.norm(Rt[:, None] - cand_centres[None], axis=-1), 1) < np.linalg.norm(half) + 0.08
            Tb, L, _ = fit_pose(self.views, t, Rt[sel] if sel.sum() > 20 else Rt, Pl, Nl, inits)
            poses.append(Tb)
            losses.append(L)
        poses = np.stack(poses)
        # shape refinement: all frames' nearby residual points in the box frame
        loc = []
        Ls = np.array(losses[1:]) if T > 1 else np.array([0.0])
        good = [0] + [t for t in range(1, T) if o["absent"][t] >= 0.05 and losses[t] <= 1.5 * max(Ls.min(), 0.05)]
        for t in good:
            Pw = self.res[t] if t == 0 else newpts[t]
            Pb = (Pw - poses[t][:3, 3]) @ poses[t][:3, :3]
            m = np.all(np.abs(Pb) < half + 0.01, 1)
            loc.append(Pb[m])
        loc = np.concatenate(loc)
        self.log(f"  free body: half={np.round(half * 1000, 1)}mm, {len(loc)} pts for refinement")
        if len(loc) > 200:
            lo = np.quantile(loc, 0.02, 0)
            hi = np.quantile(loc, 0.98, 0)
            c = (lo + hi) / 2
            half2 = np.maximum((hi - lo) / 2, 0.004)
            half = half2
            poses[:, :3, 3] += np.einsum("tij,j->ti", poses[:, :3, :3], c)
            Pl, Nl = box_surface(half)
            # re-track with the refined shape
            for t in range(T):
                if t > 0 and o["absent"][t] < 0.05:
                    poses[t] = poses[t - 1]
                    continue
                src = self.res[t] if t == 0 or len(newpts[t]) <= 30 else newpts[t]
                near = src[np.linalg.norm(src - poses[t][:3, 3], axis=1) < np.linalg.norm(half) + 0.05]
                if len(near) > 20:
                    Tb, L, _ = fit_pose(self.views, t, near, Pl, Nl, [poses[t]], iters=150, lr=0.005)
                    poses[t] = Tb
        # grasp closure: on frames where the body is lifted off the table and lies between the fingers, its
        # width along the finger axis must equal the gap between the finger pads (contact, no penetration)
        poses, half = self.grasp_closure(poses, half)
        Pl, Nl = box_surface(half)
        # grasp / attachment: object rigid w.r.t. the hand
        rel = [np.linalg.inv(Th[t]) @ poses[t] for t in range(T)]
        relt = np.stack([r[:3, 3] for r in rel])
        med = np.median(relt, 0)
        ok = np.linalg.norm(relt - med, axis=1) < 0.015
        if ok[0] and ok.mean() >= 0.6:   # rigid with the hand from the first frame on: part of the robot
            tref = int(np.where(ok)[0][0])
            poses = np.stack([Th[t] @ rel[tref] for t in range(T)])
            rel = [rel[tref]] * T
            ws = np.concatenate([(poses[t][:3, :3] @ Pl.T).T + poses[t][:3, 3] for t in range(T)])
            self.log(f"  object_{o['id']} moves rigidly with the hand in every frame -> robot attachment")
            return dict(name=f"robot_attachment_{o['id']}", type="attachment", half=half, poses=poses,
                        rel_hand=np.stack(rel), world_samples=ws)
        attached = []
        for t in range(T):
            tcp_d = np.linalg.norm(self.tcp[t] - poses[t][:3, 3])
            attached.append(bool(tcp_d < np.linalg.norm(half) + 0.02))
        ws = np.concatenate([(poses[t][:3, :3] @ Pl.T).T + poses[t][:3, 3] for t in range(T)])
        return dict(name=f"object_{o['id']}", type="free", half=half, poses=poses, losses=losses,
                    grasped=attached, world_samples=ws, rel_hand=np.stack(rel))

    def grasp_closure(self, poses, half):
        n, d = self.prep["n"], self.prep["d"]
        rob = self.robot
        with torch.no_grad():
            qt = torch.as_tensor(self.rob["q"], dtype=torch.float32, device="cuda")
            bt = torch.as_tensor(self.rob["base"], dtype=torch.float32, device="cuda")[None].expand(len(qt), 4, 4)
            P, _ = rob.surface(qt, bt)
        P = P.cpu().numpy()
        Lk = rob.link_of.cpu().numpy()
        li, ri = rob.mesh_links.index("panda_leftfinger"), rob.mesh_links.index("panda_rightfinger")
        gaps, shifts, axes = [], [], []
        T0b = poses[0]
        c0 = (np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * half) @ T0b[:3, :3].T + T0b[:3, 3]
        if (c0 @ n + d).min() > 0.005:          # already off the table at frame 0 (e.g. hand-mounted): skip
            return poses, half
        for t in range(len(poses)):
            Tb = poses[t]
            corners = (np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * half) @ Tb[:3, :3].T + Tb[:3, 3]
            if (corners @ n + d).min() < 0.005:
                continue                                           # resting on the table: not held
            Lp, Rp = P[t][Lk == li], P[t][Lk == ri]
            a = Lp.mean(0) - Rp.mean(0)
            a /= np.linalg.norm(a)                                 # finger closing axis
            c = Tb[:3, 3]
            # finger material overlapping the object's footprint (pads), inner faces along the axis
            loc = lambda Q: (Q - c) @ Tb[:3, :3]
            inL = np.all(np.abs(loc(Lp))[:, [i for i in range(3)]] < half + 0.02, 1)
            inR = np.all(np.abs(loc(Rp)) < half + 0.02, 1)
            if inL.sum() < 10 or inR.sum() < 10:
                continue
            l_in = ((Lp[inL] - c) @ a).min()                      # left finger inner face (positive side)
            r_in = ((Rp[inR] - c) @ a).max()                      # right finger inner face (negative side)
            if l_in <= r_in:
                continue
            gaps.append(l_in - r_in); shifts.append((l_in + r_in) / 2); axes.append((t, a))
        if len(gaps) == 0:
            return poses, half
        gap = float(np.median(gaps))
        t0, a0 = axes[0]
        k = int(np.argmax(np.abs(poses[t0][:3, :3].T @ a0)))       # box axis along the finger axis
        if not (0.3 * half[k] * 2 < gap < 1.5 * half[k] * 2):
            return poses, half
        old = half[k]
        # contact compliance of the simulator's fingers (calibrated: ~3.3 mm), so width in [gap, gap + 4 mm]
        half = half.copy(); half[k] = float(np.clip(half[k], gap / 2, gap / 2 + 0.002))
        poses = poses.copy()
        for (t, a), sh in zip(axes, shifts):
            poses[t][:3, 3] = poses[t][:3, 3] + sh * a             # centre the body between the pads
        self.log(f"  grasp closure on {len(gaps)} held frames: finger gap {gap * 1000:.1f} mm, width along finger axis {old * 2000:.1f} -> {half[k] * 2000:.1f} mm, centred between pads")
        return poses, half

    def fit_static(self, o, dmin):
        n, d = self.prep["n"], self.prep["d"]
        T0, half = gravity_box(o["pts0"], n, d, rest_on_table=True)
        poses = np.stack([T0] * self.obs.T)
        Pl, _ = box_surface(half)
        return dict(name=f"object_{o['id']}", type="static", half=half, poses=poses, contact_cm=dmin * 100,
                    world_samples=(T0[:3, :3] @ Pl.T).T + T0[:3, 3])

    def fit_articulated(self, o, tstar):
        """static parent + moving child; joint type/axis by search over the parent box axes."""
        n, d = self.prep["n"], self.prep["d"]
        st = o["status"]
        P0 = o["pts0"]
        moved = st[tstar] == -1
        parentP = P0[~moved]
        Tp, hp = gravity_box(parentP, n, d, rest_on_table=True)
        childP0 = P0[moved]
        T = self.obs.T
        # prismatic hypotheses along +-parent axes, revolute about parent edges (vertical/horizontal)
        new_t = self.new_points(tstar, parentP)
        new_t = new_t[np.linalg.norm(new_t - P0.mean(0), axis=1) < 0.6]
        best = None
        for ax in range(3):
            for sg in (1, -1):
                a = Tp[:3, ax] * sg
                for s in np.linspace(0.01, 0.4, 79):
                    Q = childP0 + s * a
                    sc = (knn_d(Q, new_t) < 0.01).mean() if len(new_t) else 0
                    if best is None or sc > best[0]:
                        best = (sc, a, s)
        sc, axis, _ = best
        # per-frame joint value: 1-D search + depth evidence
        vals = []
        for t in range(T):
            nt = self.new_points(t, parentP)
            nt = nt[np.linalg.norm(nt - P0.mean(0), axis=1) < 0.6]
            ss = np.linspace(0, 0.4, 161)
            scs = [(knn_d(childP0 + s * axis, nt) < 0.008).mean() if len(nt) else 0 for s in ss]
            # frame where the child is (nearly) unmoved: present test
            stay = (st[t][moved] == 1).mean()
            vals.append(0.0 if stay > 0.5 else float(ss[int(np.argmax(scs))]))
        vals = np.array(vals)
        # child shape: points of every frame mapped back to the closed configuration
        acc = [childP0]
        for t in range(T):
            nt = self.new_points(t, parentP)
            nt = nt[np.linalg.norm(nt - P0.mean(0), axis=1) < 0.6] - vals[t] * axis
            acc.append(nt[knn_d(nt, childP0) < 0.5])
        acc = np.concatenate(acc)
        # child box aligned with the parent axes, bounded by what moved
        Rl = Tp[:3, :3]
        loc = (acc - Tp[:3, 3]) @ Rl
        loc = loc[np.all(np.abs(loc) < hp + 0.05, 1)] if len(loc) else loc
        lo, hi = np.quantile(loc, 0.01, 0), np.quantile(loc, 0.99, 0)
        Tc = np.eye(4); Tc[:3, :3] = Rl; Tc[:3, 3] = Tp[:3, 3] + Rl @ ((lo + hi) / 2)
        hc = np.maximum((hi - lo) / 2, 0.005)
        poses_c = np.stack([Tc.copy() for _ in range(T)])
        for t in range(T):
            poses_c[t][:3, 3] += vals[t] * axis
        self.log(f"  articulated: prismatic axis {np.round(axis, 3)}, match {sc:.2f}, travel per frame {np.round(vals, 3)}")
        # container inference: surfaces observed well inside the child box -> it cannot be solid
        from .objects import shell_slabs, tray_slabs
        locc = (acc - Tc[:3, 3]) @ Rl
        inner = np.all(np.abs(locc) < hc - 0.015, 1)
        up = int(np.argmax(np.abs(Rl.T @ n)))
        usgn = int(np.sign((Rl.T @ n)[up]))
        child_slabs, parent_slabs = None, None
        if inner.mean() > 0.08:
            zf = float(np.quantile(locc[inner][:, up] * usgn, 0.05)) * usgn
            child_slabs = tray_slabs(hc, up, usgn, floor_top=zf * usgn if usgn > 0 else None)
            self.log(f"  child: {inner.mean():.0%} of its observed points lie inside the box -> open-top container "
                     f"(floor at {zf * 1000:.0f} mm)")
        ax_l = Rl.T @ axis
        k = int(np.argmax(np.abs(ax_l)))
        parent_slabs = shell_slabs(hp, k, int(np.sign(ax_l[k])))      # the child slides out through this face
        Pl, _ = box_surface(hc)
        ws = np.concatenate([(poses_c[t][:3, :3] @ Pl.T).T + poses_c[t][:3, 3] for t in range(T)])
        return dict(name=f"object_{o['id']}", type="articulated", parent=dict(half=hp, pose=Tp, slabs=parent_slabs),
                    child=dict(half=hc, poses=poses_c, slabs=child_slabs), joint=dict(type="prismatic", axis=axis, values=vals,
                                                                     match=sc),
                    world_samples=ws, poses=poses_c, half=hc)
