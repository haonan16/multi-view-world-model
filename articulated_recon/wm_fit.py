"""3D fit of world-model predictions (ManiSkill): robot (URDF) + objects, robust to inaccurate predicted depth.

python wm_fit.py <src> [--depth raw|table|consist|da3|da3c] [--tag T] [--stage robot|all]
  src: est:<Task>/episode-XXXX   (real RGB + model depth; GT state for every frame -> test bed)
       roll:<rollout dir>        (world-model rollout: predicted RGB + depth)
       gt:<rollout dir>          (same episode, sim GT depth)
Depth handling (--depth):
  raw      model depth as is
  table    per (frame, view) depth scale so the table lands on z = 0 (ManiSkill world, known table)
  consist  table + multiview consistency: keep a pixel only if another view's depth agrees with it (within
           tol) and no view sees through it (free-space check); RANSAC-like vote over views
  da3      Depth Anything 3 on the (predicted) RGB, known cameras, + table scale
  da3c     da3 + consistency
Outputs results/wm/<name><tag>/{robot.pkl, metrics.json}"""
import argparse, json, os, pickle, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))
import artirecon  # noqa
from artirecon.data import backproject, gt_states, load_dense, load_roll
from artirecon.fit_robot import RobotFitter, prepare
from artirecon.robot import URDFRobot

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("--depth", default="raw"); ap.add_argument("--tag", default="")
ap.add_argument("--stage", default="robot"); ap.add_argument("--tol", type=float, default=0.03)
a = ap.parse_args()
kind, ref = a.src.split(":", 1)
if kind in ("est", "dgt"):
    obs = load_dense("est" if kind == "est" else "gt", run="A", episode=ref)
    obs.meta["gt"] = gt_states(ref, obs.steps); obs.meta["episode"] = ref
    name = ref.replace("/", "_") + "_" + kind
else:
    obs = load_roll(ref, use="roll" if kind == "roll" else "gt")
    name = Path(ref).name + ("" if kind == "roll" else "_gt")
OUT = ROOT / "results/wm" / (name + "_" + a.depth + a.tag)
OUT.mkdir(parents=True, exist_ok=True)
logf = open(OUT / "log.txt", "w")


def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()


T, V, H, W = obs.T, obs.V, obs.rgb.shape[2], obs.rgb.shape[3]
log(f"=== {name} depth={a.depth}  T={T} V={V} {H}x{W}")


def table_scale(depth):
    """per (t, v): scale about the camera centre so the table pixels sit on z = 0 (plane n=(0,0,1), d=0)."""
    S = np.ones((T, V))
    for t in range(T):
        for v in range(V):
            P, _ = backproject(depth[t, v], obs.K[v], obs.c2w[v], 4)
            C = obs.c2w[v][:3, 3]
            if len(P) < 100:
                continue
            for _ in range(3):
                Ps = C + S[t, v] * (P - C)
                m = np.abs(Ps[:, 2]) < 0.03
                if m.sum() < 200:
                    break
                s = -C[2] / (P[m, 2] - C[2])
                S[t, v] = float(np.median(s))
    return depth * S[:, :, None, None].astype(np.float32), S


def consistency(depth, tol):
    """keep pixel (t,v,y,x) if >=1 other view agrees (|z - z_other| < tol*z) and no view sees through it."""
    dev = "cuda"
    D = torch.as_tensor(depth, device=dev)
    K = torch.as_tensor(obs.K, dtype=torch.float32, device=dev); c2w = torch.as_tensor(obs.c2w, dtype=torch.float32, device=dev)
    w2c = torch.linalg.inv(c2w)
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    keep = torch.zeros_like(D, dtype=torch.bool)
    for v in range(V):
        z = D[:, v]                                                            # T,H,W
        X = torch.stack([(xs - K[v, 0, 2]) / K[v, 0, 0] * z, (ys - K[v, 1, 2]) / K[v, 1, 1] * z, z], -1)
        Xw = X @ c2w[v, :3, :3].T + c2w[v, :3, 3]
        agree = torch.zeros_like(z, dtype=torch.bool); through = torch.zeros_like(z, dtype=torch.long)
        for u in range(V):
            if u == v:
                continue
            Xc = Xw @ w2c[u, :3, :3].T + w2c[u, :3, 3]
            zu = Xc[..., 2]
            pu = (Xc[..., 0] / zu.clamp(min=1e-3) * K[u, 0, 0] + K[u, 0, 2]).round().long()
            pv = (Xc[..., 1] / zu.clamp(min=1e-3) * K[u, 1, 1] + K[u, 1, 2]).round().long()
            ins = (pu >= 0) & (pu < W) & (pv >= 0) & (pv < H) & (zu > 0.05)
            Du = D[:, u][torch.arange(T, device=dev)[:, None, None].expand_as(pu), pv.clamp(0, H - 1), pu.clamp(0, W - 1)]
            ok = ins & (Du > 0)
            agree |= ok & ((zu - Du).abs() < tol * zu)
            through += (ok & (zu < Du * (1 - 2 * tol))).long()                  # u sees past the point: free space
        keep[:, v] = agree & (through == 0) & (z > 0)
    return (D * keep).cpu().numpy(), keep.float().mean().item()


depth = obs.depth
dm = a.depth
if dm.startswith("da3"):
    f = (Path(ref) if kind != "est" else None)
    zf = (f / "da3.npz") if f is not None else (ROOT / "results/wm/da3_est" / (name + ".npz"))
    depth = np.load(zf)["z"].astype(np.float32)                                # T,V,H,W
if dm != "raw":
    depth, S = table_scale(depth)
    log(f"  table scale per view (median over frames) {np.round(np.median(S, 0), 3)}")
if dm in ("consist", "da3c"):
    depth, frac = consistency(depth, a.tol)
    log(f"  multiview consistency: kept {frac * 100:.0f}% of pixels")
obs.depth = depth
gt = obs.meta["gt"]
bp = gt["base"][0]
from scipy.spatial.transform import Rotation as Rot
Tb = np.eye(4); Tb[:3, :3] = Rot.from_quat([bp[4], bp[5], bp[6], bp[3]]).as_matrix(); Tb[:3, 3] = bp[:3]
robot = URDFRobot(n_pts_per_m2=20000)
t0 = time.time()
prep = prepare(obs)
rf = RobotFitter(obs, prep, robot, log=log, calib=(dm != "raw"), known_base=Tb)
if os.environ.get("REC"):
    rf.rec = []                                   # optimisation recording (deck videos)
rob = rf.run()
W_SIL = float(os.environ.get("W_SIL", "1.0"))
VOFF = {}
if kind == "roll" and W_SIL > 0 and (Path(ref) / "sam3.npz").exists():
    # predicted depth is poor on generated views: refine the joints so the robot's silhouette matches the
    # SAM 3 'robot arm' mask in every predicted view (out-of-mask + coverage terms), depth term kept (weaker).
    # The generated views are also systematically shifted in the image (measured: up to ~16 px vs the true
    # projection, view 0 = conditioning view ~0): per-view 2D offsets are estimated on the robot (base pose
    # known) and applied to masks + depth, alternating with the refinement (GT-free view calibration).
    import cv2 as _cv2
    from artirecon.geom import DepthViews, depth_term, rho
    from artirecon.objects_v2 import Sam3Masks as _SM, shift2d
    from artirecon.splat import render_points as _rp
    from artirecon.robot import URDFRobot as _UR
    rdense = _UR(n_pts_per_m2=150000)
    depth0 = obs.depth.copy()
    base = torch.as_tensor(Tb, dtype=torch.float32, device="cuda")[None]

    def rmask(sm, t, v):
        M = np.zeros((H, W), bool)
        for i in sm.ids("robot arm", t, v):
            if sm.score[i] > 0.4:
                M |= sm.mask(i)
        return M

    def refine(q_in, sm, views):
        q_out = q_in.copy()
        for t in range(T):
            sil = []
            for v in range(V):
                M = rmask(sm, t, v)
                if M.sum() < 300:
                    continue
                dt = torch.as_tensor(_cv2.distanceTransform((~M).astype(np.uint8), _cv2.DIST_L2, 3), device="cuda")
                ys, xs = np.nonzero(M); k = np.random.default_rng(t).permutation(len(xs))[:800]
                sil.append((v, dt, torch.as_tensor(np.stack([xs[k], ys[k]], 1), dtype=torch.float32, device="cuda")))
            if not sil:
                continue
            q8 = torch.as_tensor(q_in[t:t + 1, :8], dtype=torch.float32, device="cuda").clone().requires_grad_(True)
            opt = torch.optim.Adam([q8], lr=0.01)
            for it in range(150):
                qf = torch.cat([q8, q8[:, -1:]], 1)
                P, N = robot.surface(qf, base)
                L = 0.3 * depth_term(views, P, N, [t], tau=0.02, s=0.01)[0]
                u, vv, z = views.project(P)
                Ls = 0
                for v, dt, mp in sil:
                    ui, vi = u[0, v], vv[0, v]
                    ins = ((ui >= 0) & (ui <= W - 1) & (vi >= 0) & (vi <= H - 1)).float()
                    g = torch.stack([ui.clamp(0, W - 1) / (W - 1) * 2 - 1, vi.clamp(0, H - 1) / (H - 1) * 2 - 1], -1)[None, None]
                    dpx = torch.nn.functional.grid_sample(dt[None, None], g, align_corners=True)[0, 0, 0]
                    out = (rho(dpx / 4.0, 1.0) * ins).sum() / ins.sum().clamp(min=1)
                    dc = torch.cdist(mp[None], torch.stack([ui, vi], -1)[None][:, ::3])[0].min(-1).values
                    Ls = Ls + out + rho(dc / 4.0, 1.0).mean()
                L = L + W_SIL * Ls / len(sil) + 100 * ((torch.relu(rf.lo[:8] - q8) + torch.relu(q8 - rf.hi[:8])) ** 2).sum()
                opt.zero_grad(); L.backward(); opt.step()
            q_out[t, :8] = q8.detach().cpu().numpy()[0]; q_out[t, 8] = q_out[t, 7]
        return q_out

    def offsets(q, sm_raw, rng_px=24):
        with torch.no_grad():
            RP, RN = rdense.surface(torch.as_tensor(q, dtype=torch.float32, device="cuda"), base.expand(T, 4, 4))
        RP, RN = RP.cpu().numpy(), RN.cpu().numpy()
        out = {}
        sh = list(range(-rng_px, rng_px + 1, 2))
        for v in range(V):
            score = torch.zeros(len(sh), len(sh), device="cuda"); n = 0
            for t in range(T):
                M = rmask(sm_raw, t, v)
                if M.sum() < 300:
                    continue
                _, A, _, _ = _rp([(RP[t], RN[t], (1, 1, 1))], obs.K[v], np.linalg.inv(obs.c2w[v]), H, W)
                At = torch.as_tensor(np.asarray(A.cpu() if hasattr(A, "cpu") else A), device="cuda").float(); Mt = torch.as_tensor(M, device="cuda").float()
                for i, dy in enumerate(sh):
                    for j, dx in enumerate(sh):
                        Ms = torch.roll(Mt, shifts=(-dy, -dx), dims=(0, 1))
                        score[i, j] += (Ms * At).sum() / ((Ms + At) > 0).float().sum()
                n += 1
            if n:
                i, j = np.unravel_index(int(score.argmax()), score.shape)
                out[v] = (float(sh[j]), float(sh[i]))
        return out

    q_cur = rob["q"]
    sm_raw = _SM(Path(ref) / "sam3.npz", H, W)
    for rnd in range(int(os.environ.get("VOFF_ROUNDS", "0")) + 1):
        sm_r = _SM(Path(ref) / "sam3.npz", H, W, offsets=VOFF)
        d_sh = np.stack([np.stack([shift2d(depth0[t, v], -VOFF.get(v, (0, 0))[0], -VOFF.get(v, (0, 0))[1]) for v in range(V)]) for t in range(T)]) if VOFF else depth0
        views = DepthViews(d_sh, obs.K, obs.c2w)
        q_cur = refine(q_cur, sm_r, views)
        if rnd == int(os.environ.get("VOFF_ROUNDS", "0")):
            break
        VOFF = offsets(q_cur, sm_raw)
        log(f"  generated-view offsets (px, round {rnd + 1}): {VOFF}")
    obs.depth = d_sh
    rob = dict(rob, q=q_cur, q_depth_only=rob["q"], view_offsets=VOFF)
    log(f"  robot: silhouette refinement on SAM 3 'robot arm' masks (w {W_SIL}), view offsets {VOFF}")
with torch.no_grad():
    qg = torch.tensor(gt["q"], dtype=torch.float32, device="cuda")
    B = torch.tensor(Tb, dtype=torch.float32, device="cuda")[None].expand(T, 4, 4)
    Pg, _ = robot.surface(qg, B)
    Pp, _ = robot.surface(torch.tensor(rob["q"], dtype=torch.float32, device="cuda"), B)
    e = (Pg - Pp).norm(dim=-1).mean(1).cpu().numpy() * 1000
    hand = robot.link_of >= robot.mesh_links.index("panda_hand")
    eh = (Pg - Pp).norm(dim=-1)[:, hand].mean(1).cpu().numpy() * 1000
m = dict(robot_mm=e.tolist(), robot_mm_mean=float(e.mean()), robot_mm_f0=float(e[0]), gripper_mm_mean=float(eh.mean()),
         seconds=time.time() - t0, depth=dm)
log(f"  robot vs GT: mean {e.mean():.1f} mm (frame 0 {e[0]:.1f}, gripper {eh.mean():.1f})  per frame {np.round(e, 1).tolist()}")
json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
pickle.dump(dict(rob=rob, Tb=Tb, depth_mode=dm), open(OUT / "robot.pkl", "wb"))

# ------------------------------------------------------------------ objects
if a.stage != "all":
    sys.exit(0)
import cv2
from scipy.spatial.transform import Rotation as Rot
from artirecon.objects_v2 import ObjectsV2, Sam3Masks
from artirecon.wm_objects import WMObjects
from artirecon.splat import render_points
import trimesh
TASK_OBJ = {   # name, SAM 3 prompt, role, shape, h5 actor
    "StackCube-v1": [("red_cube", "red cube", "moved", "cube", "cubeA"), ("green_cube", "green cube", "static", "cube", "cubeB")],
    "PullCubeTool-v1": [("tool", "red L-shaped tool", "moved", "hull", "l_shape_tool"), ("cube", "blue cube", "moved", "cube", "cube")],
    "PegInsertionSide-v1": [("peg", "red and white peg", "moved", "box", "peg"), ("box", "cardboard box", "static", "box", "box_with_hole")],
}
task = obs.meta.get("task") or obs.meta["episode"].split("/")[0]
d_src = Path(ref)
sam = Sam3Masks(([d_src / "color.npz"] if (d_src / "color.npz").exists() else []) + [d_src / "sam3.npz"] + ([d_src / "sam3_track.npz"] if (d_src / "sam3_track.npz").exists() else []), H, W, offsets=VOFF)
sam.first_only = True          # colour objects: colour masks only (SAM 3 masks of the same object add offset duplicates)
# annotation: the object is "clicked" where it is in the given (input) frame 0 -> projection of its frame-0 position
ann = {"objects": []}
for nm, pr, role, shape, act in TASK_OBJ[task]:
    p0 = gt["actors"][act][0, :3]
    best = None
    for v in ([0] if kind == "roll" else range(V)):        # rollout: only view 0 at frame 0 is real (the input)
        pc = np.linalg.inv(obs.c2w[v]) @ np.r_[p0, 1]
        u, vv = (obs.K[v] @ pc[:3])[:2] / pc[2]
        if 4 <= u < W - 4 and 4 <= vv < H - 4:
            ids = sam.ids(pr, 0, v)
            hit = any(sam.mask(i)[int(vv), int(u)] for i in ids)
            sc = (hit, -pc[2])
            if best is None or sc > best[0]:
                best = (sc, v, u, vv)
    if best is None:
        continue
    ann["objects"].append(dict(name=nm, prompt=pr, role=role, shape=shape, hint=dict(t=0, v=best[1], uv=[float(best[2]), float(best[3])])))
json.dump(ann, open(OUT / "ann.json", "w"), indent=1)
rob_d = dict(q=rob["q"], base=Tb)
prep_o = dict(n=prep["n"], d=prep["d"])
t1 = time.time()
_ob = (WMObjects if os.environ.get('TRI', '1') == '1' else ObjectsV2)(obs, prep_o, robot, rob_d, sam, ann, log=log)
if os.environ.get("REC"):
    _ob.rec = []
objs = _ob.run()
if os.environ.get("REC"):
    pickle.dump(dict(robot=[r for r in rf.rec if r["iters"] == 300 and r["frames"] == [0] * len(r["frames"]) and len(r["frames"]) > 1],
                     objects=_ob.rec, obj="red_cube", c2w=obs.c2w, K=obs.K, gt_q=gt["q"], depth=obs.depth.astype(np.float16),
                     rgb=obs.rgb, final_q=rob["q"], Tb=Tb), open(OUT / "optrec.pkl", "wb"))
log(f"  objects done [{time.time() - t1:.0f}s]")
# metrics: frame-0 pose vs GT (the input state), silhouettes vs SAM 3 on the predicted RGB, task check in 3D
om = {}
for ob in objs:
    spec = next(x for x in TASK_OBJ[task] if x[0] == ob["name"])
    c_mesh = ob["mesh"][0].mean(0)
    cen = np.array([ob["poses"][t][:3, :3] @ c_mesh + ob["poses"][t][:3, 3] for t in range(T)])
    e0 = float(np.linalg.norm(cen[0] - gt["actors"][spec[4]][0, :3]) * 1000)
    tm = trimesh.Trimesh(ob["mesh"][0], ob["mesh"][1], process=False)
    sp, fi = trimesh.sample.sample_surface(tm, 20000, seed=0); sp = np.asarray(sp); sn = np.asarray(tm.face_normals[fi])
    f2 = np.full((T, V), np.nan); iou = np.full((T, V), np.nan)
    for t in range(T):
        Tp = ob["poses"][t]
        for v in range(V):
            _, A, _, pid = render_points([(sp @ Tp[:3, :3].T + Tp[:3, 3], sn @ Tp[:3, :3].T, (1, 1, 1))], obs.K[v], np.linalg.inv(obs.c2w[v]), H, W, footprint=[1])
            vis = A.cpu().numpy() if hasattr(A, "cpu") else A
            sid = ob["sel"][t].get(v)
            if sid is None:
                continue
            sm = sam.mask(sid)
            iou[t, v] = (vis & sm).sum() / max(1, (vis | sm).sum())
            if vis.sum() and sm.sum():
                dv = cv2.distanceTransform((~vis).astype(np.uint8), cv2.DIST_L2, 3); ds = cv2.distanceTransform((~sm).astype(np.uint8), cv2.DIST_L2, 3)
                p_, r_ = (ds[vis] <= 2).mean(), (dv[sm] <= 2).mean(); f2[t, v] = 2 * p_ * r_ / max(p_ + r_, 1e-6)
    om[ob["name"]] = dict(f0_pos_err_mm=e0, sil_f2px=float(np.nanmean(f2)), iou=float(np.nanmean(iou)),
                          extents_mm=(2 * np.asarray(ob["half"]) * 1000).round(1).tolist(), held=np.nonzero(ob["held"])[0].tolist(),
                          start=cen[0].round(4).tolist(), end=cen[-1].round(4).tolist(), views_per_frame=[len(s) for s in ob["sel"].values()])
    log(f"  {ob['name']}: frame-0 position error {e0:.1f} mm, silhouette F@2px {np.nanmean(f2):.2f}, IoU {np.nanmean(iou):.2f}, "
        f"extents {om[ob['name']]['extents_mm']} mm, held {om[ob['name']]['held']}")
from artirecon.robot import URDFRobot as _U
rd = _U(n_pts_per_m2=150000)
with torch.no_grad():
    RP, RN = rd.surface(torch.as_tensor(rob["q"], dtype=torch.float32, device="cuda"), torch.as_tensor(Tb, dtype=torch.float32, device="cuda")[None].expand(T, 4, 4))
RP, RN = RP.cpu().numpy(), RN.cpu().numpy()
riou = np.full((T, V), np.nan)
for t in range(T):
    for v in range(V):
        M = np.zeros((H, W), bool)
        for i in sam.ids("robot arm", t, v):
            if sam.score[i] > 0.4:
                M |= sam.mask(i)
        if M.sum() < 200:
            continue
        _, A, _, _ = render_points([(RP[t], RN[t], (1, 1, 1))], obs.K[v], np.linalg.inv(obs.c2w[v]), H, W)
        A = A.cpu().numpy() if hasattr(A, "cpu") else A
        riou[t, v] = (A & M).sum() / max(1, (A | M).sum())
m["robot_sil_iou"] = float(np.nanmean(riou)); m["robot_sil_iou_views"] = np.round(np.nanmean(riou, 0), 3).tolist()
log(f"  robot silhouette IoU vs SAM 3 per view {m['robot_sil_iou_views']} (mean {m['robot_sil_iou']:.3f})")
m["objects"] = om
json.dump(m, open(OUT / "metrics.json", "w"), indent=1, default=float)
pickle.dump(dict(rob=rob, Tb=Tb, objects=objs, ann=ann, plane=(prep["n"], prep["d"]), depth_mode=dm, T=T), open(OUT / "fit.pkl", "wb"))
