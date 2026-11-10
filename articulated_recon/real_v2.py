"""Real RH20T episode, v2: robot (URDF, multiview depth) + annotated objects (SAM 3 masks + multiview depth).

python real_v2.py <episode> [--stage robot|objects|all] [--holdout V]
  inputs : real/v2/<ep>/obs.npz (prep_rh20t_episode.py --T 20 --pad_s 2), sam3.npz (seg_sam3.py), ann.json
  outputs: results/real_v2/<ep>/{robot.pkl, fit.pkl, metrics.json, overlay.mp4, overlay_grid.jpg, masks_grid.jpg}

Steps
 1. camera refinement on the robot: the URDF posed at the logged joints (proprioception, i.e. the calibration
    target) vs each camera's depth near the robot; 6-DoF ICP per camera (RH20T calibrations drift)
 2. table self-calibration (per-camera depth scale / offset) + URDF fit from multiview depth (joints NOT given)
 3. objects from the annotation + SAM 3 (artirecon/objects_v2.py)
 4. metrics: robot vs logged joints; objects: IoU of the rendered object with its SAM 3 mask per view
    (also on a held-out view when --holdout), depth agreement, physics (table contact, in-hand, penetration)
"""
import argparse
import copy
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artirecon  # noqa: F401  (pytorch3d compatibility shim first)
from pytorch3d.ops import knn_points
from artirecon.data import backproject, load_real
from artirecon.fit_robot import RobotFitter, prepare
from artirecon.geom import DepthViews, table_calibrate
from artirecon.objects_v2 import ObjectsV2, Sam3Masks
from artirecon.render import overlay, render, scene_parts
from artirecon.robot import URDFRobot

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("episode")
ap.add_argument("--stage", default="all")
ap.add_argument("--holdout", type=int, default=-1)
ap.add_argument("--no_recal", action="store_true")
ap.add_argument("--tag", default="")
a = ap.parse_args()
EP = a.episode
OUT = ROOT / "results" / "real_v2" / (EP + a.tag)
OUT.mkdir(parents=True, exist_ok=True)
logf = open(OUT / "log.txt", "a")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n"); logf.flush()


robot = URDFRobot(n_pts_per_m2=20000)                 # fitting + metrics (~18k surface samples)
obs = load_real("v2/" + EP)
gts = obs.meta["gt_states"]
log(f"=== {EP}  T={obs.T} V={obs.V}  task: {json.load(open(ROOT / 'real/v2' / EP / 'meta.json'))['task']}")


def recal(obs):
    """6-DoF per-camera refinement against the URDF at the logged joints (calibration target)."""
    with torch.no_grad():
        q = torch.tensor(np.stack([g["q"] for g in gts]), dtype=torch.float32, device="cuda")
        P, _ = robot.surface(q, torch.eye(4, device="cuda")[None].expand(len(q), 4, 4))
    o = copy.copy(obs); o.c2w = obs.c2w.copy()
    rep = {}
    rng = np.random.default_rng(0)
    for v in range(obs.V):
        # per frame: this camera's points near the robot vs the robot at that frame (batched over frames)
        Ss = []
        for t in range(obs.T):
            p = backproject(obs.depth[t, v], obs.K[v], obs.c2w[v], 2)[0]
            p = p[rng.permutation(len(p))[:8000]]
            Ss.append(np.pad(p, ((0, 8000 - len(p)), (0, 0)), constant_values=1e3))
        S = torch.as_tensor(np.stack(Ss), dtype=torch.float32, device="cuda")       # T,8000,3
        D = P                                                                       # T,N,3
        Tacc = torch.eye(4, device="cuda")
        d0 = None
        for rad in np.linspace(0.05, 0.015, 25):
            Sx = S @ Tacc[:3, :3].T + Tacc[:3, 3]
            kn = knn_points(Sx, D, K=1)
            dist = kn.dists[..., 0].sqrt()
            m = dist < rad
            if d0 is None:
                d0 = float(dist[dist < 0.05].median())
            if m.sum() < 100:
                break
            A = Sx[m]
            B = torch.gather(D, 1, kn.idx[..., :1].expand(-1, -1, 3))[m]
            ca, cb = A.mean(0), B.mean(0)
            U, _, Vt = torch.linalg.svd((A - ca).T @ (B - cb))
            R = (U @ Vt).T
            if torch.det(R) < 0:
                Vt[-1] *= -1; R = (U @ Vt).T
            Ti = torch.eye(4, device="cuda"); Ti[:3, :3] = R; Ti[:3, 3] = cb - R @ ca
            Tacc = Ti @ Tacc
        Sx = S @ Tacc[:3, :3].T + Tacc[:3, 3]
        dist = knn_points(Sx, D, K=1).dists[..., 0].sqrt()
        d1 = float(dist[dist < 0.05].median())
        Tc = Tacc.double().cpu().numpy()
        if d1 < d0:
            o.c2w[v] = Tc @ obs.c2w[v]
        ang = float(np.degrees(np.arccos(np.clip((np.trace(Tc[:3, :3]) - 1) / 2, -1, 1))))
        rep[v] = dict(rot_deg=round(ang, 2), trans_mm=round(float(np.linalg.norm(Tc[:3, 3]) * 1000), 1),
                      med_before_mm=round(d0 * 1000, 2), after_mm=round(d1 * 1000, 2), applied=bool(d1 < d0))
    log(f"  camera refinement on the robot: {rep}")
    return o, rep


def robot_metrics(rob):
    T = obs.T
    with torch.no_grad():
        qg = torch.tensor(np.stack([g["q"] for g in gts]), dtype=torch.float32, device="cuda")
        Pg, _ = robot.surface(qg, torch.eye(4, device="cuda")[None].expand(T, 4, 4))
        qp = torch.tensor(rob["q"], dtype=torch.float32, device="cuda")
        Pp, _ = robot.surface(qp, torch.tensor(rob["base"], dtype=torch.float32, device="cuda")[None].expand(T, 4, 4))
        e = (Pg - Pp).norm(dim=-1)
        hand = robot.link_of >= robot.mesh_links.index("panda_hand")
    qgt = np.stack([g["q"] for g in gts])
    return dict(robot_surface_mm=(e.mean(1) * 1000).cpu().numpy().tolist(),
                gripper_surface_mm=(e[:, hand].mean(1) * 1000).cpu().numpy().tolist(),
                joint_err_deg=np.degrees(np.abs(rob["q"][:, :7] - qgt[:, :7])).mean(1).tolist(),
                finger_err_mm=(np.abs(rob["q"][:, 7] - qgt[:, 7]) * 1000).tolist(),
                base_err_mm=float(np.linalg.norm(rob["base"][:3, 3]) * 1000))


t_start = time.time()
rp = ROOT / "results" / "real_v2" / EP / "robot.pkl"          # shared by tagged (e.g. held-out) runs
if a.stage in ("robot", "all") or not rp.exists():
    t0 = time.time()
    o, rep = (obs, {}) if a.no_recal else recal(obs)
    S, _, noise = table_calibrate(o, log=log)
    o = copy.copy(o); o.depth = o.depth * S[:, :, None, None].astype(np.float32)
    offs, nrm = table_calibrate.offsets
    o.c2w = o.c2w.copy()
    for v in range(o.V):
        o.c2w[v][:3, 3] -= offs[v] * nrm
    prep = prepare(o, fg_thr=max(0.008, 3 * noise))
    rf = RobotFitter(o, prep, robot, log=log, calib=True, known_base=np.eye(4))
    rob = rf.run()
    if rob.get("depth_scale") is not None:
        o.depth = o.depth * rob["depth_scale"][:, :, None, None].astype(np.float32)
        prep = prepare(o, fg_thr=max(0.008, 3 * noise))
    rm = robot_metrics(rob)
    log(f"  robot: {np.mean(rm['robot_surface_mm']):.1f} mm (gripper {np.mean(rm['gripper_surface_mm']):.1f}, "
        f"joints {np.mean(rm['joint_err_deg']):.2f} deg)  [{time.time() - t0:.0f}s]")
    pickle.dump(dict(rob=rob, c2w=o.c2w, depth_scale_total=(o.depth / np.maximum(obs.depth, 1e-6)).astype(np.float16)
                     if False else None, S=S, rob_scale=rob.get("depth_scale"), offs=offs, nrm=nrm, noise=noise,
                     prep_n=prep["n"], prep_d=prep["d"], recal=rep, robot_metrics=rm, robot_s=time.time() - t0),
                open(rp, "wb"))
if a.stage == "robot":
    sys.exit(0)

R = pickle.load(open(rp, "rb"))
rob = R["rob"]
o = copy.copy(obs); o.c2w = R["c2w"]
o.depth = obs.depth * R["S"][:, :, None, None].astype(np.float32)
if R["rob_scale"] is not None:
    o.depth = o.depth * R["rob_scale"][:, :, None, None].astype(np.float32)
prep = dict(n=R["prep_n"], d=R["prep_d"])
ann = json.load(open(ROOT / "real/v2" / EP / "ann.json"))
for spec in ann["objects"]:              # SAM 3D initial assets where available
    mp = ROOT / "real/v2" / EP / "sam3d" / (spec["name"] + "_20k.ply")
    if spec.get("shape") in ("hull", "mesh") and mp.exists() and not os.environ.get("NO_SAM3D"):
        spec.update(shape="mesh", mesh=str(mp), mesh_json=str(mp).replace("_20k.ply", ".json"))
sam = Sam3Masks(ROOT / "real/v2" / EP / "sam3.npz", obs.rgb.shape[2], obs.rgb.shape[3])
t1 = time.time()
ex = (a.holdout,) if a.holdout >= 0 else ()
if a.stage == "metrics":                  # re-score a saved fit (same renderer for every run)
    res = pickle.load(open(OUT / "fit.pkl", "rb"))
    objs = res["objects"]; obj_s = res["timing"]["objects_s"]
else:
    objs = ObjectsV2(o, prep, robot, rob, sam, ann, log=log, exclude_views=ex).run()
    obj_s = time.time() - t1
res = dict(label=f"real/{EP}", source="real", scene="real", condition="mv_v2", robot=rob, objects=objs,
           plane=(prep["n"], prep["d"]), timing=dict(robot_s=R["robot_s"], objects_s=obj_s), run="")
if a.stage != "metrics":
    pickle.dump(res, open(OUT / "fit.pkl", "wb"))

# ------------------------------------------------------------------ metrics + overlays
H, W = obs.rgb.shape[2:4]
T, V = obs.T, obs.V
m = dict(R["robot_metrics"])
m["recal"] = R["recal"]
iou = {ob["name"]: np.full((T, V), np.nan) for ob in objs}
f2 = {ob["name"]: np.full((T, V), np.nan) for ob in objs}
dag = {ob["name"]: np.full((T, V), np.nan) for ob in objs}
frames, mframes = [], []
crops = {}
nobj = len(objs)
from artirecon.splat import render_points
import trimesh
rob_dense = URDFRobot(n_pts_per_m2=250000)
with torch.no_grad():
    RP, RN = rob_dense.surface(torch.as_tensor(rob["q"], dtype=torch.float32, device="cuda"),
                               torch.as_tensor(rob["base"], dtype=torch.float32, device="cuda")[None].expand(T, 4, 4))
    RP, RN = RP.cpu().numpy(), RN.cpu().numpy()
osamp = []
for ob in objs:
    tm = trimesh.Trimesh(ob["mesh"][0], ob["mesh"][1], process=False)
    npt = int(max(8000, tm.area * 2500000))       # dense: objects are splatted to single pixels
    sp, fi = trimesh.sample.sample_surface(tm, npt, seed=0)
    osamp.append((np.asarray(sp), np.asarray(tm.face_normals[fi])))
COL = {"robot": (0.78, 0.86, 0.98), "free": (1.0, 0.62, 0.25), "static": (0.85, 0.80, 0.55)}
for t in range(T):
    pr = [(RP[t], RN[t], COL["robot"])]
    for ob, (sp, sn) in zip(objs, osamp):
        Tp = ob["poses"][t]
        pr.append((sp @ Tp[:3, :3].T + Tp[:3, 3], sn @ Tp[:3, :3].T, COL[ob["type"]]))
    nrob = 1
    row, mrow = [], []
    for v in range(V):
        img, alpha, dep, pid = render_points(pr, obs.K[v], np.linalg.inv(o.c2w[v]), H, W, footprint=[2] + [1] * nobj)
        ov = overlay(obs.rgb[t, v], img, alpha, 0.7)
        mv = obs.rgb[t, v].copy()
        for k, ob in enumerate(objs):
            vis = pid == nrob + k
            sid = ob["sel"][t].get(v)
            if v in ex and vis.sum() > 5:      # held-out view: best-matching SAM 3 instance of the prompt
                spec = next(x for x in ann["objects"] if x["name"] == ob["name"])
                best = (0.0, None)
                for i in sam.ids(spec["prompt"], t, v):
                    sm_ = sam.mask(i)
                    j = (vis & sm_).sum() / max(1, (vis | sm_).sum())
                    if j > best[0]:
                        best = (j, i)
                sid = best[1]
                if sid is None:
                    iou[ob["name"]][t, v] = 0.0
            if sid is not None:
                sm = sam.mask(sid)
                iou[ob["name"]][t, v] = float((vis & sm).sum() / max(1, (vis | sm).sum()))
                if vis.sum() and sm.sum():     # silhouette F-score with 2 px tolerance (size-independent)
                    dv = cv2.distanceTransform((~vis).astype(np.uint8), cv2.DIST_L2, 3)
                    ds = cv2.distanceTransform((~sm).astype(np.uint8), cv2.DIST_L2, 3)
                    pr_, rc_ = (ds[vis] <= 2).mean(), (dv[sm] <= 2).mean()
                    f2[ob["name"]][t, v] = float(2 * pr_ * rc_ / max(pr_ + rc_, 1e-6))
                cs, _ = cv2.findContours(sm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(mv, cs, -1, (0, 255, 0), 1)
            cs, _ = cv2.findContours(vis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(mv, cs, -1, (255, 0, 255), 1)
            D = o.depth[t, v]
            mk = vis & (D > 0)
            if mk.sum() > 10:
                dag[ob["name"]][t, v] = float((np.abs(dep[mk] - D[mk]) < 0.01).mean())
        cv2.putText(ov, f"t{t} v{v}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        row.append(np.concatenate([obs.rgb[t, v], ov], 0)); mrow.append(mv)
    frames.append(np.concatenate(row, 1)); mframes.append(np.concatenate(mrow, 1))
    for k, ob in enumerate(objs):           # zoomed crops around each object (green SAM 3, magenta model)
        c = ob["poses"][t][:3, 3]
        tiles = []
        for v in range(V):
            pc = np.linalg.inv(o.c2w[v]) @ np.r_[c, 1]
            uu, vv_ = (obs.K[v] @ pc[:3])[:2] / max(pc[2], 1e-3)
            r = int(np.clip(0.12 * obs.K[v][0, 0] / max(pc[2], 0.2), 24, 90))
            x0, y0 = int(uu) - r, int(vv_) - r
            big = np.pad(mrow[v], ((200, 200), (200, 200), (0, 0)))
            tile = big[y0 + 200:y0 + 200 + 2 * r, x0 + 200:x0 + 200 + 2 * r]
            tiles.append(cv2.resize(tile, (120, 120)) if tile.size else np.zeros((120, 120, 3), np.uint8))
        crops.setdefault(k, []).append(np.concatenate(tiles, 1))
frames = np.stack(frames)
iio.imwrite(OUT / "overlay_grid.jpg", np.concatenate([f[H:] for f in frames], 0)[::2, ::2], quality=85)
iio.imwrite(OUT / "masks_grid.jpg", np.concatenate(mframes, 0)[::2, ::2], quality=85)
for k, ob in enumerate(objs):
    iio.imwrite(OUT / f"crops_{ob['name']}.jpg", np.concatenate(crops[k], 0), quality=88)
iio.imwrite(OUT / "overlay.mp4", frames, fps=3, codec="libx264", macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "23"])

n, d = prep["n"], prep["d"]
for ob in objs:
    nm = ob["name"]
    P0 = ob["mesh"][0]
    if not any(x["name"] == nm for x in ann["objects"]):
        continue
    hmin = np.array([((ob["poses"][t][:3, :3] @ P0.T).T + ob["poses"][t][:3, 3]) @ n + d for t in range(T)]).min(1)
    free = ~ob["held"]
    m[f"obj/{nm}/iou_mean"] = float(np.nanmean(iou[nm]))
    m[f"obj/{nm}/sil_f2px"] = float(np.nanmean(f2[nm]))
    if a.holdout >= 0:
        m[f"obj/{nm}/sil_f2px_holdout"] = float(np.nanmean(f2[nm][:, a.holdout]))
    m[f"obj/{nm}/iou_views"] = np.round(np.nanmean(iou[nm], 0), 3).tolist()
    m[f"obj/{nm}/iou_frames"] = np.round(np.nanmean(iou[nm], 1), 3).tolist()
    if a.holdout >= 0:
        m[f"obj/{nm}/iou_holdout"] = float(np.nanmean(iou[nm][:, a.holdout]))
    m[f"obj/{nm}/depth_agree_1cm"] = float(np.nanmean(dag[nm]))
    sup = ob.get("support", np.zeros(T))
    m[f"obj/{nm}/rest_contact_mm"] = float(np.abs(hmin[free] - sup[free]).mean() * 1000) if free.any() else None
    m[f"obj/{nm}/penetration_mm"] = float(np.clip(-hmin, 0, None).max() * 1000)
    m[f"obj/{nm}/held_frames"] = np.nonzero(ob["held"])[0].tolist()
    gw = np.load(ROOT / "real/v2" / EP / "obs.npz")["gripper_width_mm"]
    gt_hold = (gw > 5) & (gw < 76)                     # logged gripper closed on something (evaluation only)
    if next(x for x in ann["objects"] if x["name"] == nm).get("role") == "moved" and gt_hold.any():
        m[f"obj/{nm}/grasp_iou_vs_logged"] = float((gt_hold & ob["held"]).sum() / max(1, (gt_hold | ob["held"]).sum()))
    m[f"obj/{nm}/extents_mm"] = np.round(2 * np.asarray(ob["half"]) * 1000, 1).tolist()
    m[f"obj/{nm}/travel_cm"] = float(np.linalg.norm(ob["poses"][-1][:3, 3] - ob["poses"][0][:3, 3]) * 100)
    m[f"obj/{nm}/views_per_frame"] = [len(s) for s in ob["sel"].values()]
m["timing"] = dict(robot_s=R["robot_s"], objects_s=obj_s)
(OUT / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
summ = {k: (round(float(np.mean(v)), 2) if isinstance(v, list) and v and isinstance(v[0], float) and "views" not in k and "frames" not in k else v)
        for k, v in m.items() if k not in ("recal",)}
log("METRICS " + json.dumps(summ, default=float))
