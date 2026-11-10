"""Videos for the slide deck: step-by-step pipeline on the peg scene (sim GT, 4 views, 11 frames)
plus orbit renders of the results.  python deck_media.py [stage ...]  -> deck/media/*.mp4"""
import pickle
import os
import sys
import time
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from artirecon.data import backproject, load_dense, load_dense_drawer, load_gt512, load_real
from artirecon.fit_robot import RobotFitter, prepare
from artirecon.objects import cluster, presence, residuals, voxel_down
from artirecon.geom import DepthViews
from artirecon.render import render, scene_parts
from artirecon.robot import PANDA_READY, URDFRobot
from artirecon.fit_rgb import lookat_w2c

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "deck/media"
OUT.mkdir(parents=True, exist_ok=True)
H, W = 720, 1280
KV = np.array([[1000., 0, W / 2], [0, 1000., H / 2], [0, 0, 1]])
BG = np.array([30, 30, 34], np.uint8)
TIMES = {}


def write(name, frames, fps=15):
    iio.imwrite(OUT / f"{name}.mp4", np.stack(frames), fps=fps, codec="libx264", macro_block_size=8,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "20"])
    iio.imwrite(OUT / f"{name}.jpg", frames[len(frames) // 2], quality=90)
    print("wrote", name, len(frames), flush=True)


def orbit(k, n, centre, radius=1.5, height=0.85, a0=-0.75, sweep=1.2):
    a = a0 + sweep * (k / max(1, n - 1) - 0.5)
    eye = centre + np.array([radius * np.cos(a), radius * np.sin(a), height])
    return lookat_w2c(eye, centre)


def splat(img, zbuf, P, C, w2c, r=2):
    Pc = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = Pc[:, 2]
    ok = z > 0.05
    Pc, C, z = Pc[ok], C[ok], z[ok]
    u = np.round(Pc[:, 0] / z * KV[0, 0] + KV[0, 2]).astype(int)
    v = np.round(Pc[:, 1] / z * KV[1, 1] + KV[1, 2]).astype(int)
    o = np.argsort(-z)
    u, v, z, C = u[o], v[o], z[o], C[o]
    for du in range(r):
        for dv in range(r):
            uu, vv = u + du, v + dv
            m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            uu, vv, zz, cc = uu[m], vv[m], z[m], C[m]
            vis = zz < zbuf[vv, uu]
            img[vv[vis], uu[vis]] = cc[vis]
    return img


def frame_img(w2c, parts=None, pts=(), mesh_alpha=1.0):
    """parts: list of (trimesh, rgb); pts: list of (P, C uint8, r)"""
    img = np.tile(BG, (H, W, 1)).astype(np.uint8)
    zb = np.full((H, W), np.inf)
    if parts:
        rgb, alpha, dep, _ = render(parts, KV, w2c, H, W)
        zb = np.where(alpha, dep, np.inf)
        m = (rgb * 255).clip(0, 255).astype(np.uint8)
        img[alpha] = (mesh_alpha * m[alpha] + (1 - mesh_alpha) * img[alpha]).astype(np.uint8)
    for P, C, r in pts:
        img = splat(img, zb, P, C, w2c, r)
    return img


def frustum_lines(img, K, c2w, w2c, Hc, Wc, col=(235, 235, 235), s=0.12):
    corners = np.array([[0, 0], [Wc, 0], [Wc, Hc], [0, Hc]], float)
    rays = np.stack([(corners[:, 0] - K[0, 2]) / K[0, 0], (corners[:, 1] - K[1, 2]) / K[1, 1], np.ones(4)], 1) * s
    Pw = np.concatenate([[c2w[:3, 3]], rays @ c2w[:3, :3].T + c2w[:3, 3]])
    Pc = Pw @ w2c[:3, :3].T + w2c[:3, 3]
    uv = np.stack([Pc[:, 0] / Pc[:, 2] * KV[0, 0] + KV[0, 2], Pc[:, 1] / Pc[:, 2] * KV[1, 1] + KV[1, 2]], 1).astype(int)
    for i in range(1, 5):
        cv2.line(img, tuple(uv[0]), tuple(uv[i]), col, 2, cv2.LINE_AA)
        cv2.line(img, tuple(uv[i]), tuple(uv[i % 4 + 1]), col, 2, cv2.LINE_AA)
    return img


def fused(obs, t, stride=2, views=None):
    P, C = [], []
    for v in (views if views is not None else range(obs.V)):
        p, pix = backproject(obs.depth[t, v], obs.K[v], obs.c2w[v], stride)
        P.append(p); C.append(obs.rgb[t, v][pix[:, 0], pix[:, 1]])
    P, C = np.concatenate(P), np.concatenate(C)
    keep = (np.abs(P[:, 0]) < 1.3) & (np.abs(P[:, 1]) < 1.3) & (P[:, 2] > -0.05) & (P[:, 2] < 1.3)
    return P[keep], C[keep]


def hold(seq, n):
    return [f for f in seq for _ in range(n)]


# ---------------------------------------------------------------------------------------------
ROBOT = URDFRobot(n_pts_per_m2=20000)
OBS = None
RES = None
CTR = np.array([-0.25, 0.0, 0.18])


def peg():
    global OBS, RES
    if OBS is None:
        OBS = load_dense("gt")
        RES = pickle.load(open(ROOT / "results/standard/gt__mv/fit.pkl", "rb"))
    return OBS, RES


def s1_input():
    obs, _ = peg()
    fr = []
    for t in range(obs.T):
        rgb = np.concatenate(list(obs.rgb[t]), 1)
        d = obs.depth[t]
        dn = np.clip((d - 0.3) / 1.6, 0, 1)
        dc = [cv2.applyColorMap((255 - dn[v] * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1] for v in range(obs.V)]
        dc = [np.where((d[v] > 0)[..., None], c, 0) for v, c in enumerate(dc)]
        g = np.concatenate([rgb, np.full((8, rgb.shape[1], 3), 255, np.uint8), np.concatenate(dc, 1)], 0)
        fr.append(g)
    write("p1_input_grid", hold(fr, 8))
    # camera rig: frustums + frame-0 points, orbit
    P, C = fused(obs, 0, 1)
    n = 90
    out = []
    for k in range(n):
        w2c = orbit(k, n, CTR, radius=2.3, height=1.4, sweep=1.6)
        img = frame_img(w2c, pts=[(P, C, 3)])
        for v in range(obs.V):
            img = frustum_lines(img, obs.K[v], obs.c2w[v], w2c, obs.rgb.shape[2], obs.rgb.shape[3], s=0.18)
        out.append(img)
    write("p1_cameras", out)


def s2_lift():
    obs, _ = peg()
    t0 = time.time()
    PC = [fused(obs, t, 1) for t in range(obs.T)]
    TIMES["lift_s"] = (time.time() - t0) / obs.T
    n = obs.T * 10
    out = []
    for k in range(n):
        t = k // 10
        P, C = PC[t]
        out.append(frame_img(orbit(k, n, CTR), pts=[(P, C, 3)]))
    write("p2_lift", out)


def s3_table():
    obs, _ = peg()
    t0 = time.time()
    prep = prepare(obs)
    TIMES["table_s"] = time.time() - t0
    n = obs.T * 10
    out = []
    for k in range(n):
        t = k // 10
        P, C = fused(obs, t, 1)
        h = P @ prep["n"] + prep["d"]
        fg = h > 0.008
        Cg = C.copy()
        Cg[~fg] = (0.35 * Cg[~fg] + 0.65 * 70).astype(np.uint8)
        F = prep["fg_all"][t]
        out.append(frame_img(orbit(k, n, CTR), pts=[(P[~fg], Cg[~fg], 3), (P[fg], np.tile([245, 120, 30], (fg.sum(), 1)).astype(np.uint8), 3)]))
    write("p3_table", out)
    return prep


def s4_search():
    """frame-0 multi-hypothesis robot fit, snapshots of all hypotheses during the optimisation"""
    obs, _ = peg()
    prep = prepare(obs)
    rf = RobotFitter(obs, prep, ROBOT)
    t0 = time.time()
    cands = rf.search_base(topk=4)
    Hs = [(2 * np.pi * k / 8, c) for c in cands for k in range(8)]
    yaw = torch.tensor([h[0] for h in Hs], device="cuda").requires_grad_(True)
    xy = torch.stack([h[1] for h in Hs]).clone().requires_grad_(True)
    q8 = torch.tensor(PANDA_READY[:8], device="cuda")[None].repeat(len(Hs), 1).requires_grad_(True)
    iters, lr = 300, 0.03
    opt = torch.optim.Adam([{"params": [q8], "lr": lr}, {"params": [yaw, xy], "lr": lr * 0.3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
    snaps = []
    for it in range(iters):
        s = 0.03 * (0.01 / 0.03) ** (it / (iters - 1))
        base = rf.base_T(yaw, xy)
        L, ch, dl = rf.loss(q8, base, [0] * len(Hs), 1.0, s)
        if it % 6 == 0 or it == iters - 1:
            snaps.append((q8.detach().clone(), base.detach().clone(), L.detach().clone()))
        opt.zero_grad(); L.sum().backward(); opt.step(); sched.step()
    TIMES["frame0_s"] = time.time() - t0
    best = int(snaps[-1][2].argmin())
    F = prep["fg_all"][0]
    Fs = F[np.random.default_rng(0).choice(len(F), min(len(F), 30000), replace=False)]
    P0, C0 = fused(obs, 0, 2)
    h = P0 @ prep["n"] + prep["d"]
    bgm = h <= 0.008
    Cb = (0.35 * C0[bgm] + 0.65 * 70).astype(np.uint8)
    rng = np.random.default_rng(1)
    pal = (rng.uniform(0.25, 0.85, (len(Hs), 3)) * 255).astype(np.uint8)
    out = []
    n = len(snaps)
    for k, (q, b, L) in enumerate(snaps):
        with torch.no_grad():
            S, _ = ROBOT.surface(rf.full_q(q), b)
        S = S.cpu().numpy()
        sub = np.arange(0, S.shape[1], 3)
        pts = [(P0[bgm], Cb, 2), (Fs, np.full((len(Fs), 3), 200, np.uint8), 2)]
        for i in range(len(Hs)):
            pts.append((S[i, sub], np.tile(pal[i], (len(sub), 1)), 2))
        out.append(frame_img(orbit(k, n + 30, CTR, radius=1.7, height=1.1), pts=pts))
    # final: best hypothesis as mesh
    qb = rf.full_q(snaps[-1][0][best:best + 1])[0].cpu().numpy()
    bb = snaps[-1][1][best].cpu().numpy()
    parts = [(m, (0.55, 0.7, 0.95)) for _, m in ROBOT.meshes_at(qb, bb)]
    for k in range(30):
        out.append(frame_img(orbit(n + k, n + 30, CTR, radius=1.7, height=1.1), parts, [(P0[bgm], Cb, 2), (Fs, np.full((len(Fs), 3), 200, np.uint8), 2)]))
    write("p4_search", out, fps=12)


def s5_track():
    obs, res = peg()
    P0 = None
    n = obs.T * 10
    out = []
    for k in range(n):
        t = k // 10
        P, C = fused(obs, t, 1)
        parts = [(m, (0.55, 0.7, 0.95)) for _, m in ROBOT.meshes_at(res["robot"]["q"][t], res["robot"]["base"])]
        out.append(frame_img(orbit(k, n, CTR, radius=1.6, height=0.9), parts, [(P, C, 3)]))
    write("p5_track", out)


def s6_residual():
    obs, res = peg()
    prep = prepare(obs)
    t0 = time.time()
    R, _ = residuals(prep, ROBOT, res["robot"]["q"], res["robot"]["base"])
    C0 = cluster(voxel_down(R[0], 0.005), eps=0.025, min_pts=15)
    C0 = [c for c in C0 if np.linalg.norm(np.quantile(c, 0.98, 0) - np.quantile(c, 0.02, 0)) < 0.8]
    views = DepthViews(obs.depth, obs.K, obs.c2w)
    st = [np.stack([presence(views, c, t) for t in range(obs.T)]) for c in C0]
    TIMES["residual_s"] = time.time() - t0
    pal = [np.array(c, np.uint8) for c in ([40, 150, 60], [230, 120, 20], [120, 80, 200], [200, 60, 60])]
    n = obs.T * 10
    out = []
    for k in range(n):
        t = k // 10
        P, C = fused(obs, t, 1)
        Cg = (0.35 * C + 0.65 * 70).astype(np.uint8)
        parts = [(m, (0.82, 0.85, 0.9)) for _, m in ROBOT.meshes_at(res["robot"]["q"][t], res["robot"]["base"])]
        pts = [(P, Cg, 3)]
        for i, c in enumerate(C0):
            s = st[i][t]
            col = np.tile(pal[i % 4], (len(c), 1))
            col[s == -1] = [225, 30, 30]                     # seen-through -> it moved
            pts.append((c, col, 4))
        out.append(frame_img(orbit(k, n, CTR, radius=1.4, height=0.8), parts, pts, mesh_alpha=0.6))
    write("p6_residual", out)


def recon_orbit(name, res, obs, ctr, radius=1.5, height=0.85, pts=True, per=10, stride=3, views=None):
    n = obs.T * per
    out = []
    for k in range(n):
        t = k // per
        parts = [(m, c) for m, c, _ in scene_parts(ROBOT, res, t)]
        pp = []
        if pts and obs.depth is not None:
            P, C = fused(obs, t, stride, views)
            pp = [(P, (0.5 * C + 0.5 * 70).astype(np.uint8), 2)]
        out.append(frame_img(orbit(k, n, ctr, radius=radius, height=height), parts, pp))
    write(name, out)


def s7_objects():
    obs, res = peg()
    recon_orbit("p7_objects", res, obs, CTR, radius=1.3, height=0.75)
    # close-up: peg + target with inferred hole, no robot
    parts_all = []
    n = obs.T * 10
    out = []
    for k in range(n):
        t = k // 10
        parts = [(m, c) for m, c, nm in scene_parts(ROBOT, res, t) if not nm.startswith("robot")]
        a = -2.4 + 2.0 * k / n
        eye = np.array([0.03, 0.0, 0.1]) + np.array([0.75 * np.cos(a), 0.75 * np.sin(a), 0.45])
        out.append(frame_img(lookat_w2c(eye, np.array([0.03, 0.0, 0.06])), parts))
    write("p7_objects_close", out)


def s8_output():
    _, res = peg()
    for nm in ["standard/gt__mv"]:
        v = iio.imread(ROOT / "results" / nm / "overlay.mp4")
        write("p8_overlay", hold(list(v), 6))


def s9_results():
    o = load_gt512("standard")
    r = pickle.load(open(ROOT / "results/standard/gt512__mv/fit.pkl", "rb"))
    recon_orbit("r_peg_3d", r, o, CTR, radius=1.4, height=0.8, per=18)
    o = load_gt512("drawer")
    r = pickle.load(open(ROOT / "results/drawer/gt512__mv/fit.pkl", "rb"))
    c = np.asarray(r["robot"]["base"])[:3, 3] * 0.5 + np.array([0.1, 0.0, 0.2])
    recon_orbit("r_drawer_3d", r, o, c, radius=1.9, height=1.0, per=18)
    o = load_real("task_0004_user_0010_scene_0007_cfg_0005_recal")
    r = pickle.load(open(ROOT / "results/real/gt__mv_cal_recal/fit.pkl", "rb"))
    recon_orbit("r_real_3d", r, o, np.array([0.45, 0.0, 0.15]), radius=1.5, height=0.9, per=10)


if __name__ == "__main__":
    stages = sys.argv[1:] or ["s1_input", "s2_lift", "s3_table", "s4_search", "s5_track", "s6_residual", "s7_objects",
                              "s8_output", "s9_results"]
    for s in stages:
        t0 = time.time()
        globals()[s]()
        print(s, f"{time.time() - t0:.0f}s", TIMES, flush=True)
