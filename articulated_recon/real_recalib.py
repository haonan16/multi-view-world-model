"""Per-camera 6-DoF extrinsic refinement on a real episode, using the robot as the calibration target:
the URDF posed at the logged joint angles (proprioception) vs each camera's depth points near the robot.
Writes real/<episode>_recal/obs.npz (same data, refined w2c) + recal.json."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import knn_points

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artirecon.data import REAL, backproject, load_real
from artirecon.robot import URDFRobot

ep = sys.argv[1] if len(sys.argv) > 1 else "task_0004_user_0010_scene_0007_cfg_0005"
obs = load_real(ep)
gts = obs.meta["gt_states"]
rob = URDFRobot(n_pts_per_m2=20000)
with torch.no_grad():
    q = torch.tensor(np.stack([g["q"] for g in gts]), dtype=torch.float32, device="cuda")
    P, _ = rob.surface(q, torch.eye(4, device="cuda")[None].expand(len(q), 4, 4))     # T,N,3
z = dict(np.load(REAL / ep / "obs.npz"))
w2c_new = z["w2c"].astype(np.float64).copy()
report = {}
for v in range(obs.V):
    src, dst = [], []
    for t in range(obs.T):
        Pv, _ = backproject(obs.depth[t, v], obs.K[v], obs.c2w[v], 2)
        src.append(Pv); dst.append(P[t])
    S = np.concatenate(src)
    S = torch.as_tensor(S, dtype=torch.float32, device="cuda")
    D = torch.cat(dst)
    Tacc = torch.eye(4, device="cuda")
    d0 = None
    for it, rad in enumerate(np.linspace(0.05, 0.015, 25)):
        Sx = S @ Tacc[:3, :3].T + Tacc[:3, 3]
        kn = knn_points(Sx[None], D[None], K=1)
        dist = kn.dists[0, :, 0].sqrt()
        m = dist < rad
        if d0 is None:
            d0 = float(dist[dist < 0.05].median())
        a, b = Sx[m], D[kn.idx[0, m, 0]]
        ca, cb = a.mean(0), b.mean(0)
        U, _, Vt = torch.linalg.svd((a - ca).T @ (b - cb))
        R = (U @ Vt).T
        if torch.det(R) < 0:
            Vt[-1] *= -1
            R = (U @ Vt).T
        Ti = torch.eye(4, device="cuda"); Ti[:3, :3] = R; Ti[:3, 3] = cb - R @ ca
        Tacc = Ti @ Tacc
    Sx = S @ Tacc[:3, :3].T + Tacc[:3, 3]
    dist = knn_points(Sx[None], D[None], K=1).dists[0, :, 0].sqrt()
    d1 = float(dist[dist < 0.05].median())
    Tc = Tacc.double().cpu().numpy()                       # world correction: x_world' = Tc x_world
    c2w = Tc @ np.linalg.inv(w2c_new[v])
    w2c_new[v] = np.linalg.inv(c2w)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Tc[:3, :3]) - 1) / 2, -1, 1))))
    report[str(v)] = dict(rot_deg=ang, trans_mm=float(np.linalg.norm(Tc[:3, 3]) * 1000),
                          robot_median_dist_mm_before=d0 * 1000, after=d1 * 1000)
    print(v, report[str(v)], flush=True)
out = REAL / (ep + "_recal")
out.mkdir(exist_ok=True)
z["w2c"] = w2c_new.astype(np.float32)
np.savez_compressed(out / "obs.npz", **z)
(out / "recal.json").write_text(json.dumps(report, indent=1))
(out / "meta.json").write_text((REAL / ep / "meta.json").read_text())
