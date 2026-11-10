"""3D + pixel hybrid: start from a depth-based fit (fit.pkl), refine the robot joints with
  (multiview depth loss, geom.py)  +  w * (soft-silhouette IoU against text-prompted masks, all views)
The 3D term puts the robot in the right basin; the pixel term sharpens edges where depth is noisy.

python hybrid.py <result_dir> <masks_allviews.npz> [w_pix] -> <result_dir>_hybrid/{fit.pkl, log.txt}
"""
import copy
import pickle
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artirecon.data import load_dense, load_gt512
from artirecon.evaluate import evaluate
from artirecon.fit_rgb import SilRenderer, SilRobot, soft_iou
from artirecon.fit_robot import RobotFitter, prepare
from artirecon.robot import URDFRobot

d = Path(sys.argv[1])
mz = np.load(sys.argv[2])
w_pix = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
res = pickle.load(open(d / "fit.pkl", "rb"))
src = res["source"]
obs = load_gt512(res["scene"]) if src == "gt512" else load_dense(src, run=res.get("run", "B"))
if res["condition"].startswith("mono"):
    obs = obs.select(views=[0])
if res["robot"].get("depth_scale") is not None:
    obs = copy.copy(obs)
    obs.depth = obs.depth * res["robot"]["depth_scale"][:, :, None, None].astype(np.float32)
t0 = time.time()
rob = URDFRobot(n_pts_per_m2=20000)
prep = prepare(obs)
rf = RobotFitter(obs, prep, rob, known_base=res["robot"]["base"])
sil = SilRobot(rob)
H, W = obs.rgb.shape[2:4]
R = SilRenderer(H, W, res=160)
V, T = obs.V, obs.T
views = res.get("views") or list(range(V))
M = torch.as_tensor(mz["robot"][:, views].astype(np.float32), device="cuda")              # T,V,H,W
M = Fnn.interpolate(M.flatten(0, 1)[:, None], size=(R.h, R.w), mode="area")[:, 0].unflatten(0, (T, V))
K = torch.as_tensor(obs.K, dtype=torch.float32, device="cuda")
W2C = torch.linalg.inv(torch.as_tensor(obs.c2w, dtype=torch.float64, device="cuda")).float()
base = torch.as_tensor(res["robot"]["base"], dtype=torch.float32, device="cuda")
q8 = torch.as_tensor(res["robot"]["q"][:, :8], dtype=torch.float32, device="cuda").clone().requires_grad_(True)
opt = torch.optim.Adam([q8], lr=0.004)
frames = list(range(T))
B = base[None].expand(T, 4, 4)
for it in range(150):
    L3, _, _ = rf.loss(q8, B, frames, 1.0, 0.01)
    v = sil.verts(torch.cat([q8, q8[:, -1:]], 1), B)                                      # T,N,3
    vv = v[:, None].expand(T, V, -1, -1).flatten(0, 1)
    S = R(vv, sil.f, K[None].expand(T, V, 3, 3).flatten(0, 1), W2C[None].expand(T, V, 4, 4).flatten(0, 1))
    Lp = soft_iou(S, M.flatten(0, 1)).view(T, V).mean(1)
    lim = (torch.relu(rf.lo[:8] - q8) + torch.relu(q8 - rf.hi[:8])).pow(2).sum(1) * 100
    tot = (L3 + w_pix * Lp + lim).sum()
    opt.zero_grad(); tot.backward(); opt.step()
    if it % 50 == 0:
        print(it, f"3D {L3.mean().item():.4f} pix(1-IoU) {Lp.mean().item():.4f}", flush=True)
out = Path(str(d) + f"_hybrid")
out.mkdir(exist_ok=True)
r2 = copy.deepcopy(res)
r2["robot"]["q"] = torch.cat([q8, q8[:, -1:]], 1).detach().cpu().numpy()
r2["condition"] = res["condition"] + "_hybrid"
r2["label"] = res["label"] + "+pixels"
r2["timing"] = dict(res["timing"], hybrid_s=time.time() - t0)
pickle.dump(r2, open(out / "fit.pkl", "wb"))
m0, m1 = evaluate(res, rob), evaluate(r2, rob)
msg = (f"robot surface {np.mean(m0['robot_surface_mm']):.1f} -> {np.mean(m1['robot_surface_mm']):.1f} mm, gripper "
       f"{np.mean(m0['gripper_surface_mm']):.1f} -> {np.mean(m1['gripper_surface_mm']):.1f} mm, joints "
       f"{np.mean(m0['joint_err_deg']):.2f} -> {np.mean(m1['joint_err_deg']):.2f} deg  [{time.time() - t0:.0f}s]")
print(msg)
(out / "log.txt").write_text(msg + "\n")
