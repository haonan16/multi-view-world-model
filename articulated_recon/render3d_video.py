"""Add a third row (the 3D reconstruction alone: robot + objects, shaded, dark background, same cameras) to
results/real_v2/<ep>/overlay.mp4 -> overlay3.mp4 (CPU; reads fit.pkl + robot.pkl).
python render3d_video.py <episode list>"""
import os
import pickle
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import trimesh

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))
import artirecon  # noqa: F401
from artirecon.data import load_real
from artirecon.robot import URDFRobot
from artirecon.splat import render_points

torch.set_num_threads(8)
robot = URDFRobot(n_pts_per_m2=150000, device="cpu")
COL = {"robot": (0.78, 0.86, 0.98), "free": (1.0, 0.62, 0.25), "static": (0.85, 0.80, 0.55)}
BG = np.array([30, 30, 34], np.uint8)
for e in open(sys.argv[1]).read().split():
    d = ROOT / "results/real_v2" / e
    out = d / "overlay3.mp4"
    if out.exists() and out.stat().st_mtime > (d / "fit.pkl").stat().st_mtime:
        continue
    res = pickle.load(open(d / "fit.pkl", "rb"))
    R = pickle.load(open(d / "robot.pkl", "rb"))
    obs = load_real("v2/" + e)
    c2w = R["c2w"]
    T, V = obs.T, obs.V
    H, W = obs.rgb.shape[2:4]
    rob = res["robot"]
    with torch.no_grad():
        RP, RN = robot.surface(torch.as_tensor(rob["q"], dtype=torch.float32),
                               torch.as_tensor(rob["base"], dtype=torch.float32)[None].expand(T, 4, 4))
    RP, RN = RP.numpy(), RN.numpy()
    samp = []
    for ob in res["objects"]:
        tm = trimesh.Trimesh(ob["mesh"][0], ob["mesh"][1], process=False)
        sp, fi = trimesh.sample.sample_surface(tm, int(max(8000, tm.area * 1500000)), seed=0)
        samp.append((np.asarray(sp), np.asarray(tm.face_normals[fi]), COL.get(ob["type"], COL["free"])))
    vid = iio.imread(d / "overlay.mp4")                     # T, 2H, V*W, 3
    frames = []
    for t in range(T):
        parts = [(RP[t], RN[t], COL["robot"])]
        for (sp, sn, col), ob in zip(samp, res["objects"]):
            P = ob["poses"][t]
            parts.append((sp @ P[:3, :3].T + P[:3, 3], sn @ P[:3, :3].T, col))
        row = []
        for v in range(V):
            img, alpha, _, _ = render_points(parts, obs.K[v], np.linalg.inv(c2w[v]), H, W, dev="cpu",
                                             footprint=[2] + [1] * len(samp))
            im = np.where(alpha[..., None], (img * 255).clip(0, 255).astype(np.uint8), BG)
            row.append(im)
        r3 = np.concatenate(row, 1)
        top = vid[min(t, len(vid) - 1)]
        frames.append(np.concatenate([top[:, :r3.shape[1]], r3], 0))
    iio.imwrite(out, np.stack(frames), fps=3, codec="libx264", macro_block_size=8,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "23"])
    print("wrote", out, flush=True)
