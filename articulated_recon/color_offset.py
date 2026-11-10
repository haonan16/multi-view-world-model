"""Per-camera colour-vs-geometry pixel offset, estimated on the robot (independent of the objects).
The robot fit (multiview depth, ~3 mm) is rendered into every view/frame; SAM 3 'robot arm' masks (colour image)
are shifted over a +-12 px grid; the shift maximising the summed IoU is the camera's colour offset (RealSense
colour/depth registration + residual extrinsic error).  -> real/v2/<ep>/color_offset.json {view: [dx, dy]}
python color_offset.py <episode list>"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))
import artirecon  # noqa: F401
from artirecon.data import load_real
from artirecon.objects_v2 import Sam3Masks
from artirecon.robot import URDFRobot
from artirecon.splat import render_points

robot = URDFRobot(n_pts_per_m2=150000)
for e in open(sys.argv[1]).read().split():
    d = ROOT / "real/v2" / e
    if not (d / "sam3_robot.npz").exists():
        print("no robot masks", e); continue
    R = pickle.load(open(ROOT / "results/real_v2" / e / "robot.pkl", "rb"))
    obs = load_real("v2/" + e)
    T, V, H, W = obs.T, obs.V, obs.rgb.shape[2], obs.rgb.shape[3]
    rob = R["rob"]
    with torch.no_grad():
        P, N = robot.surface(torch.as_tensor(rob["q"], dtype=torch.float32, device="cuda"),
                             torch.as_tensor(rob["base"], dtype=torch.float32, device="cuda")[None].expand(T, 4, 4))
    P, N = P.cpu().numpy(), N.cpu().numpy()
    sam = Sam3Masks(d / "sam3_robot.npz", H, W)
    out, rep = {}, {}
    sh = list(range(-12, 13))
    for v in range(V):
        score = np.zeros((len(sh), len(sh)))
        n = 0
        for t in range(0, T, 2):
            ids = sam.ids("robot arm", t, v)
            M = np.zeros((H, W), bool)
            for i in ids:
                if sam.score[i] > 0.4:
                    M |= sam.mask(i)
            if M.sum() < 500:
                continue
            _, A, _, _ = render_points([(P[t], N[t], (1, 1, 1))], obs.K[v], np.linalg.inv(R["c2w"][v]), H, W)
            if A.sum() < 500:
                continue
            Mt = torch.as_tensor(M, device="cuda").float(); At = torch.as_tensor(A, device="cuda").float()
            for a, dy in enumerate(sh):
                for b, dx in enumerate(sh):
                    Ms = torch.roll(Mt, shifts=(-dy, -dx), dims=(0, 1))       # mask shifted into the geometry frame
                    inter = (Ms * At).sum(); uni = ((Ms + At) > 0).float().sum()
                    score[a, b] += float(inter / uni)
            n += 1
        if n == 0:
            out[v] = [0.0, 0.0]; continue
        a, b = np.unravel_index(score.argmax(), score.shape)
        out[v] = [float(sh[b]), float(sh[a])]
        rep[v] = dict(n=n, iou0=round(score[12, 12] / n, 3), iou_best=round(score[a, b] / n, 3))
    json.dump(dict(offset_px=out, report=rep), open(d / os.environ.get("COFF_OUT", "color_offset.json"), "w"), indent=1)
    print(e, out, rep, flush=True)
