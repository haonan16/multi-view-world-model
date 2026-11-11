"""Per-view 2D similarity (scale s about the image centre + shift) of the generated views, estimated on the robot:
the fitted robot is rendered with the true cameras; the SAM 3 'robot arm' mask of the predicted image is warped by
the inverse similarity; (s, tx, ty) maximising the IoU over all frames.  Test: do the same warps make the cubes'
colour blobs triangulate consistently (cross-view reprojection error before -> after)?
python view_sim.py <rollout names...>   -> results/wm/view_sim.json"""
import json, pickle, sys
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as Fn
sys.path.insert(0, ".")
import artirecon  # noqa
from artirecon.data import load_roll
from artirecon.robot import URDFRobot
from artirecon.splat import render_points
from artirecon.objects_v2 import Sam3Masks
from artirecon.wm_objects import _ray, _tri
R = Path("/n/lab_storage/ydu_lab/Lab/akiruga/geo_runs/rollouts_A7000")
rob = URDFRobot(n_pts_per_m2=150000)
RANGES = {"red": [((0, 150, 70), (5, 255, 255)), ((171, 150, 70), (180, 255, 255))], "green": [((40, 100, 50), (90, 255, 255))]}


def blob(im, col):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV); m = np.zeros(im.shape[:2], np.uint8)
    for lo, hi in RANGES[col]:
        m |= cv2.inRange(hsv, lo, hi)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, 4]))
    return cen[k] if st[k, 4] >= 15 else None


def warp_uv(uv, sim, W, H):      # generated-image pixel -> geometry pixel (inverse similarity)
    s, tx, ty = sim; c = np.array([(W - 1) / 2, (H - 1) / 2])
    return (uv - c - [tx, ty]) / s + c


out = {}
for n in sys.argv[1:]:
    d = R / n; o = load_roll(d); T, V, H, W = o.T, o.V, o.rgb.shape[2], o.rgb.shape[3]
    F = pickle.load(open(f"results/wm/{n}_da3_final/fit_raw.pkl" if Path(f"results/wm/{n}_da3_final/fit_raw.pkl").exists() else f"results/wm/{n}_da3_final/fit.pkl", "rb"))
    sm = Sam3Masks(d / "sam3.npz", H, W)
    with torch.no_grad():
        P, N = rob.surface(torch.tensor(F["rob"]["q"], dtype=torch.float32, device="cuda"), torch.tensor(F["Tb"], dtype=torch.float32, device="cuda")[None].expand(T, 4, 4))
    P, N = P.cpu().numpy(), N.cpu().numpy()
    sims = {}
    for v in range(V):
        A, M = [], []
        for t in range(T):
            m = np.zeros((H, W), bool)
            for i in sm.ids("robot arm", t, v):
                if sm.score[i] > 0.4:
                    m |= sm.mask(i)
            if m.sum() < 300:
                continue
            _, a, _, _ = render_points([(P[t], N[t], (1, 1, 1))], o.K[v], np.linalg.inv(o.c2w[v]), H, W)
            A.append(np.asarray(a.cpu() if hasattr(a, "cpu") else a)); M.append(m)
        if not A:
            continue
        A = torch.as_tensor(np.stack(A), device="cuda").float(); M = torch.as_tensor(np.stack(M), device="cuda").float()[:, None]
        best = (-1, None)
        for s in np.arange(0.86, 1.141, 0.02):
            for tx in range(-30, 31, 2):
                ys = torch.arange(-30, 31, 2, device="cuda").float()
                # build a batch of grids over ty: geometry pixel p -> generated pixel s*(p-c)+c+t
                xs_ = torch.linspace(-1, 1, W, device="cuda"); ys_ = torch.linspace(-1, 1, H, device="cuda")
                gy, gx = torch.meshgrid(ys_, xs_, indexing="ij")
                gxs = (gx * s + tx * 2 / (W - 1))[None].expand(len(ys), H, W)
                gys = gy[None] * s + (ys * 2 / (H - 1))[:, None, None]
                grid = torch.stack([gxs, gys], -1)                                   # ty batch
                sc = torch.zeros(len(ys), device="cuda")
                for k in range(len(A)):
                    Mw = Fn.grid_sample(M[k:k + 1].expand(len(ys), 1, H, W), grid, align_corners=True)[:, 0] > 0.5
                    inter = (Mw & (A[k] > 0)).flatten(1).sum(1).float(); uni = (Mw | (A[k] > 0)).flatten(1).sum(1).float()
                    sc += inter / uni.clamp(min=1)
                j = int(sc.argmax())
                if float(sc[j]) > best[0]:
                    best = (float(sc[j]), (float(s), float(tx), float(ys[j])), float(sc[j]) / len(A))
        sims[v] = best[1]; 
    # cube consistency before / after
    res = {}
    for col in ("red", "green"):
        eb, ea = [], []
        for t in range(T):
            obs = [(v, blob(o.rgb[t, v], col)) for v in range(V)]; obs = [(v, b) for v, b in obs if b is not None]
            if len(obs) < 3:
                continue
            for corr, acc in ((False, eb), (True, ea)):
                uvs = [(v, warp_uv(b, sims[v], W, H) if corr and v in sims else b) for v, b in obs]
                X = _tri([_ray(o.K[v], o.c2w[v], uv) for v, uv in uvs])
                e = [np.linalg.norm((o.K[v] @ (np.linalg.inv(o.c2w[v]) @ np.r_[X, 1])[:3])[:2] / (np.linalg.inv(o.c2w[v]) @ np.r_[X, 1])[2] - uv) for v, uv in uvs]
                acc.append(max(e))
        res[col] = (float(np.median(eb)) if eb else None, float(np.median(ea)) if ea else None)
    out[n] = dict(sims=sims, cube_err_before_after=res)
    print(n, {v: tuple(round(x, 2) for x in s_) for v, s_ in sims.items()}, "cube worst-view err px before->after", {k: tuple(round(x, 1) if x else None for x in vv) for k, vv in res.items()}, flush=True)
json.dump(out, open("results/wm/view_sim.json", "w"), indent=1)
