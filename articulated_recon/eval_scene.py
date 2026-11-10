"""python eval_scene.py <result_dir>  -> metrics.json + overlay renders (all 4 cameras, also unused ones)"""
import json
import pickle
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artirecon.data import load_dense, load_gt512
from artirecon.evaluate import evaluate, render_agreement
from artirecon.render import overlay, render, scene_parts
from artirecon.robot import URDFRobot

d = Path(sys.argv[1])
res = pickle.load(open(d / "fit.pkl", "rb"))
robot = URDFRobot(n_pts_per_m2=20000)
m = evaluate(res, robot)
src = res["source"]
if src == "gt512":
    full = load_gt512(res["scene"])
    gt_obs = full
else:
    full = load_dense(src, run=res.get("run", "B")) if src != "gt" else load_dense("gt")
    gt_obs = None
ra = render_agreement(res, robot, full, gt_obs)
m["depth_agree_1cm"] = ra["depth_agree_1cm"].tolist()
m["iou"] = ra["iou"].tolist()
if src != "gt512" and src != "gt":           # also agreement with the real GT depth (est / roll)
    g = load_dense("gt")
    m["depth_agree_1cm_vs_gt"] = render_agreement(res, robot, g)["depth_agree_1cm"].tolist()
m["timing"] = res["timing"]
m["objects"] = [(o["name"], o["type"]) for o in res["objects"]]
(d / "metrics.json").write_text(json.dumps(m, indent=1))
# overlays: frames x views grid
rows = []
for t in range(full.T):
    parts = [(mm, c) for mm, c, _ in scene_parts(robot, res, t)]
    row = []
    for v in range(full.V):
        H, W = full.rgb.shape[2:4]
        img, alpha, dep, pid = render(parts, full.K[v], np.linalg.inv(full.c2w[v]), H, W)
        row.append(overlay(full.rgb[t, v], img, alpha, 0.7))
    rows.append(np.concatenate(row, 1))
grid = np.concatenate(rows, 0)
iio.imwrite(d / "overlay_grid.jpg", grid[::2, ::2], quality=88)
summ = {k: (np.round(np.nanmean(np.array(v, dtype=float)), 3) if isinstance(v, list) and v and not isinstance(v[0], (str, tuple, list)) else v)
        for k, v in m.items() if k not in ("depth_agree_1cm", "iou", "depth_agree_1cm_vs_gt", "objects", "timing")}
summ["iou_per_view"] = np.round(np.nanmean(np.array(m["iou"]), 0), 3).tolist()
summ["depth_agree_per_view"] = np.round(np.nanmean(np.array(m["depth_agree_1cm"]), 0), 3).tolist()
print(json.dumps(summ, default=str))
