"""python run_masks.py <source> <scene> [run] -> results/masks/<scene>_<source>.npz (view 0 only) + preview"""
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artirecon.data import load_dense, load_gt512
from artirecon.masks import segment_video

src, scene = sys.argv[1:3]
run = sys.argv[3] if len(sys.argv) > 3 else "B"
if scene == "real":
    from artirecon.data import load_real
    obs = load_real()
elif src == "gt512":
    obs = load_gt512(scene)
elif scene == "drawer":
    from artirecon.data import load_dense_drawer
    obs = load_dense_drawer(src, run=run)
else:
    obs = load_dense(src, run=run)
rgb = obs.rgb[:, 0]
m = segment_video(rgb, scene)
out = Path(os.path.dirname(os.path.abspath(__file__))) / "results" / "masks"
out.mkdir(parents=True, exist_ok=True)
tag = f"{scene}_{src}{'' if src in ('gt512', 'gt') else '_' + run}"
np.savez_compressed(out / f"{tag}.npz", robot=m["robot"], **{k.strip('.'): v for k, v in m["objects"].items()})
vis = rgb.astype(np.float32).copy()
vis[m["robot"]] = vis[m["robot"]] * 0.4 + np.array([80, 120, 255]) * 0.6
cols = [np.array([255, 140, 0]), np.array([0, 220, 120])]
for i, (k, v) in enumerate(m["objects"].items()):
    vis[v] = vis[v] * 0.4 + cols[i % 2] * 0.6
iio.imwrite(out / f"{tag}.jpg", np.concatenate(list(vis.astype(np.uint8)), 1)[::2, ::2])
print("robot px", m["robot"].sum((1, 2)), {k: v.sum((1, 2)) for k, v in m["objects"].items()})
