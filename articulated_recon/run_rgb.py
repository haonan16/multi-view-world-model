"""python run_rgb.py <source> <scene> [run]  pixel-only baseline on camera 0 (no depth, no camera).
The fit lives in the robot-base frame; for evaluation only it is placed in the world with the GT camera."""
import pickle
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artirecon.data import load_dense, load_gt512
from artirecon.fit_rgb import RGBFitter

src, scene = sys.argv[1:3]
run = sys.argv[3] if len(sys.argv) > 3 else "B"
obs = load_gt512(scene) if src == "gt512" else load_dense(src, run=run)
tag = f"{scene}_{src}{'' if src in ('gt512', 'gt') else '_' + run}"
mz = np.load(fos.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "masks", f"{tag}.npz"))
masks = {k: mz[k] for k in mz.files}
d = Path(os.path.dirname(os.path.abspath(__file__))) / "results" / scene / f"{src}{'' if src in ('gt512', 'gt') else '_' + run}__mono_rgb"
d.mkdir(parents=True, exist_ok=True)
logf = open(d / "log.txt", "w")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n"); logf.flush()


t0 = time.time()
log(f"[{obs.label}/mono_rgb] pixel-only, view 0, masks {list(masks)}")
r = RGBFitter(obs.rgb[:, 0], masks, log=log).run()
world_T_base = obs.c2w[0] @ r["w2c"]          # evaluator-side placement only
objs = []
for o in r["objects"]:
    poses = np.einsum("ij,tjk->tik", world_T_base, o["poses"])
    objs.append(dict(name=o["name"], type="free", half=o["half"], poses=poses, iou=o["iou"]))
res = dict(label=obs.label + "/mono_rgb", source=src, scene=scene, condition="mono_rgb", steps=obs.steps, views=[0],
           robot=dict(base=world_T_base, q=r["q"], iou=r["robot_iou"], w2c_pred=r["w2c"], K_pred=r["K"]),
           objects=objs, timing={"total_s": time.time() - t0}, run=run)
pickle.dump(res, open(d / "fit.pkl", "wb"))
log(f"camera centre error (world) {np.linalg.norm(np.linalg.inv(world_T_base)[:3, 3] * 0 + (world_T_base @ np.linalg.inv(r['w2c']))[:3, 3] - obs.c2w[0][:3, 3]) * 1000:.1f} mm (by construction 0: base absorbs it)")
log(f"saved {d}  {time.time() - t0:.1f}s")
