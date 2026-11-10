"""python run_scene.py <source> <scene> <condition> [out_root]
source: gt512 | gt | est | roll   condition: mv (4 views + depth + cams) | mono_depth (view 0 + depth + cam)
"""
import pickle
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artirecon.data import load_dense, load_gt512
from artirecon.robot import URDFRobot
from artirecon.scene_fit import SceneFitter

src, scene, cond = sys.argv[1:4]
out = Path(sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
run = sys.argv[5] if len(sys.argv) > 5 else "B"
obs = load_gt512(scene) if src == "gt512" else load_dense(src, run=run)
if cond == "mono_depth":
    obs = obs.select(views=[0], label=obs.label + "/mono_depth")
else:
    obs.label += "/mv"
d = out / scene / f"{src}{'' if src in ('gt512', 'gt') else '_' + run}__{cond}"
d.mkdir(parents=True, exist_ok=True)
logf = open(d / "log.txt", "w")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


t0 = time.time()
sf = SceneFitter(obs, log=log, robot=URDFRobot(n_pts_per_m2=20000)).run()
res = dict(label=obs.label, source=src, scene=scene, condition=cond, steps=obs.steps, views=obs.meta.get("views"),
           robot=sf.rob, objects=sf.objects, plane=(sf.prep["n"], sf.prep["d"]), timing=sf.timing, tcp=sf.tcp)
pickle.dump(res, open(d / "fit.pkl", "wb"))
log(f"saved {d}  total {time.time() - t0:.1f}s")
