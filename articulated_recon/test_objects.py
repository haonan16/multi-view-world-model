"""re-run only the object stage on a saved fit (robot reused): python test_objects.py <result_dir>"""
import pickle
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artirecon.scene_fit as SF
from artirecon.data import load_dense, load_gt512
from artirecon.evaluate import evaluate
from artirecon.robot import URDFRobot

d = sys.argv[1]
res = pickle.load(open(d + "/fit.pkl", "rb"))
saved = res["robot"]
SF.RobotFitter.run = lambda self, **k: saved
obs = load_gt512(res["scene"]) if res["source"] == "gt512" else load_dense(res["source"], run=res.get("run", "B"))
if res["condition"].startswith("mono"):
    obs = obs.select(views=[0])
rob = URDFRobot(n_pts_per_m2=20000)
sf = SF.SceneFitter(obs, robot=rob).run()
res["objects"] = sf.objects
from artirecon.evaluate import plausibility
m = evaluate(res, rob); m.update(plausibility(res, rob))
for k, v in m.items():
    if any(w in k for w in ("peg", "box", "travel", "axis", "drawer", "cabinet", "penetration", "gap")):
        print(k, np.round(v, 1) if isinstance(v, list) else v)
if "--save" in sys.argv:
    pickle.dump(res, open(d + "/fit.pkl", "wb"))
