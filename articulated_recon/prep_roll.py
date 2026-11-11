"""for SAM 3: write <rollout dir>/obs.npz (rgb [T,V,H,W,3] = predicted RGB) and ann.json prompts.
python prep_roll.py <dirs...>"""
import json, sys
from pathlib import Path
import numpy as np
P = {"StackCube-v1": ["red cube", "green cube"], "PullCubeTool-v1": ["red L-shaped tool", "blue cube"],
     "PegInsertionSide-v1": ["red and white peg", "cardboard box"]}
for d in map(Path, sys.argv[1:]):
    if (d / "obs.npz").exists():
        continue
    z = np.load(d / "dense.npz"); m = json.loads((d / "meta.json").read_text())
    np.savez(d / "obs.npz", rgb=z["rgb_pred"].transpose(1, 0, 2, 3, 4))
    json.dump({"prompts": P[m["episode"].split("/")[0]] + ["robot arm"]}, open(d / "ann.json", "w"))
    print(d.name)
