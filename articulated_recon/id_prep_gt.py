"""GT mirror of a rollout dir for the inverse-dynamics test: results/id/gtsrc/<rollout name>/ with dense.npz + meta.json
(symlinks) and obs.npz = the sim's own RGB (not the prediction) -> color_masks.py -> wm_fit.py gt:<dir>.
python id_prep_gt.py <rollout dirs...>"""
import os, sys
from pathlib import Path
import numpy as np
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
for d in map(Path, sys.argv[1:]):
    o = ROOT / "results/id/gtsrc" / d.name
    o.mkdir(parents=True, exist_ok=True)
    for f in ("dense.npz", "meta.json"):
        if not (o / f).exists():
            os.symlink(d / f, o / f)
    if not (o / "obs.npz").exists():
        np.savez(o / "obs.npz", rgb=np.load(d / "dense.npz")["rgb"].transpose(1, 0, 2, 3, 4))
    print(o)
