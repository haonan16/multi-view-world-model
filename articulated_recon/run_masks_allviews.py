"""robot masks for every view of a source (for the 3D + pixel hybrid). python run_masks_allviews.py <src> <scene> [run]"""
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artirecon.data import load_dense, load_gt512
from artirecon.masks import PROMPTS, Segmenter

src, scene = sys.argv[1:3]
run = sys.argv[3] if len(sys.argv) > 3 else "B"
obs = load_gt512(scene) if src == "gt512" else load_dense(src, run=run)
seg = Segmenter()
M = np.zeros(obs.rgb.shape[:4], bool)
for t in range(obs.T):
    for v in range(obs.V):
        M[t, v], _ = seg.masks(obs.rgb[t, v], PROMPTS[scene]["robot"], top=1, box_thr=0.25)
out = Path(os.path.dirname(os.path.abspath(__file__))) / "results" / "masks"
np.savez_compressed(out / f"{scene}_{src}_allviews.npz", robot=M)
print("robot px per view", M.sum((2, 3)).mean(0))
