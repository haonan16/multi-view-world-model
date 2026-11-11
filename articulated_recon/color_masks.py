"""Colour instance masks for the saturated-colour ManiSkill objects (sim render / predicted RGB): small cubes are
only ~8 px wide at 448x256, where SAM 3 text prompts miss most of them.  HSV threshold -> connected components.
-> <dir>/color.npz in the sam3.npz format (prompt names = the annotation prompts).  python color_masks.py <dirs...>"""
import json, sys
from pathlib import Path
import cv2, numpy as np
RANGES = {"red": [((0, 150, 70), (5, 255, 255)), ((171, 150, 70), (180, 255, 255))],
          "green": [((40, 100, 50), (90, 255, 255))], "blue": [((95, 100, 40), (135, 255, 255))]}
TASK = {"StackCube-v1": {"red cube": "red", "green cube": "green"},
        "PullCubeTool-v1": {"red L-shaped tool": "red", "blue cube": "blue"},
        "PegInsertionSide-v1": {}}
for d in map(Path, sys.argv[1:]):
    m = json.loads((d / "meta.json").read_text())
    spec = TASK[m["episode"].split("/")[0]]
    rgb = np.load(d / "obs.npz")["rgb"]
    T, V, H, W, _ = rgb.shape
    rec = dict(p=[], t=[], v=[], score=[], box=[], mask=[])
    prompts = list(spec)
    for pi, pr in enumerate(prompts):
        for t in range(T):
            for v in range(V):
                hsv = cv2.cvtColor(rgb[t, v], cv2.COLOR_RGB2HSV)
                mk = np.zeros((H, W), np.uint8)
                for lo, hi in RANGES[spec[pr]]:
                    mk |= cv2.inRange(hsv, lo, hi)
                mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                n, lab, st, _ = cv2.connectedComponentsWithStats(mk, connectivity=8)
                for k in range(1, n):
                    if st[k, cv2.CC_STAT_AREA] < 4:
                        continue
                    c = lab == k
                    ys, xs = np.nonzero(c)
                    rec["p"].append(pi); rec["t"].append(t); rec["v"].append(v); rec["score"].append(1.0)
                    rec["box"].append([xs.min(), ys.min(), xs.max(), ys.max()]); rec["mask"].append(np.packbits(c))
    np.savez_compressed(d / "color.npz", prompts=np.array(prompts or ["__none__"]), p=np.array(rec["p"], int),
                        t=np.array(rec["t"], int), v=np.array(rec["v"], int), score=np.array(rec["score"]),
                        box=np.array(rec["box"], np.float32).reshape(-1, 4),
                        mask=np.array(rec["mask"]).reshape(len(rec["p"]), -1) if rec["p"] else np.zeros((0, (H * W + 7) // 8), np.uint8),
                        shape=np.array([H, W]))
    print(d.name, {pr: int(sum(1 for p in rec["p"] if p == i)) for i, pr in enumerate(prompts)}, "instances over", T * V, "images")
