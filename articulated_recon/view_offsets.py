"""Per-view 2D residuals of the world model's generated views: for every rollout and frame, triangulate each
colour object from all views (largest blob) and record (observed - reprojected) per view.  A view whose
residual has a consistent direction across rollouts is systematically shifted (a per-view correction exists).
python view_offsets.py <rollout root> [task prefix]"""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from artirecon.wm_objects import _ray, _tri
from artirecon.data import MSR
RANGES = {"red": [((0, 150, 70), (5, 255, 255)), ((171, 150, 70), (180, 255, 255))], "green": [((40, 100, 50), (90, 255, 255))]}


def blob(im, col):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV); m = np.zeros(im.shape[:2], np.uint8)
    for lo, hi in RANGES[col]:
        m |= cv2.inRange(hsv, lo, hi)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return cen[k] if st[k, cv2.CC_STAT_AREA] >= 6 else None


root = Path(sys.argv[1]); pre = sys.argv[2] if len(sys.argv) > 2 else "StackCube"
res = {v: [] for v in range(4)}; res_f0 = {v: [] for v in range(4)}
for d in sorted(root.glob(pre + "*")):
    m = json.loads((d / "meta.json").read_text()); z = np.load(d / "dense.npz"); rgb = z["rgb_pred"]
    K = np.array(m["K"]); c2w = np.load(MSR / m["episode"] / "cameras.npz")["c2w"][:4].astype(float)
    for t in range(rgb.shape[1]):
        for col in ("green",):          # static object: cleanest
            obs = [(v, blob(rgb[v, t], col)) for v in range(4)]
            obs = [(v, b) for v, b in obs if b is not None]
            if len(obs) < 3:
                continue
            X = _tri([_ray(K[v], c2w[v], b) for v, b in obs])
            for v, b in obs:
                pc = np.linalg.inv(c2w[v]) @ np.r_[X, 1]
                r = b - (K[v] @ pc[:3])[:2] / pc[2]
                (res_f0 if t == 0 else res)[v].append(r)
for v in range(4):
    a = np.array(res[v]); b = np.array(res_f0[v])
    print(f"view {v}: t>0 n={len(a)} median residual {np.round(np.median(a, 0), 1)} (spread {np.round(np.median(np.abs(a - np.median(a, 0)), 0), 1)}) | "
          f"t=0 median {np.round(np.median(b, 0), 1) if len(b) else None}")
