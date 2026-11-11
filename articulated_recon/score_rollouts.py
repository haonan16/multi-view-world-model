"""Score world-model rollouts for selection (CPU, colour masks, no depth):
 multiview consistency: per frame and object, RANSAC triangulation of the colour-blob centroids over the 4 views
   -> number of agreeing views and their reprojection error (px)
 task success from the triangulated 3D positions:
   StackCube: red cube ends on the green one (centre ~4 cm above it, < 2 cm apart in xy), gripper away
   PullCubeTool: blue cube pulled towards the robot base by > 5 cm
   PegInsertionSide: not scored here (judged by eye)
python score_rollouts.py <rollout root> -> <root>/scores.json"""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from artirecon.wm_objects import _ray, _tri
from artirecon.data import MSR
RANGES = {"red": [((0, 150, 70), (5, 255, 255)), ((171, 150, 70), (180, 255, 255))],
          "green": [((40, 100, 50), (90, 255, 255))], "blue": [((95, 100, 40), (135, 255, 255))]}
OBJ = {"StackCube-v1": {"red": "red", "green": "green"}, "PullCubeTool-v1": {"tool": "red", "cube": "blue"}, "PegInsertionSide-v1": {"peg": "red"}}


def blobs(im, col):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV); m = np.zeros(im.shape[:2], np.uint8)
    for lo, hi in RANGES[col]:
        m |= cv2.inRange(hsv, lo, hi)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    return [(cen[k], st[k, cv2.CC_STAT_AREA]) for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 4]


def locate(K, c2w, cands, tol=6.0):
    best = None
    vs = [v for v in range(len(cands)) if cands[v]]
    for i, v1 in enumerate(vs):
        for v2 in vs[i + 1:]:
            for c1, a1 in cands[v1]:
                for c2, a2 in cands[v2]:
                    X = _tri([_ray(K[v1], c2w[v1], c1), _ray(K[v2], c2w[v2], c2)])
                    if X[2] < -0.02 or X[2] > 0.6:
                        continue
                    inl, err = [], []
                    for v in vs:
                        pc = np.linalg.inv(c2w[v]) @ np.r_[X, 1]
                        if pc[2] < 0.05:
                            continue
                        q = (K[v] @ pc[:3])[:2] / pc[2]
                        e = min(np.linalg.norm(c - q) for c, _ in cands[v])
                        if e < tol:
                            inl.append(v); err.append(e)
                    sc = (len(inl), -np.mean(err) if err else 0)
                    if best is None or sc > best[0]:
                        best = (sc, X, inl, float(np.mean(err)) if err else 99.0)
    return best


root = Path(sys.argv[1])
out = {}
for d in sorted(root.iterdir()):
    if not (d / "meta.json").exists():
        continue
    m = json.loads((d / "meta.json").read_text()); task = m["episode"].split("/")[0]
    z = np.load(d / "dense.npz"); rgb = z["rgb_pred"]; V, T = rgb.shape[:2]
    K = np.array(m["K"]); c2w = np.load(MSR / m["episode"] / "cameras.npz")["c2w"][:4].astype(float)
    r = dict(task=task, T=T, objects={})
    for nm, col in OBJ[task].items():
        pos, nv, err = [], [], []
        for t in range(T):
            b = locate(K, c2w, [blobs(rgb[v, t], col) for v in range(V)])
            pos.append(None if b is None else b[1].tolist()); nv.append(0 if b is None else len(b[2])); err.append(None if b is None else b[3])
        # strict consistency: the largest blob in EVERY view that has one, jointly triangulated -> worst reprojection error
        full = []
        for t in range(T):
            obs_ = []
            for v in range(V):
                b = blobs(rgb[v, t], col)
                if b:
                    obs_.append((v, max(b, key=lambda x: x[1])[0]))
            if len(obs_) < 2:
                full.append(None); continue
            X = _tri([_ray(K[v], c2w[v], uv) for v, uv in obs_])
            es = []
            for v, uv in obs_:
                pc = np.linalg.inv(c2w[v]) @ np.r_[X, 1]
                es.append(float(np.linalg.norm((K[v] @ pc[:3])[:2] / pc[2] - uv)) if pc[2] > 0 else 99.0)
            full.append(max(es))
        r["objects"][nm] = dict(pos=pos, views=nv, err_px=err, full_err_px=full)
    o = r["objects"]
    def last(nm):
        ps = [p for p in o[nm]["pos"][-2:] if p is not None]
        return np.array(ps[-1]) if ps else None
    if task == "StackCube-v1":
        a, b = last("red"), last("green")
        r["success"] = bool(a is not None and b is not None and np.linalg.norm(a[:2] - b[:2]) < 0.02 and 0.025 < a[2] - b[2] < 0.06)
        r["detail"] = None if a is None or b is None else dict(dxy_cm=float(np.linalg.norm(a[:2] - b[:2]) * 100), dz_cm=float((a[2] - b[2]) * 100))
    elif task == "PullCubeTool-v1":
        p0 = next((np.array(p) for p in o["cube"]["pos"][:2] if p is not None), None); p1 = last("cube")
        r["success"] = bool(p0 is not None and p1 is not None and (p0[0] - p1[0]) > 0.05)
        r["detail"] = None if p0 is None or p1 is None else dict(pulled_cm=float((p0[0] - p1[0]) * 100))
    else:
        r["success"] = None
    for nm in o:                                   # the object never vanishes: >= 2 agreeing views in (almost) every frame
        o[nm]["present_frac"] = float(np.mean([v >= 2 for v in o[nm]["views"]]))
    fe = [e for ob in o.values() for e in ob["full_err_px"] if e is not None]
    r["full_err_med"] = float(np.median(fe)) if fe else 99.0
    r["present_min"] = min(o[nm]["present_frac"] for nm in o)
    allv = [v for ob in o.values() for v in ob["views"]]
    alle = [e for ob in o.values() for e in ob["err_px"] if e is not None]
    r["consistency"] = dict(frac_frames_3plus_views=float(np.mean([v >= 3 for v in allv])), mean_err_px=float(np.mean(alle)) if alle else None)
    out[d.name] = r
    print(d.name, r["success"], round(r["present_min"], 2), r.get("detail"), r["consistency"], flush=True)
json.dump(out, open(root / "scores.json", "w"), indent=1)
