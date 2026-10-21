import json, sys, numpy as np
from pathlib import Path
from PIL import Image
CLOUDS = Path("/n/netscratch/ydu_lab/Lab/akiruga/mv_infer/v1_heldout15/clouds_native")
def table_normal(scene, src, t, centre, cam_pos, rng=np.random.default_rng(0)):
    d = CLOUDS / scene / src; meta = json.loads((d / "meta.json").read_text())
    V, h, w = meta["V"], meta["H"], meta["W"]; a = np.array(Image.open(d / "dc.png"))
    u, v = np.meshgrid(np.arange(w) + .5, np.arange(h) + .5); pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)
    P = []
    for i in range(V):
        blk = slice((t * V + i) * h, (t * V + i + 1) * h); m = meta["frames"][t][i]
        q = (a[blk, :, 0].astype(float) * 256 + a[blk, :, 1]) / 65535; dd = (q * (m["dmax"] - m["dmin"]) + m["dmin"]).reshape(-1)
        K, M = np.array(m["K"]), np.array(m["M"]); Xc = (pix @ np.linalg.inv(K).T) * dd[:, None]
        P.append(Xc @ M[:, :3].T + M[:, 3])
    P = np.concatenate(P); P = P[np.linalg.norm(P - centre, axis=1) < 0.45]
    best, bn, bd = 0, None, None
    for _ in range(2000):
        s = P[rng.choice(len(P), 3, replace=False)]; n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9: continue
        n /= np.linalg.norm(n); dist = np.abs((P - s[0]) @ n); c = (dist < 0.01).sum()
        if c > best: best, bn, bd = c, n, s[0] @ n
    inl = P[np.abs(P @ bn - bd) < 0.01]; c0 = inl.mean(0); _, _, Vt = np.linalg.svd(inl - c0, full_matrices=False); n = Vt[2]
    if (cam_pos.mean(0) - c0) @ n < 0: n = -n
    return n, c0, best / len(P)
if __name__ == "__main__":
    for c in sorted(Path("/tmp/akiruga_lightstage/cases").iterdir()):
        rig = json.loads((c / "rig.json").read_text())
        cp = np.array(rig["inputs"]["c2w"])[:, :3, 3]
        n, c0, frac = table_normal(rig["scene"], rig["source"], rig["t"], np.array(rig["centre"]), cp)
        up = np.array(rig["up"]); print(c.name, "table normal", n.round(3), "inlier frac", round(frac, 2),
              "angle to camera-up", round(float(np.degrees(np.arccos(np.clip(n @ up, -1, 1)))), 1), "deg; table centre", c0.round(3), "axis centre", np.array(rig["centre"]).round(3))
