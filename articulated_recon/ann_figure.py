"""Clear annotation figure for the deck: what a person provides per episode (one click + a few words) and what it
selects. python ann_figure.py <episode> <out.jpg>"""
import json, os, pickle, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))
EP, OUTF = sys.argv[1], sys.argv[2]
d = ROOT / "real/v2" / EP
z = np.load(d / "obs.npz"); rgb = z["rgb"]
ann = json.load(open(d / "ann.json"))
s3 = np.load(d / "sam3.npz"); H, W = [int(x) for x in s3["shape"]]; prompts = list(s3["prompts"])
fit = pickle.load(open(ROOT / "results/real_v2" / EP / "fit.pkl", "rb"))
R = pickle.load(open(ROOT / "results/real_v2" / EP / "robot.pkl", "rb"))
K = z["K"]; w2c = [np.linalg.inv(c) for c in R["c2w"]]
o = next(x for x in ann["objects"] if x.get("role") == "moved")
fo = next(x for x in fit["objects"] if x["name"] == o["name"])
t, v, (u, vv) = o["hint"]["t"], o["hint"]["v"], o["hint"]["uv"]
pl = o["prompt"] if isinstance(o["prompt"], list) else [o["prompt"]]
ids = np.nonzero(np.isin(s3["p"], [prompts.index(x) for x in pl if x in prompts]) & (s3["t"] == t) & (s3["v"] == v))[0]
allm, best = [], None
for i in ids:
    m = np.unpackbits(s3["mask"][i])[:H * W].reshape(H, W).astype(bool); allm.append(m)
    if m[int(vv), int(u)] and (best is None or m.sum() < best.sum()):
        best = m
T = rgb.shape[0]
def ctr(tt):
    c = fo["mesh"][0].mean(0); P = fo["poses"][tt]; X = P[:3, :3] @ c + P[:3, 3]
    Xc = w2c[v][:3, :3] @ X + w2c[v][:3, 3]
    return int(K[v][0, 0] * Xc[0] / Xc[2] + K[v][0, 2]), int(K[v][1, 1] * Xc[1] / Xc[2] + K[v][1, 2])
S = 2  # upscale for crisp drawing
def up(im): return cv2.resize(im, (W * S, H * S), interpolation=cv2.INTER_CUBIC)
F = cv2.FONT_HERSHEY_DUPLEX
# left: first frame, click + all SAM 3 candidates (thin) + chosen one (filled)
a = up(rgb[t, v]).copy()
for m in allm:
    cs, _ = cv2.findContours(cv2.resize(m.astype(np.uint8), (W * S, H * S), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(a, cs, -1, (255, 255, 255), 1, cv2.LINE_AA)
mb = cv2.resize(best.astype(np.uint8), (W * S, H * S), interpolation=cv2.INTER_NEAREST).astype(bool)
a[mb] = (0.45 * a[mb] + 0.55 * np.array([60, 255, 90])).astype(np.uint8)
cs, _ = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(a, cs, -1, (60, 255, 90), 3, cv2.LINE_AA)
cx, cy = int(u * S), int(vv * S)
cv2.circle(a, (cx, cy), 14, (255, 40, 40), 4, cv2.LINE_AA); cv2.circle(a, (cx, cy), 4, (255, 40, 40), -1, cv2.LINE_AA)
# right: last frame, where it went (arrow from start to end)
b = up(rgb[T - 1, v]).copy()
p0, p1 = ctr(0), ctr(T - 1)
cv2.circle(b, (p0[0] * S, p0[1] * S), 10, (255, 255, 255), 3, cv2.LINE_AA)
cv2.arrowedLine(b, (p0[0] * S, p0[1] * S), (p1[0] * S, p1[1] * S), (255, 220, 40), 6, cv2.LINE_AA, tipLength=0.12)
def label(im, txt):
    cv2.rectangle(im, (0, 0), (im.shape[1], 54), (255, 255, 255), -1)
    cv2.putText(im, txt, (16, 38), F, 1.1, (30, 30, 30), 2, cv2.LINE_AA)
label(a, f"first frame: one click (red) -> SAM 3 instance (green)")
label(b, f"last frame: where it went (start -> end)")
gap = np.full((H * S, 24, 3), 255, np.uint8)
img = np.concatenate([a, gap, b], 1)
cv2.imwrite(OUTF, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
print(OUTF, img.shape, o["name"], pl, o.get("shape"))
