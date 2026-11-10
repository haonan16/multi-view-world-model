"""Optimisation videos from results/real_v2/<ep>/optrec.pkl -> opt_robot.mp4, opt_object.mp4 (same dir).
python render_opt.py <episode>"""
import os, pickle, sys
from pathlib import Path
import cv2, imageio, numpy as np, torch
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))
import artirecon  # noqa
from artirecon.data import load_real
from artirecon.robot import URDFRobot

EP = sys.argv[1]; OUT = ROOT / "results/real_v2" / EP
D = pickle.load(open(OUT / "optrec.pkl", "rb"))
obs = load_real("v2/" + EP)
K, c2w = D["K"], D["c2w"]; T, V, H, W = obs.T, obs.V, obs.rgb.shape[2], obs.rgb.shape[3]
w2c = [np.linalg.inv(c) for c in c2w]
FONT = cv2.FONT_HERSHEY_DUPLEX
MAG = (255, 60, 220); GRN = (60, 255, 90)


def proj(P, v):
    Pc = P @ w2c[v][:3, :3].T + w2c[v][:3, 3]
    z = np.maximum(Pc[:, 2], 1e-4)
    return np.stack([K[v][0, 0] * Pc[:, 0] / z + K[v][0, 2], K[v][1, 1] * Pc[:, 1] / z + K[v][1, 2]], 1), Pc[:, 2]


def hue(i, n):
    c = cv2.cvtColor(np.uint8([[[int(180 * i / max(n, 1)), 200, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(x) for x in c)


def dots(img, uv, col, r=1, scale=1.0, off=(0, 0), alpha=1.0):
    lay = img.copy()
    for x, y in ((uv - off) * scale).astype(int):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(lay, (int(x), int(y)), r, col, -1, cv2.LINE_AA)
    return cv2.addWeighted(lay, alpha, img, 1 - alpha, 0) if alpha < 1 else lay


def curve(hist, best_hist, w, h, title, ymax=None):
    pan = np.full((h, w, 3), 245, np.uint8)
    cv2.putText(pan, title, (10, 22), FONT, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    if len(hist) < 2:
        return pan
    A = np.array(hist)                      # S,B
    lo, hi = np.nanmin(A), ymax or np.nanpercentile(A, 99)
    hi = max(hi, lo + 1e-6)
    S = len(A)
    def pt(i, y):
        return (int(10 + (w - 20) * i / max(S - 1, 1)), int(h - 10 - (h - 40) * (np.clip(y, lo, hi) - lo) / (hi - lo)))
    for b in range(A.shape[1]):
        col = (190, 190, 190) if A.shape[1] > 1 else MAG
        cv2.polylines(pan, [np.array([pt(i, A[i, b]) for i in range(S)])], False, col, 1, cv2.LINE_AA)
    cv2.polylines(pan, [np.array([pt(i, y) for i, y in enumerate(best_hist)])], False, MAG, 2, cv2.LINE_AA)
    return pan


# ------------------------------------------------------------------------------------------ robot
robot = URDFRobot(n_pts_per_m2=20000)
rr = D["robot"]
with torch.no_grad():
    qg = torch.tensor(D["gt_q"][:1], dtype=torch.float32, device="cuda")
    Pg, _ = robot.surface(qg, torch.eye(4, device="cuda")[None])
    sub = torch.randperm(Pg.shape[1], generator=torch.Generator().manual_seed(0))[:1800].cuda()
    snaps = []
    for r in rr:
        P, N = robot.surface(torch.tensor(r["q"], dtype=torch.float32, device="cuda"),
                             torch.tensor(r["base"], dtype=torch.float32, device="cuda"))
        err = ((P - Pg).norm(dim=-1).mean(1) * 1000).cpu().numpy()
        b_ = int(np.argmin(r["L"]))
        snaps.append((P[:, sub[:350]].cpu().numpy(), r["L"], err, r["it"], (P[b_].cpu().numpy(), N[b_].cpu().numpy())))
# 3 views where the robot is biggest
area = []
for v in range(V):
    uv, z = proj(snaps[-1][4][0], v)
    ok = (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H) & (z > 0)
    area.append(ok.sum())
rv = list(np.argsort(area)[::-1][:3])
sc = 1.0
wr = imageio.get_writer(OUT / "opt_robot.mp4", fps=12, codec="libx264", quality=8, macro_block_size=8)
hist, bh = [], []
order = None
from artirecon.splat import render_points
for k, (P, L, err, it, (Pb, Nb)) in enumerate(snaps + [snaps[-1]] * 18):
    if k < len(snaps):
        hist.append(L); bh.append(L.min())
    b = int(np.argmin(L)); B = len(L)
    tiles = []
    for v in rv:
        im = (obs.rgb[0, v].astype(np.float32) * 0.55).astype(np.uint8)
        for h_ in range(B):
            if h_ != b:
                im = dots(im, proj(P[h_], v)[0], hue(h_, B), r=1, alpha=0.5)
        rgb, A, _, _ = render_points([(Pb, Nb, (1.0, 0.25, 0.85))], K[v], w2c[v], H, W)
        rgb = rgb.cpu().numpy() if hasattr(rgb, "cpu") else rgb; A = A.cpu().numpy() if hasattr(A, "cpu") else A
        A = A.astype(np.float32)[..., None] * 0.75
        im = (im * (1 - A) + (np.clip(rgb, 0, 1) * 255) * A).astype(np.uint8)
        tiles.append(im)
    top = np.concatenate(tiles, 1)
    hdr = np.full((56, top.shape[1], 3), 255, np.uint8)
    cv2.putText(hdr, f"Robot fit from multiview depth: {B} joint hypotheses, iteration {it + 1}/300", (12, 24), FONT, 0.75, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(hdr, f"magenta = best hypothesis (others: coloured dots)   |   its error vs logged joints: {err[b]:.1f} mm  (logged joints never used)", (12, 48), FONT, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
    pan = curve(hist, bh, top.shape[1], 150, "loss of every hypothesis (grey), best (magenta)")
    wr.append_data(np.concatenate([hdr, top, pan], 0))
wr.close(); print("robot video", len(snaps))

if os.environ.get("ONLY_ROBOT"):
    sys.exit(0)
# ------------------------------------------------------------------------------------------ object
orr = D["objects"]
final = D["final"]
allP = np.concatenate([r["P"][int(np.argmin(r["L"]))].astype(np.float32) for r in orr[::20]])
cnt = []
for v in range(V):
    uv, z = proj(allP, v)
    cnt.append(((uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)).sum())
ov = list(np.argsort(cnt)[::-1][:3])
crops = {}
for v in ov:
    uv, _ = proj(allP, v)
    uv = uv[(uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)]
    x0, y0 = np.percentile(uv, 1, 0); x1, y1 = np.percentile(uv, 99, 0)
    s = max(x1 - x0, y1 - y0) * 1.5 + 40
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = min(s, H)
    crops[v] = (int(np.clip(cx - s / 2, 0, W - s)), int(np.clip(cy - s / 2, 0, H - s)), int(s))
TS = 400
wr = imageio.get_writer(OUT / "opt_object.mp4", fps=24, codec="libx264", quality=8, macro_block_size=8)
masks = {}
hist, bh, cur_call = [], [], None
nf = 0
for k, r in enumerate(orr):
    call = (r["stage"], r["t"], r["iters"])
    if r["masks"] is not None:
        masks = r["masks"]; hist, bh = [], []
    hist.append(r["L"]); bh.append(r["L"].min())
    tracking = r["stage"].startswith("track")
    if tracking and r["it"] % 9 != 0 and r["it"] != r["iters"] - 1:
        continue                              # tracking: fewer snapshots per frame
    t = r["t"]; B = len(r["L"]); b = int(np.argmin(r["L"]))
    tiles = []
    for v in ov:
        x0, y0, s = crops[v]
        im = obs.rgb[t, v][y0:y0 + s, x0:x0 + s]
        im = cv2.resize(im, (TS, TS), interpolation=cv2.INTER_CUBIC)
        im = (im.astype(np.float32) * 0.8).astype(np.uint8)
        m = masks.get((t, v))
        if m is not None:
            mc = cv2.resize(m[y0:y0 + s, x0:x0 + s].astype(np.uint8), (TS, TS), interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, GRN, 2, cv2.LINE_AA)
        for h_ in range(B):
            P = r["P"][h_].astype(np.float32)
            im = dots(im, proj(P, v)[0], MAG if h_ == b else hue(h_, B + 1), r=2 if h_ == b else 1,
                      scale=TS / s, off=(x0, y0), alpha=1.0 if h_ == b else 0.6)
        if m is None:
            cv2.putText(im, "no mask (hidden)", (10, TS - 14), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    top = np.concatenate(tiles, 1)
    hdr = np.full((60, top.shape[1], 3), 255, np.uint8)
    st = r["stage"]
    desc = {"shape+pose": "shape (per-axis scale) + 6-DoF pose", "refine": "pose refinement at the fitted shape",
            "track": "per-frame pose, initialised from the previous frame"}
    key = next(kk for kk in desc if st.startswith(kk))
    asset = " SAM 3D mesh" if "[mesh]" in st else (" 2.5D outline (hull)" if "[hull]" in st else "")
    cv2.putText(hdr, f"{D['obj']}: {desc[key]}{' -' + asset if asset else ''}" + ("  [in gripper]" if "held" in st else ""),
                (12, 25), FONT, 0.68, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(hdr, f"frame {t}   iteration {r['it'] + 1}/{r['iters']}   hypotheses {B}   loss {r['L'][b]:.3f}"
                f"     green: SAM 3 mask   magenta: model", (12, 50), FONT, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
    pan = curve(hist, bh, top.shape[1], 110, "loss over this fit (silhouette + depth + support)")
    fr = np.concatenate([hdr, top, pan], 0)
    reps = 1
    if r["it"] == r["iters"] - 1 and not tracking:
        reps = 18                              # hold the result of each shape fit
    for _ in range(reps):
        wr.append_data(fr); nf += 1
wr.close(); print("object video frames", nf)
