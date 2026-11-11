"""Optimisation videos in 3D (same style as the ManiSkill deck: orbiting virtual camera, fused depth points,
every hypothesis as a coloured point cloud, best one as a shaded surface at the end).
Reads results/real_v2/<ep>/optrec.pkl (opt_record.py) -> opt3d_robot.mp4, opt3d_object.mp4.  CPU or GPU.
python render_opt3d.py <episode>"""
import os, pickle, sys
from pathlib import Path
import cv2, imageio.v3 as iio, numpy as np, torch
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))
import artirecon  # noqa
from artirecon.data import backproject
from artirecon.robot import URDFRobot

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(16)
EP = sys.argv[1]
OUT = ROOT / "results/wm" / EP                      # wm fit dir name (e.g. StackCube-v1_episode-0982_s7_da3_rec)
D = pickle.load(open(OUT / "optrec.pkl", "rb"))
class _O: pass
obs = _O(); obs.rgb = D["rgb"]; obs.K = D["K"]; obs.T, obs.V = obs.rgb.shape[:2]
depth = D["depth"].astype(np.float32)
c2w = D["c2w"]; n_up = np.array([0., 0., 1.]); d_pl = 0.0
if (c2w[:, :3, 3].mean(0) @ n_up + d_pl) < 0:          # normal must point towards the cameras (up)
    n_up, d_pl = -n_up, -d_pl
H, W = 720, 1280
KV = np.array([[1000., 0, W / 2], [0, 1000., H / 2], [0, 0, 1]])
BG = np.array([30, 30, 34], np.float32)
FONT = cv2.FONT_HERSHEY_DUPLEX
e1 = np.cross(n_up, [1, 0, 0] if abs(n_up[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n_up, e1)


def lookat(eye, c):
    f = c - eye; f /= np.linalg.norm(f)
    r = np.cross(f, n_up); r /= np.linalg.norm(r); u = np.cross(f, r)
    Rm = np.stack([r, u, f]); w2c = np.eye(4); w2c[:3, :3] = Rm; w2c[:3, 3] = -Rm @ eye
    return w2c


cam_mean = c2w[:, :3, 3].mean(0)


def orbit(k, n, c, radius, height, sweep=1.4):
    d = cam_mean - c; a0 = np.arctan2(d @ e2, d @ e1)
    a = a0 + sweep * (k / max(1, n - 1) - 0.5)
    return lookat(c + radius * (np.cos(a) * e1 + np.sin(a) * e2) + height * n_up, c)


def splat(layers, w2c, r=2):
    """layers: list of (P [N,3], C [N,3] uint8 or (3,)) -> image, z-buffered"""
    P = np.concatenate([l[0] for l in layers]).astype(np.float32)
    C = np.concatenate([np.broadcast_to(np.asarray(l[1], np.float32), (len(l[0]), 3)) for l in layers])
    P = torch.as_tensor(P, device=dev); C = torch.as_tensor(np.ascontiguousarray(C), device=dev)
    w = torch.as_tensor(w2c, dtype=torch.float32, device=dev)
    Pc = P @ w[:3, :3].T + w[:3, 3]; z = Pc[:, 2]
    ok = z > 0.05
    u = (Pc[:, 0] / z.clamp(min=1e-3) * KV[0, 0] + KV[0, 2]).floor().long()
    v = (Pc[:, 1] / z.clamp(min=1e-3) * KV[1, 1] + KV[1, 2]).floor().long()
    zb = torch.full((H * W,), float("inf"), device=dev); img = torch.as_tensor(BG, device=dev).repeat(H * W, 1)
    sets = []
    for du in range(r):
        for dv in range(r):
            uu, vv = u + du, v + dv
            m = ok & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            fl = (vv * W + uu).clamp(0, H * W - 1)
            zb.scatter_reduce_(0, fl[m], z[m], reduce="amin"); sets.append((fl, m))
    for fl, m in sets:
        win = m & (z <= zb[fl] + 1e-6)
        img[fl[win]] = C[win]
    if os.environ.get("_MASK"):
        return img.reshape(H, W, 3).clamp(0, 255).byte().cpu().numpy(), torch.isfinite(zb).reshape(H, W).cpu().numpy()
    return img.reshape(H, W, 3).clamp(0, 255).byte().cpu().numpy()


def fused(t, ctr=None, rad=None, stride=2):
    P, C = [], []
    for v in range(obs.V):
        p, pix = backproject(depth[t, v], obs.K[v], c2w[v], stride)
        P.append(p); C.append(obs.rgb[t, v][pix[:, 0], pix[:, 1]])
    P, C = np.concatenate(P), np.concatenate(C)
    h = P @ n_up + d_pl
    keep = (h > -0.03) & (h < 1.2)
    if ctr is not None:
        keep &= np.linalg.norm(P - ctr, axis=1) < rad
    return P[keep], C[keep]


def shade(P, Nm, w2c, col):
    Pc = P @ w2c[:3, :3].T + w2c[:3, 3]; vd = -Pc / np.linalg.norm(Pc, axis=1, keepdims=True)
    s = 0.35 + 0.65 * np.abs(((Nm @ w2c[:3, :3].T) * vd).sum(1))
    return (np.asarray(col)[None] * s[:, None]).astype(np.float32)


def header(img, l1, l2):
    cv2.rectangle(img, (0, 0), (W, 70), (255, 255, 255), -1)
    cv2.putText(img, l1, (18, 30), FONT, 0.85, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(img, l2, (18, 58), FONT, 0.65, (90, 90, 90), 1, cv2.LINE_AA)
    return img


def curve_inset(img, hist, x0=W - 360, y0=H - 170, w=340, h=150):
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (245, 245, 245), -1)
    cv2.putText(img, "loss (grey: each hypothesis)", (x0 + 8, y0 + 18), FONT, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    if len(hist) < 2:
        return img
    A = np.array(hist); lo, hi = A.min(), np.percentile(A, 99); hi = max(hi, lo + 1e-6)
    pt = lambda i, y: (int(x0 + 8 + (w - 16) * i / (len(A) - 1)), int(y0 + h - 8 - (h - 34) * (np.clip(y, lo, hi) - lo) / (hi - lo)))
    for b in range(A.shape[1]):
        cv2.polylines(img, [np.array([pt(i, A[i, b]) for i in range(len(A))])], False, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.polylines(img, [np.array([pt(i, y) for i, y in enumerate(A.min(1))])], False, (255, 60, 220), 2, cv2.LINE_AA)
    return img


def write(name, frames, fps):
    iio.imwrite(OUT / f"{name}.mp4", np.stack(frames), fps=fps, codec="libx264", macro_block_size=8,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "20"])
    iio.imwrite(OUT / f"{name}.jpg", frames[int(len(frames) * 0.7)], quality=90)
    print("wrote", name, len(frames), flush=True)


rng = np.random.default_rng(1)
# ------------------------------------------------------------------ robot
if os.environ.get("PART", "all") in ("all", "robot"):
    robot = URDFRobot(n_pts_per_m2=20000, device=dev)
    rr = D["robot"]; B = len(rr[0]["L"])
    pal = (rng.uniform(0.3, 0.95, (B, 3)) * 255).astype(np.float32)
    qg = torch.tensor(D["gt_q"][:1], dtype=torch.float32, device=dev)
    with torch.no_grad():
        Pg, _ = robot.surface(qg, torch.eye(4, device=dev)[None])
    P0, C0 = fused(0)
    Cb = (0.45 * C0 + 0.55 * 60).astype(np.float32)
    ctr = Pg[0].cpu().numpy().mean(0); ctr = ctr - ((ctr @ n_up + d_pl) - 0.25) * n_up
    frames, hist = [], []
    n = len(rr) + 40
    for k, r in enumerate(rr):
        with torch.no_grad():
            S, Ns = robot.surface(torch.tensor(r["q"], dtype=torch.float32, device=dev), torch.tensor(r["base"], dtype=torch.float32, device=dev))
            err = ((S - Pg).norm(dim=-1).mean(1) * 1000).cpu().numpy()
        S = S.cpu().numpy(); b = int(np.argmin(r["L"])); hist.append(r["L"])
        sub = np.arange(0, S.shape[1], 4)
        lay = [(P0, Cb)] + [(S[i, sub], pal[i]) for i in range(B)]
        w2c = orbit(k, n, ctr, 1.6, 0.9)
        img = splat(lay, w2c)
        header(img, f"Robot on a predicted rollout: {B} starting guesses, Adam step {r['it'] + 1}/300",
               f"each colour = one guess, pulled onto Depth Anything 3 points of the predicted views (frame 0)")
        frames.append(curve_inset(img, hist))
    Sb, Nb = S[b], Ns[b].cpu().numpy()
    for k in range(40):
        w2c = orbit(len(rr) + k, n, ctr, 1.6, 0.9)
        img = splat([(P0, Cb), (Sb, shade(Sb, Nb, w2c, (140, 175, 245)))], w2c)
        header(img, "Lowest-loss guess kept", "then: every frame from the previous one, joint refinement, silhouette refinement on SAM 3 masks")
        frames.append(img)
    write("opt3d_robot", frames, 12)

# ------------------------------------------------------------------ object
if os.environ.get("PART", "all") in ("all", "object"):
    orr = D["objects"]
    ctr = np.mean([r["P"][int(np.argmin(r["L"]))].astype(np.float32).mean(0) for r in orr[::25]], 0)
    rad = 0.16
    cache = {}
    frames, hist = [], []
    keep = [k for k, r in enumerate(orr) if not r["stage"].startswith("track") or r["it"] % 15 == 0 or r["it"] == r["iters"] - 1]
    n = len(keep)
    desc = {"shape+pose": "fit size (3 scales) + pose", "refine": "refine pose at that size", "track": "next frame: start from the previous pose"}
    for j, k in enumerate(keep):
        r = orr[k]
        if r["masks"] is not None or (k and orr[k - 1]["it"] > r["it"]):
            hist = []
        hist.append(r["L"]) if not hist or len(hist[-1]) == len(r["L"]) else None
        t = r["t"]
        if t not in cache:
            cache[t] = fused(t, ctr, rad, stride=1)
        Pf, Cf = cache[t]
        B = len(r["L"]); b = int(np.argmin(r["L"]))
        pal = [(255, 60, 220)] + [(90, 220, 255), (255, 200, 60), (120, 255, 120), (255, 120, 90)]
        g = Cf.astype(np.float32).mean(1, keepdims=True)
        cloud = [(Pf, (0.35 * Cf + 0.65 * g) * 0.75)]          # desaturated depth points
        model = []
        for i in range(B):
            if i != b:
                model.append((r["P"][i].astype(np.float32), pal[1 + i % 4]))
        model.append((r["P"][b].astype(np.float32), pal[0]))
        w2c = orbit(j, n, ctr, 0.42, 0.3, sweep=1.6)
        img = splat(cloud, w2c, r=3)
        os.environ["_MASK"] = "1"; mi, mm = splat(model, w2c, r=4); del os.environ["_MASK"]
        img[mm] = (0.15 * img[mm] + 0.85 * mi[mm]).astype(np.uint8)   # model drawn on top
        st = r["stage"]; key = next(kk for kk in desc if st.startswith(kk))
        asset = "SAM 3D mesh" if "[mesh]" in st else ("2.5D outline" if "[hull]" in st else "")
        header(img, f"{D['obj']}: {desc[key]}" + (f"  ({asset})" if asset else "") + ("  - in the gripper" if "held" in st else ""),
               f"frame {t}  Adam step {r['it'] + 1}/{r['iters']}  {B} guess{'es' if B > 1 else ''}   magenta: model (best guess)   grey: depth points from all cameras")
        img = curve_inset(img, hist)
        reps = 14 if (r["it"] == r["iters"] - 1 and not st.startswith("track")) else 1
        frames += [img] * reps
    write("opt3d_object", frames, 20)
