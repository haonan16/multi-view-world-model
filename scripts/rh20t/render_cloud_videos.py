"""Render presentation videos of the dynamic point clouds (scripts/rh20t/vggt_dynamic_clouds.py).

Layout (1920x1080): a left column with 5 panels -- the input (frame 0 of the V views, the
model's conditioning) and the V generated (or GT) view videos -- and, on the right ~77%,
the dynamic point cloud seen from a slowly orbiting isometric camera, with the V camera
frustums drawn in the colours of their panels.

Cleaning (per frame, all views fused) removes only provably wrong points so thin, moving
structure such as the robot arm survives: see load_cloud.

Writes <out>/<scene>/GT/<scene>_gt.mp4 and <out>/<scene>/Pred/<scene>_{tokens,tiled}.mp4.

Usage: python scripts/rh20t/render_cloud_videos.py --root INFER_OUT --out DIR
           [--scenes 0,3] [--preview]   (preview: write a few PNG frames instead)
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

W_OUT, H_OUT = 1920, 1080
COL_W = 440                                    # left column (~23%)
FPS = 30
HOLD = 12                                      # output frames per timestep (13 steps -> 5.2 s)
LOOPS = 2
CONF_KEEP = 1.0                                # no confidence cut: VGGT-Omega is least confident on the arm
VOX = 0.01                                     # m
VOX_MIN = 12                                   # min points in the 3x3x3 voxel neighbourhood

MAX_FOOT = 0.010                               # drop far points whose pixel covers > 10 mm
GRAZE_COS = 0.2                                # drop surfaces seen at > ~78 deg (sparse striped rows)
RAMP_TOL = 0.012                               # relative depth step per pixel that marks a smear
FS_TOL = 0.08                                  # free-space test margin (relative depth)
WS_RADIUS = 0.8                                # workspace radius / median camera distance
CROP = 1.0                                     # keep points within CROP x workspace radius
BG = (255, 255, 255)
VIEW_COLORS = [(230, 85, 50), (40, 140, 220), (40, 170, 90), (200, 150, 20)]
SOURCES = {"gt": ("GT", "gt", "GROUND-TRUTH FRAMES"),
           "pred_tokens": ("Pred", "tokens", "PREDICTED · VIEWS AS TOKENS"),
           "pred_tiled": ("Pred", "tiled", "PREDICTED · VIEWS TILED")}


def font(size):
    for f in ("/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def load_cloud(d, frames_vt):
    """-> per timestep: (pts [N,3], rgb [N,3] uint8, footprint [N]); meta.

    Removes only points that are provably wrong, so thin / moving structure (the robot
    arm, which VGGT-Omega is least confident about) survives:
      - flying pixels (depth between the two sides of a discontinuity; flag from
        vggt_dynamic_clouds.py -- outlines of real objects are kept)
      - free-space violations: points that every other view seeing that spot looks
        straight through (its surface is clearly behind the point), with >= 2 such views
      - far points with a very coarse pixel footprint, and isolated points
    No confidence cut: VGGT-Omega's confidence is lowest exactly on the moving arm.
    """
    meta = json.loads((d / "meta.json").read_text())
    V, T, H, W = meta["V"], meta["T"], meta["H"], meta["W"]
    a = np.array(Image.open(d / "dc.png"))
    col = np.array(Image.open(d / "color.jpg"))                # same atlas layout as dc.png
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)
    out = []
    for t in range(T):
        depth, Ks, Ms, Xs, base = [], [], [], [], []
        for vi in range(V):
            blk = a[(t * V + vi) * H:(t * V + vi + 1) * H]
            m = meta["frames"][t][vi]
            q = (blk[..., 0].astype(np.float64) * 256 + blk[..., 1]) / 65535
            d_ = (q * (m["dmax"] - m["dmin"]) + m["dmin"]).reshape(-1)
            b = blk[..., 2].reshape(-1)
            rank, edge = (b >> 2) / 63.0, b & 1
            K, M = np.array(m["K"]), np.array(m["M"])
            Xall = (pix @ np.linalg.inv(K).T) * d_[:, None]
            foot = d_ / K[0, 0]                               # pixel footprint (m)
            cos = view_cos(Xall.reshape(H, W, 3)).reshape(-1)
            base.append(((rank >= 1 - CONF_KEEP) & (edge == 1) & (d_ > 1e-4)
                         & (foot < MAX_FOOT) & ~on_ramp(d_.reshape(H, W)).reshape(-1)
                         & (cos > GRAZE_COS),
                         foot / np.maximum(cos, 0.35)))           # splat = footprint on the surface
            depth.append(d_.reshape(H, W)); Ks.append(K); Ms.append(M)
            Xs.append(Xall @ M[:, :3].T + M[:, 3])
        P, C, F = [], [], []
        for vi in range(V):
            keep, foot = base[vi]
            keep = keep & ~free_space_violation(Xs[vi], vi, depth, Ks, Ms)
            c = col[(t * V + vi) * H:(t * V + vi + 1) * H].reshape(-1, 3)
            P.append(Xs[vi][keep]); C.append(c[keep]); F.append(foot[keep])
        out.append(density_filter(np.concatenate(P), np.concatenate(C), np.concatenate(F)))
    return out, meta


def view_cos(X, k=3):
    """|cos| between viewing ray and surface normal, normals from +-k pixel differences
    (a wide baseline keeps them stable on small objects such as the arm). X: [H,W,3]."""
    Xp = np.pad(X, ((k, k), (k, k), (0, 0)), mode="edge")
    dx = Xp[k:-k, 2 * k:] - Xp[k:-k, :-2 * k]
    dy = Xp[2 * k:, k:-k] - Xp[:-2 * k, k:-k]
    n = np.cross(dx, dy)
    return np.abs((n * X).sum(-1)) / (np.linalg.norm(n, axis=-1) * np.linalg.norm(X, axis=-1) + 1e-12)


def on_ramp(d):
    """Depth smears: VGGT-Omega sees upsampled frames, so a discontinuity becomes a ramp
    several pixels wide whose pixels float between foreground and background. A pixel is
    on a ramp if depth changes steeply on BOTH sides of it (horizontally or vertically);
    a real silhouette pixel, e.g. on the robot arm, is smooth on its own side and kept."""
    g = lambda a, b: np.abs(a - b) / np.minimum(a, b)
    out = np.zeros(d.shape, bool)
    for ax in (0, 1):
        fwd = g(np.roll(d, -1, ax), d)
        bwd = g(d, np.roll(d, 1, ax))
        out |= np.minimum(fwd, bwd) > RAMP_TOL
    return out


def free_space_violation(X, v, depth, Ks, Ms):
    """True where every other view whose image contains the point sees a surface clearly
    BEHIND it (it would have to see the point -- it is a floater), with >= 2 such views."""
    H, W = depth[0].shape
    seen = np.zeros(len(X), int)
    through = np.zeros(len(X), int)
    for u in range(len(depth)):
        if u == v:
            continue
        R, c = Ms[u][:, :3], Ms[u][:, 3]
        Xc = (X - c) @ R
        z = Xc[:, 2]
        uv = Xc @ Ks[u].T
        zs = np.maximum(z, 1e-6)
        ix, iy = np.floor(uv[:, 0] / zs).astype(int), np.floor(uv[:, 1] / zs).astype(int)
        inside = (z > 1e-3) & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        zu = np.zeros_like(z)
        zu[inside] = depth[u][iy[inside], ix[inside]]
        seen += inside
        through += inside & (zu > z * (1 + FS_TOL))
    return (through >= 2) & (through == seen)


def density_filter(P, C, I):
    """Drop points whose 3x3x3 voxel neighbourhood holds fewer than VOX_MIN points."""
    if len(P) == 0:
        return P, C, I
    k = np.floor(P / VOX).astype(np.int64)
    k -= k.min(0)
    dims = k.max(0) + 3
    key = lambda kk: ((kk[:, 0] + 1) * dims[1] + kk[:, 1] + 1) * dims[2] + kk[:, 2] + 1
    uniq, inv, cnt = np.unique(key(k), return_inverse=True, return_counts=True)
    total = np.zeros(len(P), np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                kk = key(k + np.array([dx, dy, dz]))
                pos = np.searchsorted(uniq, kk)
                pos = np.clip(pos, 0, len(uniq) - 1)
                total += np.where(uniq[pos] == kk, cnt[pos], 0)
    keep = total >= VOX_MIN
    return P[keep], C[keep], I[keep]


def scene_frame(meta, clouds):
    """Presentation frame. up = the real cameras' mean -y axis; the workspace centre is the
    point closest (least squares) to all cameras' optical axes -- every RH20T camera is
    aimed at the workspace -- and its radius is WS_RADIUS x the median camera distance.
    Points outside it (the room, people, floor) are cropped."""
    Ms = np.array([[np.array(v["M"]) for v in fr] for fr in meta["frames"]])  # T,V,3,4
    R = Ms[..., :3].reshape(-1, 3, 3)
    up = -R[:, :, 1].mean(0); up /= np.linalg.norm(up)
    fwd = R[:, :, 2].mean(0); fwd -= up * (fwd @ up); fwd /= np.linalg.norm(fwd)
    o, d = Ms[0, :, :, 3], Ms[0, :, :, 2]                        # frame-0 centres, axes
    A = sum(np.eye(3) - np.outer(di, di) for di in d)
    b = sum((np.eye(3) - np.outer(di, di)) @ oi for oi, di in zip(o, d))
    centre = np.linalg.solve(A, b)
    radius = WS_RADIUS * np.median(np.linalg.norm(o - centre, axis=1))
    allp = np.concatenate([c[0] for c in clouds])
    fit = allp[np.linalg.norm(allp - centre, axis=1) < CROP * radius][::7]
    return up, fwd, centre, radius, Ms, fit


def look_at(eye, target, up):
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])                         # rows: cam axes in world (OpenCV)


def orbit_cam(up, fwd, centre, radius, phase, W, H, fit=None):
    """Isometric-ish camera: from the front-left, 30 deg above, orbiting +-18 deg.
    fit: points used to choose the focal length so the cloud fills ~88% of the frame."""
    yaw = np.deg2rad(-35 + 18 * np.sin(2 * np.pi * phase))
    elev = np.deg2rad(28)
    side = np.cross(up, fwd)
    dir_h = np.cos(yaw) * (-fwd) + np.sin(yaw) * side
    d = np.cos(elev) * dir_h + np.sin(elev) * up
    eye = centre + d * radius * 2.4
    Rcw = look_at(eye, centre, up)
    f = 0.95 * W
    if fit is not None:
        c = (fit - eye) @ Rcw.T
        c = c[c[:, 2] > 0.05]
        ax = np.percentile(np.abs(c[:, 0] / c[:, 2]), 99)
        ay = np.percentile(np.abs(c[:, 1] / c[:, 2]), 99)
        f = 0.88 * min(W / (2 * ax), H / (2 * ay))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    return Rcw, eye, K


def splat(P, C, S, Rcw, eye, K, W, H, dev):
    """Z-buffered point splatting; each disk covers its source pixel's footprint S (m)."""
    P = torch.from_numpy(P).to(dev, torch.float32)
    C = torch.from_numpy(C).to(dev)
    S = torch.from_numpy(S).to(dev, torch.float32)
    Xc = (P - torch.from_numpy(eye).to(dev).float()) @ torch.from_numpy(Rcw).to(dev).float().T
    z = Xc[:, 2]
    ok = z > 0.05
    Xc, C, z, S = Xc[ok], C[ok], z[ok], S[ok]
    Kt = torch.from_numpy(K).to(dev).float()
    uv = (Xc @ Kt.T)[:, :2] / z[:, None]
    r = (0.75 * S * Kt[0, 0] / z).clamp(1.0, 4.5)
    img = torch.full((H * W,), float("inf"), device=dev)
    best = torch.full((H * W,), -1, dtype=torch.long, device=dev)
    idx = torch.arange(len(z), device=dev)
    R = 5
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            m = (dx * dx + dy * dy) <= r * r
            x = (uv[m, 0] + dx).long(); y = (uv[m, 1] + dy).long()
            inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            pid = (y * W + x)[inb]
            zz = z[m][inb]
            img.scatter_reduce_(0, pid, zz, "amin")
            win = zz <= img[pid]
            best[pid[win]] = idx[m][inb][win]
    out = torch.tensor(BG, device=dev, dtype=torch.uint8).repeat(H * W, 1)
    hit = best >= 0
    out[hit] = C[best[hit]]
    return out.view(H, W, 3).cpu().numpy()


def draw_frustums(im, Ms_t, Kv, Rcw, eye, K, scale=0.12):
    """Real-camera frustums (from VGGT poses aligned to the RH20T calibration)."""
    dr = ImageDraw.Draw(im)
    def proj(X):
        c = Rcw @ (X - eye)
        if c[2] <= 0.05:
            return None
        p = K @ c
        return (p[0] / p[2], p[1] / p[2])
    for vi, M in enumerate(Ms_t):
        Kinv = np.linalg.inv(np.array(Kv[vi]))
        w, h = 2 * Kv[vi][0][2], 2 * Kv[vi][1][2]
        o = M[:, 3]
        corners = [M[:, :3] @ (Kinv @ np.array([x, y, 1.0])) * scale + o
                   for x, y in ((0, 0), (w, 0), (w, h), (0, h))]
        pts = [proj(o)] + [proj(c) for c in corners]
        if any(p is None for p in pts):
            continue
        col = VIEW_COLORS[vi % 4]
        for c in pts[1:]:
            dr.line([pts[0], c], fill=col, width=3)
        dr.line(pts[1:] + [pts[1]], fill=col, width=3)


def panel(img, w, h):
    return np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def compose(t, phase, frames_vt, cloud, meta, geom, label, dev):
    V = frames_vt.shape[0]
    canvas = Image.new("RGB", (W_OUT, H_OUT), BG)
    dr = ImageDraw.Draw(canvas)
    fsmall, fbig = font(17), font(24)
    pw, ph, gap, top = 400, 200, 12, 14
    x0 = (COL_W - pw) // 2
    # input panel: frame 0 of all views as a 2x2
    grid = np.concatenate([np.concatenate([frames_vt[0, 0], frames_vt[1, 0]], 1),
                           np.concatenate([frames_vt[2, 0], frames_vt[3, 0]], 1)], 0)
    slots = [("INPUT · FRAME 0, 4 VIEWS", grid, (60, 60, 60))] + \
            [(f"VIEW {v + 1}", frames_vt[v, t], VIEW_COLORS[v]) for v in range(V)]
    y = top
    for name, im, col in slots:
        canvas.paste(Image.fromarray(panel(im, pw, ph)), (x0, y))
        dr.rectangle([x0 - 3, y - 3, x0 + pw + 2, y + ph + 2], outline=col, width=4)
        tw = dr.textlength(name, font=fsmall)
        dr.rectangle([x0, y + ph - 26, x0 + tw + 14, y + ph], fill=(255, 255, 255))
        dr.text((x0 + 7, y + ph - 24), name, fill=(20, 20, 20), font=fsmall)
        y += ph + gap
    # point cloud
    up, fwd, centre, radius, Ms, fit = geom
    RW, RH = W_OUT - COL_W, H_OUT
    Rcw, eye, K = orbit_cam(up, fwd, centre, radius, phase, RW, RH, fit)
    P, C, S = cloud[t]
    inside = np.linalg.norm(P - centre, axis=1) < CROP * radius
    P, C, S = P[inside], C[inside], S[inside]
    pc = Image.fromarray(splat(P, C, S, Rcw, eye, K, RW, RH, dev))
    draw_frustums(pc, Ms[t], [v["K"] for v in meta["frames"][t]], Rcw, eye, K,
                  scale=0.035 * radius / 0.5)
    canvas.paste(pc, (COL_W, 0))
    tw = dr.textlength(label, font=fbig)
    dr.rounded_rectangle([COL_W + 30, 26, COL_W + 30 + tw + 36, 72], radius=23,
                         fill=(255, 255, 255), outline=(30, 30, 30), width=2)
    dr.text((COL_W + 48, 36), label, fill=(20, 20, 20), font=fbig)
    tl = f"t = {t} / {len(cloud) - 1}" + ("   (input)" if t == 0 else "")
    dr.rectangle([COL_W + 24, H_OUT - 54, COL_W + 700, H_OUT - 6], fill=(255, 255, 255))
    dr.text((COL_W + 32, H_OUT - 48), tl, fill=(70, 70, 70), font=fsmall)
    dr.text((COL_W + 32, H_OUT - 26), f"{len(P)/1e3:.0f}k points · removed: depth smears, flying pixels, free-space floaters, grazing, far, isolated",
            fill=(120, 120, 120), font=font(14))
    return np.array(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--sources", default="gt,pred_tokens,pred_tiled")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--clouds", default="clouds", help="cloud dir under --root")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    scenes = json.loads((root / "metrics.json").read_text())["scenes"]
    if args.scenes:
        scenes = [scenes[int(i)] for i in args.scenes.split(",")]
    for scene in scenes:
        for src in args.sources.split(","):
            sub, tag, label = SOURCES[src]
            frames = np.load(root / scene / f"{src}.npy")           # V, T, H, W, 3
            cloud, meta = load_cloud(root / args.clouds / scene / src, frames)
            # one presentation camera per scene (from GT clouds) so GT / Pred match
            gref, gmeta = (cloud, meta) if src == "gt" else load_cloud(
                root / args.clouds / scene / "gt", np.load(root / scene / "gt.npy"))
            geom = scene_frame(gmeta, gref)
            geom = geom[:4] + (np.array([[np.array(v["M"]) for v in fr] for fr in meta["frames"]]), geom[5])
            T = len(cloud)
            n_out = T * HOLD * LOOPS
            if args.preview:
                d = out / "preview"; d.mkdir(parents=True, exist_ok=True)
                for t in (0, T // 2, T - 1):
                    im = compose(t, t / T * 0.5, frames, cloud, meta, geom, label, dev)
                    Image.fromarray(im).save(d / f"{scene[:3]}_{tag}_t{t:02d}.jpg", quality=90)
                print("preview", scene, tag, [len(c[0]) for c in cloud], flush=True)
                continue
            d = out / scene / sub
            d.mkdir(parents=True, exist_ok=True)
            w = imageio.get_writer(d / f"{scene}_{tag}.mp4", fps=FPS, codec="libx264",
                                   quality=8, macro_block_size=1, pixelformat="yuv420p")
            for i in range(n_out):
                t = (i // HOLD) % T
                w.append_data(compose(t, i / n_out, frames, cloud, meta, geom, label, dev))
            w.close()
            print("wrote", d / f"{scene}_{tag}.mp4", flush=True)


if __name__ == "__main__":
    main()
