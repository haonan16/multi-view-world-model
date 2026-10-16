"""Dynamic point clouds from multiview video predictions with VGGT-Omega.

Input: the output dir of scripts/rh20t/infer_multiview.py (<scene>/{gt,pred_tokens,
pred_tiled}.npy, uint8 [V, T, H, W, 3], and cams.npz with the RH20T calibration).

Per (scene, source) ALL V x T images go through VGGT-Omega in one forward pass (cameras
predicted, not given), so every frame shares one gauge -- no per-timestep scale drift.
The predicted camera centres are then Sim(3)-aligned (Umeyama) to the RH20T calibrated
centres (view 0 = identity, metres), which puts GT / tokens / tiled clouds in the same
metric frame. Depth and confidence are resampled to the source resolution (H x W).

Cleaning (all stored per point so the viewer can tune them live):
  - confidence: depth_conf as a percentile RANK within the (scene, source), 64 levels;
    a threshold of p keeps the top (100 - p)% of points
  - flying pixels: depth strictly between the near and far side of a 3x3 discontinuity
    (max/min > 1 + EDGE_TOL); object outlines and the background behind them are kept
  - multi-view consistency: a point is kept only if at least one OTHER view of the same
    instant sees a surface at the same depth (|z - z_other| / z < MV_TOL) when it is
    projected there -- standard MVS-fusion check, removes floaters no view agrees with

Writes <out>/<scene>/<source>/:
  dc.png     depth16 in R,G (per view-frame min/max); B = conf rank (6 bits, << 2) |
             bit1 multi-view consistent | bit0 not a depth edge; atlas rows =
             frame-major then view (row block = t * V + v), each block H x W
  color.jpg  same atlas layout, the source RGB frames
  meta.json  {V, T, H, W, frames: [[{K, M (world-from-cam 3x4), dmin, dmax}, ...V], ...T],
              sim3: {s, R, t}, align_rmse_m}
and <out>/clouds.json: per scene/source the Chamfer distance (mm) of each prediction's
cloud to the GT-frames cloud at several confidence cut-offs.

Usage: python scripts/rh20t/vggt_dynamic_clouds.py --root INFER_OUT [--out DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/n/lab_storage/ydu_lab/Lab/akiruga/vggt-omega")
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402

CKPT = next(Path("/n/netscratch/ydu_lab/Lab/akiruga/hub/models--facebook--VGGT-Omega/snapshots")
            .glob("*/vggt_omega_1b_512.pt"))
SOURCES = ("gt", "pred_tokens", "pred_tiled")
CONF_KEEP = (1.0, 0.7, 0.5, 0.3)          # fraction of points kept for the Chamfer table
EDGE_TOL = 0.05
MV_TOL = 0.03


def preprocess(frames, res=512, patch=16):
    """uint8 [N, H, W, 3] -> float [N, 3, h, w] in [0, 1], VGGT-Omega 'balanced' shape."""
    N, H, W, _ = frames.shape
    tokens = (res // patch) ** 2
    wp = np.sqrt(tokens / (H / W))
    hp, wp = int(round(tokens / wp)), int(round(wp))
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
    return F.interpolate(x, size=(hp * patch, wp * patch), mode="bicubic",
                         align_corners=False).clamp(0, 1)


def umeyama(src, dst):
    """Sim(3) s, R, t minimising |dst - (s R src + t)|. src, dst: [N, 3]."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


@torch.no_grad()
def reconstruct(model, frames_vt, c2w_gt, native=False):
    """frames_vt: uint8 [V, T, H, W, 3]. Returns depth [T,V,H,W] (metres, aligned), conf
    [T,V,H,W], K [T,V,3,3] at (H, W), M world-from-cam [T,V,3,4] in the calibrated frame."""
    V, T, H, W, _ = frames_vt.shape
    imgs = preprocess(frames_vt.transpose(1, 0, 2, 3, 4).reshape(T * V, H, W, 3)).cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(imgs[None])
    E, K = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
    hp, wp = pred["images"].shape[-2:]
    E, K = E[0].float().cpu().numpy(), K[0].float().cpu().numpy()
    depth = pred["depth"][0, ..., 0].float()
    conf = pred["depth_conf"][0].float()
    if native:                      # keep VGGT-Omega's working resolution (more points)
        H, W = hp, wp
        depth, conf = depth.cpu().numpy(), conf.cpu().numpy()
    else:
        depth = F.interpolate(depth[:, None], size=(H, W), mode="nearest")[:, 0].cpu().numpy()
        conf = F.interpolate(conf[:, None], size=(H, W), mode="area")[:, 0].cpu().numpy()
        K[:, 0] *= W / wp
        K[:, 1] *= H / hp
    Rwc = E[:, :, :3].transpose(0, 2, 1)                       # world-from-cam rotation
    C = -np.einsum("nij,nj->ni", Rwc, E[:, :, 3])               # camera centres
    C_gt = np.tile(c2w_gt[:, :3, 3], (T, 1))                   # image n = t * V + v
    s, R, t = umeyama(C, C_gt)
    rmse = float(np.sqrt(((s * C @ R.T + t - C_gt) ** 2).sum(1).mean()))
    M = np.concatenate([R @ Rwc, (s * C @ R.T + t)[:, :, None]], 2)
    shape = (T, V)
    return (depth.reshape(*shape, H, W) * s, conf.reshape(*shape, H, W),
            K.reshape(*shape, 3, 3), M.reshape(*shape, 3, 4),
            {"s": float(s), "R": R.tolist(), "t": t.tolist()}, rmse)


def unproject(d, K, M):
    """d [H, W], K [3,3], M [3,4] -> world points [H*W, 3]."""
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3) @ np.linalg.inv(K).T
    Xc = rays * d.reshape(-1, 1)
    return Xc @ M[:, :3].T + M[:, 3]


def edge_ok(d):
    """False only for flying pixels: depth strictly between the nearest and farthest depth
    of the 3x3 neighbourhood (smeared across a discontinuity). Foreground pixels on an
    object's outline (depth ~ local min) and the background behind it (~ local max) are
    kept, so thin structure like the robot arm keeps its silhouette. d: [H, W]."""
    t = torch.from_numpy(d)[None, None]
    mx = F.max_pool2d(t, 3, 1, 1)[0, 0].numpy()
    mn = -F.max_pool2d(-t, 3, 1, 1)[0, 0].numpy()
    jump = mx / np.maximum(mn, 1e-6) > 1 + EDGE_TOL
    between = (d > mn * (1 + EDGE_TOL / 3)) & (d < mx / (1 + EDGE_TOL / 3))
    return ~(jump & between)


def mv_consistent(depth, K, M):
    """depth [V,H,W], K [V,3,3], M [V,3,4] (one instant) -> bool [V,H,W]: the point is
    seen at the same depth by at least one other view."""
    V, H, W = depth.shape
    ok = np.zeros((V, H * W), bool)
    X = [unproject(depth[v], K[v], M[v]) for v in range(V)]
    for v in range(V):
        for u in range(V):
            if u == v:
                continue
            R, c = M[u][:, :3], M[u][:, 3]
            Xc = (X[v] - c) @ R                               # world -> cam u
            z = Xc[:, 2]
            uv = Xc @ K[u].T
            px, py = uv[:, 0] / np.maximum(z, 1e-6), uv[:, 1] / np.maximum(z, 1e-6)
            ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
            inside = (z > 1e-4) & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            zu = np.full_like(z, np.inf)
            zu[inside] = depth[u][iy[inside], ix[inside]]
            ok[v] |= inside & (np.abs(z - zu) < MV_TOL * z)
    return ok.reshape(V, H, W)


def chamfer_mm(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    a = torch.from_numpy(a[rng.choice(len(a), min(n, len(a)), replace=False)]).cuda()
    b = torch.from_numpy(b[rng.choice(len(b), min(n, len(b)), replace=False)]).cuda()
    d = torch.cdist(a[None].float(), b[None].float())[0]
    return float((d.min(1).values.mean() + d.min(0).values.mean()) / 2 * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--native", action="store_true",
                    help="store depth at VGGT-Omega's resolution instead of the source's")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out or root / "clouds")
    out.mkdir(parents=True, exist_ok=True)
    scenes = json.loads((root / "metrics.json").read_text())["scenes"]

    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))

    report = {}
    for scene in scenes:
        c2w = np.load(root / scene / "cams.npz")["c2w"]
        clouds = {}
        report[scene] = {}
        for src in SOURCES:
            frames = np.load(root / scene / f"{src}.npy")             # V, T, H, W, 3
            depth, conf, K, M, sim3, rmse = reconstruct(model, frames, c2w, args.native)
            V, T = frames.shape[:2]
            H, W = depth.shape[-2:]
            if frames.shape[2:4] != (H, W):                         # colour at depth resolution
                x = torch.from_numpy(frames.reshape(V * T, *frames.shape[2:])).permute(0, 3, 1, 2).float()
                x = F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False).clamp(0, 255)
                frames = x.round().byte().permute(0, 2, 3, 1).numpy().reshape(V, T, H, W, 3)
            rank = (np.searchsorted(np.sort(conf.ravel()), conf, side="right")
                    / conf.size * 255).round().astype(np.uint8)
            edge = np.stack([np.stack([edge_ok(depth[t, v]) for v in range(V)]) for t in range(T)])
            mv = np.stack([mv_consistent(depth[t], K[t], M[t]) for t in range(T)])
            clean = edge & mv
            d_atlas = np.zeros((T * V * H, W, 3), np.uint8)
            meta = {"V": V, "T": T, "H": H, "W": W, "frames": [], "sim3": sim3,
                    "align_rmse_m": rmse}
            pts = []
            for t in range(T):
                fr = []
                for v in range(V):
                    d = depth[t, v]
                    dmin, dmax = float(d.min()), float(max(d.max(), d.min() + 1e-6))
                    q = np.round((d - dmin) / (dmax - dmin) * 65535).astype(np.uint16)
                    blk = slice((t * V + v) * H, (t * V + v + 1) * H)
                    d_atlas[blk, :, 0], d_atlas[blk, :, 1] = q >> 8, q & 255
                    d_atlas[blk, :, 2] = ((rank[t, v] >> 2) << 2) | (mv[t, v] << 1) | edge[t, v]
                    fr.append({"K": K[t, v].tolist(), "M": M[t, v].tolist(),
                               "dmin": dmin, "dmax": dmax})
                    pts.append((t, unproject(d, K[t, v], M[t, v]), rank[t, v].ravel(),
                                clean[t, v].ravel()))
                meta["frames"].append(fr)
            d = out / scene / src
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(d_atlas).save(d / "dc.png", compress_level=9)
            Image.fromarray(frames.transpose(1, 0, 2, 3, 4).reshape(T * V * H, W, 3)).save(
                d / "color.jpg", quality=88)
            (d / "meta.json").write_text(json.dumps(meta))
            clouds[src] = pts
            report[scene][src] = {"align_rmse_mm": rmse * 1000, "scale": sim3["s"],
                                  "frac_edge_ok": float(edge.mean()), "frac_mv_ok": float(mv.mean()),
                                  "frac_clean": float(clean.mean())}
            print(f"{scene} {src}: cam-align rmse {rmse*1000:.1f} mm, s={sim3['s']:.3f}", flush=True)
        # geometry of each prediction vs the reconstruction of the real frames, per frame
        for src in SOURCES[1:]:
            for use_clean in (False, True):
                for keep in CONF_KEEP:
                    cut = 255 * (1 - keep)
                    sel = lambda r, c: (r >= cut) & (c | (not use_clean))
                    vals = []
                    for t in range(1, T):
                        g = np.concatenate([p[sel(r, c)] for tt, p, r, c in clouds["gt"] if tt == t])
                        p_ = np.concatenate([p[sel(r, c)] for tt, p, r, c in clouds[src] if tt == t])
                        vals.append(chamfer_mm(p_, g))
                    tag = "clean_" if use_clean else ""
                    report[scene][src][f"chamfer_mm_{tag}conf{int(keep*100)}"] = float(np.mean(vals))
        print(scene, {s: {k: round(v, 1) for k, v in report[scene][s].items()} for s in SOURCES[1:]},
              flush=True)

    keys = [k for k in report[scenes[0]]["pred_tokens"] if k.startswith("chamfer")]
    report["mean"] = {src: {k: float(np.mean([report[s][src][k] for s in scenes])) for k in keys}
                      for src in SOURCES[1:]}
    for src in SOURCES:
        for k in ("align_rmse_mm", "frac_edge_ok", "frac_mv_ok", "frac_clean"):
            report["mean"].setdefault(src, {})[k] = float(np.mean([report[s][src][k] for s in scenes]))
    (out / "clouds.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["mean"], indent=1))


if __name__ == "__main__":
    main()
