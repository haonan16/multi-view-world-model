"""Bake VGGT-Omega dynamic clouds (vggt_dynamic_clouds.py --native) into data for the local
interactive viewer (scripts/rh20t/viewer/index.html), with every cleaning filter of
render_cloud_videos.py precomputed per pixel so the page can toggle each one live.

Per (scene, source) and timestep t, writes <out>/data/<i>/<source>/:
  dc_<t>.png   views stacked vertically (V*H x W), lossless RGB:
               R,G = 12-bit depth (per view min/max) in R and the high nibble of G,
               G low nibble = VGGT-Omega depth-confidence rank (16 levels),
               B = filter bits, 1 = point passes:
                 0 not a flying pixel     1 not on a depth smear (ramp)
                 2 no free-space violation 3 not seen at a grazing angle
                 4 pixel footprint < MAX_FOOT  5 not isolated (voxel density, among
                 points passing 0-4 and 6)    6 inside the workspace sphere
  col_<t>.jpg  matching colour atlas
  meta.json    V, T, H, W, per-frame per-view K / M (world-from-cam) / dmin / dmax,
               workspace centre + radius, up / forward axes for the initial camera
and <out>/scenes.json (scene list + image and point-cloud metrics).

Usage: python scripts/rh20t/bake_cloud_viewer.py --root INFER_OUT --out VIEWER_DIR [--procs 8] [--watch]
"""
import argparse
import json
import time
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_cloud_videos as R  # noqa: E402

SOURCES = ("gt", "pred_tokens", "pred_tiled")


def density_ok(P):
    """R.density_filter as a mask: >= VOX_MIN points in the 3x3x3 voxel neighbourhood."""
    if len(P) == 0:
        return np.zeros(0, bool)
    k = np.floor(P / R.VOX).astype(np.int64)
    k -= k.min(0)
    dims = k.max(0) + 3
    key = lambda kk: ((kk[:, 0] + 1) * dims[1] + kk[:, 1] + 1) * dims[2] + kk[:, 2] + 1
    uniq, cnt = np.unique(key(k), return_counts=True)
    total = np.zeros(len(P), np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                kk = key(k + np.array([dx, dy, dz]))
                pos = np.clip(np.searchsorted(uniq, kk), 0, len(uniq) - 1)
                total += np.where(uniq[pos] == kk, cnt[pos], 0)
    return total >= R.VOX_MIN


def workspace(meta):
    """Same as R.scene_frame: centre = least-squares meeting point of the cameras' optical
    axes (frame 0), radius = WS_RADIUS x median camera distance; up / forward axes."""
    Ms = np.array([np.array(v["M"]) for v in meta["frames"][0]])
    o, d = Ms[:, :, 3], Ms[:, :, 2]
    A = sum(np.eye(3) - np.outer(di, di) for di in d)
    b = sum((np.eye(3) - np.outer(di, di)) @ oi for oi, di in zip(o, d))
    c = np.linalg.solve(A, b)
    rad = R.WS_RADIUS * np.median(np.linalg.norm(o - c, axis=1))
    up = -Ms[:, :, 1].mean(0); up /= np.linalg.norm(up)
    fwd = Ms[:, :, 2].mean(0); fwd -= up * (fwd @ up); fwd /= np.linalg.norm(fwd)
    return c, rad, up, fwd


def bake(job):
    i, scene, src, root, out, ws = job
    d = root / scene / src
    meta = json.loads((d / "meta.json").read_text())
    V, T, H, W = meta["V"], meta["T"], meta["H"], meta["W"]
    a = np.array(Image.open(d / "dc.png"))
    col = np.array(Image.open(d / "color.jpg"))
    centre, rad = np.array(ws["centre"]), ws["radius"]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)
    od = out / "data" / str(i) / src
    od.mkdir(parents=True, exist_ok=True)
    kept = []
    for t in range(T):
        depth, Ks, Ms, Xs, flags, conf, q12 = [], [], [], [], [], [], []
        for vi in range(V):
            blk = a[(t * V + vi) * H:(t * V + vi + 1) * H]
            m = meta["frames"][t][vi]
            q = blk[..., 0].astype(np.int64) * 256 + blk[..., 1]
            dd = (q / 65535 * (m["dmax"] - m["dmin"]) + m["dmin"]).reshape(-1)
            b = blk[..., 2].reshape(-1)
            K, M = np.array(m["K"]), np.array(m["M"])
            Xc = (pix @ np.linalg.inv(K).T) * dd[:, None]
            Xw = Xc @ M[:, :3].T + M[:, 3]
            f = np.zeros(H * W, np.uint8)
            f |= ((b & 1) == 1).astype(np.uint8) << 0
            f |= (~R.on_ramp(dd.reshape(H, W)).reshape(-1)).astype(np.uint8) << 1
            f |= (R.view_cos(Xc.reshape(H, W, 3)).reshape(-1) > R.GRAZE_COS).astype(np.uint8) << 3
            f |= (dd / K[0, 0] < R.MAX_FOOT).astype(np.uint8) << 4
            f |= (np.linalg.norm(Xw - centre, axis=1) < R.CROP * rad).astype(np.uint8) << 6
            depth.append(dd.reshape(H, W)); Ks.append(K); Ms.append(M); Xs.append(Xw)
            flags.append(f); conf.append(((b >> 2) >> 2).astype(np.uint8))      # 64 -> 16 levels
            q12.append((q >> 4).astype(np.uint16))
        for vi in range(V):
            fs = ~R.free_space_violation(Xs[vi], vi, depth, Ks, Ms)
            flags[vi] |= fs.astype(np.uint8) << 2
        # density among points passing every other default filter (as in the videos)
        base = [((fl & 0b1011111) == 0b1011111) & (depth[vi].reshape(-1) > 1e-4)
                for vi, fl in enumerate(flags)]
        P = np.concatenate([Xs[vi][base[vi]] for vi in range(V)])
        ok = density_ok(P)
        o = 0
        for vi in range(V):
            n = base[vi].sum()
            dens = np.zeros(H * W, bool)
            dens[np.where(base[vi])[0]] = ok[o:o + n]
            o += n
            flags[vi] |= dens.astype(np.uint8) << 5
        kept.append(int(sum(((fl == 0b1111111)).sum() for fl in flags)))
        img = np.zeros((V * H, W, 3), np.uint8)
        for vi in range(V):
            s = slice(vi * H, (vi + 1) * H)
            qq = q12[vi].reshape(H, W)
            img[s, :, 0] = (qq >> 4).astype(np.uint8)
            img[s, :, 1] = (((qq & 15) << 4) | conf[vi].reshape(H, W)).astype(np.uint8)
            img[s, :, 2] = flags[vi].reshape(H, W)
        Image.fromarray(img).save(od / f"dc_{t}.png", compress_level=6)
        Image.fromarray(col[t * V * H:(t + 1) * V * H]).save(od / f"col_{t}.jpg", quality=92)
    m2 = {"V": V, "T": T, "H": H, "W": W, "frames": meta["frames"], **ws}
    (od / "meta.json").write_text(json.dumps(m2))
    print(i, src, "points/frame after default cleaning:", int(np.mean(kept)), flush=True)
    return i, src, int(np.mean(kept))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--clouds", default="clouds_native")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--watch", action="store_true",
                    help="bake scenes as vggt_dynamic_clouds.py finishes them; scenes.json is "
                         "rewritten after each, so the page shows what is ready")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((root / "metrics.json").read_text())
    cdir = root / args.clouds
    ready = lambda s: all((cdir / s / src / "meta.json").exists() for src in SOURCES)
    done, pts = set(), {}

    def write_index():
        cj = cdir / "clouds.json"
        clouds = json.loads(cj.read_text()) if cj.exists() else {}
        scenes = [{"i": i, "name": s.split("_", 1)[1], "metrics": m["per_scene"][s],
                   "clouds": clouds.get(s, {}), "points": {src: pts[(i, src)] for src in SOURCES}}
                  for i, s in enumerate(m["scenes"]) if i in done]
        tmp = out / "scenes.json.tmp"
        tmp.write_text(json.dumps({"ckpt": m["ckpt"], "mean": m["mean"], "n_total": len(m["scenes"]),
                                   "cloud_mean": clouds.get("mean", {}), "scenes": scenes}))
        tmp.replace(out / "scenes.json")

    with Pool(args.procs) as p:
        while len(done) < len(m["scenes"]):
            todo = [(i, s) for i, s in enumerate(m["scenes"]) if i not in done and ready(s)]
            if not todo:
                if not args.watch:
                    break
                time.sleep(30)
                continue
            for i, s in todo:
                c, rad, up, fwd = workspace(json.loads((cdir / s / "gt" / "meta.json").read_text()))
                ws = {"centre": c.tolist(), "radius": float(rad), "up": up.tolist(), "fwd": fwd.tolist()}
                for _, src, n in p.map(bake, [(i, s, src, cdir, out, ws) for src in SOURCES]):
                    pts[(i, src)] = n
                done.add(i)
                write_index()
                print(f"scene {i} ready ({len(done)}/{len(m['scenes'])})", flush=True)
    write_index()


if __name__ == "__main__":
    main()
