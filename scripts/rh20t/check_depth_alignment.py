# SPDX-License-Identifier: Apache-2.0
"""Verify RH20T depth + calibration + timestamp alignment on one episode.

For every static camera t, the depth of all OTHER static cameras at the same instant is
back-projected into one world point cloud and rendered into t. If depth units, intrinsics,
extrinsics and frame pairing are right, the render matches t's colour image and t's own
measured depth where covered. A pose-shuffled render (cloud rendered into another camera's
pose, compared against t) gives the "wrong geometry" reference.

    python scripts/rh20t/check_depth_alignment.py \
        --depth-root <dir holding RH20T_cfgN/<episode>/cam_*/depth.mp4> \
        --episode RH20T_cfg3/task_0001_user_0016_scene_0001_cfg_0003 --out align.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fastvideo.dataset.rh20t_depth import (DEPTH_H, DEPTH_W, clean_depth, read_depth_frames, render_points,
                                           unproject)
from fastvideo.dataset.rh20t_multiview import CALIB_H, CALIB_W, IN_HAND, nearest_frames, read_frames

LOWRES = "/n/netscratch/ydu_lab/Lab/akiruga/rh20t_lowres/data"
CALIB = "/n/lab_storage/ydu_lab/Lab/haonan/datasets/rh20t/calib"
OUT_H, OUT_W = 180, 320


def load_calib(calib_root: Path, cfg_id: int, ts: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    d = calib_root / f"RH20T_cfg{cfg_id}" / "calib" / str(ts)
    ex = np.load(d / "extrinsics.npy", allow_pickle=True).item()
    Ks = np.load(d / "intrinsics.npy", allow_pickle=True).item()
    out = {}
    for s, m in ex.items():
        try:
            out[str(s)] = (np.asarray(Ks[s], np.float64)[:3, :3], np.asarray(m, np.float64).reshape(-1, 4, 4)[0])
        except (TypeError, ValueError, KeyError):
            continue
    return out


def scaled_K(K: np.ndarray, h: int, w: int) -> torch.Tensor:
    K = K.copy()
    K[0] *= w / CALIB_W
    K[1] *= h / CALIB_H
    return torch.from_numpy(K)


def load_views(args) -> dict[str, dict]:
    ep = Path(args.lowres) / args.episode
    cfg_id = int(args.episode.split("/")[0].replace("RH20T_cfg", ""))
    calib = load_calib(Path(args.calib), cfg_id, int(json.loads((ep / "metadata.json").read_text())["calib"]))
    cams = [
        p.name[4:] for p in sorted((Path(args.depth_root) / args.episode).glob("cam_*"))
        if p.name[4:] in calib and p.name[4:] not in IN_HAND[cfg_id] and (p / "depth.mp4").exists()
    ]
    stamps = {s: np.load(ep / f"cam_{s}" / "color" / "timestamps.npy").astype(np.int64) for s in cams}
    t_ref = np.array([float(np.median(np.concatenate(list(stamps.values()))))])
    views = {}
    for s in cams:
        j, err = nearest_frames(stamps[s], t_ref)
        if err[0] > args.max_sync_err_ms:
            continue
        color = read_frames(ep / f"cam_{s}" / "color" / "color.mp4", j)[0].double() / 255.0  # 3, 180, 320
        depth = torch.from_numpy(read_depth_frames(Path(args.depth_root) / args.episode / f"cam_{s}" / "depth.mp4",
                                                   j)[0]).double()
        K, w2c = calib[s]
        views[s] = {
            "color": color,
            "depth": clean_depth(depth),
            "K_depth": scaled_K(K, DEPTH_H, DEPTH_W),
            "K_out": scaled_K(K, OUT_H, OUT_W),
            "w2c": torch.from_numpy(w2c),
            "err_ms": float(err[0]),
        }
    return views


def cloud_from(views: dict[str, dict], exclude: str) -> tuple[torch.Tensor, torch.Tensor]:
    pts, cols = [], []
    for s, v in views.items():
        if s == exclude:
            continue
        color = F.interpolate(v["color"][None], size=(DEPTH_H, DEPTH_W), mode="bilinear", align_corners=False)[0]
        p, valid = unproject(v["depth"], v["K_depth"], torch.linalg.inv(v["w2c"]))
        pts.append(p)
        cols.append(color.permute(1, 2, 0)[valid])
    return torch.cat(pts), torch.cat(cols)


def score(views, s, pose_of) -> dict[str, float]:
    v, u = views[s], views[pose_of]
    pts, cols = cloud_from(views, exclude=s)
    img, zbuf, mask = render_points(pts, cols, u["K_out"], u["w2c"], OUT_H, OUT_W)
    own = F.interpolate(v["depth"][None, None], size=(OUT_H, OUT_W), mode="nearest")[0, 0]
    both = mask & (own > 0)
    return {
        "coverage": float(mask.float().mean()),
        "rgb_l1": float((img - v["color"]).abs().mean(0)[mask].mean()) if mask.any() else float("nan"),
        "depth_med_abs_cm": float((zbuf - own).abs()[both].median() * 100) if both.any() else float("nan"),
        "img": img,
        "mask": mask,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth-root", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--lowres", default=LOWRES)
    ap.add_argument("--calib", default=CALIB)
    ap.add_argument("--max-sync-err-ms", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=Path("depth_alignment.png"))
    args = ap.parse_args()

    views = load_views(args)
    if len(views) < 2:
        raise SystemExit(f"need >= 2 synchronised static cameras with depth, got {list(views)}")
    serials = list(views)
    rows = []
    print(f"{'camera':>14} {'sync_ms':>7} {'depth_med_m':>11} | {'cover':>5} {'rgbL1':>6} {'dz_cm':>6} "
          f"| shuffled-pose {'rgbL1':>6} {'dz_cm':>6}")
    for i, s in enumerate(serials):
        good = score(views, s, s)
        bad = score(views, s, serials[(i + 1) % len(serials)])
        d = views[s]["depth"]
        print(f"{s:>14} {views[s]['err_ms']:7.1f} {float(d[d > 0].median()):11.3f} | {good['coverage']:5.2f} "
              f"{good['rgb_l1']:6.3f} {good['depth_med_abs_cm']:6.1f} | {'':13} {bad['rgb_l1']:6.3f} "
              f"{bad['depth_med_abs_cm']:6.1f}")
        diff = (good["img"] - views[s]["color"]).abs() * good["mask"]
        rows.append(torch.cat([views[s]["color"], good["img"], diff.clamp(0, 1)], dim=2))
    grid = (torch.cat(rows, dim=1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(grid).save(args.out)
    print(f"wrote {args.out}: rows = cameras; columns = real colour | render from other cameras' depth | |diff|")


if __name__ == "__main__":
    main()
