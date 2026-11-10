"""Result / comparison clips for the deck, cut from the saved overlay videos (CPU only)."""
import os
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from artirecon.data import load_dense

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "deck/media"
OUT.mkdir(parents=True, exist_ok=True)
GEOM = {"gt512": (512, 512, 4), "gt": (256, 448, 4), "est_B": (256, 448, 4), "roll_A": (256, 448, 4),
        "roll_B": (256, 448, 4), "real": (360, 640, 5)}


def write(name, frames, fps=12):
    frames = [f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2] for f in frames]
    iio.imwrite(OUT / f"{name}.mp4", np.stack(frames), fps=fps, codec="libx264", macro_block_size=2,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "20"])
    iio.imwrite(OUT / f"{name}.jpg", frames[-1], quality=90)
    print("wrote", name, frames[0].shape, len(frames), flush=True)


def overlays(tag, kind):
    """-> T, V, h, w, 3 overlay (bottom row) images"""
    v = iio.imread(ROOT / "results" / tag / "overlay.mp4")
    h, w, V = GEOM[kind]
    return v[:, h:2 * h, :V * w].reshape(len(v), h, V, w, 3).transpose(0, 2, 1, 3, 4)


def resize_T(x, T):
    idx = np.round(np.linspace(0, len(x) - 1, T)).astype(int)
    return x[idx]


def grid2x2(O, views=(0, 1, 2, 3), gap=6):
    out = []
    for t in range(len(O)):
        a = [O[t, v] for v in views]
        h, w = a[0].shape[:2]
        g = np.full((2 * h + gap, 2 * w + gap, 3), 255, np.uint8)
        g[:h, :w], g[:h, w + gap:], g[h + gap:, :w], g[h + gap:, w + gap:] = a
        out.append(g)
    return out


def row(clips, gap=10):
    h = max(c.shape[1] for c in clips)
    T = max(len(c) for c in clips)
    clips = [resize_T(c, T) for c in clips]
    out = []
    for t in range(T):
        ims = []
        for c in clips:
            im = c[t]
            if im.shape[0] != h:
                im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA)
            ims += [im, np.full((h, gap, 3), 255, np.uint8)]
        out.append(np.concatenate(ims[:-1], 1))
    return out


def hold(seq, n):
    return [f for f in seq for _ in range(n)]


def depth_vis(d, lo=0.3, hi=1.9):
    dn = np.clip((d - lo) / (hi - lo), 0, 1)
    c = cv2.applyColorMap((255 - dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
    return np.where((d > 0)[..., None], c, 0).astype(np.uint8)


# results: 2x2 overlays
write("r_peg_overlay", hold(grid2x2(overlays("standard/gt512__mv", "gt512")), 12))
write("r_drawer_overlay", hold(grid2x2(overlays("drawer/gt512__mv", "gt512")), 12))
write("r_real_overlay", hold(grid2x2(overlays("real/gt__mv_cal_recal", "real"), views=(0, 1, 2, 4)), 6))
write("r_modelB_overlay", hold(grid2x2(overlays("standard/est_B__mv_hybrid", "est_B")), 6))
write("r_rollA_overlay", hold(grid2x2(overlays("standard/roll_A__mv_cal", "roll_A")), 6))

# comparisons: view 0, side by side
for scene in ("standard", "drawer"):
    cl = [overlays(f"{scene}/gt512__{c}", "gt512")[:, 0] for c in ("mv", "mono_depth", "mono_rgb")]
    write(f"c_inputs_{scene}", hold(row(cl), 12))
cl = [overlays(f"real/gt__{c}", "real")[:, 0] for c in ("mv", "mv_cal_recal", "mono_depth_cal", "mono_rgb")]
write("c_inputs_real", hold(row(cl), 6))

# depth source: input depth (top) + fitted overlay (bottom), view 0
tops, bots = [], []
for kind, run, tag in (("gt", "B", "gt__mv"), ("est", "B", "est_B__mv_hybrid"), ("roll", "A", "roll_A__mv_cal"),
                       ("roll", "B", "roll_B__mv_cal")):
    o = load_dense(kind, run=run)
    tops.append(np.stack([depth_vis(o.depth[t, 0]) for t in range(o.T)]))
    bots.append(overlays(f"standard/{tag}", "gt")[:, 0])
both = [np.concatenate([a, np.full((a.shape[0], 8, a.shape[2], 3), 255, np.uint8), b], 1) for a, b in zip(tops, bots)]
write("c_depth_source", hold(row(both), 6))

# SAPIEN replay renders
for scene in ("drawer", "standard"):
    ims = [iio.imread(ROOT / f"results/{scene}/gt512__mv/sim/render_t{t}.png")[..., :3] for t in range(5)]
    write(f"r_sapien_{scene}", hold(ims, 12))
