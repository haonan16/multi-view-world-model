"""Load the RH20T multiview dataset exactly as training would and log it to wandb.

Checks that frames, synchronisation and cameras are right, and measures loader speed:
  - views/<i>         the V views side by side (what the "tokens" design sees)
  - tiled/<i>         the V views packed into one canvas (what the "tiled" design sees)
  - geometry/<i>      frame 0 of each view with (a) the calibration-marker origin
                      projected through each view's own extrinsics (should hit the SAME
                      physical spot in every view) and (b) epipolar lines of 3 points
                      picked in view 0 (each line must pass through that point's
                      match in the other views)
  - raymaps/<i>       ray directions as RGB + moment magnitude, per view
  - cameras/<i>       3D plot of the V camera poses (view 0 = identity)
  - stats             sync error, cfg / rating mix, per-batch load time, throughput

Usage (from the repo root):
  python scripts/rh20t/check_dataset_wandb.py [hydra overrides...] \
      [+check.n_batches=20 +check.batch_size=4 +check.num_workers=8]
"""
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from hydra import compose, initialize_config_dir
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from datasets.rh20t import RH20TMultiViewDataset  # noqa: E402
from datasets._multiview import tile_views  # noqa: E402

COLORS = [(255, 60, 60), (60, 220, 60), (70, 140, 255), (255, 200, 0), (255, 0, 255)]


def to_uint8(x):  # [..., 3, H, W] in [-1, 1] -> uint8
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte()


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def geometry_panel(s, scale=2):
    """Marker origin + epipolar lines on frame 0 of every view."""
    V, _, _, H, W = s["videos"].shape
    K = s["intrinsics"].double().numpy()
    c2w = s["c2w"].double().numpy()
    w2c_m = s["w2c_marker"].double().numpy()
    ims = [Image.fromarray(to_uint8(s["videos"][v, 0]).permute(1, 2, 0).numpy())
           .resize((W * scale, H * scale)) for v in range(V)]
    draws = [ImageDraw.Draw(im) for im in ims]
    pts = [(0.3 * W, 0.6 * H), (0.5 * W, 0.5 * H), (0.7 * W, 0.4 * H)]
    for v in range(V):
        c = w2c_m[v] @ np.array([0, 0, 0, 1.0])
        if c[2] > 0:
            u, w_ = (K[v] @ c[:3])[:2] / c[2]
            draws[v].rectangle([u * scale - 7, w_ * scale - 7, u * scale + 7, w_ * scale + 7],
                               outline=(255, 255, 255), width=3)
    for k, (x, y) in enumerate(pts):
        col = COLORS[k]
        draws[0].ellipse([x * scale - 6, y * scale - 6, x * scale + 6, y * scale + 6], outline=col, width=3)
        for v in range(1, V):
            T = np.linalg.inv(c2w[v]) @ c2w[0]            # cam0 -> cam v
            F = np.linalg.inv(K[v]).T @ skew(T[:3, 3]) @ T[:3, :3] @ np.linalg.inv(K[0])
            a, b, c = F @ np.array([x, y, 1.0])
            if abs(b) > 1e-9:
                xs = np.array([0, W])
                ys = -(a * xs + c) / b
                draws[v].line([(xs[0] * scale, ys[0] * scale), (xs[1] * scale, ys[1] * scale)], fill=col, width=2)
    canvas = Image.new("RGB", (W * scale * V, H * scale))
    for v, im in enumerate(ims):
        canvas.paste(im, (v * W * scale, 0))
    return canvas


def raymap_panel(s):
    r = s["raymaps"]                                     # V, 6, Hr, Wr
    d = (r[:, :3] * 0.5 + 0.5).clamp(0, 1)
    m = r[:, 3:].norm(dim=1, keepdim=True)
    m = (m / m.max().clamp(min=1e-6)).expand(-1, 3, -1, -1)
    top = torch.cat(list(d), dim=-1)
    bot = torch.cat(list(m), dim=-1)
    return (torch.cat([top, bot], dim=-2).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def camera_plot(s):
    c2w = s["c2w"].numpy()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    for v, M in enumerate(c2w):
        o = M[:3, 3]
        for k, col in enumerate("rgb"):
            e = o + 0.1 * M[:3, k]
            ax.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]], color=col)
        ax.text(*o, f"v{v}")
    ax.set_title(f"cfg{s['cfg_id']} cameras (m), view0 = I")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main():
    overrides = [a for a in sys.argv[1:] if not a.startswith("+check.")]
    chk = dict(a[len("+check."):].split("=", 1) for a in sys.argv[1:] if a.startswith("+check."))
    n_batches = int(chk.get("n_batches", 20))
    bs = int(chk.get("batch_size", 4))
    nw = int(chk.get("num_workers", 8))
    n_log = int(chk.get("n_log", 6))
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configurations")):
        cfg = compose(config_name="config",
                      overrides=["dataset=rh20t_multiview",
                                 "algorithm=wan_t2v_ray_depth_mot_concat_5b"] + overrides)
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                     name=chk.get("name", "rh20t_dataset_check"), job_type="data_check",
                     config={"dataset": dict(cfg.dataset), "check": chk})

    t0 = time.time()
    ds = {sp: RH20TMultiViewDataset(cfg.dataset, sp) for sp in ("training", "test")}
    init_s = time.time() - t0
    wandb.summary.update({"n_train_episodes": len(ds["training"]),
                          "n_test_episodes": len(ds["test"]), "dataset_init_s": init_s})

    loader = torch.utils.data.DataLoader(ds["training"], batch_size=bs, num_workers=nw,
                                         shuffle=True, persistent_workers=False)
    errs, cfgs, waits, logged = [], [], [], 0
    t_last = time.time()
    t_start = t_last
    for b, batch in enumerate(loader):
        waits.append(time.time() - t_last)
        V, T = batch["videos"].shape[1:3]
        assert batch["videos"].shape[3:] == (3, cfg.dataset.height, cfg.dataset.width)
        assert torch.isfinite(batch["videos"]).all() and torch.isfinite(batch["raymaps"]).all()
        errs.append(batch["sync_err_ms"].flatten())
        cfgs += batch["cfg_id"].tolist()
        for i in range(bs):
            if logged >= n_log:
                break
            s = {k: (v[i] if torch.is_tensor(v) or isinstance(v, list) else v) for k, v in batch.items()}
            cap = f"{s['episode']} | cams {s['serials']} | max sync {s['sync_err_ms'].max():.0f} ms"
            side = to_uint8(torch.cat(list(s["videos"]), dim=-1))          # T, 3, H, V*W
            tiled = to_uint8(tile_views(s["videos"]))                       # T, 3, H, W
            wandb.log({
                f"views/{logged}": wandb.Video(side.numpy(), fps=cfg.dataset.fps, format="mp4", caption=cap),
                f"tiled/{logged}": wandb.Video(tiled.numpy(), fps=cfg.dataset.fps, format="mp4", caption=cap),
                f"geometry/{logged}": wandb.Image(geometry_panel(s), caption=cap + " | white box = marker origin; lines = epipolar of view-0 circles"),
                f"raymaps/{logged}": wandb.Image(raymap_panel(s), caption="top: ray dir as RGB; bottom: |moment|"),
                f"cameras/{logged}": wandb.Image(camera_plot(s)),
            }, step=b)
            logged += 1
        wandb.log({"loader/batch_wait_s": waits[-1]}, step=b)
        t_last = time.time()
        if b + 1 >= n_batches:
            break
    total = time.time() - t_start
    e = torch.cat(errs).numpy()
    wandb.log({"stats/sync_err_ms": wandb.Histogram(e),
               "stats/cfg_id": wandb.Histogram(np.array(cfgs), num_bins=7)})
    wandb.summary.update({
        "samples_per_s": n_batches * bs / total, "workers": nw, "batch_size": bs,
        "sync_err_ms_p50": float(np.percentile(e, 50)), "sync_err_ms_max": float(e.max()),
        "batch_wait_s_median": float(np.median(waits[1:] or waits)),
    })
    print(f"OK: {n_batches} batches x {bs} in {total:.1f}s ({n_batches * bs / total:.2f} samples/s); "
          f"sync err p50 {np.percentile(e, 50):.0f} max {e.max():.0f} ms; run {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
