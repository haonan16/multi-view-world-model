"""Compare the multiview ablation runs (tokens vs tiled) from their wandb histories.

Prints, per validation step, the held-out future-frame metrics of both designs at full
per-view resolution and at tile resolution, plus the VAE ceilings and copy-frame-0
baseline, and each run's measured GPU-hours to reach that step.

Usage: python scripts/rh20t/compare_runs.py <wandb_run_id_tokens> <wandb_run_id_tiled>
"""
import sys

import wandb

ENTITY_PROJECT = "amanim71/ei_world_model"
KEYS = ["psnr_full", "ssim_full", "lpips_full", "psnr_tile", "ssim_tile", "lpips_tile",
        "psnr_vae_full", "lpips_vae_full", "psnr_vae_tile", "psnr_copy0_full", "lpips_copy0_full"]


def history(run):
    out = {}
    for r in run.scan_history():
        if "val/psnr_full" in r and r.get("val/psnr_full") is not None:
            step = int(r.get("trainer/global_step", r.get("_step")))
            out[step] = {k: r.get(f"val/{k}") for k in KEYS}
            out[step]["runtime_h"] = r.get("_runtime", 0) / 3600
    return out


def main():
    api = wandb.Api()
    runs = {name: api.run(f"{ENTITY_PROJECT}/{rid}") for name, rid in zip(("tokens", "tiled"), sys.argv[1:3])}
    hist = {n: history(r) for n, r in runs.items()}
    steps = sorted(set(hist["tokens"]) | set(hist["tiled"]))
    fmt = lambda v: "   -  " if v is None else f"{v:6.3f}" if abs(v) < 2 else f"{v:6.2f}"
    for metric in KEYS:
        print(f"\n{metric}")
        print(f"{'step':>6} {'tokens':>8} {'tiled':>8}")
        for s in steps:
            a = hist["tokens"].get(s, {}).get(metric)
            b = hist["tiled"].get(s, {}).get(metric)
            print(f"{s:>6} {fmt(a):>8} {fmt(b):>8}")
    print("\nwall-clock hours (4 GPUs) at each validation:")
    for s in steps:
        a = hist["tokens"].get(s, {}).get("runtime_h")
        b = hist["tiled"].get(s, {}).get("runtime_h")
        print(f"{s:>6} {fmt(a):>8} {fmt(b):>8}")


if __name__ == "__main__":
    main()
