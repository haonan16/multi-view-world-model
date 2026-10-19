"""Run the tokens and tiled multiview models on held-out RH20T clips.

Exactly the validation task (algorithms/wan/wan_multiview.py): condition on frame 0 of
every view (clean latent frame 0) + per-view raymaps, sample the remaining frames with
UniPC (30 steps, shift 3), decode. The tiled design's 80x160 tiles are upsampled back to
160x320 for comparison.

For each scene writes <out>/<scene>/:
  gt.npy, pred_tokens.npy, pred_tiled.npy   uint8 [V, T, H, W, 3]
  cams.npz                                   intrinsics [V,3,3] (160x320 px), c2w [V,4,4]
                                             (view 0 = I, metric, from RH20T calibration)
  compare.mp4                                rows: GT / tokens / tiled, columns: views
  latent_<label>.npy, latent_gt.npy          with --save_latents: float16 [V, C, F, h, w]
and <out>/metrics.json with per-scene and mean PSNR / SSIM / LPIPS on frames 1.. .

Usage: python scripts/rh20t/infer_multiview.py --out DIR [--n 15]
          [--tokens_ckpt PATH] [--tiled_ckpt PATH]                       (v1: tokens vs tiled)
       python scripts/rh20t/infer_multiview.py --out DIR --models concat=concat:CKPT,prope=prope:CKPT,...
          [--text_guidance S] [--hist_guidance S]   (v2: tokens + cam_mode, 17 @ 1.5 fps, task text)
Both condition on 1 frame of history: latent frame 0 = pixel frame 0 of every view.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from algorithms.wan.wan_multiview import WanMultiView, _ssim  # noqa: E402
from datasets.rh20t import RH20TMultiViewDataset  # noqa: E402

RUNS = Path("/n/netscratch/ydu_lab/Lab/akiruga/mv_runs")
DEFAULT_CKPT = {"tokens": RUNS / "mv_tokens_v1_51672240/checkpoints/last.ckpt",
                "tiled": RUNS / "mv_tiled_v1_51672243/checkpoints/last.ckpt"}


V2_OVERRIDES = ["algorithm.text.enabled=true"]


def load_cfg(design, dataset="rh20t_multiview", extra=()):
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configurations")):
        return compose(config_name="config",
                       overrides=["experiment=exp_video", "algorithm=wan_multiview",
                                  f"dataset={dataset}", f"algorithm.design={design}",
                                  "experiment.training.lr=2e-5", *extra])


def load_algo(design, ckpt, dataset="rh20t_multiview", extra=()):
    cfg = load_cfg(design, dataset, extra)
    algo = WanMultiView(cfg.algorithm)
    algo.configure_model()
    ckpt = Path(ckpt).resolve()                       # pin the target before it rotates
    sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    step = int(sd.get("global_step", -1))
    msd = {k[len("model."):]: v for k, v in sd["state_dict"].items() if k.startswith("model.")}
    algo.model.load_state_dict(msd)
    del sd, msd
    algo = algo.cuda().eval()
    return algo, cfg, step, str(ckpt)


def u8(x):  # [..., 3, H, W] in [-1, 1] -> uint8 [..., H, W, 3]
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().movedim(-3, -1).cpu().numpy()


@torch.no_grad()
def predict(algo, sample, seed, text_guidance=0.0, hist_guidance=0.0, hist_views=None):
    batch = {k: (v[None] if torch.is_tensor(v) else [v]) for k, v in sample.items()}
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lat, cond = algo.pack(batch)
        # history = latent frame 0 (pixel frame 0) of the first hist_views views (default all);
        # every other view is generated from scratch. Task text + cameras from batch.
        hm = torch.zeros(lat.shape[:2] + lat.shape[3:4], dtype=torch.bool, device=lat.device)
        hm[:, :hist_views, :algo.cfg.n_hist_latent] = True
        z = algo.sample(lat, cond, batch=batch, hist_mask=hm,
                        text_guidance=text_guidance, hist_guidance=hist_guidance)
        x = algo._views(z, 1).float()[0]                                   # V, T, 3, h, w
    predict.latents = (z[0].float().cpu(), lat[0].float().cpu())          # sampled / GT latents
    H, W = sample["videos"].shape[-2:]
    if x.shape[-2:] != (H, W):
        x = F.interpolate(x.flatten(0, 1), size=(H, W), mode="bilinear",
                          align_corners=False).unflatten(0, x.shape[:2])
    return x.clamp(-1, 1)


def metrics(pred, gt, lp, hist_views=None):
    """over every generated (view, frame): all frames 1.. plus frame 0 of the non-history views"""
    keep = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
    keep[:hist_views, 0] = False
    p, g = (pred[keep] + 1) / 2, (gt[keep] + 1) / 2
    mse = ((p - g) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
    return {"psnr": float((-10 * torch.log10(mse)).mean()),
            "ssim": float(_ssim(p.clamp(0, 1), g).mean()),
            "lpips": float(lp(p * 2 - 1, g * 2 - 1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--tokens_ckpt", default=str(DEFAULT_CKPT["tokens"]))
    ap.add_argument("--tiled_ckpt", default=str(DEFAULT_CKPT["tiled"]))
    ap.add_argument("--models", default=None,
                    help="v2: comma list of label=cam_mode:ckpt (tokens design, rh20t_multiview_v2 "
                         "data, task text); replaces --tokens_ckpt/--tiled_ckpt")
    ap.add_argument("--text_guidance", type=float, default=0.0)
    ap.add_argument("--hist_guidance", type=float, default=0.0)
    ap.add_argument("--hist_views", type=int, default=None,
                    help="history = frame 0 of only the first K views (e.g. 1: view 0 alone); "
                         "default: frame 0 of every view")
    ap.add_argument("--save_latents", action="store_true",
                    help="also save the sampled latents latent_<label>.npy and latent_gt.npy "
                         "(float16 [V, C, F, h, w], VAE-normalised) per scene")
    args = ap.parse_args()
    if args.models:
        dataset = "rh20t_multiview_v2"
        models = [(lab, "tokens", ck, [f"algorithm.cam_mode={cm}", *V2_OVERRIDES])
                  for lab, rest in (m.split("=", 1) for m in args.models.split(","))
                  for cm, ck in [rest.split(":", 1)]]
    else:
        dataset = "rh20t_multiview"
        models = [("tokens", "tokens", args.tokens_ckpt, []), ("tiled", "tiled", args.tiled_ckpt, [])]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()

    ds = RH20TMultiViewDataset(load_cfg("tokens", dataset).dataset, "test")
    idxs = np.linspace(0, len(ds) - 1, args.n).round().astype(int)
    samples, scenes = [], []
    for i in idxs:                                     # deterministic per idx in "test"
        s = ds[int(i)]
        name = f"{i:03d}_{s['episode']}"
        if name in scenes:
            continue
        samples.append(s)
        scenes.append(name)
        d = out / name
        d.mkdir(exist_ok=True)
        np.save(d / "gt.npy", u8(s["videos"]))
        np.savez(d / "cams.npz", intrinsics=s["intrinsics"].numpy(), c2w=s["c2w"].numpy(),
                 serials=np.array(s.get("serials", [])), episode=s["episode"])
    print(f"{len(samples)} held-out scenes: {scenes}", flush=True)

    report = {"scenes": scenes, "per_scene": {n: {} for n in scenes}, "ckpt": {}}
    report["guidance"] = {"text": args.text_guidance, "hist": args.hist_guidance}
    report["hist_views"] = args.hist_views
    labels = [m[0] for m in models]
    for mi, (label, design, ckpt, extra) in enumerate(models):
        algo, _, step, path = load_algo(design, ckpt, dataset, extra)
        report["ckpt"][label] = {"path": path, "global_step": step, "overrides": extra}
        print(f"{label}: step {step} from {path}", flush=True)
        for k, (name, s) in enumerate(zip(scenes, samples)):
            pred = predict(algo, s, seed=k, text_guidance=args.text_guidance,
                           hist_guidance=args.hist_guidance, hist_views=args.hist_views)
            gt = s["videos"].cuda()
            report["per_scene"][name][label] = metrics(pred, gt, lp, args.hist_views)
            if mi == 0:   # copy-frame-0 baseline, once (an ORACLE when hist_views < V: it copies
                          # frame 0 of every view, which the model does not get)
                report["per_scene"][name]["copy0"] = metrics(gt[:, :1].expand_as(gt), gt, lp,
                                                             args.hist_views)
            np.save(out / name / f"pred_{label}.npy", u8(pred))
            if args.save_latents:
                z, zgt = predict.latents
                np.save(out / name / f"latent_{label}.npy", z.half().numpy())
                if not (out / name / "latent_gt.npy").exists():
                    np.save(out / name / "latent_gt.npy", zgt.half().numpy())
            print(f"  {name}: {report['per_scene'][name][label]}", flush=True)
        del algo
        torch.cuda.empty_cache()

    for name in scenes:                               # GT / each model, views side by side
        rows = [np.load(out / name / f"{r}.npy") for r in ["gt"] + [f"pred_{l}" for l in labels]]
        V, T = rows[0].shape[:2]
        frames = [np.concatenate([np.concatenate(list(r[:, t]), axis=1) for r in rows], axis=0)
                  for t in range(T)]
        imageio.mimwrite(out / name / "compare.mp4", frames, fps=3, macro_block_size=1)

    report["mean"] = {m: {k: float(np.mean([report["per_scene"][n][m][k] for n in scenes]))
                          for k in ("psnr", "ssim", "lpips")}
                      for m in labels + ["copy0"]}
    (out / "metrics.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["mean"], indent=1))


if __name__ == "__main__":
    main()
