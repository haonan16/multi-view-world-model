"""Smoke test of the v2 training pipeline on one GPU with the real Wan weights, real v2 data
(17 frames @ 1.5 fps, scale-normalised cameras), cached task text and the diffusion-forcing
mixture: for every cam_mode, a few optimiser steps at the per-GPU batch size (peak memory,
s/step), then one 1-frame-history sample (shape / finite check).
Usage: python scripts/rh20t/smoke_v2.py [batch_size] [modes,comma,separated]
"""
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from algorithms.wan.wan_multiview import WanMultiView  # noqa: E402
from datasets.rh20t import RH20TMultiViewDataset  # noqa: E402

bs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["concat", "cat4d", "prope", "gta", "camtoken"]
V2 = ["algorithm.text.enabled=true", "algorithm.gradient_checkpointing_rate=1",
      "algorithm.noise.mix={first_frame:0.4,rand_history:0.15,independent:0.2,view_history:0.15,none:0.1}",
      "algorithm.noise.hist_noise_prob=0.2", "algorithm.sample_steps=4"]


def cfg_for(mode):
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configurations")):
        return compose(config_name="config", overrides=[
            "experiment=exp_video", "algorithm=wan_multiview", "dataset=rh20t_multiview_v2",
            "experiment.training.lr=2e-5", f"algorithm.cam_mode={mode}"] + V2)


ds = RH20TMultiViewDataset(cfg_for("concat").dataset, "test")
samples = [ds[i] for i in range(bs)]
batch = {k: (torch.stack([s[k] for s in samples]) if torch.is_tensor(samples[0][k])
             else [s[k] for s in samples]) for k in samples[0]}
print("batch", {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)}, batch["prompts"][:2], flush=True)
for mode in modes:
    algo = WanMultiView(cfg_for(mode).algorithm)
    algo.configure_model()
    algo = algo.cuda()
    opt = torch.optim.AdamW(algo.model.parameters(), lr=2e-5)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for it in range(4):
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = algo.training_step(batch)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.time() - t0)
    mem = torch.cuda.max_memory_allocated() / 2**30
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lat, cond = algo.pack({k: (v[:1] if torch.is_tensor(v) else v[:1]) for k, v in batch.items()})
        b1 = {k: (v[:1] if torch.is_tensor(v) else v[:1]) for k, v in batch.items()}
        z = algo.sample(lat, cond, 1, batch=b1, text_guidance=2.0, hist_guidance=1.0)
    print(f"{mode:9s} bs={bs} loss={float(loss):.4f} s/step={sum(times[1:])/3:.2f} peak={mem:.1f} GiB "
          f"sample {tuple(z.shape)} finite={bool(torch.isfinite(z).all())}", flush=True)
    del algo, opt
    torch.cuda.empty_cache()
print("SMOKE_DONE")
