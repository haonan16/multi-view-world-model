"""World-model rollouts (view 0 frame 0 + cameras + text -> RGB + geometry of every view/frame) on many
held-out ManiSkill episodes, for picking successful predictions.  Same export format as eval_geo.py
(<out>/<task>_<episode>_s<seed>/{rgb.mp4, rgb_pred.mp4, depth_roll.mp4, dense.npz, meta.json}); dense.npz
has z_gt / z_roll (camera-z, metres) / valid / rgb; no estimate stream.  Skips finished episodes.
  PYTHONPATH=$PWD python scripts/geo/rollout_batch.py --out DIR --run A=depth:CKPT [--n 50] [--seeds 0]
                                                       [--shard i/k]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_geo import REPO, TASKS, batchify, load, run, turbo, u8, vid  # noqa: E402
from datasets.geo_world_mix import _sub  # noqa: E402
from datasets.maniskill_mv import ManiSkillMultiViewDataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--n", type=int, default=50); ap.add_argument("--seeds", default="0")
ap.add_argument("--shard", default="0/1"); ap.add_argument("--tasks", default=",".join(TASKS))
a = ap.parse_args()
si, sk = [int(x) for x in a.shard.split("/")]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
label, rest = a.run.split("=", 1); rep, ck = rest.split(":", 1)
dcfg = OmegaConf.load(REPO / "configurations/dataset/geo_world_mix.yaml")
dss = []
for task in a.tasks.split(","):
    c = _sub(dcfg, "maniskill"); c.tasks = [task]
    dss.append(ManiSkillMultiViewDataset(c, "validation"))
jobs = [(ds, i, int(sd)) for i in range(a.n) for ds in dss if i < len(ds) for sd in a.seeds.split(",")]
jobs = jobs[si::sk]
algo, step, path, _ = load(rep, ck)
if __import__("os").environ.get("LOWMEM"):      # small GPUs: VAE encode / decode one view at a time
    def _per(f, offload):
        def g(x, *k, **kw):
            if offload:                                # DiT off the GPU while the VAE decodes
                algo.model.to("cpu"); torch.cuda.empty_cache()
            outs = []
            for i in range(len(x)):
                outs.append(f(x[i:i + 1], *k, **kw)); torch.cuda.empty_cache()
            if offload:
                algo.model.to("cuda")
            return torch.cat(outs)
        return g
    for nm in ("encode", "encode_geo", "decode", "decode_geo"):
        setattr(algo, nm, _per(getattr(algo, nm), nm.startswith("decode")))
for ds, i, seed in jobs:
    s = ds[i]
    name = s["episode"].replace("/", "_") + f"_s{seed}"
    d = out / name
    if (d / "meta.json").exists():
        continue
    b = batchify(s)
    V, T = s["videos"].shape[:2]; nr = int(s["n_real"])
    K, c2w, cen = b["intrinsics"][0][:, None].float(), b["c2w"][0][:, None].float(), b["center"][0].float()
    rgb, g_roll = run(algo, b, hist_all=False, seed=seed)
    P_roll, z_roll = algo.geo_points(g_roll.float(), K, c2w, cen, ray_lift=True)
    d.mkdir(parents=True, exist_ok=True)
    gt = b["videos"][0].float()
    vid(d / "rgb.mp4", u8(gt)); vid(d / "rgb_pred.mp4", u8(rgb))
    zgt = b["depth"][0].float().cpu().numpy(); ok = b["depth_valid"][0].bool().cpu().numpy()
    lo, hi = np.log(np.quantile(zgt[ok], 0.01)), np.log(np.quantile(zgt[ok], 0.99))
    vid(d / "depth_roll.mp4", turbo(np.log(np.clip(z_roll.cpu().numpy(), 1e-4, None)), lo, hi, np.ones_like(ok)))
    sc = float(s["scene_scale"])
    f16 = lambda z: (z.float().cpu().numpy() * sc).astype(np.float16)
    np.savez_compressed(d / "dense.npz", z_gt=f16(b["depth"][0]), z_roll=f16(z_roll), valid=ok[:, :nr],
                        rgb=u8(gt)[:, :nr], rgb_pred=u8(rgb)[:, :nr], n_real=nr)
    c2m = s["c2w"].double().numpy().copy(); c2m[:, :3, 3] *= sc
    (d / "meta.json").write_text(json.dumps({
        "source": "maniskill", "episode": s["episode"], "seed": seed, "prompt_key": s["prompts"], "n_real": nr,
        "T": int(T), "scene_scale_m": sc, "center_m": (s["center"].double().numpy() * sc).tolist(),
        "H": int(s["videos"].shape[-2]), "W": int(s["videos"].shape[-1]), "K": s["intrinsics"].numpy().tolist(),
        "c2w": c2m.tolist(), "run": label, "step": step, "ckpt": path}))
    print(label, step, name, flush=True)
