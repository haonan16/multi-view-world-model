"""Depth Anything 3 (multi-view, known cameras, metric via the camera scale) on world-model data.
python da3_wm.py est            -> results/wm/da3_est/<Task>_episode-XXXX_est.npz  (real RGB of the eval episodes)
python da3_wm.py roll <dirs...> -> <dir>/da3.npz                                   (predicted RGB of rollouts)
z: [T, V, H, W] camera-z metres"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, imageio.v3 as iio
sys.path.insert(0, "/n/lab_storage/ydu_lab/Lab/akiruga/depth-anything-3/src")
sys.modules["xformers.ops"] = None          # xformers kernels here are sm_90-only: plain SwiGLU
from depth_anything_3.api import DepthAnything3
ROOT = Path(__file__).resolve().parent
da3 = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cuda").eval()


def run(frames, K, c2w):                      # frames [V,T,H,W,3] -> z [T,V,H,W]
    V, T, H, W = frames.shape[:4]
    w2c = np.linalg.inv(c2w).astype(np.float32)
    out = np.zeros((T, V, H, W), np.float32)
    for t in range(T):
        with torch.no_grad():
            pred = da3.inference(image=[frames[v, t] for v in range(V)], extrinsics=w2c, intrinsics=K.astype(np.float32),
                                 align_to_input_ext_scale=True, infer_gs=False, process_res=504, export_dir=None)
        dep = torch.from_numpy(np.asarray(pred.depth, dtype=np.float32))[:, None]
        out[t] = F.interpolate(dep, size=(H, W), mode="bilinear", align_corners=False)[:, 0].numpy()
    return out


if sys.argv[1] == "est":
    E = Path("/n/lab_storage/ydu_lab/Lab/akiruga/geo_runs/eval_s7000/A/A")
    od = ROOT / "results/wm/da3_est"; od.mkdir(parents=True, exist_ok=True)
    for d in sorted(E.iterdir()):
        o = od / (d.name + "_est.npz")
        if o.exists():
            continue
        m = json.loads((d / "meta.json").read_text()); z = np.load(d / "dense.npz")
        nr = int(z["n_real"]); V, H, W = 4, m["H"], m["W"]
        frames = z["rgb"][:, :nr]                                         # V,T,H,W,3
        zz = run(frames, np.array(m["K"]), np.array(m["c2w"]))
        np.savez_compressed(o, z=zz.astype(np.float16))
        g = z["z_gt"][:, :nr].astype(np.float32).transpose(1, 0, 2, 3); ok = z["valid"].transpose(1, 0, 2, 3)
        e = z["z_est"][:, :nr].astype(np.float32).transpose(1, 0, 2, 3)
        print(d.name, "absrel DA3 %.3f  model %.3f" % (np.abs(zz[ok] / g[ok] - 1).mean(), np.abs(e[ok] / g[ok] - 1).mean()), flush=True)
else:
    for d in map(Path, sys.argv[2:]):
        if (d / "da3.npz").exists():
            continue
        m = json.loads((d / "meta.json").read_text()); z = np.load(d / "dense.npz")
        zz = run(z["rgb_pred"], np.array(m["K"]), np.array(m["c2w"]))
        np.savez_compressed(d / "da3.npz", z=zz.astype(np.float16))
        g = z["z_gt"][:, :1].astype(np.float32).transpose(1, 0, 2, 3); ok = z["valid"][:, :1].transpose(1, 0, 2, 3)
        r = z["z_roll"][:, :1].astype(np.float32).transpose(1, 0, 2, 3)
        print(d.name, "frame-0 absrel DA3 %.3f  model %.3f" % (np.abs(zz[:1][ok] / g[ok] - 1).mean(), np.abs(r[ok] / g[ok] - 1).mean()), flush=True)
