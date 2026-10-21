"""Lyra 2.0 light stage from our 4 calibrated RH20T views.

Per case:
  * the 4 real views (colour + VGGT-Omega metric depth aligned to the calibrated frame, from
    clouds_native) seed Lyra's spatial memory (multiview cache, negative frame ids), so every
    generated frame is warped/retrieved from real geometry and only unseen content is hallucinated;
  * frame 0 = real camera 0; the trajectory then flies smoothly over the dome and passes exactly
    through the 64 light-stage cameras of rig.json (world frame re-based so camera 0 = identity);
  * no monocular rescaling: our depth is metric, the trajectory is metric (pose_scale = 1).
Writes <out>/<case>/{video.mp4, traj.npz, stage_frames.json, frames/ls_XXX.png}

usage (from Lyra-2 dir): python lyra_lightstage.py --cases 145_gt_t12 ... [--use_dmd] [--num_frames N]
"""
import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

CASES = Path("/tmp/akiruga_lightstage/cases")
CLOUDS = Path("/n/netscratch/ydu_lab/Lab/akiruga/mv_infer/v1_heldout15/clouds_native")
CAPTIONS = {
    "145": "A static tabletop scene in a robotics lab. A wooden table with a large brown cardboard sheet on it; "
           "a small wooden cube and a black dot marker lie on the cardboard. A white UR robot arm with a black "
           "two-finger gripper hangs above the table, black cables run across the table, and silver aluminum "
           "extrusion frames and lab shelves surround it. The camera slowly orbits around the table.",
    "000": "A static tabletop scene in a robotics lab. A white marble-pattern table with a grey and yellow "
           "emergency button box with a green cap and a small black object on it. A white robot arm with a black "
           "gripper is above the box, silver aluminum extrusion rails and lab equipment surround the table. "
           "The camera slowly orbits around the table.",
}
H, W = 416, 832
DEG_PER_FRAME = 3.0


def look_at(eye, target, up):
    z = target - eye; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
    c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def load_real(rig):
    d = CLOUDS / rig["scene"] / rig["source"]; meta = json.loads((d / "meta.json").read_text())
    V, h, w = meta["V"], meta["H"], meta["W"]
    a = np.array(Image.open(d / "dc.png")); col = np.array(Image.open(d / "color.jpg")); t = rig["t"]
    imgs, deps, Ks, c2ws = [], [], [], []
    for v in range(V):
        blk = slice((t * V + v) * h, (t * V + v + 1) * h); m = meta["frames"][t][v]
        q = (a[blk, :, 0].astype(np.float64) * 256 + a[blk, :, 1]) / 65535
        dep = q * (m["dmax"] - m["dmin"]) + m["dmin"]; dep[a[blk, :, 0].astype(int) * 256 + a[blk, :, 1] == 0] = 0
        imgs.append(col[blk]); deps.append(dep); Ks.append(np.array(m["K"], dtype=np.float64)); c2ws.append(np.vstack([m["M"], [0, 0, 0, 1]]))
    return np.stack(imgs), np.stack(deps), np.stack(Ks), np.stack(c2ws), h, w


def build_trajectory(rig, c2w0, K0):
    """Smooth dome path through all stage cameras. Returns c2w list, K list, stage frame map."""
    centre, up, r = np.array(rig["centre"]), np.array(rig["up"]), rig["radius"]
    st_c2w = np.array(rig["stage"]["c2w"]); elaz = rig["stage"]["elev_az"]
    o0 = np.array(rig["inputs"]["c2w"])[0][:3, 3]; v0 = o0 - centre; h0 = v0 - up * (v0 @ up); h0 /= np.linalg.norm(h0); h1 = np.cross(up, h0)
    def pos(el, az):
        el, az = np.deg2rad(el), np.deg2rad(az)
        return centre + r * (np.cos(el) * (np.cos(az) * h0 + np.sin(az) * h1) + np.sin(el) * up)
    Kt = np.array(rig["stage"]["K"], dtype=np.float64); Kt[0] *= W / rig["W"]; Kt[1] *= H / rig["H"]
    # visiting order: rings by elevation closest to camera 0 first, serpentine in azimuth
    el0 = rig["inputs"]["elev_az_dist"][0][0]
    rings = {}
    for i, (e, a) in enumerate(elaz):
        rings.setdefault(round(e, 3), []).append((a, i))
    order = sorted(rings, key=lambda e: abs(e - el0))
    seq, direction, az_cur = [], 1, 0.0
    for e in order:
        cams = rings[e]
        # start from the camera closest in azimuth to the current azimuth, go around in `direction`
        start = min(cams, key=lambda c: abs((c[0] - az_cur + 180) % 360 - 180))
        rel = sorted(cams, key=lambda c: ((c[0] - start[0]) * direction) % 360)
        seq += [c[1] for c in rel]; az_cur = elaz[seq[-1]][1]; direction *= -1
    c2ws, Ks, stage_frames = [c2w0], [K0], {}
    # lead-in: camera 0 -> first stage camera
    b0 = st_c2w[seq[0]]; ang = np.degrees(np.arccos(np.clip(((c2w0[:3, 3] - centre) @ (b0[:3, 3] - centre)) /
                                                             (np.linalg.norm(c2w0[:3, 3] - centre) * np.linalg.norm(b0[:3, 3] - centre)), -1, 1)))
    rot = np.degrees(np.arccos(np.clip((np.trace(c2w0[:3, :3].T @ b0[:3, :3]) - 1) / 2, -1, 1)))
    n = max(4, math.ceil(max(ang, rot) / DEG_PER_FRAME)); sl = Slerp([0, 1], Rotation.from_matrix([c2w0[:3, :3], b0[:3, :3]]))
    for k in range(1, n + 1):
        s = k / n; m = np.eye(4); m[:3, :3] = sl([s]).as_matrix()[0]; m[:3, 3] = (1 - s) * c2w0[:3, 3] + s * b0[:3, 3]
        c2ws.append(m); Ks.append((1 - s) * K0 + s * Kt)
    stage_frames[seq[0]] = len(c2ws) - 1
    for a_i, b_i in zip(seq[:-1], seq[1:]):
        (e1, z1), (e2, z2) = elaz[a_i], elaz[b_i]; dz = (z2 - z1 + 180) % 360 - 180
        p1, p2 = pos(e1, z1) - centre, pos(e2, z2) - centre
        ang = np.degrees(np.arccos(np.clip(p1 @ p2 / (np.linalg.norm(p1) * np.linalg.norm(p2)), -1, 1)))
        Ra, Rb = st_c2w[a_i][:3, :3], st_c2w[b_i][:3, :3]
        rot = np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))
        n = max(2, math.ceil(max(ang, rot) / DEG_PER_FRAME))
        for k in range(1, n + 1):
            s = k / n; c2ws.append(look_at(pos(e1 + s * (e2 - e1), z1 + s * dz), centre, up)); Ks.append(Kt.copy())
        c2ws[-1] = st_c2w[b_i]; stage_frames[b_i] = len(c2ws) - 1
    return c2ws, Ks, stage_frames, seq, (pos, centre, up, Kt, elaz)


def pad_trajectory(c2ws, Ks, N, geo):
    """Pad to N frames by continuing slowly around the last ring (extra views, never revisiting stage cams' frames)."""
    pos, centre, up, Kt, elaz = geo
    last = c2ws[-1]; v = last[:3, 3] - centre; r = np.linalg.norm(v)
    el = np.degrees(np.arcsin(np.clip(v @ up / r, -1, 1)))
    # azimuth of last frame in the dome frame
    probe = [(abs(np.linalg.norm(pos(el, a) - last[:3, 3])), a) for a in np.arange(0, 360, 0.5)]
    az = min(probe)[1]
    k = 0
    while len(c2ws) < N:
        k += 1; c2ws.append(look_at(pos(el, az + k * DEG_PER_FRAME), centre, up)); Ks.append(Kt.copy())
    return c2ws[:N], Ks[:N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out", default="/tmp/akiruga_lyra/out")
    ap.add_argument("--num_frames", type=int, default=0, help="0 = smallest 1+80k that fits")
    ap.add_argument("--use_dmd", action="store_true")
    ap.add_argument("--num_retrieval_views", type=int, default=4)
    ap.add_argument("--tag", default="")
    cli = ap.parse_args()

    from lyra_2._src.inference import lyra2_custom_traj_inference as L
    from lyra_2._src.inference.lyra2_ar_inference import run_lyra2_sample, safe_to
    from lyra_2._src.inference.depth_utils import load_da3_model
    from lyra_2._src.inference.get_t5_emb import get_umt5_embedding
    sys.argv = ["x", "--input_image_path", "x", "--trajectory_path", "x", "--experiment", "lyra2",
                "--checkpoint_dir", "checkpoints/model", "--resolution", f"{H},{W}", "--pose_scale", "1.0",
                "--no-use_moge_scale", "--num_retrieval_views", str(cli.num_retrieval_views),
                "--multiview_ids", "0", "1", "2", "3"] + (["--use_dmd"] if cli.use_dmd else [])
    args = L.parse_arguments(); L._apply_dmd_defaults(args)
    from lyra_2._ext.imaginaire.utils import misc
    misc.set_random_seed(seed=args.seed, by_rank=True)
    neg = torch.load("checkpoints/text_encoder/negative_prompt.pt", map_location="cpu", weights_only=False)
    opts = ["model.config.use_mp_policy_fsdp=False", "model.config.keep_original_net_dtype=False"]
    if args.lora_paths: opts += ["model.config.net.postpone_checkpoint=True"]
    model, _ = L.load_model_from_checkpoint(config_file="lyra_2/_src/configs/config.py", experiment_name=args.experiment,
                                            checkpoint_path=args.checkpoint_dir, enable_fsdp=False, instantiate_ema=False,
                                            load_ema_to_reg=False, experiment_opts=opts)
    if args.lora_paths:
        names = [model.load_lora_weights(p) for p in args.lora_paths]
        model.set_weights_and_activate_adapters(names, args.lora_weights)
        if hasattr(model.net, "enable_selective_checkpoint"):
            model.net.enable_selective_checkpoint(model.net.sac_config, model.net.blocks)
    dt, dev = model.tensor_kwargs.get("dtype"), model.tensor_kwargs.get("device")
    model.net = model.net.to(device=dev, dtype=dt); model.eval()
    da3 = load_da3_model(da3_model_name=args.da3_model_name, da3_model_path_custom=None, device=dev); da3.eval()

    for case in cli.cases:
        out = Path(cli.out) / (case + cli.tag); (out / "frames").mkdir(parents=True, exist_ok=True)
        if (out / "video.mp4").exists():
            print("skip", case); continue
        rig = json.loads((CASES / case / "rig.json").read_text())
        imgs, deps, Ks_n, c2w_real, h, w = load_real(rig)
        sx, sy = W / w, H / h
        K_real = Ks_n.copy(); K_real[:, 0] *= sx; K_real[:, 1] *= sy
        # re-base world so real camera 0 = identity
        M0 = c2w_real[0].copy(); base = np.linalg.inv(M0)
        c2w_real_b = np.stack([base @ c for c in c2w_real])
        rig_b = json.loads(json.dumps(rig))
        rig_b["stage"]["c2w"] = [(base @ np.array(c)).tolist() for c in rig["stage"]["c2w"]]
        rig_b["inputs"]["c2w"] = [(base @ np.array(c)).tolist() for c in c2w_real]
        rig_b["centre"] = (base @ np.append(rig["centre"], 1))[:3].tolist(); rig_b["up"] = (base[:3, :3] @ np.array(rig["up"])).tolist()
        c2ws, Ks, stage_frames, seq, geo = build_trajectory(rig_b, c2w_real_b[0], K_real[0])
        N = cli.num_frames or (1 + 80 * math.ceil((len(c2ws) - 1) / 80))
        assert len(c2ws) <= N, (len(c2ws), N)
        c2ws, Ks = pad_trajectory(c2ws, Ks, N, geo)
        w2cs = np.stack([np.linalg.inv(c) for c in c2ws]).astype(np.float32); Ks = np.stack(Ks).astype(np.float32)
        np.savez(out / "traj.npz", w2c=w2cs, intrinsics=Ks, image_height=H, image_width=W, base=base,
                 c2w_real=c2w_real_b, K_real=K_real)
        (out / "stage_frames.json").write_text(json.dumps({int(k): int(v) for k, v in stage_frames.items()}))
        print(case, "trajectory frames", N, "stage cams", len(stage_frames), "last stage frame", max(stage_frames.values()))
        # seeds: images [-1,1], metric depth, cameras
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 127.5 - 1
        x = F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False).clamp(-1, 1)
        dd = F.interpolate(torch.from_numpy(deps)[:, None].float(), size=(H, W), mode="nearest")[:, 0]
        mv_video = x.permute(1, 0, 2, 3)[None].to(dev, dt)            # [1,C,4,H,W]
        mv_depth = dd[None].to(dev)                                  # [1,4,H,W]
        mv_w2c = torch.from_numpy(np.stack([np.linalg.inv(c) for c in c2w_real_b])).float()[None].to(dev)
        mv_K = torch.from_numpy(K_real).float()[None].to(dev)
        t5 = get_umt5_embedding(CAPTIONS[case[:3]], device=dev).to(dtype=dt)
        if t5.dim() == 2: t5 = t5[None]
        batch = {
            "video": mv_video[:, :, :1].clone(),
            "t5_text_embeddings": t5, "neg_t5_text_embeddings": misc.to(neg["t5_text_embeddings"], **model.tensor_kwargs),
            "fps": torch.tensor([args.fps], dtype=torch.int32, device=dev),
            "padding_mask": torch.zeros((1, 1, H, W), dtype=dt, device=dev),
            "is_preprocessed": torch.tensor([True], dtype=torch.bool, device=dev),
            "camera_w2c": torch.from_numpy(w2cs)[None].to(dev), "intrinsics": torch.from_numpy(Ks)[None].to(dev),
            "depth": mv_depth[:, :1].repeat(1, N, 1, 1),
            "mv_video": mv_video, "mv_depth": mv_depth, "mv_w2c": mv_w2c, "mv_K": mv_K,
        }
        skip = {"camera_w2c", "intrinsics", "depth", "mv_depth", "mv_w2c", "mv_K"}
        batch = safe_to(batch, device=dev, dtype=dt, skip_keys=skip)
        args.num_frames = N
        res = run_lyra2_sample(model, batch, args, da3_model=da3, show_progress=True, log_prefix=case)
        vid = (res["video"][0].clamp(-1, 1) * 0.5 + 0.5).float().cpu()          # [C,T,H,W]
        frames = (vid.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
        import imageio.v2 as imageio
        imageio.mimwrite(out / "video.mp4", list(frames), fps=16, macro_block_size=1, quality=9)
        for i, f in stage_frames.items():
            Image.fromarray(frames[f]).save(out / "frames" / f"ls_{int(i):03d}.png")
        print(case, "saved", len(frames), "frames")
        del res, batch; torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    main()
