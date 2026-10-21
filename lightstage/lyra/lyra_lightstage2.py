"""Lyra 2.0 light stage, v2: short anchored flights.

Rig: 64 cameras (3 rings x 20 at 20/40/60 deg + 4 at 75 deg) on a sphere of radius R_DOME around a
target TARGET_H above the table (table plane/up from rig.json), looking at the target, so every
camera frames the table and the whole arm and none sits inside the robot.

Flights: each real view k starts its own flight (world re-based so view k = identity) that visits
the stage cameras of its azimuth sector; all 4 real views (colour + metric VGGT-Omega depth) seed
Lyra's spatial memory in every flight, with view k first. Short flights stay in the low-drift regime.
Writes <out>/<case>/{rig2.json, flight_k/{video.mp4, traj.npz, stage_frames.json}, frames/ls_XXX.png}
"""
import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, "/tmp/akiruga_lyra")
import lyra_lightstage as LL  # noqa: E402

H, W = LL.H, LL.W
import os
EXCL = int(os.environ.get("EXCL", -1)); KEPT = [v for v in range(4) if v != EXCL]
R_DOME, TARGET_H, DEG = 1.15, 0.20, 3.0
ELEVS, N_AZ, CAP_EL, N_CAP = (20.0, 40.0, 60.0), 20, 75.0, 4


def make_rig(rig):
    up = np.array(rig["up"]); table = np.array(rig["centre"]) - 0.05 * up; tgt = table + TARGET_H * up
    o0 = np.array(rig["inputs"]["c2w"])[0][:3, 3]; v0 = o0 - tgt; h0 = v0 - up * (v0 @ up); h0 /= np.linalg.norm(h0); h1 = np.cross(up, h0)
    def pos(el, az, r=R_DOME):
        el, az = np.deg2rad(el), np.deg2rad(az); return tgt + r * (np.cos(el) * (np.cos(az) * h0 + np.sin(az) * h1) + np.sin(el) * up)
    elaz = [(e, a) for e in ELEVS for a in np.arange(N_AZ) * 360.0 / N_AZ] + [(CAP_EL, a) for a in np.arange(N_CAP) * 360.0 / N_CAP + 45]
    c2w = [LL.look_at(pos(e, a), tgt, up) for e, a in elaz]
    Kt = np.array(rig["stage"]["K"])                       # same intrinsics as the old stage (160x320 units)
    def elaz_of(p):
        v = p - tgt; e = np.degrees(np.arcsin(np.clip(v @ up / np.linalg.norm(v), -1, 1)))
        hv = v - up * (v @ up); return float(e), float(np.degrees(np.arctan2(hv @ h1, hv @ h0)) % 360), float(np.linalg.norm(v))
    real = [elaz_of(np.array(c)[:3, 3]) for c in rig["inputs"]["c2w"]]
    return {"target": tgt.tolist(), "up": up.tolist(), "radius": R_DOME, "h0": h0.tolist(), "h1": h1.tolist(),
            "stage": {"c2w": [c.tolist() for c in c2w], "elev_az": [list(map(float, x)) for x in elaz], "K": Kt.tolist()},
            "real_elev_az_dist": real}, pos


def assign(rig2):
    """Stage camera -> real view whose azimuth is nearest (circular); balanced enough for our rigs."""
    raz = [e[1] for e in rig2["real_elev_az_dist"]]
    groups = {k: [] for k in KEPT}
    for i, (e, a) in enumerate(rig2["stage"]["elev_az"]):
        k = min(KEPT, key=lambda k: abs((a - raz[k] + 180) % 360 - 180)); groups[k].append(i)
    return groups


def flight(rig2, pos, k, cams, c2w_k, K_k, Kt):
    """Trajectory (calibrated frame) from real view k through its cameras; returns c2w list, K list, stage map."""
    tgt, up = np.array(rig2["target"]), np.array(rig2["up"]); elaz = rig2["stage"]["elev_az"]; st = rig2["stage"]["c2w"]
    rel = lambda a: (a - rig2["real_elev_az_dist"][k][1] + 180) % 360 - 180
    el0 = rig2["real_elev_az_dist"][k][0]
    rings = {}
    for i in cams: rings.setdefault(elaz[i][0], []).append(i)
    order = sorted(rings, key=lambda e: abs(e - el0)); seq, d = [], 1
    for e in order:
        cs = sorted(rings[e], key=lambda i: d * rel(elaz[i][1])); seq += cs; d *= -1
    c2ws, Ks, sm = [c2w_k], [K_k], {}
    def go(A, B, KA, KB, on_dome=None):
        pa, pb = A[:3, 3] - tgt, B[:3, 3] - tgt
        ang = np.degrees(np.arccos(np.clip(pa @ pb / (np.linalg.norm(pa) * np.linalg.norm(pb)), -1, 1)))
        rot = np.degrees(np.arccos(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1)))
        dr = abs(np.linalg.norm(pb) - np.linalg.norm(pa)) / np.linalg.norm(pb) * 57.3   # zoom counts as motion
        n = max(2, math.ceil(max(ang, rot, dr) / DEG))
        if on_dome is not None:
            (e1, z1), (e2, z2) = on_dome; dz = (z2 - z1 + 180) % 360 - 180
            for j in range(1, n + 1):
                s = j / n; c2ws.append(LL.look_at(pos(e1 + s * (e2 - e1), z1 + s * dz), tgt, up)); Ks.append(Kt.copy())
        else:
            sl = Slerp([0, 1], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))
            for j in range(1, n + 1):
                s = j / n; m = np.eye(4); m[:3, :3] = sl([s]).as_matrix()[0]; m[:3, 3] = (1 - s) * A[:3, 3] + s * B[:3, 3]
                c2ws.append(m); Ks.append((1 - s) * KA + s * KB)
        c2ws[-1] = B
    go(c2w_k, np.array(st[seq[0]]), K_k, Kt); sm[seq[0]] = len(c2ws) - 1
    for a, b in zip(seq[:-1], seq[1:]):
        go(np.array(st[a]), np.array(st[b]), Kt, Kt, on_dome=(elaz[a], elaz[b])); sm[b] = len(c2ws) - 1
    return c2ws, Ks, sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True); ap.add_argument("--out", default="/tmp/akiruga_lyra/out2")
    ap.add_argument("--use_dmd", action="store_true"); ap.add_argument("--dry", action="store_true")
    cli = ap.parse_args()
    model = None
    for case in cli.cases:
        out = Path(cli.out) / case; (out / "frames").mkdir(parents=True, exist_ok=True)
        rig = json.loads((LL.CASES / case / "rig.json").read_text()); rig2, pos = make_rig(rig)
        (out / "rig2.json").write_text(json.dumps(rig2))
        imgs, deps, Ks_n, c2w_real, h, w = LL.load_real(rig)
        K_real = Ks_n.copy(); K_real[:, 0] *= W / w; K_real[:, 1] *= H / h
        Kt = np.array(rig2["stage"]["K"], dtype=np.float64); Kt[0] *= W / rig["W"]; Kt[1] *= H / rig["H"]
        groups = assign(rig2); print(case, "cameras per real view", {k: len(v) for k, v in groups.items()})
        for k, cams in groups.items():
            fd = out / f"flight_{k}"; fd.mkdir(exist_ok=True)
            if not cams or (fd / "video.mp4").exists(): continue
            c2ws, Ks, sm = flight(rig2, pos, k, cams, c2w_real[k], K_real[k], Kt)
            N = 1 + 80 * math.ceil((len(c2ws) - 1) / 80)
            while len(c2ws) < N:                                   # hold the last camera
                c2ws.append(c2ws[-1].copy()); Ks.append(Ks[-1].copy())
            base = np.linalg.inv(c2w_real[k])                       # re-base: view k = identity
            w2cs = np.stack([np.linalg.inv(base @ c) for c in c2ws]).astype(np.float32); Ks = np.stack(Ks).astype(np.float32)
            np.savez(fd / "traj.npz", w2c=w2cs, intrinsics=Ks, base=base, image_height=H, image_width=W)
            (fd / "stage_frames.json").write_text(json.dumps({int(a): int(b) for a, b in sm.items()}))
            steps = [np.degrees(np.arccos(np.clip((np.trace(c2ws[i][:3, :3].T @ c2ws[i + 1][:3, :3]) - 1) / 2, -1, 1))) for i in range(len(c2ws) - 1)]
            print(f"  flight {k}: {len(cams)} cams, {len(sm)} stage frames, N={N}, max rot step {max(steps):.1f} deg")
            if cli.dry: continue
            if model is None:
                model, args, extras = load_lyra(cli)
            run_flight(model, args, extras, case, fd, k, imgs, deps, K_real, c2w_real, base, w2cs, Ks, N, sm, out)


def load_lyra(cli):
    from lyra_2._src.inference import lyra2_custom_traj_inference as L
    from lyra_2._src.inference.depth_utils import load_da3_model
    from lyra_2._ext.imaginaire.utils import misc
    sys.argv = ["x", "--input_image_path", "x", "--trajectory_path", "x", "--experiment", "lyra2", "--checkpoint_dir",
                "checkpoints/model", "--resolution", f"{H},{W}", "--pose_scale", "1.0", "--no-use_moge_scale",
                "--num_retrieval_views", str(len(KEPT)), "--multiview_ids"] + [str(i) for i in range(len(KEPT))] + [] + (["--use_dmd"] if cli.use_dmd else [])
    args = L.parse_arguments(); L._apply_dmd_defaults(args); misc.set_random_seed(seed=args.seed, by_rank=True)
    opts = ["model.config.use_mp_policy_fsdp=False", "model.config.keep_original_net_dtype=False"]
    if args.lora_paths: opts += ["model.config.net.postpone_checkpoint=True"]
    model, _ = L.load_model_from_checkpoint(config_file="lyra_2/_src/configs/config.py", experiment_name=args.experiment,
                                            checkpoint_path=args.checkpoint_dir, enable_fsdp=False, instantiate_ema=False,
                                            load_ema_to_reg=False, experiment_opts=opts)
    if args.lora_paths:
        names = [model.load_lora_weights(p) for p in args.lora_paths]; model.set_weights_and_activate_adapters(names, args.lora_weights)
        if hasattr(model.net, "enable_selective_checkpoint"): model.net.enable_selective_checkpoint(model.net.sac_config, model.net.blocks)
    dt, dev = model.tensor_kwargs.get("dtype"), model.tensor_kwargs.get("device")
    model.net = model.net.to(device=dev, dtype=dt); model.eval()
    da3 = load_da3_model(da3_model_name=args.da3_model_name, da3_model_path_custom=None, device=dev); da3.eval()
    neg = torch.load("checkpoints/text_encoder/negative_prompt.pt", map_location="cpu", weights_only=False)
    return model, args, {"da3": da3, "neg": neg, "misc": misc}


def run_flight(model, args, ex, case, fd, k, imgs, deps, K_real, c2w_real, base, w2cs, Ks, N, sm, out):
    from lyra_2._src.inference.lyra2_ar_inference import run_lyra2_sample, safe_to
    from lyra_2._src.inference.get_t5_emb import get_umt5_embedding
    import imageio.v2 as imageio
    dt, dev = model.tensor_kwargs.get("dtype"), model.tensor_kwargs.get("device"); misc = ex["misc"]
    order = [k] + [v for v in KEPT if v != k]                          # flight's own view first (held-out view excluded)
    x = torch.from_numpy(imgs[order]).permute(0, 3, 1, 2).float() / 127.5 - 1
    x = F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False).clamp(-1, 1)
    dd = F.interpolate(torch.from_numpy(deps[order])[:, None].float(), size=(H, W), mode="nearest")[:, 0]
    mv_video = x.permute(1, 0, 2, 3)[None].to(dev, dt); mv_depth = dd[None].to(dev)
    mv_w2c = torch.from_numpy(np.stack([np.linalg.inv(base @ c2w_real[v]) for v in order])).float()[None].to(dev)
    mv_K = torch.from_numpy(K_real[order]).float()[None].to(dev)
    t5 = get_umt5_embedding(LL.CAPTIONS[case[:3]], device=dev).to(dtype=dt)
    if t5.dim() == 2: t5 = t5[None]
    batch = {"video": mv_video[:, :, :1].clone(), "t5_text_embeddings": t5,
             "neg_t5_text_embeddings": misc.to(ex["neg"]["t5_text_embeddings"], **model.tensor_kwargs),
             "fps": torch.tensor([args.fps], dtype=torch.int32, device=dev),
             "padding_mask": torch.zeros((1, 1, H, W), dtype=dt, device=dev),
             "is_preprocessed": torch.tensor([True], dtype=torch.bool, device=dev),
             "camera_w2c": torch.from_numpy(w2cs)[None].to(dev), "intrinsics": torch.from_numpy(Ks)[None].to(dev),
             "depth": mv_depth[:, :1].repeat(1, N, 1, 1), "mv_video": mv_video, "mv_depth": mv_depth, "mv_w2c": mv_w2c, "mv_K": mv_K}
    batch = safe_to(batch, device=dev, dtype=dt, skip_keys={"camera_w2c", "intrinsics", "depth", "mv_depth", "mv_w2c", "mv_K"})
    args.num_frames = N
    res = run_lyra2_sample(model, batch, args, da3_model=ex["da3"], show_progress=True, log_prefix=f"{case}_flight{k}")
    vid = (res["video"][0].clamp(-1, 1) * 0.5 + 0.5).float().cpu(); frames = (vid.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    imageio.mimwrite(fd / "video.mp4", list(frames), fps=16, macro_block_size=1, quality=9)
    for i, f in sm.items():
        Image.fromarray(frames[f]).save(out / "frames" / f"ls_{int(i):03d}.png")
    print(case, f"flight {k} saved", len(frames)); del res, batch; torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    main()
