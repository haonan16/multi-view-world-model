"""Observation sources -> a common Obs container.

Obs: rgb [T,V,H,W,3] u8, depth [T,V,H,W] metres (0 = unknown / None if no depth),
K [V,3,3], c2w [V,4,4] (OpenCV, ManiSkill world), steps [T] control-step index of each frame,
seg [T,V,H,W] (evaluator only, never used for fitting), label str.

Sources
  gt512   : the package's observations_gt.npz (5 keyframes, 512x512) – official GT input
  gt      : the world model's GT frames (11 frames @1 Hz, 256x448 crop) – apples-to-apples GT
  est     : same frames, real RGB, depth predicted by the world model (all views given)
  roll    : world-model rollout from view-0 frame-0: predicted RGB + predicted depth for 4 views
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PKG = Path("/n/netscratch/ydu_lab/Lab/suraj/shared-maniskill/maniskill-reconstruction-scenes-20261101")
GEO = Path("/n/lab_storage/ydu_lab/Lab/akiruga/geo_runs")
MSR = Path("/n/lab_storage/ydu_lab/Lab/akiruga/maniskill_render")
FAR_REL = float(np.exp(1 / 0.9))          # datasets/_scene_norm.py: workspace range in scene units
MODEL_RUNS = {"A": GEO / "eval_s7000/A/A", "B": GEO / "eval_s9000/B/B"}


@dataclass
class Obs:
    label: str
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    c2w: np.ndarray
    steps: np.ndarray
    seg: np.ndarray = None
    meta: dict = field(default_factory=dict)

    @property
    def T(self):
        return self.rgb.shape[0]

    @property
    def V(self):
        return self.rgb.shape[1]

    def select(self, views=None, frames=None, drop_depth=False, label=None):
        v = list(range(self.V)) if views is None else list(views)
        f = list(range(self.T)) if frames is None else list(frames)
        sub = lambda a: None if a is None else a[f][:, v]
        return Obs(label or self.label, sub(self.rgb), None if drop_depth else sub(self.depth),
                   self.K[v], self.c2w[v], self.steps[f], sub(self.seg), dict(self.meta, views=v))


def load_gt512(scene):
    d = np.load(PKG / scene / "observations_gt.npz")
    w2c = d["w2c"][0].astype(np.float64)
    c2w = np.linalg.inv(w2c)
    dep = d["depth_m"].astype(np.float32) * d["valid_depth"]
    K = d["K"][0].astype(np.float64).copy()
    K[:, :2, 2] -= 0.5           # sapien K (pixel centres at +0.5) -> integer-pixel-index convention
    return Obs(f"{scene}/gt512", d["rgb"], dep, K, c2w,
               d["action_indices"].astype(int), d["segmentation"],
               {"scene": scene, "segment_names": [str(s) for s in d["segment_names"]],
                "segment_ids": d["segment_ids"].tolist(), "poses_wxyz": d["world_poses_wxyz"]})


def _vid(path):
    import imageio.v3 as iio
    return iio.imread(path)                                  # T, H, V*W, 3


def load_dense(kind, run="B", episode="PegInsertionSide-v1/episode-0950"):
    """kind in gt | est | roll : the world model's eval export of a validation episode."""
    d = MODEL_RUNS[run] / episode.replace("/", "_")
    z = np.load(d / "dense.npz")
    m = json.loads((d / "meta.json").read_text())
    nr = int(z["n_real"])
    cams = np.load(MSR / episode / "cameras.npz")
    world_c2w0 = cams["c2w"][0].astype(np.float64)
    c2m = np.array(m["c2w"], dtype=np.float64)                # scene frame (= camera 0), metres
    c2w = world_c2w0[None] @ c2m
    assert np.abs(c2w[:4] - cams["c2w"][:4]).max() < 2e-3, np.abs(c2w[:4] - cams["c2w"][:4]).max()
    c2w = cams["c2w"][:4].astype(np.float64)                  # exact world cameras (same rig)
    K = np.array(m["K"], dtype=np.float64)
    H, W = m["H"], m["W"]
    if kind == "roll":
        v = _vid(d / "rgb_pred.mp4")[:nr]
        dep = z["z_roll"][:, :nr].astype(np.float32)
    else:
        v = _vid(d / "rgb.mp4")[:nr]
        dep = z["z_gt" if kind == "gt" else "z_est"][:, :nr].astype(np.float32)
        if kind == "gt":
            dep = dep * z["valid"]
    if kind != "gt":       # same GT-free workspace rule as the training data: camera-z <= FAR_REL scene units
        dep = np.where(dep <= FAR_REL * float(m["scene_scale_m"]), dep, 0).astype(np.float32)
    rgb = v.reshape(nr, H, 4, W, 3).transpose(0, 2, 1, 3, 4)
    dep = dep.transpose(1, 0, 2, 3)                            # T,V,H,W
    fps = 1.0
    steps = np.round(np.arange(nr) / fps * 20).astype(int)
    return Obs(f"{episode.split('/')[0]}/{kind}{'' if kind == 'gt' else '_' + run}", np.ascontiguousarray(rgb),
               np.ascontiguousarray(dep), K, c2w, steps, None,
               {"scene": "standard", "kind": kind, "run": run, "valid_gt": z["valid"].transpose(1, 0, 2, 3)})


def load_dense_drawer(kind, run="B"):
    """world-model outputs on the drawer scene (out of distribution): same export format, 5 keyframes."""
    d = GEO / "eval_drawer" / run / "drawer"
    z = np.load(d / "dense.npz")
    m = json.loads((d / "meta.json").read_text())
    nr = int(z["n_real"])
    c2w = np.array(m["c2w_world"], dtype=np.float64)
    K = np.array(m["K"], dtype=np.float64)
    H, W = m["H"], m["W"]
    if kind == "roll":
        v = _vid(d / "rgb_pred.mp4")[:nr]
        dep = z["z_roll"][:, :nr].astype(np.float32)
    else:
        v = _vid(d / "rgb.mp4")[:nr]
        dep = z["z_gt" if kind == "gt" else "z_est"][:, :nr].astype(np.float32)
        if kind == "gt":
            dep = dep * z["valid"][:, :nr]
    if kind != "gt":
        dep = np.where(dep <= FAR_REL * float(m["scene_scale_m"]), dep, 0).astype(np.float32)
    rgb = v.reshape(nr, H, 4, W, 3).transpose(0, 2, 1, 3, 4)
    dep = dep.transpose(1, 0, 2, 3)
    steps = np.array([7, 179, 214, 252, 276])[:nr]
    return Obs(f"drawer/{kind}{'' if kind == 'gt' else '_' + run}", np.ascontiguousarray(rgb), np.ascontiguousarray(dep),
               K, c2w, steps, None, {"scene": "drawer", "kind": kind, "run": run})


REAL = Path("/n/lab_storage/ydu_lab/Lab/akiruga/articulated_recon/real")


def load_real(episode=None):
    import os
    episode = episode or os.environ.get("REAL_EPISODE", "task_0004_user_0010_scene_0007_cfg_0005")
    """real RH20T (cfg5 Franka) episode: 5 calibrated RealSense views, sensor depth, world = robot base frame."""
    z = np.load(REAL / episode / "obs.npz")
    c2w = np.linalg.inv(z["w2c"].astype(np.float64))
    q = z["q_gt"].astype(np.float64)
    fw = z["gripper_width_mm"].astype(np.float64) / 2000 if "gripper_width_mm" in z.files else np.zeros(len(q))
    gt = [{"robot_base": np.eye(4), "q": np.concatenate([q[t, :7], [fw[t], fw[t]]]), "actors": {}} for t in range(len(q))]
    return Obs(f"real/{episode}", z["rgb"], z["depth_m"].astype(np.float32), z["K"].astype(np.float64), c2w,
               np.asarray(z["frame_idx"]).reshape(len(q), -1)[:, 0] if np.ndim(z["frame_idx"]) > 1 else np.asarray(z["frame_idx"]),
               None, {"scene": "real", "kind": "sensor", "gt_states": gt, "episode": episode})


def backproject(depth, K, c2w, stride=1):
    """depth [H,W] -> world pts [N,3], pixel idx [N,2]"""
    H, W = depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[v, u]
    ok = z > 0
    u, v, z = u[ok], v[ok], z[ok]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z, np.ones_like(z)], 1)
    return (pc @ c2w.T)[:, :3], np.stack([v, u], 1)


H5 = {}


def gt_states(episode, steps):
    """ManiSkill demo states at the given control steps: robot qpos (9), base pose, actor poses (p, wxyz)."""
    import h5py
    task, ep = episode.split("/")
    eid = int(ep.split("-")[1])
    if task not in H5:
        H5[task] = h5py.File(MSR / "demos" / task / "trajectory.h5", "r")
    g = H5[task][f"traj_{eid}"]["env_states"]
    art = g["articulations"]
    rk = [k for k in art if k.startswith("panda")][0]
    A = art[rk][:]
    n = len(A)
    st = np.clip(np.asarray(steps), 0, n - 1)
    out = dict(q=A[st, 13:22], base=A[st, :7], actors={k: g["actors"][k][:][st, :7] for k in g["actors"]
                                                       if not k.startswith("table")}, steps=st)
    return out


def load_roll(d, use="roll"):
    """world-model rollout exported by scripts/geo/rollout_batch.py: predicted RGB + predicted depth, 4 views,
    1 fps.  use='roll' (prediction) or 'gt' (the real episode's frames + sim depth, same cameras)."""
    d = Path(d)
    z = np.load(d / "dense.npz")
    m = json.loads((d / "meta.json").read_text())
    nr = int(z["n_real"])
    cams = np.load(MSR / m["episode"] / "cameras.npz")
    c2w = cams["c2w"][:4].astype(np.float64)
    c2m = np.array(m["c2w"], dtype=np.float64)
    c2w_chk = cams["c2w"][0].astype(np.float64)[None] @ c2m
    assert np.abs(c2w_chk - c2w).max() < 2e-3
    K = np.array(m["K"], dtype=np.float64)
    if use == "roll":
        rgb = z["rgb_pred"]
        dep = z["z_roll"][:, :nr].astype(np.float32)
        dep = np.where(dep <= FAR_REL * float(m["scene_scale_m"]), dep, 0).astype(np.float32)
    else:
        rgb = z["rgb"]
        dep = (z["z_gt"][:, :nr].astype(np.float32) * z["valid"])
    rgb = rgb.transpose(1, 0, 2, 3, 4)                   # T,V,H,W,3
    dep = dep.transpose(1, 0, 2, 3)
    steps = np.round(np.arange(nr) * 20).astype(int)
    meta = {"scene": "standard", "kind": use, "episode": m["episode"], "seed": m.get("seed", 0),
            "gt": gt_states(m["episode"], steps), "task": m["episode"].split("/")[0]}
    return Obs(f"{m['episode']}/{use}", np.ascontiguousarray(rgb), np.ascontiguousarray(dep), K, c2w, steps, None, meta)
