# SPDX-License-Identifier: Apache-2.0
"""RH20T robot actions (end-effector trajectory) and their per-view rendering.

Source: the official low-dim release (``scripts/rh20t/download_lowdim.sbatch``)::

    <lowdim_root>/RH20T_cfgN/<episode>/transformed/{tcp.npy, gripper.npy}

``tcp.npy`` is ``{serial: [{"timestamp": ms, "tcp": [x, y, z, qw, qx, qy, qz], ...}, ...]}``:
the tool centre point already expressed in that camera's frame (metres), sampled at the
camera's frame times. The quaternion is scalar-first; the tool +z axis points from the wrist
to the fingertips (both checked by projecting into the colour frames, 2026-10-30).
``gripper.npy`` is ``{serial: {ms: {"gripper_info": [width, ...], ...}}}`` with width on a
0..~100 scale (cfg 4 logs a constant 84). Some poses are logged as ``None`` (a few per
episode, or all of them for some episodes) and some cameras have an empty gripper log; those
entries are dropped.

The executed end-effector trajectory stands in for the commanded actions. It covers every
clip frame, including future ones: it is the plan the world model is conditioned on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MAX_ACTION_DT_MS = 100  # nearest logged pose must be this close to the frame time
WRIST_OFFSET_M = 0.1  # second keypoint: this far from the TCP back along the tool axis
GRIPPER_SCALE = 100.0  # width / GRIPPER_SCALE -> openness in ~[0, 1]
DEPTH_SCALE_M = 2.0  # TCP depth / DEPTH_SCALE_M -> ~[0, 1]
BLOB_SIGMA = 1.0  # Gaussian blob sigma in map pixels
ACTION_MAP_CHANNELS = 4  # per frame: TCP blob, wrist blob, TCP blob x openness, TCP blob x depth


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """[..., 4] scalar-first unit quaternions -> [..., 3, 3] rotation matrices."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def load_episode_actions(episode_dir: str | Path) -> tuple[dict, dict]:
    """-> (tcp {serial: (stamps [N] int64, poses [N, 7])}, gripper {serial: (stamps [M], width [M])})."""
    d = Path(episode_dir) / "transformed"
    tcp_raw = np.load(d / "tcp.npy", allow_pickle=True).item()
    tcp = {}
    for s, entries in tcp_raw.items():
        valid = [e for e in entries if e["tcp"] is not None and np.asarray(e["tcp"]).shape == (7, )]
        if valid:
            tcp[s] = (np.array([int(e["timestamp"]) for e in valid], dtype=np.int64),
                      np.array([e["tcp"] for e in valid], dtype=np.float64))
    grip_raw = np.load(d / "gripper.npy", allow_pickle=True).item()
    gripper = {}
    for s, v in grip_raw.items():
        if not v:
            continue
        stamps = np.array(sorted(v), dtype=np.int64)
        widths = np.array([float(np.asarray(v[t]).item()["gripper_info"][0]) for t in stamps])
        gripper[s] = (stamps, widths)
    return tcp, gripper


def nearest_index(stamps: np.ndarray, query: np.ndarray, max_dt_ms: float) -> np.ndarray:
    """Index of the nearest of sorted ``stamps`` for each query time; raises if any is too far."""
    order = np.argsort(stamps)
    s = stamps[order]
    pos = np.clip(np.searchsorted(s, query), 1, len(s) - 1)
    left, right = s[pos - 1], s[pos]
    pick = np.where(np.abs(query - left) <= np.abs(right - query), pos - 1, pos)
    if np.abs(s[pick] - query).max() > max_dt_ms:
        raise ValueError(f"no logged action within {max_dt_ms} ms of a frame")
    return order[pick]


def view_actions(tcp: dict, gripper: dict, serial: str, frame_stamps: np.ndarray) -> np.ndarray:
    """-> [T, 8]: TCP xyz (camera frame, m), quaternion wxyz, gripper openness, for one view."""
    if serial not in tcp:
        raise ValueError(f"no TCP track for camera {serial}")
    stamps, poses = tcp[serial]
    pose = poses[nearest_index(stamps, frame_stamps, MAX_ACTION_DT_MS)]
    if not gripper:
        raise ValueError("no gripper log")
    g_stamps, widths = gripper.get(serial) or next(iter(gripper.values()))
    # the gripper is logged on its own clock: nearest sample, no tolerance (it changes slowly)
    width = widths[nearest_index(g_stamps, frame_stamps, np.inf)]
    return np.concatenate([pose, (width / GRIPPER_SCALE)[:, None]], axis=1)


def render_action_maps(actions: torch.Tensor, K: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """actions [T, 8] (view_actions) for one camera, K [3, 3] at (height, width)
    -> [T, ACTION_MAP_CHANNELS, height, width]. Keypoints behind the camera or far outside
    the image leave their channels empty."""
    xyz, quat, open_ = actions[:, :3], actions[:, 3:7], actions[:, 7]
    rot = torch.from_numpy(quat_wxyz_to_matrix(quat.double().numpy())).to(actions.dtype)
    wrist = xyz - WRIST_OFFSET_M * rot[:, :, 2]
    ys = torch.arange(height, dtype=actions.dtype) + 0.5
    xs = torch.arange(width, dtype=actions.dtype) + 0.5

    def blob(p: torch.Tensor) -> torch.Tensor:  # [T, 3] -> [T, H, W]
        z = p[:, 2]
        front = z > 1e-3
        u = K[0, 0] * p[:, 0] / z.clamp(min=1e-3) + K[0, 2]
        v = K[1, 1] * p[:, 1] / z.clamp(min=1e-3) + K[1, 2]
        g = torch.exp(-((ys[None, :, None] - v[:, None, None])**2 + (xs[None, None, :] - u[:, None, None])**2) /
                      (2 * BLOB_SIGMA**2))
        return g * front[:, None, None].to(g.dtype)

    tcp_blob = blob(xyz)
    depth = (xyz[:, 2] / DEPTH_SCALE_M).clamp(0, 1)
    return torch.stack(
        [tcp_blob, blob(wrist), tcp_blob * open_[:, None, None], tcp_blob * depth[:, None, None]], dim=1)
