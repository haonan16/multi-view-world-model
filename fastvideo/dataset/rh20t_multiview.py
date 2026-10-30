# SPDX-License-Identifier: Apache-2.0
"""RH20T multi-view dataset: V time-synchronised, calibrated RGB views per sample.

Ported from akiruga's ego_world_model (``datasets/rh20t.py``, ``datasets/_multiview.py``).

Source: the authors' official 320x180 release (HF ``tonyfanggalaxies/RH20T``)::

    <data_root>/RH20T_cfgN/task_TTTT_user_UUUU_scene_SSSS_cfg_CCCC/
        metadata.json                      {"calib": <ts>, "rating": 1-9, ...}
        cam_<serial>/color/color.mp4       320x180, frames at IRREGULAR times
        cam_<serial>/color/timestamps.npy  int64 ms, one per video frame
    <calib_root>/RH20T_cfgN/calib/<ts>/{intrinsics,extrinsics}.npy
        dicts keyed by serial; K is 3x4 at 1280x720; extrinsics is (1, 4, 4)
        world->camera (OpenCV axes), world origin at the calibration marker on the
        table. Some entries are None (camera not calibrated in that snapshot).

Timing: mp4 headers claim 25 fps but frames are irregular (~7-11 Hz, per-camera drops
up to >1 s), so views are synchronised by TIMESTAMP: a uniform ``fps`` grid is laid over
the cameras' common span and each view takes its nearest frame. A view is eligible only
if every grid point has a frame within ``max_sync_err_ms``.

View choice: in-hand cameras are excluded (they move with the gripper, so the static
calibration is wrong). Of the synchronisable static cameras, the ``n_views`` facing the
workspace most directly (smallest angle between optical axis and the ray to the
calibration marker) are used, in random order.

Cameras: rigs differ across the 7 cfgs and cameras were moved between calibrations, so
per-view cameras are returned (and per-view Plucker raymaps). Poses are relative to view
0 (c2w_0 = identity), in metres. Cameras are static within an episode, so cameras and
raymaps carry no time axis.

Sample keys:
    videos      [V, T, 3, H, W]  float in [-1, 1]
    raymaps     [V, 6, Hr, Wr]   unit ray direction (0:3) + Plucker moment o x d (3:6)
    intrinsics  [V, 3, 3]        pixel units at (H, W)
    c2w         [V, 4, 4]        OpenCV camera-to-world, view 0 = identity
    timestamps  [V, T] int64 ms; sync_err_ms [V, T]
    episode, cfg_id, serials (";"-joined)
    context     [V] bool         views observed at frame 0 (all True unless n_context_views)
    pc_render   [V, 3, H, W]     (point_cloud only) frame-0 RGB of the context views' fused
                                 depth point cloud rendered into each view, [-1, 1], 0 where empty
    pc_mask     [V, 1, H, W]     (point_cloud only) 1 where pc_render has a point
    depth       [V, T, H, W]     (return_depth only) sensor depth of every frame in metres,
                                 cropped like the colour, 0 = invalid. Evaluation only: it
                                 includes future frames, so it must not condition a model.
    actions     [V, T, 8]        (actions only) end-effector xyz in each view's camera frame (m),
                                 quaternion wxyz, gripper openness; every frame (the plan)
    action_maps [V, T, 4, Hr, Wr] (actions only) actions rendered into each view at raymap
                                 resolution (rh20t_actions.render_action_maps)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from fastvideo.dataset.rh20t_actions import load_episode_actions, render_action_maps, view_actions
from fastvideo.dataset.rh20t_depth import clean_depth, read_depth_frames, render_points, unproject
from fastvideo.logger import init_logger

logger = init_logger(__name__)

CALIB_H, CALIB_W = 720, 1280  # resolution the calib intrinsics refer to

# In-hand (wrist) cameras per cfg, from rh20t_api configs/configs.json ("in_hand").
IN_HAND = {
    1: {"043322070878"},
    2: {"104422070042"},
    3: {"045322071843"},
    4: {"045322071843"},
    5: {"104422070042", "135122079702"},
    6: {"135122075425", "135122070361"},
    7: {"135122075425", "135122070361"},
}
DEPTH_COMPLETE_MARKER = ".complete"  # written by scripts/rh20t/download_depth.sbatch
MAX_RESAMPLE_ATTEMPTS = 20
MAX_WINDOW_ATTEMPTS = 8
SCAN_THREADS = 32


@dataclass(frozen=True)
class RH20TMultiViewConfig:
    data_root: str
    calib_root: str
    cfgs: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    min_rating: int | None = None  # rating 1 = failed task; failures are still valid dynamics
    test_percentage: float = 0.02  # held out by a stable hash of the episode name
    n_views: int = 4
    n_frames: int = 13  # 1 + 4k -> k + 1 latent frames with the 4x temporal VAE
    fps: float = 3.0  # 13 frames at 3 Hz span 4 s; at 10 Hz RH20T barely moves
    max_sync_err_ms: float = 60.0
    height: int = 160  # centre crop of 180x320, divisible by 32
    width: int = 320
    raymap_height: int = 20
    raymap_width: int = 40
    records_cache_dir: str | None = None  # JSON cache of the record walk
    # Unseen-view task: only the first n_context_views (random) views are observed at
    # frame 0; the rest must be generated. None = every view is observed.
    n_context_views: int | None = None
    # <depth_root>/RH20T_cfgN/<episode>/cam_<serial>/depth.mp4 (see rh20t_depth.py). When set,
    # only cameras with depth are eligible (also for RGB-only runs, to keep the episode set).
    depth_root: str | None = None
    # Render the frame-0 point cloud of the context views into every view (needs depth_root
    # and n_context_views).
    point_cloud: bool = False
    # Also return every view's depth at every clip frame (needs depth_root). For evaluation
    # (e.g. cross-view reprojection); it includes future frames.
    return_depth: bool = False
    # Robot end-effector trajectory of every clip frame, raw and rendered into each view
    # (rh20t_actions.py). Needs lowdim_root: <lowdim_root>/RH20T_cfgN/<episode>/transformed/.
    actions: bool = False
    lowdim_root: str | None = None

    def __post_init__(self) -> None:
        if self.n_views < 1:
            raise ValueError(f"n_views must be >= 1, got {self.n_views}")
        if self.n_frames < 2 or (self.n_frames - 1) % 4:
            raise ValueError(f"n_frames must be 1 + 4k (k >= 1) for the Wan VAE, got {self.n_frames}")
        if self.fps <= 0 or self.max_sync_err_ms <= 0:
            raise ValueError("fps and max_sync_err_ms must be positive")
        if not 0.0 <= self.test_percentage < 1.0:
            raise ValueError(f"test_percentage must be in [0, 1), got {self.test_percentage}")
        if self.n_context_views is not None and not 1 <= self.n_context_views <= self.n_views:
            raise ValueError(f"n_context_views must be in [1, n_views={self.n_views}], got {self.n_context_views}")
        if self.point_cloud and (self.depth_root is None or self.n_context_views is None):
            raise ValueError("point_cloud=True needs depth_root and n_context_views")
        if self.return_depth and self.depth_root is None:
            raise ValueError("return_depth=True needs depth_root")
        if self.actions and self.lowdim_root is None:
            raise ValueError("actions=True needs lowdim_root")


# ---------------------------------------------------------------------------
# Geometry / packing helpers
# ---------------------------------------------------------------------------


def grid_for(n_views: int) -> tuple[int, int]:
    """Near-square (rows, cols) grid for n_views tiles, rows <= cols."""
    rows = int(math.floor(math.sqrt(n_views)))
    while n_views % rows:
        rows -= 1
    return rows, n_views // rows


def tile_views(x: torch.Tensor, grid: tuple[int, int] | None = None) -> torch.Tensor:
    """Pack views into one canvas of the SAME spatial size as a single view.

    x: [..., V, T, C, H, W] -> [..., T, C, H, W]. Each view is area-downsampled to
    (H / rows, W / cols); H and W must be divisible by rows / cols.
    """
    *lead, V, T, C, H, W = x.shape
    rows, cols = grid or grid_for(V)
    if rows * cols != V:
        raise ValueError(f"grid {rows}x{cols} does not hold {V} views")
    if H % rows or W % cols:
        raise ValueError(f"{H}x{W} not divisible by grid {rows}x{cols}")
    h, w = H // rows, W // cols
    y = F.interpolate(x.reshape(-1, C, H, W).float(), size=(h, w), mode="area").to(x.dtype)
    y = y.reshape(-1, rows, cols, T, C, h, w).permute(0, 3, 4, 1, 5, 2, 6)
    return y.reshape(*lead, T, C, H, W)


def untile_views(y: torch.Tensor, n_views: int, grid: tuple[int, int] | None = None) -> torch.Tensor:
    """Inverse layout of `tile_views` (no upsampling): [..., T, C, H, W] -> [..., V, T, C, H/rows, W/cols]."""
    *lead, T, C, H, W = y.shape
    rows, cols = grid or grid_for(n_views)
    h, w = H // rows, W // cols
    z = y.reshape(-1, T, C, rows, h, cols, w).permute(0, 3, 5, 1, 2, 4, 6)
    return z.reshape(*lead, n_views, T, C, h, w)


def camera_to_raymap(K: torch.Tensor, c2w: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Plucker raymap [B, 6, H, W]: unit world ray direction d and moment o x d.

    K: [B, 3, 3] pixel intrinsics at (height, width); c2w: [B, 4, 4] OpenCV camera-to-world.
    """
    B = K.shape[0]
    ys, xs = torch.meshgrid(torch.arange(height, dtype=K.dtype) + 0.5,
                            torch.arange(width, dtype=K.dtype) + 0.5,
                            indexing="ij")
    fx, fy, cx, cy = (K[:, 0, 0, None, None], K[:, 1, 1, None, None], K[:, 0, 2, None, None], K[:, 1, 2, None, None])
    d_cam = torch.stack([(xs - cx) / fx, (ys - cy) / fy, torch.ones(B, height, width, dtype=K.dtype)], dim=1)
    d = torch.einsum("bij,bjhw->bihw", c2w[:, :3, :3], d_cam)
    d = F.normalize(d, dim=1)
    o = c2w[:, :3, 3, None, None].expand(-1, -1, height, width)
    return torch.cat([d, torch.cross(o, d, dim=1)], dim=1)


def center_crop_box(h_src: int, w_src: int, h_dst: int, w_dst: int) -> tuple[int, int, int, int]:
    """Largest centred (top, bottom, left, right) crop of (h_src, w_src) with aspect w_dst / h_dst."""
    target = w_dst / h_dst
    if w_src / h_src > target:
        new_h, new_w = h_src, int(round(h_src * target))
    else:
        new_h, new_w = int(round(w_src / target)), w_src
    new_h, new_w = min(new_h, h_src), min(new_w, w_src)
    top, left = (h_src - new_h) // 2, (w_src - new_w) // 2
    return top, top + new_h, left, left + new_w


def crop_resize_depth(depth: torch.Tensor, box: tuple[int, int, int, int], h_src: int, w_src: int, h_dst: int,
                      w_dst: int) -> torch.Tensor:
    """Crop [T, dh, dw] depth (metres) with a colour-frame ``box`` given at (h_src, w_src), then
    clean it and resize to (h_dst, w_dst) with nearest neighbour (no mixing across edges)."""
    dh, dw = depth.shape[-2:]
    top, bottom, left, right = box
    sy, sx = dh / h_src, dw / w_src
    d = depth[..., int(round(top * sy)):int(round(bottom * sy)), int(round(left * sx)):int(round(right * sx))]
    return F.interpolate(clean_depth(d)[:, None], size=(h_dst, w_dst), mode="nearest")[:, 0]


def crop_resize_intrinsics(K: np.ndarray, box: tuple[int, int, int, int], h_dst: int, w_dst: int) -> np.ndarray:
    """Intrinsics after centre-crop to `box` then resize to (h_dst, w_dst)."""
    top, bottom, left, right = box
    sx, sy = w_dst / (right - left), h_dst / (bottom - top)
    out = np.array(K, dtype=np.float64, copy=True)
    out[0, 2] = (out[0, 2] - left) * sx
    out[1, 2] = (out[1, 2] - top) * sy
    out[0, 0] *= sx
    out[1, 1] *= sy
    return out


def marker_angle_deg(w2c: np.ndarray) -> float:
    """Angle between a camera's optical axis and its ray to the calibration marker (world
    origin, on the table in the workspace). Small = faces the work."""
    p = w2c[:3, 3]  # marker origin in camera coordinates
    return float(np.degrees(np.arctan2(np.hypot(p[0], p[1]), p[2])))


def nearest_frames(stamps: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest timestamp to every grid point, and its |error| in ms."""
    j = np.clip(np.searchsorted(stamps, grid), 1, len(stamps) - 1)
    j = np.where(np.abs(stamps[j - 1] - grid) <= np.abs(stamps[j] - grid), j - 1, j)
    return j, np.abs(stamps[j] - grid)


def read_frames(path: str | Path, indices: np.ndarray) -> torch.Tensor:
    """Decode the requested frame indices of an mp4 -> uint8 [T, 3, H, W]."""
    wanted = {int(i) for i in indices}
    last = max(wanted)
    frames: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1  # many dataloader workers; tiny 320x180 frames
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                frames[i] = frame.to_ndarray(format="rgb24")
            if i >= last:
                break
    missing = wanted - frames.keys()
    if missing:
        raise ValueError(f"{path}: video has fewer frames than timestamps (missing {sorted(missing)[:5]})")
    return torch.from_numpy(np.stack([frames[int(i)] for i in indices])).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RH20TMultiViewDataset(Dataset):
    """One record per robot episode with >= n_views calibrated static cameras."""

    def __init__(self, cfg: RH20TMultiViewConfig, split: str = "train") -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.cfg = cfg
        self.split = split
        self.data_root = Path(cfg.data_root)
        self.calib_root = Path(cfg.calib_root)
        self.depth_root = Path(cfg.depth_root) if cfg.depth_root else None
        for root in (self.data_root, self.calib_root, *([self.depth_root] if self.depth_root else [])):
            if not root.is_dir():
                raise FileNotFoundError(f"RH20T root does not exist: {root}")
        self._calib_cache: dict[tuple[int, int], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        self._calib_lock = threading.Lock()
        self._draws = 0
        self.records = self._load_records()

    def __len__(self) -> int:
        return len(self.records)

    # ---- records ----

    def _episode_dirs(self) -> list[Path]:
        out: list[Path] = []
        for c in self.cfg.cfgs:
            d = self.data_root / f"RH20T_cfg{c}"
            if d.is_dir():
                out += sorted(p for p in d.iterdir() if p.name.startswith("task_") and "_human" not in p.name)
        return out

    def load_calib(self, cfg_id: int, ts: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """serial -> (K 3x3 at 1280x720, w2c 4x4) for one calibration snapshot."""
        key = (cfg_id, ts)
        with self._calib_lock:
            if key in self._calib_cache:
                return self._calib_cache[key]
        d = self.calib_root / f"RH20T_cfg{cfg_id}" / "calib" / str(ts)
        calib: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        try:
            ex = np.load(d / "extrinsics.npy", allow_pickle=True).item()
            Ks = np.load(d / "intrinsics.npy", allow_pickle=True).item()
        except (FileNotFoundError, OSError):
            ex, Ks = {}, {}
        for serial, m in ex.items():
            try:
                w2c = np.asarray(m, dtype=np.float64).reshape(-1, 4, 4)[0]
                K = np.asarray(Ks[serial], dtype=np.float64)[:3, :3]
            except (TypeError, ValueError, KeyError):
                continue  # None entry = camera not calibrated in this snapshot
            calib[str(serial)] = (K, w2c)
        with self._calib_lock:
            self._calib_cache[key] = calib
        return calib

    def _scan_episode(self, ep: Path) -> tuple[dict[str, Any] | None, str | None]:
        try:
            meta = json.loads((ep / "metadata.json").read_text())
        except (FileNotFoundError, ValueError):
            return None, "no_meta"
        if self.cfg.min_rating is not None and meta.get("rating", 0) < self.cfg.min_rating:
            return None, "rating"
        cfg_id = int(ep.parent.name.replace("RH20T_cfg", ""))
        calib = self.load_calib(cfg_id, int(meta["calib"]))
        cams = [
            p.name[4:] for p in sorted(ep.glob("cam_*"))
            if p.name[4:] in calib and p.name[4:] not in IN_HAND.get(cfg_id, ())
            and (p / "color" / "color.mp4").exists() and (p / "color" / "timestamps.npy").exists()
            and (self.depth_root is None or self._depth_path(ep, p.name[4:], cfg_id).exists())
        ]
        if len(cams) < self.cfg.n_views:
            return None, "few_cams"
        return {
            "episode": ep.name,
            "ep_dir": str(ep),
            "cfg_id": cfg_id,
            "calib_ts": int(meta["calib"]),
            "rating": int(meta.get("rating", -1)),
            "cams": cams,
        }, None

    def _depth_path(self, ep: Path | str, serial: str, cfg_id: int) -> Path:
        assert self.depth_root is not None
        return self.depth_root / f"RH20T_cfg{cfg_id}" / Path(ep).name / f"cam_{serial}" / "depth.mp4"

    def _cache_path(self) -> Path | None:
        if not self.cfg.records_cache_dir:
            return None
        keys = ("data_root", "calib_root", "cfgs", "min_rating", "n_views", "depth_root")
        blob = json.dumps({k: asdict(self.cfg)[k] for k in keys}, sort_keys=True, default=str)
        return Path(self.cfg.records_cache_dir) / f"rh20t_mv_records_{hashlib.sha1(blob.encode()).hexdigest()[:12]}.json"

    def _depth_complete(self) -> bool:
        """True when every requested cfg's depth download finished (download_depth.sbatch
        touches <depth_root>/RH20T_cfgN/.complete). A record list built from a partial
        depth tree must not be cached: it would silently pin the partial episode set."""
        if self.depth_root is None:
            return True
        return all((self.depth_root / f"RH20T_cfg{c}" / DEPTH_COMPLETE_MARKER).exists() for c in self.cfg.cfgs)

    def _all_records(self) -> list[dict[str, Any]]:
        cache = self._cache_path()
        if cache is not None and not self._depth_complete():
            logger.warning("RH20T: depth under %s is incomplete (no %s for every cfg); not using the record cache",
                           self.depth_root, DEPTH_COMPLETE_MARKER)
            cache = None
        if cache is not None and cache.exists():
            return json.loads(cache.read_text())
        skipped = {"no_meta": 0, "rating": 0, "few_cams": 0}
        records = []
        with ThreadPoolExecutor(SCAN_THREADS) as pool:
            for rec, why in pool.map(self._scan_episode, self._episode_dirs()):
                if rec is not None:
                    records.append(rec)
                elif why:
                    skipped[why] += 1
        logger.info("RH20T: %d usable episodes under %s (skipped %s)", len(records), self.data_root, skipped)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(records))
            tmp.replace(cache)
        return records

    def _load_records(self) -> list[dict[str, Any]]:
        threshold = self.cfg.test_percentage * 10000
        want_val = self.split == "val"
        records = [
            r for r in self._all_records() if ((zlib.crc32(r["episode"].encode()) % 10000) < threshold) == want_val
        ]
        if not records:
            raise FileNotFoundError(f"No usable RH20T episodes for split {self.split!r} under {self.data_root}")
        logger.info("RH20T[%s]: %d episodes", self.split, len(records))
        return records

    # ---- sampling ----

    def _rng(self, idx: int, attempt: int = 0) -> np.random.Generator:
        # Train: fresh randomness per draw, distinct per worker (torch reseeds each worker;
        # numpy's global RNG would be duplicated across workers). Val: fixed per (index, attempt);
        # the attempt keeps a failed val draw from retrying the same plan (and resample target).
        self._draws += 1
        if self.split == "train":
            return np.random.default_rng([torch.initial_seed() % 2**32, idx, self._draws])
        return np.random.default_rng(idx if attempt == 0 else [idx, attempt])

    def plan_clip(self, rec: dict[str, Any], rng: np.random.Generator):
        """Choose a uniform time grid and n_views cameras that are all within max_sync_err_ms of it.

        Returns (serials, {serial: (frame_indices, frame_stamps, abs_err)}) or None."""
        stamps = {}
        for s in rec["cams"]:
            t = np.load(Path(rec["ep_dir"]) / f"cam_{s}" / "color" / "timestamps.npy")
            if len(t) >= 2:
                stamps[s] = t.astype(np.int64)
        if len(stamps) < self.cfg.n_views:
            return None
        step = 1000.0 / self.cfg.fps
        duration = step * (self.cfg.n_frames - 1)
        tol = self.cfg.max_sync_err_ms
        lo, hi = min(t[0] for t in stamps.values()), max(t[-1] for t in stamps.values())
        if hi - lo < duration:
            return None
        # the window only has to be covered by the views we pick, not by every camera
        for _ in range(MAX_WINDOW_ATTEMPTS):
            grid = rng.uniform(lo, hi - duration) + step * np.arange(self.cfg.n_frames)
            ok = {}
            for s, t in stamps.items():
                if t[0] > grid[0] + tol or t[-1] < grid[-1] - tol:
                    continue
                j, err = nearest_frames(t, grid)
                if err.max() <= tol:
                    ok[s] = (j, t[j], err)
            if len(ok) >= self.cfg.n_views:
                calib = self.load_calib(rec["cfg_id"], rec["calib_ts"])
                front = sorted(ok, key=lambda s: marker_angle_deg(calib[s][1]))[:self.cfg.n_views]
                return [str(s) for s in rng.permutation(front)], ok
        return None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Episodes can fail (dropped frames, short overlap); resample a new one rather than raise.
        for attempt in range(MAX_RESAMPLE_ATTEMPTS):
            rec = self.records[idx]
            rng = self._rng(idx, attempt)
            try:
                plan = self.plan_clip(rec, rng)
                if plan is not None:
                    return self.build_sample(rec, *plan)
            except Exception as exc:  # corrupt/missing media must not kill the training job
                logger.warning("RH20T: skipping %s: %r", rec["episode"], exc)
            idx = int(rng.integers(len(self.records)))
        raise RuntimeError(f"RH20T: {MAX_RESAMPLE_ATTEMPTS} consecutive episodes had no synchronisable window")

    def build_sample(self, rec: dict[str, Any], serials: list[str], ok: dict[str, Any]) -> dict[str, Any]:
        calib = self.load_calib(rec["cfg_id"], rec["calib_ts"])
        H, W = self.cfg.height, self.cfg.width
        frames, Ks, w2cs, stamps, errs, first, depths = [], [], [], [], [], [], []
        for s in serials:
            j, t, err = ok[s]
            x = read_frames(Path(rec["ep_dir"]) / f"cam_{s}" / "color" / "color.mp4", j)  # T, 3, h, w uint8
            first.append(x[0])
            h, w = x.shape[-2:]
            K, w2c = calib[s]
            K = K.copy()
            K[0] *= w / CALIB_W
            K[1] *= h / CALIB_H
            top, bottom, left, right = box = center_crop_box(h, w, H, W)
            if self.cfg.return_depth:
                d = read_depth_frames(self._depth_path(rec["ep_dir"], s, rec["cfg_id"]), j)
                depths.append(crop_resize_depth(torch.from_numpy(d), box, h, w, H, W))
            x = x[..., top:bottom, left:right].float()
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False, antialias=True)
            frames.append(x / 127.5 - 1.0)
            Ks.append(crop_resize_intrinsics(K, box, H, W))
            w2cs.append(w2c)
            stamps.append(t)
            errs.append(err)
        K = torch.from_numpy(np.stack(Ks)).float()
        c2w = np.linalg.inv(np.stack(w2cs))
        c2w_rel = torch.from_numpy(np.linalg.inv(c2w[0])[None] @ c2w).float()  # view 0 = identity
        Hr, Wr = self.cfg.raymap_height, self.cfg.raymap_width
        K_ray = K.clone()
        K_ray[:, 0] *= Wr / W
        K_ray[:, 1] *= Hr / H
        n_ctx = self.cfg.n_context_views or len(serials)
        context = torch.arange(len(serials)) < n_ctx  # serials are already in random order
        extra = {"depth": torch.stack(depths)} if self.cfg.return_depth else {}
        if self.cfg.actions:
            extra |= self.episode_actions(rec, serials, np.stack(stamps), K_ray, Hr, Wr)
        if self.cfg.point_cloud:
            extra |= self.render_context_cloud(rec, serials[:n_ctx], ok, first[:n_ctx], c2w_rel, K)
        return {
            "videos": torch.stack(frames).clamp_(-1, 1),
            "raymaps": camera_to_raymap(K_ray, c2w_rel, Hr, Wr),
            "intrinsics": K,
            "c2w": c2w_rel,
            "timestamps": torch.from_numpy(np.stack(stamps)),
            "sync_err_ms": torch.from_numpy(np.stack(errs)).float(),
            "episode": rec["episode"],
            "cfg_id": rec["cfg_id"],
            "serials": ";".join(serials),
            "context": context,
            **extra,
        }

    def episode_actions(self, rec: dict[str, Any], serials: list[str], stamps: np.ndarray, K_ray: torch.Tensor,
                        Hr: int, Wr: int) -> dict[str, torch.Tensor]:
        """Per-view end-effector trajectory at the clip's frame times and its rendering."""
        assert self.cfg.lowdim_root is not None
        tcp, gripper = load_episode_actions(Path(self.cfg.lowdim_root) / f"RH20T_cfg{rec['cfg_id']}" / rec["episode"])
        actions = torch.from_numpy(np.stack([view_actions(tcp, gripper, s, t) for s, t in zip(serials, stamps)])).float()
        maps = torch.stack([render_action_maps(a, K_ray[v], Hr, Wr) for v, a in enumerate(actions)])
        return {"actions": actions, "action_maps": maps}

    def render_context_cloud(self, rec: dict[str, Any], ctx_serials: list[str], ok: dict[str, Any],
                             ctx_frames: list[torch.Tensor], c2w_rel: torch.Tensor,
                             K_views: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fuse the context views' frame-0 depth into one coloured cloud (view-0 frame, metres)
        and render it into every view at (H, W). Only frame 0 is used: it is the only
        observation available at prediction time."""
        calib = self.load_calib(rec["cfg_id"], rec["calib_ts"])
        pts, cols = [], []
        for v, (s, rgb) in enumerate(zip(ctx_serials, ctx_frames)):
            j = ok[s][0][:1]
            depth = torch.from_numpy(read_depth_frames(self._depth_path(rec["ep_dir"], s, rec["cfg_id"]), j)[0])
            dh, dw = depth.shape  # 360x640 in the release
            K = torch.from_numpy(calib[s][0].copy()).float()
            K[0] *= dw / CALIB_W
            K[1] *= dh / CALIB_H
            p, valid = unproject(clean_depth(depth), K, c2w_rel[v])
            color = F.interpolate(rgb[None].float(), size=(dh, dw), mode="bilinear", align_corners=False)
            pts.append(p)
            cols.append((color[0] / 127.5 - 1.0).permute(1, 2, 0)[valid])
        points, colors = torch.cat(pts), torch.cat(cols)
        H, W = self.cfg.height, self.cfg.width
        renders, masks = [], []
        for v in range(len(K_views)):
            img, _, mask = render_points(points, colors, K_views[v], torch.linalg.inv(c2w_rel[v]), H, W)
            renders.append(img.clamp(-1, 1))
            masks.append(mask[None].float())
        return {"pc_render": torch.stack(renders), "pc_mask": torch.stack(masks)}
