"""RH20T multiview dataset: V time-synchronised calibrated RGB views per sample.

Source: the authors' official 320x180 release (HF `tonyfanggalaxies/RH20T`), fetched
by scripts/rh20t/lowres_download.sbatch into

    <data_root>/RH20T_cfgN/task_TTTT_user_UUUU_scene_SSSS_cfg_CCCC/
        metadata.json                      {"calib": <ts>, "rating": 1-9, ...}
        cam_<serial>/color/color.mp4       320x180, frames at IRREGULAR times
        cam_<serial>/color/timestamps.npy  int64 ms, one per video frame
    <calib_root>/RH20T_cfgN/calib/<ts>/{intrinsics,extrinsics}.npy
        dicts keyed by serial; K is 3x4 at 1280x720; extrinsics is (1,4,4)
        world->camera (OpenCV axes) with the world origin at the calibration
        marker (verified by projecting the origin into calib/<ts>/imgs: it lands
        on the marker centre in every view). Some entries are None (uncalibrated).

Timing: the mp4 headers claim 25 fps but frames are irregular (~7-11 Hz, per-camera
drops up to >1 s), so views are synchronised by TIMESTAMP: a uniform `fps` grid is
laid over the cameras' common span and each view takes its nearest frame. A view is
eligible only if every grid point has a frame within `max_sync_err_ms`.

View choice: in-hand cameras are excluded (they move); of the synchronisable static
cameras, the n_views facing the workspace most directly (smallest angle between optical
axis and the ray to the calibration marker on the table) are used, in random order.

Human demonstrations (`<episode>_human`, same rigs and calibration, no robot; cfg1-6) are
included with `include_human` (default off). They are much shorter than robot episodes, so
ones shorter than the clip window (metadata start/end) are dropped at record time.
Human clips can be sampled at their own frame rate `human_fps` (default: `fps`): people do
the same tasks ~4x faster than the robot (median demo 9.6 s vs 42 s), so a higher rate both
keeps most demos long enough and brings motion per frame closer to the robot clips.
`human_weight` sets the fraction of TRAINING draws that come from human episodes (null =
their natural share); `eval_kinds` (robot | human | both, default robot) picks which
episodes the test split serves, so validation stays comparable with robot-only runs.
Records carry "human": bool; samples carry "is_human".

Cameras: rigs differ across the 7 cfgs and cameras were moved between calibrations,
so per-view cameras are returned (and per-view Plucker raymaps). Poses are relative
to view 0 (c2w_0 = identity), in metres, or with normalize_cam_scale in units of the mean
distance from view 0 to the other cameras. Cameras are static within an episode, so
cameras/raymaps carry no time axis.

Sample keys:
    videos      [V, T, 3, H, W]  float in [-1, 1]
    raymaps     [V, 6, Hr, Wr]   unit ray direction (0:3) + Plucker moment o x d (3:6)
    intrinsics  [V, 3, 3]        pixel units at (H, W)
    c2w         [V, 4, 4]        OpenCV camera-to-world, view 0 = identity
    timestamps  [V, T] int64 ms; sync_err_ms [V, T]
    w2c_marker  [V, 4, 4]        raw calib extrinsics (for visualisation)
    episode, cfg_id, serials (";"-joined), prompts (task id, e.g. "task_0017": the
    algorithm looks up its cached text embedding), cam_scale (translation divisor; 1 unless
    normalize_cam_scale)
"""
from __future__ import annotations

import json
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import decord
import numpy as np
import torch
import torch.nn.functional as F

from ._crop_utils import adjust_K_for_crop_resize, apply_spatial_crop, center_crop_box_for_aspect
from .aria_new import camera_to_raymap
from .video_base import VideoDataset

CALIB_H, CALIB_W = 720, 1280   # resolution the calib intrinsics refer to

# In-hand (wrist) cameras move with the gripper, so their static calibration is wrong
# during an episode -> never used. From rh20t_api configs/configs.json ("in_hand").
IN_HAND = {1: {"043322070878"}, 2: {"104422070042"}, 3: {"045322071843"},
           4: {"045322071843"}, 5: {"104422070042", "135122079702"},
           6: {"135122075425", "135122070361"}, 7: {"135122075425", "135122070361"}}


def marker_angle_deg(w2c: np.ndarray) -> float:
    """Angle between a camera's optical axis and its ray to the calibration marker
    (world origin), which sits on the table in the workspace. Small = faces the work."""
    p = w2c[:3, 3]                       # marker origin in camera coords
    return float(np.degrees(np.arctan2(np.hypot(p[0], p[1]), p[2])))


class RH20TMultiViewDataset(VideoDataset):
    """One record per robot (optionally also human-demo) episode with >= n_views
    calibrated cameras."""

    def __init__(self, cfg, split: str = "training"):
        self.n_views = int(cfg.n_views)
        self.calib_root = Path(cfg.calib_root)
        self.max_sync_err_ms = float(cfg.max_sync_err_ms)
        # divide translations so the mean distance from view 0 to the other cameras is 1
        self.normalize_cam_scale = bool(cfg.get("normalize_cam_scale", False))
        self._calib_cache: Dict[str, Dict[str, Any]] = {}
        self._calib_lock = threading.Lock()
        self._draws = 0
        self.include_human = bool(cfg.get("include_human", False))
        self.human_fps = float(cfg.get("human_fps", None) or cfg.fps)
        super().__init__(cfg, split)
        if split != "training":
            kinds = str(cfg.get("eval_kinds", "robot"))
            keep = {"robot": (False,), "human": (True,), "both": (False, True)}[kinds]
            self.records = [r for r in self.records if r.get("human", False) in keep]
        hw = cfg.get("human_weight", None)
        self.human_weight = float(hw) if (hw is not None and split == "training") else None
        self._kind_idx = {k: [i for i, r in enumerate(self.records) if r.get("human", False) == k]
                          for k in (False, True)}
        if self.human_weight is not None and not all(self._kind_idx.values()):
            self.human_weight = None   # only one kind present

    # ---- records ----

    def _episode_dirs(self) -> List[Path]:
        cfgs = self.cfg.get("cfgs", None) or list(range(1, 8))
        out = []
        for c in cfgs:
            d = self.data_root / f"RH20T_cfg{c}"
            if d.is_dir():
                out += sorted(p for p in d.iterdir() if p.name.startswith("task_")
                              and (self.include_human or "_human" not in p.name))
        return out

    def _load_calib(self, cfg_id: int, ts: int):
        key = f"{cfg_id}/{ts}"
        with self._calib_lock:
            if key in self._calib_cache:
                return self._calib_cache[key]
        d = self.calib_root / f"RH20T_cfg{cfg_id}" / "calib" / str(ts)
        calib = {}
        try:
            ex = np.load(d / "extrinsics.npy", allow_pickle=True).item()
            K = np.load(d / "intrinsics.npy", allow_pickle=True).item()
        except (FileNotFoundError, OSError):
            ex, K = {}, {}
        for s, m in ex.items():
            try:
                w2c = np.asarray(m, dtype=np.float64).reshape(-1, 4, 4)[0]
                k = np.asarray(K[s], dtype=np.float64)[:3, :3]
            except (TypeError, ValueError, KeyError):
                continue   # None entry = camera not calibrated in this snapshot
            calib[str(s)] = (k, w2c)
        with self._calib_lock:
            self._calib_cache[key] = calib
        return calib

    def _load_records(self) -> List[Dict[str, Any]]:
        min_rating = self.cfg.get("min_rating", None)
        test_frac = float(self.test_percentage)
        n_skip = {"no_meta": 0, "rating": 0, "few_cams": 0, "too_short": 0}
        clip_ms = {k: 1000.0 / f * (self.n_frames - 1)
                   for k, f in ((False, self.fps), (True, self.human_fps))}

        def scan(ep):   # -> (record | None, skip reason | None); I/O bound -> threads
            try:
                meta = json.loads((ep / "metadata.json").read_text())
            except (FileNotFoundError, ValueError):
                return None, "no_meta"
            human = ep.name.endswith("_human")
            if not human and min_rating is not None and meta.get("rating", 0) < min_rating:
                return None, "rating"
            cfg_id = int(ep.parent.name.replace("RH20T_cfg", ""))
            calib = self._load_calib(cfg_id, meta["calib"])
            cams = [p.name[4:] for p in sorted(ep.glob("cam_*"))
                    if p.name[4:] in calib and p.name[4:] not in IN_HAND.get(cfg_id, ())
                    and (p / "color" / "color.mp4").exists()
                    and (p / "color" / "timestamps.npy").exists()]
            if len(cams) < self.n_views:
                return None, "few_cams"
            if human:   # often shorter than the clip window; judge by the recorded frames
                spans = []                      # (metadata start/end mark a shorter segment)
                for c in cams:
                    t = np.load(ep / f"cam_{c}" / "color" / "timestamps.npy")
                    spans.append(float(t[-1] - t[0]) if len(t) >= 2 else 0.0)
                if sorted(spans)[-self.n_views] < clip_ms[True]:   # n_views cams must cover it
                    return None, "too_short"
            # held-out split by a stable hash of the episode name
            is_test = (zlib.crc32(ep.name.encode()) % 10000) < test_frac * 10000
            if (self.split == "training") == is_test:
                return None, None
            return {
                "episode": ep.name, "ep_dir": str(ep), "cfg_id": cfg_id,
                "calib_ts": int(meta["calib"]), "rating": int(meta.get("rating", -1)),
                "cams": cams, "fps": self.human_fps if human else float(self.fps),
                "caption": "", "human": human,
                "height": self.height, "width": self.width,
            }, None

        records = []
        with ThreadPoolExecutor(32) as pool:
            for rec, why in pool.map(scan, self._episode_dirs()):
                if rec is not None:
                    records.append(rec)
                elif why:
                    n_skip[why] += 1
        print(f"RH20T[{self.split}]: {len(records)} episodes from {self.data_root} "
              f"(skipped {n_skip})")
        if not records:
            raise FileNotFoundError(f"No usable RH20T episodes under {self.data_root}")
        return records

    # ---- sampling ----

    def _rng(self, idx: int) -> np.random.Generator:
        # Training: fresh randomness per draw, distinct per worker (torch reseeds each
        # worker; numpy's global RNG would be duplicated across workers). Eval: fixed.
        self._draws += 1
        if self.split == "training":
            return np.random.default_rng([torch.initial_seed() % 2**32, idx, self._draws])
        return np.random.default_rng(idx)

    def _plan_clip(self, rec, rng):
        """Choose a time grid + V views that are all within max_sync_err_ms of it."""
        ts = {}
        for s in rec["cams"]:
            t = np.load(Path(rec["ep_dir"]) / f"cam_{s}" / "color" / "timestamps.npy")
            if len(t) >= 2:
                ts[s] = t.astype(np.int64)
        if len(ts) < self.n_views:
            return None
        # human clips may use their own rate (the kind decides, not rec["fps"]: older
        # robot-only manifests were written under a different fps)
        step = 1000.0 / (self.human_fps if rec.get("human", False) else self.fps)
        dur = step * (self.n_frames - 1)
        # the window only has to be covered by the views we pick, not by every camera
        for _ in range(8):
            lo, hi = min(t[0] for t in ts.values()), max(t[-1] for t in ts.values())
            if hi - lo < dur:
                return None
            start = rng.uniform(lo, hi - dur)
            grid = start + step * np.arange(self.n_frames)
            ok = {}
            for s, t in ts.items():
                if t[0] > grid[0] + self.max_sync_err_ms or t[-1] < grid[-1] - self.max_sync_err_ms:
                    continue
                j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
                j = np.where(np.abs(t[j - 1] - grid) <= np.abs(t[j] - grid), j - 1, j)
                err = np.abs(t[j] - grid)
                if err.max() <= self.max_sync_err_ms:
                    ok[s] = (j, t[j], err)
            if len(ok) >= self.n_views:
                # the n_views cameras facing the workspace most directly, in random order
                calib = self._load_calib(rec["cfg_id"], rec["calib_ts"])
                front = sorted(ok, key=lambda s: marker_angle_deg(calib[s][1]))[: self.n_views]
                return list(rng.permutation(front)), ok
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Episodes can fail (dropped frames, short overlap); resample a new one rather
        # than raise -- this dataset is used directly, without MixtureDataset's guard.
        if self.human_weight is not None:   # draw the episode kind first, then an episode
            rng = self._rng(idx)
            pool = self._kind_idx[bool(rng.random() < self.human_weight)]
            idx = pool[int(rng.integers(len(pool)))]
        for attempt in range(20):
            rec = self.records[idx]
            rng = self._rng(idx)
            plan = self._plan_clip(rec, rng)
            if plan is not None:
                return self._build_sample(rec, *plan)
            if self.human_weight is not None:   # retry within the same kind
                pool = self._kind_idx[bool(rec.get("human", False))]
                idx = pool[int(rng.integers(len(pool)))]
            else:
                idx = int(rng.integers(len(self.records)))
        raise RuntimeError("RH20T: 20 consecutive episodes had no synchronisable window")

    def _build_sample(self, rec, serials, ok) -> Dict[str, Any]:
        calib = self._load_calib(rec["cfg_id"], rec["calib_ts"])
        H, W = self.height, self.width
        Hr, Wr = self.cfg.raymap_height, self.cfg.raymap_width
        frames, Ks, w2cs, stamps, errs = [], [], [], [], []
        for s in serials:
            j, t, err = ok[s]
            vr = decord.VideoReader(str(Path(rec["ep_dir"]) / f"cam_{s}" / "color" / "color.mp4"),
                                    num_threads=1)
            if len(vr) < j.max() + 1:
                raise ValueError(f"{rec['episode']} cam {s}: {len(vr)} frames < timestamps")
            x = torch.from_numpy(vr.get_batch(list(j)).asnumpy()).permute(0, 3, 1, 2)  # T,3,h,w
            h, w = x.shape[-2:]
            K, w2c = calib[s]
            K = K.copy()
            K[0] *= w / CALIB_W
            K[1] *= h / CALIB_H
            box = center_crop_box_for_aspect(h, w, H, W)
            x = apply_spatial_crop(x, box).float()
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False, antialias=True)
            frames.append(x / 127.5 - 1.0)
            Ks.append(adjust_K_for_crop_resize(K, box, H, W))
            w2cs.append(w2c)
            stamps.append(t)
            errs.append(err)
        K = torch.from_numpy(np.stack(Ks)).float()
        w2c = np.stack(w2cs)
        c2w = np.linalg.inv(w2c)
        c2w_rel = torch.from_numpy(np.linalg.inv(c2w[0])[None] @ c2w).float()   # view 0 = I
        cam_scale = 1.0
        if self.normalize_cam_scale and len(c2w_rel) > 1:
            cam_scale = float(c2w_rel[1:, :3, 3].norm(dim=-1).mean().clamp(min=1e-3))
            c2w_rel[:, :3, 3] /= cam_scale
        K_ray = K.clone()
        K_ray[:, 0] *= Wr / W
        K_ray[:, 1] *= Hr / H
        rays_d, rays_m = camera_to_raymap(K_ray, c2w_rel, Hr, Wr)
        return {
            "videos": torch.stack(frames).clamp_(-1, 1),
            "raymaps": torch.cat([rays_d, rays_m], dim=1),
            "intrinsics": K,
            "c2w": c2w_rel,
            "w2c_marker": torch.from_numpy(w2c).float(),
            "timestamps": torch.from_numpy(np.stack(stamps)),
            "sync_err_ms": torch.from_numpy(np.stack(errs)).float(),
            "episode": rec["episode"],
            "cfg_id": rec["cfg_id"],
            "serials": ";".join(serials),
            "prompts": rec["episode"].split("_user")[0],   # task id, e.g. "task_0017"
            "is_human": bool(rec.get("human", False)),
            "cam_scale": cam_scale,                         # metres per normalised unit
        }
