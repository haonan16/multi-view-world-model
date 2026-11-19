"""Explicit calibrated observation contract; no guesses about camera axes/units."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Observations:
    depth: np.ndarray  # T,V,H,W, camera-z in metres; 0/NaN invalid
    hand_mask: np.ndarray  # T,V,H,W, target hand only
    K: np.ndarray  # T,V,3,3
    c2w: np.ndarray  # T,V,4,4; OpenCV x-right/y-down/z-forward
    timestamps: np.ndarray  # T seconds
    confidence: np.ndarray  # T,V,H,W, [0,1]
    keypoints: np.ndarray | None = None  # T,V,21,3: u,v,confidence

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as z:
            if str(z["depth_type"].item()) != "camera_z_m":
                raise ValueError("depth_type must explicitly be camera_z_m")
            if str(z["camera_convention"].item()) != "opencv_c2w":
                raise ValueError("camera_convention must explicitly be opencv_c2w")
            depth = z["depth"].astype(np.float32)
            if depth.ndim != 4:
                raise ValueError("depth must be T,V,H,W")
            T, V = depth.shape[:2]
            mask = z["hand_mask"].astype(bool)
            if mask.shape != depth.shape:
                raise ValueError("hand_mask and depth must have the same shape")
            K, c2w = z["K"].astype(np.float32), z["c2w"].astype(np.float32)
            if K.shape == (V, 3, 3):
                K = np.broadcast_to(K, (T, V, 3, 3)).copy()
            if c2w.shape == (V, 4, 4):
                c2w = np.broadcast_to(c2w, (T, V, 4, 4)).copy()
            if K.shape != (T, V, 3, 3) or c2w.shape != (T, V, 4, 4):
                raise ValueError("Expected static V or time-varying T,V calibration")
            if not np.isfinite(K).all() or not np.isfinite(c2w).all():
                raise ValueError("Non-finite calibration")
            if np.any(K[..., 0, 0] <= 0) or np.any(K[..., 1, 1] <= 0):
                raise ValueError("Focal lengths must be positive")
            R = c2w[..., :3, :3]
            if not np.allclose(R @ R.swapaxes(-1, -2), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(R), 1, atol=1e-4):
                raise ValueError("Camera rotations must be proper orthonormal matrices")
            if not np.allclose(c2w[..., 3, :], [0, 0, 0, 1]):
                raise ValueError("Invalid homogeneous camera transform")
            timestamps = z["timestamps"].astype(np.float64)
            if timestamps.shape != (T,) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
                raise ValueError("timestamps must be finite, strictly increasing seconds")
            confidence = z["confidence"].astype(np.float32) if "confidence" in z else np.ones_like(depth)
            if confidence.shape != depth.shape or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
                raise ValueError("confidence must match depth and lie in [0,1]")
            valid = np.isfinite(depth) & (depth > 0)
            confidence = np.where(valid, confidence, 0)
            depth = np.where(valid, depth, 0)
            kp = z["keypoints"].astype(np.float32) if "keypoints" in z else None
            if kp is not None:
                if kp.shape != (T, V, 21, 3) or not np.isfinite(kp).all():
                    raise ValueError("keypoints must be finite T,V,21,3; use confidence 0 for missing")
                if np.any((kp[..., 2] < 0) | (kp[..., 2] > 1)):
                    raise ValueError("keypoint confidence must be in [0,1]")
        return cls(depth, mask, K, c2w, timestamps, confidence, kp)

    def select_views(self, indices):
        ix = list(indices)
        return Observations(self.depth[:, ix], self.hand_mask[:, ix], self.K[:, ix],
                            self.c2w[:, ix], self.timestamps, self.confidence[:, ix],
                            None if self.keypoints is None else self.keypoints[:, ix])

    def points(self, max_per_frame=512):
        """Confidence-weighted hand-only backprojection with deterministic sampling."""
        rng = np.random.default_rng(0)
        points, weights = [], []
        for t in range(len(self.depth)):
            frame_p, frame_w = [], []
            for v in range(self.depth.shape[1]):
                y, x = np.nonzero(self.hand_mask[t, v] & (self.confidence[t, v] > 0))
                d = self.depth[t, v, y, x]
                rays = np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(self.K[t, v]).T
                p = rays * d[:, None]
                p = p @ self.c2w[t, v, :3, :3].T + self.c2w[t, v, :3, 3]
                frame_p.append(p)
                frame_w.append(self.confidence[t, v, y, x])
            p, w = np.concatenate(frame_p), np.concatenate(frame_w)
            if len(p) > max_per_frame:
                ix = rng.choice(len(p), max_per_frame, replace=False)
                p, w = p[ix], w[ix]
            points.append(p.astype(np.float32)); weights.append(w.astype(np.float32))
        return points, weights
