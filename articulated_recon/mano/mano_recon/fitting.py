"""Fit one shared MANO shape and a pose sequence; no Panda state assumptions.

Adapted observation logic from Amani's geom.py: project the same surface into
all cameras, distinguish visible surfaces from occlusion, and explain observed
foreground points. Uses native torch rather than a CUDA-only PyTorch3D kernel.
"""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt

from .model import translation_base


def project(points, K, w2c):
    camera = torch.einsum("tvij,tnj->tvni", w2c[..., :3, :3], points) + w2c[:, :, None, :3, 3]
    pixels = torch.einsum("tvij,tvnj->tvni", K, camera)
    return pixels[..., :2] / camera[..., 2:].clamp_min(1e-5), camera[..., 2]


def rho(residual, scale):
    return residual.square() / (residual.square() + scale * scale)


class SequenceFitter:
    def __init__(self, hand, observations):
        self.hand, self.obs = hand, observations
        dev = hand.faces.device
        tensor = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)
        self.K = tensor(observations.K)
        self.w2c = torch.linalg.inv(tensor(observations.c2w))
        self.depth = tensor(observations.depth)
        self.mask = tensor(observations.hand_mask)
        self.confidence = tensor(observations.confidence)
        T, V, H, W = self.depth.shape
        distance = np.empty_like(observations.depth)
        for t in range(T):
            for v in range(V):
                mask = observations.hand_mask[t, v]
                distance[t, v] = distance_transform_edt(~mask) if mask.any() else 0
        self.mask_present = self.mask.sum((-1, -2)) > 0
        self.distance = tensor(distance)
        ps, ws = observations.points()
        self.points = [tensor(p) for p in ps]
        self.weights = [tensor(w) for w in ws]
        self.keypoints = tensor(observations.keypoints) if observations.keypoints is not None else None
        self.timestamps = tensor(observations.timestamps)

    def sample(self, images, uv, mode="nearest"):
        T, V, H, W = images.shape
        grid = torch.stack((2 * uv[..., 0] / (W - 1) - 1, 2 * uv[..., 1] / (H - 1) - 1), -1)
        out = F.grid_sample(images.reshape(T * V, 1, H, W), grid.reshape(T * V, 1, -1, 2),
                            mode=mode, padding_mode="zeros", align_corners=True)
        return out.reshape(T, V, -1)

    def loss(self, q, translation, betas):
        base = translation_base(translation)
        points, _ = self.hand.surface(q, base, betas)
        _, joints = self.hand(q, betas, base)
        uv, z = project(points, self.K, self.w2c)
        H, W = self.depth.shape[-2:]
        inside = ((uv[..., 0] >= 0) & (uv[..., 0] <= W - 1) &
                  (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1) & (z > 0))
        d = self.sample(self.depth, uv)
        confidence = self.sample(self.confidence, uv)
        target = self.sample(self.mask, uv) > 0.5
        valid = inside & (d > 0) & (confidence > 0)
        residual = z - d
        # Occluded surfaces receive constant bounded cost; cannot improve by hiding.
        # Valid observations of other surfaces constrain only known free space.
        depth_cost = torch.where(residual > 0.008, rho(torch.full_like(residual, 0.008), 0.005), rho(residual, 0.005))
        evidence = valid & (target | (residual < 0))
        weight = evidence.float() * confidence
        depth_loss = (depth_cost * weight).sum() / weight.sum().clamp_min(1)
        # Silhouettes: allow occlusion by measured nearer surfaces. Missing depth
        # does not prove occlusion. Missing masks disable this term for that view.
        occluded = valid & ~target & (residual > 0.008)
        visible = ~occluded & self.mask_present[..., None]
        distance = self.sample(self.distance, uv, mode="bilinear")
        # Prevent out-of-image solutions from avoiding silhouette supervision.
        border = F.relu(-uv[..., 0]) + F.relu(uv[..., 0] - W + 1) + F.relu(-uv[..., 1]) + F.relu(uv[..., 1] - H + 1)
        silhouette = ((distance + border).square() * visible).sum() / visible.sum().clamp_min(1) / 100
        terms = []
        for t, observed in enumerate(self.points):
            if len(observed):
                dist = torch.cdist(observed[None], points[t:t+1]).amin(-1)[0]
                terms.append((rho(dist, 0.01) * self.weights[t]).sum() / self.weights[t].sum().clamp_min(1e-8))
        coverage = torch.stack(terms).mean() if terms else points.sum() * 0
        kp_loss = points.sum() * 0
        if self.keypoints is not None:
            j_uv, j_z = project(joints, self.K, self.w2c)
            weights = self.keypoints[..., 2]
            # Keypoint labels remain valid even if the current hypothesis is behind camera.
            error = F.smooth_l1_loss(j_uv, self.keypoints[..., :2], reduction="none", beta=3).sum(-1)
            kp_loss = (error * weights).sum() / weights.sum().clamp_min(1) / 10
            kp_loss = kp_loss + (F.relu(0.01 - j_z) * weights).sum() / weights.sum().clamp_min(1)
        temporal = points.sum() * 0
        if len(q) >= 3:
            dt = self.timestamps.diff()
            vel = translation.diff(dim=0) / dt[:, None]
            angular_vel = q.diff(dim=0) / dt[:, None]
            temporal = vel.diff(dim=0).square().mean() + 0.001 * angular_vel.diff(dim=0).square().mean()
        shape_prior = betas.square().mean()
        pose_prior = q[:, 3:].square().mean()
        total = depth_loss + coverage + 0.2 * silhouette + 3 * kp_loss + 0.01 * temporal + 0.001 * shape_prior + 0.0001 * pose_prior
        return total, dict(depth=depth_loss, coverage=coverage, silhouette=silhouette,
                           keypoints=kp_loss, temporal=temporal, shape_prior=shape_prior)

    def fit(self, initial_q, initial_translation, initial_betas=None, iterations=200, log=print):
        device = self.hand.faces.device
        param = lambda x: torch.nn.Parameter(torch.as_tensor(x, dtype=torch.float32, device=device).clone())
        q, translation = param(initial_q), param(initial_translation)
        betas = param(np.zeros((1, 10)) if initial_betas is None else initial_betas)
        if q.shape != (len(self.depth), self.hand.dof) or translation.shape != (len(self.depth), 3):
            raise ValueError("Initial state must provide one pose and wrist translation per frame")
        optim = torch.optim.Adam([dict(params=[q, betas], lr=0.025), dict(params=[translation], lr=0.002)])
        history, best_loss, best = [], float("inf"), None
        for i in range(iterations + 1):
            optim.zero_grad()
            loss, terms = self.loss(q, translation, betas)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite fit objective at iteration {i}")
            value = float(loss.detach())
            if value < best_loss:
                best_loss = value
                best = dict(q=q.detach().clone(), translation=translation.detach().clone(), betas=betas.detach().clone())
            if i % 25 == 0 or i == iterations:
                row = dict(iteration=i, total=value, **{k: float(v.detach()) for k, v in terms.items()})
                history.append(row)
                log(f"iteration {i}: loss={value:.5f}")
            if i == iterations:
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([q, translation, betas], 10)
            optim.step()
            with torch.no_grad():
                betas.clamp_(-3, 3)
            if i == int(iterations * 0.75):
                for group in optim.param_groups:
                    group["lr"] *= 0.2
        best["history"] = history
        return best
