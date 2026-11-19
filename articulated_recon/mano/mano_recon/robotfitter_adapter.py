"""MANO implementation of the surface/state contract consumed by RobotFitter."""
import numpy as np
import torch
from mano_recon.model import MANOHand, translation_base


def backproject(depth, K, c2w, stride=2):
    y, x = np.mgrid[:depth.shape[0]:stride, :depth.shape[1]:stride]
    d = depth[y, x]; good = np.isfinite(d) & (d > 0)
    y, x, d = y[good], x[good], d[good]
    rays = np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(K).T
    return (rays * d[:, None]) @ c2w[:3, :3].T + c2w[:3, 3], np.stack([x, y], -1)


def axis_angle(axis, angle):
    # Only identity outer-base bookkeeping uses this in the MANO port.
    a = axis / axis.norm()
    K = torch.zeros(3, 3, device=axis.device, dtype=axis.dtype)
    K[0, 1], K[0, 2], K[1, 0] = -a[2], a[1], a[2]
    K[1, 2], K[2, 0], K[2, 1] = -a[0], -a[1], a[0]
    return torch.eye(3, device=axis.device)[None] + angle.sin()[:, None, None]*K + (1-angle.cos())[:, None, None]*(K@K)


def declared_mano_side(protocol, require_explicit=False):
    side=protocol.get('mano_side')
    if side is None:
        if require_explicit:
            raise ValueError('Verified hand fitting requires an explicit mano_side declaration')
        return 'right'  # Compatibility with earlier right-only experiments.
    if side not in {'left','right'}:
        raise ValueError('mano_side must be left or right')
    return side


class MANORobot:
    def __init__(self, model_dir, initial_q, initial_translation, device='cuda', samples=768, side='right'):
        self.hand = MANOHand(model_dir, side=side, components=initial_q.shape[1]-3, samples=samples, device=device)
        self.initial = torch.tensor(np.concatenate([initial_q, initial_translation], -1), dtype=torch.float32, device=device)
        self.dof = self.initial.shape[1]
        self.betas = torch.nn.Parameter(torch.zeros(1, 10, device=device), requires_grad=False)
        self.lo = torch.tensor([-2*np.pi]*3 + [-3.]*self.hand.components + [-10.]*3, device=device)
        self.hi = -self.lo
        self.optim_scale = torch.tensor([1.]*(self.dof-3) + [.05]*3, device=device)
        self.noise_scale = torch.tensor([1.]*(self.dof-3) + [.035]*3, device=device)

    def surface(self, q, base):
        moving_base = translation_base(q[:, -3:])
        return self.hand.surface(q[:, :-3], base @ moving_base, self.betas)
