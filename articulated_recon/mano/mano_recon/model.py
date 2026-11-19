"""Differentiable human-hand backend with Amani-style surface(q, base) output."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import smplx
import torch
from torch import nn
from smplx.vertex_ids import vertex_ids


class MANOHand(nn.Module):
    """q = [wrist axis-angle (3), hand PCA coefficients (P)].

    One shared beta vector describes hand shape. Coordinates are metres.
    With no base transform, the wrist is at the origin. A supplied base is a
    world-from-wrist transform, independent of MANO's template joint offset.
    This is a surface interface adapter, not a drop-in Panda RobotFitter.
    """

    def __init__(self, model_dir="assets/mano", side="right", components=12,
                 samples=1024, device="cpu"):
        super().__init__()
        if side not in ("left", "right") or not 1 <= components <= 45:
            raise ValueError("Use left/right and 1..45 PCA components")
        self.side, self.components = side, components
        path = Path(model_dir) / f"MANO_{side.upper()}.npz"
        with np.load(path, allow_pickle=False) as z:
            data = SimpleNamespace(**{k: z[k] for k in z.files})
        # Official v1.2 data is in metres. Reject unexpected scales explicitly.
        extent = np.ptp(data.v_template, axis=0).max()
        if not 0.05 < extent < 0.4:
            raise ValueError(f"Unexpected MANO scale {extent}; expected metres")
        self.mano = smplx.MANO(str(path), data_struct=data, is_rhand=side == "right",
                              num_pca_comps=components, use_pca=True,
                              flat_hand_mean=False)
        # smplx switches 45-component requests to axis-angle internally.
        # Keep our public q convention PCA at every dimensionality, so a
        # saved 12-component fit extends exactly by appending zeros.
        if components == 45:
            self.mano.use_pca = True
            self.mano.register_buffer("hand_components", torch.as_tensor(
                data.hands_components[:45], dtype=torch.float32))
        self.mano.requires_grad_(False)
        self.register_buffer("faces", torch.as_tensor(data.f.astype(np.int64)))
        tips = [vertex_ids["mano"][n] for n in ("thumb", "index", "middle", "ring", "pinky")]
        self.register_buffer("tip_indices", torch.tensor(tips))
        # MediaPipe/OpenPose-style order: wrist; thumb,index,middle,ring,pinky.
        order = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
                 10, 11, 12, 19, 7, 8, 9, 20]
        self.register_buffer("joint_order", torch.tensor(order))
        tri = data.v_template[data.f]
        areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        rng = np.random.default_rng(0)
        fi = rng.choice(len(tri), samples, p=areas / areas.sum())
        uv = rng.random((samples, 2))
        uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
        bary = np.column_stack((1 - uv.sum(1), uv))
        self.register_buffer("sample_faces", torch.tensor(fi))
        self.register_buffer("sample_bary", torch.tensor(bary, dtype=torch.float32))
        self.to(device)

    @property
    def dof(self):
        return 3 + self.components

    def forward(self, q, betas=None, base=None):
        if q.ndim != 2 or q.shape[1] != self.dof:
            raise ValueError(f"Expected q [T,{self.dof}]")
        if betas is None:
            betas = q.new_zeros(1, 10)
        if betas.shape != (1, 10):
            raise ValueError("betas must be one shared [1,10] shape for the sequence")
        output = self.mano(global_orient=q[:, :3], hand_pose=q[:, 3:],
                           betas=betas.expand(len(q), -1), transl=q.new_zeros(len(q), 3))
        root = output.joints[:, :1]
        vertices = output.vertices - root
        joints = output.joints - root
        joints = torch.cat((joints, vertices[:, self.tip_indices]), dim=1)[:, self.joint_order]
        if base is not None:
            if base.shape != (len(q), 4, 4):
                raise ValueError("base must be [T,4,4] world-from-wrist transforms")
            vertices = vertices @ base[:, :3, :3].transpose(1, 2) + base[:, None, :3, 3]
            joints = joints @ base[:, :3, :3].transpose(1, 2) + base[:, None, :3, 3]
        return vertices, joints

    def surface(self, q, base=None, betas=None):
        vertices, _ = self(q, betas, base)
        tri = vertices[:, self.faces[self.sample_faces]]
        points = (tri * self.sample_bary[None, :, :, None]).sum(2)
        normals = torch.nn.functional.normalize(
            torch.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1), dim=-1)
        return points, normals


def translation_base(translation):
    base = torch.eye(4, dtype=translation.dtype, device=translation.device).repeat(len(translation), 1, 1)
    base[:, :3, 3] = translation
    return base
