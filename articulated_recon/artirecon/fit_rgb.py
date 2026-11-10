"""Pixel-only baseline: monocular RGB video, no depth, no camera.

Classic real-to-sim render-and-compare: text-prompted masks (masks.py) are matched by differentiable
silhouette rendering (pytorch3d soft rasteriser) of the same URDF robot + box primitives.
Unknowns: camera pose (intrinsics guessed: 60 deg horizontal FOV), joint angles per frame, object
box size and poses. The world frame is the robot base frame; the table is assumed to be the base's
mounting plane z = 0 (objects at rest sit on it). Metric scale comes only from the URDF.
"""
import math
import time

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as Fnn
from pytorch3d.renderer import (BlendParams, MeshRasterizer, MeshRenderer, RasterizationSettings,
                                SoftSilhouetteShader)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection

from .robot import PANDA_READY, URDFRobot, axis_angle, so3_exp

DEV = "cuda"


def decimate(tm, n_faces):
    tm = tm.copy()
    tm.merge_vertices()              # glb meshes are unwelded; quadric decimation needs connectivity
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(tm.vertices)),
                                  o3d.utility.Vector3iVector(np.asarray(tm.faces)))
    m = m.simplify_quadric_decimation(n_faces)
    return np.asarray(m.vertices), np.asarray(m.triangles)


class SilRobot:
    """robot link meshes (decimated) posed by FK -> batched pytorch3d meshes"""

    def __init__(self, robot, faces_per_link=1200):
        self.robot = robot
        V, F, L = [], [], []
        off = 0
        for li, l in enumerate(robot.mesh_links):
            v, f = decimate(robot.visual[l], faces_per_link)
            V.append(v); F.append(f + off); L.append(np.full(len(v), li)); off += len(v)
        self.v = torch.tensor(np.concatenate(V), dtype=torch.float32, device=DEV)
        self.f = torch.tensor(np.concatenate(F), dtype=torch.int64, device=DEV)
        self.l = torch.tensor(np.concatenate(L), device=DEV)

    def verts(self, q, base=None):
        T = self.robot.fk(q, base)
        Ts = torch.stack([T[l] for l in self.robot.mesh_links], 1)[:, self.l]
        return (Ts[..., :3, :3] @ self.v[None, :, :, None])[..., 0] + Ts[..., :3, 3]


def box_tm(half):
    s = torch.tensor([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
                     dtype=torch.float32, device=DEV)
    f = torch.tensor([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                      [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]], dtype=torch.int64, device=DEV)
    return s * half[..., None, :], f


class SilRenderer:
    def __init__(self, H, W, res=128, sigma=3e-5, fpp=50):
        self.H, self.W = H, W
        s = res / max(H, W)
        self.h, self.w = int(round(H * s)), int(round(W * s))
        self.s = s
        self.rs = RasterizationSettings(image_size=(self.h, self.w), blur_radius=math.log(1 / 1e-4 - 1) * sigma,
                                        faces_per_pixel=fpp, bin_size=0)
        self.blend = BlendParams(sigma=sigma, gamma=1e-4)

    def __call__(self, verts, faces, K, w2c):
        """verts [B,N,3], faces [F,3]; K [B,3,3] (full-res), w2c [B,4,4] -> silhouettes [B,h,w]"""
        B = verts.shape[0]
        Ks = K.clone()
        Ks[:, :2] *= self.s
        size = torch.tensor([[self.h, self.w]], device=DEV).expand(B, 2)
        cams = cameras_from_opencv_projection(w2c[:, :3, :3], w2c[:, :3, 3], Ks, size)
        mesh = Meshes(verts=verts, faces=faces[None].expand(B, -1, -1))
        r = MeshRenderer(MeshRasterizer(cameras=cams, raster_settings=self.rs), SoftSilhouetteShader(self.blend))
        return r(mesh)[..., 3]


def soft_iou(a, b):
    i = (a * b).sum((-1, -2))
    u = (a + b - a * b).sum((-1, -2))
    return 1 - i / u.clamp(min=1e-6)


def lookat_w2c(eye, target, up=(0, 0, 1)):
    eye, target, up = map(lambda x: np.asarray(x, float), (eye, target, up))
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])                  # world -> cam rows (OpenCV: y down)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = -R @ eye
    return T


class RGBFitter:
    def __init__(self, rgb, masks, robot=None, fov_deg=60.0, res=128, log=print):
        self.rgb = rgb
        self.T, self.H, self.W = rgb.shape[:3]
        self.robot = robot or URDFRobot(n_pts_per_m2=2000)
        self.sil = SilRobot(self.robot)
        self.R = SilRenderer(self.H, self.W, res)
        f = self.W / 2 / math.tan(math.radians(fov_deg) / 2)
        self.K = torch.tensor([[f, 0, (self.W - 1) / 2], [0, f, (self.H - 1) / 2], [0, 0, 1]], dtype=torch.float32, device=DEV)
        ds = lambda m: Fnn.interpolate(torch.as_tensor(m, dtype=torch.float32, device=DEV)[:, None], size=(self.R.h, self.R.w),
                                       mode="area")[:, 0]
        self.Mr = ds(masks["robot"])
        self.Mo = {k: ds(v) for k, v in masks.items() if k != "robot"}
        self.log = log
        self.lo, self.hi = self.robot.lo[:8], self.robot.hi[:8]

    def full_q(self, q8):
        return torch.cat([q8, q8[:, -1:]], 1)

    def render_robot(self, q8, w2c):
        v = self.sil.verts(self.full_q(q8))
        return self.R(v, self.sil.f, self.K[None].expand(len(q8), 3, 3), w2c)

    def opt_cam_q(self, w2c0, q0, frames, iters=150, lr=0.02, opt_cam=True):
        B = len(q0)
        R0, t0 = w2c0[:, :3, :3], w2c0[:, :3, 3]
        w = torch.zeros(B, 3, device=DEV, requires_grad=opt_cam)
        dt = torch.zeros(B, 3, device=DEV, requires_grad=opt_cam)
        q = q0.clone().requires_grad_(True)
        ps = [{"params": [q], "lr": lr}]
        if opt_cam:
            ps.append({"params": [w, dt], "lr": lr * 0.5})
        opt = torch.optim.Adam(ps)
        M = self.Mr[frames]

        def mk():
            W2C = torch.zeros(B, 4, 4, device=DEV)
            W2C[:, :3, :3] = so3_exp(w) @ R0
            W2C[:, :3, 3] = t0 + dt
            W2C[:, 3, 3] = 1
            return W2C
        for it in range(iters):
            W2C = mk()
            S = self.render_robot(q, W2C)
            lim = (torch.relu(self.lo - q) + torch.relu(q - self.hi)).pow(2).sum(1) * 100
            L = soft_iou(S, M) + lim
            opt.zero_grad(); L.sum().backward(); opt.step()
        with torch.no_grad():
            W2C = mk()
            L = soft_iou(self.render_robot(q, W2C), M)
        return W2C.detach(), q.detach(), L.detach()

    def run(self):
        t0 = time.time()
        ready = torch.tensor(PANDA_READY[:8], device=DEV)
        # camera hypotheses around the robot (base frame), look-at the workspace in front of the base
        hyps = []
        for az in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            for el in np.radians([15, 30, 45, 60]):
                for dist in (1.0, 1.5, 2.2):
                    tgt = np.array([0.45, 0, 0.25])
                    eye = tgt + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
                    hyps.append(lookat_w2c(eye, tgt))
        W = torch.tensor(np.stack(hyps), dtype=torch.float32, device=DEV)
        with torch.no_grad():
            sc = []
            for ch in torch.split(torch.arange(len(W), device=DEV), 16):
                S = self.render_robot(ready[None].expand(len(ch), -1), W[ch])
                sc.append(soft_iou(S, self.Mr[[0] * len(ch)]))
            sc = torch.cat(sc)
        top = torch.argsort(sc)[:12]
        self.log(f"  rgb cam search: best IoU {1 - sc[top[0]]:.3f}")
        W2C, q, L = self.opt_cam_q(W[top], ready[None].repeat(len(top), 1), [0] * len(top), iters=250)
        b = int(L.argmin())
        self.w2c = W2C[b:b + 1]
        qs = [q[b]]
        self.log(f"  frame 0: IoU {1 - L[b]:.3f}  [{time.time() - t0:.1f}s]")
        g = torch.Generator().manual_seed(0)
        for t in range(1, self.T):
            inits = [qs[-1], ready] + [qs[-1] + torch.randn(8, generator=g).to(DEV) * 0.35 for _ in range(6)]
            Q = torch.max(torch.min(torch.stack(inits), self.hi), self.lo)
            _, Q, L = self.opt_cam_q(self.w2c.expand(len(Q), 4, 4), Q, [t] * len(Q), iters=150, opt_cam=False)
            b = int(L.argmin())
            qs.append(Q[b])
            self.log(f"  frame {t}: IoU {1 - L[b]:.3f}")
        Q = torch.stack(qs)
        # joint refine of the camera with all frames
        Wr, Q, L = self.opt_cam_q(self.w2c.expand(self.T, 4, 4), Q, list(range(self.T)), iters=150, lr=0.01, opt_cam=True)
        # average camera (shared) -> take the one with lowest loss
        self.w2c = Wr[int(L.argmin())][None]
        _, Q, L = self.opt_cam_q(self.w2c.expand(self.T, 4, 4), Q, list(range(self.T)), iters=100, lr=0.01, opt_cam=False)
        self.Q = Q
        self.log(f"  robot IoU per frame {np.round(1 - L.cpu().numpy(), 3)}  [{time.time() - t0:.1f}s]")
        objs = []
        for k, M in self.Mo.items():
            if M.sum() < 5:
                continue
            objs.append(self.fit_object(k, M))
        return dict(w2c=self.w2c[0].cpu().numpy(), K=self.K.cpu().numpy(), q=self.full_q(Q).cpu().numpy(),
                    robot_iou=(1 - L).cpu().numpy(), objects=objs, seconds=time.time() - t0)

    def hand(self, t):
        with torch.no_grad():
            T = self.robot.fk(self.full_q(self.Q[t:t + 1]))
        return T["panda_hand"][0].cpu().numpy()

    def fit_object(self, name, M):
        """box on the table at frame 0 (size + yaw + xy), then 6-DoF per frame by silhouette."""
        T = self.T
        # frame-0 init: mask centroid ray -> table plane z = 0
        ys, xs = torch.nonzero(M[0] > 0.5, as_tuple=True)
        u, v = xs.float().mean() / self.R.s, ys.float().mean() / self.R.s
        c2w = torch.linalg.inv(self.w2c[0])
        d = c2w[:3, :3] @ torch.linalg.inv(self.K) @ torch.tensor([u, v, 1.0], device=DEV)
        o = c2w[:3, 3]
        lam = -o[2] / d[2] if d[2] < 0 else 1.0
        p0 = (o + lam * d).detach()
        B = 8
        yaw = torch.linspace(0, np.pi, B + 1, device=DEV)[:-1].clone().requires_grad_(True)
        xy = p0[:2][None].repeat(B, 1).clone().requires_grad_(True)
        lh = torch.log(torch.tensor([0.05, 0.03, 0.03], device=DEV))[None].repeat(B, 1).clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [yaw, xy], "lr": 0.01}, {"params": [lh], "lr": 0.03}])

        def pose_from(yaw, xy, half):
            Tm = torch.zeros(len(yaw), 4, 4, device=DEV)
            Tm[:, :3, :3] = axis_angle(torch.tensor([0., 0, 1], device=DEV), yaw)
            Tm[:, :2, 3] = xy
            Tm[:, 2, 3] = half[:, 2]
            Tm[:, 3, 3] = 1
            return Tm

        def ren(Tm, half, frames):
            s, f = box_tm(half)
            v = (Tm[:, None, :3, :3] @ s[..., None])[..., 0] + Tm[:, None, :3, 3]
            return self.R(v, f, self.K[None].expand(len(v), 3, 3), self.w2c.expand(len(v), 4, 4))

        for it in range(250):
            half = lh.exp()
            L = soft_iou(ren(pose_from(yaw, xy, half), half, [0] * B), M[[0] * B])
            opt.zero_grad(); L.sum().backward(); opt.step()
        with torch.no_grad():
            half = lh.exp()
            L = soft_iou(ren(pose_from(yaw, xy, half), half, [0] * B), M[[0] * B])
        b = int(L.argmin())
        half = half[b].detach()
        poses = [pose_from(yaw[b:b + 1], xy[b:b + 1], half[None])[0].detach().cpu().numpy()]
        ious = [1 - float(L[b])]
        for t in range(1, T):
            prev = poses[-1]
            inits = [prev, self.hand(t) @ np.linalg.inv(self.hand(t - 1)) @ prev]
            Ti = torch.tensor(np.stack(inits), dtype=torch.float32, device=DEV)
            n = len(Ti)
            w = torch.zeros(n, 3, device=DEV, requires_grad=True)
            dt = torch.zeros(n, 3, device=DEV, requires_grad=True)
            opt = torch.optim.Adam([w, dt], lr=0.01)
            for it in range(150):
                Tm = Ti.clone()
                Tm[:, :3, :3] = so3_exp(w) @ Ti[:, :3, :3]
                Tm[:, :3, 3] = Ti[:, :3, 3] + dt
                L = soft_iou(ren(Tm, half[None].expand(n, 3), [t] * n), M[[t] * n])
                opt.zero_grad(); L.sum().backward(); opt.step()
            with torch.no_grad():
                Tm = Ti.clone()
                Tm[:, :3, :3] = so3_exp(w) @ Ti[:, :3, :3]
                Tm[:, :3, 3] = Ti[:, :3, 3] + dt
                L = soft_iou(ren(Tm, half[None].expand(n, 3), [t] * n), M[[t] * n])
            b = int(L.argmin())
            poses.append(Tm[b].cpu().numpy())
            ious.append(1 - float(L[b]))
        self.log(f"  object '{name}': half {np.round(half.cpu().numpy() * 1000, 1)} mm, IoU {np.round(ious, 3)}")
        return dict(name=name, half=half.cpu().numpy(), poses=np.stack(poses), iou=ious)
