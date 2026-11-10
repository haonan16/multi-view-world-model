"""Differentiable articulated robot from a URDF (torch FK + surface samples + meshes).

Only what's needed for fitting: a kinematic tree of fixed / revolute / prismatic joints,
visual meshes per link (any trimesh-loadable file), surface samples with normals.
"""
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import trimesh

PANDA_URDF = os.environ.get("AR_PANDA_URDF", "/n/lab_storage/ydu_lab/akiruga/.conda/envs/maniskill/lib/python3.10/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf")


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def origin_T(el):
    T = np.eye(4)
    if el is None:
        return T
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    T[:3, :3] = rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


def axis_angle(axis, ang):
    """axis [3] tensor, ang [...] -> R [...,3,3] (Rodrigues)."""
    K = torch.zeros(3, 3, dtype=ang.dtype, device=ang.device)
    a = axis / axis.norm()
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -a[2], a[1], a[2], -a[0], -a[1], a[0]
    s, c = torch.sin(ang)[..., None, None], torch.cos(ang)[..., None, None]
    I = torch.eye(3, dtype=ang.dtype, device=ang.device)
    return I + s * K + (1 - c) * (K @ K)


def so3_exp(w):
    """w [...,3] -> R [...,3,3]"""
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = w / th
    K = torch.zeros(w.shape[:-1] + (3, 3), dtype=w.dtype, device=w.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def R_to_rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R)).as_rotvec()


class URDFRobot:
    def __init__(self, urdf=PANDA_URDF, n_pts_per_m2=60000, min_pts=150, device="cuda", mimic_fingers=True):
        self.urdf = Path(urdf)
        root = ET.parse(urdf).getroot()
        self.links, self.joints = [], []
        self.visual = {}
        for l in root.findall("link"):
            name = l.get("name")
            self.links.append(name)
            meshes = []
            for v in l.findall("visual"):
                g = v.find("geometry")
                m = g.find("mesh")
                if m is None:
                    continue
                f = self.urdf.parent / m.get("filename")
                tm = trimesh.load(f, force="mesh")
                tm.apply_transform(origin_T(v.find("origin")))
                meshes.append(tm)
            if meshes:
                self.visual[name] = trimesh.util.concatenate(meshes)
        child_of = {}
        for j in root.findall("joint"):
            jd = dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                      child=j.find("child").get("link"), T=origin_T(j.find("origin")))
            ax = j.find("axis")
            jd["axis"] = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.array([1., 0, 0])
            lim = j.find("limit")
            jd["lo"], jd["hi"] = (float(lim.get("lower", 0)), float(lim.get("upper", 0))) if lim is not None else (0, 0)
            self.joints.append(jd)
            child_of[jd["child"]] = jd
        self.root = [l for l in self.links if l not in child_of][0]
        # topological order
        order, frontier = [], [self.root]
        while frontier:
            p = frontier.pop(0)
            for jd in self.joints:
                if jd["parent"] == p:
                    order.append(jd)
                    frontier.append(jd["child"])
        self.order = order
        self.active = [jd for jd in order if jd["type"] in ("revolute", "prismatic", "continuous")]
        self.dof = len(self.active)
        self.lo = torch.tensor([j["lo"] for j in self.active], device=device)
        self.hi = torch.tensor([j["hi"] for j in self.active], device=device)
        self.device = device
        for jd in order:
            jd["Tt"] = torch.tensor(jd["T"], dtype=torch.float32, device=device)
            jd["axt"] = torch.tensor(jd["axis"], dtype=torch.float32, device=device)
        # surface samples
        self.pts, self.nrm, self.link_of = [], [], []
        self.mesh_links = [l for l in self.links if l in self.visual]
        for li, name in enumerate(self.mesh_links):
            tm = self.visual[name]
            n = max(min_pts, int(tm.area * n_pts_per_m2))
            p, fi = trimesh.sample.sample_surface_even(tm, n, seed=0)
            if len(p) < n // 2:
                p, fi = trimesh.sample.sample_surface(tm, n, seed=0)
            self.pts.append(torch.tensor(p, dtype=torch.float32))
            self.nrm.append(torch.tensor(tm.face_normals[fi], dtype=torch.float32))
            self.link_of.append(torch.full((len(p),), li))
        self.pts = torch.cat(self.pts).to(device)
        self.nrm = torch.cat(self.nrm).to(device)
        self.link_of = torch.cat(self.link_of).to(device)

    def fk(self, q, base=None):
        """q [B,dof] -> dict link -> [B,4,4] world poses. base [B,4,4] or None (identity)."""
        B = q.shape[0]
        I = torch.eye(4, device=q.device).expand(B, 4, 4)
        T = {self.root: base if base is not None else I}
        ai = {jd["name"]: i for i, jd in enumerate(self.active)}
        for jd in self.order:
            Tp = T[jd["parent"]] @ jd["Tt"]
            if jd["type"] in ("revolute", "continuous"):
                M = torch.zeros(B, 4, 4, device=q.device)
                M[:, :3, :3] = axis_angle(jd["axt"], q[:, ai[jd["name"]]])
                M[:, 3, 3] = 1
                Tp = Tp @ M
            elif jd["type"] == "prismatic":
                M = torch.eye(4, device=q.device).repeat(B, 1, 1)
                M[:, :3, 3] = jd["axt"] * q[:, ai[jd["name"]], None]
                Tp = Tp @ M
            T[jd["child"]] = Tp
        return T

    def surface(self, q, base=None):
        """-> points [B,N,3], normals [B,N,3] in world."""
        T = self.fk(q, base)
        Ts = torch.stack([T[l] for l in self.mesh_links], 1)          # B,L,4,4
        Tp = Ts[:, self.link_of]                                     # B,N,4,4
        p = (Tp[..., :3, :3] @ self.pts[None, :, :, None])[..., 0] + Tp[..., :3, 3]
        n = (Tp[..., :3, :3] @ self.nrm[None, :, :, None])[..., 0]
        return p, n

    def meshes_at(self, q, base=None):
        """numpy q [dof] -> list of (link, trimesh in world)."""
        with torch.no_grad():
            T = self.fk(torch.as_tensor(q, dtype=torch.float32, device=self.device)[None],
                        None if base is None else torch.as_tensor(base, dtype=torch.float32, device=self.device)[None])
        out = []
        for l in self.mesh_links:
            m = self.visual[l].copy()
            m.apply_transform(T[l][0].cpu().numpy())
            out.append((l, m))
        return out


PANDA_READY = [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04]
