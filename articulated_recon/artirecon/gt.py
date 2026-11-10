"""Evaluator-side ground truth (never used for fitting): robot qpos/base and object poses per frame."""
import json

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .data import PKG


def pose7_to_T(p):
    """[x,y,z,qw,qx,qy,qz] -> 4x4"""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def gt_states_keyframes(scene):
    """-> list over keyframes of {'robot_base': 4x4, 'q': [9], 'actors': {name: 4x4}}"""
    S = json.loads((PKG / scene / "evaluator_only/states.json").read_text())
    out = []
    for s in S:
        st = s.get("state", s)
        arts = st["articulations"]
        rname = [k for k in arts if "panda" in k][0]
        a = np.array(arts[rname][0])
        rec = {"robot_base": pose7_to_T(a[:7]), "q": a[13:13 + 9], "actors": {}, "articulations": {}}
        for k, v in st["actors"].items():
            rec["actors"][k] = pose7_to_T(np.array(v[0])[:7])
        for k, v in arts.items():
            if k != rname:
                v = np.array(v[0])
                rec["articulations"][k] = {"root": pose7_to_T(v[:7]), "qpos": v[13:13 + (len(v) - 13) // 2]}
        out.append(rec)
    return out


def gt_states_steps(steps, traj="traj_950"):
    """peg H5: env_states at given control-step indices (state after `step` actions)."""
    f = h5py.File(PKG / "source/trajectory.h5", "r")
    g = f[traj]["env_states"]
    arts = g["articulations"]
    rname = [k for k in arts if "panda" in k][0]
    A = arts[rname][()]
    acts = {k: g["actors"][k][()] for k in g["actors"]}
    out = []
    for s in steps:
        a = A[s]
        rec = {"robot_base": pose7_to_T(a[:7]), "q": a[13:22], "actors": {k: pose7_to_T(v[s][:7]) for k, v in acts.items()}}
        out.append(rec)
    return out


def scene_graph(scene):
    return json.loads((PKG / scene / "evaluator_only/scene_graph.json").read_text())
