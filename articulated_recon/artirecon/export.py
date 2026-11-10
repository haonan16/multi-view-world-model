"""Export a fit: scene.json (parts, joints, per-frame poses / joint values) + viewer JSON (meshes + poses)."""
import json

import numpy as np

from .fit_rgb import decimate
from .render import PALETTE

ARM_JOINTS = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6",
              "panda_joint7", "panda_finger_joint1", "panda_finger_joint2"]


def _mesh_cache(robot, faces=2500):
    if not hasattr(robot, "_viewer_meshes"):
        robot._viewer_meshes = {l: decimate(robot.visual[l], faces) for l in robot.mesh_links}
    return robot._viewer_meshes


def link_poses(robot, q, base):
    import torch
    with torch.no_grad():
        qt = torch.as_tensor(q, dtype=torch.float32, device="cuda")
        bt = torch.as_tensor(base, dtype=torch.float32, device="cuda")[None].expand(len(qt), 4, 4)
        T = robot.fk(qt, bt)
    return {l: T[l].cpu().numpy() for l in robot.mesh_links}


def box_vf(half):
    s = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) * half
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                  [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]])
    return s, f


def viewer_json(res, robot, points=None):
    """parts with local meshes and per-frame 4x4 world poses (row-major lists)."""
    T = len(res["robot"]["q"])
    parts = []
    LP = link_poses(robot, res["robot"]["q"], res["robot"]["base"])
    for l, (v, f) in _mesh_cache(robot).items():
        parts.append(dict(name=f"robot/{l}", kind="robot", color=PALETTE["robot"], verts=np.round(v, 4).ravel().tolist(),
                          faces=f.ravel().tolist(), poses=[np.round(LP[l][t], 5).ravel().tolist() for t in range(T)]))
    for ob in res["objects"]:
        if ob["type"] == "articulated":
            for role, kind in (("parent", "articulated_parent"), ("child", "articulated_child")):
                P = ob[role]
                slabs = P.get("slabs") or [(np.eye(4), P["half"])]
                for i, (Tl, hl) in enumerate(slabs):
                    v, f = box_vf(hl)
                    poses = [P["pose"] @ Tl] * T if role == "parent" else [P["poses"][t] @ Tl for t in range(T)]
                    parts.append(dict(name=f"{ob['name']}/{role}/{i}", kind=kind, color=PALETTE[kind],
                                      verts=np.round(v, 4).ravel().tolist(), faces=f.ravel().tolist(),
                                      poses=[np.round(x, 5).ravel().tolist() for x in poses]))
        elif ob["type"] == "static" and ob.get("hole"):
            from .objects import static_boxes
            for i, (Tb, hb) in enumerate(static_boxes(ob)):
                v, f = box_vf(hb)
                parts.append(dict(name=f"{ob['name']}/slab{i}", kind="static", color=PALETTE["static"],
                                  verts=np.round(v, 4).ravel().tolist(), faces=f.ravel().tolist(),
                                  poses=[np.round(Tb, 5).ravel().tolist()] * T))
        else:
            v, f = box_vf(ob["half"])
            parts.append(dict(name=ob["name"], kind=ob["type"], color=PALETTE.get(ob["type"], (0.9, 0.6, 0.3)),
                              verts=np.round(v, 4).ravel().tolist(), faces=f.ravel().tolist(),
                              poses=[np.round(ob["poses"][t], 5).ravel().tolist() for t in range(T)]))
    out = dict(label=res["label"], frames=T, steps=[int(s) for s in res["steps"]], parts=parts)
    if points is not None:
        out["points"] = points
    return out


def scene_json(res, uncertainty=None):
    """the package's requested contract: units, convention, parts, joints, keyframes."""
    T = len(res["robot"]["q"])
    base = res["robot"]["base"]
    parts, joints = [], []
    parts.append(dict(id="robot", geometry="URDF panda_v2 (ManiSkill assets), visual meshes", base_pose_world=base.tolist()))
    for ob in res["objects"]:
        if ob["type"] == "articulated":
            def geo(P, kind):
                g = dict(type="box", half_extents_m=np.asarray(P["half"]).tolist())
                if P.get("slabs"):
                    g = dict(type=kind, half_extents_m=np.asarray(P["half"]).tolist(),
                             slabs=[dict(pose_in_part=np.asarray(Tl).tolist(), half_extents_m=np.asarray(hl).tolist())
                                    for Tl, hl in P["slabs"]])
                return g
            parts.append(dict(id=ob["name"] + "/parent", geometry=geo(ob["parent"], "open_shell (walls, opening where the child slides out)"),
                              pose_world=ob["parent"]["pose"].tolist(), motion="static"))
            parts.append(dict(id=ob["name"] + "/child", geometry=geo(ob["child"], "open_top_container (floor + 4 walls)")))
            j = ob["joint"]
            vals = np.asarray(j["values"])
            joints.append(dict(parent=ob["name"] + "/parent", child=ob["name"] + "/child", type=j["type"],
                               axis_world=np.asarray(j["axis"]).tolist(),
                               frame="child box frame at joint value 0 (= child pose at frame 0)",
                               observed_range_m=[float(vals.min()), float(vals.max())],
                               limits=dict(lower=0.0, upper=None,
                                           note="upper limit not observed; at least the observed max travel"),
                               fit_match_fraction=float(j.get("match", np.nan))))
        else:
            geo = dict(type="box", half_extents_m=np.asarray(ob["half"]).tolist())
            if ob.get("hole"):
                h = ob["hole"]
                geo = dict(type="box_with_through_hole", half_extents_m=np.asarray(ob["half"]).tolist(),
                           hole_axis=int(h["axis"]), hole_centre_m={str(k): v for k, v in h["centre"].items()},
                           hole_half_m={str(k): v for k, v in h["half"].items()}, note=h["note"])
            parts.append(dict(id=ob["name"], geometry=geo, motion=ob["type"],
                              pose_world=np.asarray(ob["poses"][0]).tolist()))
            if ob["type"] == "free":
                g = ob.get("grasped")
                joints.append(dict(parent="world", child=ob["name"], type="free (6-DoF)",
                                   grasp_frames=[int(t) for t in range(T) if g and g[t]],
                                   note="rigidly carried by panda_hand on grasp frames"))
            if ob["type"] == "attachment":
                joints.append(dict(parent="robot/panda_hand", child=ob["name"], type="fixed",
                                   rel_pose=np.asarray(ob["rel_hand"][0]).tolist()))
    kf = []
    for t in range(T):
        rec = dict(frame=t, step=int(res["steps"][t]),
                   robot_joints=dict(zip(ARM_JOINTS, np.round(res["robot"]["q"][t], 5).tolist())), parts={})
        for ob in res["objects"]:
            if ob["type"] == "articulated":
                rec["parts"][ob["name"] + "/child"] = _p7(ob["child"]["poses"][t])
                rec.setdefault("joint_values", {})[ob["name"]] = float(ob["joint"]["values"][t])
            else:
                rec["parts"][ob["name"]] = _p7(ob["poses"][t])
        kf.append(rec)
    return dict(units="metres", coordinate_convention="ManiSkill world (z up) as defined by the supplied cameras; "
                "OpenCV cameras; quaternions wxyz", source=res["label"], parts=parts, joints=joints, keyframes=kf,
                timing_s=res.get("timing"))


def _p7(T):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return [float(x) for x in T[:3, 3]] + [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
