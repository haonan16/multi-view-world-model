"""Build a reconstruction (scene.json) in SAPIEN 3, replay its keyframes, render them at the supplied
cameras and run physics checks.

  python sim_replay.py <result_dir> [--cameras <observations npz with K,w2c>]
     (maniskill env: /n/lab_storage/ydu_lab/akiruga/.conda/envs/maniskill/bin/python)

Writes <result_dir>/sim/{render_t{t}_v{v}.png, sim_checks.json}
Physics checks (all from the reconstruction only):
  hold   : every keyframe, robot PD-held at the fitted joints, simulate 0.5 s, report how far each free
           body and the drawer joint drift (a consistent reconstruction is close to static equilibrium)
  joint  : drive the reconstructed prismatic joint from 0 to its observed max travel, report reached value
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import sapien
from scipy.spatial.transform import Rotation

PANDA = os.environ.get("AR_PANDA_URDF", "/n/lab_storage/ydu_lab/akiruga/.conda/envs/maniskill/lib/python3.10/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf")
ARM = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
       "panda_finger_joint1", "panda_finger_joint2"]


def T2pose(T):
    q = Rotation.from_matrix(np.asarray(T)[:3, :3]).as_quat()
    return sapien.Pose(np.asarray(T)[:3, 3], [q[3], q[0], q[1], q[2]])


def p7pose(p):
    return sapien.Pose(p[:3], p[3:])


def static_boxes(geo, T):
    H = np.asarray(geo["half_extents_m"], float)
    if geo.get("type") != "box_with_through_hole":
        return [(T, H)]
    a = geo["hole_axis"]
    b, c = [i for i in range(3) if i != a]
    hb, hc = geo["hole_half_m"][str(b)], geo["hole_half_m"][str(c)]
    cb = np.clip(geo["hole_centre_m"][str(b)], -H[b] + hb, H[b] - hb)
    cc = np.clip(geo["hole_centre_m"][str(c)], -H[c] + hc, H[c] - hc)
    out = []

    def slab(lo, hi):
        if np.any(hi - lo <= 1e-4):
            return
        Tl = np.array(T, float).copy()
        Tl[:3, 3] = Tl[:3, 3] + Tl[:3, :3] @ ((lo + hi) / 2)
        out.append((Tl, (hi - lo) / 2))
    lo, hi = -H.copy(), H.copy()
    l, h = lo.copy(), hi.copy(); h[b] = cb - hb; slab(l, h)
    l, h = lo.copy(), hi.copy(); l[b] = cb + hb; slab(l, h)
    l, h = lo.copy(), hi.copy(); l[b], h[b], h[c] = cb - hb, cb + hb, cc - hc; slab(l, h)
    l, h = lo.copy(), hi.copy(); l[b], h[b], l[c] = cb - hb, cb + hb, cc + hc; slab(l, h)
    return out


def build(scene, S, density=600.0):
    mat = scene.create_physical_material(1.0, 0.8, 0.0)
    scene.add_ground(0.0, material=mat)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(PANDA)
    robot.set_root_pose(T2pose(S["parts"][0]["base_pose_world"]))
    for j in robot.get_active_joints():
        j.set_drive_property(stiffness=1e5 if "finger" not in j.name else 1e3, damping=1e3 if "finger" not in j.name else 1e2)
    names = [j.name for j in robot.get_active_joints()]
    things = {}
    joints = {j["child"]: j for j in S["joints"]}
    for part in S["parts"][1:]:
        pid, geo = part["id"], part["geometry"]
        if pid.endswith("/parent"):
            continue
        if pid.endswith("/child"):
            par = [p for p in S["parts"] if p["id"] == pid.replace("/child", "/parent")][0]
            jd = joints[pid]
            ab = scene.create_articulation_builder()
            root = ab.create_link_builder()
            root.set_name("cabinet")
            hp = np.asarray(par["geometry"]["half_extents_m"])
            Tp = np.asarray(par["pose_world"])
            # parent as a hollow shell (5 walls) so the drawer can slide out; open on the side the joint points to
            ax_l = Tp[:3, :3].T @ np.asarray(jd["axis_world"])
            k = int(np.argmax(np.abs(ax_l))); sgn = np.sign(ax_l[k])
            t = 0.006
            for i in range(3):
                for s in (-1, 1):
                    if i == k and s == sgn:
                        continue
                    hh = hp.copy(); hh[i] = t
                    off = np.zeros(3); off[i] = s * (hp[i] - t)
                    root.add_box_collision(sapien.Pose(off), hh, material=mat)
                    root.add_box_visual(sapien.Pose(off), hh, material=[0.75, 0.75, 0.8])
            child = ab.create_link_builder(root)
            child.set_name("drawer")
            hc = np.asarray(geo["half_extents_m"])
            for sl in geo.get("slabs") or [dict(pose_in_part=np.eye(4).tolist(), half_extents_m=hc.tolist())]:
                Pp, hh = T2pose(sl["pose_in_part"]), np.asarray(sl["half_extents_m"])
                child.add_box_collision(Pp, hh * 0.97, material=mat, density=density)
                child.add_box_visual(Pp, hh, material=[0.35, 0.85, 0.55])
            Tc0 = np.asarray(S["keyframes"][0]["parts"][pid][:3])
            # joint frame: x axis = joint axis, at the child's closed-position centre
            a = np.asarray(jd["axis_world"]) / np.linalg.norm(jd["axis_world"])
            Rj = np.eye(3); Rj[:, 0] = a
            tmp = np.cross(a, [0, 0, 1.0]) if abs(a[2]) < 0.9 else np.cross(a, [1.0, 0, 0])
            Rj[:, 1] = tmp / np.linalg.norm(tmp); Rj[:, 2] = np.cross(Rj[:, 0], Rj[:, 1])
            Tj = np.eye(4); Tj[:3, :3] = Rj; Tj[:3, 3] = p7pose(S["keyframes"][0]["parts"][pid]).p - \
                a * S["keyframes"][0].get("joint_values", {}).get(pid.split("/")[0], 0.0)
            Tc = np.eye(4); Tc[:3, :3] = Rotation.from_quat(np.roll(S["keyframes"][0]["parts"][pid][3:], -1)).as_matrix()
            Tc[:3, 3] = Tj[:3, 3]
            child.set_joint_properties("prismatic", limits=[[0.0, 0.45]],
                                       pose_in_parent=T2pose(np.linalg.inv(Tp) @ Tj),
                                       pose_in_child=T2pose(np.linalg.inv(Tc) @ Tj))
            root.set_joint_properties("fixed", limits=[], pose_in_parent=T2pose(Tp), pose_in_child=sapien.Pose())
            art = ab.build(fix_root_link=True)
            art.set_name(pid.split("/")[0])
            art.set_root_pose(T2pose(Tp))
            things[pid] = ("articulated", art, pid.split("/")[0])
            continue
        motion = part.get("motion")
        if motion == "static":
            b = scene.create_actor_builder()
            T0 = np.asarray(part["pose_world"])
            for Tb, hb in static_boxes(geo, np.eye(4)):
                b.add_box_collision(T2pose(Tb), hb, material=mat)
                b.add_box_visual(T2pose(Tb), hb, material=[0.85, 0.8, 0.55])
            a = b.build_kinematic(name=pid)
            a.set_pose(T2pose(T0))
            things[pid] = ("static", a, None)
        elif motion == "free":
            b = scene.create_actor_builder()
            h = np.asarray(geo["half_extents_m"])
            b.add_box_collision(sapien.Pose(), h, material=mat, density=density)
            b.add_box_visual(sapien.Pose(), h, material=[1.0, 0.62, 0.25])
            a = b.build(name=pid)
            things[pid] = ("free", a, None)
        elif motion == "attachment":
            b = scene.create_actor_builder()
            h = np.asarray(geo["half_extents_m"])
            b.add_box_visual(sapien.Pose(), h, material=[0.78, 0.86, 0.98])
            things[pid] = ("attachment", b.build_kinematic(name=pid), None)
    return robot, names, things


def set_keyframe(robot, names, things, kf):
    q = np.array([kf["robot_joints"][n] for n in names])
    robot.set_qpos(q)
    robot.set_qvel(np.zeros_like(q))
    for j, n in zip(robot.get_active_joints(), names):
        tgt = kf["robot_joints"][n]
        if "finger" in n:             # a grasp is commanded slightly closed: squeeze force on whatever is held
            tgt = max(tgt - 0.01, 0.0)
        j.set_drive_target(tgt)
    for pid, (kind, obj, jn) in things.items():
        if kind == "articulated":
            v = kf.get("joint_values", {}).get(jn, 0.0)
            obj.set_qpos([v]); obj.set_qvel([0.0])
            obj.get_active_joints()[0].set_drive_property(0.0, 50.0)
        elif kind in ("free", "attachment") and pid in kf["parts"]:
            obj.set_pose(p7pose(kf["parts"][pid]))
            if kind == "free":
                rb = obj.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                rb.set_linear_velocity([0, 0, 0]); rb.set_angular_velocity([0, 0, 0])


def main():
    d = Path(sys.argv[1])
    S = json.loads((d / "scene.json").read_text())
    cams = None
    if "--cameras" in sys.argv:
        z = np.load(sys.argv[sys.argv.index("--cameras") + 1])
        cams = (z["K"][0], z["w2c"][0], z["rgb"].shape[2:4] if "rgb" in z.files else (512, 512))
    out = d / "sim"
    out.mkdir(exist_ok=True)
    scene = sapien.Scene()
    scene.set_timestep(1 / 200)
    scene.set_ambient_light([0.45, 0.45, 0.45])
    scene.add_directional_light([0.3, 0.3, -1], [0.9, 0.9, 0.9], shadow=True)
    robot, names, things = build(scene, S)
    report = {"hold": [], "source": S.get("source")}
    # ---- renders at the supplied cameras (kinematic replay) ----
    if cams is not None:
        import imageio.v3 as iio
        K, w2c, (H, W) = cams
        cam_objs = []
        for v in range(len(K)):
            c = scene.add_camera(f"cam{v}", int(W), int(H), 1.0, 0.01, 20.0)
            c.set_focal_lengths(float(K[v][0, 0]), float(K[v][1, 1]))
            c.set_principal_point(float(K[v][0, 2]), float(K[v][1, 2]))
            c2w = np.linalg.inv(w2c[v])
            R = c2w[:3, :3] @ np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])     # cv -> sapien camera axes
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = c2w[:3, 3]
            c.set_local_pose(T2pose(T))
            cam_objs.append(c)
        for t, kf in enumerate(S["keyframes"]):
            set_keyframe(robot, names, things, kf)
            scene.update_render()
            row = []
            for c in cam_objs:
                c.take_picture()
                row.append((np.clip(c.get_picture("Color")[..., :3], 0, 1) * 255).astype(np.uint8))
            iio.imwrite(out / f"render_t{t}.png", np.concatenate(row, 1))
    # ---- physics: hold each keyframe 0.5 s ----
    for t, kf in enumerate(S["keyframes"]):
        set_keyframe(robot, names, things, kf)
        p0 = {pid: obj.get_pose().p.copy() for pid, (k, obj, _) in things.items() if k == "free"}
        j0 = {pid: float(obj.get_qpos()[0]) for pid, (k, obj, _) in things.items() if k == "articulated"}
        for _ in range(100):
            scene.step()
        rec = {"frame": t}
        for pid, p in p0.items():
            rec[f"{pid}_drift_mm"] = float(np.linalg.norm(things[pid][1].get_pose().p - p) * 1000)
        for pid, j in j0.items():
            rec[f"{pid}_joint_drift_mm"] = float(abs(float(things[pid][1].get_qpos()[0]) - j) * 1000)
        report["hold"].append(rec)
    # ---- physics: drive the articulation along its joint ----
    for pid, (k, obj, jn) in things.items():
        if k != "articulated":
            continue
        set_keyframe(robot, names, things, S["keyframes"][0])
        robot.set_qpos(robot.get_qpos() * 0 + np.array([0, -0.8, 0, -2.6, 0, 2.0, 0.78, 0.04, 0.04]))  # arm out of the way
        for j, q in zip(robot.get_active_joints(), robot.get_qpos()):
            j.set_drive_target(q)
        jt = [j for j in S["joints"] if j["child"] == pid][0]
        target = jt["observed_range_m"][1]
        jj = obj.get_active_joints()[0]
        jj.set_drive_property(2000.0, 200.0)
        jj.set_drive_target(target)
        obj.set_qpos([0.0])
        for _ in range(400):
            scene.step()
        report[f"{pid}_drive"] = {"target_m": target, "reached_m": float(obj.get_qpos()[0])}
    (out / "sim_checks.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
