"""Offline CPU visualization of measured frozen-plan physics states; no simulator.

Object poses and robot joint positions come from records.json. Robot meshes use
known URDF collision geometry, with forward kinematics verified against logged
TCP transforms. Both panels use one fixed declared diagnostic camera. Shorter
rollouts hold their final measured state with an explicit ended label.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def link_poses(robot, qpos, base):
    values = dict(zip([f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"], qpos))
    result = {"panda_link0": base.copy()}
    remaining = dict(robot.joints)
    while remaining:
        processed = []
        for child, joint in remaining.items():
            if joint["parent"] not in result:
                continue
            motion = np.eye(4)
            if joint["kind"] in {"revolute", "continuous"}:
                motion[:3, :3] = Rotation.from_rotvec(joint["axis"] * values[joint["name"]]).as_matrix()
            elif joint["kind"] == "prismatic":
                motion[:3, 3] = joint["axis"] * values[joint["name"]]
            elif joint["kind"] != "fixed":
                raise ValueError("Unsupported URDF joint for offline visualization")
            result[child] = result[joint["parent"]] @ joint["origin"] @ motion
            processed.append(child)
        if not processed:
            raise ValueError("Disconnected robot URDF")
        for child in processed:
            remaining.pop(child)
    return result


def robot_meshes(robot, origin):
    result = []
    for link in robot.xml.findall("link"):
        for collision in link.findall("collision"):
            mesh_node, box_node = collision.find("geometry/mesh"), collision.find("geometry/box")
            if mesh_node is not None:
                mesh = trimesh.load(robot.path.parent / mesh_node.get("filename"), force="mesh")
                scale = np.fromstring(mesh_node.get("scale", "1 1 1"), sep=" ")
                vertices = np.asarray(mesh.vertices) * scale
            elif box_node is not None:
                mesh = trimesh.creation.box(extents=np.fromstring(box_node.get("size"), sep=" "))
                vertices = np.asarray(mesh.vertices)
            else:
                raise ValueError("Unsupported robot collision shape")
            local = origin(collision.find("origin"))
            vertices = vertices @ local[:3, :3].T + local[:3, 3]
            result.append((link.get("name"), vertices, np.asarray(mesh.faces)))
    return result


def camera(view="overview"):
    if view == "contact":
        eye, target = np.array([.20, -.66, .50]), np.array([-.29, -.02, .09])
        focal = 650.
    else:
        eye, target, focal = np.array([.20, -.66, .50]), np.array([-.35, 0., .12]), 475.
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, forward], axis=-1)
    pose[:3, 3] = eye
    return np.array([[focal, 0., 320.], [0., focal, 240.], [0., 0., 1.]]), pose


def render_comparison(root, robot_code, raster_code, urdf, output, view="overview"):
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(robot_code))
    sys.path.insert(0, str(raster_code))
    from offline_ik import Panda, BASE, origin
    from mano_recon.render_fast import render
    robot = Panda(urdf)
    shapes = robot_meshes(robot, origin)
    index = json.loads((root / "migration-index.json").read_text())
    variants = []
    for entry in index["variants"]:
        folder = Path(entry["folder"])
        physics = root / "physics" / entry["variant"] / "block"
        records_path = physics / "records.json"
        records = json.loads(records_path.read_text())
        summary = json.loads((physics / "summary.json").read_text())
        if summary["status"] != "complete" or summary["control_freq"] != 20:
            raise ValueError("Only complete 20 Hz recorded physics is supported")
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["command_seal"]["urdf_sha256"] != sha(urdf):
            raise ValueError("Robot model does not match frozen planner URDF")
        for relative, expected in manifest["command_seal"]["robot_collision_assets_sha256"].items():
            if sha(urdf.parent / relative) != expected:
                raise ValueError("Robot collision mesh changed since the plan was frozen")
        if not np.allclose(np.diff([r["time_s"] for r in records]), .05, atol=1e-6, rtol=0):
            raise ValueError("Recorded physics sampling is not 20 Hz")
        max_error = max(float(np.linalg.norm(robot.fk(np.asarray(r["qpos"])[:7])[:3, 3] - np.asarray(r["tcp"])[:3, 3])) for r in records)
        if max_error > .002:
            raise ValueError(f"Robot FK does not match logged TCP: {max_error} m")
        mesh = trimesh.load(folder / "object.obj", force="mesh")
        variants.append(dict(name=entry["variant"], records=records, folder=folder, physics=physics, mesh=mesh,
                             fk_error_mm=1000 * max_error, records_sha256=sha(records_path)))
    if len(variants) != 2:
        raise ValueError("This paired visualization requires both completed variants")
    K, C = camera(view)
    axis = np.linspace(-1., 1., 41)
    xx, yy = np.meshgrid(axis, axis)
    table_v = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
    table_f = []
    for y in range(40):
        for x in range(40):
            a = y * 41 + x
            table_f.extend([[a, a + 1, a + 42], [a, a + 42, a + 41]])
    background, background_depth = render(table_v, np.asarray(table_f), K, C, size=(480, 640), color=(188, 182, 168))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=20, codec="libx264", quality=8, macro_block_size=16)
    count = max(len(x["records"]) for x in variants)
    still_indices = {0, count - 1}
    for data in variants:
        z = [r["object_pose"][2][3] for r in data["records"]]
        still_indices.add(int(np.argmax(z)))
    try:
        for i in range(count):
            panels = []
            for data in variants:
                row = data["records"][min(i, len(data["records"]) - 1)]
                poses = link_poses(robot, row["qpos"], BASE)
                canvas, depth = background.copy(), background_depth.copy()

                def add(vertices, faces, transform, color):
                    image, z = render(vertices @ transform[:3, :3].T + transform[:3, 3], faces, K, C,
                                      size=(480, 640), color=color)
                    visible = (z > 0) & ((depth <= 0) | (z < depth))
                    canvas[visible], depth[visible] = image[visible], z[visible]

                add(np.asarray(data["mesh"].vertices), np.asarray(data["mesh"].faces), np.asarray(row["object_pose"]), (220, 150, 47))
                object_canvas, object_depth = canvas.copy(), depth.copy()
                for name, vertices, faces in shapes:
                    add(vertices, faces, poses[name], (190, 200, 212))
                if view == "contact":
                    robot_pixels = (depth > 0) & ((object_depth <= 0) | (depth < object_depth))
                    canvas[robot_pixels] = (.30 * canvas[robot_pixels] + .70 * object_canvas[robot_pixels]).astype(np.uint8)
                cv2.rectangle(canvas, (0, 0), (640, 32), (25, 29, 35), -1)
                label = "Before: minimal rotation" if data["name"] == "stationary_minimal_rotation" else "After: stable support"
                cv2.putText(canvas, label, (12, 23), 0, .58, (246, 246, 246), 1, cv2.LINE_AA)
                cv2.rectangle(canvas, (0, 443), (640, 480), (25, 29, 35), -1)
                phase = "Ended; final measured state held" if i >= len(data["records"]) else row["phase"].replace("_", " ")
                cv2.putText(canvas, f'{row["time_s"]:.1f}s | {phase}', (12, 468), 0, .48, (246, 246, 246), 1, cv2.LINE_AA)
                panels.append(canvas)
            header = np.full((32, 1280, 3), 25, np.uint8)
            heading = "Measured physics | offline state rendering"
            if view == "contact":
                heading += " | robot translucent for visibility"
            cv2.putText(header, heading, (15, 23), 0, .6,
                        (246, 246, 246), 1, cv2.LINE_AA)
            frame = np.concatenate([header, np.concatenate(panels, axis=1)], axis=0)
            writer.append_data(frame)
            if i in still_indices:
                imageio.imwrite(output.with_name(output.stem + f"-frame-{i:04}.png"), frame)
    finally:
        writer.close()
    metadata = {"kind": "offline_render_of_measured_physics", "physics_rerun": False,
                "source": "Recorded qpos and object_pose only; no state refinement or target animation",
                "fps": 20, "frames": count, "camera_K": K.tolist(), "camera_c2w": C.tolist(),
                "view": view, "robot_opacity": .30 if view == "contact" else 1.,
                "mesh_style": "Official Panda collision meshes; exact fixed-reference object visual mesh",
                "urdf_sha256": sha(urdf), "script_sha256": sha(__file__),
                "video_sha256": sha(output), "variants": [{k: x[k] for k in ("name", "records_sha256", "fk_error_mm")} for x in variants],
                "shorter_run": "Final measured state held and explicitly labeled ended",
                "limitation": "Legacy calibrated control; fixed recorded reconstruction proxy, not exact real-world ground truth"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "robot-code", "raster-code", "urdf", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--view", choices=("overview", "contact"), default="overview")
    print(json.dumps(render_comparison(**vars(p.parse_args())), indent=2))
