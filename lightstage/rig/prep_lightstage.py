"""Build light-stage camera rigs + per-method inputs from our 4-view RH20T clips.

For each case (scene, source, timestep): the 4 real views at that timestep (GT frames or a
model's generated frames) with their calibrated cameras (cams.npz: K at 160x320, c2w with
view 0 = identity, metres, OpenCV), and a dome of N_RING x N_AZ (+ cap) cameras all looking
at the workspace centre (least-squares intersection of the real optical axes), radius = the
real cameras' median distance, "up" = the real cameras' mean -y axis.

Writes <out>/<case>/:
  rig.json                 input + light-stage cameras (OpenCV c2w, K, image size, order)
  seva/{images/, transforms.json, train_test_split_4.json}   (OpenGL c2w for SEVA)
"""
import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, '/tmp/akiruga_lightstage')
from table_up import table_normal
sys.path.insert(0, '/tmp/akiruga_lightstage')
from table_up import table_normal
from PIL import Image

ROOT = Path("/n/netscratch/ydu_lab/Lab/akiruga/mv_infer/v1_heldout15")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/akiruga_lightstage/cases")
import os
CASES = [tuple(c.split(":")) for c in os.environ.get("CASES", "145:gt,145:pred_tokens,000:gt,000:pred_tokens").split(",")]
T_SEL = 12
ELEVS = (20.0, 40.0, 60.0)       # degrees above the dome's horizontal plane
N_AZ = 20
CAP_ELEV, N_CAP = 75.0, 4


def look_at(eye, target, up):
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)       # OpenCV: x right, y down, z forward
    y = np.cross(z, x)
    c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def dome(c2w_in, scene, src, t):
    o, d = c2w_in[:, :3, 3], c2w_in[:, :3, 2]
    A = sum(np.eye(3) - np.outer(di, di) for di in d)
    b = sum((np.eye(3) - np.outer(di, di)) @ oi for oi, di in zip(o, d))
    centre = np.linalg.solve(A, b)
    # up = table-plane normal (RANSAC on the VGGT-Omega cloud); centre = optical-axis centre
    # projected onto the table, raised 5 cm
    up, c0, _ = table_normal(scene, src, t, centre, o)
    centre = centre - up * ((centre - c0) @ up) + 0.05 * up
    r = float(np.median(np.linalg.norm(o - centre, axis=1)))
    # azimuth 0 = direction of input camera 0 projected on the horizontal plane
    v0 = o[0] - centre; h0 = v0 - up * (v0 @ up); h0 /= np.linalg.norm(h0)
    h1 = np.cross(up, h0)
    def pos(el, az):
        el, az = np.deg2rad(el), np.deg2rad(az)
        return centre + r * (np.cos(el) * (np.cos(az) * h0 + np.sin(az) * h1) + np.sin(el) * up)
    cams = []                     # serpentine: ring 1 ccw, ring 2 cw, ring 3 ccw, cap
    for k, el in enumerate(ELEVS):
        azs = np.arange(N_AZ) * 360.0 / N_AZ
        if k % 2:
            azs = azs[::-1]
        cams += [(el, a) for a in azs]
    cams += [(CAP_ELEV, a) for a in np.arange(N_CAP) * 360.0 / N_CAP]
    c2ws = np.stack([look_at(pos(el, az), centre, up) for el, az in cams])
    # real cameras' elevation / azimuth on the same dome, for reference
    def elaz(p):
        v = p - centre; el = np.degrees(np.arcsin(np.clip(v @ up / np.linalg.norm(v), -1, 1)))
        hv = v - up * (v @ up); az = np.degrees(np.arctan2(hv @ h1, hv @ h0)) % 360
        return float(el), float(az), float(np.linalg.norm(v))
    return c2ws, cams, centre, up, r, [elaz(p) for p in o]


def gl(c2w):
    m = c2w.copy(); m[:3, 1:3] *= -1; return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sc, src in CASES:
        scene = next(p.name for p in ROOT.iterdir() if p.name.startswith(sc + "_") and p.is_dir())
        cams = np.load(ROOT / scene / "cams.npz")
        K, c2w_in = cams["intrinsics"].astype(np.float64), cams["c2w"].astype(np.float64)
        frames = np.load(ROOT / scene / f"{src}.npy")[:, T_SEL]           # V,H,W,3
        V, H, W = frames.shape[:3]
        c2ws, elaz_targets, centre, up, r, elaz_in = dome(c2w_in, scene, src, T_SEL)
        Kt = np.array([[K[:, 0, 0].mean(), 0, W / 2], [0, K[:, 1, 1].mean(), H / 2], [0, 0, 1]])
        case = f"{sc}_{src}_t{T_SEL}"
        d = OUT / case
        (d / "seva" / "images").mkdir(parents=True, exist_ok=True)
        tf = {"w": W, "h": H, "frames": []}
        for v in range(V):
            p = f"images/in_{v}.png"
            Image.fromarray(frames[v]).save(d / "seva" / p)
            tf["frames"].append({"file_path": p, "transform_matrix": gl(c2w_in[v]).tolist(),
                                 "fl_x": K[v, 0, 0], "fl_y": K[v, 1, 1], "cx": K[v, 0, 2], "cy": K[v, 1, 2],
                                 "w": W, "h": H})
        black = np.zeros((H, W, 3), np.uint8)
        for i, c in enumerate(c2ws):
            p = f"images/ls_{i:03d}.png"
            Image.fromarray(black).save(d / "seva" / p)
            tf["frames"].append({"file_path": p, "transform_matrix": gl(c).tolist(),
                                 "fl_x": Kt[0, 0], "fl_y": Kt[1, 1], "cx": Kt[0, 2], "cy": Kt[1, 2],
                                 "w": W, "h": H})
        (d / "seva" / "transforms.json").write_text(json.dumps(tf, indent=1))
        (d / "seva" / "train_test_split_4.json").write_text(json.dumps(
            {"train_ids": list(range(V)), "test_ids": list(range(V, V + len(c2ws)))}))
        rig = {"case": case, "scene": scene, "source": src, "t": T_SEL, "H": H, "W": W,
               "centre": centre.tolist(), "up": up.tolist(), "radius": r,
               "inputs": {"K": K.tolist(), "c2w": c2w_in.tolist(), "elev_az_dist": elaz_in},
               "stage": {"K": Kt.tolist(), "c2w": c2ws.tolist(), "elev_az": elaz_targets}}
        (d / "rig.json").write_text(json.dumps(rig))
        print(case, f"r={r:.2f}m", "inputs (elev, az, dist):",
              [tuple(round(x, 1) for x in e) for e in elaz_in], "stage cams:", len(c2ws))


if __name__ == "__main__":
    main()
