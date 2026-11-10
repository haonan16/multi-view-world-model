"""Build a compact multiview obs.npz from one RH20T episode (colour 640x360 + sensor depth + robot state).

  python prep_rh20t_episode.py --color_dir <.../RH20T_cfgN/EP> --depth_dir <.../RH20T_cfgN/EP> --cfg 5 \
      --out /n/lab_storage/ydu_lab/Lab/akiruga/articulated_recon/real/EP [--T 10]

WORLD = robot base frame. Per camera, base->cam (OpenCV) = T(tcp in cam) @ inv(T(tcp in base)) from
transformed/tcp.npy and transformed/tcp_base.npy (RH20T quaternions are w,x,y,z); the result is constant over
time (checked) and equals calib extrinsics (marker->cam) @ M(base->marker).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

CALIB_ROOT = Path("/n/netscratch/ydu_lab/Lab/akiruga/rh20t_lowres/calib")
CFG_IN_HAND = {1: ["043322070878"], 2: ["104422070042"], 3: ["045322071843"], 4: ["045322071843"],
               5: ["104422070042", "135122079702"], 6: ["135122075425", "135122070361"],
               7: ["135122075425", "135122070361"]}


def pose(p):
    M = np.eye(4)
    M[:3, 3] = p[:3]
    M[:3, :3] = Rot.from_quat(np.asarray(p[3:7])[[1, 2, 3, 0]]).as_matrix()   # w,x,y,z -> x,y,z,w
    return M


def read_frames(path, idxs):
    cap = cv2.VideoCapture(str(path))
    out, want, i = {}, set(int(j) for j in idxs), 0
    while len(out) < len(want):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = f
        i += 1
    assert len(out) == len(want), (path, len(out), len(want))
    return [out[int(j)] for j in idxs]


def decode_depth(f):
    """RH20T depth video frame (H*2, W, 3) -> metres; 0 = invalid. Removes codec wrap artefacts at 256 mm
    boundaries of the low byte (pixels far from their 5x5 median) and out-of-range values."""
    h = f.shape[0] // 2
    lo, hi = f[:h, :, 0].astype(np.float32), f[h:, :, 0].astype(np.float32)
    d = (hi * 256 + lo) / 1000.0
    med = cv2.medianBlur(d.astype(np.float32), 5)
    bad = (np.abs(d - med) > 0.03) | (d < 0.1) | (d > 4.0)
    d[bad] = 0
    return d


def turbo(x, lo, hi, valid):
    u = np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    c = cv2.applyColorMap(u, cv2.COLORMAP_TURBO)
    c[~valid] = 0
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color_dir", required=True)
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--cfg", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--motion_thr_m", type=float, default=0.02)
    ap.add_argument("--pad_s", type=float, default=0.0, help="extend the manipulation span by this much on both sides")
    ap.add_argument("--max_sync_ms", type=float, default=100.0, help="drop cameras whose nearest frame is further off")
    a = ap.parse_args()
    cd, dd, out = Path(a.color_dir), Path(a.depth_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ep = cd.name
    md = json.loads((cd / "metadata.json").read_text())
    cal = CALIB_ROOT / f"RH20T_cfg{a.cfg}" / "calib" / str(md["calib"])
    Ex = np.load(cal / "extrinsics.npy", allow_pickle=True).item()
    Kin = np.load(cal / "intrinsics.npy", allow_pickle=True).item()
    tc = np.load(cd / "transformed/tcp.npy", allow_pickle=True).item()
    tb = np.load(cd / "transformed/tcp_base.npy", allow_pickle=True).item()
    jn = np.load(cd / "transformed/joint.npy", allow_pickle=True).item()
    gr = np.load(cd / "transformed/gripper.npy", allow_pickle=True).item()

    # usable cameras: fixed (not in-hand), colour + depth video, calibration, tcp-in-camera
    cams = []
    for c in sorted(cd.glob("cam_*")):
        s = c.name[4:]
        if s in CFG_IN_HAND[a.cfg] or not (c / "color.mp4").exists() or not (dd / c.name / "depth.mp4").exists():
            continue
        if s not in Ex or s not in Kin or s not in tc:
            continue
        cams.append(s)
    print("cameras", cams)

    # base->cam per camera from tcp(cam) vs tcp(base)
    w2c, M_marker = {}, {}
    for s in cams:
        X = np.array([pose(x["tcp"]) @ np.linalg.inv(pose(y["tcp"])) for x, y in zip(tc[s], tb[s])
                      if x["timestamp"] == y["timestamp"] and x.get("tcp") is not None and y.get("tcp") is not None])
        if len(X) == 0:
            print("no tcp-in-camera data, camera skipped:", s)
            continue
        assert X.std(0).max() < 1e-6, (s, X.std(0).max())
        w2c[s] = X.mean(0)
        M_marker[s] = np.linalg.inv(np.asarray(Ex[s], dtype=np.float64).reshape(-1, 4, 4)[0]) @ w2c[s]
    cams = [s for s in cams if s in w2c]
    Ms = np.array(list(M_marker.values()))
    print("base->marker consistency across cams (max abs dev):", float(np.abs(Ms - Ms.mean(0)).max()))

    # timestamps
    ts = {}
    for s in cams:
        t = np.load(cd / f"cam_{s}" / "timestamps.npy", allow_pickle=True).item()
        assert np.array_equal(np.asarray(t["color"]), np.asarray(t["depth"])), s
        ts[s] = np.asarray(t["color"], dtype=np.int64)
    # robot state on the union of per-camera timestamps (values identical across cameras where shared)
    rob = {}
    for s in jn:
        for k, v in jn[s].items():
            rob.setdefault(int(k), {})["q"] = np.asarray(v, dtype=np.float64)
    for s in gr:
        for k, v in gr[s].items():
            rob.setdefault(int(k), {})["g"] = (float(v["gripper_info"][0]), float(v["gripper_command"][0]))
    for s in tb:
        for x in tb[s]:
            if x.get("tcp") is None:
                continue
            rob.setdefault(int(x["timestamp"]), {})["tcp"] = np.asarray(x["tcp"][:7], dtype=np.float64)
    rk = np.array(sorted(k for k, v in rob.items() if {"q", "g", "tcp"} <= v.keys()))

    # manipulation span = TCP displaced > thr from its start / end pose
    P = np.array([rob[k]["tcp"][:3] for k in rk])
    moving = (np.linalg.norm(P - P[0], axis=1) > a.motion_thr_m) & (np.linalg.norm(P - P[-1], axis=1) > a.motion_thr_m)
    t_lo = max(t[0] for t in ts.values())
    t_hi = min(t[-1] for t in ts.values())
    if moving.any():
        t_lo = max(t_lo, rk[np.argmax(moving)])
        t_hi = min(t_hi, rk[len(moving) - 1 - np.argmax(moving[::-1])])
    t_lo = max(t_lo - a.pad_s * 1000, max(t[0] for t in ts.values()))
    t_hi = min(t_hi + a.pad_s * 1000, min(t[-1] for t in ts.values()))
    grid = np.linspace(t_lo, t_hi, a.T)
    fidx = np.zeros((a.T, len(cams)), np.int64)
    sync = np.zeros((a.T, len(cams)))
    for v, s in enumerate(cams):
        j = np.abs(ts[s][None] - grid[:, None]).argmin(1)
        fidx[:, v] = j
        sync[:, v] = ts[s][j] - grid
    keep = np.abs(sync).max(0) <= a.max_sync_ms
    dropped = {s: float(np.abs(sync[:, v]).max()) for v, s in enumerate(cams) if not keep[v]}
    print("dropped (max |sync| ms):", dropped)
    cams = [s for v, s in enumerate(cams) if keep[v]]
    fidx, sync = fidx[:, keep], sync[:, keep]
    ri = np.abs(rk[None] - grid[:, None]).argmin(1)
    rob_sync = rk[ri] - grid

    # images
    rgb, dep = [], []
    for s in cams:
        cs = read_frames(cd / f"cam_{s}" / "color.mp4", fidx[:, cams.index(s)])
        ds = read_frames(dd / f"cam_{s}" / "depth.mp4", fidx[:, cams.index(s)])
        rgb.append(np.stack([c[..., ::-1] for c in cs]))
        dep.append(np.stack([decode_depth(d) for d in ds]))
    rgb = np.stack(rgb, 1)                       # T, V, H, W, 3
    dep = np.stack(dep, 1).astype(np.float32)    # T, V, H, W
    H, W = rgb.shape[2:4]
    assert dep.shape[2:] == (H, W), (dep.shape, rgb.shape)
    Ks = []
    for s in cams:
        K = np.asarray(Kin[s], dtype=np.float64)[:3, :3].copy()
        sx, sy = W / 1280.0, H / 720.0                      # calibration resolution 1280x720
        K[0, 0] *= sx; K[1, 1] *= sy
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5                # integer pixel-centre convention
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        Ks.append(K)
    Ks = np.stack(Ks)
    W2C = np.stack([w2c[s] for s in cams])

    q = np.stack([rob[k]["q"] for k in rk[ri]])
    tcp = np.stack([rob[k]["tcp"] for k in rk[ri]])
    gw = np.array([rob[k]["g"][0] for k in rk[ri]])
    gc = np.array([rob[k]["g"][1] for k in rk[ri]])
    times_s = (grid - grid[0]) / 1000.0

    # TCP projection check
    tiles = []
    errs = {}
    for t in range(a.T):
        row = []
        for v, s in enumerate(cams):
            im = np.ascontiguousarray(rgb[t, v, ..., ::-1]).copy()
            Tb = pose(tcp[t])
            for ax, col in zip(np.eye(3), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                pts = np.stack([Tb[:3, 3], Tb[:3, 3] + Tb[:3, :3] @ ax * 0.05])
                pc = (W2C[v, :3, :3] @ pts.T + W2C[v, :3, 3:]).T
                uv = (Ks[v] @ pc.T).T
                uv = uv[:, :2] / uv[:, 2:]
                cv2.line(im, tuple(int(round(x)) for x in uv[0]), tuple(int(round(x)) for x in uv[1]), col, 2)
            pc = W2C[v, :3, :3] @ Tb[:3, 3] + W2C[v, :3, 3]
            uv = Ks[v] @ pc
            u, vv = uv[0] / uv[2], uv[1] / uv[2]
            cv2.circle(im, (int(round(u)), int(round(vv))), 6, (255, 0, 255), 2)
            ui, vi = int(round(u)), int(round(vv))
            if 0 <= ui < W and 0 <= vi < H:
                win = dep[t, v, max(vi - 4, 0):vi + 5, max(ui - 4, 0):ui + 5]
                win = win[win > 0]
                if win.size:
                    errs.setdefault(s, []).append(float(np.min(win) - pc[2]))   # sensor depth - TCP depth
            cv2.putText(im, f"t{t} {s}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            row.append(im)
        tiles.append(np.concatenate(row, 1))
    sheet = np.concatenate(tiles, 0)
    cv2.imwrite(str(out / "tcp_projection_check.jpg"), cv2.resize(sheet, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    big = np.concatenate([tiles[a.T // 2]], 0)
    cv2.imwrite(str(out / "tcp_projection_check_mid_fullres.jpg"), big)

    # previews
    rs = np.concatenate([np.concatenate(list(rgb[t, ..., ::-1]), 1) for t in range(a.T)], 0)
    cv2.imwrite(str(out / "preview_rgb.jpg"), cv2.resize(rs, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    ok = dep > 0
    lo, hi = np.quantile(dep[ok], 0.01), np.quantile(dep[ok], 0.99)
    ds_ = np.concatenate([np.concatenate([turbo(dep[t, v], lo, hi, ok[t, v]) for v in range(len(cams))], 1) for t in range(a.T)], 0)
    cv2.imwrite(str(out / "preview_depth.jpg"), cv2.resize(ds_, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST), [cv2.IMWRITE_JPEG_QUALITY, 85])

    # cross-view depth consistency: unproject view a, project into view b, compare with b's depth
    cons = []
    t = a.T // 2
    for va in range(len(cams)):
        z = dep[t, va]
        vs, us = np.nonzero(z > 0)
        zz = z[vs, us]
        xc = np.stack([(us - Ks[va, 0, 2]) / Ks[va, 0, 0] * zz, (vs - Ks[va, 1, 2]) / Ks[va, 1, 1] * zz, zz])
        Xw = np.linalg.inv(W2C[va]) @ np.vstack([xc, np.ones_like(zz)])
        for vb in range(len(cams)):
            if vb == va:
                continue
            pb = W2C[vb] @ Xw
            uvb = Ks[vb] @ pb[:3]
            ub, vb_ = np.round(uvb[0] / uvb[2]).astype(int), np.round(uvb[1] / uvb[2]).astype(int)
            m = (pb[2] > 0) & (ub >= 0) & (ub < W) & (vb_ >= 0) & (vb_ < H)
            zb = dep[t, vb][vb_[m], ub[m]]
            pz = pb[2][m]
            k = zb > 0
            # visible-surface agreement: points not occluded (|dz| small) fraction within 2 cm
            dz = zb[k] - pz[k]
            cons.append(dict(a=cams[va], b=cams[vb], n=int(k.sum()), frac_within_2cm=float(np.mean(np.abs(dz) < 0.02)),
                             median_abs_dz_m=float(np.median(np.abs(dz)))))

    np.savez_compressed(out / "obs.npz", rgb=rgb, depth_m=dep, K=Ks.astype(np.float32), w2c=W2C.astype(np.float32),
                        times_s=times_s.astype(np.float32), cam_serials=np.array(cams),
                        q_gt=q[:, :7].astype(np.float32), joint_raw14=q.astype(np.float32),
                        tcp_gt=tcp.astype(np.float32), gripper_width_mm=gw.astype(np.float32),
                        gripper_cmd_mm=gc.astype(np.float32), frame_idx=fidx, sync_err_ms=sync.astype(np.float32),
                        timestamps_ms=grid.astype(np.int64), robot_sync_err_ms=rob_sync.astype(np.float32),
                        w2c_marker=np.stack([np.asarray(Ex[s], dtype=np.float64).reshape(-1, 4, 4)[0] for s in cams]).astype(np.float32))
    td = json.load(open("/n/lab_storage/ydu_lab/Lab/akiruga/rh20t_meta/raw/task_description.json"))
    meta = {
        "episode": ep, "cfg": a.cfg, "robot": "Franka Emika Panda (RH20T cfg5, rh20t_api configs.json)" if a.cfg == 5 else f"cfg{a.cfg}",
        "task": td.get(ep[:9], {}).get("task_description_english"), "rating": md.get("rating"),
        "calib_quality": md.get("calib_quality"), "calib_ts": md["calib"],
        "cam_serials": cams, "dropped_for_sync": dropped, "excluded_in_hand": CFG_IN_HAND[a.cfg],
        "T": a.T, "V": len(cams), "H": int(H), "W": int(W),
        "timestamps_ms": grid.astype(np.int64).tolist(), "times_s": times_s.tolist(),
        "frame_idx": fidx.tolist(), "sync_err_ms": sync.round(1).tolist(),
        "max_abs_sync_err_ms": float(np.abs(sync).max()), "robot_sync_err_ms": rob_sync.tolist(),
        "frame_selection": f"T evenly spaced times over the manipulation span (TCP > {a.motion_thr_m} m from both its start and end pose), nearest frame per camera",
        "units": {"depth_m": "metres along camera z, 0 = invalid", "w2c": "metres", "q_gt": "rad (7 Franka joints)",
                  "tcp_gt": "base frame: x,y,z [m], qw,qx,qy,qz (RH20T order, w first)", "gripper_width_mm": "mm (gripper_info[0])",
                  "times_s": "seconds since first selected frame"},
        "world_frame": "robot base (Franka link0 / RH20T tcp_base frame)",
        "w2c": "OpenCV world(base)->camera = T_cam_tcp @ inv(T_base_tcp) from transformed/tcp.npy & tcp_base.npy (constant over time to 1e-14); equals calib extrinsics (marker->cam, stored as w2c_marker) @ M_base->marker",
        "base_to_marker_M": Ms.mean(0).tolist(), "base_to_marker_max_dev_across_cams": float(np.abs(Ms - Ms.mean(0)).max()),
        "K": "calib intrinsics at 1280x720 scaled to 640x360 with integer pixel-centre convention: f*0.5, c' = (c+0.5)*0.5-0.5",
        "rgb_source": "RH20T 640x360 colour videos (HF Malak-Mansour/RH20T RH20T_cfg5.tar.gz)",
        "depth_source": "RH20T depth videos (HF introvoyz043/RH20T depth/RH20T_cfg5_depth.tar.gz), 640x720 stacked halves: d=(hi*256+lo)/1000 m; same timestamps as colour (aligned/registered to colour by RealSense); pixels >3 cm from 5x5 median (codec wrap artefacts), <0.1 m or >4 m set to 0",
        "joint_raw14": "transformed/joint.npy gives 14 values per stamp; first 7 = joint positions (q_gt); last 7 near zero (likely velocities/torques; unverified)",
        "tcp_projection_sensor_minus_tcp_depth_m": {s: float(np.median(v)) for s, v in errs.items()},
        "cross_view_depth_consistency_mid_frame": cons,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ["cam_serials", "max_abs_sync_err_ms", "tcp_projection_sensor_minus_tcp_depth_m", "base_to_marker_max_dev_across_cams"]}, indent=1))
    print("consistency", [(c["a"][-4:], c["b"][-4:], round(c["frac_within_2cm"], 2), round(c["median_abs_dz_m"], 3)) for c in cons])
    print("q_gt", q[:, :7].round(3)); print("grip", gw)


if __name__ == "__main__":
    main()
