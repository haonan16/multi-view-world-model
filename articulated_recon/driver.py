"""Run many (source, scene, condition) jobs in one process: fit -> evaluate -> export.

python driver.py <scene>:<source>:<condition>[:run] ...   condition: mv | mono_depth | mono_rgb
Outputs results/<scene>/<tag>/{fit.pkl, log.txt, metrics.json, scene.json, viewer.json, overlay_grid.jpg, overlay_v*.mp4}
"""
import json
import pickle
import os
import sys
import time
import traceback
from pathlib import Path

import imageio.v3 as iio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artirecon.data import backproject, load_dense, load_gt512
from artirecon.evaluate import evaluate, plausibility, render_agreement
from artirecon.export import scene_json, viewer_json
from artirecon.render import overlay, render, scene_parts
from artirecon.robot import URDFRobot
from artirecon.scene_fit import SceneFitter

ROOT = Path(os.path.dirname(os.path.abspath(__file__))) / "results"
FORCE = "--force" in sys.argv
ROBOT = URDFRobot(n_pts_per_m2=20000)
_cache = {}


def load(src, scene, run):
    k = (src, scene, run)
    if k not in _cache:
        if scene == "real":
            from artirecon.data import load_real
            _cache[k] = load_real()
        elif src == "gt512":
            _cache[k] = load_gt512(scene)
        elif scene == "drawer":
            from artirecon.data import load_dense_drawer
            _cache[k] = load_dense_drawer(src, run=run)
        else:
            _cache[k] = load_dense(src, run=run)
    return _cache[k]


def tagof(src, cond, run):
    return f"{src}{'' if src in ('gt512', 'gt') else '_' + run}__{cond}"


def fit(scene, src, cond, run, d, log):
    obs = load(src, scene, run)
    if cond == "mono_rgb":
        from artirecon.fit_rgb import RGBFitter
        tag = f"{scene}_{src}{'' if src in ('gt512', 'gt') else '_' + run}"
        mz = np.load(ROOT / "masks" / f"{tag}.npz")
        masks = {k: mz[k] for k in mz.files}
        t0 = time.time()
        log(f"[{obs.label}/mono_rgb] pixel-only, view 0, masks {list(masks)}")
        r = RGBFitter(obs.rgb[:, 0], masks, log=log).run()
        world_T_base = obs.c2w[0] @ r["w2c"]          # evaluator-side placement only
        objs = [dict(name=o["name"], type="free", half=o["half"], iou=o["iou"],
                     poses=np.einsum("ij,tjk->tik", world_T_base, o["poses"])) for o in r["objects"]]
        return dict(label=obs.label + "/mono_rgb", source=src, scene=scene, condition=cond, steps=obs.steps, views=[0],
                    robot=dict(base=world_T_base, q=r["q"], iou=r["robot_iou"], w2c_pred=r["w2c"], K_pred=r["K"]),
                    objects=objs, timing={"total_s": time.time() - t0}, run=run)
    calib = cond.endswith("_cal")
    base = cond[:-4] if calib else cond
    o = obs.select(views=[0], label=obs.label + "/" + cond) if base == "mono_depth" else obs.select(label=obs.label + "/" + cond)
    kb = None
    if scene == "real":            # RH20T: cameras are calibrated to the robot base (world = base frame)
        kb = np.eye(4)
        log("  rig calibration: robot base = world origin (RH20T camera-to-base extrinsics)")
    elif src in ("est", "roll"):     # world-model inputs come with the calibrated rig: robot base pose known
        from artirecon.evaluate import gt_states
        kb = gt_states(dict(source=src, scene=scene, steps=o.steps, robot=dict(q=np.zeros((o.T, 9)))))[0]["robot_base"]
        log("  rig calibration: robot base pose given (camera-to-base extrinsics), joints fitted")
    sf = SceneFitter(o, log=log, robot=ROBOT, calib=calib, known_base=kb).run()
    return dict(label=o.label, source=src, scene=scene, condition=cond, steps=o.steps, views=o.meta.get("views"),
                robot=sf.rob, objects=sf.objects, plane=(sf.prep["n"], sf.prep["d"]), timing=sf.timing, tcp=sf.tcp, run=run)


def post(res, scene, src, run, d, log):
    full = load(src, scene, run)
    m = evaluate(res, ROBOT)
    gt_obs = full if src == "gt512" else None
    ra = render_agreement(res, ROBOT, full, gt_obs)
    m["depth_agree_1cm"] = ra["depth_agree_1cm"].tolist()
    m["depth_agree_3cm"] = ra["depth_agree_3cm"].tolist()
    m["iou"] = ra["iou"].tolist()
    if src in ("est", "roll"):
        m["depth_agree_1cm_vs_gt"] = render_agreement(res, ROBOT, load("gt", scene, run))["depth_agree_1cm"].tolist()
    m.update(plausibility(res, ROBOT))
    m["timing"] = res["timing"]
    m["objects"] = [(o["name"], o["type"]) for o in res["objects"]]
    (d / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    (d / "scene.json").write_text(json.dumps(scene_json(res), indent=1, default=float))
    # observed points (the input actually used) for the viewer, ~12k per frame
    pts = []
    used = res.get("views") or list(range(full.V))
    for t in range(full.T):
        P, C = [], []
        for v in used:
            if full.depth is None or res["condition"] == "mono_rgb":
                continue
            p, pix = backproject(full.depth[t, v], full.K[v], full.c2w[v], 2)
            P.append(p); C.append(full.rgb[t, v][pix[:, 0], pix[:, 1]])
        if P:
            P, C = np.concatenate(P), np.concatenate(C)
            keep = (P[:, 2] > -0.02) & (np.abs(P[:, 0]) < 1.2) & (np.abs(P[:, 1]) < 1.2) & (P[:, 2] < 1.2)
            P, C = P[keep], C[keep]
            i = np.random.default_rng(t).choice(len(P), min(len(P), 12000), replace=False)
            pts.append(dict(xyz=np.round(P[i], 3).ravel().tolist(), rgb=C[i].ravel().tolist()))
    vj = viewer_json(res, ROBOT, pts or None)
    (d / "viewer.js").write_text("window.__ar(" + json.dumps(d.parent.name + "/" + d.name) + "," + json.dumps(vj) + ");")
    # overlays (every camera, incl. ones not used for fitting)
    H, W = full.rgb.shape[2:4]
    frames = []
    for t in range(full.T):
        parts = [(mm, c) for mm, c, _ in scene_parts(ROBOT, res, t)]
        row = []
        for v in range(full.V):
            img, alpha, _, _ = render(parts, full.K[v], np.linalg.inv(full.c2w[v]), H, W)
            row.append(np.concatenate([full.rgb[t, v], overlay(full.rgb[t, v], img, alpha, 0.75)], 0))
        frames.append(np.concatenate(row, 1))
    frames = np.stack(frames)
    iio.imwrite(d / "overlay_grid.jpg", np.concatenate([f[H:] for f in frames], 0)[::2, ::2], quality=85)
    iio.imwrite(d / "overlay.mp4", frames[:, :, :, :], fps=2 if full.T <= 5 else 3, codec="libx264",
                macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "23"])
    s = {k: (round(float(np.nanmean(np.array(v, dtype=float))), 2) if isinstance(v, list) and v and isinstance(v[0], (int, float)) else v)
         for k, v in m.items() if k not in ("depth_agree_1cm", "depth_agree_3cm", "iou", "depth_agree_1cm_vs_gt", "objects", "timing", "gt_travel_m")}
    s["iou_views"] = np.round(np.nanmean(np.array(m["iou"]), 0), 3).tolist()
    s["depth_agree_views"] = np.round(np.nanmean(np.array(m["depth_agree_1cm"]), 0), 3).tolist()
    log("METRICS " + json.dumps(s, default=str))


for job in [a for a in sys.argv[1:] if not a.startswith("--")]:
    p = job.split(":")
    scene, src, cond = p[:3]
    run = p[3] if len(p) > 3 else "B"
    d = ROOT / scene / tagof(src, cond, run)
    d.mkdir(parents=True, exist_ok=True)
    logf = open(d / "log.txt", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()
    try:
        if "--objects" in sys.argv and (d / "fit.pkl").exists() and cond != "mono_rgb":
            # re-run only the object stage with the current code, reusing the saved robot fit
            import artirecon.scene_fit as SFm
            old = pickle.load(open(d / "fit.pkl", "rb"))
            saved = old["robot"]
            orig = SFm.RobotFitter.run
            SFm.RobotFitter.run = lambda self, **k: saved
            obs = load(src, scene, run)
            calib = cond.endswith("_cal")
            o = obs.select(views=[0]) if cond.startswith("mono") else obs.select()
            o.label = old["label"]
            sf = SceneFitter(o, log=log, robot=ROBOT, calib=calib).run()
            SFm.RobotFitter.run = orig
            old["objects"], old["tcp"] = sf.objects, sf.tcp
            old["plane"] = (sf.prep["n"], sf.prep["d"])
            old["timing"]["objects_s"] = sf.timing["objects_s"]
            old["timing"]["total_s"] = old["timing"].get("robot_s", 0) + sf.timing["objects_s"]
            pickle.dump(old, open(d / "fit.pkl", "wb"))
        elif FORCE or not (d / "fit.pkl").exists():
            res = fit(scene, src, cond, run, d, log)
            pickle.dump(res, open(d / "fit.pkl", "wb"))
        res = pickle.load(open(d / "fit.pkl", "rb"))
        post(res, scene, src, run, d, log)
        log(f"DONE {job}")
    except Exception:
        log("FAILED " + job + "\n" + traceback.format_exc())
