"""Build the comparison site: results/*/*/{metrics.json, viewer.js, overlay.mp4, overlay_grid.jpg} -> site/"""
import json
import shutil
import os
from pathlib import Path

import numpy as np

R = Path(os.path.dirname(os.path.abspath(__file__))) / "results"
S = Path(os.path.dirname(os.path.abspath(__file__))) / "site"
S.mkdir(exist_ok=True)
(S / "data").mkdir(exist_ok=True)
for f in ("three.min.js", "OrbitControls.js"):
    if not (S / f).exists():
        shutil.copy(f"/n/lab_storage/ydu_lab/Lab/akiruga/mv_cloud_viewer/{f}", S / f)

SRC_NAME = {"gt512": "Sim GT (package, 512px, 5 keyframes)", "gt": "Sim GT (model frames, 448×256, 11 frames)",
            "est_B": "Model B: real RGB → predicted depth", "est_A": "Model A: real RGB → predicted depth",
            "roll_B": "Model B rollout: 1 view → 4 views + depth", "roll_A": "Model A rollout: 1 view → 4 views + depth",
            }
SRC_NAME_REAL = {"gt": "Real RH20T (5 RealSense views, sensor depth)"}
COND_NAME = {"mv": "multiview video + depth + cameras", "mono_depth": "monocular video + depth + camera",
             "mv_cal": "multiview + depth + cameras, depth self-calibrated by the robot",
             "mono_depth_cal": "monocular + depth + camera, depth self-calibrated by the robot",
             "mono_rgb": "monocular video only (pixels)",
             "mv_hybrid": "multiview + depth + cameras, then + pixel silhouettes (3D→pixels hybrid)",
             "mv_cal_recal": "multiview + depth (self-calibrated) + cameras re-calibrated on the robot (logged joints, ICP)",
             "mv_cal_hybrid": "multiview + depth (self-calibrated), then + pixel silhouettes (hybrid)"}
ORDER_SRC = ["gt512", "gt", "est_B", "est_A", "roll_B", "roll_A"]
ORDER_COND = ["mv", "mv_hybrid", "mv_cal", "mv_cal_recal", "mv_cal_hybrid", "mono_depth", "mono_depth_cal", "mono_rgb"]


def mean(x):
    if x is None:
        return None
    a = np.array(x, dtype=float)
    return None if a.size == 0 or np.all(np.isnan(a)) else float(np.nanmean(a))


rows = []
for d in sorted(R.glob("*/*__*")):
    if not (d / "metrics.json").exists():
        continue
    m = json.loads((d / "metrics.json").read_text())
    scene = d.parent.name
    src, cond = d.name.split("__")
    key = f"{scene}/{d.name}"
    out = S / "data" / scene / d.name
    out.mkdir(parents=True, exist_ok=True)
    for f in ("viewer.js", "overlay.mp4", "overlay_grid.jpg", "scene.json", "metrics.json", "log.txt"):
        if (d / f).exists():
            shutil.copy(d / f, out / f)
    if (out / "overlay.mp4").exists() and (not (out / "overlay.webm").exists()
                                           or (out / "overlay.webm").stat().st_mtime < (out / "overlay.mp4").stat().st_mtime):
        import subprocess
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(out / "overlay.mp4"),
                        "-c:v", "libvpx-vp9", "-crf", "34", "-b:v", "0", str(out / "overlay.webm")])
    iou = np.array(m.get("iou", []), dtype=float)
    da = np.array(m.get("depth_agree_1cm", []), dtype=float)
    used = [0, 1, 2, 3] if cond.startswith("mv") else [0]
    held = [v for v in range(iou.shape[1] if iou.ndim == 2 else 4) if v not in used]
    r = dict(key=key, scene=scene, src=src, cond=cond, src_name=(SRC_NAME_REAL.get(src, src) if scene == "real" else SRC_NAME.get(src, src)), cond_name=COND_NAME.get(cond, cond),
             robot_mm=mean(m.get("robot_surface_mm")), grip_mm=mean(m.get("gripper_surface_mm")),
             joint_deg=mean(m.get("joint_err_deg")), base_mm=m.get("base_err_mm"),
             peg_mm=mean(m.get("peg_chamfer_mm")), peg_f0=(m.get("peg_chamfer_mm") or [None])[0],
             box_mm=mean(m.get("box_with_hole_chamfer_mm")),
             axis_deg=m.get("joint_axis_err_deg"), travel_mm=mean(m.get("travel_err_mm")), jtype=m.get("joint_type"),
             iou_used=float(np.nanmean(iou[:, used])) if iou.size and not np.all(np.isnan(iou)) else None,
             iou_held=float(np.nanmean(iou[:, held])) if iou.size and held and not np.all(np.isnan(iou[:, held])) else None,
             agree=float(np.nanmean(da)) if da.size else None,
             agree_gt=mean(m.get("depth_agree_1cm_vs_gt")), agree3=mean(m.get("depth_agree_3cm")),
             secs=(m.get("timing") or {}).get("total_s"),
             objects=m.get("objects"))
    rows.append(r)
rows.sort(key=lambda r: ({"standard": 0, "drawer": 1, "real": 2}.get(r["scene"], 3), ORDER_SRC.index(r["src"]) if r["src"] in ORDER_SRC else 9,
                         ORDER_COND.index(r["cond"]) if r["cond"] in ORDER_COND else 9))
(S / "data" / "index.js").write_text("window.__rows=" + json.dumps(rows) + ";")
tpl = (Path(__file__).parent / "site_template.html").read_text()
(S / "index.html").write_text(tpl)
print(len(rows), "rows ->", S)

# markdown table for the README
def fm(x, d=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


lines = ["| scene | source | input | robot mm | gripper mm | joint ° | peg mm | target mm | axis ° | travel mm | IoU fit | IoU held-out | depth agree | time s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['scene']} | {r['src_name']} | {r['cond_name']} | {fm(r['robot_mm'])} | {fm(r['grip_mm'])} | "
                 f"{fm(r['joint_deg'], 2)} | {fm(r['peg_mm'])} | {fm(r['box_mm'])} | {fm(r['axis_deg'])} | {fm(r['travel_mm'])} | "
                 f"{fm(r['iou_used'], 3)} | {fm(r['iou_held'], 3)} | {fm(r['agree'], 3)} | {fm(r['secs'], 0)} |")
(S / "RESULTS.md").write_text("\n".join(lines) + "\n")
