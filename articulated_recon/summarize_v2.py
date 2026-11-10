"""Aggregate results/real_v2/*/metrics.json -> results/real_v2/SUMMARY.md (+ summary.json)
python summarize_v2.py [episode list] [--tag _holdout1]"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
eps = open(args[0]).read().split() if args else sorted(p.name for p in (ROOT / "results/real_v2").iterdir() if p.is_dir())
rows, R = [], []
for e in eps:
    f = ROOT / "results/real_v2" / (e + tag) / "metrics.json"
    ann = json.load(open(ROOT / "real/v2" / e / "ann.json")) if (ROOT / "real/v2" / e / "ann.json").exists() else {}
    if not f.exists():
        gi = "" if o.get("grasp_iou") is None else " (%.2f)" % o["grasp_iou"]
        rows.append(f"| {e[:9]} {e[15:19]}/{e[26:30]} | – | – | – | not run | | | | |")
        continue
    m = json.load(open(f))
    rob = np.mean(m["robot_surface_mm"]); grip = np.mean(m["gripper_surface_mm"]); jd = np.mean(m["joint_err_deg"])
    objs = [o["name"] for o in ann.get("objects", [])]
    rec = dict(ep=e, robot_mm=rob, gripper_mm=grip, joint_deg=jd, objects={})
    for nm in objs:
        k = f"obj/{nm}/"
        if k + "iou_mean" not in m:
            continue
        role = next(o["role"] for o in ann["objects"] if o["name"] == nm)
        o = dict(role=role, iou=m[k + "iou_mean"], iou_holdout=m.get(k + "iou_holdout"), f2=m.get(k + "sil_f2px"), f2_holdout=m.get(k + "sil_f2px_holdout"), depth1=m[k + "depth_agree_1cm"],
                 contact=m[k + "rest_contact_mm"], pen=m[k + "penetration_mm"], held=len(m[k + "held_frames"]), grasp_iou=m.get(k + "grasp_iou_vs_logged"),
                 ext=m[k + "extents_mm"], travel=m[k + "travel_cm"])
        rec["objects"][nm] = o
        gi = "" if o.get("grasp_iou") is None else " (%.2f)" % o["grasp_iou"]
        rows.append(f"| {e[:9]} {e[15:19]}/{e[26:30]} | {rob:.1f} | {grip:.1f} | {jd:.2f} | {nm} ({role}) | {o['iou']:.2f}"
                    + (f" / {o['iou_holdout']:.2f}" if o['iou_holdout'] is not None else "")
                    + (f" | {o['f2']:.2f}" if o.get('f2') is not None else " | –")
                    + (f" / {o['f2_holdout']:.2f}" if o.get('f2_holdout') is not None else "")
                    + f" | {o['depth1']:.2f} | {o['contact'] if o['contact'] is None else round(o['contact'], 1)} | {o['pen']:.1f} | {o['held']}{gi} | {np.round(o['ext']).astype(int).tolist()} | {o['travel']:.1f} |")
    R.append(rec)
hdr = ("| episode (task user/scene) | robot mm | gripper mm | joints ° | object | mask IoU (fit / held-out view) | silhouette F@2px (fit / held-out) | depth ±1cm | "
       "support contact mm | table penetration mm | held frames (grasp IoU vs logged) | extents mm | travel cm |\n|" + "---|" * 13)
rob = np.array([r["robot_mm"] for r in R])
mv = [o for r in R for o in r["objects"].values() if o["role"] == "moved"]
st = [o for r in R for o in r["objects"].values() if o["role"] == "static"]
summ = dict(episodes=len(R), robot_mm_mean=float(rob.mean()) if len(rob) else None, robot_mm_median=float(np.median(rob)) if len(rob) else None,
            robot_mm_max=float(rob.max()) if len(rob) else None,
            moved_iou_mean=float(np.mean([o["iou"] for o in mv])) if mv else None,
            moved_f2_mean=float(np.nanmean([o["f2"] for o in mv if o.get("f2") is not None])) if mv else None,
            moved_f2_holdout_mean=float(np.nanmean([o["f2_holdout"] for o in mv if o.get("f2_holdout") is not None])) if any(o.get("f2_holdout") is not None for o in mv) else None,
            moved_iou_holdout_mean=float(np.mean([o["iou_holdout"] for o in mv if o["iou_holdout"] is not None])) if any(o["iou_holdout"] is not None for o in mv) else None,
            moved_held_ok=int(sum(o["held"] > 0 for o in mv)), moved_n=len(mv),
            grasp_iou_mean=float(np.mean([o["grasp_iou"] for o in mv if o.get("grasp_iou") is not None])) if any(o.get("grasp_iou") is not None for o in mv) else None,
            static_iou_mean=float(np.mean([o["iou"] for o in st])) if st else None)
txt = (f"# Real RH20T v2 — {len(R)} episodes{(' (' + tag + ')') if tag else ''}\n\n"
       f"robot surface error: mean {summ['robot_mm_mean']:.1f} mm, median {summ['robot_mm_median']:.1f}, max {summ['robot_mm_max']:.1f}  \n"
       f"moved objects: mask IoU {summ['moved_iou_mean']:.2f}"
       + (f" (held-out view {summ['moved_iou_holdout_mean']:.2f})" if summ['moved_iou_holdout_mean'] else "")
       + (f", silhouette F@2px {summ['moved_f2_mean']:.2f}" if summ.get('moved_f2_mean') else "")
       + (f" (held-out {summ['moved_f2_holdout_mean']:.2f})" if summ.get('moved_f2_holdout_mean') else "")
       + f", grasp detected in {summ['moved_held_ok']}/{summ['moved_n']}"
       + (f", grasp interval IoU vs logged gripper {summ['grasp_iou_mean']:.2f}" if summ.get('grasp_iou_mean') is not None else "") + "  \n"
       + (f"static / target objects: mask IoU {summ['static_iou_mean']:.2f}\n\n" if st else "\n")
       + hdr + "\n" + "\n".join(rows) + "\n")
(ROOT / "results/real_v2" / f"SUMMARY{tag}.md").write_text(txt)
(ROOT / "results/real_v2" / f"summary{tag}.json").write_text(json.dumps(dict(summary=summ, episodes=R), indent=1, default=float))
print(txt)
