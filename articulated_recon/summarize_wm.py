"""Summary of the final 3D fits of the 25 selected world-model predictions -> results/wm/final/SUMMARY.md + summary.json"""
import json, pickle
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
R = Path("/n/lab_storage/ydu_lab/Lab/akiruga/geo_runs/rollouts_A7000")
sc = json.load(open(R / "scores.json"))
rows = []
for n in open(ROOT / "results/wm/selected.txt").read().split():
    d = ROOT / "results/wm" / (n + "_da3_final")
    if not (d / "fit.pkl").exists():
        continue
    m = json.load(open(d / "metrics.json")); F = pickle.load(open(d / "fit.pkl", "rb"))
    ob = {o["name"]: o for o in F["objects"]}
    cen = lambda o, t: o["poses"][t][:3, :3] @ o["mesh"][0].mean(0) + o["poses"][t][:3, 3]
    r = dict(name=n, robot_sil_iou=m.get("robot_sil_iou"), robot_sil_iou_views=m.get("robot_sil_iou_views"),
             strict_view_err_px=sc[n]["full_err_med"], objects={})
    for k, v in m.get("objects", {}).items():
        r["objects"][k] = dict(sil_f2px=v["sil_f2px"], iou=v["iou"], side_mm=v["extents_mm"][0], f0_err_mm=v["f0_pos_err_mm"],
                               held=v["held"])
    if "red_cube" in ob and "green_cube" in ob:
        T = len(ob["red_cube"]["poses"]); a, b = cen(ob["red_cube"], T - 1), cen(ob["green_cube"], T - 1)
        hs = ob["green_cube"]["half"][0] + ob["red_cube"]["half"][0]
        r["stack_3d"] = dict(dxy_mm=float(np.linalg.norm(a[:2] - b[:2]) * 1000), dz_mm=float((a[2] - b[2]) * 1000), dz_expected_mm=float(hs * 1000))
        r["stacked"] = bool(r["stack_3d"]["dxy_mm"] < 25 and abs(r["stack_3d"]["dz_mm"] - r["stack_3d"]["dz_expected_mm"]) < 20)
    rows.append(r)
f = lambda key: [r["objects"][o][key] for r in rows for o in r["objects"]]
S = dict(n=len(rows), robot_sil_iou_mean=float(np.mean([r["robot_sil_iou"] for r in rows])),
         robot_sil_iou_views=np.round(np.mean([r["robot_sil_iou_views"] for r in rows], 0), 3).tolist(),
         obj_sil_f2px_median=float(np.median(f("sil_f2px"))), obj_iou_mean=float(np.mean(f("iou"))),
         cube_side_mm_median=float(np.median(f("side_mm"))), cube_side_mm_iqr=np.round(np.percentile(f("side_mm"), [25, 75]), 1).tolist(),
         stacked_3d=int(sum(r.get("stacked", False) for r in rows)),
         strict_view_err_px_median=float(np.median([r["strict_view_err_px"] for r in rows])))
json.dump(dict(summary=S, episodes=rows), open(ROOT / "results/wm/final/summary.json", "w"), indent=1)
L = ["# World-model predictions (ManiSkill StackCube), 3D fits of 25 selected rollouts", "",
     f"- robot silhouette IoU vs SAM 3 (mean): {S['robot_sil_iou_mean']:.3f}; per view {S['robot_sil_iou_views']}",
     f"- cubes: silhouette F@2px median {S['obj_sil_f2px_median']:.2f}, IoU mean {S['obj_iou_mean']:.2f}, side {S['cube_side_mm_median']:.1f} mm (IQR {S['cube_side_mm_iqr']}; true 40)",
     f"- red cube ends on the green one in 3D: {S['stacked_3d']} / {S['n']}",
     f"- strict cross-view error of the prediction itself (median px): {S['strict_view_err_px_median']:.1f}", "",
     "| rollout | robot IoU | red F@2px | green F@2px | red side | green side | stacked (dxy, dz mm) | view err px |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    o = r["objects"]; st = r.get("stack_3d", {})
    L.append(f"| {r['name'].replace('StackCube-v1_episode-', '')} | {r['robot_sil_iou']:.2f} | {o.get('red_cube', {}).get('sil_f2px', float('nan')):.2f} | "
             f"{o.get('green_cube', {}).get('sil_f2px', float('nan')):.2f} | {o.get('red_cube', {}).get('side_mm', float('nan')):.0f} | "
             f"{o.get('green_cube', {}).get('side_mm', float('nan')):.0f} | {'yes' if r.get('stacked') else 'no'} ({st.get('dxy_mm', 0):.0f}, {st.get('dz_mm', 0):.0f}) | {r['strict_view_err_px']:.0f} |")
(ROOT / "results/wm/final/SUMMARY.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
