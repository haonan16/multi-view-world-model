"""results/wm/final/site/index.html: method, summary, the 25 selected world-model predictions with their 3D fits"""
import json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent; F = ROOT / "results/wm/final"; S = F / "site"; S.mkdir(parents=True, exist_ok=True)
J = json.load(open(F / "summary.json")); sm = J["summary"]
for r in J["episodes"]:
    src = ROOT / "results/wm" / (r["name"] + "_da3_final") / "overlay.jpg"; dst = S / (r["name"] + ".jpg")
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(os.path.relpath(src, S), dst)
css = "body{font-family:system-ui,sans-serif;max-width:1500px;margin:20px auto;padding:0 16px;color:#222}img{width:100%;border:1px solid #ccc}td,th{padding:3px 8px;border-bottom:1px solid #eee;text-align:right}th{text-align:center}h3{margin-top:28px}"
h = [f"<!doctype html><meta charset=utf-8><title>World-model 3D fits</title><style>{css}</style>",
     "<h1>3D reconstruction of world-model predictions (ManiSkill StackCube)</h1>",
     "<p><b>Selection.</b> 407 rollouts of the multiview world model (view 0, frame 0 + cameras + text &rarr; RGB + depth of 4 views) on the 50 held-out StackCube episodes, several seeds. "
     "Kept: the red cube ends on the green cube (3D check on triangulated colour blobs, then every candidate checked by eye in all 4 views), ranked by the prediction's own cross-view consistency (worst-view reprojection error of the triangulated cubes); the 25 most consistent of the accepted ones (18 distinct episodes; median cross-view error 7.7 px).</p>",
     "<p><b>3D fit.</b> Robot: URDF, known base, 24 joint hypotheses on <b>Depth Anything 3</b> depth of the predicted RGB (known cameras; per-view scale from the table) &rarr; refined on SAM 3 'robot arm' silhouettes in every view. "
     "(Test with GT on 18 episodes, real RGB + predicted depth: model depth 21 mm robot error, DA3 3.9 mm.) Cubes: colour masks, multiview triangulation of the blob centroids (RANSAC over views, no depth), "
     "size from silhouette area (calibrated on real renders), silhouette-driven pose, table / contact priors.</p>",
     "<p><b>Finding.</b> The generated views are systematically misregistered against their cameras (vs the true projection at frame 0: view 1 &asymp; (-9,-16) px, view 3 &asymp; (-16,+1) px, view 0 = conditioning view &asymp; 0), so a single 3D scene can only explain them to ~5-13 px; the fit is the least-squares compromise.</p>",
     f"<h2>Summary (25)</h2><ul><li>robot silhouette IoU vs SAM 3: {sm['robot_sil_iou_mean']:.2f} (per view {sm['robot_sil_iou_views']})</li>"
     f"<li>cubes: silhouette F@2px median {sm['obj_sil_f2px_median']:.2f}; side {sm['cube_side_mm_median']:.1f} mm (IQR {sm['cube_side_mm_iqr']}, true 40)</li>"
     f"<li>red cube ends on the green one in 3D: {sm['stacked_3d']} / {sm['n']}</li><li>prediction cross-view error (median): {sm['strict_view_err_px_median']:.1f} px</li></ul>",
     "<table><tr><th>rollout</th><th>robot IoU</th><th>red F@2px</th><th>green F@2px</th><th>red / green side mm</th><th>stacked (dxy, dz mm)</th><th>view err px</th></tr>"]
for r in J["episodes"]:
    o = r["objects"]; st = r.get("stack_3d", {})
    h.append(f"<tr><td><a href='#{r['name']}'>{r['name'][20:]}</a></td><td>{r['robot_sil_iou']:.2f}</td><td>{o['red_cube']['sil_f2px']:.2f}</td><td>{o['green_cube']['sil_f2px']:.2f}</td>"
             f"<td>{o['red_cube']['side_mm']:.0f} / {o['green_cube']['side_mm']:.0f}</td><td>{'yes' if r.get('stacked') else 'no'} ({st.get('dxy_mm', 0):.0f}, {st.get('dz_mm', 0):.0f})</td><td>{r['strict_view_err_px']:.0f}</td></tr>")
h.append("</table><p>Each figure: rows = frames (1 s apart), columns = the 4 predicted views; blue = fitted robot, orange = fitted cubes, drawn over the predicted RGB.</p>")
for r in J["episodes"]:
    h.append(f"<h3 id='{r['name']}'>{r['name']}</h3><img loading=lazy src='{r['name']}.jpg'>")
(S / "index.html").write_text("\n".join(h))
print("wrote", S / "index.html")
