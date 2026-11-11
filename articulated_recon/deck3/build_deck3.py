"""deck3/wm_predictions.pptx: world-model predictions -> 3D (selection, depth, method with optimisation videos,
findings, results, all 25 three-row videos).  Same look as deck2."""
import json, os, sys
from pathlib import Path
import imageio.v3 as iio
import numpy as np
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
OLD = (HERE.parent.parent / "ego_world_model/ego_world_model_clean/articulated_recon/deck/build_deck.py")
src = OLD.read_text().split("# ---")[0]
src = src[:src.index("\n# ")] if "\n# " in src else src
exec(compile(src.split('\ndef autoplay')[0] + '\ndef autoplay' + src.split('\ndef autoplay')[1].split('\n\n\n')[0], "deck_helpers", "exec"))
M = HERE / "media"; globals()["M"] = M
W = HERE.parent / "results/wm"
J = json.load(open(W / "final/summary.json")); S = J["summary"]; E = {r["name"]: r for r in J["episodes"]}
sel = open(W / "selected.txt").read().split()


def img(slide, path, x, y, w=None, h=None):
    im = iio.imread(path); ar = im.shape[1] / im.shape[0]
    w = h * ar if w is None else w; h = w / ar if h is None else h
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h)); return w, h


def vid(slide, name, x=None, y=1.2, max_w=12.6, max_h=5.4):
    im = iio.imread(M / f"{name}.jpg"); ar = im.shape[1] / im.shape[0]
    w = min(max_w, max_h * ar)
    return video(slide, name, (SW - w) / 2 if x is None else x, y, w=w)


def table(slide, data, y=1.4, col_w=None, size=20, row_h=0.5, x=0.5):
    nr, nc = len(data), len(data[0]); col_w = col_w or [12.3 / nc] * nc
    tb = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(sum(col_w)), Inches(row_h * nr)).table
    for i in range(nr):
        tb.rows[i].height = Inches(row_h)
    for j, w_ in enumerate(col_w):
        tb.columns[j].width = Inches(w_)
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            c = tb.cell(i, j); c.text = str(val)
            for p in c.text_frame.paragraphs:
                for rr in p.runs:
                    rr.font.size = Pt(size); rr.font.name = FONT; rr.font.bold = (i == 0)
    return tb


best = sel[0]
s = new_slide("World-model predictions, reconstructed in 3D")
vid(s, best, max_h=4.9)
text(s, "rows: the model's prediction (4 views) · fitted robot + cubes overlaid · the 3D fit alone\n"
        "input to the model: one real image (view 0, frame 0) + cameras + text; everything else is generated", 0.6, 6.25, 12.2, 1.1, size=18)

s = new_slide("The problem: predicted depth is not accurate enough")
table(s, [["depth used for the 3D robot fit", "robot error vs ground truth"],
          ["simulator depth (reference)", "0.5 mm"],
          ["model's own predicted depth", "~83 mm"],
          ["  + per-view scale so the table lies at z = 0", "21 mm"],
          ["  + keep only pixels where views agree (RANSAC-like)", "173 mm (removes the robot too)"],
          ["Depth Anything 3 on the RGB, known cameras, + table scale", "3.9 mm"]], col_w=[8.0, 4.3], size=20)
text(s, "18 test episodes where the model sees real RGB and predicts depth (ground truth available for every frame)", 0.6, 5.0, 12, 0.8, size=18, color=GREY)

s = new_slide("1. Generate many predictions, keep the good ones")
img(s, M / "p_select.jpg", 0.4, 1.2, h=5.3)
text(s, "407 rollouts (50 held-out episodes per task, up to 8 seeds)\n\n"
        "PullCubeTool: the cube vanishes\nPegInsertion: the peg never goes in\nStackCube: succeeds often\n\n"
        "kept = red cube ends on the green one\n(3D check, then by eye in all 4 views)\n+ most consistent across views\n\n"
        "25 kept, 18 different episodes", 6.3, 1.3, 6.6, 5.4, size=17)

s = new_slide("Finding: the generated views are shifted")
table(s, [["view", "where the model draws a cube vs its true projection (frame 0)", "spread"],
          ["0 (the input view)", "(-0.2, -0.3) px", "0.1 px"], ["1", "(-9, -16) px", "5-9 px"],
          ["2", "(0, +4) px", "7-10 px"], ["3", "(-16, +1) px", "5-10 px"]], col_w=[2.6, 7.4, 2.3], size=19)
text(s, "not a plain shift or zoom (corrections tried on the robot made the cubes worse), so one 3D scene can only match "
        "all 4 views to a few pixels.\nThe random seed decides most of it: seed 7 gave well-aligned views in about half its rollouts, most seeds in ~1 in 20.",
     0.6, 4.4, 12.2, 2, size=18)

s = new_slide("2. Robot: depth from Depth Anything 3, then silhouettes")
video(s, "p_opt3d_robot", 0.3, 1.3, w=7.6)
text(s, "Unknowns: 7 joint angles + gripper (base known)\n\n"
        "Depth: Depth Anything 3 on the 4 predicted views (known cameras), scaled per view so the table is at z = 0\n\n"
        "Loss: depth points above the table -> robot surface (robust), surface vs each view's depth\n\n"
        "Solver: Adam, 300 steps, 24 starting poses, keep the lowest loss; then frame by frame\n\n"
        "Then: joints refined so the robot's outline matches SAM 3's 'robot arm' mask in every view", 8.1, 1.3, 5.0, 5.6, size=15)

s = new_slide("3. Cubes: triangulated, no depth")
video(s, "p_opt3d_object", 0.3, 1.3, w=7.6)
text(s, "Masks: colour (SAM 3 misses most of these 8-pixel cubes)\n\n"
        "Position: blob centres from all views triangulated (RANSAC over views); near the last position or the gripper\n\n"
        "Size: from silhouette area (calibrated on real renders)\n\n"
        "Pose: Adam on silhouettes (+ weak depth), resting on the table; a cube cannot sink into another", 8.1, 1.3, 5.0, 5.6, size=15)

s = new_slide("Results: 25 predictions")
table(s, [["", "result"],
          ["task completed in the 3D fit (red cube on green)", f"{S['stacked_3d']} / {S['n']}"],
          ["cube size (true 40 mm)", f"{S['cube_side_mm_median']:.1f} mm (IQR {S['cube_side_mm_iqr'][0]}-{S['cube_side_mm_iqr'][1]})"],
          ["cube outline vs the predicted images (F@2px, median)", f"{S['obj_sil_f2px_median']:.2f}"],
          ["robot outline vs SAM 3 masks (IoU)", f"{S['robot_sil_iou_mean']:.2f}"],
          ["the predictions' own cross-view error (median)", f"{S['strict_view_err_px_median']:.1f} px"]], col_w=[8.0, 4.3], size=20)

for k, n in enumerate(sel):
    r = E[n]
    s = new_slide(f"Prediction {k + 1}: StackCube episode {n.split('episode-')[1].split('_s')[0]}, seed {n.split('_s')[-1]}")
    vid(s, n, y=1.15, max_h=5.15)
    st = r.get("stack_3d", {})
    text(s, f"cubes F@2px {r['objects']['red_cube']['sil_f2px']:.2f} / {r['objects']['green_cube']['sil_f2px']:.2f}  ·  sides {r['objects']['red_cube']['side_mm']:.0f} / {r['objects']['green_cube']['side_mm']:.0f} mm  ·  "
            f"red on green: {st.get('dxy_mm', 0):.0f} mm off, {st.get('dz_mm', 0):.0f} mm up  ·  view error {r['strict_view_err_px']:.0f} px", 0.6, 6.55, 12.2, 0.7, size=17, color=GREY)

s = new_slide("Summary: all 25")
data = [["prediction", "red on green (dxy, dz mm)", "cube F@2px", "robot IoU", "view err px"]]
for n in sel:
    r = E[n]; st = r.get("stack_3d", {})
    data.append([n.replace("StackCube-v1_episode-", ""), f"yes ({st.get('dxy_mm', 0):.0f}, {st.get('dz_mm', 0):.0f})" if r.get("stacked") else "no",
                 f"{r['objects']['red_cube']['sil_f2px']:.2f} / {r['objects']['green_cube']['sil_f2px']:.2f}", f"{r['robot_sil_iou']:.2f}", f"{r['strict_view_err_px']:.0f}"])
table(s, data, y=1.05, col_w=[3.0, 3.4, 2.4, 1.8, 1.7], size=10, row_h=0.22)

for sl in ALL:
    autoplay(sl)
out = HERE / "wm_predictions.pptx"; prs.save(out); print("wrote", out, len(prs.slides._sldIdLst), "slides")
