"""deck2/rh20t_v2.pptx: pipeline (with times) -> results best first (3-row videos, all 25) -> comparisons."""
import json
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from pptx.enum.text import PP_ALIGN

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE.parent / "deck"))
OLD = (HERE.parent.parent / "ego_world_model/ego_world_model_clean/articulated_recon/deck/build_deck.py")
src = OLD.read_text().split("# ---")[0]
src = src[:src.index("\n# ")] if "\n# " in src else src
exec(compile(src.split('\ndef autoplay')[0] + '\ndef autoplay' + src.split('\ndef autoplay')[1].split('\n\n\n')[0], "deck_helpers", "exec"))
M = HERE / "media"
globals()["M"] = M
R = HERE.parent / "results/real_v2"
eps = open(HERE.parent / "real/v2/final25.txt").read().split()
S = json.load(open(R / "summary_final10.json"))
H = {r["ep"]: r for r in json.load(open(R / "summary_ho0_final10.json"))["episodes"]}
E = {r["ep"]: r for r in S["episodes"]}


def img(slide, path, x, y, w=None, h=None):
    im = iio.imread(path)
    ar = im.shape[1] / im.shape[0]
    if w is None:
        w = h * ar
    if h is None:
        h = w / ar
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


def vid(slide, name, y=1.25, max_w=12.6, max_h=5.6):
    im = iio.imread(M / f"{name}.jpg")
    ar = im.shape[1] / im.shape[0]
    w = min(max_w, max_h * ar)
    return video(slide, name, (SW - w) / 2, y, w=w)


# ---------------------------------------------------------------- pipeline
s = new_slide("Input: multiview video + depth + cameras")
vid(s, "p_input", max_h=4.6)
text(s, "real RH20T robot episodes: 5–6 fixed RealSense views, sensor depth, factory calibration\n20 frames over the manipulation (~2 s apart)", 0.6, 6.1, 12, 1.2, size=22)

s = new_slide("1. Look and click: what moves, where it goes", "1 min / episode")
img(s, M / "p_ann.jpg", 2.2, 1.2, h=4.6)
text(s, "by eye: object, text prompt, role (moved / target), shape class, one click", 0.6, 6.1, 12, 1, size=22)

s = new_slide("2. SAM 3 finds it in every view and frame", "25 s")
img(s, M / "p_sam3.jpg", 0.4, 1.3, w=12.5)
text(s, "text-prompted instance masks; the clicked instance is followed in 3D with depth", 0.6, 6.2, 12, 1, size=22)

s = new_slide("3. Cameras refined on the robot, robot fitted", "4 min")
img(s, M / "p_robot.jpg", 0.4, 1.3, w=12.5)
text(s, "URDF fit from multiview depth only: 2.8 mm mean over 25 episodes (v1: 13 mm)", 0.6, 6.2, 12, 1, size=22)

s = new_slide("4. SAM 3D: initial mesh from one view", "45 s / object")
img(s, M / "p_sam3d_in.jpg", 1.0, 1.4, h=4.4)
text(s, "image + mask  →  mesh\n\nmesh competes with a simple\n2.5D outline; the better fit\nto all views wins", 8.0, 1.6, 4.9, 4, size=24)

s = new_slide("5. SAM 3 video tracker through the grasp", "15 s")
img(s, M / "p_track.jpg", 0.9, 1.25, h=4.75)
text(s, "keeps the object while it is in the gripper, where per-frame detection loses it", 0.6, 6.2, 12, 1, size=22)

s = new_slide("6. Shape and pose from silhouettes + depth", "1–2 min")
img(s, M / "p_crops.jpg", 3.4, 1.2, h=4.9)
text(s, "green: SAM 3 mask   magenta: model", 0.6, 6.3, 12, 1, size=22)

s = new_slide("Output: robot + objects, every camera", "≈ 7 min total")
vid(s, eps[0], max_h=5.0)
text(s, "rows: camera image · reconstruction overlaid · reconstruction alone", 0.6, 6.5, 12, 0.8, size=22)

# ---------------------------------------------------------------- results (best first)
rows = []
for e in eps:
    r = E[e]
    mv = [(n, o) for n, o in r["objects"].items() if o["role"] == "moved"]
    n, o = mv[0]
    ho = H[e]["objects"].get(n, {})
    rows.append((o["f2"], e, n, o, ho, r))
rows.sort(key=lambda x: -x[0])
task = lambda e: json.load(open(HERE.parent / "real/v2" / e / "meta.json"))["task"]
for k, (f2, e, n, o, ho, r) in enumerate(rows):
    group = "Best results" if k < 8 else ("Results" if k < 20 else "Hardest cases")
    s = new_slide(task(e))
    vid(s, e, y=1.15, max_h=5.15)
    hof = ho.get("f2_holdout")
    text(s, f"{group}  ·  {n}: silhouette F@2px {f2:.2f} (held-out view {hof:.2f})  ·  robot {r['robot_mm']:.1f} mm", 0.6, 6.55, 12.2, 0.7, size=20, color=GREY)

# ---------------------------------------------------------------- comparisons
def table(slide, data, y=1.4, col_w=None, size=22, row_h=0.55):
    nr, nc = len(data), len(data[0])
    col_w = col_w or [12.3 / nc] * nc
    tb = slide.shapes.add_table(nr, nc, Inches(0.5), Inches(y), Inches(sum(col_w)), Inches(row_h * nr)).table
    for i in range(nr):
        tb.rows[i].height = Inches(row_h)
    for j, w in enumerate(col_w):
        tb.columns[j].width = Inches(w)
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            c = tb.cell(i, j); c.text = str(val)
            for p in c.text_frame.paragraphs:
                for rr in p.runs:
                    rr.font.size = Pt(size); rr.font.name = FONT; rr.font.bold = (i == 0)
    return tb

su, sh_ = S["summary"], json.load(open(R / "summary_ho0_final10.json"))["summary"]
F = np.array([x[0] for x in rows]); FH = np.array([x[4].get("f2_holdout") or np.nan for x in rows])
s = new_slide("Fitted views vs a held-out view")
table(s, [["moved objects (25)", "fitted views", "held-out view"],
          ["silhouette F@2px, median", f"{np.median(F):.2f}", f"{np.nanmedian(FH):.2f}"],
          ["objects with F@2px ≥ 0.8", f"{(F >= 0.8).sum()} / 25", f"{(FH >= 0.8).sum()} / 25"],
          ["mask IoU, mean (every frame counts)", f"{su['moved_iou_mean']:.2f}", f"{sh_['moved_iou_holdout_mean']:.2f}"]],
      col_w=[6.3, 3, 3])
text(s, "held-out: one camera never used for tracking or fitting, only for scoring", 0.6, 4.2, 12, 1, size=22)

s = new_slide("Robot and grasps")
table(s, [["", "v1 (point-cloud objects)", "v2 (now)"],
          ["robot surface error", "13 mm", f"{su['robot_mm_mean']:.1f} mm (2.0–4.0 per episode)"],
          ["grasps detected", "–", "23 / 23 grasped objects"],
          ["grasp timing vs logged gripper", "–", f"{su['grasp_iou_mean']:.2f}"],
          ["objects resting on support (≤ 5 mm)", "–", "25 / 25"]], col_w=[5.3, 3.2, 3.8])

s = new_slide("What made the difference")
table(s, [["change", "effect"],
          ["per-frame camera refinement on the robot", "robot 13 → 3 mm"],
          ["silhouettes as main signal (depth bleeds at edges)", "cube IoU 0.41 → 0.60"],
          ["open containers instead of solid boxes", "cube in tray 0.44 → 0.70"],
          ["grasp from motion evidence", "grasp timing 0.59 → 0.84"],
          ["per-camera colour correction (other episodes)", "F@2px 0.86 → 0.91"],
          ["SAM 3 video tracker through the grasp", "IoU 0.54 → 0.63 (knife 0.09 → 0.38)"],
          ["SAM 3D mesh vs outline, best fit wins", "phone base 0.75 → 0.78"]], col_w=[7.3, 5.0], size=20)

s = new_slide("Summary: all 25 episodes, best first")
data = [["task", "object", "F@2px", "held-out"]]
for f2, e, n, o, ho, r in rows:
    data.append([task(e)[:48], n, f"{f2:.2f}", f"{ho.get('f2_holdout') or float('nan'):.2f}"])
table(s, data, y=1.05, col_w=[7.6, 2.0, 1.3, 1.4], size=10, row_h=0.235)

for sl in ALL:
    autoplay(sl)
out = HERE / "rh20t_v2.pptx"
prs.save(out)
print("wrote", out, len(prs.slides._sldIdLst), "slides")
