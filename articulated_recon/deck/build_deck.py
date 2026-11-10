"""deck/articulated_recon.pptx from deck/media (python-pptx; videos autoplay + loop)."""
import copy
import os
from pathlib import Path

import imageio.v3 as iio
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "deck"
M = ROOT / "media"
SW, SH = 13.333, 7.5
BLACK, GREY = RGBColor(0x11, 0x11, 0x11), RGBColor(0x55, 0x55, 0x55)
FONT = "Arial"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
BLANK = prs.slide_layouts[6]
ALL = []


def text(slide, s, x, y, w, h, size=24, bold=False, color=BLACK, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, FONT
        r.font.color.rgb = color
    return tb


def new_slide(title, time=None):
    s = prs.slides.add_slide(BLANK)
    text(s, title, 0.5, 0.3, 9.6 if time else 12.3, 0.8, size=34, bold=True)
    if time:
        text(s, time, 9.9, 0.33, 2.95, 0.8, size=28, color=GREY, align=PP_ALIGN.RIGHT)
    s._videos = []
    ALL.append(s)
    return s


def video(slide, name, x, y, w=None, h=None):
    """place deck/media/<name>.mp4 at (x, y) with width w or height h (aspect kept)"""
    im = iio.imread(M / f"{name}.jpg")
    ar = im.shape[1] / im.shape[0]
    if w is None:
        w = h * ar
    if h is None:
        h = w / ar
    mv = slide.shapes.add_movie(str(M / f"{name}.mp4"), Inches(x), Inches(y), Inches(w), Inches(h),
                                poster_frame_image=str(M / f"{name}.jpg"), mime_type="video/mp4")
    slide._videos.append(mv.shape_id)
    return x, y, w, h


def centred(slide, name, y, max_w, max_h):
    im = iio.imread(M / f"{name}.jpg")
    ar = im.shape[1] / im.shape[0]
    w = min(max_w, max_h * ar)
    return video(slide, name, (SW - w) / 2, y, w=w)


NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def autoplay(slide):
    """replace python-pptx's click-to-play timing: every video starts automatically and loops"""
    ids = slide._videos
    if not ids:
        return
    sld = slide._element
    for t in sld.findall(f"{{{NS}}}timing"):
        sld.remove(t)
    n = [10]

    def nid():
        n[0] += 1
        return n[0]
    calls = "".join(
        f'<p:par><p:cTn id="{nid()}" presetID="1" presetClass="mediacall" presetSubtype="0" fill="hold" '
        f'nodeType="{"afterEffect" if i == 0 else "withEffect"}"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
        f'<p:childTnLst><p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr><p:cTn id="{nid()}" dur="1" fill="hold"/>'
        f'<p:tgtEl><p:spTgt spid="{sp}"/></p:tgtEl></p:cBhvr></p:cmd></p:childTnLst></p:cTn></p:par>'
        for i, sp in enumerate(ids))
    media = "".join(
        f'<p:video><p:cMediaNode vol="80000" mute="1"><p:cTn id="{nid()}" repeatCount="indefinite" fill="hold" display="0">'
        f'<p:stCondLst><p:cond delay="indefinite"/></p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="{sp}"/></p:tgtEl>'
        f'</p:cMediaNode></p:video>' for sp in ids)
    xml = (f'<p:timing xmlns:p="{NS}"><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot">'
           f'<p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq">'
           f'<p:childTnLst><p:par><p:cTn id="3" fill="hold"><p:stCondLst><p:cond delay="indefinite"/>'
           f'<p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond></p:stCondLst><p:childTnLst>'
           f'<p:par><p:cTn id="4" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
           f'{calls}</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn>'
           f'<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
           f'<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst>'
           f'</p:seq>{media}</p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>')
    el = etree.fromstring(xml)
    ext = sld.find(f"{{{NS}}}extLst")
    if ext is not None:
        ext.addprevious(el)
    else:
        sld.append(el)


def labels(slide, items, x0, y, total_w, size=22):
    w = total_w / len(items)
    for i, s in enumerate(items):
        text(slide, s, x0 + i * w, y, w, 1.2, size=size, align=PP_ALIGN.CENTER)


# ------------------------------------------------------------------ pipeline
s = new_slide("Input: multiview video + depth + cameras")
x, y, w, h = video(s, "p1_input_grid", 1.3, 1.2, w=11.55)
text(s, "RGB", 0.3, y + h * 0.2, 1.0, 0.5, size=18, color=GREY)
text(s, "depth", 0.3, y + h * 0.7, 1.0, 0.5, size=18, color=GREY)
x, y, w, h = video(s, "p1_cameras", 0.5, 4.95, h=2.35)
text(s, "4 calibrated cameras\n11 frames (1 per second)", x + w + 0.4, 5.5, 6.0, 1.4, size=24)

s = new_slide("1. Lift depth into 3D points", "< 0.1 s")
centred(s, "p2_lift", 1.25, 12.3, 6.0)

s = new_slide("2. Find the table, keep what is above it", "0.1 s")
centred(s, "p3_table", 1.25, 12.3, 6.0)

s = new_slide("3. Fit the robot URDF: 32 starting guesses", "43 s")
centred(s, "p4_search", 1.25, 12.3, 6.0)

s = new_slide("4. Track the robot in every frame", "7 s / frame")
centred(s, "p5_track", 1.25, 12.3, 6.0)

s = new_slide("5. Leftover points that move = objects", "1 s")
centred(s, "p6_residual", 1.25, 12.3, 5.4)
text(s, "red: a camera now sees through the point, so it moved", 0.5, 6.75, 12.3, 0.6, size=22, color=GREY,
     align=PP_ALIGN.CENTER)

s = new_slide("6. Fit objects as boxes, plus physics", "25 s")
x, y, w, h = video(s, "p7_objects", 0.5, 1.4, w=6.05)
x2, _, _, _ = video(s, "p7_objects_close", 6.78, 1.4, w=6.05)
text(s, "peg tracked in 3D", 0.5, y + h + 0.15, w, 0.6, size=22, align=PP_ALIGN.CENTER)
text(s, "hole inferred from contact", x2, y + h + 0.15, w, 1.0, size=22, align=PP_ALIGN.CENTER)

s = new_slide("Output: seen from all 4 cameras", "total ≈ 3 min")
x, y, w, h = centred(s, "p8_overlay", 1.4, 12.3, 4.4)
text(s, "fast setting: 9 s per clip", 0.5, y + h + 0.3, 12.3, 0.6, size=24, color=GREY, align=PP_ALIGN.CENTER)

# ------------------------------------------------------------------ results
def result(title, ov, d3, big):
    s = new_slide(title)
    x, y, w, h = video(s, ov, 0.5, 1.3, h=5.8)
    rest = SW - (x + w) - 0.9
    x3, y3, w3, h3 = video(s, d3, x + w + 0.4, 1.3, w=rest)
    text(s, big, x3, y3 + h3 + 0.3, w3, 2.4, size=28)
    return s


result("Peg insertion · sim · 4 views + depth", "r_peg_overlay", "r_peg_3d",
       "robot 1.2 mm\npeg 3.7 mm\ntarget box 6.5 mm")
result("Drawer · sim · 4 views + depth", "r_drawer_overlay", "r_drawer_3d",
       "robot 2.1 mm\njoint axis 0.0° off\ntravel 1.6 mm off")

s = new_slide("Simulator-ready: replayed in SAPIEN")
x, y, w, h = centred(s, "r_sapien_drawer", 1.2, 12.3, 2.8)
centred(s, "r_sapien_standard", y + h + 0.2, 12.3, 2.8)

s = new_slide("Real robot (RH20T) · 5 RealSense cameras")
x, y, w, h = video(s, "r_real_overlay", 0.5, 1.3, w=7.4)
x3, y3, w3, h3 = video(s, "r_real_3d", x + w + 0.4, 1.3, w=SW - (x + w) - 0.9)
text(s, "robot 13 mm\nblock not recovered", x3, y3 + h3 + 0.3, w3, 1.6, size=28)

s = new_slide("Real RGB + world-model depth · 4 views")
x, y, w, h = video(s, "r_modelB_overlay", 0.5, 1.3, w=8.3)
text(s, "robot 19 mm\n(27 mm without\npixel term)\n\nobjects unreliable", x + w + 0.3, 1.5, 3.7, 4.0, size=28)

s = new_slide("World-model rollout · 1 view → 4 views + depth")
x, y, w, h = video(s, "r_rollA_overlay", 0.5, 1.3, w=8.3)
text(s, "robot 40 mm\n\nthe rollout drifts\nfrom what really\nhappened", x + w + 0.3, 1.5, 3.7, 4.0, size=28)

# ------------------------------------------------------------------ comparisons
s = new_slide("Comparison: input (peg, sim)")
x, y, w, h = centred(s, "c_inputs_standard", 1.3, 12.3, 4.2)
labels(s, ["4 views + depth\nrobot 1.2 · peg 3.7 mm", "1 view + depth\nrobot 2.8 · peg 15 mm",
           "1 view, pixels only\nrobot 23 · peg 131 mm"], x, y + h + 0.2, w)

s = new_slide("Comparison: input (drawer, sim)")
x, y, w, h = centred(s, "c_inputs_drawer", 1.3, 12.3, 4.2)
labels(s, ["4 views + depth\naxis 0° · travel 1.6 mm", "1 view + depth\naxis 47° · travel 45 mm",
           "1 view, pixels only\naxis 127° · travel 490 mm"], x, y + h + 0.2, w)

s = new_slide("Comparison: input (real robot)")
x, y, w, h = centred(s, "c_inputs_real", 1.4, 12.3, 3.6)
labels(s, ["5 views + depth\n23.5 mm", "+ camera recalib.\n13.0 mm", "1 view + depth\n15.6 mm",
           "pixels only\n100 mm"], x, y + h + 0.2, w)

s = new_slide("Comparison: where the depth comes from (4 views)")
x, y, w, h = centred(s, "c_depth_source", 1.25, 12.3, 4.4)
labels(s, ["true depth\n1.3 mm", "model B depth\n19 mm", "model A rollout\n40 mm", "model B rollout\n110 mm"],
       x, y + h + 0.2, w)

s = new_slide("Summary: robot error (mm)")
rows = [("Sim, 4 views + depth", "1.2"), ("Sim, 1 view + depth", "2.8"), ("Real, 5 views + depth", "13"),
        ("Real RGB + model depth, 4 views", "19"), ("Sim, pixels only", "23"), ("Model A rollout, 4 views", "40"),
        ("Real, pixels only", "100")]
tbl = s.shapes.add_table(len(rows), 2, Inches(1.5), Inches(1.4), Inches(10.3), Inches(0.7 * len(rows))).table
tbl.columns[0].width, tbl.columns[1].width = Inches(8.0), Inches(2.3)
for i, (a, b) in enumerate(rows):
    for j, v in enumerate((a, b)):
        c = tbl.cell(i, j)
        c.text = v
        c.fill.background()
        p = c.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.RIGHT if j else PP_ALIGN.LEFT
        r = p.runs[0]
        r.font.size, r.font.name, r.font.color.rgb = Pt(26), FONT, BLACK
# plain table: drop the default table style banding
tblPr = tbl._tbl.tblPr
tblPr.set("firstRow", "0"); tblPr.set("bandRow", "0")

for sl in ALL:
    autoplay(sl)
prs.save(ROOT / "articulated_recon.pptx")
print("saved", ROOT / "articulated_recon.pptx")
