"""results/real_v2/site/index.html: summary table + per-episode overlay video, SAM 3 / model contours, object crops.
python build_site_v2.py real/v2/final25.txt  (run after summarize_v2.py)"""
import html
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
R = ROOT / "results/real_v2"
S = R / "site"
S.mkdir(exist_ok=True)
eps = open(sys.argv[1]).read().split()
summ = json.load(open(R / "summary.json"))
ho = json.load(open(R / "summary_ho0.json")) if (R / "summary_ho0.json").exists() else None
md = (R / "SUMMARY.md").read_text()
rows = []
for e in eps:
    d = R / e
    if not (d / "metrics.json").exists():
        continue
    files = ["overlay.mp4", "overlay_grid.jpg", "masks_grid.jpg"] + sorted(p.name for p in d.glob("crops_*.jpg"))
    ann = json.load(open(ROOT / "real/v2" / e / "ann.json"))
    m = json.load(open(d / "metrics.json"))
    objs = "".join(f"<li><b>{html.escape(o['name'])}</b> ({o['role']}, {o.get('shape','box')}, prompt “{html.escape(o['prompt'])}”): "
                   f"IoU {m.get('obj/' + o['name'] + '/iou_mean', float('nan')):.2f}, silhouette F@2px {m.get('obj/' + o['name'] + '/sil_f2px', float('nan')):.2f}, "
                   f"extents {m.get('obj/' + o['name'] + '/extents_mm', '–')} mm</li>" for o in ann["objects"] if 'obj/' + o['name'] + '/iou_mean' in m)
    crops = "".join(f'<figure><img src="{e}/{p}" loading="lazy"><figcaption>{p[6:-4]}: rows = frames, columns = views; '
                    f'green SAM 3 mask, magenta model</figcaption></figure>' for p in files if p.startswith("crops_"))
    import numpy as np
    rows.append(f"""<section><h2>{e}</h2><p>{html.escape(ann.get('note', ''))}</p>
<p>robot {np.mean(m['robot_surface_mm']):.1f} mm (gripper {np.mean(m['gripper_surface_mm']):.1f} mm, joints {np.mean(m['joint_err_deg']):.2f}°)</p><ul>{objs}</ul>
<video src="{e}/{"overlay3.mp4" if (d / "overlay3.mp4").exists() else "overlay.mp4"}" controls muted loop preload="none" poster="{e}/overlay_grid.jpg"></video>
<details><summary>SAM 3 vs model contours, all frames</summary><img src="{e}/masks_grid.jpg" loading="lazy"></details>
<div class="crops">{crops}</div></section>""")
page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RH20T v2 reconstruction</title><style>
body{{font-family:Arial,sans-serif;margin:16px;max-width:1400px;background:#fff;color:#111}} img,video{{max-width:100%}}
table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid #ccc;padding:3px 6px}} section{{border-top:1px solid #ccc;margin-top:24px}}
.crops{{display:flex;flex-wrap:wrap;gap:12px}} .crops figure{{margin:0;max-width:48%}} pre{{white-space:pre-wrap}}
</style></head><body><h1>Real RH20T: robot + objects, 25 episodes</h1>
<pre>{html.escape(md.split('|')[0])}</pre>{'<pre>' + html.escape((R / 'SUMMARY_ho0.md').read_text().split('|')[0]) + '</pre>' if ho else ''}
<details open><summary>table</summary>{md_table(md) if False else ''}<pre>{html.escape(md[md.index('|'):])}</pre></details>
{''.join(rows)}</body></html>"""
(S / "index.html").write_text(page)
print("wrote", S / "index.html", len(rows), "episodes")
