"""GT vs model videos straight from infer_multiview.py outputs (no 3D step).

One mp4 per held-out scene: rows GT / tokens / tiled, columns = the 4 cameras, each view
2x Lanczos-upscaled; frame 0 (the input) is marked. Also writes <out>/../scenes.json for
the simple video page.

Usage: python scripts/rh20t/make_compare_videos.py INFER_OUT OUT_DIR/vid
"""
import json, sys
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import matplotlib
from PIL import Image, ImageDraw, ImageFont
R = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
m = json.loads((R / "metrics.json").read_text())
ck = m["ckpt"]
FONT = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
F = lambda s: ImageFont.truetype(str(FONT), s)
UP, LBL, HDR = 2, 230, 34
rows = [("gt", "Ground truth"), ("pred_tokens", f"Tokens\nstep {ck['tokens']['global_step']//1000}k"),
        ("pred_tiled", f"Tiled\nstep {ck['tiled']['global_step']//1000}k")]
index = []
for i, s in enumerate(m["scenes"]):
    arr = {k: np.load(R / s / f"{k}.npy") for k, _ in rows}          # V,T,H,W,3
    V, T, H, W, _ = arr["gt"].shape
    h, w = H * UP, W * UP
    frames = []
    for t in range(T):
        im = Image.new("RGB", (LBL + V * w + (V - 1) * 4, HDR + 3 * h + 2 * 4), (18, 20, 22))
        d = ImageDraw.Draw(im)
        for v in range(V):
            d.text((LBL + v * (w + 4) + 8, 6), f"camera {v+1}", fill=(200, 205, 210), font=F(18))
        for r, (k, name) in enumerate(rows):
            y = HDR + r * (h + 4)
            d.multiline_text((12, y + h // 2 - 22), name, fill=(235, 238, 240), font=F(20), spacing=6)
            for v in range(V):
                tile = Image.fromarray(arr[k][v, t]).resize((w, h), Image.LANCZOS)
                im.paste(tile, (LBL + v * (w + 4), y))
        tag = "t = 0  (input frame)" if t == 0 else f"t = {t} / {T-1}  (generated)"
        d.text((12, 8), tag, fill=(255, 196, 80) if t == 0 else (150, 155, 160), font=F(16))
        frames.append(np.array(im))
    imageio.mimwrite(out / f"{i}.mp4", frames, fps=3, codec="libx264", quality=9,
                     macro_block_size=2, pixelformat="yuv420p")
    ps = m["per_scene"][s]
    index.append({"i": i, "name": s.split("_", 1)[1], "metrics": ps})
    print(i, s, flush=True)
(out.parent / "scenes.json").write_text(json.dumps({"ckpt": ck, "mean": m["mean"], "scenes": index}))
