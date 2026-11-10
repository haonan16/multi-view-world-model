"""SAM 3 text-prompted instance masks for every view/frame of a real episode.

python seg_sam3.py <episode dir with obs.npz> "<prompt 1>" ["<prompt 2>" ...]
-> <episode dir>/sam3.npz: for each instance: prompt id, t, v, score, box (x0,y0,x1,y1), mask (packed bits)
"""
import os
import sys
import time

import numpy as np
import torch
from transformers import Sam3Model, Sam3Processor

import json
args = sys.argv[1:]
if args and args[0] == "--many":          # python seg_sam3.py --many <list file>: one model load for all episodes
    jobs = [(l.strip(), None) for l in open(args[1]) if l.strip()]
else:
    jobs = [(args[0], args[1:] or None)]
dev = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if dev == "cuda" else torch.float32
proc = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3", dtype=DT).to(dev).eval()


def run_one(d, prompts):
  if os.environ.get("SAM3_PROMPTS"):
    prompts = os.environ["SAM3_PROMPTS"].split("|")
  if not prompts:
    prompts = json.load(open(os.path.join(d, "ann.json")))["prompts"]
  z = np.load(os.path.join(d, "obs.npz"))
  rgb = z["rgb"]                                    # T,V,H,W,3
  T, V, H, W, _ = rgb.shape

  rec = dict(p=[], t=[], v=[], score=[], box=[], mask=[])
  t0 = time.time()
  ims = [(t, v) for t in range(T) for v in range(V)]
  B = 10
  for pi, pr in enumerate(prompts):
      for i in range(0, len(ims), B):
          chunk = ims[i:i + B]
          inp = proc(images=[rgb[t, v] for t, v in chunk], text=[pr] * len(chunk), return_tensors="pt").to(dev)
          inp["pixel_values"] = inp["pixel_values"].to(DT)
          with torch.no_grad():
              out = model(**inp)
          res = proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5,
                                                        target_sizes=[(H, W)] * len(chunk))
          for (t, v), r in zip(chunk, res):
              for m, s, b in zip(r["masks"], r["scores"], r["boxes"]):
                  rec["p"].append(pi); rec["t"].append(t); rec["v"].append(v)
                  rec["score"].append(float(s)); rec["box"].append(b.float().cpu().numpy())
                  rec["mask"].append(np.packbits(m.cpu().numpy().astype(bool)))
      print(f"prompt '{pr}': {sum(1 for p in rec['p'] if p == pi)} instances", flush=True)
  print(f"{len(ims) * len(prompts)} images in {time.time() - t0:.1f}s", flush=True)
  np.savez_compressed(os.path.join(d, os.environ.get("SAM3_OUT", "sam3.npz")), prompts=np.array(prompts), p=np.array(rec["p"], int),
                      t=np.array(rec["t"], int), v=np.array(rec["v"], int), score=np.array(rec["score"]),
                      box=np.array(rec["box"]).reshape(-1, 4), mask=np.array(rec["mask"]).reshape(len(rec["p"]), -1),
                      shape=np.array([H, W]))


for d, pr in jobs:
    if os.path.exists(os.path.join(d, os.environ.get("SAM3_OUT", "sam3.npz"))) and pr is None and len(jobs) > 1:
        continue
    print(d, flush=True)
    run_one(d, pr)
