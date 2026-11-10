"""SAM 3 video tracker masks for the moved objects: per view, seed with the earliest SAM 3 instance the object
stage selected in that view (results/real_v2/<ep>/fit.pkl 'sel'), propagate forward and backward through the 20
frames.  The tracker keeps an object through partial occlusion (e.g. in the gripper), where per-frame text
detection loses it.  -> real/v2/<ep>/sam3_track.npz (same format as sam3.npz; prompt '__track__<name>')
python seg_track.py <episode list>"""
import os
import pickle
import sys
import time

import numpy as np
import torch
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from artirecon.objects_v2 import Sam3Masks  # noqa: E402

dev = "cuda"
model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to(dev).eval()
proc = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

for e in open(sys.argv[1]).read().split():
    d = os.path.join(ROOT, "real/v2", e)
    t0 = time.time()
    z = np.load(os.path.join(d, "obs.npz"))
    rgb = z["rgb"]
    T, V, H, W, _ = rgb.shape
    res = pickle.load(open(os.path.join(ROOT, "results/real_v2", e, "fit.pkl"), "rb"))
    sam = Sam3Masks(os.path.join(d, "sam3.npz"), H, W)
    prompts, rec = [], dict(p=[], t=[], v=[], score=[], box=[], mask=[])
    for ob in res["objects"]:
        if ob["type"] != "free":
            continue
        pi = len(prompts); prompts.append("__track__" + ob["name"])
        for v in range(V):
            seed = next(((t, s[v]) for t, s in sorted(ob["sel"].items()) if v in s), None)
            if seed is None:
                continue
            ts, sid = seed
            m0 = sam.mask(sid, raw=True)
            with torch.no_grad():
                sess = proc.init_video_session(video=[rgb[t, v] for t in range(T)], inference_device=dev, dtype=torch.bfloat16)
                proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=ts, obj_ids=1,
                                                     input_masks=m0.astype(np.float32))
                outs = {}
                for rev in (False, True):
                    for out in model.propagate_in_video_iterator(sess, start_frame_idx=ts, reverse=rev):
                        mk = proc.post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
                        outs[out.frame_idx] = mk[0, 0].cpu().numpy().astype(bool) if mk.ndim == 4 else mk[0].cpu().numpy().astype(bool)
            for t, m in outs.items():
                if m.sum() < 15:
                    continue
                ys, xs = np.nonzero(m)
                rec["p"].append(pi); rec["t"].append(t); rec["v"].append(v); rec["score"].append(1.0)
                rec["box"].append(np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32))
                rec["mask"].append(np.packbits(m))
    np.savez_compressed(os.path.join(d, "sam3_track.npz"), prompts=np.array(prompts or ["__none__"]), p=np.array(rec["p"], int),
                        t=np.array(rec["t"], int), v=np.array(rec["v"], int), score=np.array(rec["score"]),
                        box=np.array(rec["box"]).reshape(-1, 4), mask=np.array(rec["mask"]).reshape(len(rec["p"]), -1),
                        shape=np.array([H, W]))
    print(e, prompts, len(rec["p"]), "masks", f"{time.time() - t0:.0f}s", flush=True)
