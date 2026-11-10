"""Text-prompted 2D masks from RGB only (Grounding-DINO boxes -> SAM 2.1 masks).

This is what the pixel-only baseline gets: no depth, no cameras, no simulator segmentation.
"""
import os

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/n/netscratch/ydu_lab/Lab/akiruga/hf_cache")

PROMPTS = {"standard": {"robot": "robot arm.", "objects": ["peg.", "box."]},
           "drawer": {"robot": "robot arm.", "objects": ["drawer."]},
           "real": {"robot": "robot arm.", "objects": ["small wooden block."]}}


class Segmenter:
    def __init__(self, dev="cuda"):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.dev = dev
        self.proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(dev).eval()
        self.sam = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device=dev)

    @torch.no_grad()
    def boxes(self, img, text, box_thr=0.3, text_thr=0.25):
        inp = self.proc(images=img, text=text, return_tensors="pt").to(self.dev)
        out = self.det(**inp)
        r = self.proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=box_thr,
                                                             text_threshold=text_thr, target_sizes=[img.shape[:2]])[0]
        return r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()

    @torch.no_grad()
    def masks(self, img, text, top=1, box_thr=0.3):
        b, s = self.boxes(img, text, box_thr)
        H, W = img.shape[:2]
        if len(b) == 0:
            return np.zeros((H, W), bool), None
        o = np.argsort(-s)[:top]
        self.sam.set_image(img)
        m, sc, _ = self.sam.predict(box=b[o], multimask_output=False)
        m = np.asarray(m).reshape(-1, H, W) > 0
        return m.any(0), b[o]


def segment_video(rgb, scene):
    """rgb [T,H,W,3] -> {'robot': [T,H,W] bool, 'objects': {prompt: [T,H,W]}}"""
    seg = Segmenter()
    P = PROMPTS[scene]
    out = {"robot": [], "objects": {p: [] for p in P["objects"]}}
    for t in range(len(rgb)):
        m, _ = seg.masks(rgb[t], P["robot"], top=1, box_thr=0.25)
        out["robot"].append(m)
        for p in P["objects"]:
            mo, _ = seg.masks(rgb[t], p, top=1)
            out["objects"][p].append(mo & ~m if p != "box." else mo)
    out["robot"] = np.stack(out["robot"])
    out["objects"] = {k: np.stack(v) for k, v in out["objects"].items()}
    return out
