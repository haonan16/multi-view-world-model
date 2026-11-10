"""SAM 3D initial assets for the annotated non-box objects of many episodes, one model load.
python sam3d_batch.py <episode list>   (sam3d env)
For each object with shape 'hull' or 'mesh' (and 'sphere' skipped): image = hint view/frame RGB, mask = the
smallest SAM 3 mask (its prompt) under the click -> real/v2/<ep>/sam3d/<name>.ply (+ _cam.ply, .json) .
The mesh is decimated to ~20k faces for fitting."""
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sam3d_mesh  # noqa: E402

R = os.path.dirname(HERE)
eps = [l.strip() for l in open(sys.argv[1]) if l.strip()]
for e in eps:
    d = os.path.join(R, "real/v2", e)
    if not os.path.exists(os.path.join(d, "ann.json")) or not os.path.exists(os.path.join(d, "sam3.npz")):
        continue
    ann = json.load(open(os.path.join(d, "ann.json")))
    z = np.load(os.path.join(d, "obs.npz"))
    s3 = np.load(os.path.join(d, "sam3.npz"))
    H, W = [int(x) for x in s3["shape"]]
    prompts = list(s3["prompts"])
    for o in ann["objects"]:
        if o.get("shape", "box") not in ("hull", "mesh"):
            continue
        out = os.path.join(d, "sam3d", o["name"] + ".ply")
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        t, v, (u, vv) = o["hint"]["t"], o["hint"]["v"], o["hint"]["uv"]
        ids = np.nonzero((s3["p"] == prompts.index(o["prompt"])) & (s3["t"] == t) & (s3["v"] == v))[0]
        best = None
        for i in ids:
            m = np.unpackbits(s3["mask"][i])[:H * W].reshape(H, W).astype(bool)
            if m[int(vv), int(u)] and (best is None or m.sum() < best.sum()):
                best = m
        if best is None:
            print("no mask under click", e, o["name"], flush=True)
            continue
        ip, mp = out[:-4] + "_img.png", out[:-4] + "_mask.png"
        Image.fromarray(z["rgb"][t, v]).save(ip)
        Image.fromarray((best * 255).astype(np.uint8)).save(mp)
        dep = os.path.join(d, "sam3d", o["name"] + "_depth.npy")
        np.save(dep, z["depth_m"][t, v].astype(np.float32))
        K = z["K"][v]
        sys.argv = ["sam3d_mesh.py", "--image", ip, "--mask", mp, "--out", out, "--no_preview"]
        print("==", e, o["name"], flush=True)
        try:
            sam3d_mesh.main()
        except Exception as ex:  # keep going
            print("FAILED", e, o["name"], repr(ex), flush=True)
            continue
        try:
            import trimesh
            m = trimesh.load(out, force="mesh")
            if len(m.faces) > 20000:
                m = m.simplify_quadric_decimation(face_count=20000)
            m.export(out[:-4] + "_20k.ply")
            print("  faces", len(m.faces), "extents", np.round(m.extents, 3), flush=True)
        except Exception as ex:
            print("  decimation failed", repr(ex), flush=True)
