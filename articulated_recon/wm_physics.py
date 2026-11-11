"""Contact post-processing of a wm fit: a moved cube whose footprint overlaps a static cube cannot sink into it ->
lifted to rest on its top (and never below the table).  Updates <fit dir>/fit.pkl (keeps fit_raw.pkl).
python wm_physics.py <fit dirs...>"""
import pickle, shutil, sys
import numpy as np
for d in sys.argv[1:]:
    p = d + "/fit.pkl"; raw = d + "/fit_raw.pkl"
    if not __import__("os").path.exists(raw):
        shutil.copy(p, raw)
    F = pickle.load(open(raw, "rb"))
    n, dd = F["plane"]
    st = [o for o in F["objects"] if o["type"] == "static"]
    nfix = 0
    for o in F["objects"]:
        if o["type"] == "static":
            continue
        h = o["half"][2]
        for t in range(len(o["poses"])):
            c = o["poses"][t][:3, 3]
            lo = h                                         # table
            for s in st:
                cs = s["poses"][t][:3, 3]; hs = s["half"]
                dxy = c - cs; dxy = dxy - (dxy @ n) * n
                if np.linalg.norm(dxy) < (hs[0] + o["half"][0]) * 0.9:     # footprints overlap
                    lo = max(lo, (cs @ n + dd) + hs[2] + h)
            hc = c @ n + dd
            if hc < lo - 1e-4:
                o["poses"][t][:3, 3] = c + (lo - hc) * n; nfix += 1
    pickle.dump(F, open(p, "wb"))
    print(d.split("/")[-1], "contact fixes", nfix)
