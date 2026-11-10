"""recompute support contact from fit.pkl: distance of the object's lowest point to the nearer of the table and the
estimated support, over frames where it is not held (flags floating objects without trusting a bad support estimate)"""
import json, pickle, sys
import numpy as np
for e in open(sys.argv[1]).read().split():
    for tag in ("", "_ho0"):
        d = f"results/real_v2/{e}{tag}"
        r = pickle.load(open(d + "/fit.pkl", "rb")); m = json.load(open(d + "/metrics.json"))
        n, dd = r["plane"]
        for o in r["objects"]:
            V = o["mesh"][0]; T = len(o["poses"])
            h = np.array([(((o["poses"][t][:3, :3] @ V.T).T + o["poses"][t][:3, 3]) @ n + dd).min() for t in range(T)])
            free = ~o["held"]
            c = np.minimum(np.abs(h - o.get("support", np.zeros(T))), np.abs(h))
            m[f"obj/{o['name']}/rest_contact_mm"] = float(c[free].mean() * 1000) if free.any() else None
        json.dump(m, open(d + "/metrics.json", "w"), indent=1)
print("ok")
