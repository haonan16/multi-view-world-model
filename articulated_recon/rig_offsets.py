"""Rig-level per-camera pixel correction, leave-one-episode-out: for each episode, each camera's offset is the
median (over the OTHER episodes) of mask-minus-model centroid offsets of the fitted objects seen by that camera
serial.  -> real/v2/<ep>/color_offset.json (read by real_v2 / objects_v2: masks shifted into the geometry frame)
python rig_offsets.py <episode list>"""
import collections, json, sys
import numpy as np
eps = open([a for a in sys.argv[1:] if not a.startswith('--')][0]).read().split()
D = collections.defaultdict(list)
for e in eps:
    m = json.load(open(f"results/real_v2/{e}/metrics.json")); ser = json.load(open(f"real/v2/{e}/meta.json"))["cam_serials"]
    for k, v in m.items():
        if k.endswith("offset_px_views"):
            for vi, o in enumerate(v):
                if o[0] == o[0]:
                    D[ser[vi]].append((e, o))
for e in eps:
    ser = json.load(open(f"real/v2/{e}/meta.json"))["cam_serials"]
    off = {}
    prev = {}
    if "--accumulate" in sys.argv:           # residuals were measured with the previous offsets applied
        prev = {int(k): v for k, v in json.load(open(f"real/v2/{e}/color_offset.json"))["offset_px"].items()}
    for vi, s in enumerate(ser):
        a = np.array([o for ee, o in D.get(s, []) if ee != e])
        r = np.median(a, 0) if len(a) >= 3 else np.zeros(2)
        off[vi] = [float(x) for x in np.round(np.asarray(prev.get(vi, [0, 0])) + r, 1)]
    json.dump(dict(offset_px=off, source="rig, leave-one-episode-out medians of object mask-vs-model offsets"),
              open(f"real/v2/{e}/color_offset.json", "w"), indent=1)
print({s: np.round(np.median(np.array([o for _, o in l]), 0), 1).tolist() for s, l in D.items()})
