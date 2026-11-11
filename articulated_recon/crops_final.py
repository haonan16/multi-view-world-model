"""close-up crops around the target (green cube) at the last 2 frames, 4 views -> one row per rollout"""
import sys, json, numpy as np, imageio.v3 as iio, cv2
R = "/n/lab_storage/ydu_lab/Lab/akiruga/geo_runs/rollouts_A7000"
s = json.load(open(R + "/scores.json")); out = sys.argv[1]; rows = []
for n in sys.argv[2:]:
    d = R + "/" + n; z = np.load(d + "/dense.npz"); r = z["rgb_pred"]; T = r.shape[1]; m = json.load(open(d + "/meta.json"))
    K = np.array(m["K"]); c2w = np.load("/n/lab_storage/ydu_lab/Lab/akiruga/maniskill_render/" + m["episode"] + "/cameras.npz")["c2w"][:4]
    ob = s[n]["objects"]; key = "green" if "green" in ob else list(ob)[-1]
    p = [q for q in ob[key]["pos"] if q][-1]
    tiles = []
    for t in [T - 2, T - 1]:
        for v in range(4):
            pc = np.linalg.inv(c2w[v]) @ np.r_[p, 1]; u, vv = (K[v] @ pc[:3])[:2] / pc[2]
            x0, y0 = int(u) - 40, int(vv) - 55
            big = np.pad(r[v, t], ((100, 100), (100, 100), (0, 0)))
            tiles.append(cv2.resize(big[y0 + 100:y0 + 180, x0 + 100:x0 + 180], (140, 140), interpolation=cv2.INTER_CUBIC))
    row = np.concatenate(tiles, 1); cv2.putText(row, n[-14:], (3, 14), 0, 0.45, (255, 255, 0), 1); rows.append(row)
iio.imwrite(out, np.concatenate(rows, 0))
