"""full-res annotation crops: views (a,b) x frames (0, last) with a labelled 40 px grid -> real/v2/<ep>/ann2.jpg
python ann_sheet2.py <ep> [va vb] [t0 t1]"""
import sys, numpy as np, cv2
ep = sys.argv[1]
z = np.load(f"/n/lab_storage/ydu_lab/Lab/akiruga/articulated_recon/real/v2/{ep}/obs.npz")
r = z["rgb"]; T, V = r.shape[:2]
vs = [int(x) for x in sys.argv[2:4]] if len(sys.argv) > 3 else [1, 2]
ts = [int(x) for x in sys.argv[4:6]] if len(sys.argv) > 5 else [0, T - 1]
rows = []
for t in ts:
    ims = []
    for v in vs:
        im = np.ascontiguousarray(r[t, v][..., ::-1]).copy()
        for x in range(0, 640, 40):
            cv2.line(im, (x, 0), (x, 359), (0, 255, 0), 1); cv2.putText(im, str(x), (x + 1, 10), 0, 0.3, (0, 255, 255), 1)
        for y in range(0, 360, 40):
            cv2.line(im, (0, y), (639, y), (0, 255, 0), 1); cv2.putText(im, str(y), (1, y - 1), 0, 0.3, (0, 255, 255), 1)
        cv2.putText(im, f"t{t} v{v}", (300, 350), 0, 0.6, (255, 255, 255), 2)
        ims.append(im)
    rows.append(np.concatenate(ims, 1))
cv2.imwrite(f"/n/lab_storage/ydu_lab/Lab/akiruga/articulated_recon/real/v2/{ep}/ann2.jpg", np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
