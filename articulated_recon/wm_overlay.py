"""overlay grid of a wm fit: per frame (rows) x view (cols): predicted RGB with the fitted robot (blue) and
objects (orange) shaded on top, plus contours.  python wm_overlay.py <fit dir> <rollout dir> -> <fit dir>/overlay.jpg"""
import sys, pickle, numpy as np, torch, cv2, imageio.v3 as iio, trimesh
sys.path.insert(0, ".")
import artirecon  # noqa
from artirecon.data import load_roll
from artirecon.robot import URDFRobot
from artirecon.splat import render_points
F = pickle.load(open(sys.argv[1] + "/fit.pkl", "rb")); o = load_roll(sys.argv[2])
T, V, H, W = o.T, o.V, o.rgb.shape[2], o.rgb.shape[3]
rob = URDFRobot(n_pts_per_m2=150000)
with torch.no_grad():
    P, N = rob.surface(torch.tensor(F["rob"]["q"], dtype=torch.float32, device="cuda"), torch.tensor(F["Tb"], dtype=torch.float32, device="cuda")[None].expand(T, 4, 4))
P, N = P.cpu().numpy(), N.cpu().numpy()
osp = []
for ob in F["objects"]:
    tm = trimesh.Trimesh(ob["mesh"][0], ob["mesh"][1], process=False)
    sp, fi = trimesh.sample.sample_surface(tm, 30000, seed=0); osp.append((np.asarray(sp), np.asarray(tm.face_normals[fi]), ob))
rows = []
for t in range(T):
    tl = []
    for v in range(V):
        parts = [(P[t], N[t], (0.5, 0.65, 1.0))] + [(sp @ ob["poses"][t][:3, :3].T + ob["poses"][t][:3, 3], sn @ ob["poses"][t][:3, :3].T, (1.0, 0.55, 0.1)) for sp, sn, ob in osp]
        img, A, _, pid = render_points(parts, o.K[v], np.linalg.inv(o.c2w[v]), H, W, footprint=[2] + [1] * len(osp))
        img = img.cpu().numpy() if hasattr(img, "cpu") else img; A = A.cpu().numpy() if hasattr(A, "cpu") else A
        pid = pid.cpu().numpy() if hasattr(pid, "cpu") else pid
        from artirecon.objects_v2 import shift2d
        dx, dy = F["rob"].get("view_offsets", {}).get(v, (0, 0))        # model -> generated-image frame
        img, A, pid = shift2d(img, dx, dy), shift2d(A, dx, dy), shift2d(pid + 1, dx, dy) - 1
        base = o.rgb[t, v].astype(np.float32)
        a = A[..., None] * 0.45
        im = (base * (1 - a) + np.clip(img, 0, 1) * 255 * a).astype(np.uint8)
        for k in range(1 + len(osp)):
            cs, _ = cv2.findContours((pid == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, (60, 120, 255) if k == 0 else (255, 120, 0), 1)
        tl.append(im)
    rows.append(np.concatenate(tl, 1))
iio.imwrite(sys.argv[1] + "/overlay.jpg", np.concatenate(rows, 0)[::1, ::1], quality=85)
