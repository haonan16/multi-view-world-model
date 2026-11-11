"""3-row video per selected world-model prediction: predicted RGB (4 views) / fitted robot + cubes overlaid /
the 3D fit alone (dark background).  -> deck3/media/<name>.mp4 (+ .jpg poster)   python wm_video.py <names...>"""
import sys, pickle, numpy as np, torch, cv2, imageio.v3 as iio, trimesh
sys.path.insert(0, ".")
import artirecon  # noqa
from artirecon.data import load_roll
from artirecon.robot import URDFRobot
from artirecon.splat import render_points
R = "/n/lab_storage/ydu_lab/Lab/akiruga/geo_runs/rollouts_A7000"
rob = URDFRobot(n_pts_per_m2=150000)
BG = np.array([30, 30, 34], np.float32)
for n in sys.argv[1:]:
    F = pickle.load(open(f"results/wm/{n}_da3_final/fit.pkl", "rb")); o = load_roll(f"{R}/{n}")
    T, V, H, W = o.T, o.V, o.rgb.shape[2], o.rgb.shape[3]
    with torch.no_grad():
        P, N = rob.surface(torch.tensor(F["rob"]["q"], dtype=torch.float32, device="cuda"), torch.tensor(F["Tb"], dtype=torch.float32, device="cuda")[None].expand(T, 4, 4))
    P, N = P.cpu().numpy(), N.cpu().numpy()
    osp = []
    for ob in F["objects"]:
        tm = trimesh.Trimesh(ob["mesh"][0], ob["mesh"][1], process=False)
        sp, fi = trimesh.sample.sample_surface(tm, 30000, seed=0)
        col = (0.95, 0.2, 0.15) if ob["name"].startswith("red") else (0.2, 0.8, 0.3)
        osp.append((np.asarray(sp), np.asarray(tm.face_normals[fi]), ob, col))
    frames = []
    for t in range(T):
        r0, r1, r2 = [], [], []
        for v in range(V):
            parts = [(P[t], N[t], (0.78, 0.86, 0.98))] + [(sp @ ob["poses"][t][:3, :3].T + ob["poses"][t][:3, 3], sn @ ob["poses"][t][:3, :3].T, col) for sp, sn, ob, col in osp]
            img, A, _, _ = render_points(parts, o.K[v], np.linalg.inv(o.c2w[v]), H, W, footprint=[2] + [1] * len(osp))
            img = np.asarray(img.cpu() if hasattr(img, "cpu") else img); A = np.asarray(A.cpu() if hasattr(A, "cpu") else A)
            rend = np.clip(img, 0, 1) * 255
            a = A[..., None].astype(np.float32)
            ov = (o.rgb[t, v] * (1 - 0.6 * a) + rend * 0.6 * a).astype(np.uint8)
            solo = (BG * (1 - a) + rend * a).astype(np.uint8)
            r0.append(o.rgb[t, v]); r1.append(ov); r2.append(solo)
        fr = np.concatenate([np.concatenate(r, 1) for r in (r0, r1, r2)], 0)
        frames += [fr] * 4
    iio.imwrite(f"deck3/media/{n}.mp4", np.stack(frames), fps=4, codec="libx264", macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "20"])
    iio.imwrite(f"deck3/media/{n}.jpg", frames[-1], quality=90)
    print("video", n, flush=True)
