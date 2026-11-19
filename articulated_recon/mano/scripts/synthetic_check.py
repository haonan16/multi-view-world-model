"""Known-MANO multiview regression test, NOT a real-data performance claim.

Triangle-rendered depth and exact 2D landmarks; initialized near the true pose.
Camera 3 is excluded from every fitting loss. Evaluated on that camera and
against known vertices. No object, occlusion, calibration error, or cold start.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch

from mano_recon.fitting import SequenceFitter, project
from mano_recon.model import MANOHand, translation_base
from mano_recon.observations import Observations
from mano_recon.render import look_at, render


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--out", type=Path, default=Path("runs/synthetic-smoke"))
    p.add_argument("--without-keypoints", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    hand = MANOHand(samples=768)
    T, V, H = 5, 4, 160
    q = torch.zeros(T, hand.dof)
    q[:, :3] = torch.tensor([0.2, -0.1, 0.15])
    q[:, 3] = torch.linspace(-0.5, 0.5, T)
    q[:, 4] = torch.linspace(0.2, -0.2, T)
    translation = torch.stack((torch.linspace(0, .025, T), torch.zeros(T), torch.linspace(0, .015, T)), -1)
    beta = torch.tensor([[0.3, -0.2, 0.1, 0., 0., 0., 0., 0., 0., 0.]])
    with torch.no_grad():
        vertices, joints = hand(q, beta, translation_base(translation))
    center = vertices.mean((0, 1)).numpy()
    cameras = np.stack([look_at(center + offset, center) for offset in
                        ([.12,-.35,.28],[-.2,.3,.25],[.3,.18,-.25],[-.3,-.2,-.2])]).astype(np.float32)
    K = np.tile(np.array([[220, 0, 80], [0, 220, 80], [0, 0, 1]], np.float32), (V, 1, 1))
    depth = np.zeros((T, V, H, H), np.float32)
    rgb = np.zeros((T, V, H, H, 3), np.uint8)
    faces = hand.faces.numpy()
    for t in range(T):
        for v in range(V):
            rgb[t, v], depth[t, v] = render(vertices[t].numpy(), faces, K[v], cameras[v], H)
    Kt = torch.tensor(K)[None].expand(T,-1,-1,-1)
    ct = torch.tensor(cameras)[None].expand(T,-1,-1,-1)
    uv, _ = project(joints, Kt, torch.linalg.inv(ct))
    kp = np.concatenate((uv.numpy(), np.ones((T,V,21,1), np.float32)), -1)
    np.savez_compressed(args.out / "observations.npz", depth=depth, hand_mask=depth>0,
                        rgb=rgb, K=K, c2w=cameras, timestamps=np.arange(T)*.25,
                        depth_type="camera_z_m", camera_convention="opencv_c2w",
                        **({} if args.without_keypoints else {"keypoints": kp}))
    q0 = q + torch.randn_like(q)*.12
    tr0 = translation + torch.tensor([.018,-.012,.015])
    np.savez(args.out / "initialization.npz", q=q0.numpy(), translation=tr0.numpy())
    obs = Observations.load(args.out / "observations.npz")
    if args.without_keypoints:
        obs.keypoints = None
    start = time.monotonic()
    result = SequenceFitter(hand, obs.select_views([0,1,2])).fit(q0, tr0, iterations=args.iterations)
    elapsed = time.monotonic()-start
    with torch.no_grad():
        initial_v, initial_j = hand(q0, base=translation_base(tr0))
        fitted_v, fitted_j = hand(result["q"], result["betas"], translation_base(result["translation"]))
    def metrics(v, j):
        uv, _ = project(j, Kt[:,3:], torch.linalg.inv(ct[:,3:]))
        return dict(vertex_error_mm=float((v-vertices).norm(dim=-1).mean()*1000),
                    heldout_keypoint_error_px=float((uv-torch.tensor(kp[:,3:,:,:2])).norm(dim=-1).mean()))
    report = dict(test="synthetic MANO, near-truth initialization, clean triangle-rendered depth",
                  exact_keypoints_used_for_fitting=not args.without_keypoints,
                  frames=T, fit_views=[0,1,2], heldout_view=3,
                  initial=metrics(initial_v,initial_j), fitted=metrics(fitted_v,fitted_j),
                  seconds=elapsed, history=result["history"],
                  exclusions=["real video", "automatic initialization", "object fitting", "contact", "physics"])
    np.savez_compressed(args.out / "fit.npz", **{k:v.cpu().numpy() for k,v in result.items() if k!="history"},
                        vertices=fitted_v.numpy(), joints21=fitted_j.numpy(), faces=faces)
    (args.out / "metrics.json").write_text(json.dumps(report,indent=2))
    # Preview known synthetic trajectory, initialization, fitted result from held-out view.
    frames=[]
    for step in range(41):
        x=step/10; a=min(int(x),T-1); b=min(a+1,T-1); f=x-a
        panels=[]
        for vv,col in [(vertices,(105,174,220)),(initial_v,(225,145,92)),(fitted_v,(100,185,140))]:
            verts=(vv[a]*(1-f)+vv[b]*f).numpy()
            im,_=render(verts,faces,K[3],cameras[3],H,color=col)
            panels.append(im)
        frames.append(Image.fromarray(np.concatenate(panels,axis=1)).resize((960,320)))
    frames[0].save(args.out / "comparison.gif",save_all=True,append_images=frames[1:],duration=50,loop=0)
    frames[20].save(args.out / "comparison.png")
    print(json.dumps({k:v for k,v in report.items() if k!="history"},indent=2))
    if report["fitted"]["vertex_error_mm"] >= report["initial"]["vertex_error_mm"]*.5:
        raise SystemExit("Synthetic regression failed: geometry did not improve enough")
    if report["fitted"]["heldout_keypoint_error_px"] >= report["initial"]["heldout_keypoint_error_px"]*.5:
        raise SystemExit("Synthetic regression failed: held-out view did not improve enough")


if __name__ == "__main__":
    main()
