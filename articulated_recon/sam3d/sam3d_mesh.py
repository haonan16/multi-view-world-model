#!/usr/bin/env python
"""SAM 3D Objects: single RGB image + binary mask -> 3D mesh (+ predicted pose).

Usage (see run_sam3d.sh for env setup):
  python sam3d_mesh.py --image img.png --mask mask.png --out out.ply [--seed 0]
       [--depth depth.npy --K fx,fy,cx,cy]   # optional metric depth -> metric camera-frame pose

Outputs (for --out X.ply):
  X.ply        mesh in the model's canonical object frame (z-up, roughly in [-0.5,0.5]^3),
               per-vertex RGB colours. This is the frame the predicted pose acts on.
  X_cam.ply    same mesh posed into the OpenCV camera frame (x right, y down, z forward).
               Units: MoGe's (approx metric, scale-ambiguous) if no --depth, metres if --depth.
  X_gs.ply     3D Gaussian splat (canonical frame, 3DGS ply format).
  X.json       pose: quaternion (wxyz), scale, translation, 4x4 matrices, conventions.
  X_preview.png  4-view render of the canonical mesh + overlay of X_cam projected on the image.

Pose convention (from sam3d_objects/utils/visualization/scene_visualizer.py, PyTorch3D
row-vector Transform3d(scale).rotate(R).translate(t)):
  p_p3d = (s * p_local) @ R + t,   R = quaternion_to_matrix(q_wxyz)
  p_opencv = p_p3d * [-1, -1, 1]   (PyTorch3D camera: x left, y up, z forward)
X.json["T_obj2cam_opencv"] is the 4x4 column-vector matrix with p_cv = T @ [p_local, 1].
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = os.path.dirname(os.path.dirname(sys.executable))
os.environ.setdefault("CUDA_HOME", os.environ["CONDA_PREFIX"])
sys.path.insert(0, REPO)

import numpy as np
from PIL import Image
import torch

_PIPE = None


def _enable_lowvram():
    """Load every sub-model on CPU and move it to the GPU only while its stage runs.
    Peak VRAM ~= largest stage (ss_generator fp32 ~6.7 GB + activations) instead of ~16 GB."""
    from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
    from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
    from sam3d_objects.pipeline.depth_models import base as depth_base

    orig_load = InferencePipeline.instantiate_and_load_from_pretrained

    def load_cpu(self, config, ckpt_path, state_dict_fn=None, state_dict_key="state_dict", device="cuda"):
        with torch.device("cpu"):
            return orig_load(self, config, ckpt_path, state_dict_fn=state_dict_fn,
                             state_dict_key=state_dict_key, device="cpu")
    InferencePipeline.instantiate_and_load_from_pretrained = load_cpu

    orig_dm_init = depth_base.DepthModel.__init__

    def dm_init(self, model, device="cuda"):
        orig_dm_init(self, model, device="cpu")
        self.device = torch.device("cuda")  # inputs go to cuda; model is moved per call
    depth_base.DepthModel.__init__ = dm_init

    def staged(fn, getmods):
        def wrapped(self, *a, **k):
            mods = [m for m in getmods(self) if m is not None]
            for m in mods:
                m.to("cuda")
            try:
                return fn(self, *a, **k)
            finally:
                for m in mods:
                    m.to("cpu")
                torch.cuda.empty_cache()
        return wrapped

    P = InferencePipelinePointMap
    P.compute_pointmap = staged(P.compute_pointmap, lambda s: [s.depth_model.model])
    P.sample_sparse_structure = staged(
        P.sample_sparse_structure,
        lambda s: [s.models["ss_generator"], s.models["ss_decoder"], s.condition_embedders["ss_condition_embedder"]])
    P.sample_slat = staged(
        P.sample_slat, lambda s: [s.models["slat_generator"], s.condition_embedders["slat_condition_embedder"]])
    P.decode_slat = staged(
        P.decode_slat, lambda s: [s.models["slat_decoder_mesh"], s.models["slat_decoder_gs"], s.models["slat_decoder_gs_4"]])


def load_pipeline(config_path=None, compile=False, lowvram=True):
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    # flash_attn wheels need glibc>=2.32 (FASRC Rocky8 has 2.28) -> force sdpa / xformers.
    # The pipeline forces flash_attn when the GPU name contains A100/H100, so hide the name at import.
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
    _gdn = torch.cuda.get_device_name
    torch.cuda.get_device_name = lambda *a, **k: "GPU"
    try:
        import sam3d_objects  # noqa: F401  (registers stuff)
        import sam3d_objects.pipeline.inference_pipeline_pointmap  # noqa: F401
    finally:
        torch.cuda.get_device_name = _gdn
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "xformers"

    if lowvram:
        _enable_lowvram()
    config_path = config_path or os.path.join(REPO, "checkpoints", "hf", "pipeline.yaml")
    cfg = OmegaConf.load(config_path)
    cfg.rendering_engine = "pytorch3d"
    cfg.compile_model = compile
    cfg.workspace_dir = os.path.dirname(config_path)
    cfg.decode_formats = ["mesh", "gaussian"]  # skip unused gaussian_4 decode
    _PIPE = instantiate(cfg)
    return _PIPE


def depth_to_p3d_pointmap(depth, K):
    """metric depth HxW + K -> HxWx3 pointmap in PyTorch3D camera convention."""
    fx, fy, cx, cy = K
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = depth.astype(np.float32)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pm = np.stack([-x, -y, z], -1)
    bad = ~np.isfinite(z) | (z <= 0)
    pm[bad] = np.nan
    return torch.from_numpy(pm)


def quat_wxyz_to_R(q):
    from pytorch3d.transforms import quaternion_to_matrix
    return quaternion_to_matrix(torch.as_tensor(q, dtype=torch.float32)).numpy()


def render_preview(verts, faces, colors, path, image=None, cam_verts=None, K=None):
    """Cheap software preview: z-buffered flat-shaded triangle centroids splat."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = verts - verts.mean(0)
    v = v / (np.abs(v).max() + 1e-9)
    tri = v[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    cent = tri.mean(1)
    fcol = colors[faces].mean(1) if colors is not None else np.full((len(faces), 3), 0.8)
    views = [(30, 30), (120, 30), (210, 30), (300, -20)]
    ncols = len(views) + (1 if image is not None else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    R = 256
    for ax, (az, el) in zip(axes, views):
        a, e = np.deg2rad(az), np.deg2rad(el)
        # camera looking at origin; object frame is z-up
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
        M = Rx @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) @ Rz  # world->view (x right, y up?, z toward cam)
        c = cent @ M.T
        nn = n @ M.T
        shade = np.clip(np.abs(nn[:, 2]) * 0.7 + 0.3, 0, 1)
        px = ((c[:, 0] * 0.3 + 0.5) * R).astype(int)
        py = ((-c[:, 1] * 0.3 + 0.5) * R).astype(int)
        ok = (px >= 0) & (px < R) & (py >= 0) & (py < R)
        img = np.ones((R, R, 3))
        zb = np.full((R, R), -np.inf)
        order = np.argsort(c[:, 2])  # far to near; later overwrite
        for i in order[ok[order]]:
            img[py[i], px[i]] = fcol[i] * shade[i]
        # dilate a little to fill holes
        ax.imshow(img)
        ax.set_title(f"canonical az={az} el={el}")
        ax.axis("off")
    if image is not None:
        ax = axes[-1]
        ax.imshow(image)
        if cam_verts is not None and K is not None:
            fx, fy, cx, cy = K
            z = cam_verts[:, 2]
            ok = z > 1e-6
            u = fx * cam_verts[ok, 0] / z[ok] + cx
            vv = fy * cam_verts[ok, 1] / z[ok] + cy
            ax.scatter(u, vv, s=0.2, c="lime", alpha=0.3)
            ax.set_title("posed mesh projected (lime)")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True, help="binary mask png (>0 = object), same HxW as image")
    ap.add_argument("--out", required=True, help="output mesh .ply (canonical frame)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", default=None, help="optional metric depth .npy (HxW, metres) -> pose in metres")
    ap.add_argument("--K", default=None, help="fx,fy,cx,cy for --depth (pixels of --image)")
    ap.add_argument("--stage1_steps", type=int, default=None)
    ap.add_argument("--stage2_steps", type=int, default=None)
    ap.add_argument("--no_preview", action="store_true")
    ap.add_argument("--no_lowvram", action="store_true", help="keep all models resident on GPU (~16+ GB)")
    args = ap.parse_args()

    import trimesh

    t0 = time.time()
    rgb = np.array(Image.open(args.image).convert("RGB"))
    m = np.array(Image.open(args.mask))
    if m.ndim == 3:
        m = m[..., -1] if m.shape[-1] == 4 else m[..., 0]
    m = m > 0
    assert m.shape == rgb.shape[:2], (m.shape, rgb.shape)
    rgba = np.concatenate([rgb, (m.astype(np.uint8) * 255)[..., None]], -1)

    pointmap = None
    K = None
    if args.depth is not None:
        assert args.K is not None, "--depth needs --K fx,fy,cx,cy"
        K = [float(x) for x in args.K.split(",")]
        depth = np.load(args.depth).astype(np.float32)
        assert depth.shape == rgb.shape[:2]
        pointmap = depth_to_p3d_pointmap(depth, K)

    pipe = load_pipeline(lowvram=not args.no_lowvram)
    t1 = time.time()
    with torch.no_grad():
        out = pipe.run(
            rgba, None, args.seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=args.stage1_steps,
            stage2_inference_steps=args.stage2_steps,
            pointmap=pointmap,
        )
    t2 = time.time()

    mesh = out["mesh"][0]
    V = mesh.vertices.float().cpu().numpy()  # canonical, z-up
    F = mesh.faces.cpu().numpy()
    C = np.clip(mesh.vertex_attrs[:, :3].float().cpu().numpy(), 0, 1)

    q = out["rotation"].reshape(-1, 4)[0].float().cpu().numpy()
    s = out["scale"].reshape(-1, 3)[0].float().cpu().numpy()
    t = out["translation"].reshape(-1, 3)[0].float().cpu().numpy()
    R = quat_wxyz_to_R(q)
    V_p3d = (V * s) @ R + t
    flip = np.array([-1.0, -1.0, 1.0])
    V_cv = V_p3d * flip
    # column-vector 4x4: p_cv = diag(flip) @ R^T @ diag(s) p + diag(flip) t
    T = np.eye(4)
    T[:3, :3] = np.diag(flip) @ R.T @ np.diag(s)
    T[:3, 3] = flip * t
    assert np.allclose((V[:5] @ T[:3, :3].T + T[:3, 3]), V_cv[:5], atol=1e-4)

    base = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    vc = (C * 255).astype(np.uint8)
    trimesh.Trimesh(V, F, vertex_colors=vc, process=False).export(args.out)
    trimesh.Trimesh(V_cv, F, vertex_colors=vc, process=False).export(base + "_cam.ply")
    try:
        out["gs"].save_ply(base + "_gs.ply")
    except Exception as e:  # noqa
        print("gs save failed:", e)

    # If MoGe was used, recover its intrinsics from its pointmap for the overlay.
    if K is None:
        pm = out["pointmap"].numpy() * flip  # HxWx3 opencv
        H, W = pm.shape[:2]
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        ok = np.isfinite(pm).all(-1) & (pm[..., 2] > 1e-6)
        xn, yn = pm[ok, 0] / pm[ok, 2], pm[ok, 1] / pm[ok, 2]
        A = np.stack([xn, np.ones_like(xn)], 1)
        fx, cx = np.linalg.lstsq(A, u[ok], rcond=None)[0]
        A = np.stack([yn, np.ones_like(yn)], 1)
        fy, cy = np.linalg.lstsq(A, v[ok], rcond=None)[0]
        sx = rgb.shape[1] / W
        K_overlay = [fx * sx, fy * sx, cx * sx, cy * sx]
    else:
        K_overlay = K

    # mask IoU of the posed mesh projection (sanity check of the pose)
    fx, fy, cx, cy = K_overlay
    z = V_cv[:, 2]
    okz = z > 1e-6
    uu = np.round(fx * V_cv[okz, 0] / z[okz] + cx).astype(int)
    vv = np.round(fy * V_cv[okz, 1] / z[okz] + cy).astype(int)
    H0, W0 = m.shape
    inb = (uu >= 0) & (uu < W0) & (vv >= 0) & (vv < H0)
    proj = np.zeros_like(m)
    proj[vv[inb], uu[inb]] = True
    import scipy.ndimage as ndi
    proj = ndi.binary_closing(ndi.binary_dilation(proj, iterations=1), iterations=2)
    iou = float((proj & m).sum() / max((proj | m).sum(), 1))

    info = {
        "image": os.path.abspath(args.image),
        "mask": os.path.abspath(args.mask),
        "seed": args.seed,
        "depth_source": "user_depth" if args.depth else "MoGe(Ruicheng/moge-vitl)",
        "n_vertices": int(len(V)),
        "n_faces": int(len(F)),
        "bbox_canonical_min": V.min(0).tolist(),
        "bbox_canonical_max": V.max(0).tolist(),
        "bbox_cam_min": V_cv.min(0).tolist(),
        "bbox_cam_max": V_cv.max(0).tolist(),
        "rotation_quat_wxyz_p3d": q.tolist(),
        "scale": s.tolist(),
        "translation_p3d": t.tolist(),
        "R_rowvec_p3d": R.tolist(),
        "T_obj2cam_opencv": T.tolist(),
        "K_used_for_overlay": [float(x) for x in K_overlay],
        "proj_mask_iou": iou,
        "conventions": "canonical mesh is z-up object frame (pre-glb). p_p3d=(s*p)@R+t (PyTorch3D cam: x left,y up,z fwd); p_cv = T_obj2cam_opencv @ [p,1] (x right,y down,z fwd).",
        "time_load_s": t1 - t0,
        "time_infer_s": t2 - t1,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    with open(base + ".json", "w") as f:
        json.dump(info, f, indent=1)
    if not args.no_preview:
        render_preview(V, F, C, base + "_preview.png", image=rgb, cam_verts=V_cv, K=K_overlay)
    print(json.dumps({k: info[k] for k in ["n_vertices", "n_faces", "bbox_canonical_min", "bbox_canonical_max",
                                         "bbox_cam_min", "bbox_cam_max", "scale", "proj_mask_iou",
                                         "time_load_s", "time_infer_s", "peak_gpu_mem_gb"]}, indent=1))


if __name__ == "__main__":
    main()
