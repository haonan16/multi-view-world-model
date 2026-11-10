"""pytorch3d rendering of the reconstructed parts at the supplied OpenCV cameras (overlays, masks, depth)."""
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import (BlendParams, HardPhongShader, MeshRasterizer, MeshRenderer, PointLights,
                                RasterizationSettings, SoftSilhouetteShader, TexturesVertex)
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.utils import cameras_from_opencv_projection

PALETTE = {"robot": (0.78, 0.86, 0.98), "free": (1.0, 0.62, 0.25), "articulated_child": (0.35, 0.85, 0.55),
           "articulated_parent": (0.75, 0.75, 0.80), "static": (0.85, 0.80, 0.55), "attachment": (0.78, 0.86, 0.98)}


def cam(K, w2c, H, W, dev="cuda"):
    K = torch.as_tensor(K, dtype=torch.float32, device=dev).reshape(-1, 3, 3)
    w2c = torch.as_tensor(w2c, dtype=torch.float32, device=dev).reshape(-1, 4, 4)
    size = torch.tensor([[H, W]], device=dev).expand(len(K), 2)
    return cameras_from_opencv_projection(w2c[:, :3, :3], w2c[:, :3, 3], K, size)


def to_meshes(parts, dev="cuda"):
    """parts: list of (trimesh, rgb) -> one pytorch3d scene mesh"""
    ms = []
    for tm, col in parts:
        v = torch.as_tensor(np.asarray(tm.vertices), dtype=torch.float32, device=dev)
        f = torch.as_tensor(np.asarray(tm.faces), dtype=torch.int64, device=dev)
        c = torch.tensor(col, dtype=torch.float32, device=dev)[None].expand(len(v), 3)
        ms.append(Meshes([v], [f], TexturesVertex([c])))
    return join_meshes_as_scene(ms)


def render(parts, K, w2c, H, W, dev=None):
    """-> rgb [H,W,3] float, alpha [H,W] bool, depth [H,W] (0 = empty), part id map [H,W] (-1 empty)"""
    if dev is None:
        from .p3d_compat import NATIVE
        dev = "cuda" if NATIVE else "cpu"
    mesh = to_meshes(parts, dev)
    c = cam(K, w2c, H, W, dev)
    rs = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0)
    rast = MeshRasterizer(cameras=c, raster_settings=rs)
    frag = rast(mesh)
    C = torch.linalg.inv(torch.as_tensor(w2c, dtype=torch.float32, device=dev))[:3, 3]
    lights = PointLights(device=dev, location=C[None], ambient_color=((0.45,) * 3,), diffuse_color=((0.55,) * 3,),
                         specular_color=((0.15,) * 3,))
    shader = HardPhongShader(device=dev, cameras=c, lights=lights, blend_params=BlendParams(background_color=(0, 0, 0)))
    img = shader(frag, mesh)[0, ..., :3]
    pix = frag.pix_to_face[0, ..., 0]
    alpha = pix >= 0
    depth = torch.where(alpha, frag.zbuf[0, ..., 0], torch.zeros_like(frag.zbuf[0, ..., 0]))
    # part id from face offsets
    nf = torch.tensor([len(p[0].faces) for p in parts], device=dev).cumsum(0)
    pid = torch.where(alpha, torch.searchsorted(nf, pix, right=True), torch.full_like(pix, -1))
    return img.cpu().numpy(), alpha.cpu().numpy(), depth.cpu().numpy(), pid.cpu().numpy()


def overlay(rgb, img, alpha, a=0.85):
    out = rgb.astype(np.float32) / 255
    out[alpha] = (1 - a) * out[alpha] + a * img[alpha]
    return (out * 255).clip(0, 255).astype(np.uint8)


def box_mesh(half, T):
    m = trimesh.creation.box(extents=2 * np.asarray(half))
    m.apply_transform(T)
    return m


def scene_parts(robot, res, t, decimate=None):
    """reconstruction result (fit.pkl dict) -> list of (trimesh world, colour, name) at frame t"""
    parts = []
    for l, m in robot.meshes_at(res["robot"]["q"][t], res["robot"]["base"]):
        parts.append((m, PALETTE["robot"], f"robot/{l}"))
    for ob in res["objects"]:
        if ob.get("mesh") is not None:
            m = trimesh.Trimesh(ob["mesh"][0], ob["mesh"][1], process=False)
            m.apply_transform(ob["poses"][t])
            parts.append((m, PALETTE.get(ob["type"], PALETTE["free"]), ob["name"]))
        elif ob["type"] == "articulated":
            from .objects import part_boxes
            for Tb, hb in part_boxes(ob["parent"]["pose"], ob["parent"]["half"], ob["parent"].get("slabs")):
                parts.append((box_mesh(hb, Tb), PALETTE["articulated_parent"], ob["name"] + "/parent"))
            for Tb, hb in part_boxes(ob["child"]["poses"][t], ob["child"]["half"], ob["child"].get("slabs")):
                parts.append((box_mesh(hb, Tb), PALETTE["articulated_child"], ob["name"] + "/child"))
        elif ob["type"] == "static":
            from .objects import static_boxes
            for i, (Tb, hb) in enumerate(static_boxes(ob)):
                parts.append((box_mesh(hb, Tb), PALETTE["static"], f"{ob['name']}/{i}"))
        else:
            parts.append((box_mesh(ob["half"], ob["poses"][t]), PALETTE[ob["type"]], ob["name"]))
    return parts
