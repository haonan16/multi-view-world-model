"""Small CPU triangle renderer for synthetic validation and local previews."""
import numpy as np


def look_at(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.stack((x, y, z), axis=1)
    pose[:3, 3] = eye
    return pose


def render(vertices, faces, K, c2w, size=160, color=(105, 174, 220)):
    camera = (vertices - c2w[:3, 3]) @ c2w[:3, :3]
    pixels = camera @ K.T
    uv = pixels[:, :2] / np.maximum(camera[:, 2:], 1e-8)
    H, W = (size, size) if np.isscalar(size) else size
    depth = np.full((H, W), np.inf, np.float32)
    rgb = np.full((H, W, 3), 245, np.uint8)
    for face in faces:
        z = camera[face, 2]
        if np.any(z <= 0):
            continue
        tri = uv[face]
        lo = np.maximum(np.floor(tri.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(tri.max(0)).astype(int), [W - 1, H - 1])
        if np.any(lo > hi):
            continue
        x, y = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
        a, b, c = tri
        denom = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(denom) < 1e-8:
            continue
        u = ((b[1]-c[1])*(x-c[0]) + (c[0]-b[0])*(y-c[1])) / denom
        v = ((c[1]-a[1])*(x-c[0]) + (a[0]-c[0])*(y-c[1])) / denom
        w = 1-u-v
        cover = (u >= 0) & (v >= 0) & (w >= 0)
        zz = 1 / np.maximum(u/z[0] + v/z[1] + w/z[2], 1e-8)
        old = depth[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        update = cover & (zz < old)
        old[update] = zz[update]
        pts = camera[face]
        normal = np.cross(pts[1]-pts[0], pts[2]-pts[0])
        shade = 0.45 + 0.55 * abs(normal[2]) / max(np.linalg.norm(normal), 1e-10)
        rgb[lo[1]:hi[1]+1, lo[0]:hi[0]+1][update] = np.asarray(color) * shade
    depth[~np.isfinite(depth)] = 0
    return rgb, depth
