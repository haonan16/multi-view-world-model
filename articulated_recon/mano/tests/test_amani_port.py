import numpy as np
import torch

from mano_recon.amani_objects.geom import DepthViews
from mano_recon.render import render


def test_occlusion_reads_requested_camera():
    depth = np.stack([np.full((4, 5), 1.), np.full((4, 5), 2.)])[None]
    views = DepthViews(depth, np.tile(np.eye(3), (2, 1, 1)),
                       np.tile(np.eye(4), (2, 1, 1)), device="cpu")
    uv = torch.ones(1, 1, 3)
    value, inside = views.sample(uv, uv, [0], view_indices=[1])
    assert torch.all(value == 2) and inside.all()


def test_rectangular_renderer_preserves_projection():
    vertices = np.array([[0, 0, 1], [.4, 0, 1], [0, .3, 1]])
    K = np.array([[10., 0, 3], [0, 10., 2], [0, 0, 1]])
    rgb, depth = render(vertices, np.array([[0, 1, 2]]), K, np.eye(4), size=(8, 12))
    assert rgb.shape == (8, 12, 3)
    assert depth[2, 3] == 1 and depth[0, 0] == 0
