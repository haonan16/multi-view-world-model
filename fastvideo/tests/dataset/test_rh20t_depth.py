# SPDX-License-Identifier: Apache-2.0
"""CPU tests for RH20T depth decoding and point-cloud rendering."""
import numpy as np
import torch

from fastvideo.dataset.rh20t_depth import (clean_depth, decode_depth_frame, render_points, unproject)


def _K(h, w, f=None):
    f = f or 0.8 * w
    return torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=torch.float64)


def test_decode_depth_frame_combines_bytes_and_masks_range():
    raw = np.array([[1200, 50], [65000, 2999]], dtype=np.int32)  # mm
    gray = np.concatenate([raw % 256, raw // 256]).astype(np.uint8)
    d = decode_depth_frame(gray)
    np.testing.assert_allclose(d, [[1.2, 0.0], [0.0, 2.999]], rtol=1e-6)


def test_unproject_then_render_same_camera_is_identity():
    H, W = 12, 16
    K = _K(H, W)
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, 3] = torch.tensor([0.1, -0.2, 0.3])
    depth = torch.rand(H, W, dtype=torch.float64) + 0.5
    depth[0, 0] = 0  # invalid pixel stays empty
    colors = torch.rand(H, W, 3, dtype=torch.float64)
    pts, valid = unproject(depth, K, c2w)
    img, zbuf, mask = render_points(pts, colors[valid], K, torch.linalg.inv(c2w), H, W)
    assert torch.equal(mask, valid)
    torch.testing.assert_close(zbuf[valid], depth[valid])
    torch.testing.assert_close(img.permute(1, 2, 0)[valid], colors[valid])


def test_render_points_keeps_nearest_point():
    K = _K(4, 4)
    w2c = torch.eye(4, dtype=torch.float64)
    pts = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=torch.float64)
    cols = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    img, zbuf, mask = render_points(pts, cols, K, w2c, 4, 4)
    assert mask.sum() == 1
    assert zbuf[mask].item() == 1.0 and img[0][mask].item() == 2.0


def test_render_points_drops_points_behind_camera():
    K = _K(4, 4)
    pts = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)
    _, _, mask = render_points(pts, torch.ones(1, 1, dtype=torch.float64), K, torch.eye(4, dtype=torch.float64), 4, 4)
    assert not mask.any()


def test_clean_depth_removes_byte_spike():
    d = torch.full((5, 5), 1.0)
    d[2, 2] = 1.256  # high-byte coding error
    out = clean_depth(d)
    assert out[2, 2] == 0 and (out.sum() == 24)
