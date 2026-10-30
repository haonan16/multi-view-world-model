# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the RH20T multi-view dataset on a synthetic episode tree."""

import json
from pathlib import Path

import av
import numpy as np
import pytest
import torch

from fastvideo.dataset.rh20t_depth import read_depth_frames
from fastvideo.dataset.rh20t_multiview import (IN_HAND, RH20TMultiViewConfig, RH20TMultiViewDataset, camera_to_raymap,
                                               center_crop_box, crop_resize_intrinsics, grid_for, marker_angle_deg,
                                               nearest_frames, read_frames, tile_views, untile_views)

CFG_ID = 3
CALIB_TS = 1631153393825
VIDEO_H, VIDEO_W = 36, 64  # 16:9 like the 320x180 release
N_VIDEO_FRAMES = 40
FRAME_MS = 100  # ~10 Hz capture
SPAN_MS = 3900


def _write_mp4(path: Path, n_frames: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = VIDEO_W, VIDEO_H, "yuv420p"
        stream.options = {"crf": "0"}
        for i in range(n_frames):
            # constant colour per frame encodes the frame index in the red channel
            img = np.full((VIDEO_H, VIDEO_W, 3), rng.integers(0, 50), dtype=np.uint8)
            img[..., 0] = 5 * i
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


DEPTH_M = 1.2  # every depth pixel; in front of all fixture cameras


def _write_depth_mp4(path: Path, n_frames: int) -> None:
    """RH20T byte-split depth video: [2H, W] grey, low byte on top, high byte below (mm)."""
    raw = int(DEPTH_M * 1000)
    gray = np.concatenate([np.full((VIDEO_H, VIDEO_W), raw % 256), np.full((VIDEO_H, VIDEO_W), raw // 256)])
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.uint8)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264rgb", rate=10)
        stream.width, stream.height, stream.pix_fmt = VIDEO_W, 2 * VIDEO_H, "rgb24"
        stream.options = {"crf": "0"}
        for _ in range(n_frames):
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _look_at_w2c(cam_pos: np.ndarray) -> np.ndarray:
    """World->camera (OpenCV) for a camera at cam_pos looking at the world origin."""
    z = -cam_pos / np.linalg.norm(cam_pos)
    x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # rows = camera axes in world
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ cam_pos
    return w2c


@pytest.fixture
def rh20t_tree(tmp_path: Path) -> dict:
    data_root, calib_root = tmp_path / "data", tmp_path / "calib"
    in_hand = next(iter(IN_HAND[CFG_ID]))
    # serial -> (camera position, timestamp offset ms, frame spacing ms)
    cams = {
        "000000000001": (np.array([1.0, 0.0, 0.5]), 0, FRAME_MS),
        "000000000002": (np.array([0.0, 1.0, 0.5]), 20, FRAME_MS),
        "000000000003": (np.array([-1.0, 0.2, 0.5]), 0, FRAME_MS),
        "000000000004": (np.array([0.3, -1.0, 2.0]), 0, FRAME_MS),
        # 400 ms spacing: of two grid points 200 ms apart one is always >= 140 ms from a frame
        "000000000005": (np.array([0.5, 0.5, 0.5]), 0, 400),
        in_hand: (np.array([0.2, 0.2, 0.4]), 0, FRAME_MS),
    }
    K = np.array([[900.0, 0, 640, 0], [0, 900.0, 360, 0], [0, 0, 1, 0]])
    calib_dir = calib_root / f"RH20T_cfg{CFG_ID}" / "calib" / str(CALIB_TS)
    calib_dir.mkdir(parents=True)
    np.save(calib_dir / "intrinsics.npy", {s: K for s in cams} | {"999999999999": None}, allow_pickle=True)
    np.save(calib_dir / "extrinsics.npy", {s: _look_at_w2c(p)[None] for s, (p, _, _) in cams.items()}
            | {"999999999999": None},
            allow_pickle=True)

    depth_root = tmp_path / "depth"
    episodes = []
    for e in range(3):
        ep = data_root / f"RH20T_cfg{CFG_ID}" / f"task_000{e + 1}_user_0016_scene_0001_cfg_000{CFG_ID}"
        episodes.append(ep.name)
        for k, (serial, (_, offset, step)) in enumerate(cams.items()):
            d = ep / f"cam_{serial}" / "color"
            d.mkdir(parents=True)
            stamps = (1_000_000 + offset + np.arange(0, SPAN_MS + 1, step)).astype(np.int64)
            np.save(d / "timestamps.npy", stamps)
            _write_mp4(d / "color.mp4", N_VIDEO_FRAMES, seed=k)
            if serial != "000000000004":  # one static camera without depth
                dd = depth_root / f"RH20T_cfg{CFG_ID}" / ep.name / f"cam_{serial}"
                dd.mkdir(parents=True)
                _write_depth_mp4(dd / "depth.mp4", N_VIDEO_FRAMES)
        (ep / "metadata.json").write_text(json.dumps({"calib": CALIB_TS, "rating": 2 + e}))
    # a human demo (must be ignored) and an episode without metadata (skipped)
    (data_root / f"RH20T_cfg{CFG_ID}" / f"{episodes[0]}_human").mkdir()
    (data_root / f"RH20T_cfg{CFG_ID}" / "task_0009_user_0016_scene_0001_cfg_0003").mkdir()
    return {
        "data_root": str(data_root),
        "calib_root": str(calib_root),
        "depth_root": str(depth_root),
        "episodes": episodes,
        "in_hand": in_hand
    }


def _config(tree: dict, **overrides) -> RH20TMultiViewConfig:
    kw = dict(data_root=tree["data_root"],
              calib_root=tree["calib_root"],
              cfgs=(CFG_ID, ),
              test_percentage=0.0,
              n_views=3,
              n_frames=5,
              fps=5.0,
              max_sync_err_ms=60.0,
              height=32,
              width=64,
              raymap_height=4,
              raymap_width=8)
    kw.update(overrides)
    return RH20TMultiViewConfig(**kw)


# ---- helpers ----


@pytest.mark.parametrize("n_views,expected", [(1, (1, 1)), (2, (1, 2)), (4, (2, 2)), (6, (2, 3))])
def test_grid_for_is_near_square(n_views, expected):
    assert grid_for(n_views) == expected


def test_untile_inverts_tile_layout():
    torch.manual_seed(0)
    small = torch.rand(2, 4, 3, 3, 8, 16)  # B, V, T, C, h, w
    full = small.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)  # area-downsample recovers `small`
    canvas = tile_views(full)
    assert canvas.shape == (2, 3, 3, 16, 32)
    torch.testing.assert_close(untile_views(canvas, 4), small)
    # view 1 sits in the top-right tile of a 2x2 grid
    torch.testing.assert_close(canvas[:, :, :, :8, 16:], small[:, 1])


def test_tile_views_rejects_indivisible_canvas():
    with pytest.raises(ValueError):
        tile_views(torch.zeros(4, 1, 3, 15, 32))


def test_raymap_identity_camera():
    K = torch.tensor([[[2.0, 0, 2.0], [0, 2.0, 2.0], [0, 0, 1]]])
    c2w = torch.eye(4)[None].clone()
    c2w[0, :3, 3] = torch.tensor([0.0, 0.0, 0.0])
    rays = camera_to_raymap(K, c2w, 4, 4)
    assert rays.shape == (1, 6, 4, 4)
    torch.testing.assert_close(rays[:, :3].norm(dim=1), torch.ones(1, 4, 4))
    assert torch.all(rays[0, 2] > 0)  # all rays point forward (+z)
    torch.testing.assert_close(rays[:, 3:], torch.zeros(1, 3, 4, 4))  # origin at 0 -> zero moment


def test_raymap_moment_is_origin_cross_direction():
    K = torch.tensor([[[3.0, 0, 1.5], [0, 3.0, 1.5], [0, 0, 1]]])
    c2w = torch.eye(4)[None].clone()
    c2w[0, :3, 3] = torch.tensor([1.0, -2.0, 0.5])
    rays = camera_to_raymap(K, c2w, 3, 3)
    expected = torch.cross(c2w[0, :3, 3].view(1, 3, 1, 1).expand(1, 3, 3, 3), rays[:, :3], dim=1)
    torch.testing.assert_close(rays[:, 3:], expected)


def test_center_crop_and_intrinsics():
    box = center_crop_box(180, 320, 160, 320)
    assert box == (10, 170, 0, 320)
    K = np.array([[300.0, 0, 160], [0, 300.0, 90], [0, 0, 1]])
    K2 = crop_resize_intrinsics(K, box, 80, 160)
    np.testing.assert_allclose(K2, [[150.0, 0, 80], [0, 150.0, 40], [0, 0, 1]])


def test_marker_angle_prefers_cameras_facing_origin():
    facing = _look_at_w2c(np.array([1.0, 0.0, 0.5]))
    away = facing.copy()
    away[:3, 3] += np.array([0.5, 0.0, 0.0])  # marker now off-axis
    assert marker_angle_deg(facing) == pytest.approx(0.0, abs=1e-6)
    assert marker_angle_deg(away) > 5.0


def test_nearest_frames():
    stamps = np.array([0, 100, 210, 290])
    j, err = nearest_frames(stamps, np.array([0.0, 140.0, 300.0]))
    np.testing.assert_array_equal(j, [0, 1, 3])
    np.testing.assert_allclose(err, [0, 40, 10])


def test_config_validation():
    with pytest.raises(ValueError):
        RH20TMultiViewConfig(data_root="x", calib_root="y", n_frames=12)
    with pytest.raises(ValueError):
        RH20TMultiViewConfig(data_root="x", calib_root="y", n_views=0)


# ---- dataset ----


def test_records_skip_human_missing_meta_inhand_and_uncalibrated(rh20t_tree):
    ds = RH20TMultiViewDataset(_config(rh20t_tree), split="train")
    assert sorted(r["episode"] for r in ds.records) == sorted(rh20t_tree["episodes"])
    for rec in ds.records:
        assert rh20t_tree["in_hand"] not in rec["cams"]
        assert len(rec["cams"]) == 5


def test_split_is_disjoint_and_stable(rh20t_tree):
    train = RH20TMultiViewDataset(_config(rh20t_tree, test_percentage=0.5), split="train")
    val = RH20TMultiViewDataset(_config(rh20t_tree, test_percentage=0.5), split="val")
    names_t = {r["episode"] for r in train.records}
    names_v = {r["episode"] for r in val.records}
    assert not names_t & names_v
    assert names_t | names_v == set(rh20t_tree["episodes"])


def test_records_cache_roundtrip(rh20t_tree, tmp_path):
    cfg = _config(rh20t_tree, records_cache_dir=str(tmp_path / "cache"))
    first = RH20TMultiViewDataset(cfg).records
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1
    assert RH20TMultiViewDataset(cfg).records == first


def test_plan_clip_drops_unsynchronised_camera(rh20t_tree):
    ds = RH20TMultiViewDataset(_config(rh20t_tree, n_views=4), split="train")
    plan = ds.plan_clip(ds.records[0], np.random.default_rng(0))
    assert plan is not None
    serials, ok = plan
    assert len(serials) == 4 and len(set(serials)) == 4
    assert "000000000005" not in serials  # 400 ms spacing never syncs within 60 ms
    for s in serials:
        j, _, err = ok[s]
        assert len(j) == 5 and err.max() <= 60.0


def test_plan_clip_returns_none_when_too_few_views(rh20t_tree):
    ds = RH20TMultiViewDataset(_config(rh20t_tree, n_views=5), split="train")
    assert ds.plan_clip(ds.records[0], np.random.default_rng(0)) is None


def test_read_frames_returns_requested_indices(rh20t_tree):
    ep = Path(rh20t_tree["data_root"]) / f"RH20T_cfg{CFG_ID}" / rh20t_tree["episodes"][0]
    x = read_frames(ep / "cam_000000000001" / "color" / "color.mp4", np.array([3, 0, 7]))
    assert x.shape == (3, 3, VIDEO_H, VIDEO_W) and x.dtype == torch.uint8
    red = x[:, 0].float().mean(dim=(1, 2))
    torch.testing.assert_close(red, torch.tensor([15.0, 0.0, 35.0]), atol=3.0, rtol=0)
    with pytest.raises(ValueError):
        read_frames(ep / "cam_000000000001" / "color" / "color.mp4", np.array([N_VIDEO_FRAMES + 5]))


def test_getitem_shapes_and_camera_frame(rh20t_tree):
    ds = RH20TMultiViewDataset(_config(rh20t_tree), split="train")
    sample = ds[0]
    V, T = 3, 5
    assert sample["videos"].shape == (V, T, 3, 32, 64)
    assert sample["videos"].min() >= -1 and sample["videos"].max() <= 1
    assert sample["raymaps"].shape == (V, 6, 4, 8)
    assert sample["intrinsics"].shape == (V, 3, 3)
    torch.testing.assert_close(sample["c2w"][0], torch.eye(4))
    assert sample["timestamps"].shape == (V, T)
    assert sample["sync_err_ms"].max() <= 60.0
    assert len(sample["serials"].split(";")) == V
    # frames advance in time: red channel encodes the decoded frame index
    red = sample["videos"][0, :, 0].mean(dim=(1, 2))
    assert torch.all(red[1:] > red[:-1])


def test_val_split_is_deterministic(rh20t_tree):
    ds = RH20TMultiViewDataset(_config(rh20t_tree, test_percentage=0.9999), split="val")
    a, b = ds[0], ds[0]
    assert a["serials"] == b["serials"]
    torch.testing.assert_close(a["timestamps"], b["timestamps"])


def test_val_resample_does_not_repeat_failed_draw(rh20t_tree, monkeypatch):
    # a failing val draw used to reseed with the same rng, so it could revisit the same
    # (episode, plan) forever and exhaust MAX_RESAMPLE_ATTEMPTS
    ds = RH20TMultiViewDataset(_config(rh20t_tree, test_percentage=0.9999), split="val")
    real, calls = ds.plan_clip, []

    def flaky(rec, rng):
        calls.append(rng.random())
        return None if len(calls) == 1 else real(rec, rng)

    monkeypatch.setattr(ds, "plan_clip", flaky)
    assert "videos" in ds[0]
    assert len(calls) == 2 and calls[0] != calls[1]


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RH20TMultiViewDataset(RH20TMultiViewConfig(data_root=str(tmp_path / "nope"), calib_root=str(tmp_path)))


# ---- unseen views / depth / point cloud ----


def test_config_rejects_point_cloud_without_depth_or_context():
    with pytest.raises(ValueError):
        RH20TMultiViewConfig(data_root="x", calib_root="y", point_cloud=True, n_context_views=2)
    with pytest.raises(ValueError):
        RH20TMultiViewConfig(data_root="x", calib_root="y", point_cloud=True, depth_root="z")
    with pytest.raises(ValueError):
        RH20TMultiViewConfig(data_root="x", calib_root="y", n_views=4, n_context_views=5)


def test_read_depth_frames_decodes_metres(rh20t_tree):
    ep = Path(rh20t_tree["depth_root"]) / f"RH20T_cfg{CFG_ID}" / rh20t_tree["episodes"][0]
    d = read_depth_frames(ep / "cam_000000000001" / "depth.mp4", np.array([0, 5]))
    assert d.shape == (2, VIDEO_H, VIDEO_W)
    np.testing.assert_allclose(d, DEPTH_M, atol=0.003)


def test_default_sample_marks_every_view_as_context(rh20t_tree):
    sample = RH20TMultiViewDataset(_config(rh20t_tree), split="train")[0]
    assert sample["context"].dtype == torch.bool and sample["context"].all()
    assert "pc_render" not in sample


def test_depth_root_drops_cameras_without_depth(rh20t_tree):
    ds = RH20TMultiViewDataset(_config(rh20t_tree, depth_root=rh20t_tree["depth_root"]), split="train")
    for rec in ds.records:
        assert "000000000004" not in rec["cams"] and len(rec["cams"]) == 4


def test_point_cloud_sample_renders_context_views(rh20t_tree):
    cfg = _config(rh20t_tree, depth_root=rh20t_tree["depth_root"], n_context_views=2, point_cloud=True)
    sample = RH20TMultiViewDataset(cfg, split="train")[0]
    V, H, W = 3, 32, 64
    assert sample["context"].tolist() == [True, True, False]
    assert sample["pc_render"].shape == (V, 3, H, W) and sample["pc_mask"].shape == (V, 1, H, W)
    mask = sample["pc_mask"][:, 0].bool()
    # a context view sees its own depth: (nearly) full coverage. The red channel encodes the
    # frame index (0 -> -1 for every camera), so it checks that frame 0 was rendered even
    # where the other context camera's points occlude.
    for v in range(2):
        assert mask[v].float().mean() > 0.9
        red = sample["pc_render"][v, 0][mask[v]]
        assert (red - sample["videos"][v, 0, 0][mask[v]]).abs().median() < 0.05
    # empty pixels are exactly zero
    assert (sample["pc_render"][~mask[:, None].expand(-1, 3, -1, -1)] == 0).all()
