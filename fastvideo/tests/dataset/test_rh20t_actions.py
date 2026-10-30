# SPDX-License-Identifier: Apache-2.0
"""CPU tests for RH20T action loading and per-view rendering."""
import numpy as np
import pytest
import torch

from fastvideo.dataset.rh20t_actions import (ACTION_MAP_CHANNELS, nearest_index, quat_wxyz_to_matrix,
                                             render_action_maps, view_actions)


def test_quat_wxyz_to_matrix_identity_and_quarter_turn():
    np.testing.assert_allclose(quat_wxyz_to_matrix(np.array([1.0, 0, 0, 0])), np.eye(3), atol=1e-12)
    c = np.sqrt(0.5)  # 90 degrees about z: x -> y
    R = quat_wxyz_to_matrix(np.array([c, 0, 0, c]))
    np.testing.assert_allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)


def test_nearest_index_picks_closest_and_rejects_gaps():
    stamps = np.array([30, 0, 10, 20])  # unsorted on purpose
    np.testing.assert_array_equal(nearest_index(stamps, np.array([1, 14, 29]), 5), [1, 2, 0])
    with pytest.raises(ValueError):
        nearest_index(stamps, np.array([50]), 5)


def _K(h, w, f=10.0):
    return torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])


def test_render_action_maps_blob_location_and_channels():
    # TCP 1 m in front of the camera on the optical axis, tool axis pointing away (+z)
    a = torch.tensor([[0.0, 0.0, 1.0, 1, 0, 0, 0, 0.5]])
    maps = render_action_maps(a, _K(8, 12), 8, 12)
    assert maps.shape == (1, ACTION_MAP_CHANNELS, 8, 12)
    peak = maps[0, 0].argmax()
    assert divmod(int(peak), 12) in {(3, 5), (3, 6), (4, 5), (4, 6)}
    # wrist is on the same ray (0.1 m closer), so it also projects to the centre
    assert divmod(int(maps[0, 1].argmax()), 12) == divmod(int(peak), 12)
    torch.testing.assert_close(maps[0, 2], maps[0, 0] * 0.5)
    torch.testing.assert_close(maps[0, 3], maps[0, 0] * 0.5)  # depth 1 m / 2 m


def test_render_action_maps_behind_camera_is_empty():
    a = torch.tensor([[0.0, 0.0, -1.0, 1, 0, 0, 0, 1.0]])
    assert render_action_maps(a, _K(8, 12), 8, 12)[:, 0].abs().max() == 0


def test_view_actions_falls_back_to_any_gripper_clock():
    tcp = {"a": (np.array([0, 100]), np.array([[0, 0, 1, 1, 0, 0, 0], [0, 0, 2, 1, 0, 0, 0]], dtype=float))}
    gripper = {"other": (np.array([0, 100]), np.array([20.0, 80.0]))}
    out = view_actions(tcp, gripper, "a", np.array([95]))
    np.testing.assert_allclose(out[0], [0, 0, 2, 1, 0, 0, 0, 0.8])
    with pytest.raises(ValueError):
        view_actions(tcp, gripper, "missing", np.array([0]))


def test_load_episode_actions_drops_none_poses_and_empty_gripper_logs(tmp_path):
    from fastvideo.dataset.rh20t_actions import load_episode_actions
    d = tmp_path / "transformed"
    d.mkdir()
    pose = np.array([0, 0, 1, 1, 0, 0, 0.0])
    tcp = {
        "a": [{"timestamp": 0, "tcp": pose}, {"timestamp": 100, "tcp": None}, {"timestamp": 200, "tcp": pose}],
        "b": [{"timestamp": 0, "tcp": None}],  # no valid pose at all
    }
    gripper = {"a": {}, "b": {0: {"gripper_info": [40.0, 0, 1]}}}
    np.save(d / "tcp.npy", tcp, allow_pickle=True)
    np.save(d / "gripper.npy", gripper, allow_pickle=True)
    tcp_out, grip_out = load_episode_actions(tmp_path)
    assert list(tcp_out) == ["a"] and tcp_out["a"][0].tolist() == [0, 200]
    assert list(grip_out) == ["b"]
    # camera a has no gripper log of its own: it borrows camera b's
    np.testing.assert_allclose(view_actions(tcp_out, grip_out, "a", np.array([0]))[0, 7], 0.4)
