import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/sam3_rgb_masks.py'
    spec = importlib.util.spec_from_file_location('sam3_rgb_masks', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_mask_packing_preserves_native_grid_candidates_and_empty_results():
    m = module()
    masks = np.zeros((2, 3, 5), bool)
    masks[0, 0, 0] = masks[1, 2, 4] = True
    packed, scores, boxes = m.pack_candidates(masks, [.7, .8], [[0, 0, 1, 1], [4, 2, 5, 3]], 3, 5)
    np.testing.assert_array_equal(np.unpackbits(packed, axis=1)[:, :15].reshape(2, 3, 5), masks)
    assert packed.shape == (2, 2)
    empty, _, _ = m.pack_candidates(np.zeros((0, 3, 5), bool), [], np.empty((0, 4)), 3, 5)
    assert empty.shape == (0, 2)
    internal_empty, _, _ = m.pack_candidates(np.zeros((0, 288, 288), bool), [], np.empty((0, 4)), 3, 5)
    assert internal_empty.shape == (0, 2)
    with pytest.raises(ValueError, match='native image grid'):
        m.pack_candidates(masks[:, :2], scores, boxes, 3, 5)


def test_masks_use_only_a_clean_observed_rgb_bundle(tmp_path):
    m = module()
    import rgb_frontend
    source = tmp_path / 'source.npz'
    np.savez(source, rgb=np.zeros((2, 4, 5, 3), np.uint8), timestamps=[0., .1], K=np.eye(3))
    rgb_frontend.prepare(source, tmp_path / 'rgb')
    job = m.prepare_request({'bundle': str(tmp_path / 'rgb'), 'out': str(tmp_path / 'masks'), 'prompts': ['hand', 'tabletop']})
    assert job['rgb'].shape == (2, 4, 5, 3)
    with pytest.raises(ValueError, match='only bundle, out, prompts'):
        m.prepare_request({'bundle': str(tmp_path / 'rgb'), 'out': str(tmp_path / 'masks'), 'prompts': ['hand'], 'known_K': np.eye(3)})
