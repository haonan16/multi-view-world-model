import importlib.util
from pathlib import Path

import numpy as np


def module():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/sam3d_rgb_asset.py'
    spec = importlib.util.spec_from_file_location('sam3d_rgb_asset', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_table_gate_rejects_background_distractor_without_an_oracle_click():
    m = module()
    masks = np.zeros((2, 20, 20), bool)
    masks[0, 2:5, 2:5] = True
    masks[1, 12:15, 12:15] = True
    table = np.zeros((1, 20, 20), bool)
    table[0, 10:19, 10:19] = True
    result = m.select_candidate(masks, [.95, .8], table)
    assert result['selected_index'] == 1
    assert result['table_hull_gate_used'] is True


def test_similar_candidates_remain_ambiguous_instead_of_using_an_arbitrary_row():
    m = module()
    masks = np.zeros((2, 20, 20), bool)
    masks[0, 2:5, 2:5] = masks[1, 12:15, 12:15] = True
    result = m.select_candidate(masks, [.9, .85], np.empty((0, 20, 20), bool))
    assert result['selected_index'] is None
    assert len(result['candidates']) == 2
