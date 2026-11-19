"""Image-derived shape proportions are separate from predicted metric scale."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks import source_aspect_fit as aspect_fit
from test_source_joint_fit import synthetic_scene_and_observations


def test_canonical_aspect_ignores_global_mesh_units_and_translation():
    vertices = np.array(list(product([-.1, .1], [-.2, .2], [-.3, .3])))
    ratio, dimensions = aspect_fit.canonical_aspect(vertices)
    changed, _ = aspect_fit.canonical_aspect(vertices * 41.7 + np.array([100, -30, 50]))
    np.testing.assert_allclose(ratio, changed)
    assert np.prod(ratio) == pytest.approx(1.)
    np.testing.assert_allclose(dimensions, [.2, .4, .6])


@pytest.mark.parametrize('bad', [np.zeros((5, 3)), np.full((5, 3), np.nan), np.zeros((5, 2))])
def test_invalid_canonical_geometry_is_rejected(bad):
    with pytest.raises(ValueError): aspect_fit.canonical_aspect(bad)


def test_one_shared_scalar_preserves_prior_proportions_while_optimizing_support():
    scene, observations = synthetic_scene_and_observations()
    before = deepcopy(scene)
    # Deliberately nonsymmetric proportions demonstrate this is not a cube prior.
    ratio = np.array([1.2, .8, 1.]); ratio /= np.prod(ratio)**(1/3)
    fitted, report = aspect_fit.fit_shared_scene(scene, observations, aspect_fit.default_config(), ratio)
    assert scene == before
    dimensions = np.array(report['after']['dimensions_m'])
    np.testing.assert_allclose(dimensions / dimensions[2], ratio / ratio[2], rtol=1e-10)
    assert min(report['after']['bottom_height_m']) >= -1e-12
    assert report['SAM3D_metric_pose_or_scale_used'] is False
    assert report['simulator_accessed'] is False
    assert report['endpoint_specific_operation'] is False
    assert fitted['joint_fit']['one_metric_shape_scale'] is True
    assert report['final_cost'] == pytest.approx(sum(report['after']['normalized_soft_l1_cost'].values()))


def test_missing_evidence_remains_unobserved_with_a_shape_prior():
    scene, observations = synthetic_scene_and_observations()
    observations['evidence'][1]['observed'] = False
    fitted, _ = aspect_fit.fit_shared_scene(scene, observations, aspect_fit.default_config(), np.ones(3))
    assert fitted['objects'][0]['observed'][1] is False
