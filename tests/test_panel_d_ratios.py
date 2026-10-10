"""Tests of the actual Cartesian-image radial/conditional factorization."""
import copy

import numpy as np
import pytest

from upc_deformation.panel_d_ratios import decompose_panel_d


def _result(name, axis, image):
    # Other stages and the self source are deliberately unrelated to panel d.
    images = np.full((4, 2, len(axis), len(axis)), 43.)
    images[3, 0] = image
    return {"case": {"name": name, "beta2": .3 if name == 'beta2' else 0.,
                     "beta3": .2 if name == 'beta3' else 0., "skin_fm": .2 if name == 'skin' else 0.},
            "axis": axis, "normalized_images": images,
            "n_orientations": 32 if name in ('beta2', 'beta3') else 1,
            "stage_names": ['b_axis', 'decay_fixed', 'decay_b', 'decay_by']}


def _fixture(n=151):
    axis = np.linspace(-.2, .2, n)
    x, y = np.meshgrid(axis, axis, indexing='ij')
    return axis, x, y


def test_constant_rescaling_has_exact_unit_conditional_ratio_and_factor_closure():
    axis, x, y = _fixture()
    image = np.exp(-24*(x*x+y*y))*(1+.7*x)
    inputs = [_result('spherical', axis, image), _result('beta2', axis, 1.37*image)]
    original = copy.deepcopy(inputs)
    out = decompose_panel_d(inputs, n_radii=71, n_angles=128)
    case = out['cases']['beta2']
    np.testing.assert_allclose(case['radial_ratio'], 1.37, rtol=1e-15)
    np.testing.assert_allclose(case['angular_polar'][case['valid_polar']], 1, atol=7e-16)
    np.testing.assert_allclose(case['angular_map'][case['valid_cart']], 1, atol=7e-16)
    np.testing.assert_allclose(out['base_image'], image)
    diagnostics = out['diagnostics']['cases']['beta2']
    assert diagnostics['cart_factor_closure_max_absolute'] < 1e-15
    assert diagnostics['polar_factor_closure_max_absolute'] < 1e-15
    for before, after in zip(original, inputs):
        np.testing.assert_array_equal(before['normalized_images'], after['normalized_images'])
        np.testing.assert_array_equal(before['axis'], after['axis'])


def test_smooth_pure_radial_change_has_only_finite_cartesian_interpolation_residual():
    axis, x, y = _fixture(301)
    base = 1+.6*x
    radial_factor = 1+.5*(x*x+y*y)/.17**2
    out = decompose_panel_d([_result('spherical', axis, base),
                            _result('beta2', axis, base*radial_factor)],
                           n_radii=341, n_angles=360)
    case = out['cases']['beta2']
    # Interpolating a Cartesian product is not exactly multiplication of its
    # separately sampled factors.  The tolerance covers the known finite mesh.
    np.testing.assert_allclose(case['radial_ratio'], 1+.5*out['radii']**2/.17**2, atol=2e-5)
    np.testing.assert_allclose(case['angular_polar'][case['valid_polar']], 1, atol=2e-5)
    np.testing.assert_allclose(case['angular_map'][case['valid_cart']], 1, atol=2e-5)


def test_pure_quadrupole_angle_change_preserves_circle_mean():
    axis, x, y = _fixture(201)
    out = decompose_panel_d([_result('spherical', axis, np.ones_like(x)),
                            _result('beta3', axis, 1+3*(x*x-y*y))],
                           n_radii=86, n_angles=128)
    case = out['cases']['beta3']
    np.testing.assert_allclose(case['radial_ratio'], 1, atol=3e-16)
    expected = 1+3*out['radii'][:, None]**2*np.cos(2*out['angles'])[None, :]
    np.testing.assert_allclose(case['angular_polar'], expected, atol=4e-6)
    np.testing.assert_allclose(case['weighted_angular_mean'], 1, atol=3e-16)


def test_nonuniform_baseline_requires_density_weighted_angle_average():
    axis, x, _ = _fixture()
    base = 1+4*x
    out = decompose_panel_d([_result('spherical', axis, base),
                            _result('skin', axis, np.ones_like(x))],
                           n_radii=35, n_angles=128)
    case = out['cases']['skin']
    np.testing.assert_allclose(case['radial_ratio'], 1, atol=5e-16)
    weights = out['base_polar']/out['J_base'][:, None]
    np.testing.assert_allclose((weights*case['angular_polar']).mean(axis=1), 1, atol=5e-16)
    # At q=.17 the exact uniform mean of 1/(1+.68*cos(alpha)) is >1.36.
    assert case['angular_polar'][-1].mean() > 1.35
    assert case['full_ratio_polar'][-1].mean() != pytest.approx(case['radial_ratio'][-1], rel=.1)


def test_coordinate_order_and_periodic_endpoint_are_explicit():
    axis, x, y = _fixture()
    out = decompose_panel_d([_result('spherical', axis, np.ones_like(x)),
                            _result('beta2', axis, 1+.5*x-.8*y)],
                           n_radii=18, n_angles=64)
    case = out['cases']['beta2']
    expected = 1+out['radii'][:, None]*(.5*np.cos(out['angles'])-.8*np.sin(out['angles']))[None, :]
    np.testing.assert_allclose(case['angular_polar'], expected, atol=5e-16)
    assert case['angular_polar'][10, 0] == pytest.approx(1.05)
    assert case['angular_polar'][10, 16] == pytest.approx(.92)
    assert out['angles'][0] == 0
    assert out['angles'][-1] < 2*np.pi
    assert out['base_polar'].shape == (18, 64)
    np.testing.assert_allclose(case['angular_map'][case['valid_cart']], (1+.5*x-.8*y)[case['valid_cart']], atol=7e-16)


def test_display_mask_clipped_ring_is_not_forced_to_unit_weighted_mean():
    axis, x, _ = _fixture()
    out = decompose_panel_d([_result('spherical', axis, 1+4*x),
                            _result('skin', axis, np.ones_like(x))],
                           n_radii=35, n_angles=128, floor_fraction=.4)
    case = out['cases']['skin']
    assert np.isnan(case['weighted_angular_mean'][-1])
    assert abs(case['weighted_angular_mean_visible'][-1]-1) > .1
    assert case['visible_baseline_angular_fraction'][-1] < 1
    assert np.isnan(case['angular_polar'][~case['valid_polar']]).all()


def test_full_grid_peak_floor_and_missing_circle_stay_masked_not_epsilon_filled():
    axis, x, _ = _fixture()
    base = np.ones_like(x)
    base[0, 0] = 1000  # Outside the analysis disk, but inside the source panel.
    out = decompose_panel_d([_result('spherical', axis, base),
                            _result('skin', axis, np.ones_like(x))], floor_fraction=.01,
                           n_radii=35, n_angles=64)
    assert out['config']['display_threshold'] == 10
    assert not out['mask_cart'].any()
    assert np.isnan(out['cases']['skin']['full_ratio_cart']).all()
    assert np.isnan(out['cases']['skin']['radial_ratio']).all()
    short = np.linspace(-.1, .1, 51)
    flat = np.ones((51, 51))
    out = decompose_panel_d([_result('spherical', short, flat), _result('skin', short, flat)],
                           n_radii=18, n_angles=64)
    assert np.isnan(out['J_base'][12])  # q=.12 circle only partly in the square.
    assert out['mask_polar'][12].any()
    assert not out['cases']['skin']['valid_polar'][12].any()
    assert np.isnan(out['cases']['skin']['angular_polar'][12]).all()


def test_reject_wrong_orientation_version_and_axis():
    axis, x, _ = _fixture(31)
    base, case = _result('spherical', axis, np.ones_like(x)), _result('beta2', axis, np.ones_like(x))
    case['n_orientations'] = 64
    with pytest.raises(ValueError, match='requires 32'):
        decompose_panel_d([base, case])
    case['n_orientations'] = 32
    case['axis'] = axis+1e-5
    with pytest.raises(ValueError, match='axes must match'):
        decompose_panel_d([base, case])
    with pytest.raises(ValueError, match='exactly one spherical'):
        decompose_panel_d([])
