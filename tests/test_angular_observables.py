"""Analytic integration, conditional angular statistics, and paired errors."""
import copy
import json

import numpy as np
import pytest

from upc_deformation.angular_observables import (
    BIN_EDGES_GEV, METRIC_NAMES, analyze_angular, coefficient_bin_moments,
    orientation_angular_statistics, radial_moment,
)


def test_exact_nonuniform_linear_radial_moments_with_off_node_endpoints():
    r = np.array([0., .009, .024, .081, .111, .19])
    a, b, low, high = 2.7, -3.2, .017, .162
    values = np.stack((a + b*r, 4 - .7*r))
    for power in range(4):
        expected = np.array([
            aa*(high**(power+1)-low**(power+1))/(power+1)
            + bb*(high**(power+2)-low**(power+2))/(power+2)
            for aa, bb in [(a, b), (4, -.7)]])
        np.testing.assert_allclose(radial_moment(values, r, low, high, power), expected,
                                   rtol=3e-15, atol=1e-18)


def test_constant_annulus_has_correct_radial_measure_and_angular_moments():
    r = np.array([0., .017, .037, .087, .2])
    a0 = np.full_like(r, 2.3)
    c = np.stack((a0, .4*a0, -.2*a0))
    den, num = coefficient_bin_moments(c, r)
    lo, hi = BIN_EDGES_GEV.T
    np.testing.assert_allclose(den, np.pi*2.3*(hi**2-lo**2), rtol=1e-14)
    values = num/den[:, None]
    np.testing.assert_allclose(values[:, :3], np.tile([-.1, .05, .4/np.pi], (4, 1)), rtol=1e-14)
    np.testing.assert_allclose(values[:, 3], 2/3*(hi**3-lo**3)/(hi**2-lo**2), rtol=1e-14)


def test_quadrants_match_c2_and_cancel_sine_term_numerically():
    # Midpoints avoid angular sector boundaries.  Integrate every sector, not
    # merely the closed form used in the implementation.
    phi = (np.arange(400000)+.5)*2*np.pi/400000
    for d in [0., .31, -.45]:
        intensity = 2.0 - .5*(.7*np.cos(2*phi)+d*np.sin(2*phi))
        parallel = np.cos(2*phi) > 0
        contrast = (intensity[~parallel].sum()-intensity[parallel].sum())/intensity.sum()
        c2 = (intensity*np.cos(2*phi)).sum()/intensity.sum()
        s2 = (intensity*np.sin(2*phi)).sum()/intensity.sum()
        assert contrast == pytest.approx(-4*c2/np.pi, rel=5e-11)
        assert s2 == pytest.approx(-d/8, abs=3e-16)


def test_orientation_yield_weighting_and_joint_paired_covariance():
    r = np.array([0., .011, .047, .09, .18])
    amplitudes = np.array([1., 2., 7., 20.])
    # Perfectly correlated full and self polarization fluctuations; pairing
    # must remove them in the interaction even though either SE is nonzero.
    ratios = np.array([-.5, -.1, .2, .6])
    full = np.zeros((4, 3, len(r)))
    full[:, 0] = amplitudes[:, None]
    full[:, 1] = (amplitudes*ratios)[:, None]
    full[:, 2] = (.3*amplitudes*ratios)[:, None]
    self_coeff = 3*full
    stat = orientation_angular_statistics(full, r)
    other = orientation_angular_statistics(self_coeff, r)
    expected = -.25*np.dot(amplitudes, ratios)/amplitudes.sum()
    assert stat['values'][0, 0] == pytest.approx(expected)
    assert abs(expected - (-.25*ratios.mean())) > .04
    assert stat['se'][0, 0] > 0
    influence = stat['_influence'].reshape(4, -1)
    np.testing.assert_allclose(stat['covariance'], np.cov(influence, rowvar=False, ddof=1)/4, atol=1e-18)
    np.testing.assert_allclose(stat['_influence'].mean(0), 0, atol=1e-16)
    paired = stat['_influence']-other['_influence']
    np.testing.assert_allclose(paired, 0, atol=2e-16)
    # Same random orientation contributes to every bin, so cross-bin entries
    # must not be erased by treating bins as independent estimates.
    assert stat['covariance'][0, 4] == pytest.approx(stat['covariance'][0, 0])
    np.testing.assert_allclose(stat['covariance'][2], -4/np.pi*stat['covariance'][0], atol=1e-18)


def _cache(name='spherical', n=1, scale=1., anisotropy=.2):
    r = np.array([0., .006, .027, .066, .113, .18, .25])
    ori = np.zeros((n, 3, 2, 3, len(r)))
    for i in range(n):
        for stage in range(3):
            for branch in range(2):
                a0 = scale*(1+.2*i)*(1+.3*stage)*(1+.5*branch)*(1-.6*r)
                ori[i, stage, branch] = [a0, a0*(anisotropy+.03*i), a0*.02*i]
    axis = np.linspace(-.2, .2, 31)
    xx, yy = np.meshgrid(axis, axis, indexing='ij')
    image = np.exp(-40*(xx*xx+yy*yy))*(1+.15*xx/.2)
    images = np.tile(image, (4, 2, 1, 1))*scale
    mean = ori.mean(0)
    background = np.stack((.1*np.exp(-4*r*r), np.zeros_like(r), np.zeros_like(r)))
    return {'case': {'name': name, 'beta2': 0. if name=='spherical' else .3,
                     'beta3': 0., 'skin_fm': 0.},
            'radii': r, 'axis': axis, 'coefficients': mean,
            'orientation_coefficients': ori, 'images': images,
            'coarse_b_coefficients': mean[-1].copy(), 'background_coefficients': background}


def test_analysis_json_isotropic_dilution_pairing_and_unequal_counts():
    base, case = _cache(anisotropy=0.), _cache('beta2', n=8, scale=2., anisotropy=.2)
    original = copy.deepcopy(case)
    result = analyze_angular([base, case], {'qmax': .17})
    assert result['metric_names'] == METRIC_NAMES
    assert result['cases']['beta2']['n_orientations'] == 8
    assert list(result['cases']['beta2']['orientation_prefix']) == ['8']
    assert list(result['cases']['spherical']['orientation_prefix']) == ['1']
    sphere = result['cases']['spherical']['stages']['decay_by']['full']
    np.testing.assert_allclose(np.asarray(sphere['values'])[:, :3], 0)
    stage = result['cases']['beta2']['stages']['decay_by']
    joint = np.asarray(stage['joint_covariance'])
    covariance = np.asarray(stage['contrast']['interaction_covariance'])
    np.testing.assert_allclose(covariance, joint[:16,:16]+joint[16:,16:]-joint[:16,16:]-joint[16:,:16], atol=1e-18)
    np.testing.assert_allclose(stage['contrast']['interaction_se'], 0, atol=1e-16)
    bg = result['cases']['beta2']['background']
    np.testing.assert_allclose(np.asarray(bg['total_values'])[:, :3],
                               np.asarray(bg['full_values'])[:, :3]*np.asarray(bg['angular_dilution'])[:, None], atol=1e-16)
    np.testing.assert_allclose(np.asarray(bg['fraction_of_total'])+np.asarray(bg['angular_dilution']), 1)
    np.testing.assert_allclose(np.asarray(bg['background_to_signal'])/(1+np.asarray(bg['background_to_signal'])), bg['fraction_of_total'])
    assert result['cases']['beta2']['bridge']['known_b_x_uncertainty'] is None
    json.dumps(result, allow_nan=False)
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, case[key])


def test_empty_bins_are_marked_undefined_and_empty_disk_is_rejected():
    r = np.array([0., .01, .02, .03, .06, .1, .14, .17, .2])
    c = np.zeros((1, 3, len(r)))
    c[0, 0, :2] = 1
    stat = orientation_angular_statistics(c, r, deterministic=True)
    assert not np.any(stat['valid'])
    assert np.all(np.isnan(stat['values']))
    assert np.all(np.isnan(stat['se']))
    with pytest.raises(ValueError, match='nonpositive disk'):
        orientation_angular_statistics(np.zeros_like(c), r)
    with pytest.raises(ValueError, match='strictly increasing'):
        radial_moment(np.ones(3), [0., .1, .1], 0., .1)
    with pytest.raises(ValueError, match='inside radial support'):
        radial_moment(np.ones(3), [0., .1, .2], 0., .3)
    with pytest.raises(ValueError, match='at least'):
        analyze_angular([], {'qmax': .17})


def test_stale_mean_and_anisotropic_background_are_rejected():
    bad = _cache()
    bad['coefficients'] = bad['coefficients']*1.01
    with pytest.raises(ValueError, match='coefficient means'):
        analyze_angular([bad], {'qmax': .17})
    bad = _cache()
    bad['background_coefficients'][1] = .01
    with pytest.raises(ValueError, match='isotropic'):
        analyze_angular([bad], {'qmax': .17})


def test_zero_tail_curve_and_bins_are_json_null_without_fabricated_error():
    base = _cache(anisotropy=0.)
    base['orientation_coefficients'][..., 3:] = 0
    base['coefficients'] = base['orientation_coefficients'].mean(0)
    base['coarse_b_coefficients'] = base['coefficients'][-1].copy()
    result = analyze_angular([base], {'qmax': .17})
    stats = result['cases']['spherical']['stages']['decay_by']['full']
    assert stats['valid'] == [True, True, False, False]
    assert stats['values'][-1] == [None]*4
    assert stats['se'][-1] == [None]*4
    assert stats['curves']['C2'][-1] is None
    assert stats['curves']['C2_se'][-1] is None
    assert stats['curves']['valid'][-1] is False
    json.dumps(result, allow_nan=False)


def test_configured_curve_and_registered_bins_are_respected():
    result = analyze_angular([_cache()], {'qmax': .17, 'curve_qmin': .01, 'curve_points': 251,
                                        'angular_bins_gev': BIN_EDGES_GEV.tolist()})
    assert len(result['curve_radii_GeV']) == 251
    assert result['curve_radii_GeV'][0] == .01
    assert result['curve_radii_GeV'][-1] == .17
    assert len(result['cases']['spherical']['stages']['decay_by']['full']['curves']['C2']) == 251
    with pytest.raises(ValueError, match='four registered'):
        analyze_angular([_cache()], {'angular_bins_gev': [[.04, .08]]})
    with pytest.raises(ValueError, match='curve_points'):
        analyze_angular([_cache()], {'curve_points': 1})
    with pytest.raises(ValueError, match='curve_qmin'):
        analyze_angular([_cache()], {'curve_qmin': 0.})


def test_independent_halves_use_covariance_sum_and_within_half_pairing():
    base, endpoint = _cache(), _cache('beta2', n=64)
    result = analyze_angular([base, endpoint], {'curve_points': 7})
    halves = result['cases']['beta2']['independent_halves']
    assert halves['first_count'] == halves['second_count'] == 32
    for branch in ['full', 'self', 'interaction']:
        first, second, diff = [np.asarray(halves[k][branch]['values']) for k in ['first', 'second', 'difference']]
        np.testing.assert_allclose(diff, second-first)
        se1, se2, se_delta = [np.asarray(halves[k][branch]['se']) for k in ['first', 'second', 'difference']]
        np.testing.assert_allclose(se_delta**2, se1**2+se2**2, atol=1e-20)
        cov1, cov2, cov_delta = [np.asarray(halves[k][branch]['covariance']) for k in ['first', 'second', 'difference']]
        np.testing.assert_allclose(cov_delta, cov1+cov2, atol=1e-20)
    assert halves['first']['full']['se'][0][0] > 0
    np.testing.assert_allclose(halves['first']['interaction']['se'], 0, atol=1e-16)
    assert 'independent_halves' not in result['cases']['spherical']
    # A synthetic repeated bank is not a fresh independent shard.
    endpoint['rotation_bank_1'] = np.tile(np.eye(3), (64, 1, 1))
    endpoint['rotation_bank_2'] = endpoint['rotation_bank_1'].copy()
    with pytest.raises(ValueError, match='identical two rotation banks'):
        analyze_angular([base, endpoint], {'curve_points': 7})
