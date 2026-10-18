import numpy as np
import pytest

from upc_deformation.diffraction_analysis import (
    analyze_diffraction_case, analyze_diffraction_cases, contrast_cases,
    register_spherical_windows,
)
from upc_deformation.precision_analysis import PixelMomentProjector


def grid_and_registry():
    radii = np.linspace(0, .29, 401)
    axis = np.linspace(-.2, .2, 81)
    q = np.linspace(.01, .17, 33)
    angles = np.arange(96)*2*np.pi/96
    density = np.exp(-(radii/.065)**2)
    registry = register_spherical_windows(radii, density)
    return radii, axis, q, angles, registry


def bank(radii, amplitudes, full_c2, self_c2):
    amplitudes = np.asarray(amplitudes)
    envelope = np.exp(-(radii/.065)**2)
    out = np.zeros((len(amplitudes), 2, 3, len(radii)))
    out[:, :, 0] = amplitudes[:, None, None]*envelope
    for branch, angular in enumerate((full_c2, self_c2)):
        out[:, branch, 1] = -4*np.asarray(angular)[:, None]*out[:, branch, 0]
    return out


def test_primary_lobe_registration_uses_density_without_jacobian():
    r, _, _, _, registry = grid_and_registry()
    assert registry['peak']['q'] == 0
    half = registry['crossings']['0.5']
    expected = .065*np.sqrt(np.log(2))
    assert half['bracket_q'][0] <= expected <= half['bracket_q'][1]
    assert abs(half['q']-expected) < np.diff(r).max()
    assert registry['first_minimum']['status'] == 'right_censored'
    w = registry['windows']
    assert w['bright_lobe_inner']['low'] == .03
    assert w['bright_lobe_inner']['high'] == w['outer_slope']['low']
    assert w['bright_lobe']['high'] == w['outer_slope']['high']
    assert w['fixed_012_014']['low'] == .12
    assert w['fixed_012_014']['high'] == .14


def test_censored_boundary_never_becomes_qmax_or_secondary_lobe_crossing():
    r = np.arange(10)*.02
    d = np.array([1., .9, .7, .4, .2, .15, .2, .7, 1.2, .05])
    registry = register_spherical_windows(r, d, qmax=.17)
    assert registry['peak']['q'] == 0  # The larger second lobe is not primary.
    assert registry['first_minimum']['q'] == .1
    assert registry['outer_crossing']['q'] is None
    assert registry['outer_crossing']['status'] == 'not_reached_before_first_minimum'
    assert registry['windows']['outer_slope']['status'] == 'censored_boundary'
    assert registry['windows']['bright_lobe']['status'] == 'censored_boundary'
    assert registry['windows']['fixed_012_014']['status'] == 'registered'
    # The would-be crossing between the last accepted node and .18 is not data
    # within qmax; no interpolated .17 sample is silently manufactured.
    d = np.linspace(1., 0., len(r))
    registry = register_spherical_windows(r, d, qmax=.17, outer_fractions=(.5, .05))
    assert registry['last_observed_q'] == .16
    assert registry['outer_crossing']['status'] == 'right_censored'
    assert registry['outer_crossing']['q'] is None


def test_cartesian_curve_and_raw_annuli_match_existing_observable():
    r, a, q, phi, registry = grid_and_registry()
    c = bank(r, [1., 3., 2., 4.], [.02, .11, -.04, .07], [.01, .04, -.02, .03])
    result = analyze_diffraction_case(c, r, a, q, phi, registry, keep_samples=True, chunk_size=2)
    original = PixelMomentProjector(r, a, q, phi).project(c[:, 0])
    np.testing.assert_allclose(result['_samples']['curves_N0'][:, 0], original['orientation_denominator'])
    np.testing.assert_allclose(result['_samples']['curves_N2'][:, 0], original['orientation_C2_numerator'], atol=1e-16)
    xx, yy = np.meshgrid(a, a, indexing='ij')
    radius2 = xx*xx+yy*yy
    cos2 = np.divide(xx*xx-yy*yy, radius2, out=np.zeros_like(xx), where=radius2 > 0)
    image = original['mean_raw_image']
    for j, (lo, hi) in enumerate(result['windows']['bounds']):
        mask = (radius2 >= lo*lo) & (radius2 < hi*hi)
        n0 = image[mask].sum()*(a[1]-a[0])**2
        n2 = (image*cos2)[mask].sum()*(a[1]-a[0])**2
        np.testing.assert_allclose(result['windows']['full']['N0']['value'][j], n0)
        np.testing.assert_allclose(result['windows']['full']['C2']['value'][j], n2/n0)
    assert 'C2' not in result['curves']['cross']
    assert 'C2' not in result['windows']['cross']


def test_ratio_is_yield_weighted_not_mean_orientation_c2():
    r, a, q, phi, registry = grid_and_registry()
    weights = np.array([1., 5., 2., 4.])
    v = np.array([.02, .12, -.03, .06])
    c = bank(r, weights, v, v-.01)
    result = analyze_diffraction_case(c, r, a, q, phi, registry, keep_samples=True)
    num, den = (result['_samples'][k][:, 0] for k in ('curves_N2', 'curves_N0'))
    pointwise = num/den
    expected = np.average(pointwise, weights=weights, axis=0)
    np.testing.assert_allclose(result['curves']['full']['C2']['value'], expected, atol=1e-15)
    assert np.max(abs(expected-pointwise.mean(0))) > .01
    np.testing.assert_allclose(result['radial']['D']['value'], weights.mean()*np.exp(-(result['radial']['q']/.065)**2))


def test_full_self_covariance_and_cross_case_pairing_are_retained():
    r, a, q, phi, registry = grid_and_registry()
    weights = [1., 5., 2., 4.]
    v = np.array([.02, .12, -.03, .06])
    c = bank(r, weights, v, v-.01)
    constant_difference = analyze_diffraction_case(c, r, a, q, phi, registry)
    assert constant_difference['curves']['full']['C2']['orientation_se'].max() > .01
    assert constant_difference['curves']['full_minus_self_C2']['orientation_se'].max() < 1e-15

    ref = analyze_diffraction_case(bank(r, weights, v, -.3*v), r, a, q, phi, registry, keep_samples=True)
    case = analyze_diffraction_case(bank(r, weights, v+.01, -.3*v), r, a, q, phi, registry, keep_samples=True)
    paired = contrast_cases(case, ref, paired=True)
    independent = contrast_cases(case, ref)
    assert independent['curves']['double_difference_C2']['orientation_se'].max() > .01
    assert paired['curves']['double_difference_C2']['orientation_se'].max() < 1e-15
    for section in ('curves', 'windows'):
        np.testing.assert_allclose(paired[section]['double_difference_C2']['value'],
                                   paired[section]['full_C2']['value'], atol=1e-15)
    case.pop('_samples')
    with pytest.raises(ValueError, match='keep_samples'):
        contrast_cases(case, ref, paired=True)


def test_joint_outer_yield_ratio_and_c2_covariance_includes_common_events():
    r, a, q, phi, registry = grid_and_registry()
    c = bank(r, [1., 2., 4., 3.], [.01, .03, .06, .08], [0., 0., 0., 0.])
    # Vary radial widths in the same ordering as angular moments; this supplies
    # genuine covariance between the two measurements on one orientation bank.
    for i, width in enumerate((.060, .063, .067, .071)):
        c[i] *= (np.exp(-(r/width)**2)/np.exp(-(r/.065)**2))[None, None, :]
    out = analyze_diffraction_case(c, r, a, q, phi, registry, keep_samples=True)
    joint = out['windows']['joint_outer_ratio_C2']
    assert len(joint['labels']) == 1+len(out['windows']['names'])
    assert np.max(abs(joint['covariance'][0, 1:])) > 1e-7
    assert np.linalg.eigvalsh(joint['covariance']).min() > -1e-14
    np.testing.assert_allclose(joint['value'][0], out['windows']['outer_over_inner']['full']['value'][0])
    np.testing.assert_allclose(joint['value'][1:], out['windows']['full']['C2']['value'])


def test_deterministic_reference_and_private_arrays_are_not_retained_by_wrapper():
    r, a, q, phi, registry = grid_and_registry()
    sphere = bank(r, [1.], [0.], [0.])
    case = bank(r, [1., 2., 3., 4.], [.01, .02, .03, .04], [0., 0., 0., 0.])
    result = analyze_diffraction_cases({'spherical': sphere, 'beta2': {'orientation_coefficients': case}},
                                      r, a, q, phi, registry=registry)
    for entry in result['cases'].values():
        assert '_samples' not in entry
    np.testing.assert_allclose(result['contrasts']['beta2']['curves']['full_C2']['orientation_se'],
                               result['cases']['beta2']['curves']['full']['C2']['orientation_se'])


def test_full_self_positivity_and_coordinate_contract_are_enforced():
    r, a, q, phi, registry = grid_and_registry()
    c = bank(r, [1.], [0.], [0.])
    with pytest.raises(ValueError, match='ordered full,self'):
        analyze_diffraction_case(c, r, a, q, phi, registry, branch_names=('self', 'full'))
    with pytest.raises(ValueError, match='exclude undefined q=0'):
        analyze_diffraction_case(c, r, a, np.r_[0., q], phi, registry)
    c[:, 1, 1] = 3*c[:, 1, 0]
    with pytest.raises(ValueError, match='negative full/self'):
        analyze_diffraction_case(c, r, a, q, phi, registry)


def test_real_diffraction_zero_is_retained_while_undefined_ratios_are_masked():
    r = np.linspace(0, .3, 121)
    a = np.linspace(-.2, .2, 81)
    q = np.array([.01, .03, r[28], .1, .17])
    phi = np.arange(96)*2*np.pi/96
    c = np.zeros((1, 2, 3, len(r)))
    c[0, 0, 0] = np.exp(-(r/.065)**2)
    c[0, 1, 0] = ((r-r[28])/r[28])**2*c[0, 0, 0]
    registry = register_spherical_windows(r, c[0, 1, 0], inner_min=.005)
    result = analyze_diffraction_case(c, r, a, q, phi, registry)
    assert result['radial']['D']['value'][28] == 0
    ratio = result['radial']['J_over_D']
    assert not ratio['valid_mask'][28]
    assert ratio['omitted_q'][0] == r[28]
    assert np.isfinite(ratio['value']).all()
    diagnostic = result['curves']['analytic_C2_diagnostic']['self']
    assert not diagnostic['valid_mask'][2]
    assert np.isfinite(diagnostic['value']).all()
    # Pixel interpolation gives a finite circle average around the zero, which
    # is deliberately retained as the primary Cartesian observable.
    assert np.isfinite(result['curves']['self']['C2']['value']).all()


def test_diagnostic_figures_smoke_from_compact_summaries(tmp_path):
    from pathlib import Path
    from upc_deformation.diffraction_figures import render_diffraction_figures
    r, a, q, phi, registry = grid_and_registry()
    data = analyze_diffraction_cases({
        'spherical': bank(r, [1.], [0.], [0.]),
        'beta2': bank(r, [1., 2., 3., 4.], [.01, .02, .03, .04], [0., 0., 0., 0.])},
        r, a, q, phi, registry=registry)
    data['cases']['beta2']['case'] = {'beta2': .3, 'beta3': 0., 'skin_fm': 0.}
    data['config'] = {}
    files = render_diffraction_figures(data, tmp_path)
    assert len(files) == 6
    assert {Path(p).suffix for p in files} == {'.png', '.pdf', '.svg'}
    assert all(Path(p).is_absolute() and Path(p).stat().st_size > 1000 for p in files)
