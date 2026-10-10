import numpy as np
import pytest

from upc_deformation.spectral_weights import annular_yields, statistics_from_yields, validate_edges


def test_nonzero_lower_cut_and_disjoint_annuli_partition_pixels():
    axis = np.arange(-2., 2.01, .5)
    image = 1. + np.arange(axis.size**2).reshape(axis.size, axis.size)/100.
    edges = [.5, 1., 1.5, 2.]
    result = annular_yields(image, axis, edges)
    x, y = np.meshgrid(axis, axis, indexing='ij')
    q = np.hypot(x, y)
    selected = (q >= .5) & (q <= 2.)
    assert result['pixel_counts'].sum() == selected.sum()
    np.testing.assert_allclose(result['yield'].sum(), image[selected].sum()*.25)
    # Exactly q=1 belongs only to the middle annulus; q=2 stays in the last.
    impulses = np.zeros_like(image)
    impulses[(q == .5) | (q == 1.) | (q == 1.5) | (q == 2.)] = 1.
    selected_yields = annular_yields(impulses, axis, edges)['yield']
    np.testing.assert_array_equal(selected_yields, [1., 1., 2.])


def test_raw_mean_yields_are_combined_before_normalized_ratios():
    samples = np.array([[1., 2., 7.], [9., 3., 1.]])
    result = statistics_from_yields(samples)
    mean = samples.mean(0)
    np.testing.assert_allclose(result['fractions'], mean/mean.sum())
    np.testing.assert_allclose(result['ratios'], mean[1:]/mean[:-1])
    assert not np.allclose(result['fractions'], (samples/samples.sum(1)[:, None]).mean(0))
    assert not np.allclose(result['ratios'], (samples[:, 1:]/samples[:, :-1]).mean(0))


def test_global_intensity_rescaling_preserves_observables_and_covariance():
    samples = np.array([[1., 2., 7.], [9., 3., 1.], [3., 4., 6.]])
    before, after = [statistics_from_yields(v) for v in (samples, 237.9*samples)]
    for key in ('fractions', 'ratios', 'joint_orientation_covariance'):
        np.testing.assert_allclose(before[key], after[key], rtol=2e-14, atol=2e-15)


def test_joint_covariance_matches_independent_analytic_jacobian():
    samples = np.array([[9., 5., 2.], [10., 3., 1.], [8., 4., 3.], [11., 6., 2.]])
    result = statistics_from_yields(samples)
    mean, n = samples.mean(0), len(samples)
    f = mean/mean.sum()
    jac_f = (np.eye(3)-f[:, None])/mean.sum()
    jac_r = np.array([[-mean[1]/mean[0]**2, 1/mean[0], 0.],
                      [0., -mean[2]/mean[1]**2, 1/mean[1]]])
    jac = np.vstack((jac_f, jac_r))
    expected = jac@np.cov(samples, rowvar=False, ddof=1)@jac.T/n
    np.testing.assert_allclose(result['joint_orientation_covariance'], expected, rtol=1e-13, atol=1e-17)
    np.testing.assert_allclose(result['fractions_orientation_covariance'].sum(0), 0., atol=1e-17)
    assert np.linalg.eigvalsh(result['joint_orientation_covariance']).min() > -1e-14


def test_shared_middle_bin_gives_correlated_ratio_uncertainties():
    result = statistics_from_yields([[10., 19., 10.], [10., 21., 10.]])
    cov = result['ratios_orientation_covariance']
    assert cov[0, 1] < 0.
    np.testing.assert_allclose(cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1]), -1.)
    deterministic = statistics_from_yields([[10., 20., 10.]])
    np.testing.assert_array_equal(deterministic['joint_orientation_covariance'], np.zeros((5, 5)))


def test_rejects_unphysical_edges_and_missing_annular_weight():
    for edges in ([0., .1, .2], [-.01, .1, .2, .3], [.01, .1, .1, .3]):
        with pytest.raises(ValueError):
            validate_edges(edges)
    with pytest.raises(ValueError):
        statistics_from_yields([[1., 2., 0.], [2., 3., 0.]])
    with pytest.raises(ValueError):
        statistics_from_yields([[1., -1., 1.]])
