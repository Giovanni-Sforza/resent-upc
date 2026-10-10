"""Independent physics moments, same-pair response, migration and error checks."""
import numpy as np
import pytest

from upc_deformation.observable_response import (
    make_common_random, sample_decay_pairs, apply_track_response,
    summarize_observables, paired_contrast)


def coefficients(polarization=.8, sine=.3):
    radii = np.linspace(0, .3, 121)
    a0 = np.ones_like(radii)
    return np.stack((a0, polarization * a0, sine * a0)), radii


def test_polar_weight_recovers_independent_analytic_moments():
    # Constant A0: P(q<Q)=Q²/R², E[q|q<Q]=2Q/3.
    # Integrating the full spin-one polar density gives <cos2alpha>=-C/(4A0).
    coeff, r = coefficients()
    random = make_common_random(160_000, 239)
    pairs = sample_decay_pairs(coeff, r, random)
    reco = apply_track_response(pairs, random)
    result = summarize_observables(reco, q0=.06, qmax=.2)
    expected = np.array([.09, .4 / 3, -.2])
    assert np.all(np.abs(result.values - expected) < 5 * result.mc_se)
    weights = pairs.importance_weights
    np.testing.assert_allclose(np.mean(weights), 1, atol=.004)
    np.testing.assert_allclose(np.average(pairs.polar_cosine**2, weights=weights), .4, atol=.004)
    np.testing.assert_allclose(np.average(np.sin(2 * reco.alpha), weights=weights), -.3 / 4, atol=.004)
    np.testing.assert_allclose(reco.q, pairs.q_truth, atol=1e-15)
    np.testing.assert_allclose(np.cos(2*reco.alpha), np.cos(2*pairs.alpha_truth), atol=1e-13)


def test_nonconstant_radial_density_uses_qdq_and_exact_segment_integral():
    r = np.array([0., .07, .19, .3])
    a0 = 1 + 2 * r
    coeff = np.stack((a0, np.zeros_like(a0), np.zeros_like(a0)))
    random = make_common_random(120_000, 532)
    pairs = sample_decay_pairs(coeff, r, random)
    expected = (.3**3 / 3 + .3**4 / 2) / (.3**2 / 2 + 2 * .3**3 / 3)
    np.testing.assert_allclose(pairs.q_truth.mean(), expected, atol=.0008)


def test_joint_reconstruction_matches_explicit_smeared_vectors():
    coeff, r = coefficients()
    random = make_common_random(5000, 62)
    pairs = sample_decay_pairs(coeff, r, random)
    reco = apply_track_response(pairs, random, fractional_sigma=.01)
    plus = pairs.p_plus * (1 + .01 * random.momentum_normal[:, 0, None])
    minus = pairs.p_minus * (1 + .01 * random.momentum_normal[:, 1, None])
    qvec, kvec = plus + minus, (plus - minus) / 2
    np.testing.assert_allclose(reco.q, np.linalg.norm(qvec, axis=1))
    expected_cos2 = 2 * np.sum(qvec * kvec, axis=1)**2 / (
        np.sum(qvec*qvec, axis=1) * np.sum(kvec*kvec, axis=1)) - 1
    np.testing.assert_allclose(np.cos(2 * reco.alpha), expected_cos2, atol=1e-14)
    assert np.std(reco.q - pairs.q_truth) > .001


def test_truth_tail_can_migrate_into_selected_sample():
    coeff, r = coefficients(0, 0)
    random = make_common_random(70_000, 22)
    pairs = sample_decay_pairs(coeff, r, random)
    reco = apply_track_response(pairs, random, fractional_sigma=.01)
    migrated_in = (pairs.q_truth > .17) & (reco.q < .17)
    migrated_out = (pairs.q_truth < .17) & (reco.q > .17)
    assert migrated_in.sum() > 50
    assert migrated_out.sum() > 50
    with pytest.raises(ValueError, match="beyond"):
        summarize_observables(reco, qmax=.3)


def test_covariance_distinguishes_numerical_and_conditional_count_errors():
    coeff, r = coefficients(0, 0)
    random = make_common_random(100_000, 6)
    pairs = sample_decay_pairs(coeff, r, random)
    result = summarize_observables(apply_track_response(pairs, random), q0=.1, qmax=.2)
    f = result.values[0]
    np.testing.assert_allclose(result.event_covariance[0, 0], f * (1 - f), atol=1e-12)
    # Isotropic alpha is independent of radial selection and has Var(cos2alpha)=1/2.
    np.testing.assert_allclose(result.event_covariance[2, 2], .5, atol=.008)
    np.testing.assert_allclose(result.conditional_se(10_000) / result.conditional_se(1_000_000), 10)
    np.testing.assert_allclose(result.mc_covariance,
        result.event_covariance / (result.selected_fraction * (random.n_events - 1)), rtol=1e-12)
    narrower = summarize_observables(apply_track_response(pairs, random), q0=.1,
                                    qmax=.2, c2_window=(0., .1))
    np.testing.assert_allclose(narrower.event_covariance[2,2], .5 / narrower.c2_selected_fraction, rtol=.025)


def test_pairing_cancels_identical_models_and_preserves_scale_invariance():
    coeff, r = coefficients()
    random = make_common_random(20_000, 282)
    a = summarize_observables(apply_track_response(sample_decay_pairs(coeff, r, random), random))
    b = summarize_observables(apply_track_response(sample_decay_pairs(3*coeff, r, random), random))
    contrast = paired_contrast(a, b)
    np.testing.assert_allclose(contrast["delta"], 0, atol=1e-12)
    np.testing.assert_allclose(contrast["mc_se"], 0, atol=1e-12)
    exact = paired_contrast(a, a)
    assert exact["mc_se"] == [0., 0., 0.]


def test_physical_polar_density_bound_and_invalid_windows():
    coeff, r = coefficients(1.1, 0)
    random = make_common_random(100, 83)
    with pytest.raises(ValueError, match="Unintegrated"):
        sample_decay_pairs(coeff, r, random)
    coeff, r = coefficients()
    pairs = sample_decay_pairs(coeff, r, random)
    with pytest.raises(ValueError, match="same common"):
        apply_track_response(pairs, make_common_random(100, 84))
    with pytest.raises(ValueError, match="subset"):
        summarize_observables(apply_track_response(pairs, random), c2_window=(.1,.2))
