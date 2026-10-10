"""Independent angular moments, count conventions, pairing and orientation tests."""
import json

import numpy as np
import pytest

from upc_deformation.angular_response import (
    DEFAULT_BINS, orientation_reweight, paired_mc_covariance,
    run_angular_response, summarize_angular_bins,
)
from upc_deformation.observable_response import (
    apply_track_response, make_common_random, sample_decay_pairs,
)


def fixture_coeff(c=.7, d=.2, slope=0.):
    radii = np.linspace(0., .3, 101)
    a = 1 + slope * radii
    return np.stack((a, c*a, d*a)), radii


def case_result(name, orientation_coefficients, radii):
    ori = np.asarray(orientation_coefficients)
    raw = np.repeat(np.repeat(ori[:, None, None], 3, axis=1), 2, axis=2)
    return {"case": {"name": name}, "radii": radii,
            "orientation_coefficients": raw, "coefficients": raw.mean(0),
            "rotation_bank_1": np.repeat(np.eye(3)[None], len(ori), axis=0),
            "rotation_bank_2": np.repeat(np.eye(3)[None], len(ori), axis=0)}


def test_angular_moments_and_complementary_quadrants_match_analytic_density():
    coeff, r = fixture_coeff()
    random = make_common_random(180_000, 661)
    pairs = sample_decay_pairs(coeff, r, random)
    summary = summarize_angular_bins(apply_track_response(pairs, random))
    for ibin, (a, b) in enumerate(DEFAULT_BINS):
        expected = np.array([-.7/4, -.2/4, .7/np.pi,
                             2/3 * (b**3-a**3)/(b**2-a**2)])
        error = np.sqrt(np.diag(summary.mc_covariance))[4*ibin:4*(ibin+1)]
        assert np.all(np.abs(summary.values[ibin] - expected) < 5*error)
    # Any departure Aphi+4/pi*C2 is pure quadrature noise before response.
    vector = np.tile([4/np.pi, 0, 1, 0], (4, 1))
    for ibin in range(4):
        sub = slice(4*ibin, 4*(ibin+1))
        residual = summary.values[ibin] @ vector[ibin]
        error = np.sqrt(vector[ibin] @ summary.mc_covariance[sub, sub] @ vector[ibin])
        assert abs(residual) < 5*error


def test_total_selected_and_per_bin_event_covariances_have_distinct_count_denominators():
    coeff, r = fixture_coeff(0, 0)
    random = make_common_random(150_000, 221)
    pairs = sample_decay_pairs(coeff, r, random)
    summary = summarize_angular_bins(apply_track_response(pairs, random))
    for ibin in range(4):
        fraction = summary.denominator[ibin] / summary.selected_weight
        sub = slice(4*ibin, 4*(ibin+1))
        event_bin = summary.event_covariance_per_bin_event[ibin]
        np.testing.assert_allclose(event_bin[0, 0], .5, atol=.012)
        np.testing.assert_allclose(event_bin[2, 2], 1, atol=.002)
        np.testing.assert_allclose(event_bin[0, 2], -2/np.pi, atol=.015)
        np.testing.assert_allclose(summary.event_covariance_per_total_selected_event[sub, sub],
                                   event_bin / fraction, rtol=1e-13)
    # With unit weights, importance-quadrature error is the conditional event
    # covariance divided by the number of selected random proposals (minus 1).
    np.testing.assert_allclose(summary.mc_covariance,
        summary.event_covariance_per_total_selected_event * len(pairs.q_truth)
        / (summary.selected_weight * (len(pairs.q_truth)-1)), rtol=1e-12, atol=1e-20)


def test_compact_paired_covariance_matches_dense_and_keeps_cross_bin_terms():
    coeff, r = fixture_coeff()
    random = make_common_random(12_000, 802)
    pairs = sample_decay_pairs(coeff, r, random)
    a = summarize_angular_bins(apply_track_response(pairs, random))
    b = summarize_angular_bins(apply_track_response(pairs, random, fractional_sigma=.01))
    difference = b.influence_chunk(0, random.n_events) - a.influence_chunk(0, random.n_events)
    expected = np.cov(difference, rowvar=False, ddof=1) / len(difference)
    actual = paired_mc_covariance(b, a, chunk_size=1777)
    np.testing.assert_allclose(actual, expected, atol=2e-19, rtol=1e-11)
    assert abs(actual[0, 4]) > 1e-7
    np.testing.assert_array_equal(paired_mc_covariance(a, a), np.zeros((16, 16)))
    assert np.min(np.linalg.eigvalsh(actual)) > -1e-14


def test_finite_truth_tail_migrates_into_and_out_of_bins():
    coeff, r = fixture_coeff(0, 0)
    random = make_common_random(60_000, 987)
    pairs = sample_decay_pairs(coeff, r, random)
    summary = summarize_angular_bins(apply_track_response(pairs, random, fractional_sigma=.01))
    assert summary.migration["truth_above_qmax_fraction_of_reco_selected"] > .005
    assert summary.migration["reco_above_qmax_fraction_of_truth_selected"] > .005
    assert all(row["fraction_entered_from_outside_truth_bin"] > .03
               for row in summary.migration["per_bin"])
    with pytest.raises(ValueError, match="beyond"):
        summarize_angular_bins(apply_track_response(pairs, random, fractional_sigma=.01), qmax=.3)


def test_orientation_reweight_preserves_raw_yields_and_shared_response_kinematics():
    coeff1, radii = fixture_coeff(.1, .2)
    coeff2, _ = fixture_coeff(.8, -.3, 2)
    result = case_result("shape", [coeff1, 4*coeff2, 2*coeff1, 3*coeff2], radii)
    random = make_common_random(25_000, 84)
    pairs = sample_decay_pairs(result["coefficients"][-1, 0], radii, random)
    summaries = [summarize_angular_bins(apply_track_response(pairs, random, fractional_sigma=s))
                 for s in (0, .01)]
    orientation = orientation_reweight(result, pairs, summaries)
    np.testing.assert_allclose(orientation["closure"], 0, atol=1e-14)
    np.testing.assert_allclose(orientation["influence"].mean(0), 0, atol=1e-14)
    assert np.max(np.diag(orientation["covariance"][1])) > 1e-4
    # Ratios of averaged raw orientation integrals must not become equal-weight
    # averages of separately normalized orientation shapes.
    assert abs(summaries[0].values[0, 0] - (-.1/4-.8/4)/2) > .015


def test_runner_json_pairing_scale_invariance_and_undefined_count_diagnostics():
    coeff, r = fixture_coeff()
    spherical = case_result("spherical", [coeff], r)
    identical = case_result("rescaled", [3*coeff, 3*coeff], r)
    output = run_angular_response([spherical, identical], {
        "response_events": 12_000, "response_seed": 190,
        "fractional_resolutions": [0, .005], "include_angular_toy": True}, log=lambda _: None)
    assert len(output["responses"]) == 3
    json.dumps(output, allow_nan=False)
    reference, scaled = output["cases"]
    for row in reference["rows"]:
        assert row["vs_spherical"]["delta"] == [0.]*16
        assert row["vs_spherical"]["mc_se"] == [0.]*16
        assert row["vs_spherical"]["n3_total_selected_one_sample_diagnostic"] == [None]*16
    for row in scaled["rows"]:
        np.testing.assert_allclose(row["vs_spherical"]["delta"], 0, atol=2e-15)
        np.testing.assert_allclose(row["vs_spherical"]["combined_sampling_se"], 0, atol=2e-15)
        np.testing.assert_allclose(row["orientation_reweight_closure"], 0, atol=1e-14)
    assert scaled["rows"][0]["vs_zero_response"]["combined_sampling_se"] == [0.]*16


def test_invalid_bins_and_missing_reference_fail_explicitly():
    coeff, r = fixture_coeff()
    random = make_common_random(1000, 20)
    reco = apply_track_response(sample_decay_pairs(coeff, r, random), random)
    with pytest.raises(ValueError, match="disjoint"):
        summarize_angular_bins(reco, [[.03, .1], [.06, .14]])
    with pytest.raises(ValueError, match="spherical"):
        run_angular_response([], {"response_events": 10}, log=lambda _: None)


def test_four_term_mc_covariance_matches_independent_dense_control_variate():
    from upc_deformation.angular_response import linear_mc_covariance
    coeff, r = fixture_coeff(.7, .2)
    other, _ = fixture_coeff(.3, -.1, 3.)
    random = make_common_random(10_000, 91)
    terms = []
    for model, case_sign in ((coeff, 1), (other, -1)):
        pairs = sample_decay_pairs(model, r, random)
        for sigma, response_sign in ((.01, 1), (0, -1)):
            summary = summarize_angular_bins(
                apply_track_response(pairs, random, fractional_sigma=sigma))
            terms.append((summary, case_sign*response_sign))
    dense = sum(sign*s.influence_chunk(0, random.n_events) for s, sign in terms)
    expected = np.cov(dense, rowvar=False, ddof=1) / len(dense)
    np.testing.assert_allclose(linear_mc_covariance(terms, 1801), expected, atol=1e-19, rtol=1e-11)
    summary = terms[0][0]
    np.testing.assert_array_equal(
        linear_mc_covariance([(summary, 1), (summary, -1), (summary, -1), (summary, 1)]),
        np.zeros((16, 16)))


def test_truth_anchor_zero_is_exact_and_orientation_covariance_is_preserved():
    coeff1, r = fixture_coeff(.1, .2)
    coeff2, _ = fixture_coeff(.7, -.1)
    spherical = case_result("spherical", [coeff1], r)
    shape = case_result("shape", [coeff1, 3*coeff2, 2*coeff1, 4*coeff2], r)
    output = run_angular_response([spherical, shape], {
        "response_events": 16_000, "response_seed": 370,
        "fractional_resolutions": [0, .01], "include_angular_toy": False,
        "truth_anchored_response": True}, log=lambda _: None)
    row = output["cases"][1]["rows"][0]
    anchored = row["truth_anchored"]
    values = np.array(anchored["values_flat"]).reshape(4, 4)
    # Raw yield weights 1,3,2,4; an average of normalized shapes would be wrong.
    expected_c = (.1*(1+2)+.7*(3+4))/10
    expected_d = (.2*(1+2)-.1*(3+4))/10
    np.testing.assert_allclose(values[:, 0], -expected_c/4, atol=1e-15)
    np.testing.assert_allclose(values[:, 1], -expected_d/4, atol=1e-15)
    np.testing.assert_allclose(values[:, 2], expected_c/np.pi, atol=1e-15)
    np.testing.assert_array_equal(anchored["mc_covariance"], np.zeros((16, 16)))
    np.testing.assert_array_equal(anchored["vs_spherical"]["mc_covariance"], np.zeros((16, 16)))
    np.testing.assert_allclose(anchored["orientation_covariance"],
                               anchored["exact_truth_orientation_covariance"], atol=1e-18)
    # Independent delta-method calculation from the four known orientation yields.
    w = np.array([1., 3., 2., 4.])
    individual_c2 = -np.array([.1, .7, .1, .7])/4
    influence = w*(individual_c2+expected_c/4)/w.mean()
    np.testing.assert_allclose(np.array(anchored["orientation_covariance"])[0, 0],
                               np.var(influence, ddof=1)/4, atol=1e-18)
    json.dumps(output, allow_nan=False)


def test_truth_anchor_response_shift_and_higher_harmonics_stay_independent():
    coeff, r = fixture_coeff(.7, .2)
    baseline = case_result("spherical", [coeff], r)
    output = run_angular_response([baseline], {
        "response_events": 30_000, "response_seed": 231,
        "fractional_resolutions": [0, .03], "include_angular_toy": False,
        "truth_anchored_response": True}, log=lambda _: None)
    zero, smeared = output["cases"][0]["rows"]
    anchor = smeared["truth_anchored"]
    exact = np.array(zero["truth_anchored"]["values_flat"])
    raw_shift = np.array(smeared["values_flat"])-np.array(zero["values_flat"])
    np.testing.assert_allclose(anchor["values_flat"], exact+raw_shift, atol=1e-15)
    np.testing.assert_allclose(anchor["vs_zero_response"]["delta"],
                               smeared["vs_zero_response"]["delta"], atol=1e-15)
    np.testing.assert_allclose(anchor["mc_covariance"],
                               smeared["vs_zero_response"]["mc_covariance"], atol=1e-18)
    # This broad toy response generates higher harmonics; the sector estimator
    # must remain independently evaluated and must not be set to -4*C2/pi.
    residual = np.array([b["Aphi_harmonic_residual"] for b in anchor["bins"]])
    assert np.max(abs(residual)) > .003
