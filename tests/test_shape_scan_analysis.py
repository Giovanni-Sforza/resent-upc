import csv
from pathlib import Path

import numpy as np
import pytest

from upc_deformation.shape_scan_analysis import FEATURE_NAMES, extract_tail_features, summarize_samples


def profiles(n=8, nb=2):
    q = np.linspace(0, 1, 101)
    slopes = np.linspace(7., 15., n)
    base = np.exp(-slopes[:, None]*q[None, :]**2)
    x = np.broadcast_to(base[:, None, None, :], (n, nb, 2, len(q))).copy()
    return q, x


def test_exact_extraction_conventions_and_no_crossing_after_upturn():
    q = np.arange(6, dtype=float)
    point = extract_tail_features(q, [1, .4, .05, .04, .06, .001])
    assert point["status"] == "censored"
    assert point["Q50_GeV"] == pytest.approx(5/6)
    assert point["Q10_GeV"] is not None and point["B_mid"] is not None
    assert point["Q1_GeV"] is None and point["K_tail"] is None
    assert point["mean_pT_GeV"] > 0 and point["feature_valid"][-1]
    assert point["mean_pT_support_GeV"] == [0., 5.]
    assert point["censor_reasons"]["Q1"] == "not_reached_before_upturn"
    assert point["descending_branch_end_q_GeV"] == 3
    right = extract_tail_features(q[:4], [1, .4, .05, .04])
    assert right["censor_reasons"]["Q1"] == "right_censored"


def test_mean_is_formed_before_nonlinear_extraction():
    q, x = profiles(n=4, nb=1)
    x[3] *= 12
    result = summarize_samples(q, x, bootstrap_draws=16)
    point = result["channels"][0][0]["point"]
    expected = extract_tail_features(q, x[:, 0, 0].mean(axis=0))
    np.testing.assert_equal(point, expected)
    wrong = np.mean([extract_tail_features(q, row)["B_mid"] for row in x[:, 0, 0]])
    assert abs(point["B_mid"]-wrong) > 1
    wrong_mean_pT = np.mean([extract_tail_features(q, row)["mean_pT_GeV"] for row in x[:, 0, 0]])
    assert abs(point["mean_pT_GeV"]-wrong_mean_pT) > 1e-3
    assert result["bootstrap"]["method"] == "orientation_pair_bootstrap"


def test_joint_covariance_pairing_and_batch_invariance():
    q, x = profiles()
    # Pure intensity factors leave all features unchanged; their bootstrap
    # replicates must retain the exact same orientation fluctuations.
    x[:, 1, 1] *= 7
    a = summarize_samples(q, x, bootstrap_draws=64, bootstrap_block_size=2,
                          bootstrap_batch_size=7, orientation_bank_id="ordered-bank")
    b = summarize_samples(q, x, bootstrap_draws=64, bootstrap_block_size=2,
                          bootstrap_batch_size=13, orientation_bank_id="ordered-bank")
    np.testing.assert_allclose(a["replicate_features"], b["replicate_features"], rtol=1e-14)
    assert a["bootstrap"]["resampling_plan_id"] == b["bootstrap"]["resampling_plan_id"]
    np.testing.assert_allclose(a["replicate_features"][0, 0], a["replicate_features"][1, 1], rtol=1e-13)
    channel = a["channels"][0][0]
    cov = np.array(channel["covariance"])
    assert cov.shape == (len(FEATURE_NAMES), len(FEATURE_NAMES)) and cov[0, 1] < 0
    assert cov[0, 3] > 0 and cov[1, 3] < 0
    assert np.linalg.eigvalsh(cov).min() > -1e-13
    np.testing.assert_allclose(cov, np.cov(a["replicate_features"][0, 0], rowvar=False))
    np.testing.assert_allclose(channel["orientation_se"], np.sqrt(np.diag(cov)))
    assert channel["conditional_covariance"] is None


def test_scale_law_and_shared_plan_across_cases():
    q, x = profiles()
    a = summarize_samples(q, x, bootstrap_draws=32, bootstrap_block_size=2)
    b = summarize_samples(1.3*q, 4*x, bootstrap_draws=32, bootstrap_block_size=2)
    scale = np.array([1.3, 1/1.3**2, 1., 1.3])
    np.testing.assert_allclose(b["replicate_features"], a["replicate_features"]*scale, rtol=1e-13)
    np.testing.assert_allclose(b["channels"][0][0]["covariance"],
                              np.array(a["channels"][0][0]["covariance"])*np.outer(scale, scale), rtol=1e-12, atol=1e-17)


def test_censored_replicates_do_not_create_unconditional_se():
    q = np.arange(6, dtype=float)
    thin = np.array([1, .7, .3, .09, .03, .001])
    broad = np.array([1, .8, .5, .3, .1, .04])
    raw = np.vstack((thin, thin, thin, thin, broad))
    x = np.repeat(raw[:, None, None, :], 2, axis=2)
    result = summarize_samples(q, x, bootstrap_draws=128)
    channel = result["channels"][0][0]
    assert channel["point"]["status"] == "observed"
    assert 0 < channel["valid_fraction"] < 1
    assert channel["feature_valid_fraction"][:2] == [1., 1.]
    assert channel["covariance"] is None and channel["orientation_se"] is None
    assert all(value is not None for value in channel["marginal_orientation_se"][:2])
    assert channel["marginal_orientation_se"][2] is None
    assert channel["marginal_orientation_se"][3] is not None
    np.testing.assert_allclose(channel["marginal_orientation_se"][:2],
                               result["replicate_features"][0, 0, :, :2].std(axis=0, ddof=1))
    assert np.shape(channel["conditional_covariance"]) == (len(FEATURE_NAMES), len(FEATURE_NAMES))
    assert channel["covariance_scope"].startswith("conditional")
    valid = result["replicate_valid"][0, 0].all(axis=1)
    np.testing.assert_allclose(channel["conditional_covariance"],
                              np.cov(result["replicate_features"][0, 0, valid], rowvar=False))
    assert sum(channel["censor_reason_counts"].values()) == 128-int(valid.sum())


def test_deterministic_singleton_and_prefix_policy():
    q, x = profiles(n=1)
    out = summarize_samples(q, x, bootstrap_draws=8)
    assert out["bootstrap"]["method"] == "deterministic"
    np.testing.assert_array_equal(out["channels"][0][0]["covariance"],
                                  np.zeros((len(FEATURE_NAMES), len(FEATURE_NAMES))))
    assert out["channels"][0][0]["orientation_se"] == [0.]*len(FEATURE_NAMES)
    q, x = profiles(n=8)
    # Exactly one requested block must fall back rather than report zero MC.
    a = summarize_samples(q, x[:4], bootstrap_draws=16, bootstrap_block_size=4)
    assert a["bootstrap"]["effective_block_size"] == 1
    assert a["channels"][0][0]["orientation_se"][0] > 0
    with pytest.raises(ValueError, match="divisible"):
        summarize_samples(q, x[:7], bootstrap_draws=8, bootstrap_block_size=4)
    bad = x.copy(); bad[0, 0, 0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        summarize_samples(q, bad, bootstrap_draws=8)


def test_existing_four_case_primary_results_when_archive_is_available():
    source = (Path(__file__).parents[1]/"outputs/diffraction_imports/20261023_6988784cbfd5/analysis_n32768/radial_density.csv")
    if not source.exists():
        pytest.skip("local archived curves are not shipped as test fixtures")
    expected = {
        "spherical": (.11789214749476923, 197.70646377830917, 1.1259060633515303),
        "beta2": (.11787585994894668, 196.06489630837302, 1.008167813789098),
        "beta3": (.11742734023581525, 199.2236328056391, 1.0542718462313618),
        "skin": (.11516581266583213, 209.78652187694513, 1.121714096320723),
    }
    with source.open() as f:
        rows = list(csv.DictReader(f))
    for case, values in expected.items():
        selected = [r for r in rows if r["case"] == case]
        result = extract_tail_features([float(r["q_GeV"]) for r in selected],
                                       [float(r["J"]) for r in selected])
        assert result["status"] == "observed"
        np.testing.assert_allclose([result[k] for k in ("Q10_GeV", "B_mid", "K_tail")], values, rtol=1e-14)


def test_mean_pT_uniform_disk_exact_on_irregular_nodes_and_amplitude_invariant():
    q = np.array([0., .013, .08, .2])
    for intensity in (1e-100, 1., 7., 1e100):
        point = extract_tail_features(q, np.full_like(q, intensity))
        assert point["mean_pT_GeV"] == pytest.approx(2*q[-1]/3, rel=2e-15)
        # A disk has no 50/10/1% crossing on the saved domain. The finite-range
        # mean and its marginal uncertainty are nevertheless meaningful.
        assert point["status"] == "censored" and point["feature_valid"] == [False, False, False, True]
    samples = np.broadcast_to(np.ones(len(q)), (4, 1, 2, len(q))).copy()
    stats = summarize_samples(q, samples, bootstrap_draws=8)["channels"][0][0]
    assert stats["covariance"] is None
    assert stats["marginal_orientation_se"][:3] == [None]*3
    assert stats["marginal_orientation_se"][3] == pytest.approx(0., abs=1e-15)


def test_mean_pT_exact_linear_intensity_moments_on_saved_annulus():
    q = np.array([.02, .041, .11, .17])
    intercept, slope = 2., -3.
    a, b = q[0], q[-1]
    denominator = intercept*(b*b-a*a)/2+slope*(b**3-a**3)/3
    numerator = intercept*(b**3-a**3)/3+slope*(b**4-a**4)/4
    point = extract_tail_features(q, intercept+slope*q)
    assert point["mean_pT_GeV"] == pytest.approx(numerator/denominator, rel=2e-15)
    assert point["mean_pT_support_GeV"] == [a, b]
    scaled = extract_tail_features(1.7*q, 12*(intercept+slope*q))
    assert scaled["mean_pT_GeV"] == pytest.approx(1.7*point["mean_pT_GeV"], rel=2e-15)
