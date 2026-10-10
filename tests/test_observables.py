"""Analytic measure and averaging tests for the mechanism bridge."""
import math

import numpy as np
from scipy.special import erf

from upc_deformation.observables import (
    coefficient_metrics, deformed_ws_rms, orientation_metric_se,
)


def test_uniform_transverse_density_has_radial_jacobian_and_spin_two_moment():
    r = np.linspace(0, 1, 10001)
    amplitude, qmax, q0 = 2.3, .8, .2
    coefficients = np.stack((np.full_like(r, amplitude),
                             np.full_like(r, .4 * amplitude), np.zeros_like(r)))
    got = coefficient_metrics(coefficients, r, qmax, q0)
    assert math.isclose(got["yield"], math.pi * amplitude * qmax**2, rel_tol=1e-12)
    assert math.isclose(got["F_dip"], (q0 / qmax)**2, rel_tol=1e-12)
    assert math.isclose(got["mean_q"], 2 * qmax / 3, rel_tol=2e-8)
    assert math.isclose(got["C2"], -.1, rel_tol=1e-12)
    assert math.isclose(got["event_variance"][1], qmax**2 / 18, rel_tol=1e-7)


def test_finite_gaussian_window_matches_analytic_probability_and_mean():
    r = np.linspace(0, .3, 20001)
    sigma, qmax, q0 = .05, .17, .02
    a0 = np.exp(-r**2 / (2 * sigma**2))
    got = coefficient_metrics(np.stack((a0, .3 * a0, np.zeros_like(r))), r, qmax, q0)
    tail = math.exp(-qmax**2 / (2 * sigma**2))
    probability = (1 - math.exp(-q0**2 / (2 * sigma**2))) / (1 - tail)
    mean = (sigma * math.sqrt(math.pi / 2) * erf(qmax / (math.sqrt(2) * sigma))
            - qmax * tail) / (1 - tail)
    assert math.isclose(got["F_dip"], probability, rel_tol=5e-7)
    assert math.isclose(got["mean_q"], mean, rel_tol=5e-8)
    assert math.isclose(got["C2"], -.075, rel_tol=1e-12)


def test_orientation_average_keeps_physical_yield_weights():
    r = np.linspace(0, 1, 20001)
    coefficients = np.zeros((2, 3, len(r)))
    coefficients[0, 0] = 1
    coefficients[1, 0] = 100 * r**2
    value, se, influence = orientation_metric_se(coefficients, r, 1, .5)
    # Uniform density has yield pi, while 100*q^2 has yield 50*pi.
    expected = (.25 + 50 * .5**4) / 51
    incorrect_mean_of_normalized_shapes = (.25 + .5**4) / 2
    assert math.isclose(value[0], expected, rel_tol=2e-8)
    assert abs(value[0] - incorrect_mean_of_normalized_shapes) > .08
    assert se[0] > 0
    np.testing.assert_allclose(influence.mean(0), 0, atol=1e-15)


def test_zero_deformation_ws_rms_matches_spherical_fermi_result():
    radius, diffuseness = 5.02, .46
    # The lower-limit correction to the usual Fermi moment formula is
    # exponentially small for R/a~11 and well below this test tolerance.
    expected = math.sqrt(3 * radius**2 / 5 + 7 * math.pi**2 * diffuseness**2 / 5)
    assert math.isclose(deformed_ws_rms(radius, diffuseness), expected, rel_tol=2e-6)
