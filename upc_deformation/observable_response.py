"""Paired toy track-response screening for reduced J/psi decay distributions.

Input coefficients satisfy ``I(q, alpha) = A0 - (C cos(2 alpha) +
D sin(2 alpha))/2`` after integration over the lepton polar angle.  Here
``alpha = phi_q - phi_k`` and ``k = (p_plus - p_minus)/2``.  Sampling restores
the polar dependence with a positive importance weight proportional to
``(1 + u**2) A0 - (1 - u**2) (C cos(2 alpha) + D sin(2 alpha))``.

Kinematics use the same leading-order, massless, small-qT/M approximation as
the analytic analyzer: kT=M*sqrt(1-u**2)/2 and p_plus/minus=+/-k+q/2.
There is no finite-boost spin rotation or rapidity/eta acceptance.  Coefficients
may already include a specified rapidity average.  Fractional momentum and
azimuth errors below are explicitly toy assumptions, not detector calibration.
Both measured q and its lepton reference axis are reconstructed from the same
smeared pair, so their correlated errors and selection migration are retained.

The sample is weighted numerical quadrature, not a simulated experimental
count.  ``mc_covariance`` measures its numerical integration error;
``event_covariance / N`` is the conditional physical sampling covariance for
N selected signal events under the stated toy model.  Do not conflate them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


OBSERVABLE_NAMES = ("F_dip", "mean_q", "C2")


@dataclass(frozen=True)
class CommonRandom:
    """Reusable random numbers coupling models and response assumptions."""

    q_uniform: np.ndarray
    polar_uniform: np.ndarray
    alpha_uniform: np.ndarray
    lab_uniform: np.ndarray
    momentum_normal: np.ndarray
    azimuth_normal: np.ndarray
    seed: int

    @property
    def n_events(self) -> int:
        return len(self.q_uniform)


@dataclass(frozen=True)
class DecayPairs:
    p_plus: np.ndarray
    p_minus: np.ndarray
    q_truth: np.ndarray
    alpha_truth: np.ndarray
    polar_cosine: np.ndarray
    importance_weights: np.ndarray
    truth_qmax: float
    seed: int


@dataclass(frozen=True)
class ReconstructedPairs:
    q: np.ndarray
    alpha: np.ndarray
    importance_weights: np.ndarray
    q_truth: np.ndarray
    truth_qmax: float
    fractional_sigma: float
    angular_sigma: float
    seed: int


@dataclass(frozen=True)
class ObservableSummary:
    names: tuple[str, ...]
    values: np.ndarray
    event_covariance: np.ndarray
    mc_covariance: np.ndarray
    mc_influence: np.ndarray
    n_generated: int
    n_selected: int
    selected_fraction: float
    c2_selected_fraction: float
    q0: float
    qmax: float
    c2_window: tuple[float, float]
    truth_qmax: float
    seed: int

    @property
    def mc_se(self) -> np.ndarray:
        return np.sqrt(np.maximum(0.0, np.diag(self.mc_covariance)))

    def conditional_covariance(self, n_selected_events: int) -> np.ndarray:
        """Physical statistical covariance conditional on N accepted events."""
        if n_selected_events <= 0:
            raise ValueError("n_selected_events must be positive")
        return self.event_covariance / n_selected_events

    def conditional_se(self, n_selected_events: int) -> np.ndarray:
        return np.sqrt(np.maximum(0.0, np.diag(
            self.conditional_covariance(n_selected_events))))

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe results; event-sized paired influences are omitted."""
        return {
            "names": list(self.names),
            "values": self.values.tolist(),
            "event_covariance_per_selected_event": self.event_covariance.tolist(),
            "mc_covariance": self.mc_covariance.tolist(),
            "mc_se": self.mc_se.tolist(),
            "n_generated": self.n_generated,
            "n_selected": self.n_selected,
            "selected_fraction": self.selected_fraction,
            "c2_selected_fraction": self.c2_selected_fraction,
            "q0_gev": self.q0,
            "qmax_gev": self.qmax,
            "c2_window_gev": list(self.c2_window),
            "truth_qmax_gev": self.truth_qmax,
            "seed": self.seed,
            "kinematics": "massless leading-order transverse small-qT/M; no eta acceptance",
            "response_status": "assumed toy errors, not detector-calibrated",
            "truth_support": "finite cutoff; migration from beyond truth_qmax is absent",
        }


def make_common_random(n_events: int = 500_000, seed: int = 17123) -> CommonRandom:
    if n_events < 2:
        raise ValueError("At least two quadrature events are required")
    rng = np.random.default_rng(seed)
    return CommonRandom(
        rng.random(n_events), rng.random(n_events), rng.random(n_events),
        rng.random(n_events), rng.standard_normal((n_events, 2)),
        rng.standard_normal((n_events, 2)), int(seed))


def _numpy(values: Any) -> np.ndarray:
    # Accept ordinary arrays as well as CPU/CUDA torch outputs without importing
    # torch into this lightweight numerical module.
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _validate_coefficients(coefficients: Any, radii: Any) -> tuple[np.ndarray, np.ndarray]:
    coeff, r = _numpy(coefficients), _numpy(radii)
    if r.ndim != 1 or len(r) < 2 or coeff.shape != (3, len(r)):
        raise ValueError("Expected radii (nr,) and coefficients (3,nr)")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(coeff)):
        raise ValueError("Coefficients and radii must be finite")
    if r[0] != 0 or np.any(np.diff(r) <= 0):
        raise ValueError("radii must start at zero and increase strictly")
    scale = float(np.max(np.abs(coeff[0])))
    if scale <= 0 or np.min(coeff[0]) < -1e-12 * scale:
        raise ValueError("A0 must be nonnegative and have positive yield")
    # The stronger Stokes bound is required for the unintegrated polar density.
    # Positivity of the polar-integrated density alone would allow |C,D|<=2A0.
    if np.any(np.hypot(coeff[1], coeff[2]) > coeff[0] + 2e-7 * scale):
        raise ValueError("Unintegrated decay requires sqrt(C^2+D^2) <= A0")
    coeff = coeff.copy()
    coeff[0] = np.maximum(coeff[0], 0)
    polarization = np.hypot(coeff[1], coeff[2])
    correction = np.minimum(1.0, np.divide(coeff[0], polarization,
                              out=np.ones_like(polarization), where=polarization > 0))
    coeff[1:] *= correction
    return coeff, r


def _sample_radius(a0: np.ndarray, r: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    """Invert the exact integral of q times piecewise-linear A0(q)."""
    width = np.diff(r)
    slope = np.diff(a0) / width
    a, b, c = r[:-1] * a0[:-1], (a0[:-1] + r[:-1] * slope) / 2, slope / 3
    segment_mass = width * (a + width * (b + width * c))
    segment_mass = np.maximum(segment_mass, 0)
    cdf = np.concatenate(([0.0], np.cumsum(segment_mass)))
    if cdf[-1] <= 0:
        raise ValueError("Zero radial yield")
    targets = uniform * cdf[-1]
    index = np.searchsorted(cdf, targets, side="right") - 1
    index = np.minimum(index, len(width) - 1)
    targets -= cdf[index]
    lo, hi = np.zeros_like(targets), width[index].copy()
    ai, bi, ci = a[index], b[index], c[index]
    for _ in range(28):
        mid = (lo + hi) / 2
        integral = mid * (ai + mid * (bi + mid * ci))
        below = integral < targets
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return r[index] + (lo + hi) / 2


def sample_decay_pairs(coefficients: Any, radii: Any,
                       common_random: CommonRandom, mass_gev: float = 3.0969) -> DecayPairs:
    """Generate a common-random weighted sample of transverse decay pairs.

    The proposal draws q from q*A0, u from 3(1+u²)/8, and alpha uniformly.
    Its analytic importance weight restores the joint polar/azimuthal decay
    density.  No resampling is done: retain the weights for every observable.
    """
    if not np.isfinite(mass_gev) or mass_gev <= 0:
        raise ValueError("mass_gev must be positive and finite")
    coeff, r = _validate_coefficients(coefficients, radii)
    q = _sample_radius(coeff[0], r, common_random.q_uniform)
    a0, c, d = (np.interp(q, r, row) for row in coeff)
    u = 2 * np.sinh(np.arcsinh(4 * common_random.polar_uniform - 2) / 3)
    alpha = 2 * np.pi * common_random.alpha_uniform
    anisotropy = np.divide(c * np.cos(2 * alpha) + d * np.sin(2 * alpha),
                          a0, out=np.zeros_like(q), where=a0 > 0)
    weights = np.maximum(0.0, 1 - (1 - u * u) / (1 + u * u) * anisotropy)
    phi_q = 2 * np.pi * common_random.lab_uniform
    phi_k = phi_q - alpha
    q_vector = q[:, None] * np.column_stack((np.cos(phi_q), np.sin(phi_q)))
    kt = .5 * mass_gev * np.sqrt(np.maximum(0.0, 1 - u * u))
    k_vector = kt[:, None] * np.column_stack((np.cos(phi_k), np.sin(phi_k)))
    return DecayPairs(k_vector + q_vector / 2, -k_vector + q_vector / 2,
                      q, alpha, u, weights, float(r[-1]), common_random.seed)


def apply_track_response(pairs: DecayPairs, common_random: CommonRandom,
                         fractional_sigma: float = 0.0,
                         angular_sigma: float = 0.0) -> ReconstructedPairs:
    """Apply independent pair-track errors and reconstruct q and k jointly.

    Each transverse track magnitude is multiplied by 1+sigma*z; azimuths
    receive independent Gaussian errors in radians.  Negative scale draws
    are rejected rather than silently reversing a track.  Defaults of 0.005
    or 0.01 in a caller represent hypotheses, not published resolutions.
    """
    if (not np.isfinite(fractional_sigma) or not np.isfinite(angular_sigma)
            or fractional_sigma < 0 or angular_sigma < 0):
        raise ValueError("Response sigmas must be finite and nonnegative")
    if len(pairs.q_truth) != common_random.n_events or pairs.seed != common_random.seed:
        raise ValueError("Use the same common random sample for decay and response")
    tracks = []
    for i, track in enumerate((pairs.p_plus, pairs.p_minus)):
        scale = 1 + fractional_sigma * common_random.momentum_normal[:, i]
        if np.any(scale <= 0):
            raise ValueError("Gaussian multiplicative response produced a nonpositive momentum")
        angle = angular_sigma * common_random.azimuth_normal[:, i]
        cosine, sine = np.cos(angle), np.sin(angle)
        tracks.append(scale[:, None] * np.column_stack((
            track[:, 0] * cosine - track[:, 1] * sine,
            track[:, 0] * sine + track[:, 1] * cosine)))
    qvec, kvec = tracks[0] + tracks[1], (tracks[0] - tracks[1]) / 2
    q = np.linalg.norm(qvec, axis=1)
    alpha = np.arctan2(qvec[:, 1], qvec[:, 0]) - np.arctan2(kvec[:, 1], kvec[:, 0])
    return ReconstructedPairs(q, alpha, pairs.importance_weights, pairs.q_truth,
                              pairs.truth_qmax, float(fractional_sigma),
                              float(angular_sigma), pairs.seed)


def summarize_observables(reco: ReconstructedPairs, q0: float = .02,
                          qmax: float = .17,
                          c2_window: tuple[float, float] | None = None) -> ObservableSummary:
    """Summarize dark-region fill, diffraction scale, and angular diagnostic.

    F_dip=N(q<q0)/N(q<qmax), mean_q=<q> within q<qmax, and C2=<cos2alpha>
    within c2_window (the full selected range by default).  The event covariance
    accounts for a narrower C2 window while N always counts all q<qmax events.
    Nonzero smearing requires truth support strictly beyond the selected range;
    cutoff convergence must still be established by the caller.
    """
    if not 0 < q0 < qmax <= reco.truth_qmax:
        raise ValueError("Require 0 < q0 < qmax <= truth_qmax")
    if (reco.fractional_sigma > 0 or reco.angular_sigma > 0) and qmax >= reco.truth_qmax:
        raise ValueError("Nonzero response requires truth support beyond reconstructed qmax")
    window = (0.0, float(qmax)) if c2_window is None else tuple(map(float, c2_window))
    if len(window) != 2 or not 0 <= window[0] < window[1] <= qmax:
        raise ValueError("c2_window must be a nonempty subset of [0,qmax]")
    q, alpha, weights = reco.q, reco.alpha, reco.importance_weights
    selected = q < qmax
    w = weights * selected
    selected_weight = w.sum()
    if selected_weight <= 0:
        raise ValueError("No selected events")
    in_window = (q >= window[0]) & (q < window[1])
    denominator = np.column_stack((np.ones_like(q), np.ones_like(q), in_window))
    numerators = np.column_stack((q < q0, q, in_window * np.cos(2 * alpha)))
    weighted_denominator = (w[:, None] * denominator).sum(0)
    if np.any(weighted_denominator <= 0):
        raise ValueError("No selected events in an observable window")
    values = (w[:, None] * numerators).sum(0) / weighted_denominator
    residuals = numerators - values * denominator
    conditional_denominator = weighted_denominator / selected_weight
    event_influence = residuals / conditional_denominator
    event_covariance = (event_influence.T @ (w[:, None] * event_influence)) / selected_weight
    n = len(q)
    mc_influence = n * w[:, None] * residuals / weighted_denominator
    mc_covariance = (mc_influence.T @ mc_influence) / (n * (n - 1))
    return ObservableSummary(OBSERVABLE_NAMES, values, event_covariance, mc_covariance,
                             mc_influence, n, int(selected.sum()),
                             float(selected_weight / weights.sum()),
                             float(weighted_denominator[2] / selected_weight),
                             float(q0), float(qmax), window, float(reco.truth_qmax), reco.seed)


def paired_contrast(summary: ObservableSummary, baseline: ObservableSummary) -> dict[str, Any]:
    """Model/response difference with common-random numerical error, not data error."""
    if (summary.seed != baseline.seed or summary.n_generated != baseline.n_generated
            or summary.names != baseline.names or summary.q0 != baseline.q0
            or summary.qmax != baseline.qmax or summary.c2_window != baseline.c2_window):
        raise ValueError("Paired contrast requires matching common random sample and definitions")
    influence = summary.mc_influence - baseline.mc_influence
    n = summary.n_generated
    covariance = (influence.T @ influence) / (n * (n - 1))
    return {"names": list(summary.names), "delta": (summary.values - baseline.values).tolist(),
            "mc_covariance": covariance.tolist(),
            "mc_se": np.sqrt(np.maximum(0.0, np.diag(covariance))).tolist(),
            "interpretation": "paired numerical integration uncertainty, not experimental precision"}
