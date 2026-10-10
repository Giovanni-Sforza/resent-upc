"""Finite-q angular observables with paired, uncalibrated track-response toys.

The numerical sample is importance quadrature, not an experimental event
sample. MC, finite-orientation, and conditional counting covariances therefore
have separate names. All ratios retain per-orientation yields. Both q and the
lepton reference k are reconstructed from the same tracks by observable_response.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .angular_observables import orientation_angular_statistics
from .observable_response import (
    ReconstructedPairs, apply_track_response, make_common_random,
    sample_decay_pairs,
)

METRIC_NAMES = ("C2", "S2", "Aphi", "mean_q")
DEFAULT_BINS = ((.03, .06), (.06, .10), (.10, .14), (.14, .17))


def _se(covariance: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0., np.diag(covariance)))


def _covariance_of_mean(influence: np.ndarray) -> np.ndarray:
    n = len(influence)
    if n == 1:
        return np.zeros((influence.shape[1], influence.shape[1]))
    centered = influence - influence.mean(axis=0)
    return centered.T @ centered / (n * (n - 1))


@dataclass
class AngularSummary:
    """Compact event influence representation; never serialize event arrays."""

    values: np.ndarray
    bins: np.ndarray
    bin_index: np.ndarray
    features: np.ndarray
    weights: np.ndarray
    denominator: np.ndarray
    selected_weight: float
    event_covariance_per_bin_event: np.ndarray
    event_covariance_per_total_selected_event: np.ndarray
    mc_covariance: np.ndarray
    n_generated: int
    n_selected: int
    seed: int
    qmax: float
    truth_qmax: float
    migration: dict[str, Any]

    def influence_chunk(self, start: int, stop: int) -> np.ndarray:
        """One bin-major influence block, keeping peak storage bounded."""
        index = self.bin_index[start:stop]
        influence = np.zeros((stop - start, len(self.bins), len(METRIC_NAMES)))
        for ibin in range(len(self.bins)):
            keep = index == ibin
            influence[keep, ibin] = (
                self.n_generated * self.weights[start:stop][keep, None]
                * (self.features[start:stop][keep] - self.values[ibin])
                / self.denominator[ibin])
        return influence.reshape(stop - start, -1)


def summarize_angular_bins(reco: ReconstructedPairs, bins=DEFAULT_BINS,
                           qmax: float = .17) -> AngularSummary:
    """Weighted angular ratios and the two distinct event-count conventions.

    Bins are disjoint half-open intervals [low,high). Aphi uses -sign(cos2alpha),
    i.e. perpendicular-minus-parallel complementary quadrants. The full event
    covariance is conditional on N_total selected q<qmax signal events; a bin's
    4x4 covariance is conditional on N_bin events in that bin. Their diagonal
    blocks differ by the bin yield fraction. No detector efficiency is applied.
    """
    bins = np.asarray(bins, dtype=float)
    if (bins.ndim != 2 or bins.shape[1] != 2 or len(bins) == 0
            or not np.all(np.isfinite(bins)) or np.any(bins[:, 0] < 0)
            or np.any(bins[:, 1] <= bins[:, 0]) or np.any(bins[:, 1] > qmax)
            or np.any(bins[1:, 0] < bins[:-1, 1])):
        raise ValueError("Require ordered, disjoint nonempty bins inside [0,qmax]")
    if not 0 < qmax <= reco.truth_qmax:
        raise ValueError("qmax must lie inside positive truth support")
    if (reco.fractional_sigma or reco.angular_sigma) and qmax >= reco.truth_qmax:
        raise ValueError("Response requires truth support beyond qmax")
    q, alpha, weights = reco.q, reco.alpha, reco.importance_weights
    if (len(q) < 2 or np.any(~np.isfinite(weights)) or np.any(weights < 0)
            or np.any(~np.isfinite(q)) or np.any(~np.isfinite(alpha))):
        raise ValueError("Need finite reconstructed arrays and nonnegative weights")
    c2, s2 = np.cos(2 * alpha), np.sin(2 * alpha)
    features = np.column_stack((c2, s2, -np.sign(c2), q))
    index = np.full(len(q), -1, dtype=np.int16)
    for ibin, (low, high) in enumerate(bins):
        index[(q >= low) & (q < high)] = ibin
    selected = q < qmax
    selected_weight = float(weights[selected].sum())
    denominator = np.array([weights[index == i].sum() for i in range(len(bins))])
    if selected_weight <= 0 or np.any(denominator <= 0):
        raise ValueError("No weighted events in at least one selected bin")
    values = np.array([np.average(features[index == i], weights=weights[index == i], axis=0)
                       for i in range(len(bins))])
    n, dimensions = len(q), 4 * len(bins)
    event_bin = np.zeros((len(bins), 4, 4))
    event_total, mc = np.zeros((dimensions, dimensions)), np.zeros((dimensions, dimensions))
    for i in range(len(bins)):
        keep = index == i
        residual = features[keep] - values[i]
        event_bin[i] = residual.T @ (weights[keep, None] * residual) / denominator[i]
        section = slice(4 * i, 4 * (i + 1))
        event_total[section, section] = event_bin[i] * selected_weight / denominator[i]
        influence = n * weights[keep, None] * residual / denominator[i]
        mc[section, section] = influence.T @ influence / (n * (n - 1))
    truth_selected = reco.q_truth < qmax
    truth_weight = float(weights[truth_selected].sum())
    migration = {
        "truth_above_qmax_fraction_of_reco_selected": float(
            weights[selected & ~truth_selected].sum() / selected_weight),
        "reco_above_qmax_fraction_of_truth_selected": float(
            weights[~selected & truth_selected].sum() / truth_weight),
        "fraction_of_truth_weight_in_last_5_percent_of_support": float(
            weights[reco.q_truth >= .95 * reco.truth_qmax].sum() / weights.sum()),
        "per_bin": [],
    }
    for i, (low, high) in enumerate(bins):
        truth_in = (reco.q_truth >= low) & (reco.q_truth < high)
        reco_in = index == i
        tw = float(weights[truth_in].sum())
        migration["per_bin"].append({
            "fraction_entered_from_outside_truth_bin": float(
                weights[reco_in & ~truth_in].sum() / denominator[i]),
            "fraction_left_truth_bin": (float(weights[truth_in & ~reco_in].sum() / tw)
                                         if tw > 0 else None),
        })
    return AngularSummary(values, bins, index, features, weights, denominator,
                          selected_weight, event_bin, event_total, mc, n,
                          int(selected.sum()), reco.seed, float(qmax),
                          float(reco.truth_qmax), migration)


def paired_mc_covariance(summary: AngularSummary, baseline: AngularSummary,
                         chunk_size: int = 25_000) -> np.ndarray:
    """Full cross-bin common-random covariance of a model/response difference."""
    return linear_mc_covariance([(summary, 1.), (baseline, -1.)], chunk_size)


def linear_mc_covariance(terms: list[tuple[AngularSummary, float]],
                         chunk_size: int = 25_000) -> np.ndarray:
    """Covariance of a linear combination of coupled ratio influences.

    Four terms are required for the case-minus-reference response shift used
    by the exact-truth control variate. Identical objects cancel before floating
    arithmetic, giving exactly zero MC covariance for the zero-response anchor.
    """
    if not terms:
        raise ValueError("Need at least one influence term")
    summary = terms[0][0]
    grouped: dict[int, tuple[AngularSummary, float]] = {}
    for other, coefficient in terms:
        if (summary.seed != other.seed or summary.n_generated != other.n_generated
                or summary.qmax != other.qmax or not np.array_equal(summary.bins, other.bins)):
            raise ValueError("Paired response requires matching random draws and bins")
        if not np.isfinite(coefficient):
            raise ValueError("Influence coefficients must be finite")
        previous = grouped.get(id(other), (other, 0.))[1]
        grouped[id(other)] = (other, previous + coefficient)
    active = [(other, coefficient) for other, coefficient in grouped.values() if coefficient != 0]
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    dimensions, n = summary.values.size, summary.n_generated
    product, total = np.zeros((dimensions, dimensions)), np.zeros(dimensions)
    if not active:
        return product
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        difference = np.zeros((stop-start, dimensions))
        for other, coefficient in active:
            difference += coefficient * other.influence_chunk(start, stop)
        product += difference.T @ difference
        total += difference.sum(0)
    return (product - np.outer(total, total) / n) / (n * (n - 1))


def orientation_reweight(result: dict[str, Any], pairs: Any,
                         summaries: list[AngularSummary]) -> dict[str, Any]:
    """Push raw orientation densities through the identical reconstructed pairs.

    Each orientation is reweighted from the ensemble radial proposal; weighted
    numerators and denominators are averaged before division. This preserves
    orientation-dependent yield. The small output influence has only
    N_orientation x N_response x (N_bin*4) entries.
    """
    q, alpha, u = pairs.q_truth, pairs.alpha_truth, pairs.polar_cosine
    radii = np.asarray(result["radii"])
    proposal_a0 = np.interp(q, radii, result["coefficients"][-1, 0, 0])
    cos2, sin2, factor = np.cos(2 * alpha), np.sin(2 * alpha), (1 - u*u) / (1 + u*u)
    orientations = np.asarray(result["orientation_coefficients"])[:, -1, 0]
    nb, nr = len(summaries[0].bins), len(summaries)
    numerator = np.zeros((len(orientations), nr, nb, 4))
    denominator = np.zeros((len(orientations), nr, nb))
    for io, orientation in enumerate(orientations):
        a, c, d = (np.interp(q, radii, row) for row in orientation)
        weights = np.divide(a - factor * (c*cos2 + d*sin2), proposal_a0,
                            out=np.zeros_like(a), where=proposal_a0 > 0)
        if np.min(weights) < -1e-9:
            raise ValueError("Negative orientation decay density")
        weights = np.maximum(weights, 0.)
        for ir, summary in enumerate(summaries):
            # Sentinel nb absorbs events outside all measured bins, so the
            # interpolation weights are never expanded into an event x bin array.
            index = np.where(summary.bin_index >= 0, summary.bin_index, nb)
            denominator[io, ir] = np.bincount(index, weights=weights, minlength=nb+1)[:nb]
            for im in range(4):
                numerator[io, ir, :, im] = np.bincount(
                    index, weights=weights * summary.features[:, im], minlength=nb+1)[:nb]
    mean_denominator = denominator.mean(0)
    values = numerator.mean(0) / mean_denominator[..., None]
    influence = ((numerator - denominator[..., None] * values[None])
                 / mean_denominator[None, ..., None]).reshape(len(orientations), nr, -1)
    covariance = np.array([_covariance_of_mean(influence[:, ir]) for ir in range(nr)])
    closure = values - np.asarray([s.values for s in summaries])
    return {"values": values, "influence": influence, "covariance": covariance,
            "closure": closure, "n_orientations": len(orientations)}


def _orientation_contrast_covariance(current: dict, reference: dict,
                                     response_index: int, reference_index: int,
                                     same_bank: bool = False) -> tuple[np.ndarray, str]:
    a = current["influence"][:, response_index]
    b = reference["influence"][:, reference_index]
    if len(a) == 1 and len(b) == 1:
        return np.zeros((a.shape[1], a.shape[1])), "orientation-independent cases"
    if len(b) == 1:
        return _covariance_of_mean(a), "reference orientation-independent"
    if len(a) == 1:
        return _covariance_of_mean(b), "case orientation-independent"
    if same_bank and len(a) == len(b):
        return _covariance_of_mean(a-b), "paired identical orientation banks"
    # A conservative independent-bank sum would be misleading for overlapping
    # samples. Fail unless the caller supplies true pairing or exact reference.
    raise ValueError("Nontrivial orientation contrasts require identical banks")


def _contrast(summary: AngularSummary, reference: AngularSummary,
              orientation_covariance: np.ndarray, pairing: str,
              include_event_counts: bool = False) -> dict[str, Any]:
    mc = paired_mc_covariance(summary, reference)
    covariance = mc + orientation_covariance
    delta = (summary.values - reference.values).ravel()
    error = _se(covariance)
    resolved = np.abs(delta) > np.maximum(3 * error, 1e-12)
    # One signal sample compared to an exactly known reference prediction;
    # no background, acceptance, luminosity or systematic uncertainty included.
    count_variance = np.diag(summary.event_covariance_per_total_selected_event)
    n3 = [float(9 * v / d**2) if include_event_counts and ok and d != 0 else None
          for v, d, ok in zip(count_variance, delta, resolved)]
    return {"delta": delta.tolist(), "mc_covariance": mc.tolist(),
            "mc_se": _se(mc).tolist(),
            "orientation_covariance": orientation_covariance.tolist(),
            "orientation_se": _se(orientation_covariance).tolist(),
            "combined_sampling_covariance": covariance.tolist(),
            "combined_sampling_se": error.tolist(),
            "resolved_above_3_sampling_se": resolved.tolist(),
            "orientation_pairing": pairing,
            "resolution_absolute_floor": 1e-12,
            "event_count_diagnostics_enabled": include_event_counts,
            "n3_total_selected_one_sample_diagnostic": n3,
            "n3_scope": "Conditional pure-signal count versus an exact reference; disabled by default; null when disabled or difference does not exceed 3 combined numerical sampling SE and absolute roundoff floor 1e-12; not an experimental sensitivity forecast"}


def _same_rotation_banks(result: dict, baseline: dict) -> bool:
    return all(k in result and k in baseline and np.array_equal(result[k], baseline[k])
               for k in ("rotation_bank_1", "rotation_bank_2"))


def _uncertainty_fields(mc: np.ndarray, orientation: np.ndarray) -> dict[str, Any]:
    combined = mc + orientation
    return {"mc_covariance": mc.tolist(), "mc_se": _se(mc).tolist(),
            "orientation_covariance": orientation.tolist(),
            "orientation_se": _se(orientation).tolist(),
            "combined_sampling_covariance": combined.tolist(),
            "combined_sampling_se": _se(combined).tolist()}


def _anchored_contrast(values, reference_values, mc, orientation, pairing):
    delta = np.asarray(values).ravel() - np.asarray(reference_values).ravel()
    result = {"delta": delta.tolist(), **_uncertainty_fields(mc, orientation)}
    resolved = abs(delta) > np.maximum(3*_se(mc+orientation), 1e-12)
    result.update(resolved_above_3_sampling_se=resolved.tolist(),
                  orientation_pairing=pairing, resolution_absolute_floor=1e-12,
                  event_count_diagnostics_enabled=False,
                  n3_total_selected_one_sample_diagnostic=[None]*len(delta),
                  n3_scope="Control variate changes numerical quadrature only; physical event moments remain in the raw response row")
    return result


def _truth_anchor_information(result, summaries, orientation, bins, qmax):
    exact = orientation_angular_statistics(
        np.asarray(result["orientation_coefficients"])[:, -1, 0],
        result["radii"], bins=bins, qmax=qmax,
        deterministic=len(result["orientation_coefficients"]) == 1)
    exact_values = np.asarray(exact["values"])
    exact_influence = np.asarray(exact["_influence"])
    if exact_influence.shape != (orientation["n_orientations"], len(bins), 4):
        raise ValueError("Unexpected exact-truth orientation influence shape")
    exact_influence = exact_influence.reshape(orientation["n_orientations"], -1)
    values = exact_values[None] + (
        np.array([s.values for s in summaries]) - summaries[0].values[None])
    influence = exact_influence[:, None] + (
        orientation["influence"] - orientation["influence"][:, :1])
    covariance = np.array([_covariance_of_mean(influence[:, i]) for i in range(len(summaries))])
    return {"values": values, "influence": influence, "covariance": covariance,
            "exact_values": exact_values, "exact_covariance": exact["covariance"]}


def run_angular_response(results: list[dict], study: dict,
                         log: Callable = print) -> dict[str, Any]:
    """Evaluate all cases and response scenarios using common random numbers.

    ``study`` may be the study subsection or the complete configuration.
    Existing response_events/response_seed/fractional_resolutions keys work.
    ``include_angular_toy`` defaults true and adds sigma_p=.005, sigma_phi=.001.
    No mutation of input results and no forward-model calculation is performed.
    """
    if "study" in study:
        config = study
        study = config["study"]
        mass = config.get("physics", {}).get("physical_constants", {}).get("Mjpsi", 3.0969)
    else:
        mass = study.get("mass_gev", 3.0969)
    bins = study.get("angular_bins_gev", DEFAULT_BINS)
    qmax = float(study.get("qmax", .17))
    n_events = int(study.get("response_events", 1_000_000))
    seed = int(study.get("response_seed", 20261008))
    truth_anchor = bool(study.get("truth_anchored_response", False))
    common = make_common_random(n_events, seed)
    scenarios = [(0., 0.)]
    for sigma in study.get("fractional_resolutions", (0., .005, .01)):
        if (float(sigma), 0.) not in scenarios:
            scenarios.append((float(sigma), 0.))
    if study.get("include_angular_toy", True):
        extra = (float(study.get("angular_toy_fractional_sigma", .005)),
                 float(study.get("angular_toy_sigma_rad", .001)))
        if extra not in scenarios:
            scenarios.append(extra)
    scenario_metadata = [{"id": f"p{p:g}_a{a:g}", "fractional_sigma": p,
                          "angular_sigma": a} for p, a in scenarios]
    baseline = next((r for r in results if r["case"]["name"] == "spherical"), None)
    if baseline is None:
        raise ValueError("Need a spherical reference case")
    output: dict[str, Any] = {
        "definitions": {
            "q": "p_plus + p_minus", "k": "(p_plus - p_minus)/2",
            "alpha": "phi_q - phi_k reconstructed jointly from the same tracks",
            "C2": "<cos(2 alpha)> within reconstructed q bin",
            "S2": "<sin(2 alpha)> within reconstructed q bin; null diagnostic",
            "Aphi": "<-sign(cos(2 alpha))> = (N_perpendicular-N_parallel)/(N_perpendicular+N_parallel), complementary quadrants",
            "mean_q": "<q> within reconstructed q bin, GeV",
            "flatten_order": "bin-major, then C2,S2,Aphi,mean_q",
            "combined_sampling_covariance": "sum of MC quadrature and finite-orientation first-order covariance; no detector/systematic uncertainty or mixed second-order term",
            "event_covariance_per_total_selected_event": "divide by N_total of all selected q<qmax signal events",
            "event_covariance_per_bin_event": "divide each 4x4 block by N_bin in that bin; not N_total",
            "Aphi_harmonic_residual": "Aphi + 4/pi*C2; zero for pure second harmonic, but response can generate higher harmonics; finite-MC errors retained",
            "response_status": "uncalibrated Gaussian independent track errors, no detector acceptance or background",
            "truth_support": "finite coefficient-grid support, including migrations in and out; events beyond support absent; cutoff convergence not demonstrated",
            "orientation_reweight": "raw per-orientation density, same reconstructed pairs and response draws, yield-preserving ratio of ensemble sums",
            "truth_anchored": "Optional exact-truth control variate: exact O_truth + MC O_response - MC O_zero. Four-term paired influences for structural contrasts. Zero-response MC variance is exactly zero. Numerical integration improvement, not detector calibration; finite-ratio MC bias and mixed second-order uncertainty are not eliminated",
        },
        "metric_names": list(METRIC_NAMES), "bin_edges_gev": np.asarray(bins).tolist(),
        "qmax_gev": qmax, "n_events": n_events, "seed": seed,
        "responses": scenario_metadata,
        "event_count_diagnostics_enabled": bool(study.get("include_event_count_diagnostics", False)),
        "truth_anchored_response_enabled": truth_anchor,
        "cases": [],
    }
    baseline_summaries: list[AngularSummary] = []
    baseline_orientation: dict[str, Any] = {}
    baseline_anchor: dict[str, Any] = {}
    for result in [baseline] + [r for r in results if r is not baseline]:
        name = result["case"]["name"]
        pairs = sample_decay_pairs(result["coefficients"][-1, 0], result["radii"], common, mass)
        summaries = [summarize_angular_bins(
            apply_track_response(pairs, common, fractional_sigma=p, angular_sigma=a), bins, qmax)
            for p, a in scenarios]
        orientation = orientation_reweight(result, pairs, summaries)
        anchor = (_truth_anchor_information(result, summaries, orientation, bins, qmax)
                  if truth_anchor else None)
        if result is baseline:
            baseline_summaries, baseline_orientation = summaries, orientation
            baseline_anchor = anchor
        case = {"name": name, "n_orientations": orientation["n_orientations"], "rows": []}
        for ir, summary in enumerate(summaries):
            ori_cov = orientation["covariance"][ir]
            combined = summary.mc_covariance + ori_cov
            delta_ori, pairing = _orientation_contrast_covariance(
                orientation, baseline_orientation, ir, ir, _same_rotation_banks(result, baseline))
            response_ori, response_pairing = _orientation_contrast_covariance(
                orientation, orientation, ir, 0, True)
            row = {**scenario_metadata[ir], "response_id": scenario_metadata[ir]["id"],
                   "values_flat": summary.values.ravel().tolist(),
                   "mc_covariance": summary.mc_covariance.tolist(),
                   "orientation_covariance": ori_cov.tolist(),
                   "combined_sampling_covariance": combined.tolist(),
                   "event_covariance_per_total_selected_event": summary.event_covariance_per_total_selected_event.tolist(),
                   "selected_weight_fraction": float(summary.selected_weight / summary.weights.sum()),
                   "n_generated": n_events, "n_selected_quadrature_draws": summary.n_selected,
                   "truth_qmax_gev": summary.truth_qmax, "migration": summary.migration,
                   "orientation_reweight_closure": orientation["closure"][ir].ravel().tolist(),
                   "vs_spherical": _contrast(summary, baseline_summaries[ir], delta_ori, pairing, bool(study.get("include_event_count_diagnostics", False))),
                   "vs_zero_response": _contrast(summary, summaries[0], response_ori, response_pairing, bool(study.get("include_event_count_diagnostics", False))),
                   "bins": []}
            for ib, limits in enumerate(summary.bins):
                section = slice(4*ib, 4*(ib+1))
                contrast_vector = np.array([4/np.pi, 0., 1., 0.])
                residual = float(summary.values[ib] @ contrast_vector)
                residual_mc_se = float(np.sqrt(max(0., contrast_vector @ summary.mc_covariance[section, section] @ contrast_vector)))
                residual_ori_se = float(np.sqrt(max(0., contrast_vector @ ori_cov[section, section] @ contrast_vector)))
                row["bins"].append({
                    "q_range_gev": limits.tolist(), "values": summary.values[ib].tolist(),
                    "bin_fraction": float(summary.denominator[ib] / summary.selected_weight),
                    "mc_se": _se(summary.mc_covariance)[section].tolist(),
                    "orientation_se": _se(ori_cov)[section].tolist(),
                    "combined_sampling_se": _se(combined)[section].tolist(),
                    "event_covariance_per_bin_event": summary.event_covariance_per_bin_event[ib].tolist(),
                    "Aphi_harmonic_residual": residual,
                    "Aphi_harmonic_residual_mc_se": residual_mc_se,
                    "Aphi_harmonic_residual_orientation_se": residual_ori_se,
                })
            if truth_anchor:
                anchor_mc = np.asarray(row["vs_zero_response"]["mc_covariance"])
                anchor_ori = anchor["covariance"][ir]
                anchor_delta_ori, anchor_pairing = _orientation_contrast_covariance(
                    anchor, baseline_anchor, ir, ir, _same_rotation_banks(result, baseline))
                anchor_delta_mc = linear_mc_covariance([
                    (summary, 1.), (summaries[0], -1.),
                    (baseline_summaries[ir], -1.), (baseline_summaries[0], 1.)])
                anchored = {
                    "values_flat": anchor["values"][ir].ravel().tolist(),
                    "exact_truth_values_flat": anchor["exact_values"].ravel().tolist(),
                    "exact_truth_orientation_covariance": np.asarray(anchor["exact_covariance"]).tolist(),
                    **_uncertainty_fields(anchor_mc, anchor_ori),
                    "vs_spherical": _anchored_contrast(
                        anchor["values"][ir], baseline_anchor["values"][ir],
                        anchor_delta_mc, anchor_delta_ori, anchor_pairing),
                    "vs_zero_response": _anchored_contrast(
                        anchor["values"][ir], anchor["values"][0], anchor_mc,
                        np.asarray(row["vs_zero_response"]["orientation_covariance"]), response_pairing),
                    "event_covariance_status": "Use raw response row for physical event fluctuations; this branch improves numerical quadrature only",
                    "bins": [],
                }
                for ib, limits in enumerate(summary.bins):
                    section = slice(4*ib, 4*(ib+1))
                    value = anchor["values"][ir, ib]
                    h = np.array([4/np.pi, 0., 1., 0.])
                    anchored["bins"].append({
                        "q_range_gev": limits.tolist(), "values": value.tolist(),
                        "mc_se": _se(anchor_mc)[section].tolist(),
                        "orientation_se": _se(anchor_ori)[section].tolist(),
                        "combined_sampling_se": _se(anchor_mc+anchor_ori)[section].tolist(),
                        "Aphi_harmonic_residual": float(value @ h),
                        "Aphi_harmonic_residual_mc_se": float(np.sqrt(max(0., h @ anchor_mc[section, section] @ h))),
                        "Aphi_harmonic_residual_orientation_se": float(np.sqrt(max(0., h @ anchor_ori[section, section] @ h))),
                    })
                row["truth_anchored"] = anchored
            case["rows"].append(row)
        output["cases"].append(case)
        log(f"Angular response complete: {name} ({n_events} common draws, {orientation['n_orientations']} orientations)")
    return output
