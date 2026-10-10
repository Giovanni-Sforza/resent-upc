"""Region-based two-source diagnostics from existing observable caches.

Each stage defines its own spherical-reference dark mask.  Comparing these
masks is a stage-by-stage diagnostic, NOT transport of one original pixel
patch through the lepton-axis projection.  An outer region is descriptive:
large q alone identifies neither diffraction nor interference.

S = I_self and X = I_full - I_self use raw intensities with the SAME original
normalization.  These are theoretical two-emission-direction diagnostics;
X and the source-self sample are not experimentally selectable event classes.
Self retains coherent diffraction within either source.
"""
from __future__ import annotations

from typing import Any

import numpy as np


_DELTA_NAMES = ("self", "cross", "full")


def _regional_radial_weights(axis: np.ndarray, radii: np.ndarray,
                             mask: np.ndarray, pixel_area: float) -> np.ndarray:
    """Linear functional reproducing decay_axis_image followed by a mask sum.

    Interpolating coefficients is linear, so each pixel's interpolation
    weights can be accumulated on radial nodes once for all orientations.
    Coordinates are evaluated in float64 as in the stored decay images.
    """
    x, y = np.meshgrid(axis.astype(np.float64), axis.astype(np.float64), indexing="ij")
    q = np.hypot(x[mask], y[mask])
    angle = np.arctan2(y[mask], x[mask])
    if not len(q):
        return np.zeros((3, len(radii)))
    if q.min() < radii[0] or q.max() > radii[-1]:
        raise ValueError("Dark-region pixels exceed coefficient radial support")
    index = np.searchsorted(radii, q).clip(1, len(radii) - 1)
    fraction = (q - radii[index - 1]) / (radii[index] - radii[index - 1])
    angular = (np.ones_like(q), -.5 * np.cos(2 * angle), -.5 * np.sin(2 * angle))
    weights = []
    for component in angular:
        values = (np.bincount(index - 1, weights=component * (1 - fraction), minlength=len(radii))
                  + np.bincount(index, weights=component * fraction, minlength=len(radii)))
        weights.append(values * pixel_area)
    return np.array(weights)


def _null_errors() -> dict[str, Any]:
    return {"K_R": None, "A_R": None,
            "delta_N_self_over_Y0": None, "delta_N_cross_over_Y0": None,
            "delta_N_full_over_Y0": None,
            "delta_yield_covariance_order": list(_DELTA_NAMES),
            "delta_yield_covariance": None}


def _orientation_errors(per_orientation: np.ndarray, weights: np.ndarray,
                        raw_region_yields: np.ndarray, y0: float) -> tuple[dict, dict]:
    # orientation x (full,self) x Stokes x radial node.
    if per_orientation.ndim != 4 or per_orientation.shape[1:3] != (2, 3):
        raise ValueError("Expected per-orientation coefficients (n,2,3,nr)")
    raw_samples = np.einsum("obsr,sr->ob", per_orientation, weights)
    means = raw_samples.mean(0)
    scale = max(float(np.max(np.abs(raw_region_yields))), np.finfo(float).tiny)
    absolute_error = float(np.max(np.abs(means - raw_region_yields)))
    relative_error = absolute_error / scale
    if not np.allclose(means, raw_region_yields, rtol=2e-6, atol=2e-10 * scale):
        raise ValueError(f"Orientation regional means disagree with cached images: relative={relative_error:g}")
    n = len(raw_samples)
    full, self_terms = raw_samples.T
    cross = full - self_terms  # pair first, then calculate sampling uncertainty
    samples = np.column_stack((self_terms, cross, full)) / y0
    covariance = np.cov(samples, rowvar=False, ddof=1) / n if n > 1 else np.zeros((3, 3))
    errors = np.sqrt(np.maximum(0., np.diag(covariance)))
    if means[1] > 0:
        k = cross.mean() / self_terms.mean()
        influence_k = (cross - k * self_terms) / self_terms.mean()
        k_se = float(np.std(influence_k, ddof=1) / np.sqrt(n)) if n > 1 else 0.
    else:
        k_se = None
    error_record = {"K_R": k_se, "A_R": None,
                    **{f"delta_N_{name}_over_Y0": float(error)
                       for name, error in zip(_DELTA_NAMES, errors)},
                    "delta_yield_covariance_order": list(_DELTA_NAMES),
                    "delta_yield_covariance": covariance.tolist()}
    checks = {"orientation_mean_raw_N_full": float(means[0]),
              "orientation_mean_raw_N_self": float(means[1]),
              "orientation_mean_vs_image_max_absolute_error": absolute_error,
              "orientation_mean_vs_image_relative_error": relative_error,
              "orientation_mean_image_closure_passed": True}
    return error_record, checks


def dark_region_diagnostics(results: list[dict], config: dict) -> dict[str, Any]:
    """Return JSON-safe region mechanisms and paired orientation uncertainties.

    Each stage uses the lowest-5%-pixel threshold of that stage's spherical
    full image within q<=qmax.  Tied pixels are retained.  The same mask is
    frozen for all structural cases in the stage.  ``outer_dark`` is its
    subset q>0.9*qmax; this partition carries no quality or mechanism claim.

    Raw yield differences are divided by one shared stage-specific reference
    Y0 = integral_disk I_self,spherical, so delta self + delta cross = delta
    full exactly.  They are not differences of separately normalized shapes.
    The baseline is deterministic spherical (one orientation); reference
    sampling uncertainty is consequently zero.  A_R is the absolute value
    of the ENSEMBLE-AVERAGED cross image, not the average of |X| per event.
    Its nonsmooth absolute-value uncertainty is deliberately not estimated.
    """
    if not results:
        raise ValueError("No case results supplied")
    base = next((r for r in results if r["case"]["name"] == "spherical"), None)
    if base is None:
        raise ValueError("A spherical reference is required")
    if len(base["orientation_coefficients"]) != 1:
        raise ValueError("This diagnostic requires a deterministic one-orientation spherical reference")
    cfg = config["study"]
    qmax = float(cfg["qmax"])
    q0 = float(cfg["q0"])
    axis = np.asarray(base["axis"])
    if axis.ndim != 1 or len(axis) < 3 or not np.all(np.diff(axis) > 0):
        raise ValueError("Expected an increasing momentum axis")
    step = float(axis[1] - axis[0])
    if not np.allclose(np.diff(axis), step, rtol=1e-4, atol=1e-12):
        raise ValueError("Regional pixel integrals require a uniform image axis")
    pixel_area = step**2
    x, y = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(x, y)
    disk = radius <= qmax
    if not disk.any():
        raise ValueError("Empty analysis disk")
    stages = list(base["stage_names"])
    for case in results:
        if not np.array_equal(case["axis"], axis) or list(case["stage_names"]) != stages:
            raise ValueError("All cases must share the reference image axis and stage order")
        images = np.asarray(case["images"])
        if images.shape != (len(stages), 2, len(axis), len(axis)) or not np.isfinite(images).all():
            raise ValueError("Expected finite raw images (stage,full/self,qx,qy)")

    records, masks = [], []
    closure_values = []
    additive_errors = []
    for stage_index, stage in enumerate(stages):
        reference_full, reference_self = base["images"][stage_index]
        threshold = float(np.quantile(reference_full[disk], .05))
        all_dark = disk & (reference_full <= threshold)
        y0 = float(reference_self[disk].sum() * pixel_area)
        if y0 <= 0:
            raise ValueError("Nonpositive spherical self yield")
        for region_name, mask in (("all_dark", all_dark),
                                  ("outer_dark", all_dark & (radius > .9 * qmax))):
            count = int(mask.sum())
            radial_values = radius[mask]
            mask_record = {
                "stage": stage, "region": region_name,
                "mask_reference_case": "spherical", "mask_reference_branch": "full",
                "raw_dark_threshold": threshold, "pixel_count": count,
                "fraction_of_disk_pixels": float(count / disk.sum()),
                "q_quantile_probabilities": [0., .25, .5, .75, 1.],
                "q_quantiles_gev": np.quantile(radial_values, [0, .25, .5, .75, 1]).tolist() if count else None,
                "mask_pixel_fraction_below_q0": float(np.mean(radial_values < q0)) if count else None,
                "mask_pixel_fraction_above_0p9_qmax": float(np.mean(radial_values > .9 * qmax)) if count else None,
            }
            masks.append(mask_record)
            base_n = np.array([reference_full[mask].sum(), reference_self[mask].sum()]) * pixel_area
            radial_weights: dict[bytes, np.ndarray] = {}
            for result in results:
                name = result["case"]["name"]
                full, self_terms = result["images"][stage_index]
                cross = full - self_terms
                raw_n = np.array([full[mask].sum(), self_terms[mask].sum()]) * pixel_area
                n_full, n_self = map(float, raw_n)
                n_cross = float(cross[mask].sum() * pixel_area)
                k = n_cross / n_self if n_self > 0 else None
                activity = float(np.abs(cross[mask]).sum() * pixel_area / n_self) if n_self > 0 else None
                delta_self = float((raw_n[1] - base_n[1]) / y0)
                delta_cross = float((n_cross - (base_n[0] - base_n[1])) / y0)
                delta_full = float((raw_n[0] - base_n[0]) / y0)
                additive_error = delta_self + delta_cross - delta_full
                additive_errors.append(abs(additive_error))
                full_total = float(full[disk].sum() * pixel_area)
                self_total = float(self_terms[disk].sum() * pixel_area)
                row = {
                    "case": name, "stage": stage, "region": region_name,
                    "pixel_count": count, "status": "available" if count else "empty_region",
                    "n_orientations": int(len(result["orientation_coefficients"])),
                    "K_R": k, "A_R": activity,
                    "raw_N_full": n_full, "raw_N_self": n_self, "raw_N_cross": n_cross,
                    "region_full_yield_fraction": n_full / full_total if full_total > 0 else None,
                    "region_self_yield_fraction": n_self / self_total if self_total > 0 else None,
                    "common_Y0_spherical_self_disk": y0,
                    "delta_N_self_over_Y0": delta_self,
                    "delta_N_cross_over_Y0": delta_cross,
                    "delta_N_full_over_Y0": delta_full,
                    "additive_closure_error": float(additive_error),
                }
                if stage_index > 0 and count:
                    radii = np.asarray(result["radii"], dtype=np.float64)
                    key = radii.tobytes()
                    if key not in radial_weights:
                        radial_weights[key] = _regional_radial_weights(axis, radii, mask, pixel_area)
                    orientation = np.asarray(result["orientation_coefficients"][:, stage_index - 1], dtype=np.float64)
                    errors, checks = _orientation_errors(orientation, radial_weights[key], raw_n, y0)
                    row["orientation_se"] = errors
                    row["orientation_closure"] = checks
                    row["orientation_error_scope"] = "paired full/self orientation sampling, conditioned on deterministic spherical reference; not total numerical or experimental uncertainty"
                    closure_values.append(checks["orientation_mean_vs_image_relative_error"])
                else:
                    row["orientation_se"] = _null_errors()
                    row["orientation_closure"] = None
                    row["orientation_error_scope"] = "not available: per-orientation true-b images are absent" if stage_index == 0 else "not available: empty region"
                records.append(row)
    return {
        "definitions": {
            "mask": "Per-stage spherical full lowest 5% of pixels inside q<=qmax, threshold ties retained; frozen across cases within a stage",
            "outer_dark": "Descriptive subset of all_dark at q>0.9*qmax, not a preferred region or a mechanism classification",
            "X": "I_full - I_self using raw images with identical original normalization",
            "K_R": "integral_R X / integral_R I_self; signed net cross term",
            "A_R": "integral_R abs(X) / integral_R I_self after ensemble averaging; resolves spatial sign cancellation in K_R",
            "A_R_se": "Not estimated: absolute value is a nonsmooth functional, especially near zero cross term",
            "Y0": "Stage-specific spherical raw self yield over the analysis disk; same denominator for every structural case and branch difference",
            "delta_yields": "(N_case - N_spherical)/Y0 for self, cross, full; self+cross=full",
            "raw_yield_units": "Stage intensity-integral units, not counted experimental events; raw values across stages have different integration measures",
            "pixel_fractions": "Area/pixel counts, not fractions of event yield; yield fractions are explicitly named",
            "transport": "Masks are independently defined per stage; this is not transport of one original image patch",
            "mechanism": "High-q location and nonzero self response alone do not identify or rule out two-source interference",
            "experimental_scope": "Cross and source-self branches are theoretical diagnostics, not selectable event classes",
        },
        "qmax_gev": qmax, "q0_gev": q0, "outer_radius_fraction": .9,
        "dark_pixel_quantile": .05, "masks": masks, "records": records,
        "checks": {
            "all_orientation_mean_image_closures_passed": True,
            "max_orientation_mean_image_relative_error": max(closure_values, default=0.),
            "max_additive_closure_absolute_error": max(additive_errors, default=0.),
        },
    }
