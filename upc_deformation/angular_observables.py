"""Lepton-axis angular observables from existing unnormalized radial caches.

Only NumPy is required.  The input coefficients obey
I(q, alpha) = A0(q) - (C(q) cos(2 alpha) + D(q) sin(2 alpha))/2.
All ratios are formed after averaging intensities.  Reported errors describe
orientation integration, not event statistics or detector performance.
"""
from __future__ import annotations

from math import comb

import numpy as np


METRIC_NAMES = ["C2", "S2", "Aphi", "mean_q"]
BIN_EDGES_GEV = np.array([[.03, .06], [.06, .10], [.10, .14], [.14, .17]])
STAGE_NAMES = ["decay_fixed", "decay_b", "decay_by"]
CURVE_RADII_GEV = np.linspace(.005, .17, 331)
RELATIVE_YIELD_FLOOR = 1e-12


def _radial_inputs(values, radii):
    r, v = np.asarray(radii, dtype=float), np.asarray(values, dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.all(np.isfinite(r)) or np.any(np.diff(r) <= 0):
        raise ValueError("radii must be finite, strictly increasing, and have at least two nodes")
    if r[0] < 0 or v.shape[-1:] != r.shape or not np.all(np.isfinite(v)):
        raise ValueError("values must be finite and share the nonnegative radial axis")
    return v, r


def radial_moment(values, radii, lower, upper, power=1):
    """Exactly integrate q**power times the piecewise-linear input values.

    Interpolation is applied to the coefficient, not to an already multiplied
    q**power integrand.  The analytic segment formula works on nonuniform grids
    and includes the exact requested endpoints.
    """
    v, r = _radial_inputs(values, radii)
    if not isinstance(power, (int, np.integer)) or power < 0:
        raise ValueError("power must be a nonnegative integer")
    if not (np.isfinite(lower) and np.isfinite(upper) and r[0] <= lower < upper <= r[-1]):
        raise ValueError("integration bounds must be ordered and inside radial support")
    grid = np.r_[lower, r[(r > lower) & (r < upper)], upper]
    left, width = grid[:-1], np.diff(grid)
    index = np.clip(np.searchsorted(r, left, side="right") - 1, 0, len(r) - 2)
    slope = (v[..., index + 1] - v[..., index]) / (r[index + 1] - r[index])
    value_left = v[..., index] + slope * (left - r[index])
    w0, w1 = np.zeros_like(left), np.zeros_like(left)
    for j in range(power + 1):
        common = comb(power, j) * left ** (power - j)
        w0 += common * width ** (j + 1) / (j + 1)
        w1 += common * width ** (j + 2) / (j + 2)
    return np.sum(value_left * w0 + slope * w1, axis=-1)


def coefficient_bin_moments(coefficients, radii, bins=BIN_EDGES_GEV):
    """Return unnormalized bin yields and the four metric numerators."""
    c, r = _radial_inputs(coefficients, radii)
    if c.shape[-2] != 3:
        raise ValueError("coefficient component axis must be [A0,C,D]")
    bins = np.asarray(bins, dtype=float)
    if bins.ndim != 2 or bins.shape[1] != 2 or len(bins) == 0:
        raise ValueError("bins must be a nonempty array of [lower,upper] pairs")
    moments = np.stack([radial_moment(c, r, lo, hi, 1) for lo, hi in bins], axis=-2)
    den = 2 * np.pi * moments[..., 0]
    cnum = -.5 * np.pi * moments[..., 1]
    snum = -.5 * np.pi * moments[..., 2]
    qnum = 2 * np.pi * np.stack([radial_moment(c[..., 0, :], r, lo, hi, 2)
                                for lo, hi in bins], axis=-1)
    return den, np.stack((cnum, snum, -4 / np.pi * cnum, qnum), axis=-1)


def _covariance(influence, deterministic=False):
    x = np.asarray(influence)
    n = len(x)
    flat = x.reshape(n, -1)
    if n == 1:
        result = np.zeros((flat.shape[1], flat.shape[1])) if deterministic else np.full((flat.shape[1], flat.shape[1]), np.nan)
        invalid = ~np.isfinite(flat).all(0)
        result[invalid, :] = np.nan
        result[:, invalid] = np.nan
        return result
    centered = flat - flat.mean(0)
    return centered.T @ centered / (n * (n - 1))


def _standard_error(influence, deterministic=False):
    x = np.asarray(influence)
    if len(x) == 1:
        return np.where(np.isfinite(x[0]), 0., np.nan) if deterministic else np.full(x.shape[1:], np.nan)
    return np.std(x, axis=0, ddof=1) / np.sqrt(len(x))


def orientation_angular_statistics(coefficients, radii, bins=BIN_EDGES_GEV,
                                   qmax=.17, deterministic=False):
    """Delta-method covariance of intensity-averaged ratios across all bins.

    The private influence arrays are retained for paired source contrasts.
    Flattening order is bin-major, then METRIC_NAMES.  Zero-yield bins have
    undefined metrics (NaN internally, null in analyze_angular's JSON output).
    """
    c, r = _radial_inputs(coefficients, radii)
    if c.ndim != 3 or c.shape[1] != 3 or len(c) == 0:
        raise ValueError("expected (orientation,3,radius) coefficients")
    den, num = coefficient_bin_moments(c, r, bins)
    disk = 2 * np.pi * radial_moment(c[:, 0], r, 0., qmax, 1)
    if disk.mean() <= 0 or np.any(disk < 0):
        raise ValueError("nonpositive disk yield")
    mean_den = den.mean(0)
    if np.any(mean_den < -RELATIVE_YIELD_FLOOR * disk.mean()):
        raise ValueError("negative bin yield")
    valid = mean_den > RELATIVE_YIELD_FLOOR * disk.mean()
    safe = np.where(valid, mean_den, np.nan)
    values = num.mean(0) / safe[:, None]
    influence = (num - den[..., None] * values) / safe[:, None]
    fraction = mean_den / disk.mean()
    fraction_influence = (den - disk[:, None] * fraction) / disk.mean()
    covariance = _covariance(influence, deterministic)
    return {"values": values, "se": _standard_error(influence, deterministic),
            "covariance": covariance, "yield": mean_den,
            "yield_se": _standard_error(den, deterministic),
            "yield_fraction": fraction,
            "yield_fraction_se": _standard_error(fraction_influence, deterministic),
            "yield_fraction_covariance": _covariance(fraction_influence, deterministic),
            "disk_yield": disk.mean(), "disk_yield_se": float(_standard_error(disk, deterministic)),
            "valid": valid, "n_orientations": len(c),
            "_influence": influence, "_yield_fraction_influence": fraction_influence}


def _curve_statistics(coefficients, radii, query, deterministic):
    c = np.asarray(coefficients)
    sampled = np.array([np.interp(query, radii, v) for v in c.reshape(-1, c.shape[-1])])
    sampled = sampled.reshape(*c.shape[:-1], len(query))
    den, num = sampled[:, 0], -.25 * sampled[:, 1:3]
    average = den.mean(0)
    peak = float(np.max(c[:, 0].mean(0)))
    valid = average > RELATIVE_YIELD_FLOOR * peak
    safe = np.where(valid, average, np.nan)
    value = num.mean(0) / safe
    influence = (num - den[:, None] * value) / safe
    se = _standard_error(influence, deterministic)
    return {"C2": value[0], "S2": value[1], "C2_se": se[0], "S2_se": se[1],
            "valid": valid, "_influence": influence}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _known_b_metrics(result, disk, dark_mask):
    axis = np.asarray(result["axis"])
    # The production axis is a float32 linspace: adjacent subtractions differ
    # at its storage precision even though the underlying mesh is uniform.
    tolerance = 4*np.finfo(axis.dtype).eps*float(np.max(np.abs(axis))) if np.issubdtype(axis.dtype, np.floating) else 0.
    if len(axis) < 2 or not np.allclose(np.diff(axis), axis[1] - axis[0], rtol=1e-6, atol=tolerance):
        raise ValueError("known-b image requires a uniform transverse axis")
    pixel_area = float(axis[1] - axis[0]) ** 2
    metrics = []
    for image in np.asarray(result["images"])[0]:
        total = image[disk].sum() * pixel_area
        if total <= 0:
            raise ValueError("nonpositive known-b image yield")
        norm = image / total
        area = np.count_nonzero(norm[disk] > .5 * norm[disk].max()) * pixel_area
        metrics.append({"dark_fixed_mean": float(norm[dark_mask].mean()),
                        "sqrt_area_halfmax": float(np.sqrt(area))})
    return metrics


def _independent_halves(orientations, radii, qmax):
    """Compare two independently generated 32-orientation endpoint banks."""
    groups = []
    for subset in (orientations[:32, -1], orientations[32:, -1]):
        group = {b: orientation_angular_statistics(subset[:, i], radii, qmax=qmax)
                 for i, b in enumerate(("full", "self"))}
        paired = group["full"]["_influence"]-group["self"]["_influence"]
        group["interaction"] = {
            "values": group["full"]["values"]-group["self"]["values"],
            "se": _standard_error(paired), "covariance": _covariance(paired)}
        groups.append(group)
    first, second = groups
    difference = {}
    for key in ("full", "self", "interaction"):
        difference[key] = {
            "values": second[key]["values"]-first[key]["values"],
            "se": np.sqrt(second[key]["se"]**2+first[key]["se"]**2),
            "covariance": second[key]["covariance"]+first[key]["covariance"]}
    return {"stage": "decay_by", "first_count": 32, "second_count": 32,
            "first": first, "second": second, "difference": difference,
            "interpretation": "Second minus first. The driver supplies two independent orientation banks, so half covariances add. Full/self errors within each half are paired. Agreement is a stability diagnostic, not a convergence certificate; independence is a generation assumption, not inferred from the coefficients."}


def analyze_angular(results, study):
    """Analyze cached physical intensities, without modifying inputs or files.

    ``results`` is the loaded npz+json list used by the existing observable
    runner; ``study`` is its study configuration dictionary.  The spherical
    reference must be deterministic.  Different cases may have different
    orientation counts.  Only within-case full/self samples are paired.
    """
    if not results:
        raise ValueError("at least a deterministic spherical result is required")
    cfg = study.get("study", study)
    qmax = float(cfg.get("qmax", .17))
    if not np.isclose(qmax, .17, rtol=0, atol=1e-12):
        raise ValueError("this registered angular analysis uses qmax=.17 GeV")
    configured_bins = np.asarray(cfg.get("angular_bins_gev", BIN_EDGES_GEV), dtype=float)
    if configured_bins.shape != BIN_EDGES_GEV.shape or not np.allclose(configured_bins, BIN_EDGES_GEV, rtol=0, atol=1e-12):
        raise ValueError("angular_bins_gev must equal the four registered angular bins")
    curve_points = cfg.get("curve_points", len(CURVE_RADII_GEV))
    curve_qmin = float(cfg.get("curve_qmin", CURVE_RADII_GEV[0]))
    if isinstance(curve_points, bool) or not isinstance(curve_points, (int, np.integer)) or curve_points < 2:
        raise ValueError("curve_points must be an integer of at least two")
    if not np.isfinite(curve_qmin) or not 0 < curve_qmin < qmax:
        raise ValueError("curve_qmin must be positive and smaller than qmax")
    curve_radii = np.linspace(curve_qmin, qmax, curve_points)
    names = [result["case"]["name"] for result in results]
    if len(set(names)) != len(names) or "spherical" not in names:
        raise ValueError("unique case names and a spherical reference are required")
    base = results[names.index("spherical")]
    if (len(base["orientation_coefficients"]) != 1 or
            base["case"].get("beta2", 0) or base["case"].get("beta3", 0)):
        raise ValueError("spherical reference must have one deterministic orientation")
    # Match the original double-precision disk normalization even when the
    # cached mesh coordinates themselves were stored in float32.
    axis = np.asarray(base["axis"], dtype=np.float64)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    disk = x*x + y*y <= qmax*qmax
    base_image = np.asarray(base["images"])[0, 0]
    dark_mask = disk & (base_image <= np.quantile(base_image[disk], .05))
    base_bridge = _known_b_metrics(base, disk, dark_mask)
    output = {"schema_version": "angular_observables_v1", "metric_names": METRIC_NAMES,
              "bin_edges_GeV": BIN_EDGES_GEV, "curve_radii_GeV": curve_radii,
              "stage_names": STAGE_NAMES, "qmax_GeV": qmax,
              "covariance_order": "bin-major, metric-minor; joint covariance full block then self block",
              "cases": {}, "interpretation": {
                  "definition": "q=pT_plus+pT_minus; k=(pT_plus-pT_minus)/2; alpha=arg(q)-arg(k)",
                  "angular_kernel": "I=A0-(C*cos(2alpha)+D*sin(2alpha))/2; C2=-C/(4A0); S2=-D/(4A0)",
                  "Aphi": "(N_perpendicular-N_parallel)/(N_perpendicular+N_parallel) in four complementary quadrants; Aphi=-4*C2/pi; symmetric sectors cancel D",
                  "integration": "Exact moments of piecewise-linear coefficients, with q dq measure and intensity averaging before normalization",
                  "uncertainty": "Orientation Monte Carlo delta-method SE/covariance only; not event statistics, detector accuracy, or full numerical convergence",
                  "source_contrast": "Self retains polarized coherent single-source emission. Full-self metric contrast is not an additive decomposition or a pure interference observable",
                  "rank": "The ideal Born transverse-polarization decay analyzer retains ranks zero and two; C2 is not collective flow and beta3 does not directly map to v3",
                  "finite_bins": "A common radial envelope cancels only at exact q; finite bins retain radial reweighting",
                  "background": "Mean empirical isotropic incoherent template; illustrative dilution, not a calibrated background prediction",
                  "coarse_b": "Same cached nodes re-integrated on a coarser b grid; numerical shift, not a comprehensive systematic uncertainty",
                  "yield_floor": RELATIVE_YIELD_FLOOR}}
    prepared = {}
    for result in results:
        name = result["case"]["name"]
        r = np.asarray(result["radii"])
        orientations = np.asarray(result["orientation_coefficients"])
        if orientations.ndim != 5 or orientations.shape[1:4] != (3, 2, 3):
            raise ValueError(f"{name}: expected orientation,stage,source,component,radius")
        if "n_orientations" in result and result["n_orientations"] != len(orientations):
            raise ValueError(f"{name}: declared orientation count does not match retained samples")
        if not np.array_equal(result["axis"], axis):
            raise ValueError("known-b morphology bridge requires identical image axes")
        mean = orientations.mean(0)
        if not np.allclose(mean, result["coefficients"], rtol=1e-7, atol=max(np.max(abs(mean)), 1e-300)*1e-12):
            raise ValueError(f"{name}: coefficient means do not match retained orientation samples")
        deterministic = not (result["case"].get("beta2", 0) or result["case"].get("beta3", 0))
        entry = {"case": dict(result["case"]), "n_orientations": len(orientations), "stages": {}}
        for stage, stage_name in enumerate(STAGE_NAMES):
            branches = {}
            for branch, branch_name in enumerate(("full", "self")):
                c = orientations[:, stage, branch]
                stats = orientation_angular_statistics(c, r, qmax=qmax, deterministic=deterministic)
                stats["curves"] = _curve_statistics(c, r, curve_radii, deterministic)
                branches[branch_name] = stats
            joint = np.concatenate([branches[b]["_influence"].reshape(len(orientations), -1)
                                    for b in ("full", "self")], axis=-1)
            branches["joint_covariance"] = _covariance(joint, deterministic)
            entry["stages"][stage_name] = branches
        entry["orientation_prefix"] = {}
        for n in sorted(set([k for k in (8, 16, 32, 64) if k <= len(orientations)] + [len(orientations)])):
            entry["orientation_prefix"][str(n)] = {
                b: orientation_angular_statistics(orientations[:n, -1, i], r, qmax=qmax, deterministic=deterministic)
                for i, b in enumerate(("full", "self"))}
        if name in ("beta2", "beta3") and len(orientations) == 64:
            keys = ("rotation_bank_1", "rotation_bank_2")
            if all(key in result for key in keys) and all(
                    np.array_equal(np.asarray(result[key])[:32], np.asarray(result[key])[32:]) for key in keys):
                raise ValueError("independent-half comparison cannot reuse the identical two rotation banks")
            entry["independent_halves"] = _independent_halves(orientations, r, qmax)
        coarse = {}
        for branch, b in enumerate(("full", "self")):
            stats = orientation_angular_statistics(np.asarray(result["coarse_b_coefficients"])[None, branch], r,
                                                  qmax=qmax, deterministic=True)
            coarse[b] = {"values": stats["values"], "yield_fraction": stats["yield_fraction"],
                         "shift": stats["values"] - entry["stages"]["decay_by"][b]["values"]}
        entry["coarse_b"] = coarse
        bg = np.asarray(result["background_coefficients"])
        if bg.shape != mean[-1, 0].shape or not np.all(np.isfinite(bg)) or np.any(bg[0] < 0) or not np.allclose(bg[1:], 0, rtol=0, atol=0):
            raise ValueError("background must be a nonnegative, explicitly isotropic [A0,0,0] template")
        bg_yield, _ = coefficient_bin_moments(bg, r)
        total = orientation_angular_statistics((mean[-1, 0] + bg)[None], r, qmax=qmax, deterministic=True)
        signal = entry["stages"]["decay_by"]["full"]
        total_yield = total["yield"]
        entry["background"] = {"full_values": signal["values"], "total_values": total["values"],
                               "signal_yield": signal["yield"], "background_yield": bg_yield,
                               "fraction_of_total": np.divide(bg_yield, total_yield, out=np.full_like(bg_yield, np.nan), where=total_yield>0),
                               "background_to_signal": np.divide(bg_yield, signal["yield"], out=np.full_like(bg_yield, np.nan), where=signal["yield"]>0),
                               "angular_dilution": np.divide(signal["yield"], total_yield, out=np.full_like(bg_yield, np.nan), where=total_yield>0),
                               "total_yield_fraction": total["yield_fraction"], "orientation_se": None,
                               "uncertainty_note": "Per-orientation background samples are not retained; total-template uncertainty is not estimated"}
        bridge = {"known_b_x_uncertainty": None, "mask": "Spherical full known-b lowest 5 percent intensity pixels inside q<=.17, frozen across all cases and both source branches", "dark_mask_pixels": int(dark_mask.sum())}
        known_metrics = _known_b_metrics(result, disk, dark_mask)
        for i, b in enumerate(("full", "self")):
            metrics = known_metrics[i]
            whole = orientation_angular_statistics(orientations[:, -1, i], r, [[0., qmax]], qmax, deterministic)
            metrics.update(delta_dark_fixed_mean=metrics["dark_fixed_mean"]-base_bridge[i]["dark_fixed_mean"],
                           delta_sqrt_area_halfmax=metrics["sqrt_area_halfmax"]-base_bridge[i]["sqrt_area_halfmax"],
                           mean_q=whole["values"][0, 3], mean_q_se=whole["se"][0, 3])
            bridge[b] = metrics
        entry["bridge"] = bridge
        prepared[name] = (result, entry, deterministic)
        output["cases"][name] = entry
    reference = output["cases"]["spherical"]
    for name, (_, entry, deterministic) in prepared.items():
        for stage in STAGE_NAMES:
            current, baseline = entry["stages"][stage], reference["stages"][stage]
            df = current["full"]["values"] - baseline["full"]["values"]
            ds = current["self"]["values"] - baseline["self"]["values"]
            paired = current["full"]["_influence"] - current["self"]["_influence"]
            contrast = {"delta_full": df, "delta_self": ds, "interaction": df-ds,
                        "interaction_se": _standard_error(paired, deterministic),
                        "interaction_covariance": _covariance(paired, deterministic), "curves": {}}
            cf, cs = current["full"]["curves"], current["self"]["curves"]
            curve_se = _standard_error(cf["_influence"]-cs["_influence"], deterministic)
            for i, metric in enumerate(("C2", "S2")):
                dcf = cf[metric]-baseline["full"]["curves"][metric]
                dcs = cs[metric]-baseline["self"]["curves"][metric]
                contrast["curves"].update({f"delta_full_{metric}": dcf, f"delta_self_{metric}": dcs,
                                          f"interaction_{metric}": dcf-dcs, f"interaction_{metric}_se": curve_se[i]})
            current["contrast"] = contrast
        coarse, coarse_base = entry["coarse_b"], reference["coarse_b"]
        for b in ("full", "self"):
            coarse[b]["structural_delta"] = coarse[b]["values"] - coarse_base[b]["values"]
            coarse[b]["structural_delta_shift"] = coarse[b]["shift"] - coarse_base[b]["shift"]
            entry["bridge"][b]["delta_mean_q"] = entry["bridge"][b]["mean_q"] - reference["bridge"][b]["mean_q"]
        coarse["structural_interaction_shift"] = coarse["full"]["structural_delta_shift"]-coarse["self"]["structural_delta_shift"]
    return _json_safe(output)
