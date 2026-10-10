"""Broad-annulus spectral weights from the archived panel-d Cartesian density.

The primary estimator sums the saved density pixels in three common annuli.
Orientation uncertainty is evaluated from retained raw intensities and uses
ratios of orientation means, never averages of individually normalized ratios.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

CASE_ORDER = ("spherical", "beta2_mid", "beta2", "beta3_mid", "beta3", "skin_mid", "skin")
DEFAULT_EDGES = (.03, .08, .13, .17)


def validate_edges(edges):
    edges = np.asarray(edges, dtype=float)
    if (edges.shape != (4,) or not np.isfinite(edges).all() or edges[0] < 0
            or np.any(np.diff(edges) <= 0)):
        raise ValueError("exactly three contiguous bins with finite increasing nonnegative edges are required")
    return edges


def annular_yields(image, axis, edges=DEFAULT_EDGES):
    """Integrate one or many images, preserving their leading axes.

    Pixel centers define bin membership: [lower, upper), except that the final
    upper edge is included. No relative-density display threshold is applied.
    """
    image, axis, edges = np.asarray(image, float), np.asarray(axis, float), validate_edges(edges)
    if (axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
            or np.any(np.diff(axis) <= 0)
            or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=3e-5, atol=0)):
        raise ValueError("pixel sums require a uniform finite increasing axis")
    if (image.shape[-2:] != (len(axis), len(axis))
            or not np.isfinite(image).all() or np.any(image < 0)):
        raise ValueError("finite nonnegative Cartesian density must match axis")
    if edges[-1] > min(-axis[0], axis[-1]) + 1e-12:
        raise ValueError("the full selected momentum disk must be inside the Cartesian grid")
    x, y = np.meshgrid(axis, axis, indexing="ij")
    q2 = x*x + y*y
    masks = [(q2 >= lo*lo) & ((q2 <= hi*hi) if i == 2 else (q2 < hi*hi))
             for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]
    counts = np.asarray([m.sum() for m in masks])
    if np.any(counts == 0):
        raise ValueError("every annulus must contain pixels")
    area = float(axis[1]-axis[0])**2
    values = np.stack([image[..., m].sum(-1)*area for m in masks], axis=-1)
    if np.any(values <= 0):
        raise ValueError("every annulus must have positive yield")
    return {"yield": values, "pixel_counts": counts, "pixel_area": area}


def statistics_from_yields(orientation_yields):
    """Joint delta-method covariance of fractions and adjacent yield ratios.

    Rows are paired raw orientation samples; columns are L,M,H yields. The
    covariance describes the numerical uncertainty of the orientation mean.
    It is neither detector uncertainty nor an event-counting covariance.
    """
    values = np.asarray(orientation_yields, float)
    if (values.ndim != 2 or values.shape[1] != 3 or len(values) < 1
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("orientation yields must be a finite nonnegative n-by-3 array")
    mean = values.mean(0)
    if np.any(mean <= 0):
        raise ValueError("mean yield must be positive in every annulus")
    total = mean.sum()
    fractions = mean/total
    ratios = mean[1:]/mean[:-1]
    influence_f = (values-values.sum(1)[:, None]*fractions)/total
    influence_r = (values[:, 1:]-values[:, :-1]*ratios)/mean[:-1]
    influence = np.column_stack((influence_f, influence_r))
    centered = influence-influence.mean(0)
    n = len(values)
    cov = centered.T@centered/(n*(n-1)) if n > 1 else np.zeros((5, 5))
    return {"mean_yield": mean, "fractions": fractions, "ratios": ratios,
            "fractions_orientation_covariance": cov[:3, :3],
            "ratios_orientation_covariance": cov[3:, 3:],
            "joint_orientation_covariance": cov,
            "fractions_orientation_se": np.sqrt(np.maximum(0., np.diag(cov[:3, :3]))),
            "ratios_orientation_se": np.sqrt(np.maximum(0., np.diag(cov[3:, 3:])))}


def _orientation_yields(result, edges):
    import torch
    from .decay import decay_axis_image

    coeff = np.asarray(result["orientation_coefficients"][:, -1, 0], dtype=float)
    n = len(coeff)
    if coeff.ndim != 3 or coeff.shape[1] != 3 or n != int(result["n_orientations"]):
        raise ValueError("retained orientation samples disagree with declared count")
    with torch.no_grad():
        images = decay_axis_image(torch.from_numpy(coeff),
            torch.from_numpy(np.asarray(result["radii"], float)),
            torch.from_numpy(np.asarray(result["axis"], float))).numpy()
    archived = np.asarray(result["images"][3, 0], float)
    closure = float(np.max(np.abs(images.mean(0)-archived))/np.max(np.abs(archived)))
    if closure > 2e-12:
        raise ValueError(f"orientation images do not close the archived mean: {closure:g}")
    yields = annular_yields(images, result["axis"], edges)["yield"]
    return yields, closure


def _central(image, axis, edges):
    result = annular_yields(image, axis, edges)
    values = result["yield"]
    return {**result, "window_yield": float(values.sum()),
            "fractions": values/values.sum(), "ratios": values[1:]/values[:-1]}


def _relative(entry, baseline, covariance=True):
    for key, singular in (("fractions", "fraction"), ("ratios", "ratio")):
        denominator = baseline[key]
        entry[f"{singular}_relative_change"] = entry[key]/denominator-1.
        if covariance:
            # The baseline here is deterministic (one spherical orientation).
            cov = entry[f"{key}_orientation_covariance"]/np.outer(denominator, denominator)
            entry[f"{singular}_relative_change_orientation_covariance"] = cov
            entry[f"{singular}_relative_change_orientation_se"] = np.sqrt(np.maximum(0., np.diag(cov)))


def analyze_spectral_weights(results, edges=DEFAULT_EDGES, n_radii=341, n_angles=720,
                             boundary_variants=True):
    """Analyze all seven original b12 panel-d cases in three frozen annuli.

    Continuous spectra are bilinear circle samples of the same archived pixels,
    multiplied by 2*pi*q and divided by the selected-window pixel integral.
    Their integral mismatch measures discretization and is not renormalized away.
    """
    edges = validate_edges(edges)
    if (not isinstance(n_radii, (int, np.integer)) or n_radii < 2
            or not isinstance(n_angles, (int, np.integer)) or n_angles < 4):
        raise ValueError("n_radii >= 2 and n_angles >= 4 must be integers")
    sequence = list(results.values()) if isinstance(results, dict) else list(results)
    rows = {r["case"]["name"]: r for r in sequence}
    if len(rows) != len(sequence) or set(rows) != set(CASE_ORDER):
        raise ValueError("all seven unique original panel-d cases are required")
    axis = np.asarray(rows["spherical"]["axis"], float)
    for name, row in rows.items():
        if not np.array_equal(np.asarray(row["axis"], float), axis):
            raise ValueError("all Cartesian axes must match")
        expected_n = 32 if name.startswith("beta") else 1
        if int(row["n_orientations"]) != expected_n:
            raise ValueError(f"{name} requires original {expected_n} orientation samples")
        if row.get("stage_names", [None]*4)[3] != "decay_by":
            raise ValueError("the source must identify panel d as decay_by")
        if not np.allclose(row["b_values"], np.arange(12., 61., 2.), rtol=0, atol=1e-10):
            raise ValueError("original source b grid must be 12:2:60 fm")
    q = np.linspace(edges[0], edges[-1], n_radii)
    alpha = np.linspace(0., 2*np.pi, n_angles, endpoint=False)
    coords = np.stack((q[:, None]*np.cos(alpha), q[:, None]*np.sin(alpha)), axis=-1)
    output = {"case_order": list(CASE_ORDER), "edges_GeV": edges, "radii": q,
              "bin_labels": ["L", "M", "H"], "ratio_labels": ["M/L", "H/M"],
              "cases": {}, "boundary_variants": [],
              "config": {"stage": "decay_by", "branch": "full", "image_index": [3, 0],
                  "central_estimate": "sum original normalized_images[3,0] pixels in common annuli",
                  "bin_membership": "[lower,upper); final upper edge included",
                  "normalization": "fractions within the selected common window; ratios of raw orientation-mean yields",
                  "display_mask_applied": False, "n_radii": n_radii, "n_angles": n_angles,
                  "spectrum": "2*pi*q*complete angular mean of bilinear sampled Cartesian image / selected-window pixel integral",
                  "spectrum_uncertainty": "central estimate only",
                  "uncertainty_scope": "1SE from paired retained orientation samples; no event, detector, b-grid or model uncertainty",
                  "reference_uncertainty": "deterministic spherical baseline has zero orientation SE",
                  "boundary_choice": "exploratory common broad bins, not an optimized or validated selection"}}
    for name in CASE_ORDER:
        row = rows[name]
        image = np.asarray(row["normalized_images"][3, 0], float)
        central = _central(image, axis, edges)
        raw_yields, closure = _orientation_yields(row, edges)
        uncertainty = statistics_from_yields(raw_yields)
        if any(not np.allclose(central[key], uncertainty[key], rtol=1e-10, atol=2e-13)
               for key in ("fractions", "ratios")):
            raise ValueError("raw orientation mean observables do not close direct panel-d pixels")
        polar = RegularGridInterpolator((axis, axis), image, method="linear",
            bounds_error=False, fill_value=np.nan)(coords)
        if not np.isfinite(polar).all() or np.any(polar < 0):
            raise ValueError("complete finite nonnegative circles are required")
        spectrum = 2*np.pi*q*polar.mean(1)/central["window_yield"]
        integral = float(np.trapz(spectrum, q))
        output["cases"][name] = {"case": dict(row["case"]),
            "n_orientations": int(row["n_orientations"]), **central,
            **{key: value for key, value in uncertainty.items() if key not in ("mean_yield", "fractions", "ratios")},
            "orientation_yields_raw": raw_yields, "spectrum": spectrum,
            "angular_mean_density": polar.mean(1)/central["window_yield"],
            "diagnostics": {"orientation_image_relative_closure": closure,
                "curve_integral": integral, "curve_integral_minus_one": integral-1.,
                "fractions_sum_minus_one": float(central["fractions"].sum()-1.),
                "fraction_covariance_sum_max_absolute": float(np.max(np.abs(uncertainty["fractions_orientation_covariance"].sum(0)))),
                "raw_mean_fraction_max_closure": float(np.max(np.abs(central["fractions"]-uncertainty["fractions"]))),
                "raw_mean_ratio_max_closure": float(np.max(np.abs(central["ratios"]-uncertainty["ratios"])))}}
    baseline = output["cases"]["spherical"]
    for entry in output["cases"].values():
        _relative(entry, baseline)
        # Cancel the common 2*pi*q analytically, including the q=0 limit.
        entry["spectrum_ratio_to_spherical"] = entry["angular_mean_density"]/baseline["angular_mean_density"]
    if boundary_variants:
        for index, delta, label in [(1, -.005, "lower_split_minus_5MeV"), (1, .005, "lower_split_plus_5MeV"),
                                   (2, -.005, "upper_split_minus_5MeV"), (2, .005, "upper_split_plus_5MeV"),
                                   (0, -.01, "qmin_minus_10MeV"), (0, .01, "qmin_plus_10MeV")]:
            shifted = edges.copy()
            shifted[index] += delta
            if shifted[0] < 0 or np.any(np.diff(shifted) <= 0):
                continue
            variant = {"label": label, "edges_GeV": shifted, "cases": {}}
            for name in CASE_ORDER:
                primary = _central(rows[name]["normalized_images"][3, 0], axis, shifted)
                variant["cases"][name] = {key: primary[key] for key in ("fractions", "ratios")}
            for entry in variant["cases"].values():
                _relative(entry, variant["cases"]["spherical"], covariance=False)
            output["boundary_variants"].append(variant)
    output["diagnostics"] = {
        "max_orientation_image_relative_closure": max(e["diagnostics"]["orientation_image_relative_closure"] for e in output["cases"].values()),
        "max_abs_curve_integral_minus_one": max(abs(e["diagnostics"]["curve_integral_minus_one"]) for e in output["cases"].values()),
        "max_abs_fraction_covariance_column_sum": max(e["diagnostics"]["fraction_covariance_sum_max_absolute"] for e in output["cases"].values()),
        "boundary_variants_are_not_optimization": True,
        "interpretation": "Existing template response, not continuous parameter extraction or a detector sensitivity forecast"}
    return output
