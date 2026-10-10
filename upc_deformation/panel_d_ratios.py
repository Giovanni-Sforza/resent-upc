"""Radial/conditional-angular decomposition of the saved panel-d density.

The primitive is normalized_images[3, 0] on its original Cartesian (qx,qy)
mesh.  No coefficient-based distribution or new forward model is constructed.
Array order is (qx,qy) for Cartesian maps and (q,alpha) for polar maps.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


CASE_ORDER = ("beta2", "beta3", "skin")


def _divide(numerator, denominator, valid):
    shape = np.broadcast_shapes(np.shape(numerator), np.shape(denominator), np.shape(valid))
    return np.divide(numerator, denominator, out=np.full(shape, np.nan), where=valid)


def _maximum(values, mask):
    use = np.asarray(mask) & np.isfinite(values)
    return float(np.max(np.abs(np.asarray(values)[use]))) if np.any(use) else None


def _input(result):
    name = result["case"]["name"]
    if name not in ("spherical", *CASE_ORDER):
        raise ValueError("only original spherical, beta2, beta3 and skin cases are supported")
    axis = np.asarray(result["axis"])
    if axis.ndim != 1 or len(axis) < 2 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
        raise ValueError("Cartesian axis must be finite and strictly increasing")
    images = np.asarray(result["normalized_images"])
    if images.ndim != 4 or images.shape[:2] != (4, 2) or images.shape[2:] != (len(axis), len(axis)):
        raise ValueError("expected normalized_images[stage,source,qx,qy] with four stages and two sources")
    image = np.asarray(images[3, 0], dtype=float)
    if not np.all(np.isfinite(image)) or np.any(image < 0) or not np.any(image > 0):
        raise ValueError("saved panel-d density must be finite, nonnegative and nonempty")
    n = int(result["n_orientations"])
    expected = 32 if name in ("beta2", "beta3") else 1
    if n != expected:
        raise ValueError(f"{name}: original panel d requires {expected} orientations, not {n}")
    if "stage_names" in result and result["stage_names"][3] != "decay_by":
        raise ValueError("panel d must be the decay_by stage")
    if "b_values" in result:
        b = np.asarray(result["b_values"])
        if b.shape != (25,) or not np.allclose(b, np.arange(12., 61., 2.), rtol=0, atol=1e-10):
            raise ValueError("original panel-d b grid is 12:2:60 fm")
    return name, axis, image, n


def decompose_panel_d(results, qmax=.17, n_radii=341, n_angles=720, floor_fraction=.001):
    """Factor each saved structure/baseline ratio into radial and angular parts.

    J(q)=mean_alpha I(q,alpha), so the common 2*pi cancels from ratios.  A
    missing interpolation point invalidates the complete-circle mean; a partial
    circle is never silently renormalized.  Display thresholds use the peak of
    the *entire Cartesian baseline panel*, exactly as the source ratio plot.
    Low-denominator/out-of-support values remain NaN, with explicit masks.

    The returned NumPy arrays are intended for lossless NPZ export.  Diagnostics
    contain central estimates only: algebraic closure does not certify physical
    convergence, statistical significance or detector performance.
    """
    if not np.isfinite(qmax) or qmax <= 0:
        raise ValueError("qmax must be positive and finite")
    for value, minimum, label in [(n_radii, 2, "n_radii"), (n_angles, 4, "n_angles")]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    if not np.isfinite(floor_fraction) or not 0 <= floor_fraction < 1:
        raise ValueError("floor_fraction must lie in [0,1)")
    sequence = list(results.values()) if isinstance(results, dict) else list(results)
    parsed = [_input(result) for result in sequence]
    names = [row[0] for row in parsed]
    if len(set(names)) != len(names) or names.count("spherical") != 1:
        raise ValueError("unique case names and exactly one spherical baseline are required")
    baseline = parsed[names.index("spherical")]
    _, axis, base_image, _ = baseline
    if any(not np.array_equal(row[1], axis) for row in parsed):
        raise ValueError("all input Cartesian axes must match exactly")
    radii = np.linspace(0., qmax, n_radii)
    angles = np.linspace(0., 2*np.pi, n_angles, endpoint=False)
    px = radii[:, None]*np.cos(angles)[None, :]
    py = radii[:, None]*np.sin(angles)[None, :]
    coordinates = np.stack((px, py), axis=-1)
    interpolation_axis = np.asarray(axis, dtype=float)

    def sample(image):
        return RegularGridInterpolator((interpolation_axis, interpolation_axis), image,
            method="linear", bounds_error=False, fill_value=np.nan)(coordinates)

    base_polar = sample(base_image)
    jbase = base_polar.mean(axis=1)
    peak = float(base_image.max())
    threshold = floor_fraction*peak
    # Preserve the source plot's arithmetic/storage precision for its mask.
    # Interpolation itself uses the exact numerical values of those same axes.
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    mask_cart = (xx*xx+yy*yy <= qmax*qmax) & (base_image > threshold)
    cart_radius = np.hypot(np.asarray(xx, dtype=float), np.asarray(yy, dtype=float))
    mask_polar = np.isfinite(base_polar) & (base_polar > threshold) & (radii[:, None] <= qmax)
    base_radial_valid = np.isfinite(jbase) & (jbase > threshold)
    primitive = "scipy.interpolate.RegularGridInterpolator((axis_qx,axis_qy), normalized_images[3,0], method='linear', bounds_error=False, fill_value=NaN); J=mean over the complete uniformly sampled circle"
    output = {"axis": axis.copy(), "radii": radii, "angles": angles,
        "base_image": base_image.copy(), "base_polar": base_polar, "J_base": jbase,
        "mask_cart": mask_cart, "mask_polar": mask_polar,
        "base_radial_valid": base_radial_valid,
        "config": {"qmax": float(qmax), "n_radii": int(n_radii), "n_angles": int(n_angles),
            "floor_fraction": float(floor_fraction), "baseline_peak": peak, "display_threshold": threshold,
            "b_min_fm": 12., "b_max_fm": 60., "b_nodes": 25, "ymax": 1.,
            "stage": "decay_by", "source": "full", "image_index": [3, 0],
            "radial_denominator_rule": "complete circle and J_base > floor_fraction * full_grid_baseline_peak",
            "angles_unit": "radian", "angles_interval": "[0,2*pi)",
            "J_stored": "angular_mean; multiply by 2*pi for the angular integral, which cancels in every ratio",
            "cartesian_order": "qx,qy", "polar_order": "q,alpha"},
        "cases": {}, "diagnostics": {"projection_primitive": primitive,
            "weighted_mean_definition": "mean_alpha[(I0/J0)*R_ang]=1 on a complete circle; equivalently mean_alpha[I/J]=1. A display-mask-truncated weighted mean is not required to equal 1.",
            "mask_definition": "q<=qmax and baseline density > floor_fraction*baseline full-grid peak; per-case valid masks additionally require complete positive radial denominators",
            "uncertainty_scope": "Central estimates only; no orientation errors. Algebraic closure and circle sampling do not establish b/spatial/y convergence or physical significance.",
            "baseline_cart_display_pixels": int(mask_cart.sum()),
            "baseline_polar_display_points": int(mask_polar.sum()),
            "baseline_complete_circles": int(np.isfinite(jbase).sum()),
            "cases": {}}}
    result_by_name = {result["case"]["name"]: result for result in sequence}
    parsed_by_name = {row[0]: row for row in parsed}
    for name in CASE_ORDER:
        if name not in parsed_by_name:
            continue
        _, _, image, n = parsed_by_name[name]
        polar = sample(image)
        jcase = polar.mean(axis=1)
        radial_valid = base_radial_valid & np.isfinite(jcase) & (jcase >= 0)
        radial = _divide(jcase, jbase, radial_valid)
        direct_cart = _divide(image, base_image, mask_cart)
        radial_map = np.interp(cart_radius, radii, radial, left=np.nan, right=np.nan)
        radial_map[~mask_cart] = np.nan
        valid_cart = mask_cart & np.isfinite(radial_map) & (radial_map > 0)
        angular_map = _divide(direct_cart, radial_map, valid_cart)
        # The exact polar conditional ratio is computed from the sampled
        # primitives, rather than interpolating a separately normalized map.
        valid_polar = mask_polar & radial_valid[:, None] & (jcase[:, None] > 0) & np.isfinite(polar)
        conditional_case = _divide(polar, jcase[:, None], np.isfinite(polar) & (jcase[:, None] > 0))
        conditional_base = _divide(base_polar, jbase[:, None], np.isfinite(base_polar) & (jbase[:, None] > 0))
        angular_polar = _divide(conditional_case, conditional_base, valid_polar)
        direct_polar = _divide(polar, base_polar, mask_polar & np.isfinite(polar))
        full_polar = radial[:, None]*angular_polar
        full_circle_valid = np.all(valid_polar, axis=1)
        weighted_mean = np.full(n_radii, np.nan)
        weighted_mean[full_circle_valid] = np.mean((conditional_base*angular_polar)[full_circle_valid], axis=1)
        # For clipped rings retain the truncated weighted mean as-is, and also
        # record the baseline conditional weight represented by those pixels.
        visible_weight = np.where(valid_polar, conditional_base, 0.).sum(axis=1)
        visible_numerator = np.where(valid_polar, conditional_base*angular_polar, 0.).sum(axis=1)
        visible_mean = _divide(visible_numerator, visible_weight, visible_weight > 0)
        primitive_mean = conditional_case.mean(axis=1)
        complete_positive = np.isfinite(jcase) & (jcase > 0)
        case = {"case": dict(result_by_name[name]["case"]), "n_orientations": n,
            "J": jcase, "polar_image": polar, "full_ratio_cart": direct_cart,
            "radial_ratio": radial, "radial_valid": radial_valid,
            "radial_map": radial_map, "angular_map": angular_map,
            "angular_polar": angular_polar, "full_ratio_polar": full_polar,
            "direct_ratio_polar": direct_polar, "valid_cart": valid_cart, "valid_polar": valid_polar,
            "weighted_angular_mean": weighted_mean,
            "weighted_angular_mean_visible": visible_mean,
            "visible_baseline_angular_fraction": visible_weight/n_angles,
            "conditional_primitive_mean": primitive_mean}
        output["cases"][name] = case
        output["diagnostics"]["cases"][name] = {
            "cart_factor_closure_max_absolute": _maximum(radial_map*angular_map-direct_cart, valid_cart),
            "polar_factor_closure_max_absolute": _maximum(full_polar-direct_polar, valid_polar),
            "complete_display_ring_weighted_mean_max_error": _maximum(weighted_mean-1, full_circle_valid),
            "complete_circle_primitive_mean_max_error": _maximum(primitive_mean-1, complete_positive),
            "complete_display_rings": int(full_circle_valid.sum()),
            "valid_radial_points": int(radial_valid.sum()),
            "valid_cart_points": int(valid_cart.sum()), "valid_polar_points": int(valid_polar.sum()),
            "cart_masked_fraction": float(1-valid_cart.mean()),
            "polar_masked_fraction": float(1-valid_polar.mean()),
            "display_mask_is_not_an_observable": True}
    return output
