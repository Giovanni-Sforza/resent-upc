"""Fixed-annulus angular moments of archived panel-d Cartesian densities.

The skin-minus-sphere circular C2 curve selects two common boundaries.  Bin
observables then use the original Cartesian pixels, never the ratio image or
its display mask.  Retained orientation coefficients are used only to assess
orientation integration noise after reproducing the archived mean image.
"""
from __future__ import annotations

import numpy as np

from .angular_observables import radial_moment


CASE_ORDER = ("spherical", "beta2", "beta3", "skin")


def _edges(edges):
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] != 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError("edges must increase strictly from zero")
    return edges


def cartesian_bin_statistics(image, axis, edges):
    """Sum one or many densities over common, disjoint pixel annuli.

    Leading image axes are preserved.  The origin has zero angular harmonic;
    all other pixels use their physical (qx,qy) angle.  Each inner boundary
    belongs to the next bin, while the outermost boundary is included.
    """
    image, axis, edges = np.asarray(image, dtype=float), np.asarray(axis, dtype=float), _edges(edges)
    if (axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
            or np.any(np.diff(axis) <= 0)
            or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=3e-5, atol=0)):
        raise ValueError("pixel sums require a finite uniform increasing axis")
    if image.shape[-2:] != (len(axis), len(axis)) or not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError("finite nonnegative density must match Cartesian axes")
    x, y = np.meshgrid(axis, axis, indexing="ij")
    q2 = x*x + y*y
    c = np.divide(x*x-y*y, q2, out=np.zeros_like(q2), where=q2 > 0)
    s = np.divide(2*x*y, q2, out=np.zeros_like(q2), where=q2 > 0)
    pixel_area = float(axis[1]-axis[0])**2
    masks = [(q2 >= lo*lo) & ((q2 <= hi*hi) if i == len(edges)-2 else (q2 < hi*hi))
             for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]
    den = np.stack([image[..., mask].sum(-1)*pixel_area for mask in masks], axis=-1)
    cnum = np.stack([(image[..., mask]*c[mask]).sum(-1)*pixel_area for mask in masks], axis=-1)
    snum = np.stack([(image[..., mask]*s[mask]).sum(-1)*pixel_area for mask in masks], axis=-1)
    if np.any(den <= 0):
        raise ValueError("every annulus must have positive yield")
    return {"yield": den, "C2_numerator": cnum, "S2_numerator": snum,
            "C2": cnum/den, "S2": snum/den,
            "yield_fraction": den/den.sum(-1, keepdims=True),
            "pixel_counts": np.array([mask.sum() for mask in masks]),
            "pixel_area": pixel_area}


def angular_curve(polar, angles):
    """Density-weighted circular moments on complete uniformly sampled rings."""
    polar, angles = np.asarray(polar, dtype=float), np.asarray(angles, dtype=float)
    if (angles.ndim != 1 or len(angles) < 4 or not np.allclose(
            angles, np.arange(len(angles))*2*np.pi/len(angles), rtol=0, atol=1e-12)):
        raise ValueError("angles must sample [0,2pi) uniformly without a duplicate endpoint")
    if polar.ndim != 2 or polar.shape[1] != len(angles) or not np.isfinite(polar).all() or np.any(polar < 0):
        raise ValueError("complete finite nonnegative polar rings are required")
    den = polar.mean(-1)
    if np.any(den <= 0):
        raise ValueError("every ring must have positive density")
    cnum = (polar*np.cos(2*angles)).mean(-1)
    snum = (polar*np.sin(2*angles)).mean(-1)
    return {"J": den, "C2": cnum/den, "S2": snum/den,
            "C2_numerator": cnum, "S2_numerator": snum}


def polar_bin_statistics(polar, radii, angles, edges):
    """Crosscheck with q dq integration of full-circle sampled densities.

    Integrate the piecewise-linear angular numerator and denominator with the
    correct Jacobian, rather than averaging C2(q) uniformly over radius.
    """
    curve, edges = angular_curve(polar, angles), _edges(edges)
    moments = {key: np.array([2*np.pi*radial_moment(curve[key], radii, lo, hi)
                             for lo, hi in zip(edges[:-1], edges[1:])])
               for key in ("J", "C2_numerator", "S2_numerator")}
    den = moments["J"]
    if np.any(den <= 0):
        raise ValueError("every annulus must have positive yield")
    return {"yield": den, "C2": moments["C2_numerator"]/den,
            "S2": moments["S2_numerator"]/den, "yield_fraction": den/den.sum()}


def find_skin_boundaries(radii, delta_c2, root_search_min=.01):
    """Find the two noncentral sign-changing zeros by linear interpolation."""
    q, value = np.asarray(radii, dtype=float), np.asarray(delta_c2, dtype=float)
    if (q.ndim != 1 or value.shape != q.shape or len(q) < 3 or not np.isfinite(q).all()
            or not np.isfinite(value).all() or np.any(np.diff(q) <= 0)):
        raise ValueError("finite values on an increasing radial grid are required")
    roots = []
    for i in range(len(q)-1):
        if q[i] <= root_search_min:
            continue
        if value[i]*value[i+1] < 0:
            roots.append(q[i]-value[i]*(q[i+1]-q[i])/(value[i+1]-value[i]))
        elif value[i] == 0 and i > 0 and value[i-1]*value[i+1] < 0:
            roots.append(q[i])
    if len(roots) != 2:
        raise ValueError(f"expected two noncentral sign-changing skin roots, found {len(roots)}")
    return np.asarray(roots)


def _orientation_statistics(result, edges, primary):
    """Delta-method SE for ratios of raw orientation-averaged intensities."""
    import torch
    from .decay import decay_axis_image

    coeff = np.asarray(result["orientation_coefficients"][:, -1, 0], dtype=float)
    n = len(coeff)
    if n != int(result["n_orientations"]) or coeff.ndim != 3 or coeff.shape[1] != 3:
        raise ValueError("retained orientation samples do not match the declared count")
    with torch.no_grad():
        images = decay_axis_image(torch.from_numpy(coeff),
            torch.from_numpy(np.asarray(result["radii"], dtype=float)),
            torch.from_numpy(np.asarray(result["axis"], dtype=float))).numpy()
    archived = np.asarray(result["images"][3, 0], dtype=float)
    mean = images.mean(0)
    scale = float(np.max(np.abs(archived)))
    closure = float(np.max(np.abs(mean-archived))/scale)
    if closure > 2e-12:
        raise ValueError(f"orientation reconstruction does not close archived image: {closure:g}")
    stats = cartesian_bin_statistics(images, result["axis"], edges)
    den, cnum, snum = stats["yield"], stats["C2_numerator"], stats["S2_numerator"]
    mean_den = den.mean(0)
    result_stats = {"orientation_image_relative_closure": closure}
    for key, num in (("C2", cnum), ("S2", snum)):
        value = num.mean(0)/mean_den
        if not np.allclose(value, primary[key], rtol=1e-10, atol=2e-13):
            raise ValueError("orientation mean ratio does not reproduce direct pixels")
        influence = (num-value*den)/mean_den
        centered = influence-influence.mean(0)
        cov = centered.T@centered/(n*(n-1)) if n > 1 else np.zeros((len(edges)-1, len(edges)-1))
        result_stats[f"{key}_orientation_se"] = np.sqrt(np.maximum(0., np.diag(cov)))
        result_stats[f"{key}_orientation_covariance"] = cov
    disk = den.sum(-1)
    fraction = mean_den/disk.mean()
    influence = (den-disk[:, None]*fraction)/disk.mean()
    result_stats["yield_fraction_orientation_se"] = (influence.std(0, ddof=1)/np.sqrt(n)
                                                     if n > 1 else np.zeros(len(edges)-1))
    return result_stats


def analyze_three_bins(results, decomposition, root_search_min=.01):
    """Analyze the original four-case panel-d cache in three frozen annuli.

    ``decomposition`` accepts the flat saved NPZ mapping or the nested output
    of decompose_panel_d.  All returned arrays are suitable for NPZ export;
    root drivers may convert them to lists for JSON output.
    """
    from .panel_d_ratios import _input

    sequence = list(results.values()) if isinstance(results, dict) else list(results)
    rows = {_input(result)[0]: result for result in sequence}
    if len(rows) != len(sequence) or set(rows) != set(CASE_ORDER):
        raise ValueError("unique spherical, beta2, beta3 and skin cases are required")
    axis = np.asarray(rows["spherical"]["axis"])
    if any(not np.array_equal(result["axis"], axis) for result in sequence):
        raise ValueError("all cases must use identical Cartesian axes")
    if not np.array_equal(axis, decomposition["axis"]):
        raise ValueError("polar decomposition axes differ from source cache")
    if not np.array_equal(rows["spherical"]["normalized_images"][3, 0], decomposition["base_image"]):
        raise ValueError("polar decomposition baseline differs from source cache")

    def field(name, key):
        return decomposition["cases"][name][key] if "cases" in decomposition else decomposition[f"{name}__{key}"]

    radii, angles = np.asarray(decomposition["radii"]), np.asarray(decomposition["angles"])
    polar = {"spherical": decomposition["base_polar"],
             **{name: field(name, "polar_image") for name in CASE_ORDER[1:]}}
    curves = {name: angular_curve(polar[name], angles) for name in CASE_ORDER}
    for name in CASE_ORDER:
        curves[name]["delta_C2"] = curves[name]["C2"]-curves["spherical"]["C2"]
    roots = find_skin_boundaries(radii, curves["skin"]["delta_C2"], root_search_min)
    edges = np.r_[0., roots, radii[-1]]
    output = {"edges_GeV": edges, "radii": radii, "angles": angles,
              "skin_angular_polar": np.asarray(field("skin", "angular_polar")),
              "skin_valid_polar": np.asarray(field("skin", "valid_polar")),
              "curves": curves, "cases": {},
              "config": {"stage": "decay_by", "branch": "full", "image_index": [3, 0],
                         "boundary_source": "two sign-changing zeros of C2_skin(q)-C2_spherical(q)",
                         "boundary_root_interpolation": "linear between saved complete-circle radial samples",
                         "root_search_min_GeV": root_search_min,
                         "common_bins": True, "display_floor_applied_to_observables": False,
                         "bin_membership": "[lower,upper); final upper edge included",
                         "orientation_uncertainty_scope": "orientation Monte Carlo integration only; deterministic sphere/skin have zero orientation SE; no detector or b-grid uncertainty",
                         "central_estimate": "direct normalized_images[3,0] pixel sums; origin cos2alpha=sin2alpha=0",
                         "polar_crosscheck": "complete angular means; exact q dq integral of piecewise-linear radial numerator and denominator"},
              "diagnostics": {}}
    for name in CASE_ORDER:
        result = rows[name]
        primary = cartesian_bin_statistics(result["normalized_images"][3, 0], axis, edges)
        secondary = polar_bin_statistics(polar[name], radii, angles, edges)
        orientation = _orientation_statistics(result, edges, primary)
        output["cases"][name] = {"case": dict(result["case"]), "n_orientations": int(result["n_orientations"]),
                                  **primary, **orientation, "polar": secondary,
                                  "diagnostics": {"pixel_minus_polar_C2": primary["C2"]-secondary["C2"],
                                                  "pixel_minus_polar_yield_fraction": primary["yield_fraction"]-secondary["yield_fraction"]}}
    base = output["cases"]["spherical"]
    for name, entry in output["cases"].items():
        entry["delta_C2"] = entry["C2"]-base["C2"]
        entry["delta_C2_orientation_se"] = np.sqrt(entry["C2_orientation_se"]**2+base["C2_orientation_se"]**2)
        entry["polar"]["delta_C2"] = entry["polar"]["C2"]-base["polar"]["C2"]
        entry["diagnostics"]["pixel_minus_polar_delta_C2"] = entry["delta_C2"]-entry["polar"]["delta_C2"]
    output["diagnostics"] = {
        "root_values_linear_interpolation": np.interp(roots, radii, curves["skin"]["delta_C2"]),
        "max_abs_pixel_minus_polar_C2": max(float(np.max(np.abs(e["diagnostics"]["pixel_minus_polar_C2"]))) for e in output["cases"].values()),
        "max_abs_pixel_minus_polar_delta_C2": max(float(np.max(np.abs(e["diagnostics"]["pixel_minus_polar_delta_C2"]))) for e in output["cases"].values()),
        "max_orientation_image_relative_closure": max(e["orientation_image_relative_closure"] for e in output["cases"].values()),
        "boundary_fitting_uncertainty_included": False,
        "interpretation": "Exploratory skin-selected fixed bins, shared by all cases. Crosschecks quantify pixel/radial sampling only, not convergence or experimental sensitivity."}
    return output
