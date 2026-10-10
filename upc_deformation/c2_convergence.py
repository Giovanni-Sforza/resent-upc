"""Pixel-based angular moments and orientation-integral convergence diagnostics.

Every orientation is reconstructed on the archived Cartesian panel-d axes.
Complete circles are bilinear samples of those pixels, as in the original
32-pair C2 curves. No analytic harmonic shortcut or new forward model is used.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from .decay import decay_axis_image
from .spectral_weights import validate_edges


def ratio_statistics(numerator, denominator):
    """Ratio of raw sample means and its full delta-method mean covariance."""
    num, den = np.asarray(numerator, float), np.asarray(denominator, float)
    if (num.ndim != 2 or den.shape != num.shape or len(num) < 1
            or not np.isfinite(num).all() or not np.isfinite(den).all()
            or np.any(den <= 0)):
        raise ValueError("finite paired sample-by-coordinate arrays and positive denominators are required")
    mean_den, mean_num = den.mean(0), num.mean(0)
    value = mean_num/mean_den
    influence = (num-value*den)/mean_den
    centered = influence-influence.mean(0)
    n = len(num)
    cov = centered.T@centered/(n*(n-1)) if n > 1 else np.zeros((num.shape[1], num.shape[1]))
    return {"value": value, "orientation_se": np.sqrt(np.maximum(0., np.diag(cov))),
            "covariance": cov, "mean_numerator": mean_num, "mean_denominator": mean_den,
            "n_orientations": n}


def project_orientation_moments(coefficients, radii, axis, q, angles, edges=(.03, .08, .13, .17)):
    """Project raw orientations on original pixels and return ring/bin moments."""
    coeff, radii, axis, q, angles = [np.asarray(v, float) for v in (coefficients, radii, axis, q, angles)]
    edges = validate_edges(edges)
    if (coeff.ndim != 3 or coeff.shape[1:] != (3, len(radii)) or not len(coeff)
            or not np.isfinite(coeff).all() or not np.isfinite(radii).all()
            or np.any(np.diff(radii) <= 0)):
        raise ValueError("finite raw orientation coefficients must have shape (n,3,n_radii)")
    if (axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
            or np.any(np.diff(axis) <= 0) or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=3e-5, atol=0)):
        raise ValueError("a finite uniformly spaced increasing Cartesian axis is required")
    if (q.ndim != 1 or len(q) < 2 or not np.isfinite(q).all() or q[0] < 0
            or np.any(np.diff(q) <= 0) or q[-1] > min(-axis[0], axis[-1])):
        raise ValueError("finite increasing complete-circle momenta must be inside the pixel grid")
    if (angles.ndim != 1 or len(angles) < 4
            or not np.allclose(angles, np.arange(len(angles))*2*np.pi/len(angles), rtol=0, atol=1e-12)):
        raise ValueError("angles must uniformly sample [0,2pi) without a repeated endpoint")
    if edges[-1] > q[-1]:
        raise ValueError("diagnostic annuli must be inside the declared q range")
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    q2 = xx*xx+yy*yy
    cos2 = np.divide(xx*xx-yy*yy, q2, out=np.zeros_like(q2), where=q2 > 0)
    sin2 = np.divide(2*xx*yy, q2, out=np.zeros_like(q2), where=q2 > 0)
    masks = [(q2 >= lo*lo) & ((q2 <= hi*hi) if i == 2 else (q2 < hi*hi))
             for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]
    if any(not mask.any() for mask in masks):
        raise ValueError("each diagnostic annulus must contain pixels")
    pixel_area = float(axis[1]-axis[0])**2
    coords = np.stack((q[:, None]*np.cos(angles), q[:, None]*np.sin(angles)), axis=-1)
    cangle, sangle = np.cos(2*angles), np.sin(2*angles)
    den, cnum, snum = [], [], []
    bin_den, bin_cnum, bin_snum = [], [], []
    image_sum = np.zeros((len(axis), len(axis)))
    r_tensor, axis_tensor = torch.from_numpy(radii), torch.from_numpy(axis)
    minimum_image = np.inf
    with torch.no_grad():
        for c in coeff:
            image = decay_axis_image(torch.from_numpy(c), r_tensor, axis_tensor).numpy()
            if not np.isfinite(image).all() or np.any(image < 0):
                raise ValueError("reconstructed orientation image must be finite and nonnegative")
            minimum_image = min(minimum_image, float(image.min()))
            image_sum += image
            polar = RegularGridInterpolator((axis, axis), image, method="linear", bounds_error=True)(coords)
            den.append(polar.mean(1))
            cnum.append((polar*cangle).mean(1))
            snum.append((polar*sangle).mean(1))
            bin_den.append([image[m].sum()*pixel_area for m in masks])
            bin_cnum.append([(image[m]*cos2[m]).sum()*pixel_area for m in masks])
            bin_snum.append([(image[m]*sin2[m]).sum()*pixel_area for m in masks])
    mean_image = image_sum/len(coeff)
    disk_yield = float(mean_image[q2 <= q[-1]**2].sum()*pixel_area)
    if disk_yield <= 0:
        raise ValueError("the mean image must have positive yield inside the q disk")
    output = {"orientation_denominator": np.asarray(den),
              "orientation_C2_numerator": np.asarray(cnum),
              "orientation_S2_numerator": np.asarray(snum),
              "bin_orientation_denominator": np.asarray(bin_den),
              "bin_orientation_C2_numerator": np.asarray(bin_cnum),
              "bin_orientation_S2_numerator": np.asarray(bin_snum),
              "mean_raw_image": mean_image, "mean_normalized_image": mean_image/disk_yield,
              "normalization_yield": disk_yield,
              "mean_coefficients": coeff.mean(0), "minimum_orientation_image": minimum_image}
    if np.any(output["orientation_denominator"] <= 0) or np.any(output["bin_orientation_denominator"] <= 0):
        raise ValueError("every sampled circle and annulus needs positive raw orientation yield")
    return output


def _estimate(moments, start, stop):
    select = slice(start, stop)
    den = moments["orientation_denominator"][select]
    c = ratio_statistics(moments["orientation_C2_numerator"][select], den)
    s = ratio_statistics(moments["orientation_S2_numerator"][select], den)
    bden = moments["bin_orientation_denominator"][select]
    bc = ratio_statistics(moments["bin_orientation_C2_numerator"][select], bden)
    bs = ratio_statistics(moments["bin_orientation_S2_numerator"][select], bden)
    return {"n_orientations": stop-start, "sample_start": start, "sample_stop": stop,
            "C2": c["value"], "orientation_se": c["orientation_se"], "C2_covariance": c["covariance"],
            "S2": s["value"], "S2_orientation_se": s["orientation_se"], "S2_covariance": s["covariance"],
            "bins": {"C2": bc["value"], "orientation_se": bc["orientation_se"],
                     "C2_covariance": bc["covariance"], "S2": bs["value"],
                     "S2_orientation_se": bs["orientation_se"],
                     "yield": bden.mean(0), "fractions": bden.mean(0)/bden.mean(0).sum()}}


def summarize_orientation_moments(moments, prefix_counts=(32, 64, 128, 256, 512)):
    """Return nested prefixes and genuinely disjoint half/block estimates."""
    n = len(moments["orientation_denominator"])
    final = _estimate(moments, 0, n)
    if n == 1:
        return final
    if n % 4:
        raise ValueError("deformed sample must split into four equal disjoint blocks")
    counts = [int(k) for k in prefix_counts]
    if counts != sorted(set(counts)) or any(k < 2 or k > n for k in counts):
        raise ValueError("prefix counts must be unique, increasing and within the available sample")
    final["nested"] = {str(k): _estimate(moments, 0, k) for k in counts}
    final["halves"] = [_estimate(moments, i*n//2, (i+1)*n//2) for i in range(2)]
    final["blocks"] = [_estimate(moments, i*n//4, (i+1)*n//4) for i in range(4)]
    first, second = final["halves"]
    difference = first["C2"]-second["C2"]
    difference_se = np.sqrt(first["orientation_se"]**2+second["orientation_se"]**2)
    final["half_difference"] = difference
    final["half_difference_se"] = difference_se
    final["half_difference_z_pointwise"] = np.divide(difference, difference_se,
        out=np.zeros_like(difference), where=difference_se > 0)
    return final


def check_original_prefix(entry, original_curves, name, atol=1e-12):
    """Require reconstructed old32 curve and SE to match the saved reference."""
    previous = entry["nested"]["32"] if "nested" in entry else entry
    c2 = np.asarray(original_curves[f"{name}_C2"])
    se = np.asarray(original_curves[f"{name}_orientation_se"])
    if previous["C2"].shape != c2.shape or previous["orientation_se"].shape != se.shape:
        raise ValueError("the saved original curve shape differs")
    c2_error = float(np.max(np.abs(previous["C2"]-c2)))
    se_error = float(np.max(np.abs(previous["orientation_se"]-se)))
    if c2_error > atol or se_error > atol:
        raise ValueError(f"{name}: original prefix closure failed: C2={c2_error:g}, SE={se_error:g}")
    return {"max_absolute_C2_error": c2_error, "max_absolute_SE_error": se_error}
