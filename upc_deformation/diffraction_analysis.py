"""Pure analysis of paired full/self diffraction campaigns.

No forward model, event generator, or sensitivity-window search is run here.
``D`` and ``J`` are radial *two-dimensional densities* (self/full A0), without
a q Jacobian.  Angular curves use the archived Cartesian reconstruction and
circle interpolation.  Window moments are sums of raw Cartesian pixels.

All quoted errors are delta-method orientation-integration standard errors.
They exclude grid, window-location, detector, and model uncertainty.  A sphere
registers fixed windows for every case; no window is optimized on a response.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .c2_convergence import ratio_statistics
from .precision_analysis import PixelMomentProjector, paired_ratio_difference


def _mean_statistics(samples):
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite sample-by-coordinate arrays are required")
    n = len(x)
    se = x.std(axis=0, ddof=1)/np.sqrt(n) if n > 1 else np.zeros(x.shape[1])
    return {"value": x.mean(axis=0), "orientation_se": se, "n_orientations": n}


def _radial_ratio(numerator, denominator, q):
    """Keep raw diffraction zeros; a ratio exists only where mean D is positive.

    Individual orientation denominators may vanish without making a ratio of
    means undefined.  No epsilon is added to either individual or mean yields.
    """
    num, den = np.asarray(numerator, float), np.asarray(denominator, float)
    if np.any(den < 0):
        raise ValueError("negative self radial density")
    valid = den.mean(axis=0) > 0
    num, den = num[:, valid], den[:, valid]
    if np.all(den > 0):
        result = ratio_statistics(num, den)
    else:
        n = len(den)
        mn, md = num.mean(axis=0), den.mean(axis=0)
        value = mn/md
        u = (num-value*den)/md
        u -= u.mean(axis=0)
        cov = u.T@u/(n*(n-1)) if n > 1 else np.zeros((len(md), len(md)))
        result = {"value": value, "orientation_se": np.sqrt(np.maximum(0, np.diag(cov))),
                  "covariance": cov, "mean_numerator": mn, "mean_denominator": md,
                  "n_orientations": n}
    result.update(q=np.asarray(q)[valid], valid_mask=valid,
                  omitted_q=np.asarray(q)[~valid],
                  domain="positive ensemble-mean self density; no pseudocounts")
    return result


def register_spherical_windows(radii, sphere_self_A0, *, qmax=.17,
                               inner_min=.03, outer_fractions=(.5, .1),
                               fixed_window=(.12, .14)):
    """Register a finite-q bright subset from the sphere's first radial lobe.

    The peak is the first change from rising/flat to declining, including the
    origin when the density initially decreases.  Crossings are the first
    declining brackets after that peak and before the first observed minimum.
    Crossing positions are explicitly linear interpolations of saved nodes;
    no density sample is invented at qmax.  Missing boundaries remain censored
    and dependent windows are omitted, never silently clipped to the domain.
    """
    r, d = np.asarray(radii, float), np.asarray(sphere_self_A0, float)
    fractions = tuple(float(x) for x in outer_fractions)
    fixed = tuple(float(x) for x in fixed_window)
    if (r.ndim != 1 or d.shape != r.shape or len(r) < 3
            or not np.isfinite(r).all() or not np.isfinite(d).all()
            or np.any(np.diff(r) <= 0) or r[0] != 0 or np.any(d < 0)
            or not np.isfinite(qmax) or qmax <= 0 or qmax > r[-1]
            or not np.isfinite(inner_min) or inner_min < 0
            or len(fractions) != 2 or not 0 < fractions[1] < fractions[0] < 1
            or len(fixed) != 2 or not 0 <= fixed[0] < fixed[1]
            or not np.isfinite(fixed).all()):
        raise ValueError("invalid radial density, domain, fractions, or fixed window")
    select = r <= qmax
    r, d = r[select], d[select]
    if len(r) < 3 or d.max() <= 0:
        raise ValueError("the registered domain needs three nodes and positive density")
    declining = np.flatnonzero(np.diff(d) < 0)
    peak_observed = bool(len(declining))
    peak_i = int(declining[0]) if peak_observed else len(r)-1
    peak = {"q": float(r[peak_i]), "value": float(d[peak_i]),
            "index": peak_i, "status": "observed" if peak_observed else "right_censored"}
    minimum_i = None
    if peak_observed:
        # A strict decline followed by a flat bottom and then a rise counts as
        # an observed minimum; retain an actual saved node at the first bottom.
        for i in range(peak_i+1, len(r)-1):
            if d[i] >= d[i-1]:
                continue
            j = i+1
            while j < len(r) and d[j] == d[i]:
                j += 1
            if j < len(r) and d[j] > d[i]:
                minimum_i = i
                break
    minimum = ({"status": "observed", "q": float(r[minimum_i]),
                "value": float(d[minimum_i]), "index": minimum_i}
               if minimum_i is not None else
               {"status": "right_censored", "q": None, "value": None,
                "last_observed_q": float(r[-1])})

    def crossing(fraction):
        threshold = fraction*d[peak_i]
        stop = minimum_i if minimum_i is not None else len(r)-1
        for j in range(peak_i+1, stop+1):
            if d[j] <= threshold <= d[j-1] and d[j] < d[j-1]:
                q = r[j-1] + (threshold-d[j-1])*(r[j]-r[j-1])/(d[j]-d[j-1])
                return {"status": "observed", "fraction": fraction,
                        "q": float(q), "bracket_q": [float(r[j-1]), float(r[j])],
                        "bracket_values": [float(d[j-1]), float(d[j])],
                        "location_method": "linear interpolation of saved radial nodes"}
        return {"status": ("not_reached_before_first_minimum" if minimum_i is not None
                           else "right_censored"), "fraction": fraction, "q": None,
                "last_observed_q": float(r[stop])}

    crossings = {str(f): crossing(f) for f in dict.fromkeys((*fractions, .5))}
    first, last = (crossings[str(f)] for f in fractions)
    half = crossings[str(.5)]

    def window(low, high, dependency_ok=True):
        if not dependency_ok:
            return {"status": "censored_boundary", "low": low, "high": high}
        if low is None or high is None or not low < high or high > qmax:
            return {"status": "outside_domain_or_empty", "low": low, "high": high}
        return {"status": "registered", "low": float(low), "high": float(high),
                "upper_inclusive": bool(high == qmax)}

    windows = {
        "bright_lobe_inner": window(inner_min, first["q"], first["status"] == "observed"),
        "outer_slope": window(first["q"], last["q"],
                              first["status"] == last["status"] == "observed"),
        "bright_lobe": window(inner_min, last["q"], last["status"] == "observed"),
        "fixed_012_014": window(*fixed),
    }
    return {"qmax": float(qmax), "last_observed_q": float(r[-1]),
            "inner_min": float(inner_min), "outer_fractions": list(fractions),
            "fixed_window": list(fixed), "peak": peak, "first_minimum": minimum,
            "crossings": crossings, "inner_crossing": first, "outer_crossing": last,
            "outer_half_height_q": half["q"],
            "outer_half_width": None if half["q"] is None else half["q"]-peak["q"],
            "windows": windows,
            "definitions": {"density": "sphere self A0; no q Jacobian",
                "bright_lobe": "fraction-defined finite-q subset, not the entire first lobe",
                "uncertainty": "crossing brackets are discretization diagnostics, not error bars",
                "fixed_012_014": "legacy key; actual bounds are given by fixed_window"}}


def _window_map(projector, registry):
    xx, yy = np.meshgrid(projector.axis, projector.axis, indexing="ij")
    q2 = xx*xx + yy*yy
    cos2 = np.divide(xx*xx-yy*yy, q2, out=np.zeros_like(q2), where=q2 > 0)
    names, bounds, masks = [], [], []
    for name, w in registry["windows"].items():
        if w["status"] != "registered":
            continue
        low, high = w["low"], w["high"]
        mask = (q2 >= low*low) & ((q2 <= high*high) if w["upper_inclusive"] else q2 < high*high)
        if not mask.any():
            raise ValueError(f"registered window {name} contains no Cartesian pixels")
        names.append(name)
        bounds.append((low, high))
        masks.append(mask)
    if not names:
        return names, np.empty((0, 2)), sparse.csr_matrix((0, projector.pixel_map.shape[1]))
    weights = np.asarray([(m*f*projector.pixel_area).ravel()
                          for f in (np.ones_like(q2), cos2) for m in masks])
    return names, np.asarray(bounds), sparse.csr_matrix(weights) @ projector.pixel_map


def _summarize_moments(n0, n2):
    # Each input has shape (orientation, branch, coordinate).  Cross terms are
    # signed raw moments, never independently normalized angular distributions.
    result = {}
    for branch, i in (("full", 0), ("self", 1)):
        result[branch] = {"N0": _mean_statistics(n0[:, i]),
                          "N2": _mean_statistics(n2[:, i]),
                          "C2": ratio_statistics(n2[:, i], n0[:, i])}
    result["cross"] = {"N0": _mean_statistics(n0[:, 0]-n0[:, 1]),
                       "N2": _mean_statistics(n2[:, 0]-n2[:, 1])}
    result["full_minus_self_C2"] = paired_ratio_difference(
        n2[:, 0], n0[:, 0], n2[:, 1], n0[:, 1])
    return result


def analyze_diffraction_case(orientation_coefficients, radii, axis, q, angles,
                             registry, *, branch_names=("full", "self"),
                             chunk_size=128, keep_samples=False):
    """Analyze one case/one b level, keeping large temporary arrays bounded.

    Input shape is (orientation, full/self, A0/C/D, radius).  Set keep_samples
    only when paired comparisons between runs are required; private ``_samples``
    then contains circle/window N0,N2, not another coefficient-bank copy.
    """
    c = np.asarray(orientation_coefficients, dtype=float)
    r, a, q, phi = [np.asarray(v, float) for v in (radii, axis, q, angles)]
    if (c.ndim != 4 or c.shape[1:] != (2, 3, len(r)) or not len(c)
            or tuple(branch_names) != ("full", "self") or chunk_size < 1):
        raise ValueError("coefficients must be (n,2,3,nr), ordered full,self")
    if (q.ndim != 1 or len(q) < 2 or q[0] <= 0
            or not np.isclose(q[-1], registry["qmax"], rtol=0, atol=1e-12)):
        raise ValueError("angular q grid must exclude undefined q=0 and end at the registered qmax")
    # These three auxiliary bins only satisfy the old projector's constructor;
    # actual observables below use independently registered Cartesian windows.
    projector = PixelMomentProjector(r, a, q, phi, edges=np.linspace(0, q[-1], 4))
    names, bounds, window_map = _window_map(projector, registry)
    nq, nw, n = len(q), len(names), len(c)
    transform = sparse.vstack((projector.moment_map[:2*nq], window_map), format="csr")
    n0, n2 = np.empty((n, 2, nq)), np.empty((n, 2, nq))
    w0, w2 = np.empty((n, 2, nw)), np.empty((n, 2, nw))
    for start in range(0, n, chunk_size):
        stop = min(start+chunk_size, n)
        for branch in range(2):
            block = c[start:stop, branch]
            if not np.isfinite(block).all():
                raise ValueError("nonfinite orientation coefficients")
            lower = block[:, 0]-.5*np.hypot(block[:, 1], block[:, 2])
            if lower.min() < -1e-12*max(float(np.abs(block).max()), np.finfo(float).tiny):
                raise ValueError("negative full/self angular density")
            values = (transform @ block.reshape(stop-start, -1).T).T
            n0[start:stop, branch], n2[start:stop, branch] = values[:, :nq], values[:, nq:2*nq]
            w0[start:stop, branch], w2[start:stop, branch] = values[:, 2*nq:2*nq+nw], values[:, 2*nq+nw:]
    if np.any(n0 <= 0) or np.any(w0 <= 0):
        raise ValueError("positive full/self circle and window yields required; exclude undefined q=0 when needed")

    curves = _summarize_moments(n0, n2)
    curves.update(q=q.copy(), alpha_defined=q > 0)
    mean = c.mean(axis=0)
    curves["analytic_C2_diagnostic"] = {}
    for i, branch in enumerate(("full", "self")):
        denominator = np.interp(q, r, mean[i, 0])
        numerator = -.25*np.interp(q, r, mean[i, 1])
        valid = denominator > 0
        curves["analytic_C2_diagnostic"][branch] = {
            "q": q[valid], "value": numerator[valid]/denominator[valid],
            "valid_mask": valid, "omitted_q": q[~valid]}
    windows = _summarize_moments(w0, w2) if nw else {}
    windows.update(names=names, bounds=bounds)
    if "outer_slope" in names and "bright_lobe_inner" in names:
        oi, ii = names.index("outer_slope"), names.index("bright_lobe_inner")
        windows["outer_over_inner"] = {
            branch: ratio_statistics(w0[:, i, oi:oi+1], w0[:, i, ii:ii+1])
            for i, branch in enumerate(("full", "self"))}
        windows["joint_outer_ratio_C2"] = ratio_statistics(
            np.column_stack((w0[:, 0, oi], w2[:, 0])),
            np.column_stack((w0[:, 0, ii], w0[:, 0])))
        windows["joint_outer_ratio_C2"]["labels"] = ["full_outer_over_inner"] + ["full_C2_"+x for x in names]
    keep = r <= registry["qmax"]
    j, d = c[:, 0, 0, keep], c[:, 1, 0, keep]
    radial = {"q": r[keep], "J": _mean_statistics(j), "D": _mean_statistics(d),
              "J_minus_D": _mean_statistics(j-d), "J_over_D": _radial_ratio(j, d, r[keep])}
    result = {"n_orientations": n, "registry": registry, "radial": radial,
              "curves": curves, "windows": windows,
              "definitions": {
                  "D_J": "self/full mean A0 radial density, without q Jacobian",
                  "C2": "ratio of raw mean Cartesian circle/annulus moments; not mean of orientation C2",
                  "cross": "signed full-self raw N0/N2; no normalized cross C2 is defined",
                  "C2_at_zero": "alpha is experimentally undefined at q=0, which is excluded from angular curves",
                  "uncertainty": "orientation-integration delta-method SE only; n=1 variance is not estimable",
                  "mechanism": "full-minus-self C2 is a nonlinear diagnostic, not an additive interference observable"}}
    if keep_samples:
        result["_samples"] = {"curves_N0": n0, "curves_N2": n2,
                              "windows_N0": w0, "windows_N2": w2}
    return result


def _difference_statistics(case, reference):
    if np.shape(case["value"]) != np.shape(reference["value"]):
        raise ValueError("contrast coordinates differ")
    covariance = case["covariance"]+reference["covariance"]
    return {"value": case["value"]-reference["value"], "covariance": covariance,
            "orientation_se": np.sqrt(np.maximum(0, np.diag(covariance)))}


def _influence(num, den):
    value = num.mean(axis=0)/den.mean(axis=0)
    return (num-value*den)/den.mean(axis=0)


def contrast_cases(case, reference, *, paired=False):
    """C2 structural differences and the paired full/self double difference.

    Pairing between cases/runs must be explicitly asserted by the caller after
    checking orientation identities.  Full/self pairing is always retained.
    A reference with one orientation is treated as deterministic; callers must
    not use that convention to certify a deformed one-sample reference.
    """
    if (case["registry"] != reference["registry"]
            or not np.array_equal(case["curves"]["q"], reference["curves"]["q"])
            or case["windows"]["names"] != reference["windows"]["names"]):
        raise ValueError("comparisons require identical fixed windows and q grids")
    n, nr = case["n_orientations"], reference["n_orientations"]
    use_pair = paired and n > 1 and nr > 1
    if use_pair and (n != nr or "_samples" not in case or "_samples" not in reference):
        raise ValueError("paired case comparisons require equal counts and keep_samples=True")
    result = {"pairing": "paired orientations" if use_pair else "independent or deterministic reference"}
    for section in ("curves", "windows"):
        if section == "windows" and not case[section]["names"]:
            result[section] = {}
            continue
        result[section] = {}
        for metric in ("full", "self", "full_minus_self_C2"):
            ca = case[section][metric] if metric == "full_minus_self_C2" else case[section][metric]["C2"]
            re = reference[section][metric] if metric == "full_minus_self_C2" else reference[section][metric]["C2"]
            estimate = _difference_statistics(ca, re)
            if use_pair:
                influences = []
                for entry in (case, reference):
                    den, num = (entry["_samples"][section+suffix] for suffix in ("_N0", "_N2"))
                    if metric == "full_minus_self_C2":
                        influences.append(_influence(num[:, 0], den[:, 0])-_influence(num[:, 1], den[:, 1]))
                    else:
                        i = 0 if metric == "full" else 1
                        influences.append(_influence(num[:, i], den[:, i]))
                u = influences[0]-influences[1]
                u -= u.mean(axis=0)
                cov = u.T@u/(n*(n-1))
                estimate.update(covariance=cov, orientation_se=np.sqrt(np.maximum(0, np.diag(cov))))
            key = "double_difference_C2" if metric == "full_minus_self_C2" else metric+"_C2"
            result[section][key] = estimate
    result["definitions"] = {"double_difference_C2": "(C2full-C2self)case-(C2full-C2self)reference",
                              "uncertainty": "orientation integration only; fixed boundaries and numerical grids are conditioned on"}
    return result


def analyze_diffraction_cases(cases, radii, axis, q, angles, *, sphere_name="spherical",
                              registry=None, paired_with_sphere=(),
                              window_options=None, chunk_size=128):
    """Convenience wrapper for small banks; production CLIs should stream cases.

    A value is either the coefficient bank itself or a storage mapping with
    ``orientation_coefficients``.  Stored b_level_coefficients are deliberately
    not expanded here: analyze each level separately with one shared registry.
    """
    def coefficients(entry):
        return entry["orientation_coefficients"] if isinstance(entry, dict) else entry

    if sphere_name not in cases:
        raise ValueError("a spherical reference is required")
    if registry is None:
        sphere = np.asarray(coefficients(cases[sphere_name]), float)
        registry = register_spherical_windows(radii, sphere[:, 1, 0].mean(axis=0),
                                               **(window_options or {}))
    paired_names = set(paired_with_sphere)
    results = {name: analyze_diffraction_case(coefficients(entry), radii, axis, q, angles,
                    registry, chunk_size=chunk_size,
                    keep_samples=bool(paired_names) and (name == sphere_name or name in paired_names))
               for name, entry in cases.items()}
    contrasts = {name: contrast_cases(result, results[sphere_name], paired=name in paired_names)
                 for name, result in results.items() if name != sphere_name}
    for result in results.values():
        result.pop("_samples", None)
    return {"registry": registry, "cases": results, "contrasts": contrasts}
