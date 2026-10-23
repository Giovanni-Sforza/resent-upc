"""Shape features of orientation-averaged radial densities and joint MC errors.

The sampling unit is a *pair of nuclear orientations*.  A replicate averages
raw A0 profiles before finding its own peak, crossings and finite-range mean
pT. It never averages per-orientation nonlinear features. Crossings use A0
without a Jacobian; mean pT includes the radial q dq area measure.

For large banks, nonoverlapping equal-size orientation blocks are resampled.
This is a block approximation to orientation-integration uncertainty, not an
event bootstrap.  Identical plans preserve pairing across b levels/branches
and, when the caller verifies the same ordered rotation bank, across cases.
Small replicate-feature arrays support paired contrasts without a dense
covariance matrix over the whole parameter scan.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


FEATURE_NAMES = ("Q10_GeV", "B_mid", "K_tail", "mean_pT_GeV")
BRANCH_NAMES = ("full", "self")
_PLAN_ALGORITHM = "PCG64-SeedSequence(seed,n_blocks,replicate)-multinomial-v1"


def _radial_grid(q):
    q = np.asarray(q, dtype=float)
    if (q.ndim != 1 or len(q) < 3 or not np.isfinite(q).all()
            or q[0] < 0 or np.any(np.diff(q) <= 0)):
        raise ValueError("q must contain at least three finite increasing nonnegative nodes")
    return q


def extract_tail_features(q, mean_A0):
    """Extract own-peak, same-descending-branch 50/10/1 percent features.

    The saved-node maximum, first subsequent upturn, and linear interpolation
    are exactly the primary conventions of ``analyze_single_image_tail_lobes``.
    The caller supplies the registered q domain; no endpoint is synthesized.
    Missing Q1 censors K_tail but does not discard observed Q10 or B_mid.
    Mean pT uses the full saved domain, including any later upturn, and exact
    q/q-squared moments of the piecewise-linear A0. It stays defined when
    crossings are censored; it is a finite-range mean, not an extrapolation.
    """
    q = _radial_grid(q)
    j = np.asarray(mean_A0, dtype=float)
    if (j.shape != q.shape or not np.isfinite(j).all()
            or np.any(j < 0) or j.max() <= 0):
        raise ValueError("mean_A0 must be a finite nonnegative density with positive maximum")
    # With q=a+h*t and J=j0*(1-t)+j1*t, integrate the two basis
    # functions analytically. Positive endpoint weights avoid cancellation;
    # normalizing the intensity cancels in the ratio and limits overflow.
    a, h = q[:-1], np.diff(q)
    normalized = j/j.max()
    left, right = normalized[:-1], normalized[1:]
    radial_yield = np.sum(h*(left*(a/2+h/6)+right*(a/2+h/3)))
    momentum_sum = np.sum(h*(left*(a*a/2+a*h/3+h*h/12)
                            + right*(a*a/2+2*a*h/3+h*h/4)))
    mean_pT = float(momentum_sum/radial_yield)
    peak = int(j.argmax())
    rises = np.flatnonzero(np.diff(j)[peak:] > 0)
    end = peak + int(rises[0]) if len(rises) else len(q)-1
    result = {
        "status": "observed", "peak_node_index": peak,
        "peak_q_GeV": float(q[peak]), "peak_height": float(j[peak]),
        "peak_method": "saved-node maximum", "interpolation": "linear",
        "local_peak_count": int(np.sum((j[1:-1] > j[:-2]) & (j[1:-1] >= j[2:]))),
        "descending_branch_end_index": end,
        "descending_branch_end_q_GeV": float(q[end]),
        "branch_end_observed_upturn": bool(len(rises)),
        "crossings": {}, "censor_reason": None,
        "mean_pT_GeV": mean_pT,
        "mean_pT_support_GeV": [float(q[0]), float(q[-1])],
        "mean_pT_integration": "exact q and q^2 moments of piecewise-linear A0; radial q dq measure; no extrapolation",
    }
    missing = {}
    for name, fraction in (("Q50", .5), ("Q10", .1), ("Q1", .01)):
        threshold = fraction*j[peak]
        candidates = [i for i in range(peak, end)
                      if j[i] >= threshold >= j[i+1] and j[i] > j[i+1]]
        if not candidates:
            reason = "not_reached_before_upturn" if len(rises) else "right_censored"
            result[name+"_GeV"] = None
            result["crossings"][name] = {"status": "censored", "reason": reason,
                "fraction": fraction, "last_q_GeV": float(q[end])}
            missing[name] = reason
            continue
        k = candidates[0]
        value = q[k] + (threshold-j[k])*(q[k+1]-q[k])/(j[k+1]-j[k])
        result[name+"_GeV"] = float(value)
        result["crossings"][name] = {"status": "observed", "fraction": fraction,
            "bracket_indices": [k, k+1], "bracket_q_GeV": q[k:k+2].tolist(),
            "bracket_density": j[k:k+2].tolist()}
    a, b, c = (result[key+"_GeV"] for key in ("Q50", "Q10", "Q1"))
    result.update(B_mid=None, B_tail=None, K_tail=None)
    if a is not None and b is not None:
        if not a < b:
            raise ValueError("unordered 50/10 percent crossings")
        result["B_mid"] = float(np.log(5)/(b*b-a*a))
        if c is not None:
            if not b < c:
                raise ValueError("unordered 10/1 percent crossings")
            result["B_tail"] = float(np.log(10)/(c*c-b*b))
            result["K_tail"] = result["B_tail"]/result["B_mid"]
    if missing:
        result["status"] = "censored"
        result["censor_reason"] = "; ".join(f"{key}:{value}" for key, value in missing.items())
    result["censor_reasons"] = missing
    result["feature_valid"] = [result[name] is not None for name in FEATURE_NAMES]
    return result


def _feature_vector(feature):
    return np.array([np.nan if feature[name] is None else feature[name]
                     for name in FEATURE_NAMES], dtype=float)


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _count_batches(n_blocks, draws, seed, batch_size):
    """Replay the same counts regardless of case, channel, or batch size."""
    probabilities = np.full(n_blocks, 1./n_blocks)
    for first in range(0, draws, batch_size):
        count = min(batch_size, draws-first)
        weights = np.empty((count, n_blocks), dtype=float)
        for row, replicate in enumerate(range(first, first+count)):
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                [seed, n_blocks, replicate])))
            weights[row] = rng.multinomial(n_blocks, probabilities)/n_blocks
        yield first, weights


def _channel_statistics(point, values, reasons, deterministic):
    valid = np.isfinite(values)
    joint = valid.all(axis=1)
    surviving = values[joint]
    n_valid = int(joint.sum())
    all_valid = bool(joint.all() and all(point["feature_valid"]))
    covariance = None
    if n_valid >= 2:
        covariance = (np.zeros((len(FEATURE_NAMES), len(FEATURE_NAMES))) if deterministic else
                      np.cov(surviving, rowvar=False, ddof=1))
    se = None if covariance is None else np.sqrt(np.maximum(0, np.diag(covariance)))
    marginal_se = [float(0. if deterministic else values[:, i].std(ddof=1))
                   if valid[:, i].all() and point["feature_valid"][i] else None
                   for i in range(len(FEATURE_NAMES))]
    return {
        "point": point, "value": [point[name] for name in FEATURE_NAMES],
        "covariance": covariance.tolist() if all_valid and covariance is not None else None,
        "orientation_se": se.tolist() if all_valid and se is not None else None,
        "marginal_orientation_se": marginal_se,
        "conditional_covariance": covariance.tolist() if not all_valid and covariance is not None else None,
        "conditional_orientation_se": se.tolist() if not all_valid and se is not None else None,
        "covariance_scope": ("all_replicates" if all_valid else
                             "conditional_on_all_features_observed"),
        "valid_fraction": float(joint.mean()), "censored_fraction": float(1-joint.mean()),
        "feature_valid_fraction": valid.mean(axis=0).tolist(),
        "n_joint_valid": n_valid, "n_bootstrap": len(values),
        "censor_reason_counts": dict(sorted(reasons.items())),
        "bootstrap_quantiles_16_50_84": (np.quantile(surviving, [.16, .5, .84], axis=0).tolist()
                                          if all_valid else None),
        "conditional_bootstrap_quantiles_16_50_84": (
            np.quantile(surviving, [.16, .5, .84], axis=0).tolist()
            if not all_valid and n_valid else None),
    }


def summarize_samples(q, samples, *, bootstrap_draws=1024,
                      bootstrap_seed=20261024, bootstrap_block_size=32,
                      bootstrap_batch_size=32, orientation_bank_id=None):
    """Summarize a single case/prefix, preserving all-channel bootstrap pairing.

    ``samples`` has shape (orientation, b level, full/self, q).  The raw first
    axis must follow the campaign's shared ordered rotation bank.  All channels
    receive exactly the same resampling counts.  Call this function separately
    on prefixes with the same seed/count/ordering across cases to pair *within*
    each prefix; these plans do not estimate cross-prefix difference errors.

    When n > block_size, n must divide evenly into contiguous blocks.  For
    n <= block_size, use individual orientation pairs rather than a single
    block with fictitiously zero variance.  A singleton is interpreted as an
    explicitly deterministic spherical calculation and has zero orientation
    MC error, not zero systematic uncertainty.

    Returns ``channels[b_index][branch_index]`` with JSON-safe point statistics
    and a covariance ordered as FEATURE_NAMES. ``replicate_features``
    is a small (nb,2,nboot,nfeatures) array, with NaN for censored features, for NPZ
    storage and paired comparisons.  ``replicate_valid`` records its mask.
    A censored replicate suppresses the unconditional covariance/SE; any
    surviving-replicate covariance is explicitly conditional and uses one
    common complete-case mask, never pairwise deletion.
    ``marginal_orientation_se`` remains available for an individual feature
    only when its point and every replicate are observed, even if another
    feature censors the joint covariance.
    """
    q = _radial_grid(q)
    x = np.asarray(samples)
    if (x.ndim != 4 or x.shape[2:] != (2, len(q))
            or not x.shape[0] or not x.shape[1] or not np.issubdtype(x.dtype, np.number)
            or np.iscomplexobj(x)):
        raise ValueError("samples must be real (orientation, b level, 2 branches, q) arrays")
    draws = _positive_integer(bootstrap_draws, "bootstrap_draws", 2)
    seed = _positive_integer(bootstrap_seed, "bootstrap_seed", 0)
    block_size = _positive_integer(bootstrap_block_size, "bootstrap_block_size")
    batch_size = _positive_integer(bootstrap_batch_size, "bootstrap_batch_size")
    n, nb, _, nq = x.shape
    effective = 1 if n <= block_size else block_size
    if n % effective:
        raise ValueError("n must be divisible by bootstrap_block_size; no orientations are dropped")
    for first in range(0, n, max(32, effective)):
        part = x[first:first+max(32, effective)]
        if not np.isfinite(part).all() or np.any(part < 0):
            raise ValueError("raw A0 samples must be finite and nonnegative")
    mean = x.mean(axis=0, dtype=np.float64)
    points = [[extract_tail_features(q, mean[level, branch]) for branch in range(2)]
              for level in range(nb)]
    n_blocks = n//effective
    blocks = x.reshape(n_blocks, effective, nb, 2, nq).mean(axis=1, dtype=np.float64)
    plan = {"algorithm": _PLAN_ALGORITHM, "seed": seed, "draws": draws,
            "n_orientations": n, "effective_block_size": effective, "n_blocks": n_blocks}
    plan_id = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    values = np.empty((nb, 2, draws, len(FEATURE_NAMES)), dtype=float)
    reasons = [[Counter() for _ in range(2)] for _ in range(nb)]
    if n == 1:
        for level in range(nb):
            for branch in range(2):
                values[level, branch] = _feature_vector(points[level][branch])
                if points[level][branch]["censor_reason"]:
                    reasons[level][branch][points[level][branch]["censor_reason"]] = draws
    else:
        flattened = blocks.reshape(n_blocks, -1)
        for first, weights in _count_batches(n_blocks, draws, seed, batch_size):
            profiles = (weights@flattened).reshape(len(weights), nb, 2, nq)
            for offset, profile in enumerate(profiles):
                for level in range(nb):
                    for branch in range(2):
                        feature = extract_tail_features(q, profile[level, branch])
                        values[level, branch, first+offset] = _feature_vector(feature)
                        if feature["censor_reason"]:
                            reasons[level][branch][feature["censor_reason"]] += 1
    channels = [[_channel_statistics(points[level][branch], values[level, branch],
                                    reasons[level][branch], n == 1)
                 for branch in range(2)] for level in range(nb)]
    return {
        "feature_names": list(FEATURE_NAMES), "branch_names": list(BRANCH_NAMES),
        "n_orientations": n, "q": q.copy(), "mean_A0": mean,
        "channels": channels, "replicate_features": values,
        "replicate_valid": np.isfinite(values),
        "bootstrap": {**plan, "requested_block_size": block_size,
            "resampling_plan_id": plan_id, "orientation_bank_id": orientation_bank_id,
            "method": ("deterministic" if n == 1 else
                       "orientation_pair_bootstrap" if effective == 1 else
                       "nonoverlapping_equal_block_bootstrap"),
            "approximation": "block approximation to orientation MC; not an event bootstrap",
            "pairing_requirement": "same ordered rotation bank AND matching resampling_plan_id; deterministic references are constant",
            "prefix_scope": "cross-case pairing within each prefix only; no cross-prefix difference covariance",
        },
        "uncertainty_scope": "orientation integration only; no q-grid, b/y quadrature, detector, event-count or model uncertainty",
    }
