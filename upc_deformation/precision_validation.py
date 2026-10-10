"""Independent numerical diagnostics for the explicit-precision panel-d kernel.

The comparison projects the returned coefficients through the established
Cartesian pixel sampler.  Cross-device checks compare like precision, whereas
FP32/FP64 differences are reported separately without reusing an acceptance
threshold to declare the precision change harmless.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any, Callable

import numpy as np

from .c2_convergence import project_orientation_moments, ratio_statistics


HIGH_PT_C2_ATOL = 1e-6
PEAK_RELATIVE_TOLERANCES = {"float32": 1e-4, "float64": 1e-8}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def scientific_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(jsonable(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def rotation_pairs(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproducible independent Haar rotations shared by every device/dtype."""
    if count < 1:
        raise ValueError("at least one orientation pair is required")
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(2):
        quat = rng.normal(size=(count, 4))
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        w, x, y, z = quat.T
        result.append(np.stack((
            1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
            2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
            2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=1).reshape(count, 3, 3))
    return result[0], result[1]


def tiny_configuration(config: dict[str, Any]) -> dict[str, Any]:
    """Coarse, genuine forward smoke configuration, never a production proxy."""
    cfg = copy.deepcopy(config)
    cfg["physics"]["grid_parameters"] = {
        "WS": {"N": 16, "L": 24.}, "T_prim": {"N": 16, "L": 24.},
        "output": {"N": 80, "L": 160.}}
    cfg["physics"].setdefault("output_settings", {}).update(
        safe_padding_factor=2, p_limit_q=.17, target_size_q=49)
    cfg["study"].update(b_min_fm=12., b_max_fm=60., b_step_fm=12., b_nodes=5,
                        n_y=3, ymax=1., y_batch_size=3, n_radial=65, n_phi=16)
    return cfg


def validate_payload(payload: dict[str, Any]) -> None:
    coeff = np.asarray(payload["coefficients"])
    levels = np.asarray(payload["b_level_coefficients"])
    radii = np.asarray(payload["radii"])
    steps = np.asarray(payload["b_steps_fm"])
    if (coeff.ndim != 3 or coeff.shape[1:] != (3, len(radii)) or coeff.shape[0] < 1
            or levels.shape != (len(steps), *coeff.shape)
            or not np.isfinite(coeff).all() or not np.isfinite(levels).all()
            or not len(steps) or np.any(steps <= 0) or np.any(np.diff(steps) <= 0)):
        raise ValueError("invalid per-orientation coefficients or b levels")
    if not np.array_equal(coeff, levels[0]):
        raise ValueError("main coefficients must equal the finest retained b level")


def merge_serial_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve order when comparing a batch to the same pairs run one at a time."""
    if not payloads:
        raise ValueError("no serial payloads")
    first = payloads[0]
    for payload in payloads:
        validate_payload(payload)
        if len(payload["coefficients"]) != 1:
            raise ValueError("serial payloads must contain exactly one orientation")
        for key in ("axis", "radii", "b_steps_fm"):
            if not np.array_equal(payload[key], first[key]):
                raise ValueError(f"serial payloads disagree on {key}")
    out = dict(first)
    out["coefficients"] = np.concatenate([p["coefficients"] for p in payloads], axis=0)
    out["b_level_coefficients"] = np.concatenate([p["b_level_coefficients"] for p in payloads], axis=1)
    out["n_orientations"] = len(payloads)
    for key in ("peak_memory_allocated_bytes", "peak_memory_reserved_bytes"):
        out[key] = max(int(p.get(key, 0)) for p in payloads)
    return out


def project_payload(payload: dict[str, Any], q: np.ndarray, angles: np.ndarray) -> dict[str, Any]:
    """Use the original direct Cartesian projection independently of new analyzers."""
    validate_payload(payload)
    q = np.asarray(q, float)
    if q[-1] < .17 or q[0] < 0:
        raise ValueError("validation must cover the registered high-pT range through 0.17 GeV")
    records = []
    for step, coeff in zip(payload["b_steps_fm"], payload["b_level_coefficients"]):
        moments = project_orientation_moments(coeff, payload["radii"], payload["axis"], q, angles)
        numerator, denominator = moments["orientation_C2_numerator"], moments["orientation_denominator"]
        pooled = ratio_statistics(numerator, denominator)
        records.append({
            "b_step_fm": float(step), "C2": pooled["value"],
            "orientation_C2": numerator/denominator,
            "mean_raw_image": moments["mean_raw_image"],
            "minimum_image": float(moments["minimum_orientation_image"]),
        })
    return {"q": q, "levels": records}


def _peak_relative(actual, expected):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("compared arrays must have equal shapes and finite values")
    difference = float(np.max(np.abs(actual-expected)))
    peak = float(np.max(np.abs(expected)))
    if peak <= 0:
        raise ValueError("reference must have nonzero magnitude")
    return difference, difference/peak


def compare_payloads(actual: dict[str, Any], expected: dict[str, Any],
                     actual_projection: dict[str, Any], expected_projection: dict[str, Any],
                     *, precision: str, comparison_kind: str,
                     high_pt_min: float = .10) -> dict[str, Any]:
    """Compare like-precision implementations or explicitly ungated precision changes."""
    if precision not in PEAK_RELATIVE_TOLERANCES:
        raise ValueError("precision must be float32 or float64")
    kinds = ("same_precision_cross_device", "same_precision_batch", "precision_diagnostic")
    if comparison_kind not in kinds:
        raise ValueError("unknown comparison kind")
    validate_payload(actual)
    validate_payload(expected)
    if not np.array_equal(actual["b_steps_fm"], expected["b_steps_fm"]):
        raise ValueError("comparisons require identical retained b steps")
    q = np.asarray(expected_projection["q"])
    if not np.array_equal(q, actual_projection["q"]):
        raise ValueError("comparisons require identical projected momenta")
    high = (q >= high_pt_min) & (q <= .17)
    if not np.any(high):
        raise ValueError("no samples inside the high-pT gate")
    if len(actual_projection["levels"]) != len(expected_projection["levels"]) or len(actual_projection["levels"]) != len(actual["b_steps_fm"]):
        raise ValueError("projection b-level mismatch")
    rows = []
    for i, (a, e) in enumerate(zip(actual_projection["levels"], expected_projection["levels"])):
        absolute, relative = _peak_relative(actual["b_level_coefficients"][i], expected["b_level_coefficients"][i])
        image_absolute, image_relative = _peak_relative(a["mean_raw_image"], e["mean_raw_image"])
        pooled = np.abs(np.asarray(a["C2"])-e["C2"])
        individual = np.abs(np.asarray(a["orientation_C2"])-e["orientation_C2"])
        if pooled.shape != q.shape or individual.ndim != 2 or individual.shape[-1] != len(q):
            raise ValueError("invalid moment comparison shapes")
        if not np.isfinite(pooled).all() or not np.isfinite(individual).all():
            raise ValueError("nonfinite moment comparison")
        rows.append({
            "b_step_fm": float(actual["b_steps_fm"][i]),
            "coefficient_max_absolute": absolute, "coefficient_difference_over_reference_peak": relative,
            "image_max_absolute": image_absolute, "image_difference_over_reference_peak": image_relative,
            "C2_mean_max_absolute_all_q": float(np.max(pooled)),
            "C2_mean_max_absolute_high_pT": float(np.max(pooled[high])),
            "C2_individual_max_absolute_high_pT": float(np.max(individual[:, high])),
        })
    gated = comparison_kind != "precision_diagnostic"
    tol = PEAK_RELATIVE_TOLERANCES[precision]
    passed = (all(r["C2_mean_max_absolute_high_pT"] < HIGH_PT_C2_ATOL
                  and r["C2_individual_max_absolute_high_pT"] < HIGH_PT_C2_ATOL
                  and r["coefficient_difference_over_reference_peak"] < tol
                  and r["image_difference_over_reference_peak"] < tol for r in rows) if gated else None)
    return {"comparison_kind": comparison_kind, "precision": precision,
            "status": ("passed" if passed else "failed") if gated else "diagnostic_only",
            "passed": passed, "acceptance_gate_applied": gated,
            "high_pT_interval_GeV": [float(high_pt_min), .17],
            "high_pT_C2_absolute_tolerance": HIGH_PT_C2_ATOL if gated else None,
            "coefficient_and_image_peak_relative_tolerance": tol if gated else None,
            "levels": rows,
            "scope": ("Same configuration and precision; numerical implementation/batch closure only."
                      if gated else "FP32-versus-FP64 diagnostic, never substituted for the cross-device acceptance test.")}


def benchmark(run_once: Callable[[], dict[str, Any]], synchronize: Callable[[], None],
              *, warmup: int, repeats: int, pairs: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Time only post-warmup calls; caller supplies explicit device synchronization."""
    if warmup < 1 or repeats < 1 or pairs < 1:
        raise ValueError("positive warmup, repeats, and pairs are required")
    for _ in range(warmup):
        run_once()
        synchronize()
    seconds, allocated, reserved = [], [], []
    result = None
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        result = run_once()
        synchronize()
        seconds.append(time.perf_counter()-start)
        allocated.append(int(result.get("peak_memory_allocated_bytes", 0)))
        reserved.append(int(result.get("peak_memory_reserved_bytes", 0)))
    return result, {"warmup_calls_excluded": warmup, "timed_repeats": repeats,
                    "pairs_per_timed_call": pairs, "elapsed_seconds": seconds,
                    "median_seconds": float(np.median(seconds)),
                    "minimum_seconds": float(np.min(seconds)),
                    "pairs_per_second_at_median": float(pairs/np.median(seconds)),
                    "peak_memory_allocated_bytes": max(allocated),
                    "peak_memory_reserved_bytes": max(reserved),
                    "timing_scope": "Synchronized wall time including returned CPU arrays; model construction excluded."}
