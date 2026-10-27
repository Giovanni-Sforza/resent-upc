"""Measured same-grid closure certificates for the separate fast scan engine.

Certificates bind the scientific reference, every implementation source, actual
runtime environment, and each tested orientation/rapidity batch combination.
They are implementation checks, not quadrature or physical convergence tests.
"""
from __future__ import annotations

from copy import deepcopy
import gc
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import yaml

from .precision_campaign import (
    GRID_KEYS, OutputLock, _case_rotations, _digest_arrays, atomic_json,
    configure_runtime, rotation_bank, sha256,
)
from .shape_scan_analysis import FEATURE_NAMES, extract_tail_features
from .shape_scan_campaign import build_refiner as build_reference
from .shape_scan_physics import resolve_case

SCHEMA = "shape_scan_fast_validation_v1"
# Per orientation / b level / branch, with a separate peak for every channel.
# The peak-scaled floor handles destructive zeros without dividing by zero.
RTOL = 2e-10
PEAK_ATOL = 2e-12
FEATURE_RTOL = 5e-9


def representative_cases(cfg):
    """Smallest and largest deformations of every registered density variant."""
    chosen = {}
    for variant in cfg["scan"]["density_variants"]:
        cases = [c for c in cfg["study"]["cases"] if c["density_id"] == variant["id"]]
        key = lambda c: (abs(c["beta2"]) + abs(c["beta3"]), abs(c["beta2"]), abs(c["beta3"]))
        for case in (min(cases, key=key), max(cases, key=key)):
            chosen[case["name"]] = case
    return list(chosen.values())


def benchmark_case(cfg):
    cases = cfg["study"]["cases"]
    halo = [c for c in cases if c["density_type"] == "halo"]
    return max(halo or cases, key=lambda c: (abs(c["beta2"]), abs(c["beta3"])))


def build_fast(cfg, case):
    from .shape_scan_fast_forward import ShapeScanFastRefiner
    kernel_cfg, kernel_case, _ = resolve_case(cfg, case)
    return ShapeScanFastRefiner(kernel_cfg, kernel_case,
        device=cfg["runtime"]["device"], dtype=cfg["runtime"]["dtype"], fft_mode="separate")


def reference_payload(payload, qmax):
    indices = np.flatnonzero(payload["radii"] <= qmax)
    values = payload["b_level_coefficients"][:, :, :, 0, :][:, :, :, indices]
    return {**{key: payload[key] for key in GRID_KEYS},
            "stored_indices": indices, "stored_q": payload["radii"][indices],
            "A0": values.transpose(1, 0, 2, 3), "branch_names": payload["branch_names"]}


def compare_payloads(reference, actual):
    grid_equal = all(key in actual and np.array_equal(reference[key], actual[key])
                     for key in (*GRID_KEYS, "stored_indices", "stored_q"))
    a, b = np.asarray(reference["A0"]), np.asarray(actual.get("A0", []))
    valid = (grid_equal and a.shape == b.shape and a.ndim == 4
             and tuple(actual.get("branch_names", ())) == ("full", "self")
             and np.isfinite(b).all() and np.all(b >= 0))
    result = {"passed": False, "grid_equal": grid_equal,
              "shape_finite_nonnegative": bool(valid), "rtol": RTOL,
              "atol_per_channel_peak_fraction": PEAK_ATOL}
    if not valid:
        return result
    peak = np.max(np.abs(a), axis=-1, keepdims=True)
    difference = np.abs(b-a)
    allowance = RTOL*np.abs(a) + PEAK_ATOL*peak
    within = difference <= allowance
    ratio = np.divide(difference, peak, out=np.zeros_like(difference), where=peak > 0)
    result.update(max_absolute_error=float(difference.max()),
                  max_error_over_channel_peak=float(ratio.max()),
                  failed_values=int(np.count_nonzero(~within)),
                  compared_values=int(a.size))
    features = []
    amean, bmean = a.mean(axis=0), b.mean(axis=0)
    for level in range(a.shape[1]):
        for branch in range(2):
            if not amean[level, branch].max() > 0:
                features.append({"level": level, "branch": branch,
                                 "passed": bool(np.array_equal(amean[level, branch], bmean[level, branch])),
                                 "reason": "zero_channel"})
                continue
            expected = extract_tail_features(reference["stored_q"], amean[level, branch])
            observed = extract_tail_features(reference["stored_q"], bmean[level, branch])
            same = (expected["status"] == observed["status"]
                    and expected["censor_reasons"] == observed["censor_reasons"]
                    and expected["peak_node_index"] == observed["peak_node_index"])
            comparisons = {}
            for key in ("Q50_GeV", "Q1_GeV", *FEATURE_NAMES):
                x, y = expected[key], observed[key]
                ok = x is y if x is None or y is None else bool(np.isclose(x, y, rtol=FEATURE_RTOL, atol=1e-12))
                comparisons[key] = {"reference": x, "fast": y, "passed": ok}
                same = same and ok
            features.append({"level": level, "branch": branch, "passed": bool(same),
                             "features": comparisons})
    result["feature_checks"] = features
    result["passed"] = bool(within.all() and all(item["passed"] for item in features))
    return result


def _sync(env):
    if env["device_type"] == "cuda":
        torch.cuda.synchronize(env["device"])


def _release(env):
    gc.collect()
    if env["device_type"] == "cuda":
        torch.cuda.empty_cache()


def _measure(kernel, rotations, env, warmup, repeats):
    elapsed, allocated, reserved, payload = [], 0, 0, None
    for index in range(warmup + repeats):
        _sync(env)
        started = time.perf_counter()
        payload = kernel.calculate(*rotations, return_nodes=False)
        _sync(env)
        duration = time.perf_counter() - started
        if index >= warmup:
            elapsed.append(duration)
        allocated = max(allocated, int(payload.get("peak_memory_allocated_bytes", 0)))
        reserved = max(reserved, int(payload.get("peak_memory_reserved_bytes", 0)))
    median = statistics.median(elapsed)
    return payload, {"elapsed_seconds": elapsed, "median_seconds": median,
        "pairs_per_second": len(rotations[0])/median,
        "peak_memory_allocated_bytes": allocated, "peak_memory_reserved_bytes": reserved,
        "warmup_calls_excluded": warmup, "timed_repeats": repeats}


def _reference_prefix(cfg, case, banks, count, batch):
    kernel, _ = build_reference(cfg, case)
    blocks, first = [], None
    try:
        for start in range(0, count, batch):
            rotations = _case_rotations(case, banks, list(range(start, min(count, start+batch))))
            value = reference_payload(kernel.calculate(*rotations), cfg["study"]["qmax"])
            blocks.append(value["A0"])
            if first is None:
                first = value
        return {**first, "A0": np.concatenate(blocks)}
    finally:
        del kernel


def run_benchmark(cfg, output, *, batches=(8, 16, 32), y_batches=(1, 5),
                  warmup=1, repeats=2, validation_pairs=2, max_memory_fraction=.8):
    from .shape_scan_fast_campaign import code_hashes, scientific_reference_fingerprint
    if cfg["runtime"]["dtype"] != "float64":
        raise ValueError("The fast production path currently requires float64")
    batches, y_batches = list(batches), list(y_batches)
    if (not batches or not y_batches or any(type(n) is not int or n < 1 for n in batches+y_batches)
            or len(set(batches)) != len(batches) or len(set(y_batches)) != len(y_batches)
            or max(y_batches) > cfg["study"]["n_y"]):
        raise ValueError("Unique positive batches and y_batches within n_y are required")
    if (type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1
            or type(validation_pairs) is not int or validation_pairs < 2
            or not 0 < max_memory_fraction <= 1):
        raise ValueError("Require warmup >= 0, repeats >= 1, validation_pairs >= 2 and valid memory fraction")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Benchmark output must be new or empty")
    reference_cfg = deepcopy(cfg)
    reference_cfg["study"]["y_batch_size"] = 1
    reference_batch = min(8, max(batches))
    with OutputLock(output):
        if any(p.name != ".precision.lock" for p in output.iterdir()):
            raise ValueError("Benchmark output is no longer empty")
        env = configure_runtime(cfg["runtime"])
        sources = code_hashes()
        banks, bank_meta = rotation_bank(cfg["orientations"], max(max(batches), validation_pairs))
        report = {"schema": SCHEMA, "status": "running", "config": deepcopy(cfg),
            "reference_fingerprint": scientific_reference_fingerprint(cfg), "code_sha256": sources,
            "environment": env, "orientation_bank": bank_meta,
            "rotation_prefix_sha256": _digest_arrays({"r1": banks[0], "r2": banks[1]}),
            "request": {"batches": batches, "y_batches": y_batches, "warmup": warmup,
                "repeats": repeats, "validation_pairs": validation_pairs,
                "max_memory_fraction": max_memory_fraction},
            "representative_checks": [], "configurations": [], "recommendation": None,
            "scope": "Same-grid implementation closure and warmed synchronized host-return timing. "
                     "Not quadrature convergence or a whole-campaign wall-time measurement."}
        path = output/"benchmark.json"
        (output/"effective_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        atomic_json(path, report)
        try:
            # Include spherical cancellation and deformed none/skin/halo. Test
            # every requested rapidity schedule on the same reference samples.
            for case in representative_cases(cfg):
                count = 1 if case["beta2"] == case["beta3"] == 0 else validation_pairs
                expected = _reference_prefix(reference_cfg, case, banks, count, reference_batch)
                _release(env)
                rotations = _case_rotations(case, banks, list(range(count)))
                for y_batch in y_batches:
                    local = deepcopy(cfg); local["study"]["y_batch_size"] = y_batch
                    kernel = build_fast(local, case)
                    try:
                        actual = kernel.calculate(*rotations)
                    finally:
                        del kernel
                    check = compare_payloads(expected, actual)
                    report["representative_checks"].append(
                        {"case": case["name"], "y_batch_size": y_batch, "pairs": count, **check})
                    atomic_json(path, report)
                    print(f"closure {case['name']} y_batch={y_batch}: {check['passed']}", flush=True)
                    if not check["passed"]:
                        raise ValueError("Reference/fast representative closure failed")
                    _release(env)
            case = benchmark_case(cfg)
            report["benchmark_case"] = case
            expected = _reference_prefix(reference_cfg, case, banks, max(batches), reference_batch)
            _release(env)
            kernel, _ = build_reference(reference_cfg, case)
            try:
                _, baseline = _measure(kernel, _case_rotations(case, banks, list(range(reference_batch))),
                                       env, warmup, repeats)
            finally:
                del kernel
            report["reference_timing"] = {"batch_size": reference_batch, "y_batch_size": 1, **baseline}
            _release(env)
            for batch, y_batch in sorted(((b, y) for b in batches for y in y_batches), key=lambda p: p[0]*p[1]):
                local = deepcopy(cfg); local["study"]["y_batch_size"] = y_batch
                row = {"batch_size": batch, "y_batch_size": y_batch, "passed": False}
                kernel = None
                try:
                    kernel = build_fast(local, case)
                    actual, timing = _measure(kernel, _case_rotations(case, banks, list(range(batch))),
                                              env, warmup, repeats)
                    check = compare_payloads({**expected, "A0": expected["A0"][:batch]}, actual)
                    memory_ok = (env["device_type"] != "cuda" or timing["peak_memory_reserved_bytes"]
                                 <= max_memory_fraction*env["device_total_memory_bytes"])
                    row.update(status="ok" if check["passed"] else "closure_failed", passed=check["passed"],
                               closure=check, memory_eligible=bool(memory_ok), **timing)
                    row["speedup_vs_reference"] = row["pairs_per_second"]/baseline["pairs_per_second"]
                    row["kernel_metadata"] = {key: actual[key] for key in (
                        "engine", "fourier_backend", "native_fft_vertices", "evaluated_intensity_vertices",
                        "native_dft_side", "czt_crop_side", "projection_vertices") if key in actual}
                except (torch.OutOfMemoryError, MemoryError) as error:
                    row.update(status="oom", error=str(error), memory_eligible=False)
                finally:
                    del kernel
                    _release(env)
                report["configurations"].append(row)
                atomic_json(path, report)
                print(f"batch={batch}, y_batch={y_batch}: {row['status']}"
                      + (f", {row['pairs_per_second']:.4g} pairs/s, {row['speedup_vs_reference']:.3g}x"
                         if row["status"] == "ok" else ""), flush=True)
                if row["status"] == "closure_failed":
                    raise ValueError("Reference/fast batch closure failed")
            eligible = [r for r in report["configurations"] if r["status"] == "ok" and r["memory_eligible"]]
            if not eligible:
                raise ValueError("No measured configuration passed within the memory limit")
            best = max(eligible, key=lambda r: r["pairs_per_second"])
            report["recommendation"] = {key: best[key] for key in (
                "batch_size", "y_batch_size", "pairs_per_second", "speedup_vs_reference",
                "peak_memory_reserved_bytes")}
            if sources != code_hashes() or report["reference_fingerprint"] != scientific_reference_fingerprint(cfg):
                raise ValueError("Source or scientific reference changed during benchmark")
            report["status"] = "passed"
            atomic_json(path, report)
            return report
        except BaseException as error:
            report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error_type=type(error).__name__, error=str(error), recommendation=None)
            atomic_json(path, report)
            raise


def validate_report(path, cfg):
    """Reject stale, wrong-device, different-grid or unmeasured certificates."""
    from .shape_scan_fast_campaign import code_hashes, scientific_reference_fingerprint
    report = json.loads(Path(path).read_text())
    if report.get("schema") != SCHEMA or report.get("status") != "passed":
        raise ValueError("A passed shape-scan fast validation report is required")
    if report.get("reference_fingerprint") != scientific_reference_fingerprint(cfg):
        raise ValueError("Validation scientific reference differs from this run")
    if report.get("code_sha256") != code_hashes():
        raise ValueError("Validation source hashes differ; rerun the fast benchmark")
    batch, y_batch = cfg["runtime"]["batch_size"], cfg["study"].get("y_batch_size", 1)
    rows = [r for r in report.get("configurations", [])
            if r.get("batch_size") == batch and r.get("y_batch_size") == y_batch
            and r.get("status") == "ok" and r.get("passed") and r.get("memory_eligible")
            and r.get("closure", {}).get("passed")]
    if not rows:
        raise ValueError("Requested batch_size/y_batch_size lacks a passed eligible measurement")
    checks = report.get("representative_checks", [])
    for case in representative_cases(cfg):
        minimum = 1 if case["beta2"] == case["beta3"] == 0 else 2
        if not any(c.get("case") == case["name"] and c.get("y_batch_size") == y_batch
                   and c.get("pairs", 0) >= minimum and c.get("passed") for c in checks):
            raise ValueError("Missing representative-case closure for the requested rapidity batch")
    env = configure_runtime(cfg["runtime"])
    keys = ["device_type", "dtype", "numpy", "torch", "cuda_build"]
    if env["device_type"] == "cuda":
        keys += ["device_name", "device_total_memory_bytes", "device_capability"]
    if any(report.get("environment", {}).get(key) != env.get(key) for key in keys):
        raise ValueError("Validation runtime/device differs; benchmark on the production environment")
    if cfg["runtime"]["checkpoint_size"] % batch:
        raise ValueError("checkpoint_size must be a multiple of the validated batch_size")
    return report
