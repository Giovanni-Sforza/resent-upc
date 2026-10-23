#!/usr/bin/env python3
"""Benchmark explicitly requested batches on the registered shape-scan grid.

This writes timing/provenance reports, never production checkpoints. A
recommendation applies only to successful configurations measured in this run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import yaml

from upc_deformation.precision_campaign import (
    OutputLock, _case_rotations, _digest_arrays, atomic_json,
    configure_runtime, rotation_bank, sha256,
)

CODE_FILES = (
    "scripts/benchmark_shape_scan.py", "upc_deformation/shape_scan_campaign.py",
    "upc_deformation/shape_scan_physics.py", "upc_deformation/diffraction_forward.py",
    "upc_deformation/precision_campaign.py", "upc_deformation/precision_forward.py",
    "upc_deformation/observables.py", "upc_deformation/forward.py",
    "upc_deformation/source_amplitudes.py", "upc_deformation/decay.py",
    "upc_deformation/vendor/upc_forward.py",
)


def code_hashes():
    return {name: sha256(ROOT / name) for name in CODE_FILES}


def resolve_case(cfg, case):
    # The benchmark CLI remains importable before the scan package is installed.
    from upc_deformation.shape_scan_physics import resolve_case as resolve
    return resolve(cfg, case)


def build_refiner(cfg, case):
    from upc_deformation.diffraction_forward import DiffractionRefiner
    runtime = cfg["runtime"]
    return DiffractionRefiner(cfg, case, device=runtime["device"],
                             dtype=runtime["dtype"], fft_mode=runtime["fft_mode"])


def select_case(cases, name=None):
    if not cases:
        raise ValueError("No registered scan cases")
    if name is not None:
        matches = [case for case in cases if case["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one registered case named {name!r}")
        return deepcopy(matches[0])
    maxima = {key: max(abs(float(case[key])) for case in cases)
              for key in ("beta2", "beta3")}
    matches = [case for case in cases
               if case.get("density_type", case.get("density_variant", {}).get("type")) == "halo"
               and all(abs(float(case[key])) == maxima[key] for key in maxima)]
    if not matches:
        raise ValueError("No halo case has both maximum |beta2| and |beta3|; specify --case")
    # Registration order chooses the representative density variant explicitly;
    # different parameterizations need not have comparable strength coordinates.
    return deepcopy(matches[0])


def _positive_ints(values, name):
    values = list(values)
    if not values or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in values):
        raise ValueError(f"{name} must be a nonempty list of positive integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not repeat a configuration")
    return values


def _sync(environment):
    if environment["device_type"] == "cuda":
        torch.cuda.synchronize(environment["device"])


def _memory(environment):
    if environment["device_type"] != "cuda":
        return {"peak_memory_allocated_bytes": 0, "peak_memory_reserved_bytes": 0}
    device = environment["device"]
    return {"peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}


def _release(environment):
    gc.collect()
    if environment["device_type"] == "cuda":
        torch.cuda.empty_cache()


def _validate_payload(payload, pairs, nr):
    coefficients = np.asarray(payload["coefficients"])
    levels = np.asarray(payload["b_level_coefficients"])
    if (tuple(payload.get("branch_names", ())) != ("full", "self")
            or coefficients.shape != (pairs, 2, 3, nr)
            or levels.shape != (len(payload["b_steps_fm"]), pairs, 2, 3, nr)
            or not np.isfinite(levels).all()
            or not np.array_equal(coefficients, levels[0])
            or np.any(levels[:, :, :, 0] < 0)):
        raise ValueError("Invalid paired full/self benchmark payload")


def measure_configuration(cfg, case, banks, environment, batch, y_batch, warmup, repeats):
    """Time one scheme; arrays are discarded after shape/finite-value checks."""
    local = deepcopy(cfg)
    local["runtime"]["batch_size"] = batch
    local["study"]["y_batch_size"] = y_batch
    rotations = _case_rotations(case, banks, list(range(batch)))
    kernel = None
    try:
        if environment["device_type"] == "cuda":
            torch.cuda.reset_peak_memory_stats(environment["device"])
        before = time.perf_counter()
        kernel = build_refiner(local, case)
        _sync(environment)
        construction = time.perf_counter() - before
        peaks = _memory(environment)
        elapsed = []
        for call in range(warmup + repeats):
            _sync(environment)
            before = time.perf_counter()
            payload = kernel.calculate(*rotations, return_nodes=False)
            _sync(environment)
            duration = time.perf_counter() - before
            _validate_payload(payload, batch, local["study"]["n_radial"])
            measured = _memory(environment)
            for key in peaks:
                peaks[key] = max(peaks[key], measured[key], int(payload.get(key, 0)))
            del payload
            if call >= warmup:
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("Nonpositive or nonfinite benchmark duration")
                elapsed.append(duration)
        median = statistics.median(elapsed)
        return {"construction_seconds": construction, "elapsed_seconds": elapsed,
                "median_seconds": median, "pairs_per_second": batch / median,
                "warmup_calls_excluded": warmup, "timed_repeats": repeats,
                "pairs_per_timed_call": batch, **peaks}
    finally:
        del kernel


def choose_recommendation(rows, environment, max_memory_fraction):
    cuda = environment["device_type"] == "cuda"
    total = environment.get("device_total_memory_bytes")
    if cuda and (not isinstance(total, (int, float)) or not math.isfinite(total) or total <= 0):
        raise ValueError("CUDA total memory is required for a memory-limited recommendation")
    eligible = [row for row in rows if row["status"] == "ok"
                and (not cuda or row["peak_memory_reserved_bytes"] <= max_memory_fraction * total)]
    if not eligible:
        return None
    best = max(eligible, key=lambda row: row["pairs_per_second"])
    return {key: best[key] for key in ("batch_size", "y_batch_size", "pairs_per_second",
                                       "peak_memory_allocated_bytes", "peak_memory_reserved_bytes")}


def run_benchmark(cfg, output, *, batches=(8, 16, 32, 64), y_batches=(1, 5),
                  case_name=None, max_memory_fraction=.8, warmup=1, repeats=1,
                  config_path=None):
    batches = _positive_ints(batches, "batches")
    y_batches = _positive_ints(y_batches, "y_batches")
    if any(y > cfg["study"]["n_y"] for y in y_batches):
        raise ValueError("y_batches must not exceed the registered n_y")
    if (not math.isfinite(max_memory_fraction) or not 0 < max_memory_fraction <= 1
            or not isinstance(warmup, int) or warmup < 0
            or not isinstance(repeats, int) or repeats < 1):
        raise ValueError("Require 0 < memory fraction <= 1, nonnegative warmup and positive repeats")
    if cfg["runtime"]["fft_mode"] != "separate":
        raise ValueError("The shape-scan benchmark requires separate full/self FFTs")
    case = select_case(cfg["study"]["cases"], case_name)
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Benchmark output must be new or empty; refusing to overwrite existing files")
    input_path = Path(config_path).resolve() if config_path is not None else None
    source_hashes = code_hashes()
    input_hash = sha256(input_path) if input_path else None
    with OutputLock(output):
        # Recheck after locking: another process may have finished between the
        # first emptiness check and acquiring the lock.
        if any(path.name != ".precision.lock" for path in output.iterdir()):
            raise ValueError("Benchmark output is no longer empty; refusing to overwrite existing files")
        effective = output / "effective_config.yaml"
        effective.write_text(yaml.safe_dump(deepcopy(cfg), sort_keys=False))
        report = {
            "schema": "shape_scan_benchmark_v1", "status": "running",
            "input_config": str(input_path) if input_path else None,
            "input_config_sha256": input_hash, "effective_config_sha256": sha256(effective),
            "code_sha256": source_hashes, "case": case,
            "default_case_selection": "first registered halo variant at both maximum absolute deformations",
            "request": {"batches": batches, "y_batches": y_batches, "warmup": warmup,
                        "repeats": repeats, "max_memory_fraction": max_memory_fraction},
            "config": deepcopy(cfg), "configurations": [], "recommendation": None,
            "timing_scope": "Synchronized kernel wall time including returned CPU arrays; construction and payload checks excluded. One warmup by default.",
            "scope": "Registered grid and one representative case only. No production outputs; no grid/physics convergence claim. Fastest eligible measured scheme, not an untested optimum for an 80 GB GPU.",
        }
        report_path = output / "benchmark.json"
        atomic_json(report_path, report)
        try:
            environment = configure_runtime(cfg["runtime"])
            report["environment"] = environment
            report["memory_limit_applied"] = environment["device_type"] == "cuda"
            if report["memory_limit_applied"]:
                choose_recommendation([], environment, max_memory_fraction)
            banks, bank_meta = rotation_bank(cfg["orientations"], max(batches))
            report["orientation_bank"] = bank_meta
            report["orientation_prefix_sha256"] = _digest_arrays(
                {f"rotation_bank_{i+1}": bank for i, bank in enumerate(banks)})
            kernel_cfg, kernel_case, density = resolve_case(deepcopy(cfg), case)
            report["resolved_density"] = density
            report["kernel_case"] = kernel_case
            report["resolved_physics"] = kernel_cfg["physics"]
            plans = sorted(((batch, y) for batch in batches for y in y_batches),
                           key=lambda pair: (pair[0] * pair[1], pair[0], pair[1]))
            oom_product = None
            for batch, y_batch in plans:
                row = {"batch_size": batch, "y_batch_size": y_batch,
                       "batch_times_y_batch": batch * y_batch}
                if oom_product is not None and batch * y_batch > oom_product:
                    row.update(status="skipped_after_oom", reason=f"Larger batch*y_batch than an OOM scheme ({oom_product}); not measured")
                else:
                    _release(environment)
                    try:
                        row.update(measure_configuration(kernel_cfg, kernel_case, banks, environment,
                                                         batch, y_batch, warmup, repeats))
                        row["status"] = "ok"
                    except Exception as error:
                        oom = isinstance(error, (torch.OutOfMemoryError, MemoryError)) or (
                            environment["device_type"] == "cuda" and "out of memory" in str(error).lower())
                        row.update(status="oom" if oom else "failed", error_type=type(error).__name__,
                                   error=str(error), **_memory(environment))
                        if oom:
                            oom_product = batch * y_batch
                    # Release again after the exception traceback has gone out of scope.
                    _release(environment)
                report["configurations"].append(row)
                atomic_json(report_path, report)
                print(f"batch={batch}, y_batch={y_batch}: {row['status']}"
                      + (f", {row['pairs_per_second']:.4g} pairs/s" if row["status"] == "ok" else ""), flush=True)
            if (source_hashes != code_hashes() or (input_path and input_hash != sha256(input_path))
                    or (bank_meta.get("import_sha256") and
                        sha256(cfg["orientations"]["import_bank"]) != bank_meta["import_sha256"])):
                raise ValueError("Benchmark input or source changed during execution")
            report["input_and_code_unchanged"] = True
            report["recommendation"] = choose_recommendation(report["configurations"], environment, max_memory_fraction)
            report["status"] = ("completed" if report["recommendation"] is not None
                                else "no_eligible_configuration")
            report["failed_configurations"] = sum(row["status"] == "failed" for row in report["configurations"])
            atomic_json(report_path, report)
            return report
        except BaseException as error:
            report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error_type=type(error).__name__, error=str(error), recommendation=None)
            atomic_json(report_path, report)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/shape_scan_a100.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--y-batches", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--case", dest="case_name")
    parser.add_argument("--max-memory-fraction", type=float, default=.8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    from upc_deformation.shape_scan_campaign import load_config
    cfg = load_config(args.config, {"device": args.device})
    report = run_benchmark(cfg, args.output, batches=args.batches, y_batches=args.y_batches,
                           case_name=args.case_name, max_memory_fraction=args.max_memory_fraction,
                           warmup=args.warmup, repeats=args.repeats, config_path=args.config)
    print(f"{report['status']}: {args.output / 'benchmark.json'}")
    if report["recommendation"]:
        best = report["recommendation"]
        print(f"Fastest eligible measured scheme: batch={best['batch_size']}, y_batch={best['y_batch_size']}")
    return 0 if report["recommendation"] is not None and not report["failed_configurations"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
