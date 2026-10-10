#!/usr/bin/env python3
"""Validate explicit CPU/CUDA precision with paired genuine forward calculations.

Example local smoke (does not validate CUDA or production resolution):
  python scripts/validate_panel_d_precision.py --device cpu --tiny --pairs 2

On a CUDA server, --device cuda runs both CPU and CUDA references at each
requested precision.  Unavailable CUDA is an explicit skip with nonzero exit;
the command never falls back to CPU and calls that a CUDA validation.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from upc_deformation.precision_validation import (
    benchmark, compare_payloads, jsonable, merge_serial_payloads,
    project_payload, rotation_pairs, scientific_fingerprint, tiny_configuration,
)


CODE_PATHS = (
    "scripts/validate_panel_d_precision.py", "upc_deformation/precision_validation.py",
    "upc_deformation/precision_forward.py", "upc_deformation/c2_convergence.py",
    "upc_deformation/decay.py", "upc_deformation/forward.py",
    "upc_deformation/observables.py", "upc_deformation/spectral_weights.py",
    "upc_deformation/vendor/upc_forward.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False,
                               allow_nan=False)+"\n")


@contextmanager
def disable_tf32():
    """Do not mix TF32 tensor-core arithmetic into the stated FP32 comparison."""
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _run_mode(config, case, device, dtype, r1, r2, args, q, angles, out):
    from upc_deformation.precision_forward import PrecisionPanelDRefiner
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    sync()
    before = time.perf_counter()
    model = PrecisionPanelDRefiner(config, case, device=device, dtype=dtype,
                                  fft_mode=args.fft_mode)
    sync()
    construction_seconds = time.perf_counter()-before
    batch, batch_perf = benchmark(lambda: model.calculate(r1, r2, return_nodes=False), sync,
                                   warmup=args.warmup, repeats=args.repeats, pairs=args.pairs)
    batch_projection = project_payload(batch, q, angles)
    serial_check = None
    serial_perf = None
    if args.pairs > 1:
        def run_serial():
            return merge_serial_payloads([model.calculate(r1[i:i+1], r2[i:i+1], return_nodes=False)
                                          for i in range(args.pairs)])
        serial, serial_perf = benchmark(run_serial, sync, warmup=args.warmup,
                                         repeats=args.repeats, pairs=args.pairs)
        serial_projection = project_payload(serial, q, angles)
        serial_check = compare_payloads(batch, serial, batch_projection, serial_projection,
                                         precision=dtype, comparison_kind="same_precision_batch")
    saved = {k: v for k, v in batch.items() if isinstance(v, np.ndarray)}
    saved.update(q=q, angles=angles,
                 C2_by_b_level=np.stack([r["C2"] for r in batch_projection["levels"]]),
                 individual_C2_by_b_level=np.stack([r["orientation_C2"] for r in batch_projection["levels"]]))
    np.savez_compressed(out/f"{device}_{dtype}.npz", **saved)
    description = {"device": device, "dtype": dtype, "pairs": args.pairs,
                   "construction_seconds": construction_seconds,
                   "cuda_peak_memory_measured": device == "cuda",
                   "batch": batch_perf, "serial": serial_perf,
                   "batch_consistency": serial_check if serial_check is not None else
                   {"status": "not_tested", "passed": None, "reason": "--pairs must exceed 1 to test batching"},
                   "b_steps_fm": batch["b_steps_fm"], "shape": list(batch["coefficients"].shape),
                   "minimum_reconstructed_intensity_by_b_level": [r["minimum_image"] for r in batch_projection["levels"]]}
    del model
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return batch, batch_projection, description


def _report(result, path):
    lines = ["# Precision/device validation", "", f"Status: `{result['status']}`.", "",
             result["scope"], "", "- The requested device is explicit; CUDA is never replaced by CPU.",
             "- Same-precision gates compare raw coefficients, Cartesian images and high-pT C2; FP32/FP64 changes are separate diagnostics.",
             "- C2 uses the existing direct Cartesian pixel projection, with no harmonic shortcut.",
             "- Timings exclude model construction and warmups, synchronize CUDA, and include exported CPU result arrays.",
             "- No detector validation, event-statistics claim, or b/space/y convergence certificate is supplied.", ""]
    if result.get("reason"):
        lines += [result["reason"], ""]
    for name, mode in result.get("modes", {}).items():
        perf = mode["batch"]
        memory_note = (f"peak allocated GPU memory: {perf['peak_memory_allocated_bytes']} bytes."
                       if mode["cuda_peak_memory_measured"] else "GPU peak memory was not measured in this CPU run.")
        lines += [f"## {name}", "", f"Batch median: {perf['median_seconds']:.6g} s for {mode['pairs']} pairs; "
                  + memory_note,
                  f"Batch consistency: `{mode['batch_consistency']['status']}`.", ""]
    for category in ("cross_device_same_precision", "precision_diagnostics"):
        for name, comparison in result.get(category, {}).items():
            lines += [f"## {category}: {name}", "", f"Status: `{comparison['status']}`.", "",
                      "| b step (fm) | max high-pT C2 difference, mean | max high-pT C2 difference, individual | coefficient relative peak error |",
                      "|---:|---:|---:|---:|"]
            for row in comparison["levels"]:
                lines.append(f"| {row['b_step_fm']:g} | {row['C2_mean_max_absolute_high_pT']:.6g} | {row['C2_individual_max_absolute_high_pT']:.6g} | {row['coefficient_difference_over_reference_peak']:.6g} |")
            lines.append("")
    lines += [f"CUDA validation passed: `{result.get('cuda_validation_passed', False)}`.", "",
              "Complete numerical settings, code/input hashes, paired rotations and raw coefficients accompany this report."]
    path.write_text("\n".join(lines)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/panel_d_precision_a100.yaml")
    parser.add_argument("--output", default="outputs/panel_d_precision_validation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64", "both"), default="both")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--case", default="beta2")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", "--loops", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fft-mode", choices=("combined", "separate"), default="combined")
    parser.add_argument("--b-step", type=float, help="Override the finest impact-parameter spacing in fm")
    parser.add_argument("--n-y", type=int, help="Override the number of rapidity quadrature nodes")
    parser.add_argument("--y-batch-size", type=int, help="Override rapidity chunk size")
    args = parser.parse_args(argv)
    if min(args.pairs, args.warmup, args.repeats, args.threads) < 1:
        parser.error("pairs, warmup, repeats and threads must be positive")
    source = (ROOT/args.config).resolve()
    config = yaml.safe_load(source.read_text())
    if not isinstance(config, dict) or "physics" not in config or "study" not in config:
        parser.error("config must contain complete physics and study mappings")
    cases = {c["name"]: c for c in config["study"]["cases"]}
    if args.case not in cases:
        parser.error(f"unknown case {args.case}")
    effective = tiny_configuration(config) if args.tiny else copy.deepcopy(config)
    if args.b_step is not None:
        span = float(effective["study"]["b_max_fm"])-float(effective["study"]["b_min_fm"])
        if args.b_step <= 0 or not np.isfinite(args.b_step):
            parser.error("--b-step must be finite and positive")
        intervals = int(round(span/args.b_step))
        if intervals < 1 or not np.isclose(intervals*args.b_step, span, rtol=1e-12, atol=1e-12):
            parser.error("--b-step must exactly divide the configured b window")
        effective["study"].update(b_step_fm=args.b_step, b_nodes=intervals+1)
    if args.n_y is not None:
        if args.n_y < 1:
            parser.error("--n-y must be positive")
        effective["study"]["n_y"] = args.n_y
    if args.y_batch_size is not None:
        if args.y_batch_size < 1:
            parser.error("--y-batch-size must be positive")
        effective["study"]["y_batch_size"] = args.y_batch_size
    identity = {"scientific_configuration": effective, "case": cases[args.case],
                "pairs": args.pairs, "seed": args.seed, "tiny": args.tiny,
                "device": args.device, "dtype": args.dtype, "warmup": args.warmup,
                "repeats": args.repeats, "threads": args.threads, "fft_mode": args.fft_mode}
    fingerprint = scientific_fingerprint(identity)
    out = (ROOT/args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"validation.json").exists():
        previous = json.loads((out/"validation.json").read_text())
        if previous.get("fingerprint") != fingerprint:
            raise ValueError("output belongs to a different validation configuration; select another --output")
    input_hash = sha(source)
    code_hashes = {name: sha(ROOT/name) for name in CODE_PATHS}
    cuda_available = bool(torch.cuda.is_available())
    result = {"status": "running", "fingerprint": fingerprint, "request": vars(args),
              "scope": ("Tiny coarse-grid real-forward smoke only; it does not validate production resolution."
                        if args.tiny else "Requested scientific configuration, limited paired numerical validation; no physical-convergence claim."),
              "input_config": str(source), "input_config_sha256": input_hash,
              "effective_config": effective, "code_sha256": code_hashes,
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "python_version": platform.python_version(), "platform": platform.platform(),
              "cuda_build_version": torch.version.cuda, "cuda_available": cuda_available,
              "cuda_validation_passed": False, "modes": {}, "cross_device_same_precision": {},
              "precision_diagnostics": {}, "tf32_enabled_during_validation": False,
              "amp_enabled": False,
              "rotations": {"method": "independent normalized Gaussian quaternions, common FP64 source bank", "seed": args.seed}}
    (out/"effective_config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False))
    if args.device == "cuda" and not cuda_available:
        result.update(status="skipped_cuda_unavailable", reason="CUDA was requested but unavailable; no CPU fallback or CUDA passing claim.")
        save_json(out/"validation.json", result)
        _report(result, out/"report.md")
        print(result["reason"], flush=True)
        return 2
    if cuda_available:
        result["cuda_device_name"] = torch.cuda.get_device_name()
        result["cuda_device_total_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(args.threads)
    r1, r2 = rotation_pairs(args.pairs, args.seed)
    np.savez_compressed(out/"rotation_pairs.npz", rotation_bank_1=r1, rotation_bank_2=r2)
    q = np.linspace(0, .17, 65 if args.tiny else 341)
    angles = np.arange(128 if args.tiny else 720)*2*np.pi/(128 if args.tiny else 720)
    result["pixel_projection"] = {"q_points": len(q), "angular_points": len(angles),
                                  "q_range_GeV": [0., .17], "high_pT_gate_GeV": [.10, .17]}
    precisions = ("float32", "float64") if args.dtype == "both" else (args.dtype,)
    devices = ("cpu", "cuda") if args.device == "cuda" else ("cpu",)
    payloads, projections = {}, {}
    try:
        with disable_tf32():
            for device in devices:
                for precision in precisions:
                    name = f"{device}_{precision}"
                    print(f"Run {name}: {args.pairs} paired orientations; tiny={args.tiny}", flush=True)
                    payloads[name], projections[name], result["modes"][name] = _run_mode(
                        effective, cases[args.case], device, precision, r1, r2, args, q, angles, out)
                    save_json(out/"validation.json", result)
            if args.device == "cuda":
                for precision in precisions:
                    gpu, cpu = f"cuda_{precision}", f"cpu_{precision}"
                    result["cross_device_same_precision"][precision] = compare_payloads(
                        payloads[gpu], payloads[cpu], projections[gpu], projections[cpu],
                        precision=precision, comparison_kind="same_precision_cross_device")
            if len(precisions) == 2:
                for device in devices:
                    low, high = f"{device}_float32", f"{device}_float64"
                    result["precision_diagnostics"][device] = compare_payloads(
                        payloads[low], payloads[high], projections[low], projections[high],
                        precision="float32", comparison_kind="precision_diagnostic")
        checks = [m["batch_consistency"]["passed"] for m in result["modes"].values()
                  if m["batch_consistency"]["passed"] is not None]
        checks += [c["passed"] for c in result["cross_device_same_precision"].values()]
        passed = all(checks) if checks else None
        result["executed_acceptance_gate_count"] = len(checks)
        result["all_executed_acceptance_gates_passed"] = passed
        result["cuda_validation_passed"] = bool(args.device == "cuda" and passed and result["cross_device_same_precision"])
        if passed is None:
            result["status"] = "cpu_forward_completed_no_acceptance_gates_cuda_not_tested"
        else:
            result["status"] = ("cuda_checks_passed" if args.device == "cuda" else "cpu_checks_passed_cuda_not_tested") if passed else "failed"
        result["input_and_code_unchanged"] = (sha(source) == input_hash and all(sha(ROOT/n) == h for n, h in code_hashes.items()))
        if not result["input_and_code_unchanged"]:
            raise RuntimeError("input configuration or validation/forward code changed during the run")
        result["outputs_sha256"] = {p.name: sha(p) for p in out.iterdir() if p.suffix in (".npz", ".yaml")}
        exit_code = 0 if passed is not False else 1
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}", cuda_validation_passed=False)
        save_json(out/"validation.json", result)
        _report(result, out/"report.md")
        raise
    save_json(out/"validation.json", result)
    _report(result, out/"report.md")
    print(f"Completed: {result['status']}; CUDA validated={result['cuda_validation_passed']}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
