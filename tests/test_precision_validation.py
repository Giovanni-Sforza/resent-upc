"""Acceptance logic, independent pixel projection, and benchmark bookkeeping."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from upc_deformation.precision_validation import (
    HIGH_PT_C2_ATOL, benchmark, compare_payloads, merge_serial_payloads,
    project_payload, rotation_pairs, scientific_fingerprint, tiny_configuration,
)


def payload_and_projection():
    q = np.linspace(0, .17, 9)
    coeff = np.ones((2, 3, 13))
    payload = {"coefficients": coeff, "b_level_coefficients": coeff[None],
               "b_steps_fm": np.array([2.]), "radii": np.linspace(0, .3, 13),
               "axis": np.linspace(-.2, .2, 25)}
    projection = {"q": q, "levels": [{"b_step_fm": 2., "C2": np.full(9, .01),
                  "orientation_C2": np.full((2, 9), .01), "mean_raw_image": np.ones((25, 25))}]}
    return payload, projection


def test_gate_catches_orientation_errors_hidden_by_mean_cancellation():
    expected, reference = payload_and_projection()
    candidate = copy.deepcopy(reference)
    candidate["levels"][0]["orientation_C2"][:, -1] += np.array([2e-6, -2e-6])
    result = compare_payloads(expected, expected, candidate, reference, precision="float32",
                              comparison_kind="same_precision_cross_device")
    assert result["levels"][0]["C2_mean_max_absolute_high_pT"] == 0
    assert result["passed"] is False


def test_same_precision_c2_gate_is_strict_and_low_pt_is_not_silently_gated():
    payload, reference = payload_and_projection()
    candidate = copy.deepcopy(reference)
    candidate["levels"][0]["C2"][0] += .02
    result = compare_payloads(payload, payload, candidate, reference, precision="float64",
                              comparison_kind="same_precision_cross_device")
    assert result["passed"] is True
    assert result["levels"][0]["C2_mean_max_absolute_all_q"] == pytest.approx(.02)
    candidate["levels"][0]["C2"][-1] += HIGH_PT_C2_ATOL*1.01
    assert compare_payloads(payload, payload, candidate, reference, precision="float64",
                             comparison_kind="same_precision_cross_device")["passed"] is False


def test_scale_error_cannot_hide_behind_normalized_c2():
    payload, projection = payload_and_projection()
    candidate = copy.deepcopy(payload)
    candidate["coefficients"] *= 2
    candidate["b_level_coefficients"] = candidate["coefficients"][None]
    result = compare_payloads(candidate, payload, projection, projection, precision="float32",
                              comparison_kind="same_precision_batch")
    assert result["passed"] is False
    assert result["levels"][0]["coefficient_difference_over_reference_peak"] == 1


def test_float32_vs_float64_is_diagnostic_not_acceptance():
    payload, reference = payload_and_projection()
    candidate = copy.deepcopy(reference)
    candidate["levels"][0]["C2"][-1] += .1
    result = compare_payloads(payload, payload, candidate, reference, precision="float32",
                              comparison_kind="precision_diagnostic")
    assert result["status"] == "diagnostic_only"
    assert result["passed"] is None
    assert result["acceptance_gate_applied"] is False
    assert result["high_pT_C2_absolute_tolerance"] is None


def test_serial_merge_retains_pair_order_and_rejects_grid_changes():
    payload, _ = payload_and_projection()
    rows = []
    for value in (2., 5.):
        row = copy.deepcopy(payload)
        row["coefficients"] = np.full((1, 3, 13), value)
        row["b_level_coefficients"] = row["coefficients"][None]
        rows.append(row)
    merged = merge_serial_payloads(rows)
    np.testing.assert_array_equal(merged["coefficients"][:, 0, 0], [2., 5.])
    rows[1]["radii"][2] += 1e-6
    with pytest.raises(ValueError, match="radii"):
        merge_serial_payloads(rows)


def test_direct_pixel_projection_preserves_isotropic_zero_c2():
    payload, _ = payload_and_projection()
    coeff = np.zeros_like(payload["coefficients"])
    coeff[:, 0, :] = np.asarray([1., 3.])[:, None]
    payload["coefficients"] = coeff
    payload["b_level_coefficients"] = coeff[None]
    projection = project_payload(payload, np.linspace(0, .17, 35), np.arange(64)*2*np.pi/64)
    assert np.max(np.abs(projection["levels"][0]["C2"])) < 1e-15
    assert np.max(np.abs(projection["levels"][0]["orientation_C2"])) < 1e-15


def test_paired_rotation_bank_is_reproducible_proper_and_independent():
    a, b = rotation_pairs(7, 312)
    a2, b2 = rotation_pairs(7, 312)
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_array_equal(b, b2)
    np.testing.assert_allclose(a@a.transpose(0, 2, 1), np.broadcast_to(np.eye(3), a.shape), atol=2e-15)
    np.testing.assert_allclose(np.linalg.det(a), 1, atol=2e-15)
    assert not np.array_equal(a, b)


def test_tiny_scope_is_copied_and_keeps_full_b_window():
    config = {"physics": {"grid_parameters": {}}, "study": {"n_y": 5}}
    original = copy.deepcopy(config)
    tiny = tiny_configuration(config)
    assert config == original
    assert tiny["study"]["b_min_fm"] == 12
    assert tiny["study"]["b_max_fm"] == 60
    assert tiny["physics"]["output_settings"]["p_limit_q"] >= .17
    assert scientific_fingerprint(config) != scientific_fingerprint(tiny)


def test_benchmark_excludes_warmup_and_explicitly_synchronizes(monkeypatch):
    times = iter([1., 3., 10., 14.])
    monkeypatch.setattr("upc_deformation.precision_validation.time.perf_counter", lambda: next(times))
    calls, syncs = [], []
    def run():
        calls.append(1)
        return {"peak_memory_allocated_bytes": 14, "peak_memory_reserved_bytes": 21}
    _, stats = benchmark(run, lambda: syncs.append(1), warmup=3, repeats=2, pairs=2)
    assert len(calls) == 5
    assert len(syncs) == 7
    assert stats["elapsed_seconds"] == [2., 4.]
    assert stats["median_seconds"] == 3
    assert stats["peak_memory_allocated_bytes"] == 14


def test_cli_cuda_unavailable_is_explicit_skip_without_forward(monkeypatch, tmp_path):
    import yaml
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("validate_panel_d_precision_script", root/"scripts/validate_panel_d_precision.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    config = {"physics": {}, "study": {"cases": [{"name": "beta2", "beta2": .3, "beta3": 0., "skin_fm": 0.}]}}
    source = tmp_path/"config.yaml"
    source.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(script, "_run_mode", lambda *a, **k: pytest.fail("unavailable CUDA triggered a CPU fallback"))
    assert script.main(["--config", str(source), "--output", str(tmp_path/"out"), "--device", "cuda"]) == 2
    import json
    result = json.loads((tmp_path/"out"/"validation.json").read_text())
    assert result["status"] == "skipped_cuda_unavailable"
    assert result["cuda_validation_passed"] is False
    assert result["modes"] == {}
