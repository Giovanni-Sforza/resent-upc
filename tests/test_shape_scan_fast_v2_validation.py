"""Scientific and runtime gates for fast scan production certificates."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from upc_deformation.shape_scan_campaign import ROOT, load_config
from upc_deformation.shape_scan_fast_v2_validation import (
    compare_payloads, run_benchmark, validate_report,
)


@pytest.fixture
def cfg(tmp_path):
    raw = yaml.safe_load((ROOT/"configs/shape_scan_smoke.yaml").read_text())
    raw["physics"]["grid_parameters"].update(
        WS={"N": 8, "L": 24.}, T_prim={"N": 8, "L": 24.},
        output={"N": 32, "L": 160.})
    raw["physics"]["output_settings"].update(p_limit_q=.05, target_size_q=13)
    raw["study"].update(qmax=.04, n_radial=31, n_phi=8)
    raw["runtime"].update(output=str(tmp_path/"run"), batch_size=2, checkpoint_size=4)
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def test_real_reference_closure_certificate_and_runtime_guards(cfg, tmp_path):
    directory = tmp_path/"benchmark"
    report = run_benchmark(cfg, directory, batches=[1, 2], y_batches=[1], warmup=0, repeats=1)
    assert report["status"] == "passed"
    assert len(report["representative_checks"]) == 6
    assert all(c["passed"] for c in report["representative_checks"])
    path = directory/"benchmark.json"
    assert validate_report(path, cfg)["recommendation"]
    changed = deepcopy(cfg); changed["runtime"]["batch_size"] = 3
    with pytest.raises(ValueError, match="lacks a passed"):
        validate_report(path, changed)
    changed = deepcopy(cfg); changed["study"]["qmax"] = .035
    with pytest.raises(ValueError, match="scientific reference"):
        validate_report(path, changed)
    for change, message in (
        (lambda r: r["code_sha256"].update({"invented": "wrong"}), "source hashes"),
        (lambda r: r["representative_checks"].clear(), "representative-case"),
        (lambda r: r["environment"].update(torch="wrong-version"), "runtime/device"),
        (lambda r: r.update(schema="shape_scan_fast_validation_v1"), "passed shape-scan"),
        (lambda r: r.update(status="running"), "passed shape-scan"),
    ):
        bad = deepcopy(report); change(bad)
        candidate = tmp_path/"bad.json"; candidate.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=message):
            validate_report(candidate, cfg)
    with pytest.raises(ValueError, match="new or empty"):
        run_benchmark(cfg, directory, batches=[2], y_batches=[1], warmup=0, repeats=1)


def test_channel_scaled_tolerance_does_not_hide_small_branch_errors():
    from upc_deformation.precision_campaign import GRID_KEYS
    q = np.linspace(0, .17, 31)
    a = np.exp(-400*q*q)[None, None, None, :]*np.ones((2, 3, 2, 1))
    a[:, :, 0] *= 1e-9
    reference = {key: np.arange(3) for key in GRID_KEYS}
    reference.update(stored_indices=np.arange(31), stored_q=q, A0=a, branch_names=["full", "self"])
    actual = deepcopy(reference)
    assert compare_payloads(reference, actual)["passed"]
    actual["A0"][0, 0, 0, 10] *= 1.001
    result = compare_payloads(reference, actual)
    assert not result["passed"] and result["failed_values"] == 1
    actual = deepcopy(reference); actual["stored_q"][1] += 1e-8
    assert not compare_payloads(reference, actual)["passed"]
    actual = deepcopy(reference); actual["A0"][0, 0, 0, 0] = np.nan
    assert not compare_payloads(reference, actual)["passed"]
