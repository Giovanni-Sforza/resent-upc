"""Benchmark scheduling/provenance checks use mocks, never the production grid."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "scripts/benchmark_shape_scan.py"
SPEC = importlib.util.spec_from_file_location("shape_scan_benchmark_under_test", PATH)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


@pytest.fixture
def cfg():
    cases = [
        {"name": "none", "beta2": 0., "beta3": 0., "density_type": "none"},
        {"name": "halo_lower", "beta2": .1, "beta3": .2, "density_type": "halo"},
        {"name": "halo_max", "beta2": .3, "beta3": .2, "density_type": "halo"},
        {"name": "skin_max", "beta2": .3, "beta3": .2, "density_type": "skin"},
    ]
    return {"physics": {"test_only": True},
            "study": {"cases": cases, "n_y": 5, "n_radial": 7, "y_batch_size": 1},
            "runtime": {"device": "cpu", "dtype": "float64", "fft_mode": "separate",
                        "threads": 1, "batch_size": 1, "output": "unused-production-directory"},
            "orientations": {"seed": 99, "block_size": 4, "import_bank": None}}


@pytest.fixture
def mocks(monkeypatch):
    environment = {"device_type": "cuda", "device": "cuda", "device_total_memory_bytes": 1000}
    monkeypatch.setattr(benchmark, "code_hashes", lambda: {"test": "source-v1"})
    monkeypatch.setattr(benchmark, "configure_runtime", lambda runtime: deepcopy(environment))
    monkeypatch.setattr(benchmark, "resolve_case", lambda cfg, case: (cfg, case, {"resolved": True}))
    monkeypatch.setattr(benchmark, "_release", lambda environment: None)
    monkeypatch.setattr(benchmark, "_memory", lambda environment: {
        "peak_memory_allocated_bytes": 10, "peak_memory_reserved_bytes": 20})
    return environment


def measured(pairs_per_second, reserved=100):
    return {"pairs_per_second": pairs_per_second, "median_seconds": 1.,
            "elapsed_seconds": [1.], "peak_memory_allocated_bytes": reserved // 2,
            "peak_memory_reserved_bytes": reserved}


def test_select_representative_case(cfg):
    assert benchmark.select_case(cfg["study"]["cases"])["name"] == "halo_max"
    assert benchmark.select_case(cfg["study"]["cases"], "none")["name"] == "none"
    with pytest.raises(ValueError, match="specify --case"):
        benchmark.select_case([case for case in cfg["study"]["cases"] if case["density_type"] != "halo"])
    with pytest.raises(ValueError, match="registered case"):
        benchmark.select_case(cfg["study"]["cases"], "missing")


def test_recommends_fastest_eligible_measured_scheme_and_records_provenance(cfg, mocks, monkeypatch, tmp_path):
    calls = []
    original = deepcopy(cfg)

    def measure(local, case, banks, environment, batch, y, warmup, repeats):
        calls.append((batch, y))
        assert local["study"]["n_radial"] == 7
        assert len(banks[0]) == 4
        return measured(batch * y, reserved=900 if (batch, y) == (4, 5) else 100)

    monkeypatch.setattr(benchmark, "measure_configuration", measure)
    path = tmp_path / "benchmark"
    report = benchmark.run_benchmark(cfg, path, batches=[4, 2], y_batches=[5, 1], warmup=0)
    assert calls == [(2, 1), (4, 1), (2, 5), (4, 5)]
    assert report["recommendation"]["batch_size"] == 2
    assert report["recommendation"]["y_batch_size"] == 5
    assert report["case"]["name"] == "halo_max"
    assert report["memory_limit_applied"] is True
    assert report["input_and_code_unchanged"] is True
    assert report["code_sha256"] == {"test": "source-v1"}
    assert report["orientation_bank"]["seed"] == 99
    assert cfg == original
    assert set(p.name for p in path.iterdir()) == {"benchmark.json", "effective_config.yaml", ".precision.lock"}
    assert json.loads((path / "benchmark.json").read_text())["status"] == "completed"


def test_oom_is_not_retried_and_larger_products_are_explicitly_skipped(cfg, mocks, monkeypatch, tmp_path):
    calls = []

    def measure(local, case, banks, environment, batch, y, warmup, repeats):
        calls.append((batch, y))
        if batch == 4:
            raise torch.OutOfMemoryError("synthetic CUDA out of memory")
        return measured(2.)

    monkeypatch.setattr(benchmark, "measure_configuration", measure)
    report = benchmark.run_benchmark(cfg, tmp_path / "oom", batches=[2, 4, 8], y_batches=[1, 5])
    assert calls == [(2, 1), (4, 1)]
    assert [row["status"] for row in report["configurations"]] == ["ok", "oom"] + ["skipped_after_oom"] * 4
    assert report["recommendation"]["batch_size"] == 2
    assert report["configurations"][1]["error_type"] == "OutOfMemoryError"


def test_non_oom_failure_is_recorded_without_repeating_or_pruning(cfg, mocks, monkeypatch, tmp_path):
    calls = []

    def measure(local, case, banks, environment, batch, y, warmup, repeats):
        calls.append(batch)
        if batch == 2:
            raise ValueError("invalid test payload")
        return measured(4.)

    monkeypatch.setattr(benchmark, "measure_configuration", measure)
    report = benchmark.run_benchmark(cfg, tmp_path / "failed", batches=[2, 4], y_batches=[1])
    assert calls == [2, 4]
    assert [row["status"] for row in report["configurations"]] == ["failed", "ok"]
    assert report["failed_configurations"] == 1
    assert report["recommendation"]["batch_size"] == 4


def test_no_recommendation_when_all_exceed_memory_limit(cfg, mocks, monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "measure_configuration", lambda *args: measured(100., reserved=801))
    report = benchmark.run_benchmark(cfg, tmp_path / "over_limit", batches=[2], y_batches=[1])
    assert report["status"] == "no_eligible_configuration"
    assert report["recommendation"] is None
    assert report["configurations"][0]["status"] == "ok"


def test_nonempty_output_refused_before_runtime_or_forward(cfg, monkeypatch, tmp_path):
    path = tmp_path / "existing"
    path.mkdir()
    sentinel = path / "user-file.txt"
    sentinel.write_text("keep")
    monkeypatch.setattr(benchmark, "configure_runtime", lambda _: pytest.fail("runtime must not start"))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        benchmark.run_benchmark(cfg, path, batches=[2], y_batches=[1])
    assert sentinel.read_text() == "keep"
    assert list(path.iterdir()) == [sentinel]


def test_cuda_unavailable_records_failure_without_cpu_fallback(cfg, monkeypatch, tmp_path):
    cfg["runtime"]["device"] = "cuda"
    monkeypatch.setattr(benchmark, "code_hashes", lambda: {"test": "v1"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(benchmark, "build_refiner", lambda *args: pytest.fail("no forward permitted"))
    path = tmp_path / "unavailable"
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        benchmark.run_benchmark(cfg, path, batches=[2], y_batches=[1])
    report = json.loads((path / "benchmark.json").read_text())
    assert report["status"] == "failed"
    assert report["configurations"] == []
    assert report["recommendation"] is None
    with pytest.raises(ValueError, match="refusing to overwrite"):
        benchmark.run_benchmark(cfg, path, batches=[2], y_batches=[1])


def test_timing_excludes_warmup_and_discards_validated_payload(cfg, monkeypatch):
    calls = []

    class FakeKernel:
        def __init__(self, local, case):
            assert local["runtime"]["batch_size"] == 2
            assert local["study"]["y_batch_size"] == 5

        def calculate(self, r1, r2, return_nodes):
            assert return_nodes is False
            calls.append(r1.copy())
            c = np.ones((len(r1), 2, 3, 7))
            return {"branch_names": ["full", "self"], "coefficients": c,
                    "b_level_coefficients": np.stack((c, c, c)), "b_steps_fm": [.5, 1., 2.]}

    monkeypatch.setattr(benchmark, "build_refiner", FakeKernel)
    times = iter([0., .1, 10., 110., 200., 202., 300., 304.])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    banks, _ = benchmark.rotation_bank(cfg["orientations"], 2)
    result = benchmark.measure_configuration(cfg, cfg["study"]["cases"][2], banks,
        {"device_type": "cpu", "device": "cpu"}, 2, 5, 1, 2)
    assert len(calls) == 3
    assert result["elapsed_seconds"] == [2., 4.]
    assert result["pairs_per_second"] == pytest.approx(2 / 3)
    assert result["construction_seconds"] == pytest.approx(.1)
    assert result["peak_memory_reserved_bytes"] == 0
    assert cfg["study"]["y_batch_size"] == 1


@pytest.mark.parametrize("kwargs", [{"batches": [2, 2]}, {"batches": [0]},
    {"y_batches": [6]}, {"max_memory_fraction": 1.01}, {"repeats": 0}])
def test_invalid_schemes_fail_before_output_creation(cfg, tmp_path, kwargs):
    args = {"batches": [2], "y_batches": [1], **kwargs}
    path = tmp_path / "invalid"
    with pytest.raises(ValueError):
        benchmark.run_benchmark(cfg, path, **args)
    assert not path.exists()
