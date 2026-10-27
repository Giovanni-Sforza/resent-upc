"""Legacy/v1-prefix preservation and independent second-generation provenance."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from upc_deformation import shape_scan_campaign as legacy
from upc_deformation import shape_scan_fast_campaign as v1
from upc_deformation import shape_scan_fast_v2_campaign as fast
from upc_deformation.precision_campaign import OutputLock


class FakeLegacy:
    calls = []

    def __init__(self, cfg, case):
        self.cfg, self.case = cfg, case

    def calculate(self, r1, r2, return_nodes=False):
        type(self).calls.append((self.case["name"], len(r1)))
        study = self.cfg["study"]
        q = np.linspace(0, .3, study["n_radial"])
        intensity = (3+r1[:, 0, 0]+.2*r2[:, 0, 0])[:, None]*np.exp(-200*q*q)
        coeff = np.zeros((len(r1), 2, 3, len(q)))
        coeff[:, :, 0] = intensity[:, None]
        levels = np.stack([coeff, 1.01*coeff])
        return {"branch_names": ["full", "self"], "coefficients": coeff,
            "b_level_coefficients": levels, "radii": q, "axis": np.linspace(-.2, .2, 5),
            "p_full_axis": np.linspace(-1, 1, 32),
            "b_values": np.linspace(12, 60, study["b_nodes"]),
            "b_weights": np.ones(study["b_nodes"]), "b_steps_fm": np.array([24., 48.]),
            "b_level_weights": np.ones((2, study["b_nodes"])),
            "y_values": np.zeros(1), "y_weights": np.ones(1), "elapsed_seconds": .001}


class FakeFast(FakeLegacy):
    calls = []
    fail_at = None

    def calculate(self, *args, **kwargs):
        result = super().calculate(*args, **kwargs)
        if self.fail_at == len(self.calls):
            raise RuntimeError("simulated fast kernel failure")
        indices = np.flatnonzero(result["radii"] <= self.cfg["study"]["qmax"])
        result.update(stored_indices=indices, stored_q=result["radii"][indices])
        result["A0"] = result["b_level_coefficients"][:, :, :, 0, :][:, :, :, indices].transpose(1, 0, 2, 3)
        return result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = yaml.safe_load((legacy.ROOT/"configs/shape_scan_smoke.yaml").read_text())
    source["scan"]["density_variants"] = source["scan"]["density_variants"][:1]
    source["study"]["n_radial"] = 31
    source["runtime"].update(output=str(tmp_path/"legacy"), checkpoint_size=2)
    path = tmp_path/"config.yaml"; path.write_text(yaml.safe_dump(source))
    monkeypatch.setattr(legacy, "code_hashes", lambda: {"reference": "fixed"})
    monkeypatch.setattr(v1, "code_hashes", lambda: {"reference": "fixed", "fast": "fixed"})
    monkeypatch.setattr(fast, "code_hashes", lambda: {"reference": "fixed", "fast": "fixed", "v2": "fixed"})
    monkeypatch.setattr(v1, "build_refiner", lambda c, k: (FakeFast(c, k), {"test": True}))
    monkeypatch.setattr(v1, "validate_report", lambda path, cfg: {"status": "passed", "code_sha256": v1.code_hashes()})
    monkeypatch.setattr(legacy, "build_refiner", lambda c, k: (FakeLegacy(c, k), {"test": True}))
    monkeypatch.setattr(fast, "build_refiner", lambda c, k: (FakeFast(c, k), {"test": True}))
    monkeypatch.setattr(fast, "validate_report", lambda path, cfg: {
        "status": "passed", "code_sha256": fast.code_hashes()})
    report = tmp_path/"validation.json"; report.write_text('{"status":"passed"}')
    old = legacy.load_config(path)
    cfg = deepcopy(old); cfg["runtime"]["output"] = str(tmp_path/"fast")
    FakeLegacy.calls = []; FakeFast.calls = []; FakeFast.fail_at = None
    return old, cfg, report


def _files(directory):
    return {str(path.relative_to(directory)): fast.sha256(path)
            for path in Path(directory).rglob("*") if path.is_file()}


def test_import_reuses_exact_arrays_preserves_source_and_completes(setup, tmp_path):
    old, cfg, report = setup
    original = legacy.run_campaign(old, max_new_pairs=3)
    source, out = Path(old["runtime"]["output"]), Path(cfg["runtime"]["output"])
    frozen = _files(source)
    result = fast.run_campaign(cfg, import_run=source, validation_path=report)
    assert _files(source) == frozen
    assert result["completed_pairs"] == 13 and result["new_pairs_this_session"] == 10
    assert sum(n for _, n in FakeFast.calls) == 10
    assert result["fingerprint"] != original["fingerprint"]
    assert result["import_record"]["completed_pairs"] == 3
    for path in source.glob("cases/*/chunk_*.npz"):
        with np.load(path, allow_pickle=False) as before, np.load(out/path.relative_to(source), allow_pickle=False) as after:
            for key in ("A0", "rotation1", "rotation2", "data_sha256"):
                np.testing.assert_array_equal(before[key], after[key])
            assert str(after["fingerprint"]) == result["fingerprint"]
            assert str(after["origin_chunk_sha256"]) == fast.sha256(path)
            assert str(after["origin_fingerprint"]) == original["fingerprint"]
            session = out/str(after["session_file"])
            assert fast.sha256(session) == str(before["session_sha256"])
            assert json.loads(session.read_text())["fingerprint"] == original["fingerprint"]
    # Existing analyzer still verifies and consumes both producers' checkpoints.
    spec = importlib.util.spec_from_file_location("analyze_fast_campaign_test", legacy.ROOT/"scripts/analyze_shape_scan.py")
    analyzer = importlib.util.module_from_spec(spec); spec.loader.exec_module(analyzer)
    summary = analyzer.analyze_run(out, tmp_path/"analysis", bootstrap_draws=8, no_plots=True)
    assert summary["status"] == "completed"
    # Original code and directory retain their own resume path.
    assert legacy.run_campaign(old)["new_pairs_this_session"] == 10


def test_import_prepare_then_produce_and_resume_with_new_schedule(setup):
    old, cfg, report = setup
    legacy.run_campaign(old, max_new_pairs=3)
    prepared = fast.run_campaign(cfg, import_run=old["runtime"]["output"], prepare_only=True)
    assert prepared["completed_pairs"] == 3 and prepared["status"] == "prepared"
    assert not FakeFast.calls
    partial = fast.run_campaign(cfg, validation_path=report, max_new_pairs=2)
    assert partial["completed_pairs"] == 5
    out = Path(cfg["runtime"]["output"])
    chunks = {path: fast.sha256(path) for path in out.glob("cases/*/chunk_*.npz")}
    cfg["runtime"].update(batch_size=3, checkpoint_size=3)
    assert fast.run_campaign(cfg, validation_path=report)["status"] == "completed"
    assert all(fast.sha256(path) == digest for path, digest in chunks.items())
    cfg["orientations"]["n_orientations"] = 8
    extended = fast.run_campaign(cfg, validation_path=report)
    assert extended["completed_pairs"] == 25 and extended["new_pairs_this_session"] == 12
    assert all(fast.sha256(path) == digest for path, digest in chunks.items())


def test_fast_failure_keeps_saved_prefix_and_resumes(setup):
    _, cfg, report = setup
    FakeFast.fail_at = 3
    with pytest.raises(RuntimeError, match="simulated"):
        fast.run_campaign(cfg, validation_path=report)
    out = Path(cfg["runtime"]["output"])
    assert json.loads((out/"production_manifest.json").read_text())["status"] == "interrupted"
    chunks = {path: fast.sha256(path) for path in out.glob("cases/*/chunk_*.npz")}
    FakeFast.fail_at = None
    assert fast.run_campaign(cfg, validation_path=report)["status"] == "completed"
    assert all(fast.sha256(path) == digest for path, digest in chunks.items())


def test_interrupted_import_requires_explicit_resume_and_never_skips_prefix(setup, monkeypatch):
    old, cfg, report = setup
    legacy.run_campaign(old, max_new_pairs=5)
    source = Path(old["runtime"]["output"])
    frozen = _files(source)
    writer, calls = fast.atomic_npz, []

    def interrupt(path, **values):
        if Path(path).name.startswith("chunk_"):
            calls.append(path)
            if len(calls) == 2:
                raise KeyboardInterrupt("simulated interruption during import")
        return writer(path, **values)

    monkeypatch.setattr(fast, "atomic_npz", interrupt)
    with pytest.raises(KeyboardInterrupt):
        fast.run_campaign(cfg, import_run=source, prepare_only=True)
    monkeypatch.setattr(fast, "atomic_npz", writer)
    with pytest.raises(ValueError, match="import is incomplete"):
        fast.run_campaign(cfg, validation_path=report)
    result = fast.run_campaign(cfg, import_run=source, validation_path=report)
    assert result["new_pairs_this_session"] == 8
    assert result["import_record"]["completed_pairs"] == 5
    assert _files(source) == frozen


def test_import_rejects_running_source_and_same_or_nested_directory(setup):
    old, cfg, _ = setup
    legacy.run_campaign(old, max_new_pairs=3)
    source = Path(old["runtime"]["output"])
    with OutputLock(source):
        with pytest.raises(RuntimeError, match="source campaign is running"):
            fast.run_campaign(cfg, import_run=source, prepare_only=True)
    frozen = _files(source)
    cfg["runtime"]["output"] = str(source/"nested")
    with pytest.raises(ValueError, match="non-nested"):
        fast.run_campaign(cfg, import_run=source, prepare_only=True)
    assert _files(source) == frozen
    assert not (source/"nested").exists()
    cfg["runtime"]["output"] = str(source)
    with pytest.raises(ValueError, match="non-nested"):
        fast.run_campaign(cfg, import_run=source, prepare_only=True)
    assert _files(source) == frozen


@pytest.mark.parametrize("kind", ["chunk", "session", "grid", "bank"])
def test_import_rejects_corrupt_source_without_modifying_it(setup, kind):
    old, cfg, _ = setup
    legacy.run_campaign(old, max_new_pairs=3)
    source = Path(old["runtime"]["output"])
    root = json.loads((source/"production_manifest.json").read_text())
    path = {"chunk": next(source.glob("cases/*/chunk_*.npz")),
            "session": next((source/"sessions").glob("*.json")),
            "grid": source/"grid.npz", "bank": source/root["bank_file"]}[kind]
    path.write_bytes(path.read_bytes()+b"tampered")
    frozen = _files(source)
    with pytest.raises(ValueError):
        fast.run_campaign(cfg, import_run=source, prepare_only=True)
    assert _files(source) == frozen


def test_import_rejects_different_science_and_source_code(setup, monkeypatch):
    old, cfg, _ = setup
    legacy.run_campaign(old, max_new_pairs=3)
    cfg["study"]["qmax"] = .16
    with pytest.raises(ValueError, match="configuration/reference code"):
        fast.run_campaign(cfg, import_run=old["runtime"]["output"], prepare_only=True)
    cfg["study"]["qmax"] = .17
    cfg["runtime"]["output"] += "_different_code"
    monkeypatch.setattr(legacy, "code_hashes", lambda: {"reference": "changed"})
    with pytest.raises(ValueError, match="configuration/reference code"):
        fast.run_campaign(cfg, import_run=old["runtime"]["output"], prepare_only=True)


def test_validation_gate_precedes_any_fast_production_write(setup, monkeypatch):
    _, cfg, report = setup
    with pytest.raises(ValueError, match="requires --validation"):
        fast.run_campaign(cfg)
    assert not Path(cfg["runtime"]["output"]).exists()

    def fail(path, cfg):
        raise ValueError("batch was not validated")

    monkeypatch.setattr(fast, "validate_report", fail)
    with pytest.raises(ValueError, match="batch was not validated"):
        fast.run_campaign(cfg, validation_path=report)
    assert not Path(cfg["runtime"]["output"]).exists()


def test_import_archive_and_target_chunks_are_bound_on_resume(setup):
    old, cfg, report = setup
    legacy.run_campaign(old, max_new_pairs=3)
    fast.run_campaign(cfg, import_run=old["runtime"]["output"], prepare_only=True)
    out = Path(cfg["runtime"]["output"])
    record = out/fast.IMPORT_RECORD
    original = record.read_bytes()
    record.write_bytes(original+b" ")
    with pytest.raises(ValueError, match="import record changed"):
        fast.run_campaign(cfg, validation_path=report)
    record.write_bytes(original)
    archive = out/"imports/source/provenance.json"
    archive.write_bytes(archive.read_bytes()+b" ")
    with pytest.raises(ValueError, match="provenance archive changed"):
        fast.run_campaign(cfg, validation_path=report)


def test_imported_target_tamper_is_rejected(setup):
    old, cfg, report = setup
    legacy.run_campaign(old, max_new_pairs=3)
    fast.run_campaign(cfg, import_run=old["runtime"]["output"], prepare_only=True)
    out = Path(cfg["runtime"]["output"])
    path = next(out.glob("cases/*/chunk_*.npz"))
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="imported source checkpoint changed"):
        fast.run_campaign(cfg, validation_path=report)


def test_orphan_fast_chunk_after_root_commit_failure_is_recovered(setup, monkeypatch):
    _, cfg, report = setup
    partial = fast.run_campaign(cfg, validation_path=report, max_new_pairs=3)
    writer = fast.atomic_json

    def fail(path, data):
        if Path(path).name == "production_manifest.json" and data["completed_pairs"] > partial["completed_pairs"]:
            raise RuntimeError("root commit interrupted")
        writer(path, data)

    monkeypatch.setattr(fast, "atomic_json", fail)
    with pytest.raises(RuntimeError, match="root commit"):
        fast.run_campaign(cfg, validation_path=report)
    monkeypatch.setattr(fast, "atomic_json", writer)
    assert fast.run_campaign(cfg, validation_path=report)["status"] == "completed"
    assert sum(n for _, n in FakeFast.calls) == 13


def test_completed_campaign_is_noop_and_source_changes_are_rejected(setup, monkeypatch):
    _, cfg, report = setup
    done = fast.run_campaign(cfg, validation_path=report)
    out = Path(cfg["runtime"]["output"])
    frozen = {p: fast.sha256(p) for p in out.rglob("*") if p.is_file() and p.name != ".precision.lock"}
    assert fast.run_campaign(cfg, validation_path=report) == done
    assert all(fast.sha256(p) == digest for p, digest in frozen.items())
    monkeypatch.setattr(fast, "code_hashes", lambda: {"fast": "changed"})
    with pytest.raises(ValueError, match="identity changed"):
        fast.run_campaign(cfg, validation_path=report)


@pytest.mark.parametrize("change", ["delete", "modify"])
def test_archived_fast_validation_report_is_verified_on_resume(setup, change):
    _, cfg, report = setup
    fast.run_campaign(cfg, validation_path=report, max_new_pairs=3)
    out = Path(cfg["runtime"]["output"])
    archived = next((out/"validation").glob("*.json"))
    if change == "delete":
        archived.unlink()
    else:
        archived.write_bytes(archived.read_bytes()+b" ")
    with pytest.raises(ValueError, match="validation report changed or missing"):
        fast.run_campaign(cfg, validation_path=report)


def test_float32_is_rejected_before_prepare_or_migration(setup):
    _, cfg, _ = setup
    cfg["runtime"]["dtype"] = "float32"
    with pytest.raises(ValueError, match="dtype=float64"):
        fast.run_campaign(cfg, prepare_only=True)
    assert not Path(cfg["runtime"]["output"]).exists()


def test_wrong_saved_node_metadata_is_rejected(setup, monkeypatch):
    _, cfg, report = setup
    calculate = FakeFast.calculate

    def wrong(self, *args, **kwargs):
        result = calculate(self, *args, **kwargs)
        result["stored_q"] = result["stored_q"]+.001
        return result

    monkeypatch.setattr(FakeFast, "calculate", wrong)
    with pytest.raises(ValueError, match="invalid fast kernel grid"):
        fast.run_campaign(cfg, validation_path=report)


def test_source_change_between_validation_and_identity_is_rejected(setup, monkeypatch):
    _, cfg, report = setup
    before = fast.code_hashes()

    def validate(path, cfg):
        monkeypatch.setattr(fast, "code_hashes", lambda: {"fast": "changed after validation"})
        return {"status": "passed", "code_sha256": before}

    monkeypatch.setattr(fast, "validate_report", validate)
    with pytest.raises(ValueError, match="source files changed after validation"):
        fast.run_campaign(cfg, validation_path=report)
    assert not Path(cfg["runtime"]["output"]).exists()


def _make_v1_source(setup, tmp_path, *, mixed):
    old, cfg, report = setup
    source_cfg = deepcopy(cfg)
    source_cfg["runtime"]["output"] = str(tmp_path/"source_v1")
    if mixed:
        legacy.run_campaign(old, max_new_pairs=3)
        v1.run_campaign(source_cfg, import_run=old["runtime"]["output"],
                        validation_path=report, max_new_pairs=2)
    else:
        v1.run_campaign(source_cfg, validation_path=report, max_new_pairs=5)
    FakeFast.calls = []
    return Path(source_cfg["runtime"]["output"])


@pytest.mark.parametrize("mixed", [False, True])
def test_import_v1_prefix_preserves_arrays_source_and_entire_provenance_chain(setup, tmp_path, mixed):
    _, cfg, report = setup
    source = _make_v1_source(setup, tmp_path, mixed=mixed)
    frozen = _files(source)
    source_root = json.loads((source/"production_manifest.json").read_text())
    partial = fast.run_campaign(cfg, import_run=source, validation_path=report, max_new_pairs=2)
    assert partial["completed_pairs"] == 7 and partial["new_pairs_this_session"] == 2
    assert partial["import_record"]["source_engine"] == v1.ENGINE
    assert partial["import_record"]["source_fingerprint"] == source_root["fingerprint"]
    out = Path(cfg["runtime"]["output"])
    plan = json.loads((out/fast.IMPORT_PLAN).read_text())
    for relative in ("production_manifest.json", "provenance.json", "effective_config.yaml"):
        assert (out/"imports/source"/relative).read_bytes() == (source/relative).read_bytes()
    for path in source.glob("cases/*/chunk_*.npz"):
        relative = str(path.relative_to(source))
        with np.load(path, allow_pickle=False) as before, np.load(out/relative, allow_pickle=False) as after:
            for key in ("A0", "rotation1", "rotation2", "data_sha256"):
                np.testing.assert_array_equal(before[key], after[key])
            assert str(after["producer"]) == "imported_campaign_prefix"
            assert str(after["origin_engine"]) == v1.ENGINE
            assert str(after["origin_fingerprint"]) == source_root["fingerprint"]
            assert str(after["origin_chunk_sha256"]) == fast.sha256(path)
            metadata = plan["chunks"][relative]["producer_metadata"]
            assert metadata == {key: before[key].tolist() for key in before.files
                                if key not in ("A0", "rotation1", "rotation2")}
            archived_session = out/str(after["session_file"])
            assert fast.sha256(archived_session) == str(before["session_sha256"])
    for path in (source/"validation").glob("*.json"):
        assert (out/"imports/source"/path.relative_to(source)).read_bytes() == path.read_bytes()
    if mixed:
        for path in (source/"imports").rglob("*"):
            if path.is_file():
                assert (out/"imports/source"/path.relative_to(source)).read_bytes() == path.read_bytes()
    assert _files(source) == frozen
    complete = fast.run_campaign(cfg, validation_path=report)
    assert complete["completed_pairs"] == 13 and complete["new_pairs_this_session"] == 6
    assert sum(count for _, count in FakeFast.calls) == 8
    assert _files(source) == frozen
    spec = importlib.util.spec_from_file_location("analyze_fast_v2_campaign_test", legacy.ROOT/"scripts/analyze_shape_scan.py")
    analyzer = importlib.util.module_from_spec(spec); spec.loader.exec_module(analyzer)
    result = analyzer.analyze_run(out, tmp_path/"v2_analysis", bootstrap_draws=8, no_plots=True)
    assert result["status"] == "completed"


@pytest.mark.parametrize("dependency", ["validation", "import_plan", "import_record", "ancestor_provenance"])
def test_v1_source_broken_ancestry_is_rejected_before_import(setup, tmp_path, dependency):
    _, cfg, _ = setup
    source = _make_v1_source(setup, tmp_path, mixed=True)
    paths = {"validation": next((source/"validation").glob("*.json")),
             "import_plan": source/v1.IMPORT_PLAN, "import_record": source/v1.IMPORT_RECORD,
             "ancestor_provenance": source/"imports/source/provenance.json"}
    path = paths[dependency]
    path.write_bytes(path.read_bytes()+b" ")
    frozen = _files(source)
    with pytest.raises(ValueError):
        fast.run_campaign(cfg, import_run=source, prepare_only=True)
    assert _files(source) == frozen
    assert not (Path(cfg["runtime"]["output"])/fast.IMPORT_PLAN).exists()


def test_v1_source_wrong_current_code_identity_is_rejected(setup, tmp_path, monkeypatch):
    _, cfg, _ = setup
    source = _make_v1_source(setup, tmp_path, mixed=False)
    monkeypatch.setattr(v1, "code_hashes", lambda: {"fast": "different v1 code"})
    with pytest.raises(ValueError, match="configuration/reference code"):
        fast.run_campaign(cfg, import_run=source, prepare_only=True)


def test_v1_nested_validation_archive_is_bound_by_v2_import_record(setup, tmp_path):
    _, cfg, report = setup
    source = _make_v1_source(setup, tmp_path, mixed=True)
    fast.run_campaign(cfg, import_run=source, prepare_only=True)
    out = Path(cfg["runtime"]["output"])
    archived_report = next((out/"imports/source/validation").glob("*.json"))
    archived_report.unlink()
    with pytest.raises(ValueError, match="source provenance archive changed"):
        fast.run_campaign(cfg, validation_path=report)
