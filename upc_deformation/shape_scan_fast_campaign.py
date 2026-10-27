"""A0-only execution with explicit, verified reuse of legacy orientation prefixes.

The legacy campaign and its source identity are never modified. A fast campaign
has its own identity and output directory. Imported arrays retain their original
data digest, while their new envelope records the source chunk and session.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import yaml

from . import shape_scan_campaign as legacy
from .precision_campaign import (ROOT, GRID_KEYS, OutputLock, atomic_json,
    atomic_npz, sha256, rotation_bank, configure_runtime, _digest_arrays,
    _case_rotations)
from .shape_scan_physics import resolve_case

SCHEMA = legacy.SCHEMA
ENGINE = "shape_scan_fast_A0_v1"
CODE_FILES = legacy.CODE_FILES + (
    "upc_deformation/shape_scan_fast_forward.py",
    "upc_deformation/shape_scan_fast_campaign.py",
    "upc_deformation/shape_scan_fast_validation.py",
    "upc_deformation/shape_scan_czt.py",
    "upc_deformation/shape_scan_analysis.py",
    "scripts/run_shape_scan_fast.py",
    "scripts/benchmark_shape_scan_fast.py",
)
IMPORT_PLAN = "imports/plan.json"
IMPORT_RECORD = "imports/manifest.json"


def code_hashes():
    return {name: sha256(ROOT/name) for name in CODE_FILES}


def scientific_reference_fingerprint(cfg):
    _, metadata = rotation_bank(cfg["orientations"], cfg["orientations"]["n_orientations"])
    return legacy.scientific_identity(cfg, metadata)[1]


def scientific_identity(cfg, bank_metadata):
    identity, reference = legacy.scientific_identity(cfg, bank_metadata)
    identity.update(execution_engine=ENGINE, reference_fingerprint=reference,
                    code_sha256=code_hashes())
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                             allow_nan=False).encode()).hexdigest()
    return identity, fingerprint


def validate_report(path, cfg):
    # Lazy import keeps the validation module free to use the identity helpers.
    from .shape_scan_fast_validation import validate_report as validate
    return validate(path, cfg)


def build_refiner(cfg, case):
    from .shape_scan_fast_forward import ShapeScanFastRefiner
    kernel_cfg, kernel_case, density = resolve_case(cfg, case)
    r = cfg["runtime"]
    return (ShapeScanFastRefiner(kernel_cfg, kernel_case, device=r["device"],
                                dtype=r["dtype"], fft_mode="separate"), density)


def _inside(directory, relative):
    path = (directory/relative).resolve()
    if directory.resolve() not in path.parents:
        raise ValueError(f"path escapes campaign: {relative}")
    return path


def _json(path):
    return json.loads(Path(path).read_text())


def _load_grid(path):
    with np.load(path, allow_pickle=False) as data:
        grid = {key: data[key] for key in data.files}
    if (set(grid) != set(GRID_KEYS) | {"stored_q", "stored_indices"}
            or not all(np.isfinite(value).all() for value in grid.values())):
        raise ValueError("invalid registered grid")
    return grid


def _copy_immutable(source, target, expected):
    if sha256(source) != expected:
        raise ValueError(f"source changed while importing: {source}")
    if target.exists():
        if sha256(target) != expected:
            raise ValueError(f"imported file changed: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+".partial")
    with Path(source).open("rb") as incoming, temporary.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
        outgoing.flush(); os.fsync(outgoing.fileno())
    if sha256(temporary) != expected:
        raise ValueError(f"imported copy checksum mismatch: {source}")
    os.replace(temporary, target)


@contextmanager
def _source_lock(source):
    """Acquire the existing source flock without changing its PID or contents."""
    lock = source/".precision.lock"
    if not lock.is_file():
        raise ValueError("legacy source has no campaign lock file")
    with lock.open("rb") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("legacy source is running; stop it before importing") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _verify_prefixes(out, cases, fingerprint, banks, grid, grid_hash, root):
    states = {case["name"]: legacy.inspect_case(out, case, fingerprint, banks,
                                               grid, grid_hash) for case in cases}
    for name, entry in root.get("cases", {}).items():
        if name not in states:
            raise ValueError("unregistered case in production manifest")
        count = entry["completed_pairs"]
        if count and legacy.committed_state_hash(states[name], count) != entry["committed_state_sha256"]:
            raise ValueError("previously committed case state changed")
    for name, state in states.items():
        if state["completed_pairs"] and "density" not in state:
            raise ValueError(f"completed case has no density metadata: {name}")
    return states


def _verify_execution_sessions(out, states, fingerprint, hashes):
    """Follow the fast producer's session -> validation-report hash chain."""
    imported = (_json(out/IMPORT_RECORD)["chunks"] if (out/IMPORT_RECORD).exists() else {})
    checked = set()
    for name, state in states.items():
        for filename in state["chunks"]:
            relative = f"cases/{name}/{filename}"
            with np.load(out/relative, allow_pickle=False) as row:
                producer = str(row["producer"]) if "producer" in row.files else None
                if relative in imported:
                    if producer != "imported_legacy_diffraction":
                        raise ValueError("imported checkpoint producer changed")
                    continue  # Entire origin archive is bound by _verify_import.
                if producer != ENGINE:
                    raise ValueError("unregistered fast checkpoint producer")
                session_name = str(row["session_file"])
            if session_name in checked:
                continue
            checked.add(session_name)
            session = _json(_inside(out, session_name))
            if (session.get("fingerprint") != fingerprint or session.get("execution_engine") != ENGINE
                    or session.get("code_sha256") != hashes):
                raise ValueError("fast session producer identity changed")
            binding = session.get("validation")
            if (not isinstance(binding, dict) or not binding.get("file")
                    or not binding.get("sha256")
                    or not _inside(out, binding["file"]).is_file()
                    or sha256(_inside(out, binding["file"])) != binding["sha256"]):
                raise ValueError("fast session validation report changed or missing")


def _verify_import(out, root):
    path = out/IMPORT_RECORD
    binding = root.get("import_record")
    if binding and (binding.get("file") != IMPORT_RECORD or not path.exists()
                    or sha256(path) != binding.get("sha256")):
        raise ValueError("legacy import record changed")
    if not path.exists():
        if (out/IMPORT_PLAN).exists():
            raise ValueError("legacy import is incomplete; repeat --import-run with the original source")
        return None
    record = _json(path)
    if (record.get("schema") != "shape_scan_legacy_import_v1"
            or record.get("status") != "completed"
            or sha256(out/IMPORT_PLAN) != record.get("plan_sha256")):
        raise ValueError("invalid legacy import record/plan")
    for relative, digest in record["archived_files"].items():
        if sha256(_inside(out, relative)) != digest:
            raise ValueError(f"legacy provenance archive changed: {relative}")
    for relative, entry in record["chunks"].items():
        if sha256(_inside(out, relative)) != entry["target_sha256"]:
            raise ValueError(f"imported legacy checkpoint changed: {relative}")
    return {"file": IMPORT_RECORD, "sha256": sha256(path),
            "source_fingerprint": record["source_fingerprint"],
            "completed_pairs": record["completed_pairs"]}


def import_legacy_prefix(cfg, out, source, fingerprint, banks, bank_metadata):
    """Copy verified legacy arrays into a new campaign, preserving their origin.

    An interrupted import must be explicitly resumed with the same source. The
    immutable plan binds the exact source snapshot before the first data write.
    """
    source = Path(source).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("legacy source and fast output must be separate, non-nested directories")
    if not source.is_dir():
        raise ValueError("legacy source directory does not exist")
    with _source_lock(source):
        root = _json(source/"production_manifest.json")
        provenance = _json(source/"provenance.json")
        reference_identity, reference = legacy.scientific_identity(cfg, bank_metadata)
        if (provenance != {"identity": reference_identity, "fingerprint": reference}
                or root.get("schema") != SCHEMA or root.get("fingerprint") != reference
                or root.get("code_sha256") != reference_identity["code_sha256"]):
            raise ValueError("legacy source scientific configuration/reference code does not match")
        if root["n_orientations"] > cfg["orientations"]["n_orientations"]:
            raise ValueError("cannot reduce the imported orientation target")
        for relative, digest in (("effective_config.yaml", root["effective_config_sha256"]),
                                 (root["bank_file"], root["bank_sha256"]),
                                 ("grid.npz", root["grid_sha256"])):
            if sha256(_inside(source, relative)) != digest:
                raise ValueError(f"legacy registered checksum mismatch: {relative}")
        source_cfg = yaml.safe_load((source/"effective_config.yaml").read_text())
        if source_cfg["study"]["cases"] != cfg["study"]["cases"]:
            raise ValueError("legacy case registry changed")
        with np.load(_inside(source, root["bank_file"]), allow_pickle=False) as data:
            for i, bank in enumerate(banks, 1):
                previous = data[f"rotation_bank_{i}"]
                if (previous.shape != (root["n_orientations"], 3, 3)
                        or not np.array_equal(previous, bank[:len(previous)])):
                    raise ValueError("legacy orientation bank differs")
        grid = _load_grid(source/"grid.npz")
        states = _verify_prefixes(source, cfg["study"]["cases"], reference, banks,
                                 grid, root["grid_sha256"], root)
        snapshots = {name: sha256(source/name) for name in
                     ("production_manifest.json", "provenance.json", "effective_config.yaml",
                      "grid.npz", root["bank_file"])}
        entries, sessions = {}, {}
        for case in cfg["study"]["cases"]:
            name = case["name"]
            state = states[name]
            expected = 1 if case["beta2"] == case["beta3"] == 0 else root["n_orientations"]
            if state["completed_pairs"] > expected:
                raise ValueError("legacy prefix exceeds its registered target")
            manifest = source/"cases"/name/"manifest.json"
            if manifest.exists():
                snapshots[str(manifest.relative_to(source))] = sha256(manifest)
            for filename, entry in state["chunks"].items():
                relative = f"cases/{name}/{filename}"
                with np.load(source/relative, allow_pickle=False) as row:
                    session = str(row["session_file"])
                    session_sha = str(row["session_sha256"])
                    sessions[session] = session_sha
                    entries[relative] = {**entry, "data_sha256": str(row["data_sha256"]),
                        "session_file": session, "session_sha256": session_sha}
        if not entries:
            raise ValueError("legacy source contains no completed checkpoints to import")
        plan = {"schema": "shape_scan_legacy_import_plan_v1", "source": str(source),
                "source_fingerprint": reference, "target_fingerprint": fingerprint,
                "snapshots": snapshots, "sessions": sessions, "chunks": entries,
                "completed_pairs": sum(row["completed_pairs"] for row in states.values())}
        plan_path = out/IMPORT_PLAN
        if plan_path.exists():
            if _json(plan_path) != plan:
                raise ValueError("legacy source snapshot differs from the immutable import plan")
        else:
            if (out/"production_manifest.json").exists() or any((out/"cases").glob("*/chunk_*.npz")):
                raise ValueError("legacy import must precede fast production")
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(plan_path, plan)
        if (out/IMPORT_RECORD).exists():
            return _verify_import(out, {})
        _copy_immutable(source/"grid.npz", out/"grid.npz", root["grid_sha256"])
        archived = {}
        for relative, digest in {**snapshots, **sessions}.items():
            archived_relative = "imports/source/"+relative
            _copy_immutable(_inside(source, relative), _inside(out, archived_relative), digest)
            archived[archived_relative] = digest
        imported = {}
        for case in cfg["study"]["cases"]:
            name, state = case["name"], states[case["name"]]
            if not state["completed_pairs"]:
                continue
            directory = out/"cases"/name
            directory.mkdir(parents=True, exist_ok=True)
            chunks = {}
            for filename, entry in state["chunks"].items():
                relative = f"cases/{name}/{filename}"
                origin = entries[relative]
                source_row = legacy.read_chunk(source/relative, fingerprint=reference,
                    case=case, banks=banks, grid=grid, grid_sha256=root["grid_sha256"])
                if sha256(source/relative) != origin["sha256"]:
                    raise ValueError("legacy checkpoint changed during import")
                target = directory/filename
                if not target.exists():
                    source_row.update(fingerprint=np.asarray(fingerprint),
                        session_file=np.asarray("imports/source/"+origin["session_file"]),
                        origin_fingerprint=np.asarray(reference),
                        origin_chunk_sha256=np.asarray(origin["sha256"]),
                        origin_session_file=np.asarray(origin["session_file"]),
                        origin_session_sha256=np.asarray(origin["session_sha256"]),
                        producer=np.asarray("imported_legacy_diffraction"))
                    atomic_npz(target, **source_row)
                actual = legacy.read_chunk(target, fingerprint=fingerprint, case=case,
                    banks=banks, grid=grid, grid_sha256=root["grid_sha256"])
                if (str(actual.get("origin_fingerprint")) != reference
                        or str(actual.get("origin_chunk_sha256")) != origin["sha256"]
                        or str(actual["data_sha256"]) != origin["data_sha256"]
                        or str(actual["session_file"]) != "imports/source/"+origin["session_file"]
                        or str(actual["session_sha256"]) != origin["session_sha256"]):
                    raise ValueError("existing imported checkpoint has different provenance/data")
                digest = sha256(target)
                chunks[filename] = {"start": entry["start"], "stop": entry["stop"], "sha256": digest}
                imported[relative] = {"origin_sha256": origin["sha256"],
                    "data_sha256": origin["data_sha256"], "target_sha256": digest}
            atomic_json(directory/"manifest.json", {**state, "fingerprint": fingerprint,
                "chunks": chunks, "imported_pairs": state["completed_pairs"],
                "legacy_fingerprint": reference})
        for relative, digest in {**snapshots, **sessions}.items():
            if sha256(_inside(source, relative)) != digest:
                raise ValueError("legacy source changed during import")
        atomic_json(out/IMPORT_RECORD, {"schema": "shape_scan_legacy_import_v1",
            "status": "completed", "source_fingerprint": reference,
            "target_fingerprint": fingerprint, "plan_sha256": sha256(plan_path),
            "completed_pairs": plan["completed_pairs"], "chunks": imported,
            "archived_files": archived,
            "arrays": "A0 and both rotation arrays retain their original data_sha256; no forward recomputation"})
        return _verify_import(out, {})


def run_campaign(cfg, *, config_path=None, validation_path=None, import_run=None,
                 prepare_only=False, max_new_pairs=None, selected_cases=None):
    legacy.validate_config(cfg)
    if cfg["runtime"]["dtype"] != "float64":
        raise ValueError("fast A0 production requires dtype=float64")
    if max_new_pairs is not None and (isinstance(max_new_pairs, bool)
                                     or not isinstance(max_new_pairs, int) or max_new_pairs < 0):
        raise ValueError("max_new_pairs must be a nonnegative integer")
    out = Path(cfg["runtime"]["output"]).resolve()
    if import_run is not None:
        source = Path(import_run).resolve()
        if source == out or source in out.parents or out in source.parents:
            raise ValueError("legacy source and fast output must be separate, non-nested directories")
    target, cases = cfg["orientations"]["n_orientations"], cfg["study"]["cases"]
    names = {case["name"] for case in cases}
    selected = names if selected_cases is None else set(selected_cases)
    if not selected <= names:
        raise ValueError(f"unknown selected cases: {selected-names}")
    if not prepare_only:
        if validation_path is None:
            raise ValueError("fast production requires --validation with a passing report")
        validation_path = Path(validation_path).resolve()
        validation_digest = sha256(validation_path)
        validation = validate_report(validation_path, cfg)
        if sha256(validation_path) != validation_digest:
            raise ValueError("validation report changed while checking")
    else:
        validation = None
    banks, bank_meta = rotation_bank(cfg["orientations"], target)
    identity, fingerprint = scientific_identity(cfg, bank_meta)
    if validation is not None and validation.get("code_sha256") != identity["code_sha256"]:
        raise ValueError("source files changed after validation report verification")
    counts = {case["name"]: 1 if case["beta2"] == case["beta3"] == 0 else target for case in cases}
    with OutputLock(out):
        provenance_path = out/"provenance.json"
        if provenance_path.exists():
            if _json(provenance_path) != {"identity": identity, "fingerprint": fingerprint}:
                raise ValueError("fast scientific identity changed; use a new output directory")
        elif any(path.name != ".precision.lock" for path in out.iterdir()):
            raise ValueError("unregistered nonempty fast output directory")
        else:
            atomic_json(provenance_path, {"identity": identity, "fingerprint": fingerprint})
        root_path = out/"production_manifest.json"
        previous = _json(root_path) if root_path.exists() else {}
        if previous and (previous.get("schema") != SCHEMA or previous.get("fingerprint") != fingerprint):
            raise ValueError("different fast campaign schema/fingerprint")
        if target < previous.get("n_orientations", 0):
            raise ValueError("orientation target may be extended but not reduced")
        if previous.get("bank_file"):
            path = _inside(out, previous["bank_file"])
            if sha256(path) != previous["bank_sha256"]:
                raise ValueError("registered bank changed")
            with np.load(path, allow_pickle=False) as row:
                if any(not np.array_equal(row[f"rotation_bank_{i+1}"], bank[:len(row[f"rotation_bank_{i+1}"])])
                       for i, bank in enumerate(banks)):
                    raise ValueError("orientation prefix changed")
        bank_path = out/f"orientation_banks_{target:08d}.npz"
        if not bank_path.exists():
            atomic_npz(bank_path, rotation_bank_1=banks[0], rotation_bank_2=banks[1])
        else:
            with np.load(bank_path, allow_pickle=False) as row:
                if any(not np.array_equal(row[f"rotation_bank_{i+1}"], bank) for i, bank in enumerate(banks)):
                    raise ValueError("existing bank differs")
        if import_run is not None:
            import_legacy_prefix(cfg, out, import_run, fingerprint, banks, bank_meta)
        imported = _verify_import(out, previous)
        grid_path = out/"grid.npz"
        grid = _load_grid(grid_path) if grid_path.exists() else None
        grid_hash = sha256(grid_path) if grid is not None else None
        if previous.get("grid_sha256") and previous["grid_sha256"] != grid_hash:
            raise ValueError("registered grid changed")
        states = _verify_prefixes(out, cases, fingerprint, banks, grid, grid_hash, previous)
        _verify_execution_sessions(out, states, fingerprint, identity["code_sha256"])
        if any(states[name]["completed_pairs"] > counts[name] for name in names):
            raise ValueError("completed prefix exceeds target")
        if previous.get("status") == "completed" and previous["n_orientations"] == target:
            if any(states[name]["completed_pairs"] != counts[name] for name in names):
                raise ValueError("completed campaign lost data")
            if sha256(out/"effective_config.yaml") != previous["effective_config_sha256"]:
                raise ValueError("effective configuration changed")
            for name, digest in previous.get("case_manifest_sha256", {}).items():
                if sha256(out/"cases"/name/"manifest.json") != digest:
                    raise ValueError("case manifest changed")
            return previous
        for case in cases:
            state = states[case["name"]]
            if state["completed_pairs"]:
                atomic_json(out/"cases"/case["name"]/"manifest.json", state)
        effective = out/"effective_config.yaml"
        temporary = effective.with_suffix(".yaml.partial")
        temporary.write_text(yaml.safe_dump(cfg, sort_keys=False)); os.replace(temporary, effective)
        environment = ({"requested_device": cfg["runtime"]["device"], "executed_forward": False}
                       if prepare_only else configure_runtime(cfg["runtime"]))
        sessions = out/"sessions"; sessions.mkdir(exist_ok=True)
        session_path = sessions/f"fast_{time.time_ns()}_{os.getpid()}.json"
        validation_binding = None
        if validation is not None:
            report_path = out/"validation"/f"{validation_digest}.json"
            _copy_immutable(validation_path, report_path, validation_digest)
            validation_binding = {"file": str(report_path.relative_to(out)), "sha256": validation_digest}
        atomic_json(session_path, {"fingerprint": fingerprint, "execution_engine": ENGINE,
            "environment": environment, "runtime": cfg["runtime"],
            "code_sha256": identity["code_sha256"], "validation": validation_binding,
            "import_record": imported, "y_batch_size": cfg["study"].get("y_batch_size"),
            "target": target, "selected_cases": sorted(selected), "max_new_pairs": max_new_pairs,
            "effective_config_sha256": sha256(effective), "created_utc": datetime.now(timezone.utc).isoformat()})
        manifest = {"schema": SCHEMA, "execution_engine": ENGINE,
            "status": "prepared" if prepare_only else "running", "fingerprint": fingerprint,
            "n_orientations": target, "npairs": counts, "target_pairs_total": sum(counts.values()),
            "new_pairs_this_session": 0, "bank_file": bank_path.name, "bank_sha256": sha256(bank_path),
            "environment": environment, "code_sha256": identity["code_sha256"],
            "effective_config_sha256": sha256(effective), "config_path": str(config_path),
            "session_file": str(session_path.relative_to(out)), "import_record": imported,
            "validation": validation_binding}

        def record(status=None):
            if status:
                manifest["status"] = status
            manifest["completed_pairs"] = sum(state["completed_pairs"] for state in states.values())
            manifest["cases"] = {case["name"]: {"case": case,
                "completed_pairs": states[case["name"]]["completed_pairs"],
                "target_pairs": counts[case["name"]],
                "committed_state_sha256": legacy.committed_state_hash(states[case["name"]], states[case["name"]]["completed_pairs"])}
                for case in cases}
            if grid is not None:
                manifest["grid_sha256"] = sha256(grid_path)
            manifest["case_manifest_sha256"] = {case["name"]: sha256(out/"cases"/case["name"]/"manifest.json")
                for case in cases if (out/"cases"/case["name"]/"manifest.json").exists()}
            atomic_json(root_path, manifest)

        record()
        if prepare_only:
            return manifest
        refiner = None
        try:
            for case in cases:
                name, state = case["name"], states[case["name"]]
                if name not in selected or state["completed_pairs"] >= counts[name]:
                    continue
                if max_new_pairs is not None and manifest["new_pairs_this_session"] >= max_new_pairs:
                    break
                refiner, density = build_refiner(cfg, case)
                if state.get("density") is not None and state["density"] != density:
                    raise ValueError("resolved density changed across producers")
                state["density"] = density
                directory = out/"cases"/name; directory.mkdir(parents=True, exist_ok=True)
                while state["completed_pairs"] < counts[name]:
                    start = state["completed_pairs"]
                    stop = min(start+cfg["runtime"]["checkpoint_size"], counts[name])
                    if max_new_pairs is not None:
                        stop = min(stop, start+max_new_pairs-manifest["new_pairs_this_session"])
                    if stop <= start:
                        break
                    blocks, elapsed = [], 0.
                    for begin in range(start, stop, cfg["runtime"]["batch_size"]):
                        end = min(begin+cfg["runtime"]["batch_size"], stop)
                        rotations = _case_rotations(case, banks, list(range(begin, end)))
                        result = refiner.calculate(*rotations, return_nodes=False)
                        current = {key: np.asarray(result[key]) for key in GRID_KEYS}
                        indices = np.flatnonzero(current["radii"] <= cfg["study"]["qmax"])
                        current.update(stored_indices=indices, stored_q=current["radii"][indices])
                        if (not all(np.isfinite(value).all() for value in current.values())
                                or cfg["study"]["qmax"] > current["radii"][-1]
                                or len(indices) < 3 or tuple(result["branch_names"]) != legacy.BRANCHES
                                or not np.array_equal(result.get("stored_indices"), indices)
                                or not np.array_equal(result.get("stored_q"), current["stored_q"])):
                            raise ValueError("invalid fast kernel grid/branches")
                        if grid is None:
                            grid = current; atomic_npz(grid_path, **grid); grid_hash = sha256(grid_path)
                        elif any(not np.array_equal(grid[key], current[key]) for key in grid):
                            raise ValueError("kernel grid changed between batches/cases/producers")
                        a0 = np.asarray(result["A0"])
                        expected = (end-begin, len(grid["b_steps_fm"]), 2, len(indices))
                        if (a0.shape != expected or a0.dtype != np.float64
                                or not np.isfinite(a0).all() or np.any(a0 < 0)):
                            raise ValueError("invalid fast A0 payload")
                        blocks.append(a0); elapsed += float(result["elapsed_seconds"])
                    rotations = _case_rotations(case, banks, list(range(start, stop)))
                    values = {"A0": np.concatenate(blocks), "rotation1": rotations[0], "rotation2": rotations[1]}
                    path = directory/f"chunk_{start:08d}_{stop:08d}.npz"
                    if path.exists():
                        raise ValueError("refusing to overwrite immutable checkpoint")
                    atomic_npz(path, **values, schema=np.asarray(SCHEMA), fingerprint=np.asarray(fingerprint),
                        producer=np.asarray(ENGINE), case=np.asarray(name), branch_names=np.asarray(legacy.BRANCHES),
                        start=np.asarray(start), stop=np.asarray(stop), grid_sha256=np.asarray(grid_hash),
                        data_sha256=np.asarray(_digest_arrays(values)),
                        session_file=np.asarray(str(session_path.relative_to(out))), session_sha256=np.asarray(sha256(session_path)))
                    state["chunks"][path.name] = {"start": start, "stop": stop, "sha256": sha256(path)}
                    state["completed_pairs"] = stop
                    state["forward_seconds"] = state.get("forward_seconds", 0.)+elapsed
                    atomic_json(directory/"manifest.json", state)
                    manifest["new_pairs_this_session"] += stop-start
                    record()
                    print(f"{name}: {stop}/{counts[name]} pairs; total {manifest['completed_pairs']}/{manifest['target_pairs_total']}; {elapsed:.2f}s/chunk [fast A0]", flush=True)
                del refiner; refiner = None; gc.collect()
                if torch.device(cfg["runtime"]["device"]).type == "cuda":
                    torch.cuda.empty_cache()
            if code_hashes() != identity["code_sha256"]:
                raise ValueError("fast/reference source files changed during execution")
            if cfg["orientations"].get("import_bank") and sha256(cfg["orientations"]["import_bank"]) != bank_meta["import_sha256"]:
                raise ValueError("imported orientation bank changed during execution")
            record("completed" if all(states[name]["completed_pairs"] == counts[name] for name in names) else "incomplete")
        except BaseException as error:
            manifest["error"] = f"{type(error).__name__}: {error}"
            record("interrupted")
            raise
        finally:
            if refiner is not None:
                del refiner
        return manifest
