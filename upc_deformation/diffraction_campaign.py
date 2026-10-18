"""Isolated full/self diffraction campaign with immutable paired checkpoints.

Only this module's schema is accepted on resume. Every checkpoint stores both
branches from the same two source amplitudes and orientation pair; extending a
run preserves the existing bank prefix and all completed checkpoint bytes.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from . import precision_campaign as shared
from .precision_campaign import (ROOT, GRID_KEYS, OutputLock, atomic_json,
    atomic_npz, sha256, rotation_bank, configure_runtime, _numpy,
    _digest_arrays, _case_rotations)

SCHEMA = "diffraction_full_self_v1"
BRANCH_NAMES = ("full", "self")
CODE_FILES = (
    "scripts/run_diffraction_observables.py",
    "upc_deformation/diffraction_campaign.py",
    "upc_deformation/diffraction_forward.py",
    "upc_deformation/precision_campaign.py", "upc_deformation/precision_forward.py",
    "upc_deformation/observables.py", "upc_deformation/forward.py",
    "upc_deformation/source_amplitudes.py", "upc_deformation/decay.py",
    "upc_deformation/vendor/upc_forward.py",
)


def load_config(path, overrides=None):
    cfg = shared.load_config(path, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    shared.validate_config(cfg)
    if cfg["runtime"]["fft_mode"] != "separate":
        raise ValueError("Full/self diffraction requires fft_mode=separate")
    span = float(cfg["study"]["b_max_fm"]) - float(cfg["study"]["b_min_fm"])
    step = float(cfg["study"]["b_step_fm"])
    if (not np.isfinite(step) or step <= 0 or
            not np.isclose(span / step, round(span / step), rtol=0, atol=1e-10) or
            cfg["study"]["b_nodes"] != int(round(span / step)) + 1):
        raise ValueError("b_step_fm must divide the registered interval and match b_nodes")
    # Do not permit accidental reuse of a legacy precision output, even before
    # any new provenance file could be written there.
    out = Path(cfg["runtime"]["output"])
    if (out / "production_manifest.json").exists():
        previous = json.loads((out / "production_manifest.json").read_text())
        if previous.get("schema") != SCHEMA:
            raise ValueError("Output belongs to a different campaign schema; use a separate directory")
    if not (out / "provenance.json").exists() and (
            (out / "grid.npz").exists() or any((out / "checkpoints").glob("*/*.npz"))):
        raise ValueError("Unregistered existing campaign files; use a separate output directory")


def code_hashes():
    return {path: sha256(ROOT / path) for path in CODE_FILES}


def scientific_identity(cfg, bank_metadata):
    study = deepcopy(cfg["study"])
    for key in ("n_orientations", "orientation_seed", "threads", "y_batch_size"):
        study.pop(key, None)
    identity = {
        "schema": SCHEMA, "branch_names": list(BRANCH_NAMES),
        "branch_definition": "full=|A1+A2|^2; self=|A1|^2+|A2|^2, paired source Stokes tensors",
        "coefficient_order": ["A0", "C", "D_sin"],
        "physics": cfg["physics"], "study": study, "analysis": cfg["analysis"],
        "orientation_bank": bank_metadata,
        "numerics": {"dtype": cfg["runtime"]["dtype"],
            "device_type": torch.device(cfg["runtime"]["device"]).type,
            "fft_mode": cfg["runtime"]["fft_mode"], "tf32": False, "amp": False},
        "code_sha256": code_hashes(),
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return identity, fingerprint


def build_refiner(cfg, case):
    from .diffraction_forward import DiffractionRefiner
    runtime = cfg["runtime"]
    return DiffractionRefiner(cfg, case, device=runtime["device"],
        dtype=runtime["dtype"], fft_mode=runtime["fft_mode"])


def _read_checkpoint(path, fingerprint, case, index, rotations, nr, nlevels=None):
    with np.load(path, allow_pickle=False) as saved:
        row = {key: saved[key] for key in saved.files}
    if (str(row.get("schema")) != SCHEMA or
            tuple(row.get("branch_names", ())) != BRANCH_NAMES or
            str(row.get("fingerprint")) != fingerprint or
            str(row.get("case")) != case["name"] or int(row.get("index", -1)) != index or
            row["coefficients"].shape != (2, 3, nr) or
            row["b_level_coefficients"].ndim != 4 or
            row["b_level_coefficients"].shape[1:] != (2, 3, nr) or
            (nlevels is not None and len(row["b_level_coefficients"]) != nlevels)):
        raise ValueError(f"Checkpoint identity/shape/branch mismatch: {path}")
    numeric = {key: row[key] for key in
               ("coefficients", "b_level_coefficients", "rotation1", "rotation2")}
    if (not all(np.isfinite(value).all() for value in numeric.values()) or
            not np.array_equal(row["coefficients"], row["b_level_coefficients"][0]) or
            _digest_arrays(numeric) != str(row["data_sha256"]) or
            not all(np.array_equal(row[f"rotation{j+1}"], value) for j, value in enumerate(rotations))):
        raise ValueError(f"Checkpoint data/rotation mismatch: {path}")
    return row


def run_campaign(cfg, *, config_path=None, prepare_only=False, max_new_pairs=None):
    """Execute a bounded or complete full/self run; each pair is atomic."""
    validate_config(cfg)
    if max_new_pairs is not None and max_new_pairs < 0: raise ValueError("max_new_pairs must be nonnegative")
    out = Path(cfg["runtime"]["output"])
    with OutputLock(out):
        environment = configure_runtime(cfg["runtime"])
        environment["y_batch_size"] = int(cfg["study"].get("y_batch_size", cfg["study"]["n_y"]))
        target = cfg["orientations"]["n_orientations"]
        banks, bank_meta = rotation_bank(cfg["orientations"], target)
        identity, fingerprint = scientific_identity(cfg, bank_meta)
        provenance_path = out / "provenance.json"
        if provenance_path.exists():
            previous = json.loads(provenance_path.read_text())
            if previous["fingerprint"] != fingerprint or previous["identity"] != identity:
                raise ValueError("Scientific identity changed; use a separate output directory")
        else:
            atomic_json(provenance_path, {"identity": identity, "fingerprint": fingerprint})
        prior = json.loads((out / "production_manifest.json").read_text()) if (out / "production_manifest.json").exists() else {}
        if target < prior.get("n_orientations", 0): raise ValueError("A registered orientation target may be extended but not reduced")
        if prior and prior.get("fingerprint") != fingerprint:
            raise ValueError("Production manifest scientific identity changed")
        if prior.get("bank_file"):
            previous_bank = out / prior["bank_file"]
            if sha256(previous_bank) != prior["bank_sha256"]: raise ValueError("Previous registered bank changed")
            with np.load(previous_bank, allow_pickle=False) as old:
                if any(not np.array_equal(old[f"rotation_bank_{j+1}"], banks[j][:len(old[f"rotation_bank_{j+1}"])]) for j in (0, 1)):
                    raise ValueError("Orientation extension does not preserve the existing prefix")
        bank_path = out / f"orientation_banks_{target:08d}.npz"
        if not bank_path.exists(): atomic_npz(bank_path, rotation_bank_1=banks[0], rotation_bank_2=banks[1])
        else:
            with np.load(bank_path, allow_pickle=False) as saved:
                if any(not np.array_equal(saved[f"rotation_bank_{j+1}"], banks[j]) for j in (0, 1)):
                    raise ValueError("Registered orientation bank differs")
        cases = cfg["study"]["cases"]
        counts = {c["name"]: (1 if c["beta2"] == 0 and c["beta3"] == 0 else target) for c in cases}
        available = {name: set() for name in counts}
        checkpoint_hashes = {}
        nr = cfg["study"]["n_radial"]
        grid_path = out / "grid.npz"
        grid = None
        if grid_path.exists():
            with np.load(grid_path, allow_pickle=False) as z: grid = {k: z[k] for k in z.files}
            if set(grid) != set(GRID_KEYS) or not all(np.isfinite(v).all() for v in grid.values()):
                raise ValueError("Registered grid keys/data changed")
            if prior.get("grid_sha256") and sha256(grid_path) != prior["grid_sha256"]: raise ValueError("Registered grid metadata changed")
        for case in cases:
            name = case["name"]; directory = out / "checkpoints" / name; directory.mkdir(parents=True, exist_ok=True)
            for index in range(counts[name]):
                path = directory / f"{index:08d}.npz"
                if not path.exists():
                    if str(path.relative_to(out)) in prior.get("checkpoint_sha256", {}):
                        raise ValueError(f"Previously completed checkpoint is missing: {path}")
                    continue
                rotations = [r[0] for r in _case_rotations(case, banks, [index])]
                row = _read_checkpoint(path, fingerprint, case, index, rotations, nr,
                    len(grid["b_steps_fm"]) if grid is not None else None)
                if grid is None or str(row["grid_sha256"]) != sha256(grid_path): raise ValueError("Checkpoint grid identity differs")
                session_source = (out / str(row["session_file"])).resolve()
                if out.resolve() not in session_source.parents or sha256(session_source) != str(row["session_sha256"]):
                    raise ValueError("Checkpoint runtime provenance changed")
                relative = str(path.relative_to(out)); digest = sha256(path)
                if relative in prior.get("checkpoint_sha256", {}) and prior["checkpoint_sha256"][relative] != digest:
                    raise ValueError("Previously completed checkpoint changed")
                checkpoint_hashes[relative] = digest; available[name].add(index)
            if len(available[name]) < prior.get("cases", {}).get(name, {}).get("completed_pairs", 0):
                raise ValueError(f"Previously completed checkpoint is missing for {name}")
        if prior.get("status") == "completed" and target == prior.get("n_orientations"):
            if any(len(available[name]) != count for name, count in counts.items()):
                raise ValueError("Completed campaign has missing checkpoint data")
            effective = out / "effective_config.yaml"
            if not effective.is_file() or sha256(effective) != prior.get("effective_config_sha256"):
                raise ValueError("Completed campaign effective configuration changed")
            for name, expected in prior.get("session_sha256", {}).items():
                session = (out / name).resolve()
                if out.resolve() not in session.parents or not session.is_file() or sha256(session) != expected:
                    raise ValueError("Completed campaign session provenance changed")
            expected_outputs = {f"{name}.npz" for name in counts}
            if set(prior.get("output_sha256", {})) != expected_outputs:
                raise ValueError("Completed campaign aggregate provenance is incomplete")
            for name, expected in prior["output_sha256"].items():
                path = out / name
                if not path.is_file() or sha256(path) != expected:
                    raise ValueError(f"Completed campaign aggregate changed or is missing: {name}")
            # A completed same-target run has no new work. Preserve its
            # effective configuration, session history and completed claims.
            # Scheduling overrides are not a new production session, and
            # repacking the multi-GiB aggregate files would change their hashes.
            return deepcopy(prior)
        effective_path = out / "effective_config.yaml"
        effective_temp = effective_path.with_name(effective_path.name + ".partial")
        effective_temp.write_text(yaml.safe_dump(cfg, sort_keys=False))
        os.replace(effective_temp, effective_path)
        sessions = out / "sessions"; sessions.mkdir(exist_ok=True)
        session_path = sessions / f"{time.time_ns()}_{os.getpid()}.json"
        atomic_json(session_path, {"fingerprint": fingerprint, "environment": environment,
                    "requested_pairs_per_deformation": target, "max_new_pairs": max_new_pairs,
                    "effective_config_sha256": sha256(effective_path), "runtime": cfg["runtime"],
                    "created_utc": datetime.now(timezone.utc).isoformat()})
        manifest = {"schema": SCHEMA, "branch_names": list(BRANCH_NAMES), "status": "prepared" if prepare_only else "running", "fingerprint": fingerprint,
                    "config_path": str(Path(config_path).resolve()) if config_path else None,
                    "effective_config": effective_path.name, "effective_config_sha256": sha256(effective_path),
                    "n_orientations": target, "npairs": counts, "target_pairs_total": sum(counts.values()),
                    "bank_file": bank_path.name, "bank_sha256": sha256(bank_path),
                    "environment": environment, "code_sha256": identity["code_sha256"],
                    "cases": {c["name"]: {"case": c, "n_pairs": counts[c["name"]]} for c in cases},
                    "started_utc": datetime.now(timezone.utc).isoformat(), "new_pairs_this_session": 0}
        manifest["session_file"] = str(session_path.relative_to(out))
        def record(status=None):
            if status: manifest["status"] = status
            manifest["completed_pairs"] = sum(len(v) for v in available.values())
            for name in counts: manifest["cases"][name]["completed_pairs"] = len(available[name])
            # A 32768-pair run has 65538 checkpoints. Rewriting their complete
            # hash map after every batch would generate quadratic metadata IO.
            # Each checkpoint always has its own content/rotation/grid digest;
            # bounded stops and completed runs additionally freeze file hashes.
            if manifest["status"] != "running":
                manifest["checkpoint_sha256"] = dict(checkpoint_hashes)
            if grid is not None: manifest["grid_sha256"] = sha256(grid_path)
            atomic_json(out / "production_manifest.json", manifest)
        record()
        if prepare_only: return manifest
        refiners = {}; batch_size = int(cfg["runtime"]["batch_size"]); started = time.monotonic()
        try:
            for start in range(0, target, batch_size):
                for case in cases:
                    name = case["name"]
                    indices = [i for i in range(start, min(start+batch_size, counts[name])) if i not in available[name]]
                    if max_new_pairs is not None:
                        remaining = max_new_pairs-manifest["new_pairs_this_session"]
                        indices = indices[:max(0, remaining)]
                    if not indices: continue
                    if name not in refiners: refiners[name] = build_refiner(cfg, case)
                    rotations = _case_rotations(case, banks, indices)
                    with torch.autocast(device_type=environment["device_type"], enabled=False):
                        result = refiners[name].calculate(*rotations, return_nodes=False)
                    coeff, levels = _numpy(result["coefficients"]), _numpy(result["b_level_coefficients"])
                    current_grid = {k: _numpy(result[k]) for k in GRID_KEYS}
                    if not all(np.isfinite(v).all() for v in current_grid.values()):
                        raise ValueError("Kernel returned nonfinite grid metadata")
                    if (tuple(result.get("branch_names", ())) != BRANCH_NAMES or
                            coeff.shape != (len(indices), 2, 3, nr) or levels.ndim != 5 or levels.shape[1:] != coeff.shape
                            or levels.shape[0] != len(current_grid["b_steps_fm"]) or not np.array_equal(coeff, levels[0])
                            or not np.isfinite(coeff).all() or not np.isfinite(levels).all()):
                        raise ValueError("Kernel returned invalid batched coefficients")
                    if grid is None:
                        grid = current_grid; atomic_npz(grid_path, **grid)
                    elif any(not np.array_equal(grid[k], current_grid[k]) for k in GRID_KEYS):
                        raise ValueError("Kernel projection/quadrature grid changed between cases/batches")
                    grid_hash = sha256(grid_path)
                    for position, index in enumerate(indices):
                        values = {"coefficients": coeff[position], "b_level_coefficients": levels[:, position],
                                  "rotation1": rotations[0][position], "rotation2": rotations[1][position]}
                        path = out / "checkpoints" / name / f"{index:08d}.npz"
                        if path.exists(): raise ValueError(f"Refusing to overwrite immutable checkpoint: {path}")
                        atomic_npz(path, **values, schema=np.asarray(SCHEMA), branch_names=np.asarray(BRANCH_NAMES),
                                   fingerprint=np.asarray(fingerprint), case=np.asarray(name), index=np.asarray(index),
                                   grid_sha256=np.asarray(grid_hash), data_sha256=np.asarray(_digest_arrays(values)),
                                   session_file=np.asarray(str(session_path.relative_to(out))),
                                   session_sha256=np.asarray(sha256(session_path)))
                        checkpoint_hashes[str(path.relative_to(out))] = sha256(path)
                        available[name].add(index); manifest["new_pairs_this_session"] += 1
                    manifest["elapsed_seconds"] = time.monotonic()-started
                    manifest["last_batch"] = {"case": name, "indices": indices,
                        "elapsed_seconds": float(result.get("elapsed_seconds", 0)),
                        "peak_memory_allocated_bytes": int(result.get("peak_memory_allocated_bytes", 0)),
                        "peak_memory_reserved_bytes": int(result.get("peak_memory_reserved_bytes", 0))}
                    record()
                    print(f"Completed {manifest['completed_pairs']}/{manifest['target_pairs_total']} pairs; {name} {indices[0]}–{indices[-1]}", flush=True)
                if max_new_pairs is not None and manifest["new_pairs_this_session"] >= max_new_pairs: break
            if code_hashes() != identity["code_sha256"]: raise ValueError("Production code changed during this run")
            if bank_meta["import_sha256"] and sha256(cfg["orientations"]["import_bank"]) != bank_meta["import_sha256"]:
                raise ValueError("Imported orientation bank changed during this run")
            if any(len(available[name]) != count for name, count in counts.items()):
                record("incomplete"); return manifest
            outputs = {}
            for case in cases:
                name = case["name"]; collected = []
                for index in range(counts[name]):
                    path = out / "checkpoints" / name / f"{index:08d}.npz"
                    rotations = [r[0] for r in _case_rotations(case, banks, [index])]
                    collected.append(_read_checkpoint(path, fingerprint, case, index, rotations, nr, len(grid["b_steps_fm"])))
                rotations = _case_rotations(case, banks, list(range(counts[name])))
                path = out / f"{name}.npz"
                atomic_npz(path, orientation_coefficients=np.stack([r["coefficients"] for r in collected]),
                           b_level_coefficients=np.stack([r["b_level_coefficients"] for r in collected], axis=1),
                           rotation_bank_1=rotations[0], rotation_bank_2=rotations[1],
                           fingerprint=np.asarray(fingerprint), schema=np.asarray(SCHEMA),
                           branch_names=np.asarray(BRANCH_NAMES), coefficient_names=np.asarray(["A0", "C", "D_sin"]), **grid)
                outputs[path.name] = sha256(path)
            manifest.update(output_sha256=outputs, checkpoint_sha256=checkpoint_hashes,
                            session_sha256={str(p.relative_to(out)): sha256(p) for p in sessions.glob("*.json")},
                            completed_utc=datetime.now(timezone.utc).isoformat(),
                            elapsed_seconds=time.monotonic()-started)
            record("completed")
            return manifest
        except BaseException as error:
            manifest["error"] = repr(error); manifest["elapsed_seconds"] = time.monotonic()-started
            record("interrupted")
            raise
