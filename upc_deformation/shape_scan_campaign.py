"""Isolated multi-parameter scan of full/self radial A0, with atomic resume.

The proven diffraction kernel is unchanged. Cases are evaluated sequentially
with one live GPU model; immutable chunks store each orientation, all b levels,
and both source branches once. Scheduling changes do not change the physics ID.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .precision_campaign import (ROOT, GRID_KEYS, OutputLock, atomic_json,
    atomic_npz, sha256, rotation_bank, configure_runtime, _digest_arrays,
    _case_rotations, clean)
from .shape_scan_physics import build_cases, resolve_case

SCHEMA = "single_image_shape_scan_A0_v1"
BRANCHES = ("full", "self")
CODE_FILES = ("scripts/run_shape_scan.py", "upc_deformation/shape_scan_campaign.py",
    "upc_deformation/shape_scan_physics.py", "upc_deformation/diffraction_forward.py",
    "upc_deformation/precision_forward.py", "upc_deformation/precision_campaign.py",
    "upc_deformation/observables.py", "upc_deformation/forward.py",
    "upc_deformation/source_amplitudes.py", "upc_deformation/decay.py",
    "upc_deformation/vendor/upc_forward.py")


def load_config(path, overrides=None):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    for key in ("physics", "study", "scan", "orientations", "runtime", "analysis"):
        if not isinstance(cfg.get(key), dict):
            raise ValueError(f"missing configuration section: {key}")
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        section = {"n_orientations": "orientations", "b_step_fm": "study", "n_y": "study",
                   "y_batch_size": "study"}.get(key, "runtime")
        cfg[section][key] = value
    bank = cfg["orientations"].get("import_bank")
    if bank:
        cfg["orientations"]["import_bank"] = str((path.parent/bank).resolve())
    cfg["runtime"]["output"] = str((ROOT/Path(cfg["runtime"]["output"])).resolve())
    cfg["runtime"].setdefault("checkpoint_size", 128)
    cfg["runtime"].setdefault("fft_mode", "separate")
    cfg["study"]["cases"] = build_cases(cfg)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    s, o, r = cfg["study"], cfg["orientations"], cfg["runtime"]
    for key in ("n_orientations", "block_size"):
        if not isinstance(o[key], int) or o[key] < 1:
            raise ValueError(f"positive integer orientations.{key} required")
    if o["n_orientations"] < 4 or o["n_orientations"] % 4:
        raise ValueError("orientation target must be >=4 and divisible by four")
    if not isinstance(o["seed"], int) or o["seed"] < 0:
        raise ValueError("nonnegative integer seed required")
    if r["dtype"] not in ("float32", "float64") or r["fft_mode"] != "separate":
        raise ValueError("float32/float64 and separate source FFT required")
    if torch.device(r["device"]).type not in ("cpu", "cuda"):
        raise ValueError("explicit cpu/cuda device required")
    for key in ("batch_size", "threads", "checkpoint_size"):
        if not isinstance(r[key], int) or r[key] < 1:
            raise ValueError(f"positive integer runtime.{key} required")
    if s.get("longitudinal_units") != "physical":
        raise ValueError("physical longitudinal units required")
    span, step = float(s["b_max_fm"])-float(s["b_min_fm"]), float(s["b_step_fm"])
    if not np.isfinite(span+step) or span <= 0 or step <= 0 or not np.isclose(span/step, round(span/step), rtol=0, atol=1e-10):
        raise ValueError("b_step_fm must divide the positive interval")
    s["b_nodes"] = int(round(span/step))+1
    if s["n_y"] < 1 or s.get("y_batch_size", 1) < 1 or s["n_phi"] < 8 or s["n_phi"] % 4 or s["n_radial"] < 3:
        raise ValueError("invalid quadrature grid")
    if not np.isfinite(s["qmax"]) or s["qmax"] <= 0:
        raise ValueError("positive finite qmax required")
    if s["cases"] != build_cases(cfg):
        raise ValueError("case registry differs from registered scan grid")
    if not s["cases"]:
        raise ValueError("empty scan")


def code_hashes():
    return {p: sha256(ROOT/p) for p in CODE_FILES}


def scientific_identity(cfg, bank_metadata):
    study = deepcopy(cfg["study"])
    for key in ("y_batch_size", "threads"):
        study.pop(key, None)
    identity = clean({"schema": SCHEMA, "branch_names": BRANCHES,
        "physics": cfg["physics"], "study": study, "scan": cfg["scan"],
        "storage": "per-orientation unnormalized A0, all b levels, q<=registered qmax",
        "orientation_bank": bank_metadata,
        "numerics": {"dtype": cfg["runtime"]["dtype"],
            "device_type": torch.device(cfg["runtime"]["device"]).type,
            "fft_mode": "separate", "tf32": False, "amp": False},
        "code_sha256": code_hashes()})
    return identity, hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def build_refiner(cfg, case):
    from .diffraction_forward import DiffractionRefiner
    kernel_cfg, kernel_case, density = resolve_case(cfg, case)
    r = cfg["runtime"]
    refiner = DiffractionRefiner(kernel_cfg, kernel_case, device=r["device"], dtype=r["dtype"], fft_mode="separate")
    return refiner, density


def read_chunk(path, *, fingerprint, case, banks, grid, grid_sha256):
    with np.load(path, allow_pickle=False) as z:
        row = {k: z[k] for k in z.files}
    start, stop = int(row["start"]), int(row["stop"])
    if (str(row["schema"]) != SCHEMA or str(row["fingerprint"]) != fingerprint
            or str(row["case"]) != case["name"] or tuple(row["branch_names"]) != BRANCHES
            or str(row["grid_sha256"]) != grid_sha256 or start < 0 or stop <= start
            or row["A0"].shape != (stop-start, len(grid["b_steps_fm"]), 2, len(grid["stored_q"]))):
        raise ValueError(f"checkpoint identity/shape mismatch: {path}")
    if not np.isfinite(row["A0"]).all() or np.any(row["A0"] < 0):
        raise ValueError(f"invalid A0 data: {path}")
    rotations = _case_rotations(case, banks, list(range(start, stop)))
    values = {k: row[k] for k in ("A0", "rotation1", "rotation2")}
    if (_digest_arrays(values) != str(row["data_sha256"])
            or any(not np.array_equal(row[f"rotation{i+1}"], rotations[i]) for i in (0, 1))):
        raise ValueError(f"checkpoint data/rotation mismatch: {path}")
    return row


def inspect_case(out, case, fingerprint, banks, grid, grid_hash):
    directory = out/"cases"/case["name"]
    previous = json.loads((directory/"manifest.json").read_text()) if (directory/"manifest.json").exists() else {}
    if previous and (previous["fingerprint"] != fingerprint or previous["case"] != case):
        raise ValueError(f"case identity changed: {case['name']}")
    registered = previous.get("chunks", {})
    for filename in registered:
        if not (directory/filename).is_file():
            raise ValueError(f"registered checkpoint missing: {filename}")
    count, entries = 0, {}
    for path in sorted(directory.glob("chunk_*.npz")):
        if grid is None:
            raise ValueError("checkpoint without registered grid")
        row = read_chunk(path, fingerprint=fingerprint, case=case, banks=banks, grid=grid, grid_sha256=grid_hash)
        if int(row["start"]) != count:
            raise ValueError(f"checkpoint gap/overlap: {path}")
        digest = sha256(path)
        if path.name in registered and digest != registered[path.name]["sha256"]:
            raise ValueError(f"registered checkpoint changed: {path}")
        session = (out/str(row["session_file"])).resolve()
        if out.resolve() not in session.parents or not session.is_file() or sha256(session) != str(row["session_sha256"]):
            raise ValueError("checkpoint session provenance changed")
        count = int(row["stop"])
        entries[path.name] = {"start": int(row["start"]), "stop": count, "sha256": digest}
    return {**previous, "schema": SCHEMA, "fingerprint": fingerprint, "case": case,
            "completed_pairs": count, "chunks": entries}


def committed_state_hash(state, count):
    """Anchor the root-committed prefix while allowing a case commit ahead of it."""
    chunks = {name: row for name, row in state["chunks"].items() if row["stop"] <= count}
    if max((row["stop"] for row in chunks.values()), default=0) != count:
        raise ValueError("committed checkpoint boundary disappeared")
    payload = {"case": state["case"], "density": state.get("density"),
               "completed_pairs": count, "chunks": chunks}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def run_campaign(cfg, *, config_path=None, prepare_only=False, max_new_pairs=None, selected_cases=None):
    validate_config(cfg)
    if max_new_pairs is not None and max_new_pairs < 0:
        raise ValueError("max_new_pairs must be nonnegative")
    out = Path(cfg["runtime"]["output"])
    target = cfg["orientations"]["n_orientations"]
    cases = cfg["study"]["cases"]
    names = {c["name"] for c in cases}
    selected = names if selected_cases is None else set(selected_cases)
    if not selected <= names:
        raise ValueError(f"unknown selected cases: {selected-names}")
    with OutputLock(out):
        if not (out/"provenance.json").exists() and any(p.name != ".precision.lock" for p in out.iterdir()):
            raise ValueError("unregistered nonempty output; use a new directory")
        banks, bank_meta = rotation_bank(cfg["orientations"], target)
        identity, fingerprint = scientific_identity(cfg, bank_meta)
        if (out/"provenance.json").exists():
            old = json.loads((out/"provenance.json").read_text())
            if old != {"identity": identity, "fingerprint": fingerprint}:
                raise ValueError("scientific identity changed; use a new output directory")
        else:
            atomic_json(out/"provenance.json", {"identity": identity, "fingerprint": fingerprint})
        old_manifest = json.loads((out/"production_manifest.json").read_text()) if (out/"production_manifest.json").exists() else {}
        if old_manifest and (old_manifest.get("schema") != SCHEMA or old_manifest.get("fingerprint") != fingerprint):
            raise ValueError("different campaign schema/fingerprint")
        if target < old_manifest.get("n_orientations", 0):
            raise ValueError("orientation target may be extended but not reduced")
        if old_manifest.get("bank_file"):
            old_bank = out/old_manifest["bank_file"]
            if sha256(old_bank) != old_manifest["bank_sha256"]:
                raise ValueError("registered bank changed")
            with np.load(old_bank, allow_pickle=False) as z:
                for i in (0, 1):
                    previous = z[f"rotation_bank_{i+1}"]
                    if not np.array_equal(previous, banks[i][:len(previous)]):
                        raise ValueError("orientation prefix changed")
        bank_path = out/f"orientation_banks_{target:08d}.npz"
        if not bank_path.exists():
            atomic_npz(bank_path, rotation_bank_1=banks[0], rotation_bank_2=banks[1])
        else:
            with np.load(bank_path, allow_pickle=False) as z:
                if any(not np.array_equal(z[f"rotation_bank_{i+1}"], banks[i]) for i in (0, 1)):
                    raise ValueError("existing bank differs")
        grid_path = out/"grid.npz"
        grid = None
        if grid_path.exists():
            if old_manifest.get("grid_sha256") and sha256(grid_path) != old_manifest["grid_sha256"]:
                raise ValueError("registered grid changed")
            with np.load(grid_path, allow_pickle=False) as z:
                grid = {k: z[k] for k in z.files}
            if set(grid) != set(GRID_KEYS) | {"stored_q", "stored_indices"} or not all(np.isfinite(x).all() for x in grid.values()):
                raise ValueError("invalid stored grid")
        grid_hash = sha256(grid_path) if grid is not None else None
        for name in old_manifest.get("case_manifest_sha256", {}):
            if not (out/"cases"/name/"manifest.json").is_file():
                raise ValueError("registered case manifest missing")
        states = {c["name"]: inspect_case(out, c, fingerprint, banks, grid, grid_hash) for c in cases}
        for case in cases:
            if states[case["name"]]["completed_pairs"] and "density" not in states[case["name"]]:
                _, _, states[case["name"]]["density"] = resolve_case(cfg, case)
        for name, entry in old_manifest.get("cases", {}).items():
            if entry["completed_pairs"]:
                if committed_state_hash(states[name], entry["completed_pairs"]) != entry["committed_state_sha256"]:
                    raise ValueError("previously committed case state changed")
        counts = {c["name"]: 1 if c["beta2"] == 0 and c["beta3"] == 0 else target for c in cases}
        if any(states[name]["completed_pairs"] > counts[name] for name in names):
            raise ValueError("completed prefix exceeds target")
        if old_manifest.get("status") == "completed" and old_manifest["n_orientations"] == target:
            if any(states[name]["completed_pairs"] != counts[name] for name in names):
                raise ValueError("completed campaign lost data")
            if sha256(out/"effective_config.yaml") != old_manifest["effective_config_sha256"]:
                raise ValueError("effective configuration changed")
            for name, expected in old_manifest.get("case_manifest_sha256", {}).items():
                if sha256(out/"cases"/name/"manifest.json") != expected:
                    raise ValueError("case manifest changed")
            return old_manifest
        # A process can stop after atomic chunk rename but before manifest
        # rename. Adopt only fully verified orphan chunks without recomputing.
        for case in cases:
            state = states[case["name"]]
            if state["completed_pairs"]:
                if "density" not in state:
                    _, _, state["density"] = resolve_case(cfg, case)
                atomic_json(out/"cases"/case["name"]/"manifest.json", state)
        # Preparing configs does not initialize CUDA or perform a CPU fallback.
        environment = ({"requested_device": cfg["runtime"]["device"], "executed_forward": False}
                       if prepare_only else configure_runtime(cfg["runtime"]))
        effective = out/"effective_config.yaml"
        temp = effective.with_suffix(".yaml.partial")
        temp.write_text(yaml.safe_dump(cfg, sort_keys=False)); os.replace(temp, effective)
        sessions = out/"sessions"; sessions.mkdir(exist_ok=True)
        session_path = sessions/f"{time.time_ns()}_{os.getpid()}.json"
        atomic_json(session_path, {"fingerprint": fingerprint, "environment": environment,
            "runtime": cfg["runtime"], "y_batch_size": cfg["study"].get("y_batch_size"),
            "target": target, "selected_cases": sorted(selected), "max_new_pairs": max_new_pairs,
            "effective_config_sha256": sha256(effective), "created_utc": datetime.now(timezone.utc).isoformat()})
        manifest = {"schema": SCHEMA, "status": "prepared" if prepare_only else "running",
            "fingerprint": fingerprint, "n_orientations": target, "npairs": counts,
            "target_pairs_total": sum(counts.values()), "new_pairs_this_session": 0,
            "bank_file": bank_path.name, "bank_sha256": sha256(bank_path),
            "environment": environment, "code_sha256": identity["code_sha256"],
            "effective_config_sha256": sha256(effective), "config_path": str(config_path),
            "session_file": str(session_path.relative_to(out))}

        def record(status=None):
            if status:
                manifest["status"] = status
            manifest["completed_pairs"] = sum(x["completed_pairs"] for x in states.values())
            manifest["cases"] = {c["name"]: {"case": c, "completed_pairs": states[c["name"]]["completed_pairs"],
                "target_pairs": counts[c["name"]],
                "committed_state_sha256": committed_state_hash(states[c["name"]], states[c["name"]]["completed_pairs"])} for c in cases}
            if grid is not None:
                manifest["grid_sha256"] = sha256(grid_path)
            manifest["case_manifest_sha256"] = {c["name"]: sha256(out/"cases"/c["name"]/"manifest.json")
                for c in cases if (out/"cases"/c["name"]/"manifest.json").exists()}
            atomic_json(out/"production_manifest.json", manifest)

        record()
        if prepare_only:
            return manifest
        refiner = None
        try:
            for case in cases:
                name = case["name"]
                state = states[name]
                if name not in selected or state["completed_pairs"] >= counts[name]:
                    continue
                if max_new_pairs is not None and manifest["new_pairs_this_session"] >= max_new_pairs:
                    break
                refiner, density = build_refiner(cfg, case)
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
                        current = {k: np.asarray(result[k]) for k in GRID_KEYS}
                        if not all(np.isfinite(value).all() for value in current.values()):
                            raise ValueError("nonfinite kernel grid")
                        if cfg["study"]["qmax"] > current["radii"][-1]:
                            raise ValueError("requested qmax exceeds the native radial grid")
                        indices = np.flatnonzero(current["radii"] <= cfg["study"]["qmax"])
                        current.update(stored_indices=indices, stored_q=current["radii"][indices])
                        if len(indices) < 3 or tuple(result["branch_names"]) != BRANCHES:
                            raise ValueError("invalid kernel branch/grid output")
                        if grid is None:
                            grid = current; atomic_npz(grid_path, **grid); grid_hash = sha256(grid_path)
                        elif any(not np.array_equal(grid[k], current[k]) for k in grid):
                            raise ValueError("kernel grid changed between batches/cases")
                        levels = np.asarray(result["b_level_coefficients"])
                        expected = (len(grid["b_steps_fm"]), end-begin, 2, 3, len(grid["radii"]))
                        if levels.shape != expected or not np.array_equal(levels[0], result["coefficients"]):
                            raise ValueError("invalid kernel coefficient shape")
                        a0 = levels[:, :, :, 0, :][:, :, :, indices].transpose(1, 0, 2, 3).copy()
                        if not np.isfinite(a0).all() or np.any(a0 < 0):
                            raise ValueError("nonfinite/negative A0")
                        blocks.append(a0); elapsed += float(result["elapsed_seconds"])
                    rotations = _case_rotations(case, banks, list(range(start, stop)))
                    values = {"A0": np.concatenate(blocks), "rotation1": rotations[0], "rotation2": rotations[1]}
                    path = directory/f"chunk_{start:08d}_{stop:08d}.npz"
                    if path.exists():
                        raise ValueError("refusing to overwrite immutable checkpoint")
                    atomic_npz(path, **values, schema=np.asarray(SCHEMA), fingerprint=np.asarray(fingerprint),
                        case=np.asarray(name), branch_names=np.asarray(BRANCHES), start=np.asarray(start), stop=np.asarray(stop),
                        grid_sha256=np.asarray(grid_hash), data_sha256=np.asarray(_digest_arrays(values)),
                        session_file=np.asarray(str(session_path.relative_to(out))), session_sha256=np.asarray(sha256(session_path)))
                    state["chunks"][path.name] = {"start": start, "stop": stop, "sha256": sha256(path)}
                    state["completed_pairs"] = stop
                    state["forward_seconds"] = state.get("forward_seconds", 0.)+elapsed
                    atomic_json(directory/"manifest.json", state)
                    manifest["new_pairs_this_session"] += stop-start
                    record()
                    print(f"{name}: {stop}/{counts[name]} pairs; total {manifest['completed_pairs']}/{manifest['target_pairs_total']}; {elapsed:.2f}s/chunk", flush=True)
                del refiner; refiner = None; gc.collect()
                if torch.device(cfg["runtime"]["device"]).type == "cuda":
                    torch.cuda.empty_cache()
            if code_hashes() != identity["code_sha256"]:
                raise ValueError("forward source files changed during execution")
            imported = cfg["orientations"].get("import_bank")
            if imported and sha256(imported) != bank_meta["import_sha256"]:
                raise ValueError("imported orientation bank changed during execution")
            record("completed" if all(states[n]["completed_pairs"] == counts[n] for n in names) else "incomplete")
        except BaseException as error:
            manifest["error"] = f"{type(error).__name__}: {error}"
            record("interrupted")
            raise
        finally:
            if refiner is not None:
                del refiner
        return manifest


def load_case_samples(run, case_name, *, n=None, verify=True):
    """Read one case at a time; never concatenate the entire scan in memory."""
    run = Path(run)
    root = json.loads((run/"production_manifest.json").read_text())
    info = json.loads((run/"cases"/case_name/"manifest.json").read_text())
    if verify and sha256(run/"cases"/case_name/"manifest.json") != root["case_manifest_sha256"][case_name]:
        raise ValueError("case manifest hash changed")
    if n is not None and (isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1):
        raise ValueError("requested prefix n must be a positive integer")
    if n is not None and n > info["completed_pairs"]:
        raise ValueError("requested prefix exceeds completed orientations")
    target = info["completed_pairs"] if n is None else n
    if target < 1:
        raise ValueError("no completed orientations")
    if verify and sha256(run/"grid.npz") != root["grid_sha256"]:
        raise ValueError("grid hash changed")
    with np.load(run/"grid.npz", allow_pickle=False) as z:
        q, steps = z["stored_q"], z["b_steps_fm"]
    samples = np.empty((target, len(steps), 2, len(q)), dtype=float)
    count = 0
    for filename, chunk in sorted(info["chunks"].items()):
        if count >= target:
            break
        path = run/"cases"/case_name/filename
        if chunk["start"] != count or (verify and sha256(path) != chunk["sha256"]):
            raise ValueError("checkpoint gap/hash mismatch")
        with np.load(path, allow_pickle=False) as z:
            if str(z["fingerprint"]) != root["fingerprint"] or str(z["case"]) != case_name:
                raise ValueError("checkpoint identity mismatch")
            end = min(int(z["stop"]), target)
            samples[count:end] = z["A0"][:end-count]
        count = end
    if count != target:
        raise ValueError("incomplete checkpoint prefix")
    return q, steps, samples, info
