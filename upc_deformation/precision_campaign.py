"""Resumable precision campaigns with immutable per-orientation checkpoints.

The scientific identity excludes scheduling choices and the requested sample
count. Increasing the target therefore appends samples without changing earlier
rotations or checkpoints. A changed grid, numerical kernel, dtype or device type
requires a separate output directory.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
BANK_ALGORITHM = "pcg64-seedsequence-shoemake-fixed-block-v1"
CODE_FILES = (
    "scripts/run_panel_d_precision.py", "upc_deformation/precision_campaign.py",
    "upc_deformation/precision_forward.py", "upc_deformation/observables.py",
    "upc_deformation/forward.py", "upc_deformation/source_amplitudes.py",
    "upc_deformation/decay.py", "upc_deformation/vendor/upc_forward.py",
)
GRID_KEYS = ("axis", "radii", "p_full_axis", "b_values", "b_weights",
             "b_steps_fm", "b_level_weights", "y_values", "y_weights")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    with temp.open("w") as stream:
        json.dump(clean(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class OutputLock(AbstractContextManager):
    """OS-managed exclusive lock; a stale PID file does not block resumption."""
    def __init__(self, output):
        self.path = Path(output) / ".precision.lock"
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.stream.close()
            raise RuntimeError(f"Another process owns campaign output: {self.path.parent}") from error
        self.stream.seek(0); self.stream.truncate()
        self.stream.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))
        self.stream.flush()
        return self

    def __exit__(self, *unused):
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def load_config(path, overrides=None):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    for key in ("physics", "study", "orientations", "runtime", "analysis"):
        if not isinstance(cfg.get(key), dict): raise ValueError(f"Missing configuration section: {key}")
    for key, value in (overrides or {}).items():
        if value is None: continue
        section = {"n_orientations": "orientations", "b_step_fm": "study", "n_y": "study", "y_batch_size": "study"}.get(key, "runtime")
        cfg[section][key] = value
    bank = cfg["orientations"].get("import_bank")
    if bank:
        cfg["orientations"]["import_bank"] = str((path.parent / bank).resolve())
    output = Path(cfg["runtime"]["output"])
    cfg["runtime"]["output"] = str((ROOT / output).resolve())
    span = float(cfg["study"]["b_max_fm"]) - float(cfg["study"]["b_min_fm"])
    step = float(cfg["study"]["b_step_fm"])
    if step <= 0 or span <= 0 or not np.isclose(span / step, round(span / step), rtol=0, atol=1e-10):
        raise ValueError("b_step_fm must divide the positive registered b interval exactly")
    cfg["study"]["b_nodes"] = int(round(span / step)) + 1
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    study, orient, runtime = cfg["study"], cfg["orientations"], cfg["runtime"]
    n = orient["n_orientations"]
    if not isinstance(n, int) or n < 4 or n % 4:
        raise ValueError("The deformation target must be at least four and divisible by four")
    if not isinstance(orient["block_size"], int) or orient["block_size"] < 1:
        raise ValueError("The fixed orientation block size must be positive")
    if not isinstance(orient["seed"], int) or orient["seed"] < 0:
        raise ValueError("The orientation seed must be a nonnegative integer")
    if runtime["dtype"] not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if torch.device(runtime["device"]).type not in ("cpu", "cuda"):
        raise ValueError("Only explicitly requested CPU or CUDA execution is supported")
    if runtime["fft_mode"] not in ("combined", "separate"):
        raise ValueError("fft_mode must be combined or separate")
    if int(runtime["batch_size"]) < 1 or int(runtime["threads"]) < 1:
        raise ValueError("batch_size and threads must be positive")
    if study["longitudinal_units"] != "physical" or study["b_min_fm"] != 12 or study["b_max_fm"] != 60:
        raise ValueError("This campaign retains physical longitudinal units and b=12–60 fm")
    if study["n_y"] < 1 or study["n_phi"] < 8 or study["n_phi"] % 4 or study["n_radial"] < 2:
        raise ValueError("Invalid y/radial/angular quadrature")
    if int(study.get("y_batch_size", study["n_y"])) < 1:
        raise ValueError("y_batch_size must be positive")
    cases = study["cases"]
    if [c["name"] for c in cases] != ["spherical", "beta2", "beta3", "skin"]:
        raise ValueError("Register exactly spherical, beta2, beta3, skin in that order")
    if any(c["beta2"] != 0 or c["beta3"] != 0 for c in (cases[0], cases[3])):
        raise ValueError("The spherical and skin references must be undeformed")


def rotation_bank(orientations, target):
    """Independent nuclear streams; fixed-size blocks make all prefixes stable."""
    imported = orientations.get("import_bank")
    prefix = []
    imported_hash = None
    if imported:
        imported_hash = sha256(imported)
        expected = orientations.get("import_sha256")
        if expected and expected != imported_hash: raise ValueError("Imported orientation-bank hash differs")
        with np.load(imported, allow_pickle=False) as saved:
            prefix = [np.asarray(saved[f"rotation_bank_{j}"], dtype=np.float64) for j in (1, 2)]
        if prefix[0].shape != prefix[1].shape or prefix[0].ndim != 3 or prefix[0].shape[1:] != (3, 3):
            raise ValueError("Imported bank must contain matching (n,3,3) rotation arrays")
        if not len(prefix[0]): raise ValueError("Imported bank is empty")
    n_prefix = len(prefix[0]) if prefix else 0
    size = orientations["block_size"]
    result = []
    for nucleus in (1, 2):
        blocks = [prefix[nucleus-1]] if prefix else []
        for block in range((max(0, target-n_prefix) + size-1) // size):
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([orientations["seed"], nucleus, block])))
            u = rng.random((size, 3))
            x = np.sqrt(1-u[:, 0])*np.sin(2*np.pi*u[:, 1])
            y = np.sqrt(1-u[:, 0])*np.cos(2*np.pi*u[:, 1])
            z = np.sqrt(u[:, 0])*np.sin(2*np.pi*u[:, 2])
            w = np.sqrt(u[:, 0])*np.cos(2*np.pi*u[:, 2])
            rows = np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                             2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                             2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(size, 3, 3)
            blocks.append(rows)
        bank = np.concatenate(blocks)[:target]
        if (not np.isfinite(bank).all() or not np.allclose(bank @ bank.transpose(0, 2, 1), np.eye(3), atol=2e-5, rtol=0)
                or not np.allclose(np.linalg.det(bank), 1, atol=2e-5, rtol=0)):
            raise ValueError("Orientation bank contains invalid SO(3) matrices")
        if len(np.unique(bank.reshape(target, -1), axis=0)) != target:
            raise ValueError("Orientation bank contains repeated rotations")
        result.append(bank)
    metadata = {"algorithm": BANK_ALGORITHM, "seed": orientations["seed"], "block_size": size,
                "import_sha256": imported_hash, "imported_prefix_count": n_prefix,
                "precision": "float64; imported float32 values retained exactly"}
    return result, metadata


def code_hashes():
    return {p: sha256(ROOT / p) for p in CODE_FILES}


def scientific_identity(cfg, bank_metadata):
    study = deepcopy(cfg["study"])
    for key in ("n_orientations", "orientation_seed", "threads", "y_batch_size"):
        study.pop(key, None)
    identity = {"schema_version": 1, "physics": cfg["physics"], "study": study,
                "analysis": cfg["analysis"], "orientation_bank": bank_metadata,
                "numerics": {"dtype": cfg["runtime"]["dtype"],
                    "device_type": torch.device(cfg["runtime"]["device"]).type,
                    "fft_mode": cfg["runtime"]["fft_mode"], "tf32": False, "amp": False},
                "code_sha256": code_hashes()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return identity, fingerprint


def configure_runtime(runtime):
    device = torch.device(runtime["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback is permitted")
        torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    torch.set_num_threads(int(runtime["threads"]))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": str(device), "device_type": device.type, "dtype": runtime["dtype"],
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32, "amp_enabled": False,
            "threads": torch.get_num_threads(), "batch_size": int(runtime["batch_size"]),
            "host": socket.gethostname(), "pid": os.getpid()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        meta.update(device_name=props.name, device_total_memory_bytes=props.total_memory,
                    device_capability=list(torch.cuda.get_device_capability(device)))
    return meta


def build_refiner(cfg, case):
    from .precision_forward import PrecisionPanelDRefiner
    r = cfg["runtime"]
    return PrecisionPanelDRefiner(cfg, case, device=r["device"], dtype=r["dtype"], fft_mode=r["fft_mode"])


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _digest_arrays(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode()); h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes())
    return h.hexdigest()


def _case_rotations(case, banks, indices):
    if case["beta2"] == 0 and case["beta3"] == 0:
        return [np.broadcast_to(np.eye(3), (len(indices), 3, 3)).copy() for _ in (1, 2)]
    return [b[indices] for b in banks]


def _read_checkpoint(path, fingerprint, case, index, rotations, nr):
    with np.load(path, allow_pickle=False) as saved:
        row = {k: saved[k] for k in saved.files}
    if (str(row["fingerprint"]) != fingerprint or str(row["case"]) != case["name"]
            or int(row["index"]) != index or row["coefficients"].shape != (3, nr)
            or row["b_level_coefficients"].ndim != 3 or row["b_level_coefficients"].shape[1:] != (3, nr)):
        raise ValueError(f"Checkpoint identity/shape mismatch: {path}")
    numeric = {k: row[k] for k in ("coefficients", "b_level_coefficients", "rotation1", "rotation2")}
    if (not all(np.isfinite(v).all() for v in numeric.values())
            or _digest_arrays(numeric) != str(row["data_sha256"])
            or not all(np.array_equal(row[f"rotation{j+1}"], value) for j, value in enumerate(rotations))):
        raise ValueError(f"Checkpoint data/rotation mismatch: {path}")
    return row


def run_campaign(cfg, *, config_path=None, prepare_only=False, max_new_pairs=None):
    """Execute a bounded or complete run; interruption preserves atomic samples."""
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
        cases = cfg["study"]["cases"]
        counts = {c["name"]: (1 if c["beta2"] == 0 and c["beta3"] == 0 else target) for c in cases}
        available = {name: set() for name in counts}
        checkpoint_hashes = {}
        nr = cfg["study"]["n_radial"]
        grid_path = out / "grid.npz"
        grid = None
        if grid_path.exists():
            with np.load(grid_path, allow_pickle=False) as z: grid = {k: z[k] for k in z.files}
            if prior.get("grid_sha256") and sha256(grid_path) != prior["grid_sha256"]: raise ValueError("Registered grid metadata changed")
        for case in cases:
            name = case["name"]; directory = out / "checkpoints" / name; directory.mkdir(parents=True, exist_ok=True)
            for index in range(counts[name]):
                path = directory / f"{index:08d}.npz"
                if not path.exists(): continue
                rotations = [r[0] for r in _case_rotations(case, banks, [index])]
                row = _read_checkpoint(path, fingerprint, case, index, rotations, nr)
                if grid is None or str(row["grid_sha256"]) != sha256(grid_path): raise ValueError("Checkpoint grid identity differs")
                session_source = (out / str(row["session_file"])).resolve()
                if out.resolve() not in session_source.parents or sha256(session_source) != str(row["session_sha256"]):
                    raise ValueError("Checkpoint runtime provenance changed")
                relative = str(path.relative_to(out)); digest = sha256(path)
                if relative in prior.get("checkpoint_sha256", {}) and prior["checkpoint_sha256"][relative] != digest:
                    raise ValueError("Previously completed checkpoint changed")
                checkpoint_hashes[relative] = digest; available[name].add(index)
        manifest = {"status": "prepared" if prepare_only else "running", "fingerprint": fingerprint,
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
                    if (coeff.shape != (len(indices), 3, nr) or levels.ndim != 4 or levels.shape[1:] != coeff.shape
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
                        atomic_npz(path, **values, fingerprint=np.asarray(fingerprint), case=np.asarray(name), index=np.asarray(index),
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
                    collected.append(_read_checkpoint(path, fingerprint, case, index, rotations, nr))
                rotations = _case_rotations(case, banks, list(range(counts[name])))
                path = out / f"{name}.npz"
                atomic_npz(path, orientation_coefficients=np.stack([r["coefficients"] for r in collected]),
                           b_level_coefficients=np.stack([r["b_level_coefficients"] for r in collected], axis=1),
                           rotation_bank_1=rotations[0], rotation_bank_2=rotations[1],
                           fingerprint=np.asarray(fingerprint), **grid)
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
