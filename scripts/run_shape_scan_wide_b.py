#!/usr/bin/env python3
"""Scan cumulative impact-parameter cutoffs and extract paired full/self radial features."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from upc_deformation.precision_campaign import (
    OutputLock, _case_rotations, _digest_arrays, atomic_json, atomic_npz,
    configure_runtime, rotation_bank, sha256,
)
from upc_deformation.shape_scan_analysis import BRANCH_NAMES, extract_tail_features
from upc_deformation.shape_scan_campaign import load_config

DEFAULT_CUTOFFS = (60., 120., 240., 480., 1000., 10000.)
SCHEMA = "wide_b_shape_scan_v1"


def select_cases(registry, requested, all_cases):
    if all_cases:
        return list(registry)
    if not requested:
        return [next(c for c in registry if c["beta2"] == c["beta3"] == 0
                     and c["density_type"] == "none")]
    names = set(requested)
    missing = names.difference(c["name"] for c in registry)
    if missing:
        raise ValueError(f"Unregistered case(s): {', '.join(sorted(missing))}")
    return [c for c in registry if c["name"] in names]


def cutoff_metrics(case, n, q, cutoffs, mean_a0):
    """Quantities are extracted after averaging the raw spectra over orientation pairs."""
    h = np.diff(q)
    left_weight = h*(q[:-1]/2 + h/6)
    right_weight = h*(q[:-1]/2 + h/3)
    rows = []
    for branch_index, branch in enumerate(BRANCH_NAMES):
        branch_rows = []
        for cutoff, spectra in zip(cutoffs, mean_a0):
            spectrum = spectra[branch_index]
            f = extract_tail_features(q, spectrum)
            radial_yield = float(left_weight @ spectrum[:-1] + right_weight @ spectrum[1:])
            branch_rows.append(dict(case=case["name"], density=case["density_id"],
                beta2=case["beta2"], beta3=case["beta3"], n_pairs=n,
                b_max_fm=float(cutoff), branch=branch, Q10_GeV=f["Q10_GeV"],
                K_tail=f["K_tail"], qdq_yield=radial_yield, censor_reason=f["censor_reason"]))
        for row in branch_rows:
            for key in ("Q10_GeV", "K_tail", "qdq_yield"):
                ref = branch_rows[0][key]
                row[key + "_over_first_cutoff"] = (row[key]/ref
                    if row[key] is not None and ref is not None and ref > 0 else None)
            row["qdq_yield_over_last_cutoff"] = row["qdq_yield"]/branch_rows[-1]["qdq_yield"]
            rows.append(row)
    return rows


def validate_a0(a0, shape):
    if (a0.shape != shape or a0.dtype != np.float64 or not np.isfinite(a0).all()
            or np.any(a0 < 0)):
        raise ValueError(f"Invalid A0 array (expected float64 {shape}, finite and nonnegative)")


def resume_chunks(directory, fingerprint, case, target, banks, shape):
    count = 0
    summed = np.zeros(shape, dtype=np.float64)
    for path in sorted(directory.glob("chunk_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            start, stop = int(data["start"]), int(data["stop"])
            if (str(data["fingerprint"]) != fingerprint or str(data["case"]) != case["name"]
                    or start != count or not start < stop <= target
                    or path.name != f"chunk_{start:08d}_{stop:08d}.npz"):
                raise ValueError(f"Checkpoint identity or orientation interval differs: {path}")
            rotations = _case_rotations(case, banks, list(range(start, stop)))
            if any(not np.array_equal(data[f"rotation{i}"], r) for i, r in enumerate(rotations, 1)):
                raise ValueError(f"Checkpoint orientation bank differs: {path}")
            a0 = data["A0"]
            validate_a0(a0, (stop-start, *shape))
            summed += a0.sum(axis=0)
            count = stop
    return count, summed


def register_grid(out, refiner, cutoffs):
    grid = {"stored_q": np.asarray(refiner.stored_q), "b_nodes": np.asarray(refiner.b_nodes),
            "cutoffs_fm": np.asarray(refiner.cutoffs)}
    if (not np.array_equal(grid["cutoffs_fm"], cutoffs)
            or any(v.ndim != 1 or not len(v) or not np.isfinite(v).all()
                   or np.any(np.diff(v) <= 0) for v in grid.values())
            or len(grid["stored_q"]) < 3):
        raise ValueError("Invalid cumulative cutoff, b or radial grid")
    path = out/"grid.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if any(not np.array_equal(old[k], v) for k, v in grid.items()):
                raise ValueError("Wide-b run grid changed")
    else:
        atomic_npz(path, **grid)
    return grid


def run(args):
    from upc_deformation.shape_scan_wide_b import WideBRefiner

    cfg = load_config(args.config, dict(device=args.device, batch_size=args.batch_size, threads=args.threads))
    if cfg["runtime"]["dtype"] != "float64":
        raise ValueError("The wide-b implementation requires float64")
    cutoffs = np.asarray(args.cutoffs, dtype=np.float64)
    if (cutoffs.ndim != 1 or not len(cutoffs) or not np.isfinite(cutoffs).all()
            or cutoffs[0] <= cfg["study"]["b_min_fm"] or np.any(np.diff(cutoffs) <= 0)):
        raise ValueError("Cutoffs must be finite, strictly increasing and above b_min_fm")
    if (min(args.n_orientations, args.batch_size, args.checkpoint_size) < 1
            or not np.isfinite(args.step_scale) or args.step_scale <= 0):
        raise ValueError("Positive orientation count, batch size, checkpoint size and step scale required")
    cases = select_cases(cfg["study"]["cases"], args.cases, args.all_cases)
    banks, bank_meta = rotation_bank(cfg["orientations"], args.n_orientations)
    identity = {
        "schema": SCHEMA, "physics": cfg["physics"],
        "base_study": {k: v for k, v in cfg["study"].items() if k not in ("cases", "y_batch_size")},
        "cases": cases, "orientation_bank": bank_meta, "n_deformed_pairs": args.n_orientations,
        "device_type": args.device.split(":")[0], "dtype": cfg["runtime"]["dtype"],
        "cutoffs_fm": cutoffs.tolist(), "step_scale": args.step_scale, "branch_names": list(BRANCH_NAMES),
        "forward_sources": {k: sha256(ROOT/"upc_deformation"/k) for k in (
            "shape_scan_wide_b.py", "shape_scan_free_field.py", "shape_scan_physics.py", "observables.py", "forward.py",
            "precision_campaign.py", "precision_forward.py", "diffraction_forward.py",
            "shape_scan_fast_forward.py", "shape_scan_fast_v2_forward.py", "shape_scan_crop_czt.py",
            "shape_scan_czt.py", "source_amplitudes.py", "decay.py", "vendor/upc_forward.py")},
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    out = Path(args.output).resolve()
    provenance = out/"wide_b_run.json"
    if out.exists() and not provenance.exists() and any(
            p.name not in (".precision.lock", "wide_b_run.json.partial") for p in out.iterdir()):
        raise ValueError("Choose a dedicated new wide-b output directory")
    configure_runtime(cfg["runtime"])
    with OutputLock(out):
        record = {"fingerprint": fingerprint, "identity": identity}
        if provenance.exists() and json.loads(provenance.read_text()) != record:
            raise ValueError("Wide-b settings or orientation identity differ; choose another output directory")
        if not provenance.exists():
            atomic_json(provenance, record)
        rows = []
        for case in cases:
            directory = out/"cases"/case["name"]
            directory.mkdir(exist_ok=True, parents=True)
            target = 1 if case["beta2"] == case["beta3"] == 0 else args.n_orientations
            refiner = WideBRefiner(cfg, case, device=args.device, step_scale=args.step_scale, cutoffs=cutoffs)
            grid = register_grid(out, refiner, cutoffs)
            case_id = fingerprint + ":" + _digest_arrays(grid)
            q = grid["stored_q"]
            shape = (len(cutoffs), len(BRANCH_NAMES), len(q))
            count, summed = resume_chunks(directory, case_id, case, target, banks, shape)
            while count < target:
                stop = min(count+args.checkpoint_size, target)
                blocks = []
                for i in range(count, stop, args.batch_size):
                    end = min(i+args.batch_size, stop)
                    rotations = _case_rotations(case, banks, list(range(i, end)))
                    a0 = np.asarray(refiner.calculate(*rotations))
                    validate_a0(a0, (end-i, *shape))
                    blocks.append(a0)
                a0 = np.concatenate(blocks)
                r1, r2 = _case_rotations(case, banks, list(range(count, stop)))
                atomic_npz(directory/f"chunk_{count:08d}_{stop:08d}.npz", A0=a0,
                           start=count, stop=stop, rotation1=r1, rotation2=r2,
                           case=case["name"], fingerprint=case_id)
                summed += a0.sum(axis=0)
                count = stop
                print(f"{case['name']} {count}/{target} orientation pairs", flush=True)
            mean = summed/target
            atomic_npz(directory/"mean_A0.npz", mean_A0=mean, n_pairs=target, stored_q=q,
                       cutoffs_fm=cutoffs, branch_names=np.asarray(BRANCH_NAMES))
            rows.extend(cutoff_metrics(case, target, q, cutoffs, mean))
            del refiner
        path = out/"cutoff_metrics.csv"
        temp = path.with_suffix(".csv.partial")
        with temp.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
        return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT/"configs/shape_scan_a100.yaml"))
    parser.add_argument("--output", required=True, help="Dedicated wide-b campaign directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case", dest="cases", action="append", help="Registered case name (repeatable)")
    group.add_argument("--all-cases", action="store_true", help="Compute the complete registered grid")
    parser.add_argument("--n-orientations", type=int, default=4,
                        help="Ordered pairs per deformed case (default 4); spherical cases use one")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, help="CPU threads (default from configuration)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-size", type=int, default=32)
    parser.add_argument("--step-scale", type=float, default=1.)
    parser.add_argument("--cutoffs", nargs="+", type=float, default=DEFAULT_CUTOFFS)
    args = parser.parse_args(argv)
    print(f"Results: {run(args)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
