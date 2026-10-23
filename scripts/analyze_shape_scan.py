#!/usr/bin/env python3
"""Analyze registered single-image shape-scan checkpoints, one case at a time.

No forward calculation is performed. Bootstrap replicates average raw A0
before extracting Q10, B_mid, K_tail and finite-range mean pT, retaining their
joint dependence.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"upc_shape_scan_matplotlib"))

import numpy as np
import yaml

from upc_deformation.shape_scan_analysis import (
    BRANCH_NAMES, FEATURE_NAMES, extract_tail_features, summarize_samples,
)
from upc_deformation.shape_scan_campaign import SCHEMA, load_case_samples
from upc_deformation.precision_campaign import atomic_json, atomic_npz, sha256

REPORT_SCHEMA = "single_image_shape_scan_report_v2"
ANALYSIS_FILES = ("scripts/analyze_shape_scan.py", "upc_deformation/shape_scan_analysis.py",
                  "upc_deformation/shape_scan_campaign.py")
DENSITY_FIELDS = ("R_ws_p_fm", "R_ws_n_fm", "a_p_fm", "a_n_fm", "proton_rms_fm",
                  "neutron_rms_fm", "delta_r_np_fm")
FIGURE_CONTRACT = {
    "conclusion": "Explore the four correlated single-image responses across the registered parameter grid.",
    "archetype": "quantitative grid", "backend": "Python/matplotlib",
    "panels": "One density variant per panel; one full-J observable per figure, finest b level only.",
    "comparability": "Shared colour limits and beta coordinates within each figure; no response-based cut selection.",
    "scope": "Exploratory model response, not measured data, convergence or parameter selectivity evidence.",
    "statistics": "Own-feature extraction from the orientation mean; orientation-pair/block MC errors in CSV, no event errors or hypothesis tests.",
    "exports": ["PNG (300 dpi)", "PDF (TrueType text)", "SVG (editable text)"],
    "source_data": "metrics.csv", "missing": "Grey cells: x=censored, open circle=case unavailable; no interpolation across cells.",
}


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _inside(run, relative):
    path = (run/relative).resolve()
    if run not in path.parents:
        raise ValueError(f"input path escapes run: {relative}")
    return path


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def load_header(run, *, n=None, allow_partial=False):
    """Verify registry/provenance and select complete cases before writing output."""
    run = Path(run).resolve()
    root = json.loads((run/"production_manifest.json").read_text())
    if root.get("schema") != SCHEMA:
        raise ValueError("not a registered single-image A0 shape scan")
    inputs = {"production_manifest.json": sha256(run/"production_manifest.json")}
    for path, digest in (("effective_config.yaml", root["effective_config_sha256"]),
                         (root["bank_file"], root["bank_sha256"]),
                         ("grid.npz", root.get("grid_sha256"))):
        if digest is None or sha256(_inside(run, path)) != digest:
            raise ValueError(f"registered checksum mismatch: {path}")
        inputs[path] = digest
    provenance = json.loads((run/"provenance.json").read_text())
    identity_hash = hashlib.sha256(json.dumps(provenance["identity"], sort_keys=True,
                                             allow_nan=False).encode()).hexdigest()
    if identity_hash != root["fingerprint"] or provenance["fingerprint"] != identity_hash:
        raise ValueError("scientific identity fingerprint mismatch")
    inputs["provenance.json"] = sha256(run/"provenance.json")
    cfg = yaml.safe_load((run/"effective_config.yaml").read_text())
    cases = cfg["study"]["cases"]
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names) or set(root["cases"]) != set(names):
        raise ValueError("configuration/production case registry mismatch")
    if provenance["identity"]["study"]["cases"] != cases:
        raise ValueError("configuration/provenance case identity mismatch")
    target = _integer(root["n_orientations"], "registered n_orientations")
    requested = target if n is None else _integer(n, "--n")
    if requested > target:
        raise ValueError("--n exceeds the registered orientation target; no silent truncation")
    if requested == 1 and any(case["beta2"] != 0 or case["beta3"] != 0 for case in cases):
        raise ValueError("--n=1 is not a deterministic deformed case; use at least two orientations")
    chosen, missing = [], []
    for case in cases:
        name, row = case["name"], root["cases"][case["name"]]
        expected = 1 if case["beta2"] == case["beta3"] == 0 else target
        if row["case"] != case or row["target_pairs"] != expected or root["npairs"][name] != expected:
            raise ValueError(f"case target/identity mismatch: {name}")
        count = row["completed_pairs"]
        if not 0 <= count <= expected:
            raise ValueError(f"invalid completed count: {name}")
        if count == expected:
            chosen.append(case)
        else:
            missing.append({"case": name, "completed_pairs": count, "target_pairs": expected,
                            "reason": "case_incomplete"})
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} registered cases incomplete; use --allow-partial for complete cases only")
    if not chosen:
        raise ValueError("no complete registered cases to analyze")
    with np.load(_inside(run, root["bank_file"]), allow_pickle=False) as data:
        banks = [data[f"rotation_bank_{i}"] for i in (1, 2)]
    if any(bank.shape != (target, 3, 3) or not np.isfinite(bank).all() for bank in banks):
        raise ValueError("registered rotation bank shape/content mismatch")
    with np.load(run/"grid.npz", allow_pickle=False) as data:
        q, steps = data["stored_q"], data["b_steps_fm"]
    if (steps.ndim != 1 or not len(steps) or not np.isfinite(steps).all()
            or np.any(steps <= 0) or len(np.unique(steps)) != len(steps)):
        raise ValueError("invalid registered b refinements")
    return cfg, root, chosen, missing, requested, q, steps, banks, inputs


def _verify_case(run, case, info, root, banks, used, inputs):
    """Bind paired bootstrap labels to the saved rotations, not just a seed."""
    name = case["name"]
    if (info["case"] != case or info["fingerprint"] != root["fingerprint"]
            or info["completed_pairs"] != root["cases"][name]["completed_pairs"]):
        raise ValueError(f"case manifest identity/count mismatch: {name}")
    if not isinstance(info.get("density"), dict):
        raise ValueError(f"missing resolved density metadata: {name}")
    relative = f"cases/{name}/manifest.json"
    inputs[relative] = root["case_manifest_sha256"][name]
    deterministic = case["beta2"] == case["beta3"] == 0
    count = 0
    for filename, entry in sorted(info["chunks"].items()):
        if count >= used:
            break
        relative = f"cases/{name}/{filename}"
        path = _inside(run, relative)
        start, stop = entry["start"], entry["stop"]
        if start != count or stop <= start:
            raise ValueError("checkpoint ordering mismatch")
        with np.load(path, allow_pickle=False) as data:
            if (str(data["schema"]) != SCHEMA or tuple(data["branch_names"]) != BRANCH_NAMES
                    or str(data["grid_sha256"]) != root["grid_sha256"]
                    or int(data["start"]) != start or int(data["stop"]) != stop):
                raise ValueError("checkpoint grid/branch/range mismatch")
            end = min(stop, used)
            for i, bank in enumerate(banks, 1):
                expected = np.eye(3)[None] if deterministic else bank[start:end]
                if not np.array_equal(data[f"rotation{i}"][:end-start], expected):
                    raise ValueError(f"saved rotations do not match shared bank: {name}")
            session = str(data["session_file"])
            session_hash = str(data["session_sha256"])
            if sha256(_inside(run, session)) != session_hash:
                raise ValueError("checkpoint session checksum mismatch")
            inputs[session] = session_hash
        inputs[relative] = entry["sha256"]  # load_case_samples verifies this hash.
        count = stop


def _base_row(case, density, level, step, branch, n):
    return {"case": case["name"], "density_id": case["density_id"],
            "density_type": case["density_type"], "density_mode": case["density_mode"],
            "beta2": case["beta2"], "beta3": case["beta3"], "b_level": level,
            "b_step_fm": float(step), "branch": branch, "n_orientations": n,
            **{key: density.get(key) for key in DENSITY_FIELDS},
            "density_metadata_json": _json(density)}


def _metric_row(base, channel):
    p = channel["point"]
    row = {**base, "status": p["status"], "censor_reason": p["censor_reason"],
           **{key: p[key] for key in ("Q50_GeV", "Q1_GeV", "B_tail", *FEATURE_NAMES)},
           "peak_q_GeV": p["peak_q_GeV"], "peak_height": p["peak_height"],
           "point_json": _json(p)}
    for i, feature in enumerate(FEATURE_NAMES):
        row[feature+"_orientation_se"] = (None if channel["orientation_se"] is None
                                          else channel["orientation_se"][i])
        row[feature+"_marginal_orientation_se"] = channel["marginal_orientation_se"][i]
        row[feature+"_conditional_orientation_se"] = (
            None if channel["conditional_orientation_se"] is None
            else channel["conditional_orientation_se"][i])
    for key in ("valid_fraction", "censored_fraction", "n_joint_valid", "n_bootstrap", "covariance_scope"):
        row[key] = channel[key]
    for key in ("covariance", "conditional_covariance", "feature_valid_fraction", "censor_reason_counts"):
        row[key+"_json"] = _json(channel[key])
    return row


def _contrast_statistics(point, samples):
    feature_valid = np.isfinite(samples)
    valid = feature_valid.all(axis=1)
    complete = bool(valid.all() and np.isfinite(point).all())
    cov = np.cov(samples[valid], rowvar=False, ddof=1) if valid.sum() >= 2 else None
    se = None if cov is None else np.sqrt(np.maximum(0, np.diag(cov)))
    marginal_se = [float(samples[:, i].std(ddof=1))
                   if feature_valid[:, i].all() and np.isfinite(point[i]) else None
                   for i in range(len(FEATURE_NAMES))]
    return {"value": [float(v) if np.isfinite(v) else None for v in point],
            "covariance": cov.tolist() if complete and cov is not None else None,
            "orientation_se": se.tolist() if complete and se is not None else None,
            "marginal_orientation_se": marginal_se,
            "conditional_covariance": cov.tolist() if not complete and cov is not None else None,
            "conditional_orientation_se": se.tolist() if not complete and se is not None else None,
            "valid_fraction": float(valid.mean()), "n_joint_valid": int(valid.sum()),
            "covariance_scope": "all_replicates" if complete else "conditional_on_all_features_observed"}


def _pairing(a, b):
    aa, bb = a["bootstrap"], b["bootstrap"]
    if aa["draws"] != bb["draws"] or aa["orientation_bank_id"] != bb["orientation_bank_id"]:
        raise ValueError("paired contrast requires the same registered orientation bank and draw count")
    if aa["method"] != "deterministic" and bb["method"] != "deterministic":
        if aa["resampling_plan_id"] != bb["resampling_plan_id"]:
            raise ValueError("paired contrast requires the same resampling plan")
    return "shared_ordered_bank_and_plan; deterministic_references_are_constant"


def paired_rows(cases, steps):
    """Cross-case, branch and b-grid diagnostics from shared replicate indices."""
    rows, notes = [], []
    finest = int(np.argmin(steps))
    names = list(cases)
    comparisons = []
    for name in names:
        a = cases[name]; case = a["case"]
        sphere = [key for key, row in cases.items() if row["case"]["density_id"] == case["density_id"]
                  and row["case"]["beta2"] == row["case"]["beta3"] == 0]
        if not sphere:
            notes.append(f"{name}: same-density sphere absent; spherical response skipped")
        for ref in sphere:
            if ref != name:
                comparisons.extend(("same_density_vs_sphere", name, ref, level, branch, level, branch)
                                   for level in range(len(steps)) for branch in range(2))
        none = [key for key, row in cases.items() if row["case"]["density_type"] == "none"
                and row["case"]["beta2"] == case["beta2"] and row["case"]["beta3"] == case["beta3"]]
        if not none:
            notes.append(f"{name}: matched-beta none case absent; density response skipped")
        for ref in none:
            if ref != name:
                comparisons.extend(("matched_beta_vs_none", name, ref, level, branch, level, branch)
                                   for level in range(len(steps)) for branch in range(2))
        comparisons.extend(("full_vs_self", name, name, level, 0, level, 1) for level in range(len(steps)))
        comparisons.extend(("b_refinement_vs_finest", name, name, level, branch, finest, branch)
                           for level in range(len(steps)) if level != finest for branch in range(2))
    for kind, name, ref, level, branch, ref_level, ref_branch in comparisons:
        a, b = cases[name], cases[ref]
        pairing = _pairing(a, b)
        x = np.array([np.nan if v is None else v for v in a["channels"][level][branch]["value"]])
        y = np.array([np.nan if v is None else v for v in b["channels"][ref_level][ref_branch]["value"]])
        xx, yy = a["replicate_features"][level, branch], b["replicate_features"][ref_level, ref_branch]
        ratio = np.divide(x, y, out=np.full_like(x, np.nan), where=np.isfinite(y) & (y != 0))
        ratio_reps = np.divide(xx, yy, out=np.full_like(xx, np.nan), where=np.isfinite(yy) & (yy != 0))
        difference, ratio_stats = _contrast_statistics(x-y, xx-yy), _contrast_statistics(ratio, ratio_reps)
        for i, feature in enumerate(FEATURE_NAMES):
            row = {"contrast": kind, "case": name, "reference_case": ref,
                   "density_id": a["case"]["density_id"], "reference_density_id": b["case"]["density_id"],
                   "beta2": a["case"]["beta2"], "beta3": a["case"]["beta3"],
                   "b_level": level, "b_step_fm": float(steps[level]), "branch": BRANCH_NAMES[branch],
                   "reference_b_level": ref_level, "reference_b_step_fm": float(steps[ref_level]),
                   "reference_branch": BRANCH_NAMES[ref_branch], "feature": feature, "pairing": pairing}
            for prefix, stats in (("difference", difference), ("ratio", ratio_stats)):
                row[prefix] = stats["value"][i]
                for key in ("orientation_se", "marginal_orientation_se", "conditional_orientation_se"):
                    row[prefix+"_"+key] = None if stats[key] is None else stats[key][i]
                for key in ("valid_fraction", "n_joint_valid", "covariance_scope"):
                    row[prefix+"_"+key] = stats[key]
                for key in ("covariance", "conditional_covariance"):
                    row[prefix+"_"+key+"_json"] = _json(stats[key])
            row["fractional_response"] = None if row["ratio"] is None else row["ratio"]-1
            rows.append(row)
    return rows, notes


def plot_responses(rows, cfg, steps, output, n):
    """Registered cells only; identical colour limits across density variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.right": False,
        "axes.spines.top": False, "legend.frameon": False})
    variants = cfg["scan"]["density_variants"]
    b2, b3 = (np.sort(np.asarray(cfg["scan"][key], float)) for key in ("beta2_values", "beta3_values"))
    finest = int(np.argmin(steps))
    lookup = {(row["density_id"], row["beta2"], row["beta3"]): row for row in rows
              if row["b_level"] == finest and row["branch"] == "full"}

    def edges(nodes):
        if len(nodes) == 1:
            return np.array([nodes[0]-.025, nodes[0]+.025])
        mid = (nodes[1:]+nodes[:-1])/2
        return np.r_[2*nodes[0]-mid[0], mid, 2*nodes[-1]-mid[-1]]

    def ticks(nodes):
        # The production grid has ten long repeating-decimal coordinates.
        # Keep all cells, but label a readable subset of their exact positions.
        return nodes[np.linspace(0, len(nodes)-1, min(5, len(nodes)), dtype=int)]

    files, qa = [], {}
    labels = {"Q10_GeV": r"$Q_{10}$ [GeV]", "B_mid": r"$B_{\rm mid}$ [GeV$^{-2}$]",
              "K_tail": r"$K_{\rm tail}$", "mean_pT_GeV": r"$\langle p_T\rangle_{\rm saved\ range}$ [GeV]"}
    for feature in FEATURE_NAMES:
        label = labels[feature]
        observed = [row[feature] for row in lookup.values() if row[feature] is not None]
        lo, hi = (min(observed), max(observed)) if observed else (0., 1.)
        if lo == hi:
            pad = max(abs(lo)*.01, 1e-6); lo, hi = lo-pad, hi+pad
        norm = Normalize(lo, hi)
        cmap = plt.get_cmap("cividis").copy(); cmap.set_bad("#e8e8e8")
        cols = min(3, len(variants)); nrows = (len(variants)+cols-1)//cols
        fig, axes = plt.subplots(nrows, cols, figsize=(3.0*cols+.45, 2.65*nrows+.65),
                                 squeeze=False, layout="constrained")
        active = []
        for index, (ax, variant) in enumerate(zip(axes.flat, variants)):
            active.append(ax)
            values = np.full((len(b3), len(b2)), np.nan)
            for iy, y in enumerate(b3):
                for ix, x in enumerate(b2):
                    row = lookup.get((variant["id"], x, y))
                    if row is None or row[feature] is None:
                        if row is None:
                            ax.scatter(x, y, marker="o", s=22, facecolors="none",
                                       edgecolors="#555555", linewidths=.8, zorder=3)
                        else:
                            ax.scatter(x, y, marker="x", s=22, color="#555555", linewidths=.8, zorder=3)
                    else:
                        values[iy, ix] = row[feature]
            mesh = ax.pcolormesh(edges(b2), edges(b3), np.ma.masked_invalid(values), cmap=cmap,
                                 norm=norm, shading="flat", rasterized=False)
            ax.set(xticks=ticks(b2), yticks=ticks(b3), xlabel=r"$\beta_2$", ylabel=r"$\beta_3$",
                   title=f"{variant.get('label', variant['id'])}\n{variant['type']}; {variant['mode']}")
            ax.set_xticklabels([f"{value:.3g}" for value in ticks(b2)])
            ax.set_yticklabels([f"{value:.3g}" for value in ticks(b3)])
            ax.tick_params(labelsize=7)
            ax.text(-.13, 1.11, chr(ord("a")+index), transform=ax.transAxes, fontweight="bold")
        for ax in list(axes.flat)[len(variants):]:
            ax.set_visible(False)
        if observed:
            fig.colorbar(mesh, ax=active, label=label, shrink=.85, pad=.025)
        else:
            active[0].text(.5, .5, "No observed values", ha="center", transform=active[0].transAxes)
        fig.suptitle(f"{label}: full J, finest b step {steps[finest]:g} fm\n"
                     f"N={n} orientation pairs (spherical: 1); exploratory model response", fontsize=10)
        fig.supxlabel("Registered cells only; x: censored, open circle: unavailable. MC errors: metrics.csv", fontsize=7)
        fig.canvas.draw()
        qa[feature] = {"observed_cells": len(observed), "colour_limits": [lo, hi] if observed else None,
                       "panels": len(variants), "figure_inches": list(fig.get_size_inches()),
                       "renderer_draw_completed": True}
        for extension in ("png", "pdf", "svg"):
            path = output/f"{feature}_response.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            files.append(path.name)
        plt.close(fig)
    return files, qa


def analyze_run(run, output, *, n=None, bootstrap_draws=None, allow_partial=False, no_plots=False):
    run, output = Path(run).resolve(), Path(output).resolve()
    analysis_hashes = {path: sha256(ROOT/path) for path in ANALYSIS_FILES}
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("refusing to overwrite a nonempty analysis output")
    cfg, production, selected, missing, requested, q, steps, banks, inputs = load_header(
        run, n=n, allow_partial=allow_partial)
    settings = cfg["analysis"]
    draws = settings.get("bootstrap_draws", 1024) if bootstrap_draws is None else bootstrap_draws
    options = {"bootstrap_draws": _integer(draws, "bootstrap_draws", 2),
               "bootstrap_seed": _integer(settings.get("bootstrap_seed", 20261024), "bootstrap_seed", 0),
               "bootstrap_block_size": _integer(settings.get("bootstrap_block_size", 32), "bootstrap_block_size"),
               "bootstrap_batch_size": _integer(settings.get("bootstrap_batch_size", 32), "bootstrap_batch_size"),
               "orientation_bank_id": production["bank_sha256"]}
    block = options["bootstrap_block_size"]
    if requested > block and requested % block and any(case["beta2"] or case["beta3"] for case in selected):
        raise ValueError("--n must be divisible by bootstrap_block_size above the small-bank fallback; no samples discarded")
    prefixes = settings.get("prefixes", settings.get("orientation_prefixes", []))
    prefixes = sorted({_integer(p, "analysis.prefixes") for p in prefixes if p <= requested} | {requested})
    output.mkdir(parents=True, exist_ok=True)
    (output/"cases").mkdir()
    cases, rows, prefix_rows, npz = {}, [], [], {}
    radial_means = {"q": q, "b_steps_fm": steps, "branch_names": np.asarray(BRANCH_NAMES)}
    for case in selected:
        name = case["name"]
        used = 1 if case["beta2"] == case["beta3"] == 0 else requested
        cq, cs, samples, info = load_case_samples(run, name, n=used, verify=True)
        if len(samples) != used or not np.array_equal(cq, q) or not np.array_equal(cs, steps):
            raise ValueError("requested prefix/grid was not returned exactly")
        _verify_case(run, case, info, production, banks, used, inputs)
        result = summarize_samples(q, samples, **options)
        result.update(case=case, density=info["density"], b_steps_fm=steps.tolist())
        result.update(nominal_qmax_GeV=float(cfg["study"]["qmax"]), effective_saved_qmax_GeV=float(q[-1]))
        npz_key = f"cases__{name}__replicate_features"
        result["replicate_npz_key"] = npz_key
        result["radial_mean_npz_key"] = f"cases__{name}__mean_A0"
        radial_means[result["radial_mean_npz_key"]] = result["mean_A0"]
        npz[npz_key] = result["replicate_features"]
        npz[f"cases__{name}__replicate_valid"] = result["replicate_valid"]
        for level, step in enumerate(steps):
            for branch, label in enumerate(BRANCH_NAMES):
                base = _base_row(case, info["density"], level, step, label, used)
                base.update(nominal_qmax_GeV=float(cfg["study"]["qmax"]), effective_saved_qmax_GeV=float(q[-1]))
                rows.append(_metric_row(base, result["channels"][level][branch]))
        for prefix in prefixes:
            count = 1 if used == 1 else prefix
            mean = samples[:count].mean(axis=0, dtype=float)
            for level, step in enumerate(steps):
                for branch, label in enumerate(BRANCH_NAMES):
                    point = extract_tail_features(q, mean[level, branch])
                    prefix_rows.append({**_base_row(case, info["density"], level, step, label, count),
                        "requested_prefix": prefix, "is_final_analysis_prefix": prefix == requested,
                        "status": point["status"], "censor_reason": point["censor_reason"],
                        **{key: point[key] for key in ("Q50_GeV", "Q1_GeV", *FEATURE_NAMES)},
                        "point_json": _json(point)})
        compact = {key: value for key, value in result.items()
                   if key not in ("q", "mean_A0", "replicate_features", "replicate_valid")}
        atomic_json(output/"cases"/f"{name}.json", compact)
        cases[name] = {**compact, "replicate_features": result["replicate_features"]}
        del samples, result, mean
        gc.collect()
        print(f"analyzed {name}: {used} orientation pairs", flush=True)
    _write_csv(output/"metrics.csv", rows)
    _write_csv(output/"prefix_stability.csv", prefix_rows)
    responses, comparison_notes = paired_rows(cases, steps)
    _write_csv(output/"paired_response.csv", responses)
    npz.update(feature_names=np.asarray(FEATURE_NAMES), branch_names=np.asarray(BRANCH_NAMES),
               case_names=np.asarray(list(cases)), b_steps_fm=steps,
               orientation_bank_id=np.asarray(production["bank_sha256"]))
    atomic_npz(output/"bootstrap_replicates.npz", **npz)
    atomic_npz(output/"radial_means.npz", **radial_means)
    atomic_json(output/"missing_cases.json", missing)
    plot_files, plot_qa = ([], {}) if no_plots else plot_responses(rows, cfg, steps, output, requested)
    if sha256(run/"production_manifest.json") != inputs["production_manifest.json"]:
        raise ValueError("production manifest changed during analysis; report is not a completed snapshot")
    if {path: sha256(ROOT/path) for path in ANALYSIS_FILES} != analysis_hashes:
        raise ValueError("analysis source files changed during execution")
    if sha256(_inside(run, production["bank_file"])) != inputs[production["bank_file"]]:
        raise ValueError("registered rotation bank changed during analysis")
    report = {
        "schema": REPORT_SCHEMA, "status": "completed", "production_fingerprint": production["fingerprint"],
        "requested_n": requested, "allow_partial": allow_partial, "registered_cases": len(cfg["study"]["cases"]),
        "analyzed_cases": list(cases), "missing_cases": missing, "feature_names": list(FEATURE_NAMES),
        "branch_names": list(BRANCH_NAMES), "b_steps_fm": steps.tolist(), "prefixes": prefixes,
        "nominal_qmax_GeV": float(cfg["study"]["qmax"]), "effective_saved_qmax_GeV": float(q[-1]),
        "mean_pT_support_GeV": [float(q[0]), float(q[-1])],
        "mean_pT_definition": "integral(q^2 A0 dq)/integral(q A0 dq), exact moments of piecewise-linear A0 on saved support",
        "bootstrap_options": options, "case_bootstrap": {key: row["bootstrap"] for key, row in cases.items()},
        "input_sha256": inputs, "analysis_code_sha256": analysis_hashes,
        "comparison_notes": comparison_notes, "figure_contract": FIGURE_CONTRACT,
        "figure_qa": plot_qa, "figures": plot_files,
        "limitations": [
            "Orientation Monte Carlo integration uncertainty only; not experimental event statistics.",
            "Nonoverlapping blocks approximate orientation resampling; all b levels/branches and same-plan cases share counts.",
            "Q10, B_mid, K_tail and mean pT are correlated; retain their joint 4x4 covariance and replicate indices.",
            "Any censored replicate suppresses unconditional joint covariance/SE; surviving-replicate statistics are conditional.",
            "Individual marginal MC SE remains available only when that feature's point and every replicate are observed; it is not a substitute for the joint covariance.",
            "Each image uses its own saved-node global maximum, linear 50/10/1 percent crossings, on one declining branch; no extrapolation.",
            "The effective saved q endpoint can be below nominal qmax; absent thresholds are censored, never replaced by qmax.",
            "Thresholds apply directly to the two-dimensional radial density A0, without a q Jacobian; their definitions do not use a spherical template.",
            "Mean pT includes the q dq radial area measure, uses the entire saved q domain with exact piecewise-linear intensity moments, and is not an all-pT mean.",
            "K_tail is not a pure interference observable: structural response can persist in self.",
            "Self excludes inter-source cross terms; it is not incoherent nuclear breakup. Full/self normalized-feature contrasts are nonadditive.",
            "Every matched-beta none variant is compared independently; unmatched proton density parameters can also change the response.",
            "Prefix rows are mean-feature diagnostics, not independent measurements or cross-prefix covariance estimates.",
            "b refinement contrasts retain pairing but do not establish q/y/detector/model convergence.",
            "Saved spherical singleton calculations have zero orientation MC error, not zero systematic uncertainty.",
            "Plots show point estimates on registered grid cells; no parameter-selectivity or detector-feasibility claim.",
        ],
    }
    atomic_json(output/"summary.json", report)
    text = ["# Single-image shape scan", "", f"Analyzed {len(cases)}/{report['registered_cases']} registered cases at requested N={requested}.",
            f"Feature order: {', '.join(FEATURE_NAMES)}. All {len(steps)} b levels and full/self branches are retained.", "",
            "Q10 is the own-peak 10% outer crossing. B_mid = ln(5)/(Q10²−Q50²), B_tail = ln(10)/(Q1²−Q10²), K_tail = B_tail/B_mid.",
            f"Mean pT = ∫q²A0 dq / ∫qA0 dq on [{q[0]:.12g}, {q[-1]:.12g}] GeV. The radial q dq measure is included; moments are integrated exactly for piecewise-linear A0. No missing endpoint or tail is extrapolated.",
            "", "## Interpretation and uncertainty", ""]
    text.extend(f"- {item}" for item in report["limitations"])
    text.extend(["", "## Outputs", "", "- metrics.csv and cases/: point/censor details, resolved densities, joint covariance and MC SE.",
        "- bootstrap_replicates.npz: (b level, branch, replicate, feature), NaN for censored features; valid masks supplied. Case JSON gives keys and pairing plan.",
        "- radial_means.npz: native q, b steps and each mean_A0 (b level, full/self, q), allowing threshold inspection without raw orientation samples.",
        "- paired_response.csv: paired differences/ratios against same-density sphere, every matched-beta none, full/self and b refinements. Conditional columns are not unconditional errors.",
        "- prefix_stability.csv: configured prefixes up to N plus N; spherical rows use the same deterministic singleton.",
        "- missing_cases.json: omitted incomplete cases (only with --allow-partial).", "",
        "## Figure contract", "", FIGURE_CONTRACT["conclusion"], FIGURE_CONTRACT["scope"],
        FIGURE_CONTRACT["comparability"], FIGURE_CONTRACT["missing"],
        "Source data: metrics.csv. Point estimates are shown; joint MC uncertainty is tabulated. No smoothing across parameter cells.",
        "" if plot_files else "Plots were not requested (--no-plots).", "", "## Skipped comparisons", ""])
    text.extend(comparison_notes or ["None."])
    (output/"report.md").write_text("\n".join(text)+"\n")
    manifest = {"schema": REPORT_SCHEMA, "status": "completed", "production_fingerprint": production["fingerprint"],
                "input_sha256": inputs, "analysis_code_sha256": report["analysis_code_sha256"],
                "output_sha256": {str(path.relative_to(output)): sha256(path)
                    for path in sorted(output.rglob("*")) if path.is_file()}}
    atomic_json(output/"report_manifest.json", manifest)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n", type=int, help="Exact common prefix; spherical cases retain their deterministic singleton")
    parser.add_argument("--bootstrap-draws", type=int, help="Override analysis.bootstrap_draws (default 1024)")
    parser.add_argument("--allow-partial", action="store_true", help="Include complete cases only and report missing cases")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    report = analyze_run(**vars(args))
    print(f"{report['status']}: {len(report['analyzed_cases'])} cases; {args.output}")


if __name__ == "__main__":
    main()
