#!/usr/bin/env python3
"""Verify saved finite-q angular observables without launching forward runs.

Invariant closure is distinct from quadrature convergence, orientation sampling
precision, physical event-count precision and experimental qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

METRICS = ["C2", "S2", "Aphi", "mean_q"]
BINS = np.array([[.03, .06], [.06, .10], [.10, .14], [.14, .17]])
STAGES = ["decay_fixed", "decay_b", "decay_by"]


def verify(destination: str | Path) -> dict[str, Any]:
    """Validate analytic.json and response.json and write qa.json beside them."""
    destination = Path(destination)
    analytic = json.loads((destination / "analytic.json").read_text())
    response = json.loads((destination / "response.json").read_text())
    config = yaml.safe_load((destination / "config.yaml").read_text())
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    invariants: list[dict] = []
    warnings: list[dict] = []
    comparisons: list[dict] = []
    prefixes: list[dict] = []

    def check(name, passed, detail=None):
        row = {"name": name, "passed": bool(passed)}
        if detail is not None:
            row["detail"] = detail
        invariants.append(row)

    def warn(code, location, message, **detail):
        warnings.append({"code": code, "location": location, "message": message, **detail})

    def finite_tree(value, path):
        bad = []
        def walk(item, key):
            if isinstance(item, dict):
                for k, v in item.items():
                    walk(v, f"{key}.{k}")
            elif isinstance(item, list):
                for k, v in enumerate(item):
                    walk(v, f"{key}[{k}]")
            elif isinstance(item, (float, int)) and not isinstance(item, bool):
                if not np.isfinite(item):
                    bad.append(key)
        walk(value, path)
        check(path + ": all supplied numbers finite", not bad, bad)

    def array(value, shape, location):
        try:
            a = np.asarray(value, dtype=float)
            ok = a.shape == shape and np.all(np.isfinite(a))
        except (ValueError, TypeError):
            a, ok = None, False
        check(location + ": finite shape", ok, {"expected_shape": list(shape)})
        return a if ok else None

    def close(location, actual, expected, rtol=2e-8, atol=2e-13):
        actual, expected = np.asarray(actual), np.asarray(expected)
        difference = float(np.max(np.abs(actual - expected))) if actual.size else 0.
        check(location, np.allclose(actual, expected, rtol=rtol, atol=atol),
              {"max_absolute_error": difference, "rtol": rtol, "atol": atol})

    def covariance(value, dimension, location):
        a = array(value, (dimension, dimension), location)
        if a is None:
            return None
        scale = max(float(np.max(np.abs(a))), 1e-30)
        tolerance = max(1e-15, 2e-8 * scale)
        check(location + ": symmetric", np.max(abs(a-a.T)) <= tolerance)
        eigenvalues = np.linalg.eigvalsh((a+a.T)/2)
        check(location + ": positive semidefinite", eigenvalues.min() >= -tolerance,
              {"minimum_eigenvalue": float(eigenvalues.min()), "tolerance": tolerance})
        return a

    def harmonic_redundancy(values, covariance_array, location):
        close(location + ": Aphi = -4 C2/pi", values[:, 2], -4/np.pi*values[:, 0])
        if covariance_array is not None:
            operator = np.zeros((4, 16))
            for ibin in range(4):
                operator[ibin, 4*ibin] = 4/np.pi
                operator[ibin, 4*ibin+2] = 1
            close(location + ": Aphi covariance redundancy", operator @ covariance_array,
                  np.zeros((4, 16)), atol=max(1e-15, np.max(abs(covariance_array))*2e-8))

    def analytic_values(value, location):
        v = array(value, (4, 4), location)
        if v is not None:
            check(location + ": Born angular polarization bound",
                  np.all(np.hypot(v[:, 0], v[:, 1]) <= .25 + 1e-10))
            check(location + ": Aphi range", np.all(abs(v[:, 2]) <= 1 + 1e-12))
            check(location + ": mean q inside bin",
                  np.all((v[:, 3] >= BINS[:, 0]-1e-12) & (v[:, 3] <= BINS[:, 1]+1e-12)))
        return v

    def validate_stats(stats, location):
        v = analytic_values(stats.get("values"), location + ".values")
        c = covariance(stats.get("covariance"), 16, location + ".covariance")
        se = array(stats.get("se"), (4, 4), location + ".se")
        if se is not None:
            check(location + ": nonnegative SE", np.all(se >= 0))
            if c is not None:
                close(location + ": SE from covariance", se.ravel()**2, np.diag(c))
        if v is not None:
            harmonic_redundancy(v, c, location)
        yields = array(stats.get("yield"), (4,), location + ".yield")
        fraction = array(stats.get("yield_fraction"), (4,), location + ".yield_fraction")
        disk = stats.get("disk_yield", 0.)
        check(location + ": positive disk yield", disk is not None and disk > 0)
        if yields is not None:
            check(location + ": positive bin yields", np.all(yields > 0))
        if fraction is not None:
            check(location + ": physical yield fractions",
                  np.all(fraction > 0) and fraction.sum() <= 1 + 1e-12)
            if yields is not None and disk and disk > 0:
                close(location + ": fraction yield closure", fraction, yields / disk)
        fcov = covariance(stats.get("yield_fraction_covariance"), 4,
                          location + ".yield_fraction_covariance")
        fse = array(stats.get("yield_fraction_se"), (4,), location + ".yield_fraction_se")
        if fcov is not None and fse is not None:
            close(location + ": fraction SE closure", fse**2, np.diag(fcov))
        return v, c

    finite_tree(analytic, "analytic")
    finite_tree(response, "response")
    finite_tree(config, "config")
    check("analytic metric order", analytic.get("metric_names") == METRICS)
    check("response metric order", response.get("metric_names") == METRICS)
    check("analytic registered bins", np.array_equal(np.asarray(analytic.get("bin_edges_GeV")), BINS))
    check("response registered bins", np.array_equal(np.asarray(response.get("bin_edges_gev")), BINS))
    check("analytic registered stages", analytic.get("stage_names") == STAGES)
    cfg = config.get("study", config)
    check("current b window", cfg.get("fixed_b_fm") == 12 and cfg.get("b_min_fm") == 12
          and cfg.get("b_max_fm") == 60)
    check("selected qmax", analytic.get("qmax_GeV") == .17 and response.get("qmax_gev") == .17)
    check("truth-anchor setting", bool(response.get("truth_anchored_response_enabled", False))
          == bool(cfg.get("truth_anchored_response", False)))
    acases = analytic.get("cases", {})
    rcases_list = response.get("cases", [])
    rcases = {c["name"]: c for c in rcases_list}
    check("unique response cases", len(rcases) == len(rcases_list))
    check("matching case sets", set(acases) == set(rcases))
    check("spherical reference present", "spherical" in acases and "spherical" in rcases)
    if "spherical" not in acases or "spherical" not in rcases:
        raise ValueError("Saved products have no common spherical reference")
    check("deterministic spherical reference", acases["spherical"].get("n_orientations") == 1)
    for record in manifest.get("inputs", []):
        error = record.get("orientation_mean_relative_error")
        check(record.get("case", "unknown") + ": saved raw-coefficient mean closure",
              error is not None and np.isfinite(error) and 0 <= error <= 1e-12,
              {"relative_error": error})
    for name, case in acases.items():
        check(name + ": positive orientation count", case.get("n_orientations", 0) >= 1)
        for stage in STAGES:
            location = f"analytic.{name}.{stage}"
            branches = case["stages"][stage]
            stats = {branch: validate_stats(branches[branch], location + "." + branch)
                     for branch in ("full", "self")}
            joint = covariance(branches["joint_covariance"], 32, location + ".joint_covariance")
            contrast = branches["contrast"]
            ci = covariance(contrast["interaction_covariance"], 16, location + ".interaction_covariance")
            if joint is not None:
                for index, branch in enumerate(("full", "self")):
                    if stats[branch][1] is not None:
                        close(location + ": joint " + branch + " block",
                              joint[index*16:(index+1)*16, index*16:(index+1)*16], stats[branch][1])
                if ci is not None:
                    close(location + ": paired full-self covariance",
                          ci, joint[:16, :16] + joint[16:, 16:] - joint[:16, 16:] - joint[16:, :16])
            df = np.asarray(contrast["delta_full"])
            ds = np.asarray(contrast["delta_self"])
            close(location + ": interaction mean", contrast["interaction"], df-ds)
            for branch, delta in (("full", df), ("self", ds)):
                reference = acases["spherical"]["stages"][stage][branch]
                close(location + ": structural delta " + branch, delta,
                      np.asarray(branches[branch]["values"]) - np.asarray(reference["values"]))
                curves = branches[branch]["curves"]
                c2, s2 = np.asarray(curves["C2"]), np.asarray(curves["S2"])
                check(location + "." + branch + ": curve bound",
                      np.isfinite(c2).all() and np.isfinite(s2).all()
                      and np.all(np.hypot(c2, s2) <= .25 + 1e-10))
            ise = array(contrast["interaction_se"], (4, 4), location + ".interaction_se")
            if ci is not None and ise is not None:
                close(location + ": interaction SE covariance", ise.ravel()**2, np.diag(ci))
                harmonic_redundancy(np.asarray(contrast["interaction"]), ci, location + ".interaction")
        bg = case["background"]
        full = np.asarray(case["stages"]["decay_by"]["full"]["values"])
        total = analytic_values(bg["total_values"], f"analytic.{name}.background.total_values")
        signal, background = np.asarray(bg["signal_yield"]), np.asarray(bg["background_yield"])
        check(name + ": background yields nonnegative", np.all(signal > 0) and np.all(background >= 0))
        dilution = signal / (signal+background)
        close(name + ": background fraction", bg["fraction_of_total"], 1-dilution)
        close(name + ": background to signal", bg["background_to_signal"], background/signal)
        close(name + ": background dilution", bg["angular_dilution"], dilution)
        close(name + ": background signal values", bg["full_values"], full)
        close(name + ": background signal yield", signal,
              case["stages"]["decay_by"]["full"]["yield"])
        if total is not None:
            close(name + ": isotropic background angular dilution", total[:, :3], full[:, :3]*dilution[:, None])
        prefix_entries = case.get("orientation_prefix", {})
        for count, entries in prefix_entries.items():
            check(name + ": prefix count " + count, 1 <= int(count) <= case["n_orientations"])
            for branch in ("full", "self"):
                validate_stats(entries[branch], f"analytic.{name}.prefix{count}.{branch}")
        if "32" in prefix_entries and case["n_orientations"] > 32:
            final = np.asarray(case["stages"]["decay_by"]["full"]["values"])
            prefix = np.asarray(prefix_entries["32"]["full"]["values"])
            prefixes.append({"case": name, "n_final": case["n_orientations"],
                             "delta_final_minus_first32": (final-prefix).tolist(),
                             "interpretation": "Nested overlapping samples; not independent convergence evidence or an additional error bar"})
        coarse = case["coarse_b"]
        for branch in ("full", "self"):
            values = np.asarray(case["stages"]["decay_by"][branch]["values"])
            close(name + ": coarse-b shift " + branch, coarse[branch]["shift"], np.asarray(coarse[branch]["values"])-values)
        bshift = np.asarray(coarse["full"]["structural_delta_shift"])[:, 0]
        ori_se = np.asarray(case["stages"]["decay_by"]["full"]["se"])[:, 0]
        relevant = np.flatnonzero(abs(bshift) > np.maximum(ori_se, 1e-5))
        if name != "spherical" and len(relevant):
            warn("coarse_b_shift", name, "C2 structural-response shift under coarser b quadrature exceeds the larger of orientation SE and 1e-5; not a convergence certificate", bins=relevant.tolist(), shift=bshift.tolist())

    scenarios = response.get("responses", [])
    expected_ids = [s["id"] for s in scenarios]
    check("unique response IDs", len(expected_ids) == len(set(expected_ids)))
    reference_rows = {row["response_id"]: row for row in rcases["spherical"]["rows"]}
    for name, case in rcases.items():
        rows = case["rows"]
        check(name + ": response scenario set", [r["response_id"] for r in rows] == expected_ids)
        zeros = [r for r in rows if r["fractional_sigma"] == 0 and r["angular_sigma"] == 0]
        check(name + ": one zero-response reference", len(zeros) == 1)
        check(name + ": consistent orientation count", name in acases and case["n_orientations"] == acases[name]["n_orientations"])
        if not zeros:
            continue
        zero = zeros[0]
        for row in rows:
            identifier = row["response_id"]
            location = f"response.{name}.{identifier}"
            values = array(row["values_flat"], (16,), location + ".values")
            mc = covariance(row["mc_covariance"], 16, location + ".mc_covariance")
            ori = covariance(row["orientation_covariance"], 16, location + ".orientation_covariance")
            combined = covariance(row["combined_sampling_covariance"], 16, location + ".combined_sampling_covariance")
            event = covariance(row["event_covariance_per_total_selected_event"], 16, location + ".event_covariance_per_total_selected_event")
            if mc is not None and ori is not None and combined is not None:
                close(location + ": combined sampling covariance", combined, mc+ori)
            closure = np.asarray(row["orientation_reweight_closure"])
            check(location + ": raw-orientation reweight closure", np.isfinite(closure).all() and np.max(abs(closure)) < 1e-10,
                  {"max_absolute_error": float(np.max(abs(closure)))})
            check(location + ": finite truth migration support", row["truth_qmax_gev"] > .17)
            check(location + ": generated count", row["n_generated"] == response["n_events"])
            check(location + ": four bins", len(row["bins"]) == 4)
            fractions = []
            for ibin, b in enumerate(row["bins"]):
                sub = slice(ibin*4, (ibin+1)*4)
                check(location + f": bin {ibin} limits", np.array_equal(b["q_range_gev"], BINS[ibin]))
                v = np.asarray(b["values"])
                close(location + f": bin {ibin} flatten order", v, values[sub])
                check(location + f": bin {ibin} response moment bounds", np.hypot(v[0], v[1]) <= 1+1e-12 and abs(v[2]) <= 1+1e-12)
                check(location + f": bin {ibin} mean q", BINS[ibin, 0]-1e-12 <= v[3] <= BINS[ibin, 1]+1e-12)
                fraction = b["bin_fraction"]
                fractions.append(fraction)
                check(location + f": bin {ibin} fraction", 0 < fraction <= 1)
                ec = covariance(b["event_covariance_per_bin_event"], 4, location + f".bin{ibin}.event_covariance")
                if event is not None and ec is not None:
                    close(location + f": bin {ibin} count convention scaling", event[sub, sub], ec/fraction)
                    close(location + f": bin {ibin} quadrant event variance", ec[2, 2], 1-v[2]**2, atol=2e-12)
                    close(location + f": bin {ibin} circular event identity", ec[0, 0]+ec[1, 1]+v[0]**2+v[1]**2, 1., atol=2e-12)
                for key, covariance_array in (("mc_se", mc), ("orientation_se", ori), ("combined_sampling_se", combined)):
                    if covariance_array is not None:
                        close(location + f": bin {ibin} {key}", np.asarray(b[key])**2, np.diag(covariance_array)[sub])
                close(location + f": bin {ibin} harmonic diagnostic definition", b["Aphi_harmonic_residual"], v[2]+4/np.pi*v[0])
                if row is zero:
                    residual_error = b["Aphi_harmonic_residual_mc_se"]
                    if abs(b["Aphi_harmonic_residual"]) > 5*residual_error + 1e-10:
                        warn("zero_response_harmonic_sampling", location, "Zero-response quadrant/Fourier identity exceeds 5 MC SE; stochastic warning", bin=ibin)
            check(location + ": measured bins subset of selected sample", sum(fractions) <= 1+1e-12)
            if event is not None:
                off_diagonal_bins = event.copy()
                for ibin in range(4):
                    off_diagonal_bins[4*ibin:4*ibin+4, 4*ibin:4*ibin+4] = 0.
                close(location + ": disjoint-bin counting covariance", off_diagonal_bins, np.zeros((16, 16)))
            for contrast_name, reference in (("vs_spherical", reference_rows[identifier]), ("vs_zero_response", zero)):
                c = row[contrast_name]
                prefix = location + "." + contrast_name
                close(prefix + ": paired mean difference", c["delta"], values-np.asarray(reference["values_flat"]))
                c_mc = covariance(c["mc_covariance"], 16, prefix + ".mc_covariance")
                c_ori = covariance(c["orientation_covariance"], 16, prefix + ".orientation_covariance")
                c_both = covariance(c["combined_sampling_covariance"], 16, prefix + ".combined_sampling_covariance")
                if c_mc is not None and c_ori is not None and c_both is not None:
                    close(prefix + ": covariance sum", c_both, c_mc+c_ori)
                    for key, cov in (("mc_se", c_mc), ("orientation_se", c_ori), ("combined_sampling_se", c_both)):
                        close(prefix + ": " + key, np.asarray(c[key])**2, np.diag(cov))
                delta, error = np.asarray(c["delta"]), np.asarray(c["combined_sampling_se"])
                expected_resolved = abs(delta) > np.maximum(3*error, c.get("resolution_absolute_floor", 0.))
                check(prefix + ": resolution gate", np.array_equal(c["resolved_above_3_sampling_se"], expected_resolved))
                n3 = c.get("n3_total_selected_one_sample_diagnostic", [None]*16)
                check(prefix + ": unresolved event-count estimates suppressed",
                      all(v is None for v, ok in zip(n3, expected_resolved) if not ok))
                if not c.get("event_count_diagnostics_enabled", False):
                    check(prefix + ": optional event counts disabled", all(v is None for v in n3))
            if cfg.get("truth_anchored_response", False):
                check(location + ": truth anchor present", "truth_anchored" in row)
            if "truth_anchored" in row:
                anchored = row["truth_anchored"]
                prefix = location + ".truth_anchored"
                av = array(anchored["values_flat"], (16,), prefix + ".values")
                exact = np.asarray(acases[name]["stages"]["decay_by"]["full"]["values"]).ravel()
                exact_cov = np.asarray(acases[name]["stages"]["decay_by"]["full"]["covariance"])
                close(prefix + ": exact truth values", anchored["exact_truth_values_flat"], exact)
                close(prefix + ": exact truth orientation covariance", anchored["exact_truth_orientation_covariance"], exact_cov)
                close(prefix + ": control variate definition", av,
                      exact + (values-np.asarray(zero["values_flat"])))
                amc = covariance(anchored["mc_covariance"], 16, prefix + ".mc_covariance")
                aori = covariance(anchored["orientation_covariance"], 16, prefix + ".orientation_covariance")
                atotal = covariance(anchored["combined_sampling_covariance"], 16, prefix + ".combined_sampling_covariance")
                if amc is not None:
                    close(prefix + ": paired response-shift MC covariance", amc, row["vs_zero_response"]["mc_covariance"])
                if amc is not None and aori is not None and atotal is not None:
                    close(prefix + ": combined covariance", atotal, amc+aori)
                    for key, cov in (("mc_se", amc), ("orientation_se", aori), ("combined_sampling_se", atotal)):
                        close(prefix + ": " + key, np.asarray(anchored[key])**2, np.diag(cov))
                zero_anchor = zero["truth_anchored"]
                reference_anchor = reference_rows[identifier]["truth_anchored"]
                for contrast_name, ref in (("vs_spherical", reference_anchor), ("vs_zero_response", zero_anchor)):
                    contrast = anchored[contrast_name]
                    cp = prefix + "." + contrast_name
                    close(cp + ": mean difference", contrast["delta"], av-np.asarray(ref["values_flat"]))
                    cmc = covariance(contrast["mc_covariance"], 16, cp + ".mc_covariance")
                    cori = covariance(contrast["orientation_covariance"], 16, cp + ".orientation_covariance")
                    ctotal = covariance(contrast["combined_sampling_covariance"], 16, cp + ".combined_sampling_covariance")
                    if cmc is not None and cori is not None and ctotal is not None:
                        close(cp + ": covariance sum", ctotal, cmc+cori)
                        for key, cov in (("mc_se", cmc), ("orientation_se", cori), ("combined_sampling_se", ctotal)):
                            close(cp + ": " + key, np.asarray(contrast[key])**2, np.diag(cov))
                    error = np.asarray(contrast["combined_sampling_se"])
                    expected = abs(np.asarray(contrast["delta"])) > np.maximum(3*error, contrast.get("resolution_absolute_floor", 0.))
                    check(cp + ": resolution gate", np.array_equal(expected, contrast["resolved_above_3_sampling_se"]))
                    if contrast_name == "vs_zero_response":
                        close(cp + ": raw response shift preserved", contrast["delta"], row["vs_zero_response"]["delta"])
                        close(cp + ": raw response MC covariance preserved", contrast["mc_covariance"], row["vs_zero_response"]["mc_covariance"])
                        close(cp + ": raw response orientation covariance preserved", contrast["orientation_covariance"], row["vs_zero_response"]["orientation_covariance"])
                    elif cori is not None and aori is not None:
                        close(cp + ": deterministic-reference orientation covariance", cori, aori)
                        # Cauchy-Schwarz bound for the four-term paired MC
                        # contrast. Independence cannot be assumed here.
                        bound = 2*(amc+np.asarray(reference_anchor["mc_covariance"]))-cmc
                        covariance(bound, 16, cp + ".paired_covariance_Cauchy_bound")
                for ibin, b in enumerate(anchored["bins"]):
                    sub = slice(4*ibin, 4*ibin+4)
                    close(prefix + f": bin {ibin} flatten order", b["values"], av[sub])
                    for key, cov in (("mc_se", amc), ("orientation_se", aori), ("combined_sampling_se", atotal)):
                        close(prefix + f": bin {ibin} {key}", np.asarray(b[key])**2, np.diag(cov)[sub])
                    close(prefix + f": bin {ibin} independent quadrant diagnostic",
                          b["Aphi_harmonic_residual"], av[4*ibin+2]+4/np.pi*av[4*ibin])
                if row is zero:
                    close(prefix + ": zero response exact truth", av, exact, rtol=0, atol=1e-15)
                    check(prefix + ": zero response MC exactly zero", np.array_equal(amc, np.zeros((16, 16))))
                    check(prefix + ": zero structural MC exactly zero",
                          np.array_equal(anchored["vs_spherical"]["mc_covariance"], np.zeros((16, 16))))
                    close(prefix + ": zero orientation covariance exact", aori, exact_cov)
                    harmonic_redundancy(av.reshape(4, 4), aori, prefix + ".zero")
            structural = row.get("truth_anchored", row)["vs_spherical"]
            if name != "spherical":
                unresolved = np.flatnonzero(~np.asarray(structural["resolved_above_3_sampling_se"])[::4])
                if len(unresolved):
                    warn("unresolved_structural_C2", location, "C2 structural difference is not resolved above 3 combined orientation+MC sampling SE in these bins", bins=unresolved.tolist(), estimator="truth_anchored" if "truth_anchored" in row else "raw")
            if row is zero and name in acases and values is not None and mc is not None:
                ideal = np.asarray(acases[name]["stages"]["decay_by"]["full"]["values"]).ravel()
                delta = values-ideal
                error = np.sqrt(np.maximum(0., np.diag(mc)))
                bad = np.flatnonzero(abs(delta) > 5*error+1e-7)
                comparison = {"case": name, "delta_response_zero_minus_analytic": delta.tolist(),
                              "mc_se": error.tolist(), "within_5_mc_se_plus_tolerance": not len(bad),
                              "absolute_integration_tolerance": 1e-7}
                comparisons.append(comparison)
                if len(bad):
                    warn("zero_response_vs_analytic_sampling", name, "Zero-response quadrature differs from exact radial moments by more than 5 MC SE plus 1e-7; stochastic warning, not an invariant failure", flat_indices=bad.tolist())

    hashes = {name: hashlib.sha256((destination/name).read_bytes()).hexdigest()
              for name in ("analytic.json", "response.json", "config.yaml")}
    if manifest_path.exists():
        hashes["manifest.json"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    report = {"pass": all(x["passed"] for x in invariants),
              "invariant_count": len(invariants),
              "passed_invariant_count": sum(x["passed"] for x in invariants),
              "warning_count": len(warnings), "invariants": invariants, "warnings": warnings,
              "zero_response_comparisons": comparisons, "orientation_prefix_diagnostics": prefixes,
              "input_sha256": hashes,
              "scope": {
                  "invariants": "Saved numerical and physical identities; no forward simulation performed",
                  "covariance_order": "bin-major C2,S2,Aphi,mean_q; full/self joint blocks in analytic data",
                  "orientation_precision": "Finite sampling errors retained. Nested prefixes are descriptive and do not establish convergence",
                  "b12_spatial_and_rapidity_convergence": "unassessed: historical b20 checks are not applied as certification of b12 or the new angular bins",
                  "coarse_b": "Diagnostic shift only; finer current-window b convergence not certified",
                  "truth_tail_convergence": "unassessed beyond the saved finite support; reported migrations are within that support",
                  "response": "Uncalibrated toy track errors; no acceptance, backgrounds, or experimental precision claim",
                  "Aphi_after_response": "Computed independently from complementary quadrants. Aphi=-4 C2/pi is deliberately not enforced after smearing",
                  "truth_anchored": "Exact zero-response anchor for numerical quadrature; variance reduction for nonzero-response structural contrasts is not guaranteed. Raw outputs and physical event covariance remain; finite-ratio MC bias, mixed higher-order errors, and detector systematics remain unassessed",
                  "source_provenance": "Manifest retained and hashed; this verifier does not independently rerun physical source caches",
              }}
    (destination / "qa.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path)
    parser.add_argument("--output", type=Path, help="Alias for destination directory")
    args = parser.parse_args()
    destination = args.output or args.destination
    if destination is None:
        parser.error("provide a saved output directory")
    report = verify(destination)
    print(json.dumps({key: report[key] for key in ("pass", "invariant_count", "passed_invariant_count", "warning_count")}, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
