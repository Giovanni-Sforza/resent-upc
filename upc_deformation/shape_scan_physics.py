"""Explicit Woods--Saxon variants for the single-image shape scan.

The scan declares proton/neutron density parameters, not manuscript defaults.
RMS radii retain the legacy origin-based convention: there is no recentering or
volume-preserving radius correction.  This module adapts cases to the unchanged
``case_physics``/precision kernel, including its neutron-radius root bracket.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math
from numbers import Real
import re

from scipy.optimize import brentq

from .observables import case_physics, deformed_ws_rms


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _variant(value):
    if not isinstance(value, dict):
        raise ValueError("Each density variant must be a mapping")
    result = deepcopy(value)
    ident = result.get("id")
    if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", ident):
        raise ValueError("Density variant id must contain only letters, digits, '_' or '-'")
    kind, mode = result.get("type"), result.get("mode")
    if kind not in ("none", "skin", "halo"):
        raise ValueError("Density type must be none, skin or halo")
    if mode not in ("direct_ws", "rms_matched"):
        raise ValueError("Density mode must be direct_ws or rms_matched")
    fields = {"R_p", "a_p", "R_n", "a_n"} if mode == "direct_ws" else {
        "R_p", "a_p", "delta_r_np_fm"}
    missing = fields - result.keys()
    extra = result.keys() - fields - {"id", "type", "mode", "label", "source", "notes"}
    if missing or extra:
        raise ValueError(f"Density variant {ident}: missing {sorted(missing)}, unexpected {sorted(extra)}")
    for key in fields:
        result[key] = _number(result[key], f"{ident}.{key}", positive=key != "delta_r_np_fm")
    if mode == "rms_matched":
        delta = result["delta_r_np_fm"]
        if (kind == "none" and delta != 0) or (kind != "none" and delta <= 0):
            raise ValueError("rms_matched none requires delta_r_np_fm=0; skin/halo require a positive value")
    else:
        rp, ap, rn, an = (result[k] for k in ("R_p", "a_p", "R_n", "a_n"))
        same_r = math.isclose(rp, rn, rel_tol=0, abs_tol=1e-12)
        same_a = math.isclose(ap, an, rel_tol=0, abs_tol=1e-12)
        valid = ((kind == "none" and same_r and same_a)
                 or (kind == "skin" and rn > rp and same_a)
                 or (kind == "halo" and same_r and an > ap))
        if not valid:
            raise ValueError("direct_ws type is inconsistent: none needs equal R/a, skin larger R_n only, halo larger a_n only")
    return result


def _axis(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"scan.{name} must be a nonempty list")
    result = [_number(value, name) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"scan.{name} contains duplicate values")
    return result


def build_cases(cfg):
    """Return the Cartesian product, in variant/beta2/beta3 input order.

    Names use the explicit variant id and zero-based grid indices.  Numeric beta
    values and the full variant declaration remain in each case/provenance;
    changing the scan grid therefore changes its scientific identity.
    """
    scan = cfg.get("scan")
    if not isinstance(scan, dict):
        raise ValueError("Missing scan mapping")
    b2 = _axis(scan.get("beta2_values"), "beta2_values")
    b3 = _axis(scan.get("beta3_values"), "beta3_values")
    variants = scan.get("density_variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("scan.density_variants must be a nonempty list")
    variants = [_variant(value) for value in variants]
    if len({value["id"] for value in variants}) != len(variants):
        raise ValueError("Density variant ids must be unique")
    return [{"name": f"{v['id']}__b2_{i:03d}__b3_{j:03d}",
             "beta2": x, "beta3": y, "beta2_index": i, "beta3_index": j,
             "density_id": v["id"], "density_type": v["type"], "density_mode": v["mode"],
             "density_variant": deepcopy(v)}
            for v in variants for i, x in enumerate(b2) for j, y in enumerate(b3)]


@lru_cache(maxsize=4096)
def _rms(radius, diffuseness, beta2, beta3):
    return deformed_ws_rms(radius, diffuseness, beta2, beta3)


def resolve_case(cfg, case):
    """Return ``(kernel_cfg, kernel_case, density_metadata)`` without mutation.

    ``direct_ws`` retains all four declared WS parameters and derives its RMS
    difference at this deformation. ``rms_matched`` instead holds the requested
    origin-based RMS difference fixed at every deformation: skin solves R_n at
    a_n=a_p, halo solves a_n at R_n=R_p, and none uses identical p/n profiles.
    The returned case has legacy ``skin_fm`` even for halo; its meaning is only
    the derived RMS difference, while the resolved diffuseness lives in config.
    """
    variant = _variant(case.get("density_variant"))
    beta2, beta3 = (_number(case.get(key), key) for key in ("beta2", "beta3"))
    for key, field in (("density_id", "id"), ("density_type", "type"), ("density_mode", "mode")):
        if key in case and case[key] != variant[field]:
            raise ValueError(f"Case {key} differs from its density_variant")
    kernel_cfg, kernel_case = deepcopy(cfg), deepcopy(case)
    physics = kernel_cfg.get("physics")
    if not isinstance(physics, dict) or not isinstance(physics.get("nuclear_parameters"), dict):
        raise ValueError("Missing physics.nuclear_parameters mapping")
    if any(physics.get(f"nuclear_{i}") for i in (1, 2)):
        raise ValueError("Shape scans require shared nuclear parameters, without per-nucleus overrides")
    nuc = physics["nuclear_parameters"]
    if nuc.get("density_model", "woods_saxon") != "woods_saxon":
        raise ValueError("Shape scans require woods_saxon density_model")
    kind, mode = variant["type"], variant["mode"]
    rp, ap = variant["R_p"], variant["a_p"]
    proton_rms = _rms(rp, ap, beta2, beta3)
    if mode == "direct_ws":
        rn, an = variant["R_n"], variant["a_n"]
        delta = _rms(rn, an, beta2, beta3) - proton_rms
    else:
        delta, rn, an = variant["delta_r_np_fm"], rp, ap
        if kind == "halo":
            target = proton_rms + delta
            upper = max(2 * ap, ap + delta)
            for _ in range(20):
                if _rms(rp, upper, beta2, beta3) >= target:
                    break
                upper *= 2
            else:
                raise ValueError("Unable to bracket halo diffuseness for requested RMS difference")
            an = brentq(lambda value: _rms(rp, value, beta2, beta3) - target,
                        ap, upper, xtol=1e-12)
    nuc.update(density_model="woods_saxon", R_ws_p=rp, R_ws_n=rn,
               d_ws_p=ap, d_ws_n=an, beta2=beta2, beta3=beta3)
    kernel_case.update(beta2=beta2, beta3=beta3, skin_fm=float(delta))
    try:
        resolved_physics, _ = case_physics(kernel_cfg, kernel_case)
    except ValueError as exc:
        raise ValueError("Density is outside the legacy neutron-radius solver bracket [0.6 R_p, 1.8 R_p]") from exc
    resolved_rn = float(resolved_physics["nuclear_parameters"]["R_ws_n"])
    if (mode == "direct_ws" or kind in ("none", "halo")) and not math.isclose(
            resolved_rn, rn, rel_tol=0, abs_tol=2e-9):
        raise ValueError("Legacy RMS adapter did not reconstruct the declared neutron radius")
    neutron_rms = _rms(resolved_rn, an, beta2, beta3)
    if not math.isclose(neutron_rms - proton_rms, delta, rel_tol=0, abs_tol=2e-9):
        raise ValueError("Resolved density does not match its requested RMS difference")
    kernel_cfg["physics"] = resolved_physics
    metadata = {
        "id": variant["id"], "type": kind, "mode": mode,
        "beta2": beta2, "beta3": beta3,
        "R_ws_p_fm": rp, "R_ws_n_fm": resolved_rn, "a_p_fm": ap, "a_n_fm": float(an),
        "proton_rms_fm": proton_rms, "neutron_rms_fm": neutron_rms,
        "delta_r_np_fm": neutron_rms - proton_rms,
        "rms_definition": "sqrt(<r^2>) about Woods-Saxon parameter origin",
        "shared_proton_neutron_beta": True, "shared_proton_neutron_orientation": True,
        "center_of_mass_correction": False, "volume_preserving_radius_correction": False,
        "density_normalization": "Separate proton/neutron grid integrals to Z/N at each orientation",
        "declared_variant": deepcopy(variant),
    }
    return kernel_cfg, kernel_case, metadata
