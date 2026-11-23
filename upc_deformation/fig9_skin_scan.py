"""Spherical Woods–Saxon profiles at fixed neutron-skin thickness for Fig. 9."""

import numpy as np

from .observables import case_physics, deformed_ws_rms
from .shape_scan_analysis import extract_tail_features
from .shape_scan_fast_v2_forward import ShapeScanFastV2Refiner
from .shape_scan_physics import resolve_case


def resolve_spherical_profile(base_cfg, delta_r_np_fm, profile):
    """Resolve a radius-only, diffuseness-only or midpoint-diffuseness profile."""
    endpoint_type = {"radius": "skin", "diffuseness": "halo", "mixed": "halo"}[profile]
    kind = "none" if delta_r_np_fm == 0 else endpoint_type
    skin = next(v for v in base_cfg["scan"]["density_variants"] if v["id"] == "skin")
    case = {
        "beta2": 0.0,
        "beta3": 0.0,
        "density_variant": {
            "id": kind, "type": kind, "mode": "rms_matched",
            "R_p": skin["R_p"], "a_p": skin["a_p"], "delta_r_np_fm": delta_r_np_fm,
        },
    }
    cfg, kernel_case, density = resolve_case(base_cfg, case)
    density["profile"] = profile
    if profile != "mixed" or delta_r_np_fm == 0:
        return cfg, kernel_case, density

    # The halo solver fixes R_n=R_p. Half of its change in diffuseness sets
    # a_n for the mixed profile. The production RMS solver then determines R_n.
    a_endpoint = density["a_n_fm"]
    a_mid = (density["a_p_fm"] + a_endpoint) / 2
    cfg["physics"]["nuclear_parameters"]["d_ws_n"] = a_mid
    kernel_case = {"beta2": 0.0, "beta3": 0.0, "skin_fm": delta_r_np_fm}
    cfg["physics"], _ = case_physics(cfg, kernel_case)
    rn = cfg["physics"]["nuclear_parameters"]["R_ws_n"]
    neutron_rms = deformed_ws_rms(rn, a_mid)
    delta_actual = neutron_rms - density["proton_rms_fm"]
    if not np.isclose(delta_actual, delta_r_np_fm, rtol=0, atol=2e-9):
        raise ValueError(f"Mixed profile attained {delta_actual} fm for target {delta_r_np_fm} fm")
    density.update(
        id="mixed", type="mixed", R_ws_n_fm=rn, a_n_fm=a_mid, neutron_rms_fm=neutron_rms,
        delta_r_np_fm=delta_actual,
        declared_variant={"id": "mixed", "mode": "rms_matched", "delta_r_np_fm": delta_r_np_fm,
                          "a_n_fm": a_mid, "diffuseness_only_a_n_fm": a_endpoint,
                          "diffuseness_fraction": 0.5},
    )
    return cfg, kernel_case, density


def compute_spherical_point(base_cfg, delta_r_np_fm, device="cuda", *, profile="radius"):
    """Return resolved WS parameters, q in GeV, full coherent A0 and its features.

    One identity rotation pair exactly represents the spherical orientation average.
    """
    cfg, kernel_case, density = resolve_spherical_profile(base_cfg, delta_r_np_fm, profile)
    runtime = cfg["runtime"]
    refiner = ShapeScanFastV2Refiner(
        cfg, kernel_case, device=device, dtype=runtime["dtype"], fft_mode=runtime["fft_mode"]
    )
    rotation = np.eye(3)[None]
    result = refiner.calculate(rotation, rotation)
    finest = np.argmin(result["b_steps_fm"])
    full = result["branch_names"].index("full")
    q = result["stored_q"]
    a0 = result["A0"][0, finest, full]
    return density, q, a0, extract_tail_features(q, a0)
