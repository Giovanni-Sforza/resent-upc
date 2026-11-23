"""Spherical, fixed-diffuseness neutron-radius scan for the final figure."""

import numpy as np

from .shape_scan_analysis import extract_tail_features
from .shape_scan_fast_v2_forward import ShapeScanFastV2Refiner
from .shape_scan_physics import resolve_case


def compute_spherical_point(base_cfg, delta_r_np_fm, device="cuda"):
    """Return resolved WS parameters, q in GeV, full coherent A0 and its features.

    delta_r_np_fm is the requested difference of RMS radii. The density solver
    varies only the neutron half-density radius at the registered proton R and a.
    The spherical density is invariant under rotations, so one identity pair
    integrates orientation exactly as in the original fast-v2 scan.
    """
    skin = next(v for v in base_cfg["scan"]["density_variants"] if v["id"] == "skin")
    kind = "none" if delta_r_np_fm == 0 else "skin"
    case = {
        "beta2": 0.0,
        "beta3": 0.0,
        "density_variant": {
            "id": kind, "type": kind, "mode": "rms_matched",
            "R_p": skin["R_p"], "a_p": skin["a_p"], "delta_r_np_fm": delta_r_np_fm,
        },
    }
    cfg, kernel_case, density = resolve_case(base_cfg, case)
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
