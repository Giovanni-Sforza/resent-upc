#!/usr/bin/env python3
"""Compute the four spherical spectra added to Fig. 9."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from upc_deformation.fig9_skin_scan import compute_spherical_point
from upc_deformation.precision_campaign import configure_runtime


TARGETS_FM = (0.04, 0.08, 0.12, 0.16)
WS_FIELDS = ("R_ws_p_fm", "R_ws_n_fm", "a_p_fm", "a_n_fm", "proton_rms_fm", "neutron_rms_fm")
FEATURES = ("Q50_GeV", "Q10_GeV", "Q1_GeV", "B_mid", "B_tail", "K_tail")


def write_json(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/shape_scan_a100.yaml").read_text())
    config["runtime"].update(device=args.device, batch_size=1)
    runtime = config["runtime"]
    write_json(output / "settings.json", {
        "base_config": "configs/shape_scan_a100.yaml",
        "physics": config["physics"], "study": config["study"],
        "runtime": {k: runtime[k] for k in ("device", "dtype", "threads", "fft_mode", "batch_size")},
        "method": "fast_v2, full coherent A0, finest b grid, one identity orientation pair",
        "density_variant": "skin", "diffuseness_constraint": "a_n = a_p",
        "varied_parameter": "R_ws_n", "beta2": 0.0, "beta3": 0.0,
        "target_delta_r_np_fm": TARGETS_FM, "reused_delta_r_np_fm": [0.0, 0.20],
    })
    environment = configure_runtime(runtime)
    write_json(output / "environment.json", environment)
    print(f"Device: {environment.get('device_name', args.device)}, {runtime['dtype']}", flush=True)

    spectra = output / "spectra"
    spectra.mkdir()
    fields = ("target_delta_r_np_fm", "delta_r_np_fm", *WS_FIELDS, *FEATURES, "status")
    with (output / "new_skin_points.csv").open("w", newline="") as stream:
        table = csv.DictWriter(stream, fields, lineterminator="\n")
        table.writeheader()
        for index, target in enumerate(TARGETS_FM, 1):
            print(f"[{index}/{len(TARGETS_FM)}] Computing delta_r_np = {target:.2f} fm", flush=True)
            density, q, a0, features = compute_spherical_point(config, target, device=args.device)
            name = f"skin_{target:.2f}fm"
            np.savez(spectra / f"{name}.npz", q_GeV=q, A0_full=a0)
            write_json(spectra / f"{name}.json", {"density": density, "features": features})
            table.writerow({
                "target_delta_r_np_fm": target, "delta_r_np_fm": density["delta_r_np_fm"],
                **{k: density[k] for k in WS_FIELDS}, **{k: features[k] for k in FEATURES},
                "status": features["status"],
            })
            stream.flush()
            if features["status"] != "observed":
                raise ValueError(f"{name}: {features['censor_reason']}")
            print(f"  R_n = {density['R_ws_n_fm']:.8f} fm, "
                  f"p_T,10 = {features['Q10_GeV']:.9f} GeV/c, K_tail = {features['K_tail']:.9f}",
                  flush=True)


if __name__ == "__main__":
    main()
