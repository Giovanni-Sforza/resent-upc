#!/usr/bin/env python3
"""Run or resume the independent full-precision panel-d campaign."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from upc_deformation.precision_campaign import load_config, run_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/panel_d_precision_a100.yaml"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-new-pairs", type=int, help="Stop safely after this many new orientation pairs across all cases")
    parser.add_argument("--device", help="Explicit cpu or cuda[:index]; CUDA never falls back to CPU")
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-orientations", type=int, help="Deformation target, divisible by four; supports prefix extension")
    parser.add_argument("--b-step", type=float, dest="b_step_fm")
    parser.add_argument("--n-y", type=int)
    parser.add_argument("--y-batch-size", type=int, help="Rapidity chunk size; preserves the configured y quadrature")
    parser.add_argument("--output")
    args = parser.parse_args()
    overrides = {k: getattr(args, k) for k in ("device", "dtype", "batch_size", "n_orientations", "b_step_fm", "n_y", "y_batch_size", "output")}
    cfg = load_config(args.config, overrides)
    result = run_campaign(cfg, config_path=args.config, prepare_only=args.prepare_only, max_new_pairs=args.max_new_pairs)
    print(f"{result['status']}: {result['completed_pairs']}/{result['target_pairs_total']} pairs; {cfg['runtime']['output']}")


if __name__ == "__main__":
    main()
