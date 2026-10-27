#!/usr/bin/env python3
"""Validate and benchmark the separate fast shape-scan engine on the chosen grid."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from upc_deformation.shape_scan_campaign import load_config
from upc_deformation.shape_scan_fast_validation import run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--y-batches", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--validation-pairs", type=int, default=2)
    parser.add_argument("--max-memory-fraction", type=float, default=.8)
    args = parser.parse_args()
    cfg = load_config(args.config, {"device": args.device})
    report = run_benchmark(cfg, args.output, batches=args.batches, y_batches=args.y_batches,
        warmup=args.warmup, repeats=args.repeats, validation_pairs=args.validation_pairs,
        max_memory_fraction=args.max_memory_fraction)
    best = report["recommendation"]
    print(f"passed: {Path(args.output)/'benchmark.json'}")
    print(f"Fastest validated scheme: batch={best['batch_size']}, y_batch={best['y_batch_size']}; "
          f"{best['pairs_per_second']:.5g} pairs/s, {best['speedup_vs_reference']:.3g}x reference")


if __name__ == "__main__":
    main()
