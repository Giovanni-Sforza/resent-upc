#!/usr/bin/env python3
"""Run/resume the validated second-generation A0 scan in its own output directory."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from upc_deformation.shape_scan_campaign import load_config
from upc_deformation.shape_scan_fast_v2_campaign import run_campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True,
                        help="Separate fast campaign directory; never the legacy production directory")
    parser.add_argument("--validation", help="Passing report from benchmark_shape_scan_fast_v2.py")
    parser.add_argument("--import-run", help="Stopped legacy or fast-v1 campaign to import without recomputing its prefix")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--case", action="append", dest="selected_cases")
    parser.add_argument("--max-new-pairs", type=int)
    parser.add_argument("--n-orientations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint-size", type=int)
    parser.add_argument("--y-batch-size", type=int)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, {key: getattr(args, key) for key in
        ("n_orientations", "batch_size", "checkpoint_size", "y_batch_size", "device", "output")})
    if args.list_cases:
        print(json.dumps(cfg["study"]["cases"], indent=2)); return 0
    result = run_campaign(cfg, config_path=args.config, validation_path=args.validation,
        import_run=args.import_run, prepare_only=args.prepare_only,
        max_new_pairs=args.max_new_pairs, selected_cases=args.selected_cases)
    print(f"{result['status']}: {result['completed_pairs']}/{result['target_pairs_total']} pairs; {cfg['runtime']['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
