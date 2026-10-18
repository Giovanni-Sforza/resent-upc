#!/usr/bin/env python3
"""Check the paired kernel against the previous full kernel and CPU/CUDA."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from upc_deformation.diffraction_campaign import load_config, code_hashes
from upc_deformation.diffraction_forward import DiffractionRefiner
from upc_deformation.precision_campaign import atomic_json, configure_runtime, rotation_bank, sha256
from upc_deformation.precision_forward import PrecisionPanelDRefiner
from upc_deformation.precision_validation import (
    compare_payloads, project_payload, tiny_configuration,
)


def branch_payload(payload, branch):
    return {**payload, 'coefficients': payload['coefficients'][:, branch],
            'b_level_coefficients': payload['b_level_coefficients'][:, :, branch]}


def run_validation(cfg, output, *, tiny=False, pairs=2, case_name='beta2'):
    if pairs < 1:
        raise ValueError('pairs must be positive')
    cfg = tiny_configuration(cfg) if tiny else deepcopy(cfg)
    environment = configure_runtime(cfg['runtime'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'validation.json').exists():
        raise ValueError('validation already exists; choose a new output directory')
    banks, bank_meta = rotation_bank(cfg['orientations'], pairs)
    case = next(c for c in cfg['study']['cases'] if c['name'] == case_name)
    dtype, device = cfg['runtime']['dtype'], cfg['runtime']['device']
    kernel = DiffractionRefiner(cfg, case, device=device, dtype=dtype)
    actual = kernel.calculate(*banks)
    singles = [kernel.calculate(banks[0][i:i+1], banks[1][i:i+1]) for i in range(pairs)]
    serial = {**actual,
              'coefficients': np.concatenate([p['coefficients'] for p in singles]),
              'b_level_coefficients': np.concatenate([p['b_level_coefficients'] for p in singles], axis=1)}
    q = np.unique(np.r_[np.linspace(0, .17, 86), .12, .13, .14])
    angles = np.arange(180)*2*np.pi/180
    comparisons = {}
    for index, branch in enumerate(('full', 'self')):
        a, s = branch_payload(actual, index), branch_payload(serial, index)
        comparisons[f'batch_{branch}'] = compare_payloads(
            a, s, project_payload(a, q, angles), project_payload(s, q, angles),
            precision=dtype, comparison_kind='same_precision_batch')
    # The previous production configuration FFTs the summed source fields.
    # Comparing it here also checks linearity against the new two-FFT path.
    old = PrecisionPanelDRefiner(cfg, case, device=device, dtype=dtype, fft_mode='combined').calculate(*banks)
    full = branch_payload(actual, 0)
    comparisons['legacy_full'] = compare_payloads(
        full, old, project_payload(full, q, angles), project_payload(old, q, angles),
        precision=dtype, comparison_kind='same_precision_batch')
    if environment['device_type'] == 'cuda':
        cpu = DiffractionRefiner(cfg, case, device='cpu', dtype=dtype).calculate(*banks)
        for index, branch in enumerate(('full', 'self')):
            a, c = branch_payload(actual, index), branch_payload(cpu, index)
            comparisons[f'cuda_cpu_{branch}'] = compare_payloads(
                a, c, project_payload(a, q, angles), project_payload(c, q, angles),
                precision=dtype, comparison_kind='same_precision_cross_device')
    report = {
        'status': 'passed' if all(v['passed'] for v in comparisons.values()) else 'failed',
        'cuda_validation_performed': environment['device_type'] == 'cuda',
        'tiny_grid': tiny, 'case': case, 'pairs': pairs, 'config': cfg,
        'environment': environment, 'orientation_bank': bank_meta,
        'code_sha256': {**code_hashes(), 'scripts/validate_diffraction_observables.py': sha256(__file__)},
        'comparisons': comparisons,
        'legacy_full_fft_mode': 'combined',
        'forward_elapsed_seconds': actual['elapsed_seconds'],
        'peak_memory_allocated_bytes': actual.get('peak_memory_allocated_bytes', 0),
        'scope': 'Implementation and batch/device closure only; not physical or grid convergence. '
                 'Timing is an un-warmed diagnostic, not a benchmark.',
    }
    atomic_json(output/'validation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/diffraction_observables_a100.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    parser.add_argument('--pairs', type=int, default=2)
    parser.add_argument('--b-step', type=float, dest='b_step_fm')
    parser.add_argument('--y-batch-size', type=int)
    parser.add_argument('--case', choices=('spherical', 'beta2', 'beta3', 'skin'), default='beta2')
    parser.add_argument('--tiny', action='store_true', help='Real forward smoke on deliberately coarse grids')
    args = parser.parse_args()
    cfg = load_config(args.config, {k: getattr(args, k) for k in ('device', 'dtype', 'b_step_fm', 'y_batch_size')})
    report = run_validation(cfg, args.output, tiny=args.tiny, pairs=args.pairs, case_name=args.case)
    print(f"{report['status']}: {args.output/'validation.json'}")
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
