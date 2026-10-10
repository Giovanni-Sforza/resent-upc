#!/usr/bin/env python3
"""Targeted quadrature/grid changes on four matched deformation orientations.

This screens selected numerical directions against the structure signal; it
does not certify convergence of all dimensions or the full orientation average.
The production calculation implementation is imported unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from upc_deformation.observables import calculate_case, coefficient_metrics, METRIC_NAMES


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def save_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')


def load_case(directory, name):
    result = json.loads((directory / f'{name}.json').read_text())
    with np.load(directory / f'{name}.npz') as arrays:
        result.update({key: arrays[key] for key in arrays.files})
    return result


def save_case(directory, result):
    name = result['case']['name']
    np.savez_compressed(directory / f'{name}.npz',
                        **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    save_json(directory / f'{name}.json',
              {k: v for k, v in result.items() if not isinstance(v, np.ndarray)})


def selected_metrics(result, q0, qmax):
    return np.array([[coefficient_metrics(branch, result['radii'], qmax, q0)[name]
                      for name in METRIC_NAMES]
                     for branch in result['coefficients'][-1]])


def relative_signal_shift(shift, signal):
    return np.divide(np.abs(shift), np.abs(signal),
                     out=np.full_like(shift, np.nan), where=np.abs(signal) > 1e-14)


def finite_json(value):
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def summarize(results, config):
    scans = {}
    qmax = config['study']['qmax']
    for q0 in config['study']['q0_candidates']:
        variants = {}
        for variant, cases in results.items():
            spherical = selected_metrics(cases['spherical'], q0, qmax)
            beta2 = selected_metrics(cases['beta2'], q0, qmax)
            delta = beta2 - spherical
            variants[variant] = {
                'spherical_values': spherical, 'beta2_first4_values': beta2,
                'structure_delta': delta,
                'structure_delta_full_minus_self': delta[0] - delta[1],
            }
        baseline = variants['base']
        for variant, item in variants.items():
            if variant == 'base':
                continue
            shift = item['structure_delta'] - baseline['structure_delta']
            ratio = relative_signal_shift(shift, baseline['structure_delta'])
            differential = (item['structure_delta_full_minus_self']
                            - baseline['structure_delta_full_minus_self'])
            differential_ratio = relative_signal_shift(
                differential, baseline['structure_delta_full_minus_self'])
            item.update({
                'change_of_structure_delta': shift,
                'change_over_absolute_base_signal': ratio,
                'below_10pct_of_base_signal': ratio < .1,
                'change_of_full_minus_self_structure_delta': differential,
                'change_over_absolute_base_full_minus_self_signal': differential_ratio,
                'full_minus_self_below_10pct_of_base_signal': differential_ratio < .1,
                'spherical_absolute_metric_shift': item['spherical_values'] - baseline['spherical_values'],
            })
        scans[str(q0)] = variants
    return finite_json({
        'status': 'targeted checks only; no full convergence certification',
        'metric_names': METRIC_NAMES, 'branch_names': ['full', 'source_self'],
        'comparison': 'same first four random orientations of each nucleus, beta2=0.30 versus spherical',
        'qmax_gev': qmax, 'q0_scan': scans,
        'criterion': 'change in structure contrast smaller than 10% of its base value; componentwise only',
        'scope': 'rapidity Gauss 5 to 7; WS/T grid 64 to 80 and output grid 320 to 400 at fixed extents; b and orientation convergence not certified',
    })


def write_report(destination, summary):
    lines = ['# Targeted numerical checks', '', summary['status'], '',
             summary['comparison'], '', summary['scope'], '',
             'All values below use q0=0.02 GeV/c and qmax=0.17 GeV/c.', '',
             '| Variation | Branch | Observable | Base structure difference | Shift of difference | Shift / base signal | <10% |',
             '|---|---|---|---:|---:|---:|---|']
    rows = summary['q0_scan']['0.02']
    for variant in ['rapidity7', 'spatial400']:
        item = rows[variant]
        for branch_index, branch in enumerate(summary['branch_names']):
            for metric_index, metric in enumerate(summary['metric_names']):
                base = rows['base']['structure_delta'][branch_index][metric_index]
                shift = item['change_of_structure_delta'][branch_index][metric_index]
                ratio = item['change_over_absolute_base_signal'][branch_index][metric_index]
                criterion = item['below_10pct_of_base_signal'][branch_index][metric_index]
                ratio_text = f'{ratio:.3%}' if ratio is not None else 'undefined'
                lines.append(f'| {variant} | {branch} | {metric} | {base:.8g} | {shift:.8g} | {ratio_text} | {criterion} |')
    lines += ['', 'The JSON includes all registered q0 candidates and the difference between full and source-self structure responses.',
              'A component exceeding the threshold remains numerically unresolved by this check. Four orientations are used only to match integration points; they do not represent a converged orientation ensemble.',
              'Each variant directory retains its exact config, quadrature nodes/weights, rotation banks, coefficients and density metadata.', '']
    (destination / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/jpsi_observables_revision.yaml')
    parser.add_argument('--production', default='outputs/jpsi_observables_revision/main_20261007')
    parser.add_argument('--output', default='outputs/jpsi_observables_revision/convergence')
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    production = Path(args.production).resolve()
    config = yaml.safe_load(Path(args.config).read_text())
    config['study'].update(n_orientations=4, threads=2, n_phi=256, n_radial=757)
    cases = [c for c in config['study']['cases'] if c['name'] in ['spherical', 'beta2']]
    config['study']['cases'] = cases
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    log_file = (destination / 'run.log').open('a', buffering=1)

    def log(message, **kwargs):
        print(message, **kwargs)
        log_file.write(str(message) + '\n')

    variants = {'base': copy.deepcopy(config), 'rapidity7': copy.deepcopy(config),
                'spatial400': copy.deepcopy(config)}
    variants['rapidity7']['study']['n_y'] = 7
    for key in ['WS', 'T_prim']:
        variants['spatial400']['physics']['grid_parameters'][key]['N'] = 80
    variants['spatial400']['physics']['grid_parameters']['output']['N'] = 400
    results = {}
    started = time.monotonic()
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [Path(__file__), ROOT / 'upc_deformation/observables.py',
                        ROOT / 'upc_deformation/source_amplitudes.py', ROOT / 'upc_deformation/forward.py',
                        ROOT / 'upc_deformation/decay.py', ROOT / 'upc_deformation/vendor/upc_forward.py']}
    save_json(destination / 'manifest.json', {'status': 'running', 'source_hashes': hashes,
                                             'command': sys.argv, 'threads': 2})
    # Refined jobs run first, letting an already-running production job finish
    # its first deformation cache, when possible.
    for variant in ['rapidity7', 'spatial400', 'base']:
        cfg = variants[variant]
        folder = destination / variant
        folder.mkdir(exist_ok=True)
        frozen = yaml.safe_dump(cfg, sort_keys=False)
        if (folder / 'config.yaml').exists() and (folder / 'config.yaml').read_text() != frozen:
            raise ValueError(f'Existing {folder} configuration differs; choose a new output')
        (folder / 'config.yaml').write_text(frozen)
        results[variant] = {}
        for case in cases:
            name = case['name']
            if (folder / f'{name}.npz').exists():
                result = load_case(folder, name)
                log(f'Reused {variant}/{name}', flush=True)
            elif variant == 'base' and name == 'spherical' and (production / 'spherical.npz').exists():
                result = load_case(production, name)
                result['convergence_source'] = str(production / f'{name}.npz')
                save_case(folder, result)
                log('Reused production spherical cache', flush=True)
            elif variant == 'base' and name == 'beta2' and (production / 'beta2.npz').exists():
                archived = load_case(production, name)
                result = {key: archived[key] for key in ['case', 'density', 'physics',
                          'stage_names', 'axis', 'radii', 'b_values', 'b_weights',
                          'y_values', 'y_weights']}
                for key in ['rotation_bank_1', 'rotation_bank_2', 'orientation_coefficients']:
                    result[key] = archived[key][:4]
                result['coefficients'] = result['orientation_coefficients'].mean(0)
                result['n_orientations'] = 4
                result['convergence_source'] = str(production / 'beta2.npz')
                result['cache_scope'] = 'first four orientation coefficients and matched rotation banks; no reconstructed per-node subset'
                save_case(folder, result)
                log('Reused production beta2 first-four-orientation coefficients', flush=True)
            else:
                log(f'Starting {variant}/{name}', flush=True)
                result = calculate_case(cfg, case, log)
                save_case(folder, result)
            results[variant][name] = result
    # Random rotations must be exactly shared across all variants.
    for key in ['rotation_bank_1', 'rotation_bank_2']:
        base_bank = results['base']['beta2'][key]
        for variant in ['rapidity7', 'spatial400']:
            np.testing.assert_array_equal(base_bank, results[variant]['beta2'][key])
        if (production / 'beta2.npz').exists():
            with np.load(production / 'beta2.npz') as cached:
                np.testing.assert_array_equal(base_bank, cached[key][:4])
    summary = summarize(results, config)
    save_json(destination / 'comparison.json', summary)
    write_report(destination, summary)
    save_json(destination / 'manifest.json', {'status': 'completed_targeted_checks_only',
              'source_hashes': hashes, 'command': sys.argv, 'threads': 2,
              'elapsed_seconds': time.monotonic() - started})
    log(f'Completed targeted checks in {time.monotonic()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
