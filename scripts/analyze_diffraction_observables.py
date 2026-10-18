#!/usr/bin/env python3
"""Analyze a completed paired full/self campaign and package compact reports."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'upc_diffraction_matplotlib'))
import numpy as np
import yaml

from upc_deformation.diffraction_campaign import SCHEMA, BRANCH_NAMES
from upc_deformation.diffraction_analysis import (
    analyze_diffraction_case, contrast_cases, register_spherical_windows,
)
from upc_deformation.precision_campaign import GRID_KEYS, OutputLock, atomic_json, atomic_npz, sha256

CASES = ('spherical', 'beta2', 'beta3', 'skin')
ANALYSIS_FILES = (
    'scripts/analyze_diffraction_observables.py', 'upc_deformation/diffraction_analysis.py',
    'upc_deformation/diffraction_figures.py', 'upc_deformation/precision_analysis.py',
    'upc_deformation/c2_convergence.py', 'upc_deformation/decay.py',
    'upc_deformation/spectral_weights.py',
)


def load_header(run):
    manifest = json.loads((run/'production_manifest.json').read_text())
    if manifest.get('schema') != SCHEMA or tuple(manifest.get('branch_names', [])) != BRANCH_NAMES:
        raise ValueError('this analyzer requires the paired diffraction schema')
    if manifest.get('status') != 'completed':
        raise ValueError('resume production first; analysis requires a completed campaign')
    if sha256(run/'effective_config.yaml') != manifest['effective_config_sha256']:
        raise ValueError('completed configuration checksum mismatch')
    if sha256(run/manifest['bank_file']) != manifest['bank_sha256']:
        raise ValueError('registered orientation bank checksum mismatch')
    cfg = yaml.safe_load((run/'effective_config.yaml').read_text())
    if [c['name'] for c in cfg['study']['cases']] != list(CASES):
        raise ValueError('unexpected case registration')
    with np.load(run/manifest['bank_file'], allow_pickle=False) as z:
        banks = [z[f'rotation_bank_{i}'] for i in (1, 2)]
    inputs = {k: sha256(run/k) for k in ('production_manifest.json', 'effective_config.yaml')}
    inputs.update(manifest['output_sha256'])
    inputs[manifest['bank_file']] = manifest['bank_sha256']
    return cfg, manifest, banks, inputs


def load_case(run, name, manifest, banks, reference_grid=None):
    path = run/f'{name}.npz'
    if sha256(path) != manifest['output_sha256'][path.name]:
        raise ValueError(f'{name}: production checksum mismatch')
    with np.load(path, allow_pickle=False) as z:
        row = {k: z[k] for k in z.files}
    if (str(row['schema']) != SCHEMA or tuple(row['branch_names']) != BRANCH_NAMES
            or str(row['fingerprint']) != manifest['fingerprint']):
        raise ValueError(f'{name}: branch/schema/fingerprint mismatch')
    n = manifest['npairs'][name]
    c, levels = row['orientation_coefficients'], row['b_level_coefficients']
    if (c.shape != (n, 2, 3, len(row['radii']))
            or levels.shape != (len(row['b_steps_fm']), *c.shape)
            or not np.array_equal(c, levels[0])):
        raise ValueError(f'{name}: coefficient count or b-level identity mismatch')
    for i, bank in enumerate(banks, 1):
        expected = np.eye(3)[None] if name in ('spherical', 'skin') else bank[:n]
        if not np.array_equal(row[f'rotation_bank_{i}'], expected):
            raise ValueError(f'{name}: paired orientation identity mismatch')
    if reference_grid is not None:
        for key in GRID_KEYS:
            if not np.array_equal(row[key], reference_grid[key]):
                raise ValueError(f'{name}: mismatched {key}')
    return row


def array_leaves(value, prefix='', target=None):
    target = {} if target is None else target
    if isinstance(value, np.ndarray):
        target[prefix] = value
    elif isinstance(value, dict):
        for k, v in value.items():
            if not k.startswith('_'):
                array_leaves(v, f'{prefix}__{k}' if prefix else k, target)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            array_leaves(v, f'{prefix}__{i}', target)
    return target


def expanded_valid(values, size, key='value'):
    """Expand a compact valid-column diagnostic to nullable CSV coordinates."""
    mask = values.get('valid_mask', np.ones(size, dtype=bool))
    result = [None]*size
    for index, value in zip(np.flatnonzero(mask), values[key]):
        result[index] = float(value)
    return result


def diagonal_ratio(num, den):
    value = num.mean(0)/den.mean(0)
    u = (num-value*den)/den.mean(0)
    se = u.std(0, ddof=1)/np.sqrt(len(u)) if len(u) > 1 else np.zeros_like(value)
    return value, se, u


def convergence_rows(name, row, prefixes):
    n = row['n_orientations']
    selections = [('final', n, slice(None))]
    selections += [('prefix', p, slice(0, p)) for p in sorted(set(prefixes)) if 2 <= p < n]
    if n >= 4:
        selections += [('half1', n//2, slice(0, n//2)), ('half2', n-n//2, slice(n//2, n))]
    for section in ('curves', 'windows'):
        den, num = (row['_samples'][section+k] for k in ('_N0', '_N2'))
        coordinates = row['curves']['q'] if section == 'curves' else row['windows']['names']
        for kind, count, select in selections:
            stats = [diagonal_ratio(num[select, b], den[select, b]) for b in (0, 1)]
            u = stats[0][2]-stats[1][2]
            estimates = [('full', stats[0][0], stats[0][1]), ('self', stats[1][0], stats[1][1]),
                         ('full_minus_self', stats[0][0]-stats[1][0],
                          u.std(0, ddof=1)/np.sqrt(count) if count > 1 else np.zeros(u.shape[1]))]
            for branch, values, se in estimates:
                for i, coordinate in enumerate(coordinates):
                    yield [name, section, coordinate, kind, count, branch, values[i], se[i]]


def radial_difference(coarse, fine):
    delta = coarse-fine
    n = len(delta)
    return delta.mean(0), delta.std(0, ddof=1)/np.sqrt(n) if n > 1 else np.zeros(delta.shape[1:])


def _write_tables(out, data):
    sphere = data['cases']['spherical']
    with (out/'radial_density.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'q_GeV', 'D', 'D_orientation_se', 'J', 'J_orientation_se',
                         'X_J_minus_D', 'X_orientation_se', 'J_over_D', 'J_over_D_se',
                         'D_over_own_D_at_zero', 'D_shape_over_spherical', 'dN_full_dq_2piqJ'])
        base_d = sphere['radial']['D']['value']
        for name, row in data['cases'].items():
            radial = row['radial']; d, j = radial['D']['value'], radial['J']['value']
            ratio = expanded_valid(radial['J_over_D'], len(d))
            ratio_se = expanded_valid(radial['J_over_D'], len(d), key='orientation_se')
            for i, q in enumerate(radial['q']):
                writer.writerow([name, q, d[i], radial['D']['orientation_se'][i], j[i], radial['J']['orientation_se'][i],
                    radial['J_minus_D']['value'][i], radial['J_minus_D']['orientation_se'][i],
                    ratio[i], ratio_se[i], d[i]/d[0] if d[0] > 0 else None,
                    (d[i]/d[0])/(base_d[i]/base_d[0]) if d[0] > 0 and base_d[i] > 0 and base_d[0] > 0 else None,
                    2*np.pi*q*j[i]])
    with (out/'curves.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'q_GeV', 'C2_full', 'C2_full_se', 'C2_self', 'C2_self_se',
            'C2_full_minus_self', 'paired_se', 'delta_C2_full', 'delta_C2_full_se',
            'delta_C2_self', 'delta_C2_self_se', 'delta_full_minus_self_C2', 'delta_paired_se',
            'N0_full', 'N0_self', 'N0_cross', 'N2_full', 'N2_self', 'N2_cross',
            'C2_full_radial_analytic_diagnostic', 'C2_self_radial_analytic_diagnostic'])
        for name, row in data['cases'].items():
            c, delta = row['curves'], data['contrasts'][name]['curves']
            analytic = {b: expanded_valid(c['analytic_C2_diagnostic'][b], len(c['q'])) for b in ('full', 'self')}
            for i, q in enumerate(c['q']):
                values = [name, q]
                for estimate in (c['full']['C2'], c['self']['C2'], c['full_minus_self_C2'],
                                 delta['full_C2'], delta['self_C2'], delta['double_difference_C2']):
                    values += [estimate['value'][i], estimate['orientation_se'][i]]
                for moment in ('N0', 'N2'):
                    values += [c[b][moment]['value'][i] for b in ('full', 'self', 'cross')]
                values += [analytic[b][i] for b in ('full', 'self')]
                writer.writerow(values)
    with (out/'window_metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'window', 'q_low', 'q_high', 'C2_full', 'C2_full_se', 'C2_self', 'C2_self_se',
            'C2_full_minus_self', 'paired_se', 'delta_C2_full', 'delta_C2_full_se',
            'delta_C2_self', 'delta_C2_self_se', 'delta_full_minus_self_C2', 'delta_paired_se',
            'yield_full', 'yield_self', 'yield_cross', 'N2_full', 'N2_self', 'N2_cross',
            'yield_full_over_spherical', 'yield_self_over_spherical'])
        for name, row in data['cases'].items():
            w, delta = row['windows'], data['contrasts'][name]['windows']
            for i, label in enumerate(w['names']):
                values = [name, label, *w['bounds'][i]]
                for estimate in (w['full']['C2'], w['self']['C2'], w['full_minus_self_C2'],
                                 delta['full_C2'], delta['self_C2'], delta['double_difference_C2']):
                    values += [estimate['value'][i], estimate['orientation_se'][i]]
                for moment in ('N0', 'N2'):
                    values += [w[b][moment]['value'][i] for b in ('full', 'self', 'cross')]
                values += [w[b]['N0']['value'][i]/sphere['windows'][b]['N0']['value'][i] for b in ('full', 'self')]
                writer.writerow(values)
    with (out/'outer_slope_ratios.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'branch', 'outer_over_inner', 'orientation_se', 'relative_change_spherical', 'relative_change_se'])
        for name, row in data['cases'].items():
            for branch, ratio in row['windows'].get('outer_over_inner', {}).items():
                base = sphere['windows']['outer_over_inner'][branch]['value'][0]
                writer.writerow([name, branch, ratio['value'][0], ratio['orientation_se'][0],
                                 ratio['value'][0]/base-1, ratio['orientation_se'][0]/base])


def package_reports(out):
    archive = out.with_name(out.name+'_reports.tar.gz')
    files = sorted(p for p in out.rglob('*') if p.is_file() and not any(s.startswith('.') for s in p.relative_to(out).parts))
    temporary = archive.with_name(archive.name+'.partial')
    with tarfile.open(temporary, 'w:gz') as tar:
        for path in files:
            tar.add(path, arcname=out.name+'/'+str(path.relative_to(out)), recursive=False)
    os.replace(temporary, archive)
    atomic_json(archive.with_suffix(archive.suffix+'.manifest.json'), {
        'archive_sha256': sha256(archive), 'bytes': archive.stat().st_size,
        'files': {str(p.relative_to(out)): sha256(p) for p in files},
        'scope': 'Compact analysis only; retain original per-orientation production files on the server.',
    })
    return archive


def run_analysis(run, output=None, *, figures=True, pack=True):
    run = Path(run).resolve()
    # Hold the production lock for the full read, preventing concurrent extension.
    with OutputLock(run):
        cfg, manifest, banks, inputs = load_header(run)
        n = manifest['npairs']['beta2']
        out = Path(output).resolve() if output else run/f'analysis_n{n}'
        if out == run or out in run.parents:
            raise ValueError('analysis requires a separate directory, not the production or its ancestor')
        out.mkdir(parents=True, exist_ok=True)
        with OutputLock(out):
            return _analyze_locked(run, out, cfg, manifest, banks, inputs, figures, pack)


def _analyze_locked(run, out, cfg, manifest, banks, inputs, figures, pack):
    acfg = cfg['analysis']; qmax = float(cfg['study']['qmax'])
    nq, nphi = int(acfg.get('q_points', 341)), int(acfg.get('angular_points', 720))
    if nq < 3 or nphi < 8:
        raise ValueError('analysis needs at least three q nodes and eight angular nodes')
    fixed = np.asarray(acfg.get('fixed_pT_GeV', []), float)
    if not np.isfinite(fixed).all() or np.any((fixed < 0) | (fixed > qmax)):
        raise ValueError('fixed diagnostic q must be inside the analysis domain')
    q = np.unique(np.r_[np.linspace(0, qmax, nq)[1:], fixed[fixed > 0]])
    angles = np.arange(nphi)*2*np.pi/nphi
    identity = {'input_sha256': inputs, 'config': acfg, 'figures': figures,
                'code_sha256': {p: sha256(ROOT/p) for p in ANALYSIS_FILES}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    previous = out/'analysis_manifest.json'
    if previous.exists() and json.loads(previous.read_text()).get('fingerprint') != fingerprint:
        raise ValueError('data or analysis changed; choose a new --output directory')
    state = {**identity, 'fingerprint': fingerprint, 'status': 'running'}
    atomic_json(previous, state)
    try:
        sphere_raw = load_case(run, 'spherical', manifest, banks)
        grid = {k: sphere_raw[k] for k in GRID_KEYS}
        registry = register_spherical_windows(grid['radii'], sphere_raw['orientation_coefficients'][0, 1, 0],
            qmax=qmax, inner_min=float(acfg.get('bright_inner_min_GeV', .03)),
            outer_fractions=acfg.get('outer_fractions', [.5, .1]),
            fixed_window=acfg.get('fixed_window_GeV', [.12, .14]))
        data = {'config': cfg, 'registry': registry, 'cases': {}, 'contrasts': {}, 'b_grid': {}}
        sphere_b_changes = {}
        with (out/'orientation_convergence.csv').open('w', newline='') as conv, (out/'b_grid_comparison.csv').open('w', newline='') as bg:
            cw, bw = csv.writer(conv), csv.writer(bg)
            cw.writerow(['case', 'section', 'q_or_window', 'sample', 'n_orientations', 'branch', 'C2', 'orientation_se'])
            bw.writerow(['case', 'b_step_fm', 'section', 'q_or_window', 'metric', 'coarse_minus_fine', 'paired_orientation_se', 'coarse_minus_fine_structure_response'])
            for name in CASES:
                raw = sphere_raw if name == 'spherical' else load_case(run, name, manifest, banks, grid)
                coeff = raw['orientation_coefficients']
                row = analyze_diffraction_case(coeff, grid['radii'], grid['axis'], q, angles, registry, keep_samples=True)
                row['case'] = next(c for c in cfg['study']['cases'] if c['name'] == name)
                # Per-case locations diagnose width response; they never change the common windows.
                row['self_profile_locations'] = register_spherical_windows(grid['radii'], coeff[:, 1, 0].mean(0),
                    qmax=qmax, inner_min=registry['inner_min'], outer_fractions=registry['outer_fractions'], fixed_window=registry['fixed_window'])
                row['mean_coefficients'] = coeff.mean(0)
                data['cases'][name] = row
                data['contrasts'][name] = contrast_cases(row, data['cases']['spherical'])
                cw.writerows(convergence_rows(name, row, acfg.get('orientation_prefixes', [])))
                data['b_grid'][name] = []
                for level, step in enumerate(raw['b_steps_fm'][1:], 1):
                    coarse_coeff = raw['b_level_coefficients'][level]
                    coarse = analyze_diffraction_case(coarse_coeff, grid['radii'], grid['axis'], q, angles, registry, keep_samples=True)
                    difference = contrast_cases(coarse, row, paired=True)
                    if name == 'spherical':
                        sphere_b_changes[level] = difference
                    for section in ('curves', 'windows'):
                        coordinates = q if section == 'curves' else row['windows']['names']
                        for metric, estimate in difference[section].items():
                            base = sphere_b_changes[level][section][metric]['value']
                            for i, coordinate in enumerate(coordinates):
                                bw.writerow([name, step, section, coordinate, metric, estimate['value'][i],
                                             estimate['orientation_se'][i], estimate['value'][i]-base[i]])
                    keep = grid['radii'] <= qmax
                    rv, re = radial_difference(coarse_coeff[:, :, 0, keep], coeff[:, :, 0, keep])
                    for branch, metric in enumerate(('J_density', 'D_density')):
                        for i, coordinate in enumerate(grid['radii'][keep]):
                            bw.writerow([name, step, 'radial', coordinate, metric, rv[branch, i], re[branch, i], ''])
                    data['b_grid'][name].append({'b_step_fm': float(step), 'C2_coarse_minus_fine': difference,
                                                'radial_coarse_minus_fine': rv, 'radial_paired_se': re})
                    del coarse, coarse_coeff
                row.pop('_samples')
                del coeff, raw
                gc.collect()
                print(f'Analyzed {name}: {row["n_orientations"]} paired full/self orientations', flush=True)
        _write_tables(out, data)
        if any(sha256(ROOT/p) != checksum for p, checksum in identity['code_sha256'].items()):
            raise ValueError('analysis source code changed during execution; rerun with a new output directory')
        atomic_json(out/'windows.json', registry)
        atomic_npz(out/'summary_arrays.npz', radii=grid['radii'], q=q, **array_leaves(data))
        summary = {'n_orientations': manifest['npairs'], 'registry': registry,
                   'b_steps_fm': grid['b_steps_fm'], 'cases': {},
                   'uncertainty_scope': 'Orientation Monte Carlo SE only; b differences are diagnostics, not strict error bounds. '
                        'No detector, event-count, model or boundary-location uncertainty included.',
                   'definitions': data['cases']['spherical']['definitions']}
        for name, row in data['cases'].items():
            summary['cases'][name] = {'case': row['case'], 'n_orientations': row['n_orientations'],
                'self_profile_locations': row['self_profile_locations'], 'windows': row['windows'],
                'window_structure_response': data['contrasts'][name]['windows']}
        atomic_json(out/'summary.json', summary)
        if figures:
            from upc_deformation.diffraction_figures import render_diffraction_figures
            render_diffraction_figures(data, out)
        report = (
            '# Diffraction reference and angular-response diagnostics\n\n'
            f'Production fingerprint: `{manifest["fingerprint"]}`.\n\n'
            f'Orientation counts: `{manifest["npairs"]}`.\n\n'
            'Read `windows.json` for frozen spherical-self boundaries and censored minima; '
            '`radial_density.csv` for D, J and X on the native radial grid; `curves.csv` for '
            'Cartesian C2 and paired structural contrasts; `window_metrics.csv` and '
            '`outer_slope_ratios.csv` for fixed-annulus observables. `summary_arrays.npz` '
            'contains compact means/covariances, not the large per-orientation production banks.\n\n'
            '`orientation_convergence.csv` retains prefixes and disjoint halves. '
            '`b_grid_comparison.csv` uses the same windows and paired orientations at every b step.\n\n'
            'D is self A0, a model reference density without a q Jacobian. '
            'self remains coherent within each nucleus. Full-minus-self C2 is a diagnostic difference, '
            'not a normalized cross-term moment. Window boundaries are reference choices, not optimized cuts.\n\n'
            + summary['uncertainty_scope']+'\n'
        )
        (out/'report.md').write_text(report)
        state.update(status='completed', output_sha256={str(p.relative_to(out)): sha256(p)
            for p in sorted(out.rglob('*')) if p.is_file() and p != previous and not p.name.startswith('.')})
        atomic_json(previous, state)
        if pack:
            archive = package_reports(out)
            print(f'Return report archive: {archive}', flush=True)
        return summary
    except BaseException as error:
        atomic_json(previous, {**state, 'status': 'failed', 'error': repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('--no-package', action='store_true')
    args = parser.parse_args()
    run_analysis(args.run, args.output, figures=not args.no_figures, pack=not args.no_package)


if __name__ == '__main__':
    main()
