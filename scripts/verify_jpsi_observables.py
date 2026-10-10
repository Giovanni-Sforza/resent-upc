#!/usr/bin/env python3
"""Check saved bridge invariants and report unresolved numerical precision.

An invariant pass is not a detector validation or a scientific qualification.
Cached b-quadrature comparisons and finite-orientation sampling errors are
assessed here.  When matched targeted rapidity/spatial refinement evidence is
present, it is linked with its limited four-orientation scope; this is not a
global convergence certification.  No forward simulation is launched here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from upc_deformation.observables import (
    METRIC_NAMES, coefficient_metrics, orientation_metric_se,
)


def _linear_moment(coeff, radii, upper, power):
    """Exact integral of q**power times piecewise-linear coefficient values."""
    r = np.r_[radii[radii < upper], upper]
    v = np.interp(r, radii, coeff)
    h = np.diff(r)
    slopes = np.diff(v) / h
    total = 0.
    # q=r_i+t, coefficient=v_i+s_i*t on each interval.
    import math
    for k in range(power + 1):
        factor = math.comb(power, k) * r[:-1]**(power-k)
        total += np.sum(factor * (v[:-1] * h**(k+1)/(k+1)
                                  + slopes * h**(k+2)/(k+2)))
    return float(total)


def _exact_linear_metrics(coeff, radii, qmax, q0):
    denominator = _linear_moment(coeff[0], radii, qmax, 1)
    return np.array([
        _linear_moment(coeff[0], radii, q0, 1) / denominator,
        _linear_moment(coeff[0], radii, qmax, 2) / denominator,
        -.25 * _linear_moment(coeff[1], radii, qmax, 1) / denominator,
    ])


def _values(coeff, r, cfg):
    metrics = coefficient_metrics(coeff, r, cfg['qmax'], cfg['q0'])
    return np.array([metrics[key] for key in METRIC_NAMES])


def _ratio(numerator, denominator):
    return [float(a / b) if b > 0 else None
            for a, b in zip(np.abs(numerator), np.abs(denominator))]


def _verify_b_window_reuse(result, name, check):
    """Verify saved extension provenance, endpoint closure and untouched tail."""
    reuse = result.get('b_window_reuse')
    if not reuse:
        return None
    prefix = f'{name}: b-window reuse '
    n = result['n_orientations']
    check(prefix+'node geometry', np.array_equal(result['b_values'], np.arange(12.,61.,2.))
          and reuse.get('source_b_window_fm') == [20.,60.]
          and reuse.get('extended_b_window_fm') == [12.,60.]
          and reuse.get('evaluated_b_nodes_fm') == [12.,14.,16.,18.,20.]
          and reuse.get('new_b_nodes_fm') == [12.,14.,16.,18.]
          and reuse.get('recomputed_seam_b_fm') == 20.)
    check(prefix+'calculation counts', reuse.get('source_calls_per_orientation') == 5
          and reuse.get('new_nodes_per_orientation') == 4
          and reuse.get('seam_calls_per_orientation') == 1
          and reuse.get('retained_old_b_nodes') == 21
          and reuse.get('total_source_calls') == 5*n
          and reuse.get('full_replay_source_calls_avoided') == 20*n)
    closures = reuse.get('b20_closure', {})
    orientation_rows = closures.get('per_orientation_fixed_coefficients', [])
    check(prefix+'orientation closure count', len(orientation_rows) == n)
    named_rows = {key: closures.get(key, {}) for key in (
        'mean_coefficients','half_coefficients','fixed_images','empirical_probability')}
    named_rows.update({f'orientation_{i}': row for i,row in enumerate(orientation_rows)})
    ratios = []
    for label, row in named_rows.items():
        error = row.get('max_error_over_archived_peak', float('nan'))
        absolute = row.get('max_absolute_error', float('nan'))
        rtol = row.get('relative_tolerance', float('nan'))
        atol = row.get('absolute_tolerance_over_peak', float('nan'))
        valid = (np.isfinite([error,absolute,rtol,atol]).all()
                 and 0 <= error <= rtol+atol and absolute >= 0
                 and 0 < rtol <= 5e-5 and 0 <= atol <= 2e-6)
        check(prefix+'b20 '+label, valid, row)
        if np.isfinite(error): ratios.append(float(error))
    source = Path(reuse.get('cache_source', ''))
    source_meta = source.with_suffix('.json')
    source_available = source.is_file() and source_meta.is_file()
    check(prefix+'source preserved', source_available)
    if source_available:
        old_meta = json.loads(source_meta.read_text())
        check(prefix+'source hash identity',
              old_meta.get('calculation_hash') == reuse.get('source_calculation_hash'))
        check(prefix+'same case', all(old_meta['case'].get(key) == result['case'].get(key)
              for key in ('name','beta2','beta3','skin_fm')))
        old_physics, physics = copy.deepcopy(old_meta['physics']), copy.deepcopy(result['physics'])
        old_physics['collision_parameters'].pop('b')
        physics['collision_parameters'].pop('b')
        check(prefix+'same effective physics apart from initial b', old_physics == physics)
        check(prefix+'same nuclear density', old_meta['density'] == result['density'])
        with np.load(source, allow_pickle=False) as archived:
            check(prefix+'old node geometry', np.array_equal(archived['b_values'],np.arange(20.,61.,2.)))
            for key in ('rotation_bank_1','rotation_bank_2'):
                check(prefix+'same '+key, np.array_equal(archived[key],result[key]))
            for key, closure_key in (
                ('node_coefficients','mean_coefficients'),
                ('half_node_coefficients','half_coefficients'),
                ('incoherent_probabilities','empirical_probability')):
                old = archived[key]
                check(prefix+'unchanged b>=22 '+key, np.array_equal(result[key][5:],old[1:]))
                peak = max(float(np.abs(old[0]).max()),1e-30)
                seam_error = float(np.abs(result[key][4]-old[0]).max())
                # Uniform 2 fm steps give equal left/right half weights at 20.
                # A weighted average can add float64 arithmetic roundoff even
                # when the two float32 source evaluations are bitwise equal.
                bound = .5*closures[closure_key]['max_absolute_error']+5e-14*peak
                check(prefix+'endpoint weighted mean '+key, seam_error <= bound,
                      {'max_absolute_error':seam_error,'bound':bound})
    return {'source_cache':str(source), 'b20_max_error_over_archived_peak':max(ratios,default=None),
            'source_calls':reuse.get('total_source_calls'),
            'full_replay_calls_avoided':reuse.get('full_replay_source_calls_avoided'),
            'meaning':'Reuse/integration consistency only; does not validate the added b=12--20 physics or convergence.'}


def _verify_dark_region_diagnostics(report, destination, config, results, check):
    """Check raw regional decomposition and paired covariance identities."""
    path = destination/'dark_region_diagnostics.json'
    if not path.exists():
        return
    diagnostic = json.loads(path.read_text())
    cfg, base = config['study'], results.get('spherical')
    check('dark-region diagnostics: reference available', base is not None)
    if base is None:
        return
    check('dark-region diagnostics: same observable windows',
          diagnostic.get('qmax_gev') == cfg['qmax'] and diagnostic.get('q0_gev') == cfg['q0'])
    check('dark-region diagnostics: declared mask construction',
          diagnostic.get('outer_radius_fraction') == .9 and diagnostic.get('dark_pixel_quantile') == .05)
    axis = base['axis']
    x, y = np.meshgrid(axis,axis,indexing='ij')
    radius = np.hypot(x,y)
    disk = radius <= cfg['qmax']
    pixel_area = float(axis[1]-axis[0])**2
    stages = list(base['stage_names'])
    masks, reference_yields = {}, {}
    for si, stage in enumerate(stages):
        reference = base['images'][si]
        threshold = np.quantile(reference[0,disk],.05)
        all_dark = disk & (reference[0] <= threshold)
        masks[stage,'all_dark'] = all_dark
        masks[stage,'outer_dark'] = all_dark & (radius > .9*cfg['qmax'])
        reference_yields[stage] = float(reference[1,disk].sum()*pixel_area)
    expected = {(name,stage,region) for name in results for stage in stages
                for region in ('all_dark','outer_dark')}
    rows = diagnostic.get('records',[])
    keys = [(row['case'],row['stage'],row['region']) for row in rows]
    check('dark-region diagnostics: complete unique case-stage-region coverage',
          len(keys) == len(expected) and set(keys) == expected)
    orientation_closures = []
    additive_errors = []
    covariance_identity_errors = []
    for row in rows:
        key = row['case'],row['stage'],row['region']
        label = 'dark '+ '/'.join(key)
        if key not in expected:
            continue
        name, stage, region = key
        si = stages.index(stage)
        mask = masks[stage,region]
        count = int(mask.sum())
        image = results[name]['images'][si]
        raw = np.array([float(image[i,mask].sum()*pixel_area) for i in (0,1)])
        nfull,nself = raw
        cross_image = image[0]-image[1]
        ncross = float(cross_image[mask].sum()*pixel_area)
        y0 = reference_yields[stage]
        scale = max(float(np.abs(raw).max()),np.finfo(float).tiny)
        check(label+': pixel count/status', row['pixel_count'] == count
              and row['status'] == ('available' if count else 'empty_region'))
        check(label+': raw image yields/common denominator',
              np.allclose([row['raw_N_full'],row['raw_N_self'],row['raw_N_cross']],
                          [nfull,nself,ncross],rtol=1e-12,atol=1e-14*scale)
              and np.isclose(row['common_Y0_spherical_self_disk'],y0,rtol=1e-12))
        k, activity = row['K_R'], row['A_R']
        if nself > 0:
            expected_k = ncross/nself
            expected_activity = float(np.abs(cross_image[mask]).sum()*pixel_area/nself)
            finite = k is not None and activity is not None and np.isfinite([k,activity]).all()
            check(label+': bounded K/A and absolute K<=A', finite
                  and -1-2e-6 <= k <= 1+2e-6 and -2e-10 <= activity <= 1+2e-6
                  and abs(k) <= activity+2e-10)
            check(label+': K/A from signed raw cross image', finite
                  and np.allclose([k,activity],[expected_k,expected_activity],rtol=1e-12,atol=1e-14))
        else:
            check(label+': undefined empty ratios', k is None and activity is None)
        baseline = base['images'][si]
        base_full = float(baseline[0,mask].sum()*pixel_area)
        base_self = float(baseline[1,mask].sum()*pixel_area)
        expected_delta = np.array([(nself-base_self)/y0,
                                   (ncross-(base_full-base_self))/y0,
                                   (nfull-base_full)/y0])
        delta = np.array([row[f'delta_N_{branch}_over_Y0'] for branch in ('self','cross','full')])
        additive_error = float(delta[0]+delta[1]-delta[2])
        additive_errors.append(abs(additive_error))
        check(label+': common-normalization additive closure',
              np.allclose(delta,expected_delta,rtol=1e-11,atol=1e-13)
              and abs(additive_error) <= 1e-12
              and abs(row['additive_closure_error']-additive_error) <= 1e-14)
        errors = row['orientation_se']
        check(label+': no unsupported A_R error', errors['A_R'] is None)
        if si > 0 and count:
            closure = row['orientation_closure']
            recorded_means = np.array([closure['orientation_mean_raw_N_full'],
                                       closure['orientation_mean_raw_N_self']])
            actual_error = float(np.abs(recorded_means-raw).max())
            relative_error = actual_error/scale
            orientation_closures.append(relative_error)
            check(label+': orientation means match images',
                  closure['orientation_mean_image_closure_passed'] is True
                  and np.allclose(recorded_means,raw,rtol=2e-6,atol=2e-10*scale)
                  and np.isclose(closure['orientation_mean_vs_image_max_absolute_error'],
                                 actual_error,rtol=1e-9,atol=1e-14*scale))
            covariance = np.asarray(errors['delta_yield_covariance'],dtype=float)
            valid_covariance = covariance.shape == (3,3) and np.isfinite(covariance).all()
            check(label+': paired covariance schema', valid_covariance
                  and errors['delta_yield_covariance_order'] == ['self','cross','full'])
            if valid_covariance:
                cov_peak = max(float(np.abs(covariance).max()),np.finfo(float).tiny)
                residual = covariance[:,2]-covariance[:,0]-covariance[:,1]
                covariance_identity_errors.append(float(np.abs(residual).max()/cov_peak))
                se = np.array([errors[f'delta_N_{branch}_over_Y0'] for branch in ('self','cross','full')])
                check(label+': paired covariance full=self+cross',
                      np.allclose(covariance,covariance.T,rtol=1e-10,atol=1e-12*cov_peak)
                      and np.abs(residual).max() <= 1e-9*cov_peak
                      and np.linalg.eigvalsh(covariance).min() >= -1e-10*cov_peak
                      and np.isfinite(se).all() and (se >= 0).all()
                      and np.allclose(se**2,np.diag(covariance),rtol=1e-10,atol=1e-12*cov_peak))
        else:
            check(label+': unavailable orientation errors stay null',
                  row['orientation_closure'] is None and errors['delta_yield_covariance'] is None)
    report['dark_region_evidence'] = {
        'path':str(path),'records_checked':len(rows),
        'max_additive_closure_absolute_error':max(additive_errors,default=0.),
        'max_orientation_mean_image_relative_error':max(orientation_closures,default=0.),
        'max_covariance_identity_error_over_covariance_peak':max(covariance_identity_errors,default=0.),
        'bounds':'-1<=K_R<=1; 0<=A_R<=1; abs(K_R)<=A_R (float32 tolerance)',
        'interpretation':'Raw regional algebra and paired numerical errors; source-self/cross remain theoretical diagnostics.',
    }


def _attach_targeted_evidence(report, destination, config, results):
    """Link existing refinement evidence only after checking its applicability."""
    directory = destination.parent/'convergence'
    required = [directory/'report.md', directory/'comparison.json',
                directory/'manifest.json', directory/'base/config.yaml',
                directory/'base/beta2.npz']
    if not all(path.exists() for path in required):
        return
    comparison = json.loads((directory/'comparison.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    base_config = yaml.safe_load((directory/'base/config.yaml').read_text())
    config_keys = ['longitudinal_units', 'fixed_b_fm', 'b_min_fm', 'b_max_fm',
                   'b_nodes', 'ymax', 'n_y', 'qmax', 'q0', 'n_radial', 'n_phi',
                   'orientation_seed']
    checks = {
        'completed_targeted_run': manifest.get('status') == 'completed_targeted_checks_only',
        'same_base_physics_config': base_config['physics'] == config['physics'],
        'same_base_quadrature_and_observable_definitions': all(
            base_config['study'].get(key) == config['study'].get(key) for key in config_keys),
    }
    run_manifest_path = destination/'manifest.json'
    run_hashes = (json.loads(run_manifest_path.read_text()).get('source_hashes', {})
                  if run_manifest_path.exists() else {})
    core_files = ['upc_deformation/observables.py', 'upc_deformation/source_amplitudes.py',
                  'upc_deformation/forward.py', 'upc_deformation/decay.py',
                  'upc_deformation/vendor/upc_forward.py']
    checks['same_recorded_calculation_sources'] = all(
        key in run_hashes and run_hashes[key] == manifest.get('source_hashes', {}).get(key)
        for key in core_files)
    checks['same_first_four_orientation_pairs'] = False
    if 'beta2' in results:
        with np.load(directory/'base/beta2.npz', allow_pickle=False) as base:
            checks['same_first_four_orientation_pairs'] = all(
                base[key].shape[0] == 4 and np.array_equal(base[key], results['beta2'][key][:4])
                for key in ('rotation_bank_1', 'rotation_bank_2'))
    applicable = all(checks.values())
    rows = comparison.get('q0_scan', {}).get(str(config['study']['q0']), {})
    evidence = {
        'report_path': str(directory/'report.md'),
        'comparison_path': str(directory/'comparison.json'),
        'status': comparison.get('status'), 'scope': comparison.get('scope'),
        'comparison': comparison.get('comparison'),
        'applicable_to_this_run': applicable, 'applicability_checks': checks,
        'role': ('current_run_targeted_check' if applicable
                 else 'historical_reference_not_current_window_validation'),
        'metric_names': comparison.get('metric_names'),
        'branch_names': comparison.get('branch_names'),
        'q0_gev': config['study']['q0'], 'qmax_gev': comparison.get('qmax_gev'),
        ('results_at_registered_q0' if applicable else 'historical_results_at_q0'): {
            variant: {key: row.get(key) for key in (
                'change_of_structure_delta', 'change_over_absolute_base_signal',
                'below_10pct_of_base_signal', 'change_of_full_minus_self_structure_delta',
                'change_over_absolute_base_full_minus_self_signal')}
            for variant, row in rows.items() if variant != 'base'},
        'interpretation': 'Four matched orientations of beta2=0.30 versus spherical only; '
            'does not certify all structures, all orientations, or simultaneous global convergence.',
    }
    report['targeted_numerical_evidence'] = evidence
    if applicable:
        report['unassessed_convergence'] = [item for item in report['unassessed_convergence']
                                            if item not in ('rapidity quadrature', 'spatial grids')]
        report['unassessed_convergence'].append(
            'rapidity/spatial refinement across all cases and the complete orientation ensemble')
    else:
        report['warnings'].append('Targeted refinement evidence exists, but its base calculation '
                                 'does not fully match this run; it does not certify the current '
                                 'b window or its added low-b region. See applicability_checks.')


def verify(destination):
    destination = Path(destination).resolve()
    config = yaml.safe_load((destination / 'config.yaml').read_text())
    cfg = config['study']
    report = {
        'scope': 'Saved numerical invariants and precision screening; not detector validation',
        'pass': True, 'invariants': [], 'cases': {}, 'contrasts': {}, 'warnings': [],
        'unassessed_convergence': ['rapidity quadrature', 'spatial grids',
            'polar sampling grid', 'truth-tail cutoff', 'b-window truncation',
            'physics-model and detector systematics'],
    }

    def check(name, condition, detail=None):
        row = {'name': name, 'pass': bool(condition)}
        if detail is not None:
            row['detail'] = detail
        report['invariants'].append(row)
        report['pass'] &= bool(condition)

    def close(name, actual, expected, rtol=5e-10, relative_atol=1e-12):
        actual, expected = np.asarray(actual), np.asarray(expected)
        scale = max(float(np.max(np.abs(expected))), 1e-30)
        error = float(np.max(np.abs(actual - expected)))
        check(name, np.allclose(actual, expected, rtol=rtol, atol=relative_atol*scale),
              {'max_error': error, 'max_error_over_peak': error/scale})

    summary_path = destination / 'observable_summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary_cases = {row['case']['name']: row for row in summary.get('cases', [])}
    if not summary:
        report['warnings'].append('No observable_summary.json: response products could not be checked.')
    results = {}
    for case in cfg['cases']:
        name = case['name']
        npz_path, meta_path = destination / f'{name}.npz', destination / f'{name}.json'
        check(f'{name}: complete cache', npz_path.exists() and meta_path.exists())
        if not npz_path.exists() or not meta_path.exists():
            continue
        result = json.loads(meta_path.read_text())
        result.update(summary_cases.get(name, {}))
        with np.load(npz_path, allow_pickle=False) as data:
            result.update({key: data[key] for key in data.files})
        results[name] = result
        axis = result['axis'].astype(np.float64)
        r = result['radii'].astype(np.float64)
        b, bw, y, yw = (result[key] for key in ('b_values', 'b_weights', 'y_values', 'y_weights'))
        coeff, node = result['coefficients'], result['node_coefficients']
        images, normalized = result['images'], result['normalized_images']
        check(f'{name}: image schema', images.shape == normalized.shape == (4, 2, len(axis), len(axis)))
        check(f'{name}: coefficient schema', coeff.shape == (3, 2, 3, len(r))
              and node.shape == (len(b), len(y), 2, 3, len(r)))
        check(f'{name}: finite arrays', all(np.isfinite(value).all()
              for value in result.values() if isinstance(value, np.ndarray)))
        check(f'{name}: ordered axes', all(np.all(np.diff(x) > 0) for x in (axis, r, b, y)))
        close(f'{name}: b quadrature area', bw.sum(), np.pi*(b[-1]**2-b[0]**2))
        close(f'{name}: rapidity quadrature measure', yw.sum(), 2*cfg['ymax'])
        bi = int(np.argmin(abs(b-cfg['fixed_b_fm'])))
        yi = int(np.argmin(abs(y)))
        close(f'{name}: fixed node contraction', coeff[0], node[bi, yi])
        close(f'{name}: b contraction', coeff[1], np.einsum('b,bscr->scr', bw, node[:, yi]))
        close(f'{name}: b,y contraction', coeff[2], np.einsum('b,y,byscr->scr', bw, yw, node))
        close(f'{name}: orientation contraction', coeff, result['orientation_coefficients'].mean(0))
        qx, qy = np.meshgrid(axis, axis, indexing='ij')
        disk = qx*qx+qy*qy <= cfg['qmax']**2
        pixel_area = (axis[1]-axis[0])**2
        close(f'{name}: unit disk normalization', (normalized*disk).sum((-1,-2))*pixel_area,
              np.ones((4, 2)), rtol=2e-8)
        close(f'{name}: saved yields', (images*disk).sum((-1,-2))*pixel_area, result['yields'])
        close(f'{name}: normalize after averaging', normalized, images/result['yields'][..., None, None])
        peak = float(images.max())
        check(f'{name}: nonnegative physical intensities', float(images.min()) >= -1e-7*peak)
        a0, c, d = np.moveaxis(coeff, -2, 0)
        check(f'{name}: decay polarization bound', np.all(np.hypot(c, d) <= a0+1e-6*np.max(a0)))
        radius, angle = np.hypot(qx, qy), np.arctan2(qy, qx)
        rebuilt = np.empty_like(images[1:])
        for stage in range(3):
            for branch in range(2):
                aa, cc, dd = [np.interp(radius, r, row) for row in coeff[stage, branch]]
                rebuilt[stage, branch] = aa-.5*(cc*np.cos(2*angle)+dd*np.sin(2*angle))
        close(f'{name}: decay image reconstruction', images[1:], rebuilt, rtol=2e-7, relative_atol=2e-7)
        cross = coeff[:, 0]-coeff[:, 1]
        scale = np.max(np.abs(coeff))
        detail = {
            'axis_change_disk_yield_relative': (result['yields'][1]/result['yields'][0]-1).tolist(),
            'axis_change_interpretation': 'Cartesian/radial discretization closure; no universal pass threshold',
            'signed_interference_A0_range': [float(cross[:,0].min()),float(cross[:,0].max())],
            'interference_has_negative_A0': bool(cross[:,0].min() < -1e-8*scale),
            'interference_has_positive_A0': bool(cross[:,0].max() > 1e-8*scale),
            'interference_policy': 'full minus self is retained with its sign; it is not a probability density',
        }
        if result['n_orientations'] == 1 and not np.any(result['half_node_coefficients']):
            report['warnings'].append(f'{name}: legacy n=1 half cache is zero; do not use it for convergence.')
        reuse_evidence = _verify_b_window_reuse(result,name,check)
        if reuse_evidence is not None:
            detail['b_window_reuse'] = reuse_evidence
        report['cases'][name] = detail
        for row in result.get('response', []):
            values, mc_se = np.asarray(row['values']), np.asarray(row['mc_se'])
            check(f'{name}: finite response at sigma={row["fractional_sigma"]}',
                  np.isfinite(values).all() and np.isfinite(mc_se).all() and np.all(mc_se >= 0))
            covariance = np.asarray(row['event_covariance'])
            check(f'{name}: positive event covariance at sigma={row["fractional_sigma"]}',
                  np.linalg.eigvalsh(covariance).min() >= -1e-12)
            if name == 'spherical':
                close('spherical: response baseline contrast', row['delta_vs_spherical'], np.zeros(3), relative_atol=0)
                close('spherical: paired baseline MC error', row['delta_mc_se'], np.zeros(3), relative_atol=0)
            if row['fractional_sigma'] == 0 and cfg['angular_sigma_rad'] == 0:
                deterministic = _values(coeff[-1,0], r, cfg)
                exact = _exact_linear_metrics(coeff[-1,0], r, cfg['qmax'], cfg['q0'])
                discretization = abs(deterministic-exact)
                residual = abs(values-deterministic)
                within = residual <= 5*mc_se+discretization+1e-12
                detail['zero_response_closure'] = {
                    'residual': residual.tolist(), 'mc_se': mc_se.tolist(),
                    'piecewise_linear_discretization': discretization.tolist(),
                    'within_5_MC_SE_plus_discretization': within.tolist(),
                }
                if not np.all(within):
                    report['warnings'].append(f'{name}: zero-response closure exceeds 5 MC SE plus radial discretization.')

    check('required physical endpoints present', all(key in results for key in ('spherical','beta2','beta3','skin')))
    baseline = results.get('spherical')
    if baseline is not None:
        r = baseline['radii']
        base_fine = _values(baseline['coefficients'][-1,0], r, cfg)
        base_coarse = _values(baseline['coarse_b_coefficients'][0], r, cfg)
        for name, result in results.items():
            if name == 'spherical':
                continue
            fine = _values(result['coefficients'][-1,0], result['radii'], cfg)
            coarse = _values(result['coarse_b_coefficients'][0], result['radii'], cfg)
            delta, coarse_delta = fine-base_fine, coarse-base_coarse
            _, se, _ = orientation_metric_se(result['orientation_coefficients'][:,-1,0],
                result['radii'], cfg['qmax'], cfg['q0'])
            # Spherical reference is deterministic, so the contrast's Monte
            # Carlo orientation error equals that of the deformed numerator.
            check(f'{name}: deterministic spherical contrast reference', baseline['n_orientations'] == 1)
            b_shift = coarse_delta-delta
            numerical_flags = {
                'orientation_signal_below_3_SE': (abs(delta) < 3*se).tolist(),
                'coarse_b_shift_at_least_signal': (abs(b_shift) >= abs(delta)).tolist(),
            }
            report['contrasts'][name] = {
                'names': METRIC_NAMES, 'delta_vs_spherical': delta.tolist(),
                'orientation_MC_SE': se.tolist(), 'signal_over_orientation_SE': _ratio(delta,se),
                'coarse_b_delta_vs_spherical': coarse_delta.tolist(),
                'coarse_minus_fine_contrast': b_shift.tolist(),
                'coarse_b_shift_over_signal': _ratio(b_shift,delta),
                **numerical_flags,
            }
            for metric, unresolved, b_unresolved in zip(METRIC_NAMES,
                    numerical_flags['orientation_signal_below_3_SE'],
                    numerical_flags['coarse_b_shift_at_least_signal']):
                if unresolved or b_unresolved:
                    report['warnings'].append(f'{name}/{metric}: signal unresolved against '
                        + ', '.join(label for label,flag in [('orientation Monte Carlo error',unresolved),
                                                             ('b-quadrature coarsening',b_unresolved)] if flag))
    _attach_targeted_evidence(report, destination, config, results)
    _verify_dark_region_diagnostics(report,destination,config,results,check)
    direct_qa_path = destination/'direct_spherical_extension_qa.json'
    if direct_qa_path.exists():
        direct_qa = json.loads(direct_qa_path.read_text())
        current_config_hash = hashlib.sha256((destination/'config.yaml').read_bytes()).hexdigest()
        spherical = results.get('spherical', {})
        same_cache_path = Path(direct_qa.get('extension_cache_path', '')).resolve() == (destination/'spherical.npz').resolve()
        same_calculation_hash = (bool(spherical.get('calculation_hash'))
            and direct_qa.get('extension_calculation_hash') == spherical['calculation_hash'])
        manifest_path = destination/'manifest.json'
        final_hashes = (json.loads(manifest_path.read_text()).get('source_hashes', {})
                        if manifest_path.exists() else {})
        recorded_hashes = direct_qa.get('source_hashes', {})
        required_files = {'upc_deformation/observables.py', 'upc_deformation/source_amplitudes.py',
            'upc_deformation/forward.py', 'upc_deformation/decay.py',
            'upc_deformation/vendor/upc_forward.py', 'upc_deformation/observable_b_extension.py'}
        same_sources = required_files.issubset(recorded_hashes) and all(
            name in final_hashes and value == final_hashes[name]
            for name,value in recorded_hashes.items())
        check('direct spherical extension check: same saved configuration',
              direct_qa.get('config_sha256') == current_config_hash)
        check('direct spherical extension check: same loaded cache path',same_cache_path)
        check('direct spherical extension check: same loaded calculation hash',same_calculation_hash)
        check('direct spherical extension check: recorded physical sources match final manifest',same_sources)
        check('direct spherical extension check: arrays and scalars agree',
              direct_qa.get('pass') is True)
        report['direct_spherical_extension_evidence'] = {
            'path':str(direct_qa_path),'scope':direct_qa.get('scope'),
            'pass':direct_qa.get('pass'),
            'provenance_matches_current_run': bool(same_cache_path and same_calculation_hash
                and same_sources and direct_qa.get('config_sha256') == current_config_hash),
            'direct_execution_threads':direct_qa.get('direct_execution_threads'),
            'extension_execution_threads':direct_qa.get('extension_execution_threads'),
            'max_array_error_over_peak':max(row['max_error_over_peak']
                for row in direct_qa['array_checks'].values()),
        }
    report['qualification'] = ('Invariant checks passed; numerical warnings and unassessed convergence remain.'
        if report['pass'] else 'At least one saved-data invariant failed.')
    (destination/'qa.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', default='outputs/jpsi_observables_revision/main')
    args = parser.parse_args()
    report = verify(args.output)
    failed = [row['name'] for row in report['invariants'] if not row['pass']]
    print(json.dumps({'pass':report['pass'], 'invariant_checks':len(report['invariants']),
        'failures':failed, 'warnings':report['warnings'],
        'qa_path':str(Path(args.output).resolve()/'qa.json')},indent=2,ensure_ascii=False))
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
