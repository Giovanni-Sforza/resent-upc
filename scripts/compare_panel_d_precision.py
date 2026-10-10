#!/usr/bin/env python3
"""Paired numerical diagnostics between two completed panel-d precision runs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.analyze_panel_d_precision import CASES, load_run, sha, write_json
from upc_deformation.c2_convergence import ratio_statistics
from upc_deformation.precision_analysis import PixelMomentProjector, paired_ratio_difference


def physics_signature(config):
    """Exclude declared integration/grid controls, retaining physical constants."""
    physics = deepcopy(config['physics'])
    physics.pop('grid_parameters', None)
    for key in ('p_limit_q', 'target_size_q', 'safe_padding_factor'):
        physics.get('output_settings', {}).pop(key, None)
    # These are nominal defaults; actual b and y integration is in study.
    for key in ('b', 'y'):
        physics.get('collision_parameters', {}).pop(key, None)
    cases = {c['name']: {k: v for k, v in c.items() if k not in ('label', 'role')}
             for c in config['study']['cases'] if c['name'] in CASES}
    if set(cases) != set(CASES):
        raise ValueError('all four physical cases must be declared')
    return {'physics': physics, 'cases': cases,
            'longitudinal_units': config['study'].get('longitudinal_units', 'physical')}


def validate_pair(reference_config, candidate_config, reference, candidate):
    if physics_signature(reference_config) != physics_signature(candidate_config):
        raise ValueError('reference and candidate physical parameters differ')
    for name in CASES:
        n = len(reference[name]['orientation_coefficients'])
        if len(candidate[name]['orientation_coefficients']) != n:
            raise ValueError(f'{name}: paired comparison requires equal orientation counts')
        for key in ('rotation_bank_1', 'rotation_bank_2'):
            if (reference[name][key].shape != (n, 3, 3)
                    or not np.array_equal(reference[name][key], candidate[name][key])):
                raise ValueError(f'{name}: exact rotation bank and ordering must match')
    qmax = float(reference_config['study'].get('qmax', .17))
    if float(candidate_config['study'].get('qmax', .17)) < qmax:
        raise ValueError('candidate qmax must cover the reference comparison window')


def compare_loaded(reference_config, candidate_config, reference, candidate):
    """Compare each run's own Cartesian pixels on common reference circles."""
    validate_pair(reference_config, candidate_config, reference, candidate)
    acfg = reference_config.get('analysis', {})
    q = np.linspace(0, float(reference_config['study'].get('qmax', .17)), int(acfg.get('q_points', 341)))
    nphi = int(acfg.get('angular_points', 720))
    phi = np.arange(nphi)*2*np.pi/nphi
    edges = np.asarray(acfg.get('edges_GeV', acfg.get('diagnostic_edges_GeV', [.03, .08, .13, .17])))
    moments, estimates, closure = {}, {}, {}
    for tag, source in (('reference', reference), ('candidate', candidate)):
        first = source['spherical']
        projector = PixelMomentProjector(first['radii'], first['axis'], q, phi, edges)
        moments[tag], estimates[tag], closure[tag] = {}, {}, {}
        for name in CASES:
            coefficients = source[name]['orientation_coefficients']
            closure[tag][name] = projector.check_direct(coefficients)
            m = projector.project(coefficients)
            moments[tag][name] = m
            estimates[tag][name] = {
                'C2': ratio_statistics(m['orientation_C2_numerator'], m['orientation_denominator']),
                'ratios': ratio_statistics(m['bin_orientation_denominator'][:, 1:],
                                            m['bin_orientation_denominator'][:, :-1])}
    result = {'q': q, 'edges_GeV': edges, 'cases': {}, 'projection_closure': closure}
    for name in CASES:
        r, c = moments['reference'][name], moments['candidate'][name]
        delta = paired_ratio_difference(c['orientation_C2_numerator'], c['orientation_denominator'],
                                       r['orientation_C2_numerator'], r['orientation_denominator'])
        rd, cd = r['bin_orientation_denominator'], c['bin_orientation_denominator']
        spectral = paired_ratio_difference(cd[:, 1:], cd[:, :-1], rd[:, 1:], rd[:, :-1])
        # Work with both adjacent ratios internally, avoiding a scalar covariance
        # special case; the delivered observable is the second ratio H/M.
        sphere_r = estimates['reference']['spherical']['ratios']['value']
        sphere_c = estimates['candidate']['spherical']['ratios']['value']
        response = paired_ratio_difference(cd[:, 1:]/sphere_c, cd[:, :-1],
                                           rd[:, 1:]/sphere_r, rd[:, :-1])
        er, ec = estimates['reference'][name], estimates['candidate'][name]
        row = {'n_orientations': len(rd),
               'reference_C2': er['C2']['value'], 'candidate_C2': ec['C2']['value'],
               'C2_shift': delta['value'], 'C2_shift_paired_se': delta['orientation_se'],
               'C2_shift_covariance': delta['covariance'],
               'HM_reference': float(er['ratios']['value'][1]),
               'HM_candidate': float(ec['ratios']['value'][1]),
               'HM_shift': float(spectral['value'][1]),
               'HM_shift_paired_se': float(spectral['orientation_se'][1]),
               'HM_response_reference': float(er['ratios']['value'][1]/sphere_r[1]-1),
               'HM_response_candidate': float(ec['ratios']['value'][1]/sphere_c[1]-1),
               'HM_response_shift': float(response['value'][1]),
               'HM_response_shift_paired_se': float(response['orientation_se'][1])}
        for base, label in (('spherical', 'sphere'), ('skin', 'skin')):
            reference_response = er['C2']['value']-estimates['reference'][base]['C2']['value']
            candidate_response = ec['C2']['value']-estimates['candidate'][base]['C2']['value']
            row[f'{label}_response_reference'] = reference_response
            row[f'{label}_response_candidate'] = candidate_response
            row[f'{label}_response_shift'] = candidate_response-reference_response
            # Both reference structures have exactly one deterministic sample.
            row[f'{label}_response_shift_paired_se'] = delta['orientation_se'].copy()
        result['cases'][name] = row
    return result


def run_comparison(reference_run, candidate_run, output):
    reference_run, candidate_run, output = [Path(p).resolve() for p in (reference_run, candidate_run, output)]
    if output in (reference_run, candidate_run):
        raise ValueError('comparison output must not overwrite a production directory')
    rcfg, rmanifest, reference, rinputs = load_run(reference_run)
    ccfg, cmanifest, candidate, cinputs = load_run(candidate_run)
    files = ('scripts/compare_panel_d_precision.py', 'scripts/analyze_panel_d_precision.py',
             'upc_deformation/precision_analysis.py', 'upc_deformation/c2_convergence.py',
             'upc_deformation/decay.py', 'upc_deformation/spectral_weights.py')
    identity = {'reference_run': str(reference_run), 'candidate_run': str(candidate_run),
                'input_sha256': {'reference': rinputs, 'candidate': cinputs},
                'production_fingerprints': {'reference': rmanifest['fingerprint'], 'candidate': cmanifest['fingerprint']},
                'code_sha256': {p: sha(ROOT/p) for p in files}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'comparison_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get('fingerprint') != fingerprint:
        raise ValueError('changed comparison inputs/code require a new output directory')
    write_json(manifest_path, {**identity, 'fingerprint': fingerprint, 'status': 'running'})
    result = compare_loaded(rcfg, ccfg, reference, candidate)
    columns = ('reference_C2', 'candidate_C2', 'C2_shift', 'C2_shift_paired_se',
               'sphere_response_reference', 'sphere_response_candidate', 'sphere_response_shift',
               'sphere_response_shift_paired_se', 'skin_response_reference', 'skin_response_candidate',
               'skin_response_shift', 'skin_response_shift_paired_se')
    with (output/'C2_comparison.csv').open('w') as handle:
        writer = csv.writer(handle); writer.writerow(('case', 'pT_GeV', *columns))
        for name, row in result['cases'].items():
            for i, q in enumerate(result['q']): writer.writerow((name, q, *[row[k][i] for k in columns]))
    hkeys = ('HM_reference', 'HM_candidate', 'HM_shift', 'HM_shift_paired_se',
             'HM_response_reference', 'HM_response_candidate', 'HM_response_shift', 'HM_response_shift_paired_se')
    with (output/'HM_comparison.csv').open('w') as handle:
        writer = csv.writer(handle); writer.writerow(('case', *hkeys))
        for name, row in result['cases'].items(): writer.writerow((name, *[row[k] for k in hkeys]))
    high = result['q'] >= .12
    summary = {'direction': 'candidate minus reference', 'high_pT_threshold_GeV': .12,
               'common_q_GeV': result['q'], 'edges_GeV': result['edges_GeV'],
               'reference_config': rcfg, 'candidate_config': ccfg,
               'projection_closure': result['projection_closure'], 'cases': {},
               'interpretation': 'Paired numerical diagnostics only; differences are not error upper bounds, detector uncertainties, or an automatic convergence decision.'}
    stored = {'q': result['q'], 'edges_GeV': result['edges_GeV']}
    for name, row in result['cases'].items():
        summary['cases'][name] = {'n_orientations': row['n_orientations'], **{k: row[k] for k in hkeys},
            'max_absolute_high_pT': {k: float(np.max(np.abs(row[k][high]))) if high.any() else None
                                    for k in ('C2_shift', 'sphere_response_shift', 'skin_response_shift')},
            'maximum_high_pT_paired_se': float(np.max(row['C2_shift_paired_se'][high])) if high.any() else None}
        stored.update({f'{name}__{key}': value for key, value in row.items() if isinstance(value, np.ndarray)})
    np.savez_compressed(output/'arrays.npz', **stored)
    write_json(output/'summary.json', summary)
    lines = ['# Panel d 数值精度变体比较', '', f'参照：`{reference_run}`', f'候选：`{candidate_run}`', '',
        '所有差值定义为候选减参照。四种结构的物理参数保持相同；每种结构的取向数目、顺序和两核旋转矩阵逐项核对。',
        '各自用自己的axis/radii重建Cartesian panel d，再在参照配置的共同pT/角度网格求角矩。区间H/M直接来自各自像素环带求和。',
        '结构响应分别减去各数值设置下重算的球形或中子皮参照。配对SE保留同一取向在两个设置中的协方差；确定性球形/皮没有取向SE。',
        '这些是数值诊断，差值不是误差上界，也不构成自动物理收敛判定或实验显著性。若b/y积分范围改变，差值也包括积分选择的变化。', '',
        '公式：R=mean(N)/mean(D)，u_i=(N_i−R D_i)/mean(D)；差值SE=std(u_candidate−u_reference, ddof=1)/sqrt(n)。',
        'H/M相对球形响应使用各自的球形H/M；其配对影响函数为u_candidate/R_sph,candidate−u_reference/R_sph,reference。', '',
        '| 案例 | H/M参照 | H/M候选 | H/M差值 ± 配对SE | 相对球形响应变化（百分点） |', '|---|---:|---:|---:|---:|']
    for name, row in result['cases'].items():
        lines.append(f'| {name} | {row["HM_reference"]:.9g} | {row["HM_candidate"]:.9g} | {row["HM_shift"]:+.4g} ± {row["HM_shift_paired_se"]:.3g} | {100*row["HM_response_shift"]:+.4g} ± {100*row["HM_response_shift_paired_se"]:.3g} |')
    lines += ['', '全pT结果见C2_comparison.csv；高pT（≥0.12 GeV）最大绝对差见summary.json。arrays.npz保存差值协方差。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    for tag, run, inputs in (('reference', reference_run, rinputs), ('candidate', candidate_run, cinputs)):
        if any(sha(run/p) != value for p, value in inputs.items()):
            raise ValueError(f'{tag} inputs changed during comparison')
    if any(sha(ROOT/p) != value for p, value in identity['code_sha256'].items()):
        raise ValueError('comparison code changed during execution')
    write_json(manifest_path, {**identity, 'fingerprint': fingerprint, 'status': 'completed',
        'output_sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file() and p != manifest_path}})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(run_comparison(args.reference, args.candidate, args.output))


if __name__ == '__main__':
    main()
