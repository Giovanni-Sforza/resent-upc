#!/usr/bin/env python3
"""Analyze a completed CPU/CUDA precision campaign using the panel-d pixels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml

from upc_deformation.precision_analysis import PixelMomentProjector, moment_estimate, paired_ratio_difference
from upc_deformation.precision_figures import render_precision_figures

CASES = ('spherical', 'beta2', 'beta3', 'skin')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    return x


def write_json(path, x):
    path.write_text(json.dumps(clean(x), indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def arrays(x, prefix='', result=None):
    result = {} if result is None else result
    if isinstance(x, np.ndarray): result[prefix] = x
    elif isinstance(x, dict):
        for k, v in x.items(): arrays(v, f'{prefix}__{k}' if prefix else k, result)
    elif isinstance(x, (list, tuple)):
        for k, v in enumerate(x): arrays(v, f'{prefix}__{k}', result)
    return result


def load_run(run):
    manifest = json.loads((run/'production_manifest.json').read_text())
    if manifest.get('status') != 'completed':
        raise ValueError('analyze only a completed production; resume incomplete runs first')
    if sha(run/'effective_config.yaml') != manifest['effective_config_sha256']:
        raise ValueError('effective configuration differs from the completed production')
    cfg = yaml.safe_load((run/'effective_config.yaml').read_text())
    inputs = {'production_manifest.json': sha(run/'production_manifest.json'), 'effective_config.yaml': sha(run/'effective_config.yaml')}
    rows = {}
    for name in CASES:
        path = run/f'{name}.npz'
        if sha(path) != manifest['output_sha256'][path.name]:
            raise ValueError(f'production checksum mismatch: {path.name}')
        inputs[path.name] = sha(path)
        with np.load(path, allow_pickle=False) as z: rows[name] = {k: z[k] for k in z.files}
    base = rows['spherical']
    for name, row in rows.items():
        if str(row['fingerprint']) != manifest['fingerprint']:
            raise ValueError(f'{name}: production fingerprint mismatch')
        if len(row['orientation_coefficients']) != manifest['npairs'][name]:
            raise ValueError(f'{name}: production sample count mismatch')
        for key in ('axis', 'radii', 'b_steps_fm'):
            if not np.array_equal(row[key], base[key]):
                raise ValueError(f'{name}: mismatched {key}')
        c, levels = row['orientation_coefficients'], row['b_level_coefficients']
        if levels.shape != (len(base['b_steps_fm']), *c.shape) or not np.array_equal(c, levels[0]):
            raise ValueError(f'{name}: invalid fine/coarse coefficient identity')
        if name in ('spherical', 'skin') and len(c) != 1:
            raise ValueError('sphere and spherical skin must be deterministic')
    for key in ('rotation_bank_1', 'rotation_bank_2'):
        if not np.array_equal(rows['beta2'][key], rows['beta3'][key]):
            raise ValueError('deformation comparisons require a common orientation bank')
    return cfg, manifest, rows, inputs


def run_analysis(run, output=None, figures=True):
    cfg, manifest, source, inputs = load_run(run)
    n = len(source['beta2']['orientation_coefficients'])
    out = output or run/f'analysis_n{n}'
    if out.resolve() == run.resolve(): raise ValueError('analysis needs a separate subdirectory')
    out.mkdir(parents=True, exist_ok=True)
    acfg = cfg.get('analysis', {})
    qmax = float(cfg['study'].get('qmax', .17))
    fixed_q = np.asarray(acfg.get('fixed_pT_GeV', []), dtype=float)
    if not np.isfinite(fixed_q).all() or np.any((fixed_q < 0) | (fixed_q > qmax)):
        raise ValueError('fixed diagnostic pT points must lie inside the analyzed interval')
    q = np.unique(np.r_[np.linspace(0, qmax, int(acfg.get('q_points', 341))), fixed_q])
    angles = np.arange(int(acfg.get('angular_points', 720)))*2*np.pi/int(acfg.get('angular_points', 720))
    edges = np.asarray(acfg.get('edges_GeV', acfg.get('diagnostic_edges_GeV', [.03, .08, .13, .17])))
    identity = {'input_sha256': inputs, 'analysis_config': acfg, 'code_sha256': {
        p: sha(ROOT/p) for p in ('scripts/analyze_panel_d_precision.py', 'upc_deformation/precision_analysis.py',
                                'upc_deformation/precision_figures.py', 'upc_deformation/c2_convergence.py',
                                'upc_deformation/decay.py', 'upc_deformation/spectral_weights.py',
                                'upc_deformation/panel_d_ratio_figures.py')}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    previous = out/'analysis_manifest.json'
    if previous.exists() and json.loads(previous.read_text()).get('fingerprint') != fingerprint:
        raise ValueError('changed data/analysis require a new --output directory')
    write_json(previous, {**identity, 'fingerprint': fingerprint, 'status': 'running'})
    ref = source['spherical']
    projector = PixelMomentProjector(ref['radii'], ref['axis'], q, angles, edges)
    data = {'config': cfg, 'q': q, 'edges_GeV': edges, 'b_steps_fm': ref['b_steps_fm'], 'cases': {}, 'grid': {}}
    fine_moments = {}
    closure = {}
    for name in CASES:
        raw = source[name]
        closure[name] = projector.check_direct(raw['orientation_coefficients'])
        moment = projector.project(raw['orientation_coefficients'])
        fine_moments[name] = moment
        row = moment_estimate(moment)
        row['case'] = next(c for c in cfg['study']['cases'] if c['name'] == name)
        data['cases'][name] = {**row, **moment}
        data['grid'][name] = []
        for level, coeff in enumerate(raw['b_level_coefficients']):
            coarse = moment if level == 0 else projector.project(coeff)
            entry = moment_estimate(coarse, prefixes=())
            difference = paired_ratio_difference(coarse['orientation_C2_numerator'], coarse['orientation_denominator'],
                                                 moment['orientation_C2_numerator'], moment['orientation_denominator'])
            data['grid'][name].append({'b_step_fm': float(ref['b_steps_fm'][level]),
                                      'C2': entry['C2'], 'orientation_se': entry['orientation_se'],
                                      'coarse_minus_fine': difference})
        print(f'Analyzed {name}: {len(raw["orientation_coefficients"])} orientations', flush=True)
    for name in CASES:
        row = data['cases'][name]
        row['delta_C2'] = row['C2']-data['cases']['spherical']['C2']
        row['delta_C2_skin'] = row['C2']-data['cases']['skin']['C2']
        row['spectral']['relative_change'] = row['spectral']['value']/data['cases']['spherical']['spectral']['value']-1
        row['spectral']['relative_change_se'] = row['spectral']['orientation_se']/data['cases']['spherical']['spectral']['value']
    a, b = fine_moments['beta2'], fine_moments['beta3']
    data['paired_beta2_minus_beta3'] = paired_ratio_difference(a['orientation_C2_numerator'], a['orientation_denominator'], b['orientation_C2_numerator'], b['orientation_denominator'])

    with (out/'curves.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['case', 'pT_GeV', 'C2', 'orientation_se', 'delta_C2_spherical', 'delta_C2_skin', 'S2', 'S2_orientation_se'])
        for name, row in data['cases'].items():
            for i, p in enumerate(q): w.writerow([name, p, *[row[k][i] for k in ('C2', 'orientation_se', 'delta_C2', 'delta_C2_skin', 'S2', 'S2_orientation_se')]])
    with (out/'b_grid_comparison.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['case', 'b_step_fm', 'fine_step_fm', 'pT_GeV', 'C2', 'coarse_minus_fine_C2', 'paired_orientation_se', 'coarse_minus_fine_structure_response'])
        for name, levels in data['grid'].items():
            for j, row in enumerate(levels):
                d = row['coarse_minus_fine']; base_d = data['grid']['spherical'][j]['coarse_minus_fine']['value']
                for i, p in enumerate(q): w.writerow([name, row['b_step_fm'], ref['b_steps_fm'][0], p, row['C2'][i], d['value'][i], d['orientation_se'][i], d['value'][i]-base_d[i]])
    with (out/'fixed_pT_diagnostics.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['case', 'pT_GeV', 'C2', 'orientation_se', 'delta_C2_spherical', 'S2', 'S2_orientation_se'])
        for name, row in data['cases'].items():
            for p in fixed_q:
                i = int(np.searchsorted(q, p))
                w.writerow([name, p, *[row[k][i] for k in ('C2', 'orientation_se', 'delta_C2', 'S2', 'S2_orientation_se')]])
    with (out/'bin_metrics.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['case', 'bin', 'q_low', 'q_high', 'C2', 'orientation_se', 'delta_C2_spherical', 'yield_fraction_selected', 'H_over_M', 'H_over_M_se'])
        for name, row in data['cases'].items():
            for j, label in enumerate('LMH'):
                b = row['bins']; w.writerow([name, label, *edges[j:j+2], b['C2'][j], b['orientation_se'][j], b['C2'][j]-data['cases']['spherical']['bins']['C2'][j], b['fractions'][j], row['spectral']['value'][1], row['spectral']['orientation_se'][1]])
    with (out/'orientation_convergence.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['case', 'kind', 'label', 'n_orientations', 'pT_GeV', 'C2', 'orientation_se'])
        for name, row in data['cases'].items():
            entries = [('final', str(row['n_orientations']), row)] + [('nested', k, v) for k, v in row.get('nested', {}).items()] + [('half', str(i+1), h) for i, h in enumerate(row.get('halves', []))]
            for kind, label, entry in entries:
                for i, p in enumerate(q): w.writerow([name, kind, label, entry['n_orientations'], p, entry['C2'][i], entry['orientation_se'][i]])
    np.savez_compressed(out/'arrays.npz', **arrays(data))
    high = q >= .12
    summary = {'n_orientations': n, 'config': cfg, 'cases': {}, 'pixel_projection_closure': closure,
               'uncertainty_scope': 'Pointwise orientation integration SE only. Grid changes are paired diagnostics, not strict error bounds or detector errors.'}
    for name, row in data['cases'].items():
        summary['cases'][name] = {'case': row['case'], 'n_orientations': row['n_orientations'], 'bins': row['bins'],
                                 'spectral': row['spectral'], 'maximum_high_pT_orientation_se': float(row['orientation_se'][high].max()) if high.any() else None}
    write_json(out/'summary.json', summary)
    files = render_precision_figures(data, out) if figures else []
    lines = ['# Panel d 精度计算分析', '', f'形变各 {n} 对取向；b 步长为 {ref["b_steps_fm"][0]:g} fm。球形及中子皮在同一套数值设置下重算。', '',
             '观测量仍由 Cartesian panel d 定义。稀疏线性组合加速像素重建、圆周双线性插值与求和；每个案例均与原逐像素算法核对。', '',
             'C2 是原始角矩均值除以原始产额均值。阴影仅为取向积分点态1SE；相邻pT相关，嵌套前缀相关。两半样本互不重叠。', '',
             'b 网格比较复用相同取向和细网格节点，差值SE包含配对协方差。粗细差不等于细网格误差上界，尚需独立检查空间/快度/浮点精度。', '',
             'H/M、三区间C2与份额直接对像素求和；三区间边界见summary。没有实验显著性或自动挑选最优窗口。', '',
             '| 案例 | H/M | 取向SE | 相对球形变化 |', '|---|---:|---:|---:|']
    for name, row in data['cases'].items():
        s = row['spectral']; lines.append(f'| {name} | {s["value"][1]:.8f} | {s["orientation_se"][1]:.3g} | {100*s["relative_change"][1]:+.4f}% |')
    lines += ['', '## 图件', '']
    lines += [f'![{Path(p).stem}]({Path(p).name})' for p in files if p.endswith('.png')]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    for path, digest in inputs.items():
        if sha(run/path) != digest:
            raise RuntimeError('production changed during analysis; rerun into a fresh output directory')
    for path, digest in identity['code_sha256'].items():
        if sha(ROOT/path) != digest:
            raise RuntimeError('analysis source changed during execution; rerun into a fresh output directory')
    write_json(previous, {**identity, 'fingerprint': fingerprint, 'status': 'completed', 'files': files,
                         'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'analysis_manifest.json'}})
    print(out, flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--no-figures', action='store_true')
    a = p.parse_args()
    run_analysis(a.run.resolve(), a.output.resolve() if a.output else None, not a.no_figures)


if __name__ == '__main__': main()
