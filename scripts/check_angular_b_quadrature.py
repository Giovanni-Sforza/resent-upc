#!/usr/bin/env python3
"""Independent spherical/skin b-quadrature diagnostic; never edits old caches.

Example:
  python scripts/check_angular_b_quadrature.py --threads 2
The default refines 12:2:60 fm to 12:1:60 fm with unchanged spatial/y physics.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
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

from upc_deformation.angular_observables import BIN_EDGES_GEV, METRIC_NAMES, orientation_angular_statistics
from upc_deformation.observables import calculate_case


CORE_FILES = ["upc_deformation/observables.py", "upc_deformation/source_amplitudes.py",
              "upc_deformation/forward.py", "upc_deformation/decay.py",
              "upc_deformation/vendor/upc_forward.py"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items() if not key.startswith('_')}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load_case(directory, name):
    result = json.loads((directory/f'{name}.json').read_text())
    with np.load(directory/f'{name}.npz') as archive:
        result.update({key: archive[key] for key in archive.files})
    return result


def save_case(directory, result):
    name = result['case']['name']
    target = directory/f'{name}.npz'
    temporary = target.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    temporary.replace(target)
    save_json(directory/f'{name}.json', {k: v for k, v in result.items() if not isinstance(v, np.ndarray)})


def metrics(result, bins=BIN_EDGES_GEV):
    return {branch: orientation_angular_statistics(result['orientation_coefficients'][:, -1, index],
                result['radii'], bins=bins, qmax=.17, deterministic=True)
            for index, branch in enumerate(['full', 'self'])}


def compare(original, refined, step):
    old, new = {k: metrics(v) for k, v in original.items()}, {k: metrics(v) for k, v in refined.items()}
    result = {'scope': 'Only deterministic spherical and skin=0.2 fm cases; final decay_by projection',
              'not_certified': ['deformation endpoint b convergence', 'spatial/y convergence',
                                'rigorous quadrature error bound', 'detector accuracy'],
              'metric_names': METRIC_NAMES, 'bin_edges_GeV': BIN_EDGES_GEV,
              'original_b_step_fm': 2., 'refined_b_step_fm': step, 'cases': {},
              'structural_contrast': {}, 'overlap_closure': {}}
    for name in original:
        result['cases'][name] = {branch: {
            'original_values': old[name][branch]['values'], 'refined_values': new[name][branch]['values'],
            'shift': new[name][branch]['values']-old[name][branch]['values'],
            'original_yield_fraction': old[name][branch]['yield_fraction'],
            'refined_yield_fraction': new[name][branch]['yield_fraction']}
            for branch in ['full', 'self']}
        indices = [int(np.argmin(abs(refined[name]['b_values']-b))) for b in original[name]['b_values']]
        if not np.allclose(refined[name]['b_values'][indices], original[name]['b_values'], rtol=0, atol=1e-10):
            raise ValueError('refined grid does not retain original b nodes')
        reference = original[name]['node_coefficients']
        residual = refined[name]['node_coefficients'][indices]-reference
        scale = float(np.max(abs(reference)))
        closure = float(np.max(abs(residual))/scale)
        result['overlap_closure'][name] = {'max_absolute_difference': float(np.max(abs(residual))),
                                         'max_difference_over_reference_peak': closure,
                                         'tolerance_over_peak': 3e-6, 'pass': closure < 3e-6}
        if closure >= 3e-6:
            raise ValueError(f'{name}: overlapping-node closure failed')
    for branch in ['full', 'self']:
        before = old['skin'][branch]['values']-old['spherical'][branch]['values']
        after = new['skin'][branch]['values']-new['spherical'][branch]['values']
        shift = after-before
        relative = np.divide(abs(shift), abs(after), out=np.full_like(shift, np.nan), where=abs(after)>1e-15)
        result['structural_contrast'][branch] = {'definition': 'skin minus spherical',
            'original': before, 'refined': after, 'shift': shift,
            'absolute_shift_over_refined_signal': relative}
    result['source_interaction_contrast'] = {
        key: result['structural_contrast']['full'][key]-result['structural_contrast']['self'][key]
        for key in ['original', 'refined', 'shift']}
    whole_old = {k: metrics(v, [[0., .17]]) for k, v in original.items()}
    whole_new = {k: metrics(v, [[0., .17]]) for k, v in refined.items()}
    whole = {'q_range_GeV': [0., .17], 'units': 'GeV', 'cases': {}, 'structural_contrast': {}}
    for name in original:
        whole['cases'][name] = {}
        for branch in ['full', 'self']:
            before = float(whole_old[name][branch]['values'][0, 3])
            after = float(whole_new[name][branch]['values'][0, 3])
            whole['cases'][name][branch] = {'original': before, 'refined': after, 'shift': after-before}
    for branch in ['full', 'self']:
        before = whole['cases']['skin'][branch]['original']-whole['cases']['spherical'][branch]['original']
        after = whole['cases']['skin'][branch]['refined']-whole['cases']['spherical'][branch]['refined']
        whole['structural_contrast'][branch] = {'definition': 'skin minus spherical',
            'original': before, 'refined': after, 'shift': after-before,
            'absolute_shift_over_refined_signal': abs((after-before)/after) if abs(after)>1e-15 else None}
    result['whole_disk_mean_q'] = whole
    return result


def write_report(destination, comparison, provenance):
    step = comparison['refined_b_step_fm']
    lines = ['# 球形与中子皮的 b 求积加密诊断', '',
        f'仅将 b=12–60 fm 的步长从 2 fm 改为 {step:g} fm；空间网格、快度求积、物理参数和四个动量分箱保持原设置。两例均为确定性单取向。', '',
        '表中 Δ 表示 skin−spherical，求积变化为加密值减原值。这里比较两级网格的变化，不把它当作严格误差上界。', '',
        '| 分支 | q 区间 [GeV] | 原 ΔC2 | 加密 ΔC2 | 变化/加密信号 | 原 Δmeanq [GeV] | 加密 Δmeanq [GeV] | 变化/加密信号 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for branch in ['full', 'self']:
        row = comparison['structural_contrast'][branch]
        for index, (low, high) in enumerate(BIN_EDGES_GEV):
            old, fine, relative = row['original'][index], row['refined'][index], row['absolute_shift_over_refined_signal'][index]
            lines.append(f'| {branch} | {low:.2f}–{high:.2f} | {old[0]:.7g} | {fine[0]:.7g} | {relative[0]:.2%} | {old[3]:.7g} | {fine[3]:.7g} | {relative[3]:.2%} |')
    full = comparison['structural_contrast']['full']['absolute_shift_over_refined_signal']
    lines += ['', f'球形到皮厚 0.2 fm 的 full 结构差，C2 的最大相对变化为 {np.max(full[:,0]):.2%}，meanq 的最大相对变化为 {np.max(full[:,3]):.2%}。各例绝对值、self 对照及所有分箱数值见 comparison.json。',
        '', '与原明区宽度支线对应的全区均值使用 0≤q≤0.17 GeV，不由四个环带的归一化均值拼接：', '',
        '| 分支 | 原全区 Δmeanq [GeV] | 加密全区 Δmeanq [GeV] | 求积变化 [GeV] | 变化/加密信号 |',
        '|---|---:|---:|---:|---:|']
    for branch in ['full', 'self']:
        row = comparison['whole_disk_mean_q']['structural_contrast'][branch]
        lines.append(f"| {branch} | {row['original']:.8g} | {row['refined']:.8g} | {row['shift']:.6g} | {row['absolute_shift_over_refined_signal']:.3%} |")
    lines += [
        '', '本诊断不能证明 β2 或 β3 端点的 b 积分收敛，也没有检验空间网格、快度积分或实验响应。若某分箱的物理结构差很小，其相对求积变化可以放大；应结合绝对变化阅读。',
        '', '重叠的旧 b 节点与新计算通过浮点闭合检查，支持把此处变化归因于求积网格改变。该闭合不等于求积收敛。',
        '', f"预计两例运行约 {provenance['estimated_two_case_seconds']:.0f} 秒，保存的原始两例计算共 {sum(provenance['cached_case_calculation_seconds'].values()):.1f} 秒；累计前向计算 {provenance['new_calculation_seconds']:.1f} 秒，本次命令新增计算 {provenance['this_invocation_new_calculation_seconds']:.1f} 秒（重放可复用缓存）。为保持本轮低成本，没有继续加密到 0.5 fm。",
        '', 'provenance.json 记录旧缓存、配置、代码哈希和运行线程；原 b12 缓存未写入。']
    (destination/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT/'outputs/jpsi_observables_revision/b12_20261007')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/jpsi_angular_revision/b12_20261008/b_quadrature_check')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--step', type=float, choices=[1., .5], default=1.)
    args = parser.parse_args()
    baseline, destination = args.baseline.resolve(), args.output.resolve()
    if baseline == destination or baseline in destination.parents:
        raise ValueError('diagnostic output must be separate from original cache directory')
    source_hashes = {key: digest(ROOT/key) for key in CORE_FILES}
    baseline_manifest = json.loads((baseline/'manifest.json').read_text())
    if any(source_hashes[key] != baseline_manifest['source_hashes'].get(key) for key in CORE_FILES):
        raise ValueError('current physical core does not match the baseline manifest')
    config = yaml.safe_load((baseline/'config.yaml').read_text())
    if (config['study']['b_min_fm'], config['study']['b_max_fm'], config['study']['b_nodes']) != (12., 60., 25):
        raise ValueError('expected original 12:2:60 fm baseline')
    original = {name: load_case(baseline, name) for name in ['spherical', 'skin']}
    if any(r['n_orientations'] != 1 or r['case']['beta2'] or r['case']['beta3'] for r in original.values()):
        raise ValueError('this diagnostic is restricted to deterministic spherical/skin cases')
    fine_config = copy.deepcopy(config)
    fine_config['study']['b_nodes'] = int(round(48/args.step))+1
    fine_config['study']['threads'] = args.threads
    fine_config['study']['cases'] = [copy.deepcopy(r['case']) for r in original.values()]
    signature = hashlib.sha256(json.dumps({'config': fine_config, 'core_source_hashes': source_hashes}, sort_keys=True).encode()).hexdigest()
    reference_seconds = json.loads((baseline/'direct_spherical_extension_qa.json').read_text())['direct_elapsed_seconds']
    estimate = reference_seconds * fine_config['study']['b_nodes']/25*2
    destination.mkdir(parents=True, exist_ok=True)
    previous_provenance = (json.loads((destination/'provenance.json').read_text())
                           if (destination/'provenance.json').exists() else {})
    if previous_provenance and previous_provenance.get('calculation_hash') != signature:
        raise ValueError('existing diagnostic provenance has a different calculation signature')
    config_path = destination/'config.yaml'
    if config_path.exists() and yaml.safe_load(config_path.read_text()) != fine_config:
        raise ValueError('existing diagnostic configuration differs; choose a different output directory')
    config_path.write_text(yaml.safe_dump(fine_config, sort_keys=False, allow_unicode=True))
    provenance = {'scope': 'spherical and skin only; no deformation convergence certification',
        'status': 'running', 'started_utc': previous_provenance.get('started_utc', datetime.now(timezone.utc).isoformat()),
        'baseline_directory': str(baseline), 'output_directory': str(destination),
        'baseline_config_sha256': digest(baseline/'config.yaml'), 'config_sha256': digest(config_path),
        'source_hashes': {**source_hashes, 'upc_deformation/angular_observables.py': digest(ROOT/'upc_deformation/angular_observables.py'),
                          'scripts/check_angular_b_quadrature.py': digest(__file__)},
        'calculation_hash': signature, 'execution_threads': args.threads,
        'estimated_two_case_seconds': estimate, 'baseline_inputs': {},
        'registered_change': {'b_step_fm': [2., args.step], 'b_nodes': [25, fine_config['study']['b_nodes']]},
        'replay_command': f'{sys.executable} {Path(__file__).resolve()} --baseline {baseline} --output {destination} --threads {args.threads} --step {args.step}'}
    if previous_provenance:
        provenance['initial_source_hashes'] = previous_provenance.get('initial_source_hashes', previous_provenance['source_hashes'])
    for name in original:
        provenance['baseline_inputs'][name] = {'calculation_hash': original[name].get('calculation_hash'),
            **{suffix+'_sha256': digest(baseline/f'{name}.{suffix}') for suffix in ['npz','json']}}
    save_json(destination/'provenance.json', provenance)
    torch.set_num_threads(args.threads)
    refined, elapsed = {}, 0.
    print(f'Estimated calculation time for two cases: {estimate:.1f}s; b step {args.step:g} fm', flush=True)
    with (destination/'run.log').open('a') as log_file:
        def log(message, **kwargs):
            print(message, flush=True)
            print(message, file=log_file, flush=True)
        for name, old in original.items():
            cache_json, cache_npz = destination/f'{name}.json', destination/f'{name}.npz'
            if cache_json.exists() and cache_npz.exists():
                result = load_case(destination, name)
                if result.get('calculation_hash') != signature:
                    raise ValueError('existing raw cache has different calculation provenance')
                log(f'Reusing diagnostic {name}')
            else:
                started = time.monotonic()
                result = calculate_case(fine_config, old['case'], log=log)
                elapsed += time.monotonic()-started
                result.update(calculation_hash=signature, execution_threads=args.threads,
                              diagnostic_scope='b quadrature refinement of spherical/skin only')
                save_case(destination, result)
            refined[name] = result
    comparison = compare(original, refined, args.step)
    if source_hashes != {key: digest(ROOT/key) for key in CORE_FILES}:
        raise ValueError('physical core changed during calculation')
    for name in original:
        if any(digest(baseline/f'{name}.{suffix}') != provenance['baseline_inputs'][name][suffix+'_sha256'] for suffix in ['npz','json']):
            raise ValueError('original cache changed during calculation')
    now = datetime.now(timezone.utc).isoformat()
    provenance.update(status='complete', finished_utc=previous_provenance.get('finished_utc', now) if elapsed==0 else now,
                      last_postprocess_utc=now,
                      new_calculation_seconds=previous_provenance.get('new_calculation_seconds', 0.)+elapsed,
                      this_invocation_new_calculation_seconds=elapsed,
                      cached_case_calculation_seconds={k: v['elapsed_seconds'] for k,v in refined.items()},
                      output_hashes={f'{k}.{ext}': digest(destination/f'{k}.{ext}') for k in refined for ext in ['npz','json']},
                      physical_core_unchanged=True, original_caches_unchanged=True)
    save_json(destination/'comparison.json', comparison)
    write_report(destination, comparison, provenance)
    provenance['product_hashes'] = {name: digest(destination/name) for name in ['comparison.json', 'report.md']}
    save_json(destination/'provenance.json', provenance)
    print(f'Completed diagnostic: {destination}', flush=True)


if __name__ == '__main__':
    main()
