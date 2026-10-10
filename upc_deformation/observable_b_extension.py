"""Extend the archived 20--60 fm bridge to 12--60 fm without replaying its tail.

Only b=12,14,16,18,20 fm are evaluated.  The common 20 fm node supplies one
half-endpoint weight to each trapezoid, so adding the two integrals is exact.
The extension operates on unnormalized Stokes/intensity coefficients, including
each individual orientation; it never averages normalized distributions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .decay import b_quadrature, decay_axis_image, decay_coefficients, normalize_in_disk
from .forward import UPCForwardModel
from .observables import STAGES, case_physics, coefficient_moments
from .source_amplitudes import source_stokes


_CORE_FILES = (
    'upc_deformation/observables.py', 'upc_deformation/source_amplitudes.py',
    'upc_deformation/forward.py', 'upc_deformation/decay.py',
    'upc_deformation/vendor/upc_forward.py',
)


def _core_calculation_hash():
    root = Path(__file__).resolve().parents[1]
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in _CORE_FILES}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def _without_initial_b(physics):
    answer = copy.deepcopy(physics)
    answer['collision_parameters'].pop('b')
    return answer


def _require(condition, message):
    if not bool(condition):
        raise ValueError('Cannot reuse b-window cache: '+message)


def _closure(current, archived, *, rtol=5e-5, peak_atol=2e-6, name='array'):
    current, archived = np.asarray(current), np.asarray(archived)
    _require(current.shape == archived.shape, f'{name} shape differs')
    _require(np.isfinite(current).all() and np.isfinite(archived).all(), f'{name} is nonfinite')
    peak = max(float(np.abs(archived).max()), 1e-30)
    error = float(np.abs(current-archived).max())
    _require(np.allclose(current, archived, rtol=rtol, atol=peak*peak_atol),
             f'{name} closure error {error/peak:.4g} times its peak')
    return {'max_absolute_error': error, 'max_error_over_archived_peak': error/peak,
            'relative_tolerance': rtol, 'absolute_tolerance_over_peak': peak_atol}


def _validate_reuse(config, case, old_result, physics, density, model, b, y, yw, axis, r, banks):
    _require('cache_config' in old_result and 'cache_source' in old_result,
             'supply cache_config (old config dict) and cache_source (old npz path)')
    old_cfg, cfg = old_result['cache_config']['study'], config['study']
    _require(old_result.get('calculation_hash') == _core_calculation_hash(),
             'archived calculation hash differs from the unchanged physical implementation')
    for key in ('name', 'beta2', 'beta3', 'skin_fm'):
        _require(old_result['case'].get(key) == case.get(key), f'case {key} differs')
    expected_old_physics, expected_old_density = case_physics(old_result['cache_config'], old_result['case'])
    _require(old_result['physics'] == expected_old_physics,
             'recorded effective physics does not match the archived config/case')
    _require(_without_initial_b(physics) == _without_initial_b(old_result['physics']),
             'effective physics differs beyond collision_parameters.b')
    _require(density == old_result['density'] == expected_old_density,
             'nuclear density/RMS definitions differ')
    for key in ('longitudinal_units', 'ymax', 'n_y', 'n_radial', 'n_phi', 'orientation_seed'):
        _require(old_cfg.get(key) == cfg.get(key), f'study.{key} differs')
    _require(old_cfg['b_min_fm'] == 20 and old_cfg['b_max_fm'] == 60
             and old_cfg['fixed_b_fm'] == 20 and old_cfg['b_nodes'] == 21,
             'archived window must be fixed b=20 and nodes 20:2:60 fm')
    _require(cfg['b_min_fm'] == 12 and cfg['b_max_fm'] == 60
             and cfg['fixed_b_fm'] == 12 and cfg['b_nodes'] == 25,
             'new window must be fixed b=12 and nodes 12:2:60 fm')
    n = len(banks[0])
    expected_old_n = old_cfg['n_orientations'] if case['beta2'] or case['beta3'] else 1
    _require(n == old_result['n_orientations'] == expected_old_n, 'orientation counts differ')
    _require(old_result['stage_names'] == STAGES, 'archived stage meanings differ')
    exact_arrays = {
        'axis': axis.numpy(), 'radii': r.numpy(), 'b_values': b.numpy()[4:],
        'y_values': y, 'y_weights': yw,
        'rotation_bank_1': banks[0].numpy(), 'rotation_bank_2': banks[1].numpy(),
    }
    for key, expected in exact_arrays.items():
        _require(np.array_equal(old_result[key], expected), f'{key} differs')
    old_weights = b_quadrature(b[4:]).numpy()
    _closure(old_result['b_weights'], old_weights, rtol=1e-12, peak_atol=1e-14, name='old b weights')
    expected_shapes = {
        'node_coefficients': (21, len(y), 2, 3, len(r)),
        'half_node_coefficients': (21, len(y), 2, 3, len(r)),
        'orientation_coefficients': (n, 3, 2, 3, len(r)),
        'coefficients': (3, 2, 3, len(r)),
        'images': (4, 2, len(axis), len(axis)),
        'incoherent_probabilities': (21, len(y)),
    }
    for key, shape in expected_shapes.items():
        _require(old_result[key].shape == shape and np.isfinite(old_result[key]).all(),
                 f'{key} has incompatible shape or nonfinite values')
    yi = int(np.argmin(abs(y)))
    coeff, node = old_result['coefficients'], old_result['node_coefficients']
    contractions = {
        'orientation mean': old_result['orientation_coefficients'].mean(0),
        'fixed node': node[0, yi],
        'b integral': np.einsum('b,bscr->scr', old_weights, node[:, yi]),
        'b,y integral': np.einsum('b,y,byscr->scr', old_weights, yw, node),
    }
    targets = [coeff, coeff[0], coeff[1], coeff[2]]
    for (name, current), archived in zip(contractions.items(), targets):
        _closure(current, archived, rtol=5e-10, peak_atol=1e-12, name='archived '+name)
    if n == 1:
        _closure(old_result['half_node_coefficients'], node, rtol=1e-12,
                 peak_atol=1e-14, name='single-orientation half cache')
    _require(model._output_L/2 > cfg['b_max_fm']+16, 'field domain lacks nuclear support')


def extend_case_b_window(config, case, old_result, log=print):
    """Extend one verified 20--60 fm case, returning the standard case schema.

    ``old_result`` is the loaded old npz + json, augmented with ``cache_config``
    (its saved config.yaml dict) and ``cache_source`` (its npz provenance path).
    The caller saves outputs and assigns the new runner's calculation hash.
    No existing files, model constants, or physical prescriptions are changed.

    The old full per-orientation 20--60 integrals are retained.  Only the new
    12--20 leg is computed, including its 20 fm endpoint for closure and the
    additional half weight.  At the seam, the node cache stores the weighted
    mean of the two floating-point evaluations.  This exactly represents the
    two trapezoids, rather than silently discarding numerical seam differences.
    """
    cfg = config['study']
    physics, density = case_physics(config, case)
    model = UPCForwardModel(physics, cfg['longitudinal_units']).eval()
    b = torch.linspace(cfg['b_min_fm'], cfg['b_max_fm'], cfg['b_nodes'], dtype=torch.float64)
    bw = b_quadrature(b)
    leg_b = b[:5]
    leg_w = b_quadrature(leg_b).numpy()
    old_w = b_quadrature(b[4:]).numpy()
    y, yw = np.polynomial.legendre.leggauss(cfg['n_y'])
    y, yw = y*cfg['ymax'], yw*cfg['ymax']
    _require(cfg['n_y'] % 2 == 1, 'odd rapidity quadrature is required for the y=0 fixed stage')
    yi = int(np.argmin(abs(y)))
    axis = model._p_axis_q
    r = torch.linspace(0, float(axis.abs().max())*math.sqrt(2)*(1+1e-6), cfg['n_radial'])
    n = cfg['n_orientations'] if case['beta2'] or case['beta3'] else 1
    banks = []
    for seed in (cfg['orientation_seed'], cfg['orientation_seed']+1):
        generator = torch.Generator(device='cpu').manual_seed(seed)
        banks.append(model._sample_uniform_rotations(1, n, torch.device('cpu'), torch.float32, generator)[0]
                     if n > 1 else torch.eye(3)[None])
    _validate_reuse(config, case, old_result, physics, density, model, b, y, yw, axis, r, banks)
    started = time.monotonic()
    shape = (5, len(y), 2, 3, len(r))
    leg_sums = np.zeros(shape, dtype=np.float64)
    leg_half = np.zeros_like(leg_sums)
    leg_inc = np.zeros((5, len(y)), dtype=np.float64)
    fixed_maps = np.zeros((2, len(axis), len(axis)), dtype=np.float64)
    seam_maps = np.zeros_like(fixed_maps)
    per_orientation = np.empty_like(old_result['orientation_coefficients'])
    seam_orientation_errors = []
    center = len(axis)//2
    with torch.inference_mode():
        for oi in range(n):
            node = np.empty(shape, dtype=np.float64)
            for bi, bvalue in enumerate(leg_b):
                full, self_map, pfull, inc = source_stokes(
                    model, torch.tensor(y[None], dtype=torch.float32), [float(bvalue)],
                    banks[0][oi:oi+1], banks[1][oi:oi+1], return_incoherent=True)
                maps = torch.stack((full[0,0], self_map[0,0]), dim=1)
                node[bi] = decay_coefficients(maps, pfull, r, cfg['n_phi']).double().numpy()
                leg_inc[bi] += inc[0,0,:,center,center].double().numpy()*np.pi/model.b_slope_incoh/n
                if bi in (0, 4):
                    projection, _ = model.project_full_grid_intensity_to_output(maps[yi])
                    if bi == 0:
                        fixed_maps += projection[:,0].double().numpy()/n
                    else:
                        seam_maps += projection[:,0].double().numpy()/n
                del full, self_map, maps, inc
            seam_orientation_errors.append(_closure(node[-1,yi],
                old_result['orientation_coefficients'][oi,0], name=f'b20 orientation {oi} fixed coefficients'))
            leg_sums += node/n
            if oi < n//2:
                leg_half += node/max(1, n//2)
            per_orientation[oi,0] = node[0,yi]
            per_orientation[oi,1] = old_result['orientation_coefficients'][oi,1] + np.einsum(
                'b,bscr->scr', leg_w, node[:,yi])
            per_orientation[oi,2] = old_result['orientation_coefficients'][oi,2] + np.einsum(
                'b,y,byscr->scr', leg_w, yw, node)
            log(f'{case["name"]}: extension orientations {oi+1}/{n}, '
                f'{time.monotonic()-started:.1f}s', flush=True)
    if n == 1:
        leg_half = leg_sums.copy()
    closure = {
        'mean_coefficients': _closure(leg_sums[-1], old_result['node_coefficients'][0], name='b20 mean coefficients'),
        'half_coefficients': _closure(leg_half[-1], old_result['half_node_coefficients'][0], name='b20 half coefficients'),
        'fixed_images': _closure(seam_maps, old_result['images'][0], name='b20 fixed images'),
        'empirical_probability': _closure(leg_inc[-1], old_result['incoherent_probabilities'][0], name='b20 empirical yield'),
        'per_orientation_fixed_coefficients': seam_orientation_errors,
    }
    # At b20 the two half-endpoint weights combine to the full interior weight.
    # Keep their weighted values so cached-node contractions also close exactly
    # when a different thread count changes a float32 evaluation by roundoff.
    def join(left, right):
        combined = np.concatenate((left[:-1], right.copy()), axis=0)
        combined[4] = (leg_w[-1]*left[-1]+old_w[0]*right[0])/float(bw[4])
        return combined
    sums = join(leg_sums, old_result['node_coefficients'])
    half_sums = join(leg_half, old_result['half_node_coefficients'])
    incoherent = join(leg_inc, old_result['incoherent_probabilities'])
    coeff = per_orientation.mean(0)
    _closure(coeff[-1], np.einsum('b,y,byscr->scr', bw.numpy(), yw, sums),
             rtol=5e-10, peak_atol=1e-12, name='extended node/orientation integration')
    images = np.empty((4,2,len(axis),len(axis)))
    images[0] = fixed_maps
    for stage in range(3):
        images[stage+1] = decay_axis_image(torch.from_numpy(coeff[stage]), r.double(), axis.double()).numpy()
    norm, yields = normalize_in_disk(torch.from_numpy(images), axis.double(), cfg['qmax'])
    slope = physics['output_settings']['b_slope_incoh']
    background = np.zeros((3,len(r)))
    background[0] = (np.einsum('b,y,by->', bw.numpy(), yw, incoherent)*slope/np.pi
                     *np.exp(-slope*r.numpy()**2))
    coarse = np.arange(0,len(b),2)
    coarse_coeff = np.einsum('b,y,byscr->scr', b_quadrature(b[coarse]).numpy(), yw, sums[coarse])
    endpoint_fraction = []
    total, _ = coefficient_moments(coeff[-1,0], r.numpy(), cfg['qmax'], cfg['q0'])
    for sl in (slice(0,6), slice(len(b)-6,len(b))):
        part = np.einsum('b,y,byscr->scr', b_quadrature(b[sl]).numpy(), yw, sums[sl])
        partial, _ = coefficient_moments(part[0], r.numpy(), cfg['qmax'], cfg['q0'])
        endpoint_fraction.append(float(partial/total))
    checks = {
        'min_full_intensity': float(images[:,0].min()), 'min_self_intensity': float(images[:,1].min()),
        'fixed_axis_vs_decay_yield_relative': float(yields[1,0]/yields[0,0]-1),
        'b_weights_sum_fm2': float(bw.sum()), 'y_weights_sum': float(yw.sum()),
        'edge_b_band_yield_fractions': endpoint_fraction,
    }
    reuse = {
        'cache_source': str(old_result['cache_source']),
        'source_calculation_hash': old_result['calculation_hash'],
        'source_b_window_fm': [20.,60.], 'extended_b_window_fm': [12.,60.],
        'evaluated_b_nodes_fm': leg_b.tolist(), 'new_b_nodes_fm': leg_b[:-1].tolist(),
        'recomputed_seam_b_fm': 20., 'retained_old_b_nodes': 21,
        'source_calls_per_orientation': 5, 'new_nodes_per_orientation': 4,
        'seam_calls_per_orientation': 1, 'total_source_calls': 5*n,
        'full_replay_source_calls_avoided': 20*n,
        'seam_policy': 'Sum left/right trapezoid half weights; node cache stores their weighted mean.',
        'b20_closure': closure,
        'initial_b_dependence': 'No b-dependent initialization buffers; fixed-b source API sets and restores b for every evaluation.',
    }
    return {
        'case': copy.deepcopy(case), 'density': density, 'physics': physics, 'stage_names': STAGES,
        'axis': axis.numpy(), 'radii': r.numpy(), 'b_values': b.numpy(), 'b_weights': bw.numpy(),
        'y_values': y, 'y_weights': yw, 'coefficients': coeff, 'node_coefficients': sums,
        'half_node_coefficients': half_sums, 'orientation_coefficients': per_orientation,
        'coarse_b_coefficients': coarse_coeff, 'images': images, 'normalized_images': norm.numpy(),
        'yields': yields.numpy(), 'incoherent_probabilities': incoherent,
        'background_coefficients': background, 'total_coefficients': coeff[-1,0]+background,
        'rotation_bank_1': banks[0].numpy(), 'rotation_bank_2': banks[1].numpy(),
        'checks': checks, 'n_orientations': n, 'elapsed_seconds': time.monotonic()-started,
        'execution_threads': torch.get_num_threads(), 'b_window_reuse': reuse,
    }
