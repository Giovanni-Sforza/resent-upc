"""Tiny-grid direct/extended comparisons exercise the common endpoint weight."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from upc_deformation.observables import calculate_case
from upc_deformation.observable_b_extension import extend_case_b_window, _core_calculation_hash


@pytest.fixture(scope='module', params=['spherical', 'beta2'])
def archived_case(request):
    torch.set_num_threads(2)
    path = Path(__file__).resolve().parents[1]/'configs/jpsi_observables_revision.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['physics']['collision_parameters']['b'] = 20.
    cfg['physics']['grid_parameters'] = {
        'WS': {'N': 16, 'L': 24.}, 'T_prim': {'N': 16, 'L': 24.},
        'output': {'N': 96, 'L': 160.},
    }
    cfg['physics']['output_settings'].update(safe_padding_factor=2, target_size_q=51)
    cfg['physics']['incoherent_quadrature'].update(n_r=24, n_phi=32)
    cfg['study'].update(b_min_fm=20., b_max_fm=60., b_nodes=21, fixed_b_fm=20.,
                        n_orientations=2, n_y=3, n_radial=81, n_phi=32)
    # Both deterministic sphere and orientation-dependent deformation must
    # preserve the endpoint weights and the first-half orientation cache.
    case = next(c for c in cfg['study']['cases'] if c['name'] == request.param)
    old = calculate_case(cfg, case, lambda *args, **kwargs: None)
    old.update(cache_config=deepcopy(cfg), cache_source='tiny-grid-test-cache.npz',
               calculation_hash=_core_calculation_hash())
    new = deepcopy(cfg)
    new['physics']['collision_parameters']['b'] = 12.
    new['study'].update(b_min_fm=12., b_nodes=25, fixed_b_fm=12.)
    return old, new, case


def test_extension_equals_full_replay_with_no_double_counted_endpoint(archived_case):
    old, config, case = archived_case
    original_nodes = old['node_coefficients'].copy()
    extended = extend_case_b_window(config, case, old, lambda *a, **kw: None)
    direct = calculate_case(config, case, lambda *a, **kw: None)
    for key in ('coefficients', 'node_coefficients', 'half_node_coefficients',
                'orientation_coefficients', 'coarse_b_coefficients', 'images',
                'normalized_images', 'yields', 'background_coefficients',
                'incoherent_probabilities', 'total_coefficients'):
        scale = float(np.abs(direct[key]).max())
        np.testing.assert_allclose(extended[key], direct[key], rtol=3e-7, atol=1e-9*scale,
                                   err_msg=key)
    np.testing.assert_array_equal(old['node_coefficients'], original_nodes)
    np.testing.assert_array_equal(extended['b_values'], np.arange(12.,61.,2.))
    np.testing.assert_allclose(extended['b_weights'].sum(), np.pi*(60**2-12**2), rtol=1e-14)
    reuse = extended['b_window_reuse']
    assert reuse['total_source_calls'] == 5*extended['n_orientations']
    assert reuse['full_replay_source_calls_avoided'] == 20*extended['n_orientations']
    assert reuse['b20_closure']['mean_coefficients']['max_error_over_archived_peak'] < 1e-6


@pytest.mark.parametrize('change', ['seed', 'n_phi', 'physics', 'bank', 'hash'])
def test_incompatible_cache_is_rejected_before_forward(archived_case, monkeypatch, change):
    old, config, case = archived_case
    old, config = deepcopy(old), deepcopy(config)
    if change == 'seed': config['study']['orientation_seed'] += 1
    if change == 'n_phi': config['study']['n_phi'] *= 2
    if change == 'physics': config['physics']['physical_constants']['Mjpsi'] += .1
    if change == 'bank': old['rotation_bank_1'][0,0,0] += .01
    if change == 'hash': old['calculation_hash'] = 'stale'
    def forbidden(*args, **kwargs):
        pytest.fail('Incompatible cache reached a forward calculation')
    monkeypatch.setattr('upc_deformation.observable_b_extension.source_stokes', forbidden)
    with pytest.raises(ValueError, match='Cannot reuse b-window cache'):
        extend_case_b_window(config, case, old, lambda *a, **kw: None)
