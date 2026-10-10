from copy import deepcopy

import numpy as np
import pytest

from scripts.compare_panel_d_precision import compare_loaded, validate_pair


def example():
    names = ('spherical', 'beta2', 'beta3', 'skin')
    config = {'physics': {'collision_parameters': {'gamma': 100, 'b': 12, 'y': 0},
                          'grid_parameters': {'WS': {'N': 16}},
                          'output_settings': {'target_size_q': 33, 'p_limit_q': .2}},
              'study': {'qmax': .17, 'longitudinal_units': 'physical',
                        'cases': [dict(name=n, beta2=.3 if n == 'beta2' else 0,
                                       beta3=.2 if n == 'beta3' else 0,
                                       skin_fm=.2 if n == 'skin' else 0) for n in names]},
              'analysis': {'q_points': 15, 'angular_points': 32}}
    radii, axis = np.linspace(0, .3, 51), np.linspace(-.2, .2, 33)
    rows = {}
    for name in names:
        count = 4 if name in ('beta2', 'beta3') else 1
        rotations = np.tile(np.eye(3), (count, 1, 1))
        if count > 1:
            rotations[1] = np.diag([-1, -1, 1])
        coef = np.tile(np.stack((np.ones(len(radii)), .1*np.ones(len(radii)),
                                np.zeros(len(radii)))), (count, 1, 1))
        rows[name] = {'radii': radii, 'axis': axis, 'orientation_coefficients': coef,
                      'rotation_bank_1': rotations, 'rotation_bank_2': rotations.copy()}
    return config, rows


def test_pairing_rejects_physics_change_reordered_samples_and_shorter_window():
    cfg, rows = example()
    numeric = deepcopy(cfg)
    numeric['physics']['grid_parameters']['WS']['N'] = 32
    numeric['study'].update(n_y=9, b_step_fm=.5)
    validate_pair(cfg, numeric, rows, rows)
    changed = deepcopy(cfg); changed['physics']['collision_parameters']['gamma'] = 200
    with pytest.raises(ValueError, match='physical parameters'):
        validate_pair(cfg, changed, rows, rows)
    reordered = deepcopy(rows)
    reordered['beta2']['rotation_bank_1'] = reordered['beta2']['rotation_bank_1'][::-1]
    with pytest.raises(ValueError, match='ordering'):
        validate_pair(cfg, cfg, rows, reordered)
    shorter = deepcopy(cfg); shorter['study']['qmax'] = .16
    with pytest.raises(ValueError, match='qmax'):
        validate_pair(cfg, shorter, rows, rows)


def test_each_variant_recomputes_its_own_spherical_and_skin_references():
    cfg, rows = example()
    candidate = deepcopy(rows)
    for row in candidate.values():
        row['orientation_coefficients'][:, 1] += .08
    result = compare_loaded(cfg, cfg, rows, candidate)
    for row in result['cases'].values():
        assert np.max(abs(row['C2_shift'])) > .01
        np.testing.assert_allclose(row['sphere_response_shift'], 0, atol=2e-16)
        np.testing.assert_allclose(row['skin_response_shift'], 0, atol=2e-16)
        assert row['HM_response_shift'] == pytest.approx(0, abs=2e-15)


def test_global_yield_rescaling_cancels_in_both_observables():
    cfg, rows = example()
    candidate = deepcopy(rows)
    for i, row in enumerate(candidate.values()):
        row['orientation_coefficients'] *= 1.7+i
    result = compare_loaded(cfg, cfg, rows, candidate)
    for row in result['cases'].values():
        np.testing.assert_allclose(row['C2_shift'], 0, atol=2e-15)
        assert row['HM_shift'] == pytest.approx(0, abs=3e-15)
        assert row['HM_response_shift'] == pytest.approx(0, abs=3e-15)


def test_candidate_uses_its_own_cartesian_grid_on_the_common_q_window():
    cfg, rows = example()
    candidate = deepcopy(rows)
    for row in candidate.values():
        row['axis'] = np.linspace(-.2, .2, 49)
    result = compare_loaded(cfg, cfg, rows, candidate)
    # Same nonzero angular coefficients sampled on different Cartesian grids
    # have a finite interpolation difference; reusing the reference grid would
    # incorrectly force this diagnostic to zero.
    assert np.max(abs(result['cases']['beta2']['C2_shift'])) > 1e-5
    np.testing.assert_allclose(result['q'], np.linspace(0, .17, 15), rtol=0, atol=0)
