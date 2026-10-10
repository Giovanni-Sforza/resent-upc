import numpy as np
import pytest
import torch

from upc_deformation.orientation_refinement import PanelDOrientationRefiner


def test_rotation_contract_rejects_reflection_and_nonorthogonal_input():
    eye = np.eye(3)
    assert PanelDOrientationRefiner._rotation(eye).shape == (1, 3, 3)
    for value in (np.diag([1., 1., -1.]), eye*2, np.zeros((2, 3, 3))):
        with pytest.raises(ValueError):
            PanelDOrientationRefiner._rotation(value)


def test_refinement_refuses_changed_numerical_scope_before_model_construction():
    config = {'study': {'longitudinal_units': 'physical', 'b_min_fm': 10,
                       'b_max_fm': 60, 'b_nodes': 25, 'n_y': 5, 'ymax': 1,
                       'n_radial': 757, 'n_phi': 256}}
    with pytest.raises(ValueError, match='preserve original'):
        PanelDOrientationRefiner(config, {})
    with pytest.raises(ValueError, match='fft_mode'):
        PanelDOrientationRefiner(config, {}, fft_mode='approximate')


def _fake_refiner():
    obj = object.__new__(PanelDOrientationRefiner)
    obj.model = type('Model', (), {'b_imp': 7., '_p_full_q': torch.linspace(-1., 1., 9)})()
    obj.fft_mode = 'combined'
    obj.axis = torch.linspace(-.2, .2, 5)
    obj.radii = torch.linspace(0., .3, 7)
    obj.b_values = torch.linspace(12., 60., 25, dtype=torch.float64)
    from upc_deformation.decay import b_quadrature
    obj.b_weights = b_quadrature(obj.b_values)
    obj.y_values, obj.y_weights = np.polynomial.legendre.leggauss(5)
    return obj


def test_b_independent_prepare_runs_once_and_b_is_restored():
    obj = _fake_refiner()
    calls = []
    obj._prepare = lambda r1, r2: calls.append('prepare') or None
    def node(prepared):
        calls.append(obj.model.b_imp)
        return torch.full((5, 3, 7), obj.model.b_imp, dtype=torch.float32)
    obj._node = node
    result = obj.calculate(np.eye(3), np.eye(3))
    assert calls.count('prepare') == 1
    assert calls[1:] == obj.b_values.tolist()
    assert obj.model.b_imp == 7.
    expected = (obj.b_weights*obj.b_values).sum().item()*obj.y_weights.sum()
    np.testing.assert_allclose(result['coefficients'], expected, rtol=1e-14)
    assert result['b_node_coefficients'].shape == (25, 5, 3, 7)
    assert result['coefficients'].dtype == np.float64


def test_b_is_restored_if_node_raises():
    obj = _fake_refiner()
    obj._prepare = lambda r1, r2: None
    def failure(prepared):
        raise RuntimeError('node failed')
    obj._node = failure
    with pytest.raises(RuntimeError, match='node failed'):
        obj.calculate(np.eye(3), np.eye(3))
    assert obj.model.b_imp == 7.


def test_combined_fft_preserves_stokes_projection_against_separate_branches(monkeypatch):
    from upc_deformation import orientation_refinement as module
    obj = object.__new__(PanelDOrientationRefiner)
    generator = torch.Generator().manual_seed(402)
    branch_fields = [torch.complex(torch.randn(5, 8, 8, generator=generator),
                                  torch.randn(5, 8, 8, generator=generator)) for _ in range(4)]
    class Model:
        Ep_1 = Ep_2 = 1.
        A_1 = A_2 = 96
        def _compute_amplitude_at_grid(self, nuclear_ID, **kwargs):
            return branch_fields[2*(nuclear_ID-1):2*nuclear_ID]
        def _coherent_amplitude_q(self, ax, ay):
            return torch.fft.fft2(ax), torch.fft.fft2(ay), torch.arange(8)
    obj.model = Model()
    obj.radii, obj.n_phi = torch.arange(3), 256
    # Retain the nonlinear Stokes operation; only replace unrelated ring sampling.
    monkeypatch.setattr(module, 'decay_coefficients', lambda stokes, *args: stokes.mean((-1, -2)))
    prepared = dict.fromkeys(('t1', 't2', 'thick1', 'thick2', 'vx1', 'vy1', 'vx2', 'vy2', 'energy1', 'energy2'))
    obj.fft_mode = 'combined'
    combined = obj._node(prepared)
    obj.fft_mode = 'separate'
    separate = obj._node(prepared)
    torch.testing.assert_close(combined, separate, rtol=5e-6, atol=2e-5)
