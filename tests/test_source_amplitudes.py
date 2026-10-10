"""CPU closure and interference checks for the source-resolved adapter."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from upc_deformation.decay import stokes_maps
from upc_deformation.forward import UPCForwardModel
from upc_deformation.source_amplitudes import source_stokes


def tiny_model(beta2=0.0, beta3=0.0):
    path = Path(__file__).resolve().parents[1] / "configs/jpsi_decay_axis.yaml"
    cfg = deepcopy(yaml.safe_load(path.read_text())["physics"])
    cfg["nuclear_parameters"].update(beta2=beta2, beta3=beta3)
    cfg["grid_parameters"] = {
        "WS": {"N": 16, "L": 24.0},
        "T_prim": {"N": 16, "L": 24.0},
        "output": {"N": 32, "L": 64.0},
    }
    cfg["output_settings"].update(safe_padding_factor=2, target_size_q=33)
    return UPCForwardModel(cfg, "physical")


@pytest.mark.parametrize("deformed", [False, True])
def test_full_stokes_close_to_existing_forward_and_self_is_physical(deformed):
    model = tiny_model(.3 if deformed else 0, .15 if deformed else 0)
    y_views = torch.tensor([[-.7, 0, .7], [-.4, .2, .8]])
    r1 = torch.eye(3).repeat(2, 1, 1)
    r2 = r1.clone()
    r2[1] = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    original_b = model.b_imp
    full, self_terms, axis = source_stokes(model, y_views, [16., 23.], r1, r2)
    mx, my, expected_axis = model.fixed_b_complex_amplitude_multi_y(
        None, None, y_views, [16., 23.], R_1=r1, R_2=r2,
    )
    expected = stokes_maps(mx, my)
    assert full.shape == self_terms.shape == (2, 2, 3, 3, 64, 64)
    torch.testing.assert_close(axis, expected_axis, rtol=0, atol=0)
    # The native path adds spatial amplitudes before FFT; the branch API adds
    # their FFTs.  Float32 roundoff therefore needs a peak-scaled tolerance.
    scale = float(expected.abs().max())
    torch.testing.assert_close(full, expected, rtol=3e-5, atol=scale * 1e-6)
    assert model.b_imp == original_b
    assert bool(torch.isfinite(full).all() and torch.isfinite(self_terms).all())
    assert bool((self_terms[..., 0, :, :] >= 0).all())
    # Each Stokes map is a positive polarization density matrix, also after
    # adding branch intensities.  S1/S2 themselves need not be positive.
    for maps in (full, self_terms):
        s0, s1, s2 = maps.unbind(-3)
        assert bool((s0.square() + scale**2 * 1e-6 >= s1.square() + s2.square()).all())


def test_spherical_midrapidity_origin_cancels_only_with_cross_term():
    model = tiny_model()
    full, self_terms, axis = source_stokes(model, torch.tensor([[0., .8]]), [20.])
    zero = int(axis.abs().argmin())
    assert float(axis[zero]) == 0
    self_origin = self_terms[0, 0, :, 0, zero, zero]
    full_origin = full[0, 0, :, 0, zero, zero]
    assert float(self_origin[0]) > 0
    assert float(full_origin[0] / self_origin[0]) < 1e-6
    # Unequal photon energies at finite y break the equal-source cancellation.
    assert float(full_origin[1] / self_origin[1]) > 1e-3


def test_optional_empirical_background_is_positive_on_separate_output_grid():
    model = tiny_model()
    full, self_terms, axis, empirical = source_stokes(
        model, torch.tensor([[0., .8]]), [20.], return_incoherent=True,
    )
    assert empirical.shape == (1, 1, 2, 33, 33)
    assert len(axis) == 64
    assert bool(torch.isfinite(empirical).all() and (empirical >= 0).all())
    assert bool((empirical.sum((-1, -2)) > 0).all())
    # This optional comparison must not fill the coherent origin by accident.
    zero = int(axis.abs().argmin())
    assert float(full[0, 0, 0, 0, zero, zero] / self_terms[0, 0, 0, 0, zero, zero]) < 1e-6


def test_rejects_invalid_inputs_and_restores_model_b_after_branch_failure(monkeypatch):
    model = tiny_model()
    with pytest.raises(ValueError, match="finite rapidities"):
        source_stokes(model, torch.tensor([[float("nan")]]), [20.])
    with pytest.raises(ValueError, match="shape"):
        source_stokes(model, torch.zeros(2, 1), [20.], torch.eye(3)[None])
    with pytest.raises(ValueError, match="strictly increasing"):
        source_stokes(model, torch.zeros(1, 1), [23., 16.])

    original_b = model.b_imp
    def failure(*args, **kwargs):
        raise RuntimeError("deliberate branch error")
    monkeypatch.setattr(model, "_compute_amplitude_at_grid", failure)
    with pytest.raises(RuntimeError, match="deliberate branch error"):
        source_stokes(model, torch.zeros(1, 1), [17.])
    assert model.b_imp == original_b
