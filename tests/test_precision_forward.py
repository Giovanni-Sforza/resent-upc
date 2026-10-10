import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from upc_deformation.decay import b_quadrature
from upc_deformation.forward import UPCForwardModel
from upc_deformation.orientation_refinement import PanelDOrientationRefiner
from upc_deformation.precision_forward import PrecisionPanelDRefiner, PrecisionUPCForwardModel


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    result = yaml.safe_load((Path(__file__).resolve().parents[1]/
                            "configs/panel_d_precision_a100.yaml").read_text())
    result["physics"]["grid_parameters"] = {
        "WS": {"N": 16, "L": 24.}, "T_prim": {"N": 16, "L": 24.},
        "output": {"N": 48, "L": 96.}}
    result["physics"]["output_settings"].update(safe_padding_factor=2,
        p_limit_q=.10, target_size_q=25)
    result["study"].update(b_min_fm=12., b_max_fm=16., b_step_fm=1.,
        n_y=3, ymax=1., n_radial=31, n_phi=16)
    return result


def case(beta2=.3, beta3=0):
    return dict(name="test", beta2=beta2, beta3=beta3, skin_fm=0.)


def rotations():
    angle = .731
    ry = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                   [-np.sin(angle), 0, np.cos(angle)]])
    rz = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return np.stack((np.eye(3), ry)), np.stack((rz, np.eye(3)))


def test_grids_and_spherical_caches_are_constructed_in_selected_precision(config):
    physics = config["physics"]
    # Non-binary spacing exposes whether FP32 coordinates were merely upcast.
    physics["grid_parameters"]["WS"]["L"] = 23.7
    physics["grid_parameters"]["T_prim"]["L"] = 23.7
    old = UPCForwardModel(copy.deepcopy(physics), "physical")
    single = PrecisionUPCForwardModel(copy.deepcopy(physics), dtype="float32")
    double = PrecisionUPCForwardModel(copy.deepcopy(physics), dtype="float64")
    for name, values in old.named_buffers():
        torch.testing.assert_close(dict(single.named_buffers())[name], values, rtol=0, atol=0)
    expected = torch.arange(16, dtype=torch.float64)*(23.7/16)-23.7/2
    torch.testing.assert_close(double._WS_x_grid, expected, rtol=0, atol=0)
    assert not torch.equal(double._WS_x_grid, single._WS_x_grid.double())
    assert double._static_ws_rho_p_1.dtype == torch.float64
    assert double._static_ws_F_3d_1.dtype == torch.complex128
    assert not torch.equal(double._static_ws_rho_p_1, single._static_ws_rho_p_1.double())


def test_real_batched_forward_matches_single_orientations_and_y_chunks(config):
    r1, r2 = rotations()
    full = PrecisionPanelDRefiner(config, case(), dtype="float64")
    batched = full.calculate(r1, r2)
    singles = np.concatenate([full.calculate(a, b)["coefficients"] for a, b in zip(r1, r2)])
    np.testing.assert_allclose(batched["coefficients"], singles, rtol=3e-13, atol=1e-13)
    chunked_config = copy.deepcopy(config)
    chunked_config["study"]["y_batch_size"] = 1
    chunked = PrecisionPanelDRefiner(chunked_config, case(), dtype="float64").calculate(r1, r2)
    np.testing.assert_allclose(chunked["b_level_coefficients"], batched["b_level_coefficients"],
                               rtol=3e-13, atol=1e-13)
    assert batched["coefficients"].shape == (2, 3, 31)
    assert batched["coefficients"].dtype == np.float64
    assert batched["elapsed_seconds"] >= batched["preparation_seconds"] > 0
    assert full.model.b_imp == config["physics"]["collision_parameters"]["b"]


def test_float32_matches_frozen_legacy_kernel_on_identical_small_grids(config):
    current = PrecisionPanelDRefiner(config, case(), dtype="float32")
    # Construct only the old wrapper's small-grid metadata by hand: its public
    # constructor deliberately refuses anything except the frozen full grid.
    # The actual old preparation, forward, projection and integration run here.
    legacy = object.__new__(PanelDOrientationRefiner)
    legacy.model = UPCForwardModel(copy.deepcopy(current.physics), "physical")
    legacy.fft_mode = "combined"
    legacy.axis, legacy.radii = legacy.model._p_axis_q, current.radii.clone()
    legacy.b_values = current.b_values.clone()
    legacy.b_weights = b_quadrature(legacy.b_values)
    legacy.y_values, legacy.y_weights = current.y_values.copy(), current.y_weights.copy()
    legacy.y_tensor, legacy.n_phi = current.y_tensor[None].clone(), current.n_phi
    r1, r2 = rotations()
    expected = legacy.calculate(r1[1], r2[1], return_nodes=True)
    actual = current.calculate(r1[1], r2[1], return_nodes=True)
    np.testing.assert_allclose(actual["coefficients"][0], expected["coefficients"], rtol=2e-7, atol=1e-10)
    np.testing.assert_array_equal(actual["b_node_coefficients"][:, 0], expected["b_node_coefficients"])


def test_coarse_levels_reintegrate_exact_subset_nodes_with_same_endpoints(config):
    r1, r2 = rotations()
    refiner = PrecisionPanelDRefiner(config, case(beta2=0, beta3=.2))
    actual = refiner.calculate(r1[:1], r2[:1], return_nodes=True)
    np.testing.assert_array_equal(actual["b_steps_fm"], [1., 2., 4.])
    for level, stride in enumerate((1, 2, 4)):
        subset_b = actual["b_values"][::stride]
        weights = b_quadrature(torch.from_numpy(subset_b)).numpy()
        expected = np.einsum("b,y,boycr->ocr", weights, actual["y_weights"],
                             actual["b_node_coefficients"][::stride])
        np.testing.assert_allclose(actual["b_level_coefficients"][level], expected,
                                   rtol=3e-14, atol=1e-13)
        coarser = copy.deepcopy(config)
        coarser["study"]["b_step_fm"] = float(stride)
        recomputed = PrecisionPanelDRefiner(coarser, case(beta2=0, beta3=.2)).calculate(r1[:1], r2[:1])
        np.testing.assert_allclose(actual["b_level_coefficients"][level], recomputed["coefficients"],
                                   rtol=3e-13, atol=1e-13)


def test_contract_rejects_bad_rotations_and_restores_b_after_failure(config, monkeypatch):
    with pytest.raises(ValueError, match="dtype"):
        PrecisionPanelDRefiner(config, case(), dtype="float16")
    invalid = copy.deepcopy(config)
    invalid["study"]["b_step_fm"] = 1.3
    with pytest.raises(ValueError, match="divide"):
        PrecisionPanelDRefiner(invalid, case())
    refiner = PrecisionPanelDRefiner(config, case())
    with pytest.raises(ValueError, match="proper orthogonal"):
        refiner.calculate(np.diag([1, 1, -1]), np.eye(3))
    r1, r2 = rotations()
    with pytest.raises(ValueError, match="same number"):
        refiner.calculate(r1, r2[:1])
    original = refiner.model.b_imp
    def failure(_):
        raise RuntimeError("intentional node failure")
    monkeypatch.setattr(refiner, "_node", failure)
    with pytest.raises(RuntimeError, match="intentional"):
        refiner.calculate(r1[:1], r2[:1])
    assert refiner.model.b_imp == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable on this host")
def test_cuda_cpu_same_precision_real_forward_closure(config):
    r1, r2 = rotations()
    cpu = PrecisionPanelDRefiner(config, case(), device="cpu").calculate(r1, r2)
    cuda = PrecisionPanelDRefiner(config, case(), device="cuda").calculate(r1, r2)
    scale = np.max(np.abs(cpu["coefficients"]))
    np.testing.assert_allclose(cuda["coefficients"], cpu["coefficients"], rtol=2e-10, atol=2e-10*scale)
    assert cuda["peak_memory_allocated_bytes"] > 0
    assert cuda["device"].startswith("cuda")
