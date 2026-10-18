"""Small-grid closure of the shared full/self precision forward calculation."""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from upc_deformation.decay import b_quadrature, stokes_maps
from upc_deformation.diffraction_forward import DiffractionRefiner
from upc_deformation.precision_campaign import GRID_KEYS
from upc_deformation.precision_forward import PrecisionPanelDRefiner


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
                          "configs/panel_d_precision_a100.yaml").read_text())
    cfg["physics"]["grid_parameters"] = {
        "WS": {"N": 16, "L": 24.}, "T_prim": {"N": 16, "L": 24.},
        "output": {"N": 48, "L": 96.},
    }
    cfg["physics"]["output_settings"].update(
        safe_padding_factor=2, p_limit_q=.10, target_size_q=25)
    cfg["study"].update(b_min_fm=12., b_max_fm=16., b_step_fm=1.,
                        n_y=3, ymax=1., n_radial=31, n_phi=16, y_batch_size=3)
    return cfg


def case(beta2=.3, beta3=.2):
    return dict(name="test", beta2=beta2, beta3=beta3, skin_fm=0.)


def rotations():
    angle = .731
    ry = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                   [-np.sin(angle), 0., np.cos(angle)]])
    rz = np.array([[np.cos(angle), -np.sin(angle), 0.],
                   [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    return np.stack((np.eye(3), ry)), np.stack((rz, np.eye(3)))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_full_branch_matches_existing_separate_fft_kernel_and_metadata(config, dtype):
    r1, r2 = rotations()
    old = PrecisionPanelDRefiner(config, case(), dtype=dtype, fft_mode="separate")
    new = DiffractionRefiner(config, case(), dtype=dtype)
    expected = old.calculate(r1, r2, return_nodes=True)
    actual = new.calculate(r1, r2, return_nodes=True)
    tolerance = 3e-7 if dtype == "float32" else 3e-13
    for key, selection in (("coefficients", (slice(None), 0)),
                           ("b_level_coefficients", (slice(None), slice(None), 0)),
                           ("b_node_coefficients", (slice(None), slice(None), slice(None), 0))):
        np.testing.assert_allclose(actual[key][selection], expected[key], rtol=tolerance, atol=1e-13)
    assert actual["coefficients"].shape == (2, 2, 3, 31)
    assert actual["b_level_coefficients"].shape == (3, 2, 2, 3, 31)
    assert actual["b_node_coefficients"].shape == (5, 2, 3, 2, 3, 31)
    assert actual["coefficients"].dtype == np.float64
    assert actual["branch_names"] == ["full", "self"]
    for key in GRID_KEYS:
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in ("fft_mode", "device", "dtype", "y_batch_size", "n_orientations"):
        assert actual[key] == expected[key]
    assert actual["elapsed_seconds"] >= actual["preparation_seconds"] > 0
    assert actual["peak_memory_allocated_bytes"] == actual["peak_memory_reserved_bytes"] == 0
    assert new.model.b_imp == config["physics"]["collision_parameters"]["b"]


def test_native_bilinears_precede_projection_and_cross_is_amplitude_interference(config, monkeypatch):
    refiner = DiffractionRefiner(config, case())
    r1, r2 = (refiner._rotation(r) for r in rotations())
    captured = []
    project = refiner._project_stokes

    def record(stokes):
        captured.append(stokes.clone())
        return project(stokes)

    monkeypatch.setattr(refiner, "_project_stokes", record)
    with torch.inference_mode():
        prepared = refiner._prepare_y(refiner._prepare(r1, r2), len(r1), 0, 3)
        actual = refiner._node(prepared)
        amplitudes = []
        model = refiner.model
        for target, source in ((1, 2), (2, 1)):
            fields = model._compute_amplitude_at_grid(
                target, T_prim_target=prepared[f"t{target}"],
                T_thick_source=prepared[f"thick{source}"],
                Vx_source=prepared[f"vx{source}"], Vy_source=prepared[f"vy{source}"],
                Egamma_target=prepared[f"energy{source}"],
                Ep_target=getattr(model, f"Ep_{target}"), A_source=getattr(model, f"A_{source}"))
            amplitudes.append(model._coherent_amplitude_q(*fields)[:2])
        (x1, y1), (x2, y2) = amplitudes
        self_terms = stokes_maps(x1, y1) + stokes_maps(x2, y2)
        xx, yy = 2 * (x1 * x2.conj()).real, 2 * (y1 * y2.conj()).real
        xy = 2 * (x1 * y2.conj() + x2 * y1.conj()).real
        cross = torch.stack((xx + yy, xx - yy, xy), dim=-3) / (2 * math.pi)
        assert len(captured) == 2
        assert captured[0].shape == captured[1].shape == (6, 3, 96, 96)
        torch.testing.assert_close(captured[1], self_terms, rtol=0, atol=0)
        torch.testing.assert_close(captured[0] - captured[1], cross,
                                   rtol=1e-12, atol=1e-13 * float(self_terms.abs().max()))
        torch.testing.assert_close(actual[:, 1], project(self_terms), rtol=0, atol=0)
        torch.testing.assert_close(actual[:, 0] - actual[:, 1], project(cross),
                                   rtol=1e-12, atol=1e-13 * float(actual.abs().max()))


def test_branches_remain_unnormalized_and_preserve_destructive_origin(config, monkeypatch):
    refiner = DiffractionRefiner(config, case(beta2=0., beta3=0.))
    baseline = refiner.calculate(np.eye(3), np.eye(3), return_nodes=True)
    mid_y = int(np.argmin(np.abs(baseline["y_values"])))
    full = baseline["b_node_coefficients"][:, 0, mid_y, 0, 0, 0]
    self_terms = baseline["b_node_coefficients"][:, 0, mid_y, 1, 0, 0]
    assert np.all(self_terms > 0)
    # Half-grid b shifts on this deliberately coarse spatial grid leave a
    # small interpolation residue, as in the existing source-amplitude test.
    assert np.all(full < 1e-6 * self_terms)
    intensities = baseline["coefficients"][:, :, 0]
    assert np.all(intensities >= 0)
    assert np.all(intensities[:, 0] <= 2 * intensities[:, 1] * (1 + 1e-14))
    original = refiner.model._compute_amplitude_at_grid

    def scale_amplitudes(*args, **kwargs):
        return tuple(3 * field for field in original(*args, **kwargs))

    monkeypatch.setattr(refiner.model, "_compute_amplitude_at_grid", scale_amplitudes)
    scaled = refiner.calculate(np.eye(3), np.eye(3))
    expected = 9 * baseline["coefficients"]
    np.testing.assert_allclose(scaled["coefficients"], expected,
                               rtol=3e-13, atol=1e-13 * np.max(np.abs(expected)))


def test_orientation_batches_and_rapidity_chunks_preserve_both_branches(config):
    r1, r2 = rotations()
    refiner = DiffractionRefiner(config, case())
    together = refiner.calculate(r1, r2, return_nodes=True)
    singles = [refiner.calculate(a, b, return_nodes=True) for a, b in zip(r1, r2)]
    for key, axis in (("coefficients", 0), ("b_level_coefficients", 1), ("b_node_coefficients", 1)):
        expected = np.concatenate([result[key] for result in singles], axis=axis)
        np.testing.assert_allclose(together[key], expected, rtol=3e-13, atol=1e-13)
    chunk_config = copy.deepcopy(config)
    chunk_config["study"]["y_batch_size"] = 2  # final rapidity chunk has one node
    chunked = DiffractionRefiner(chunk_config, case()).calculate(r1, r2, return_nodes=True)
    for key in ("coefficients", "b_level_coefficients", "b_node_coefficients"):
        np.testing.assert_allclose(chunked[key], together[key], rtol=3e-13, atol=1e-13)


def test_b_levels_use_exact_subset_nodes_for_both_branches(config):
    r1, r2 = rotations()
    result = DiffractionRefiner(config, case()).calculate(r1[:1], r2[:1], return_nodes=True)
    np.testing.assert_array_equal(result["b_steps_fm"], [1., 2., 4.])
    for level, stride in enumerate((1, 2, 4)):
        weights = b_quadrature(torch.from_numpy(result["b_values"][::stride])).numpy()
        expected = np.einsum("b,y,boyscr->oscr", weights, result["y_weights"],
                             result["b_node_coefficients"][::stride])
        np.testing.assert_allclose(result["b_level_coefficients"][level], expected,
                                   rtol=3e-14, atol=1e-13)
        coarse = copy.deepcopy(config)
        coarse["study"]["b_step_fm"] = float(stride)
        recomputed = DiffractionRefiner(coarse, case()).calculate(r1[:1], r2[:1])
        np.testing.assert_allclose(result["b_level_coefficients"][level], recomputed["coefficients"],
                                   rtol=3e-13, atol=1e-13)


def test_rejects_combined_fft_and_restores_geometry_after_failure(config, monkeypatch):
    with pytest.raises(ValueError, match="separate"):
        DiffractionRefiner(config, case(), fft_mode="combined")
    refiner = DiffractionRefiner(config, case())
    r1, r2 = rotations()
    with pytest.raises(ValueError, match="same number"):
        refiner.calculate(r1, r2[:1])
    with pytest.raises(ValueError, match="proper orthogonal"):
        refiner.calculate(np.diag([1., 1., -1.]), np.eye(3))
    original_b = refiner.model.b_imp

    def fail(_):
        raise RuntimeError("intentional node failure")

    monkeypatch.setattr(refiner, "_node", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        refiner.calculate(r1[:1], r2[:1])
    assert refiner.model.b_imp == original_b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable on this host")
def test_cuda_cpu_closure_for_full_and_self(config):
    r1, r2 = rotations()
    cpu = DiffractionRefiner(config, case(), device="cpu").calculate(r1, r2, return_nodes=True)
    cuda = DiffractionRefiner(config, case(), device="cuda").calculate(r1, r2, return_nodes=True)
    for key in ("coefficients", "b_level_coefficients", "b_node_coefficients"):
        scale = np.max(np.abs(cpu[key]))
        np.testing.assert_allclose(cuda[key], cpu[key], rtol=2e-10, atol=2e-10 * scale)
    assert cuda["peak_memory_allocated_bytes"] > 0
    assert cuda["device"].startswith("cuda")
