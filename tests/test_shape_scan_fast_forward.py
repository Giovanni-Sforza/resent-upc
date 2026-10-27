"""Numerical closure of optimized saved-A0 evaluation against the frozen kernel."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from upc_deformation.decay import stokes_maps
from upc_deformation.diffraction_forward import DiffractionRefiner
from upc_deformation.precision_campaign import GRID_KEYS
from upc_deformation.shape_scan_fast_forward import ShapeScanFastRefiner
from upc_deformation.shape_scan_physics import build_cases, resolve_case
from upc_deformation.vendor.upc_forward import _interp2d_batched


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/shape_scan_smoke.yaml").read_text())
    cfg["physics"]["grid_parameters"] = {
        "WS": {"N": 16, "L": 24.}, "T_prim": {"N": 16, "L": 24.},
        "output": {"N": 48, "L": 96.}}
    cfg["physics"]["output_settings"].update(safe_padding_factor=2, p_limit_q=.10, target_size_q=25)
    cfg["study"].update(b_min_fm=12., b_max_fm=16., b_step_fm=1., n_y=3,
                        ymax=1., n_radial=31, n_phi=16, y_batch_size=2, qmax=.08)
    return cfg


def rotations():
    angle = .731
    c, s = np.cos(angle), np.sin(angle)
    ry = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return np.stack((np.eye(3), ry)), np.stack((rz, np.eye(3)))


def resolved(cfg, kind="halo", spherical=False):
    chosen = next(case for case in build_cases(cfg) if case["density_type"] == kind
                  and case["beta2"] == (0. if spherical else .3)
                  and case["beta3"] == (0. if spherical else .2))
    kernel_cfg, case, _ = resolve_case(cfg, chosen)
    return kernel_cfg, case


def reference_a0(result, indices):
    levels = result["b_level_coefficients"][:, :, :, 0, :][..., indices]
    nodes = result["b_node_coefficients"][:, :, :, :, 0, :][..., indices]
    return levels.transpose(1, 0, 2, 3), nodes


def assert_close(actual, expected, rtol=3e-13):
    # Scale absolute tolerance by the reference peak, so destructive zeros do
    # not produce an unbounded relative error. No absolute floor hides a small
    # signal. This tolerance is tighter than the existing CPU/CUDA closure.
    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol*scale)


@pytest.mark.parametrize("kind", ["none", "skin", "halo"])
@pytest.mark.parametrize("spherical", [False, True])
def test_saved_a0_and_each_node_match_frozen_kernel(config, kind, spherical):
    cfg, case = resolved(config, kind, spherical)
    r1, r2 = rotations()
    old = DiffractionRefiner(cfg, case).calculate(r1, r2, return_nodes=True)
    kernel = ShapeScanFastRefiner(cfg, case)
    actual = kernel.calculate(r1, r2, return_nodes=True)
    expected, nodes = reference_a0(old, actual["stored_indices"])
    assert_close(actual["A0"], expected)
    assert_close(actual["b_node_A0"], nodes)
    for key in GRID_KEYS:
        np.testing.assert_array_equal(actual[key], old[key])
    np.testing.assert_array_equal(actual["stored_q"], old["radii"][old["radii"] <= .08])
    assert actual["A0"].dtype == np.float64
    assert actual["A0"].shape == (2, 3, 2, len(actual["stored_q"]))
    assert actual["branch_names"] == ["full", "self"]
    assert actual["evaluated_intensity_vertices"] < actual["native_fft_vertices"]
    assert kernel.model.b_imp == config["physics"]["collision_parameters"]["b"]


def test_sparse_projection_squares_before_interpolation(config):
    kernel = ShapeScanFastRefiner(*resolved(config))
    generator = torch.Generator().manual_seed(731)
    n = kernel.model._target_points
    x = torch.randn((2, n, n), generator=generator, dtype=torch.complex128)
    y = torch.randn((2, n, n), generator=generator, dtype=torch.complex128)
    s0 = stokes_maps(x, y)[:, 0]
    lower, upper = kernel._projection_bounds
    queries = kernel._saved_queries[None].expand(2, -1, -1)
    expected = _interp2d_batched(s0, queries, lower, upper, lower, upper)
    expected = expected.reshape(2, len(kernel.stored_q), kernel.n_phi).mean(-1)
    vertices = s0.flatten(-2).index_select(-1, kernel._projection_vertices)
    actual = kernel._project_intensity_vertices(vertices)
    torch.testing.assert_close(actual, expected, rtol=3e-14, atol=3e-14)
    # Interpolating complex amplitudes and then squaring is *not* this contract.
    xx = _interp2d_batched(x, queries, lower, upper, lower, upper)
    yy = _interp2d_batched(y, queries, lower, upper, lower, upper)
    wrong = (xx.abs().square()+yy.abs().square())/(2*np.pi)
    wrong = wrong.reshape_as(expected[..., None].expand(-1, -1, kernel.n_phi)).mean(-1)
    assert not torch.allclose(actual, wrong, rtol=.01, atol=.001)


def test_sparse_fft_matches_native_padding_phase_across_repeated_batches(config):
    kernel = ShapeScanFastRefiner(*resolved(config))
    generator = torch.Generator().manual_seed(421)
    n = kernel.model._output_N
    # Alternating batches detect stale transform data; asymmetric random fields
    # detect an omitted padding/fftshift phase. CZT and dense FFT evaluate the
    # same finite DFT with different floating-point operation order.
    for batch in (2, 2, 1, 2):
        x = torch.randn((batch, n, n), dtype=torch.complex128, generator=generator)
        y = torch.randn((batch, n, n), dtype=torch.complex128, generator=generator)
        mx, my, _ = kernel.model._coherent_amplitude_q(x, y)
        for field, reference in ((x, mx), (y, my)):
            expected = reference.flatten(-2).index_select(-1, kernel._projection_vertices)
            torch.testing.assert_close(kernel._fft_at_vertices(field), expected,
                                       rtol=3e-13, atol=3e-13*float(expected.abs().max()))


def test_batches_y_chunks_and_b_levels_preserve_saved_a0(config):
    cfg, case = resolved(config)
    r1, r2 = rotations()
    kernel = ShapeScanFastRefiner(cfg, case)
    together = kernel.calculate(r1, r2, return_nodes=True)
    singles = [kernel.calculate(a, b)["A0"] for a, b in zip(r1, r2)]
    assert_close(together["A0"], np.concatenate(singles))
    cfg = deepcopy(cfg)
    cfg["study"]["y_batch_size"] = 3
    changed = ShapeScanFastRefiner(cfg, case).calculate(r1, r2, return_nodes=True)
    assert_close(changed["A0"], together["A0"])
    assert_close(changed["b_node_A0"], together["b_node_A0"])
    expected = np.einsum("lb,y,boysr->olsr", together["b_level_weights"],
                         together["y_weights"], together["b_node_A0"])
    assert_close(together["A0"], expected)


def test_destructive_spherical_origin_and_failure_restores_geometry(config, monkeypatch):
    cfg, case = resolved(config, spherical=True)
    kernel = ShapeScanFastRefiner(cfg, case)
    actual = kernel.calculate(np.eye(3), np.eye(3), return_nodes=True)
    middle = np.argmin(abs(actual["y_values"]))
    full = actual["b_node_A0"][:, 0, middle, 0, 0]
    self_terms = actual["b_node_A0"][:, 0, middle, 1, 0]
    assert np.all(self_terms > 0)
    assert np.all(full < 1e-6*self_terms)
    original_b = kernel.model.b_imp

    def fail(_):
        raise RuntimeError("intentional node failure")

    monkeypatch.setattr(kernel, "_node_A0", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        kernel.calculate(np.eye(3), np.eye(3))
    assert kernel.model.b_imp == original_b


def test_rejects_bad_rotations_and_invalid_stored_interval(config):
    cfg, case = resolved(config)
    with pytest.raises(ValueError, match="float64"):
        ShapeScanFastRefiner(cfg, case, dtype="float32")
    kernel = ShapeScanFastRefiner(cfg, case)
    with pytest.raises(ValueError, match="same number"):
        kernel.calculate(np.stack((np.eye(3), np.eye(3))), np.eye(3))
    with pytest.raises(ValueError, match="proper orthogonal"):
        kernel.calculate(np.diag([1., 1., -1.]), np.eye(3))
    for qmax in (0., -1., float("nan"), 1.):
        bad = deepcopy(cfg); bad["study"]["qmax"] = qmax
        with pytest.raises(ValueError, match="qmax"):
            ShapeScanFastRefiner(bad, case)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable on this host")
def test_cuda_saved_a0_closure_against_frozen_cuda_and_cpu(config):
    cfg, case = resolved(config)
    r1, r2 = rotations()
    reference = DiffractionRefiner(cfg, case, device="cuda").calculate(r1, r2, return_nodes=True)
    fast = ShapeScanFastRefiner(cfg, case, device="cuda").calculate(r1, r2, return_nodes=True)
    expected, nodes = reference_a0(reference, fast["stored_indices"])
    assert_close(fast["A0"], expected, rtol=2e-10)
    assert_close(fast["b_node_A0"], nodes, rtol=2e-10)
    cpu = ShapeScanFastRefiner(cfg, case).calculate(r1, r2)
    assert_close(fast["A0"], cpu["A0"], rtol=2e-10)
