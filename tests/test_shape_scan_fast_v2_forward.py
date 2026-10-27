"""Closure and exact-support proofs for the independent v2 A0 kernel."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from upc_deformation.diffraction_forward import DiffractionRefiner
from upc_deformation.precision_campaign import GRID_KEYS
from upc_deformation.shape_scan_fast_forward import ShapeScanFastRefiner
from upc_deformation.shape_scan_fast_v2_forward import ShapeScanFastV2Refiner
from upc_deformation.shape_scan_physics import build_cases, resolve_case


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
        "output": {"N": 64, "L": 128.}}
    cfg["physics"]["output_settings"].update(safe_padding_factor=2, p_limit_q=.2, target_size_q=49)
    cfg["study"].update(b_min_fm=12., b_max_fm=14., b_step_fm=.5, n_y=3,
                        ymax=1., n_radial=61, n_phi=32, y_batch_size=2, qmax=.17)
    return cfg


def resolved(cfg, kind="halo", spherical=False):
    chosen = next(case for case in build_cases(cfg) if case["density_type"] == kind
                  and case["beta2"] == (0. if spherical else .3)
                  and case["beta3"] == (0. if spherical else .2))
    kernel_cfg, case, _ = resolve_case(cfg, chosen)
    return kernel_cfg, case


def rotations():
    angle = .731
    c, s = np.cos(angle), np.sin(angle)
    ry = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return np.stack((np.eye(3), ry)), np.stack((rz, np.eye(3)))


def assert_close(actual, expected, rtol=3e-12):
    expected = np.asarray(expected)
    # Per-channel peak absolute budget protects destructive zeros without
    # allowing a large branch to hide a smaller branch's discrepancy.
    peak = np.max(np.abs(expected), axis=-1, keepdims=True)
    difference = np.abs(np.asarray(actual)-expected)
    assert np.all(difference <= rtol*np.abs(expected)+rtol*peak), float(difference.max())


def legacy_saved(payload, qmax):
    indices = np.flatnonzero(payload["radii"] <= qmax)
    return (payload["b_level_coefficients"][:, :, :, 0, :][..., indices].transpose(1, 0, 2, 3),
            payload["b_node_coefficients"][:, :, :, :, 0, :][..., indices])


@pytest.mark.parametrize("kind", ["none", "skin", "halo"])
@pytest.mark.parametrize("spherical", [False, True])
def test_each_node_and_integrated_a0_match_original_and_v1(config, kind, spherical):
    cfg, case = resolved(config, kind, spherical)
    r1, r2 = rotations()
    legacy = DiffractionRefiner(cfg, case).calculate(r1, r2, return_nodes=True)
    v1 = ShapeScanFastRefiner(cfg, case).calculate(r1, r2, return_nodes=True)
    actual = ShapeScanFastV2Refiner(cfg, case).calculate(r1, r2, return_nodes=True)
    old_a0, old_nodes = legacy_saved(legacy, cfg["study"]["qmax"])
    assert_close(actual["A0"], old_a0)
    assert_close(actual["b_node_A0"], old_nodes)
    assert_close(actual["A0"], v1["A0"])
    assert_close(actual["b_node_A0"], v1["b_node_A0"])
    for key in (*GRID_KEYS, "stored_indices", "stored_q"):
        np.testing.assert_array_equal(actual[key], v1[key])
    assert actual["projection_after_integration"]
    assert actual["thickness_reused_across_y"] and actual["probability_reused_across_y"]
    assert actual["spatial_support_points_max"] < actual["original_spatial_points"]


@pytest.mark.parametrize("region", [(22., 26., .5), (36., 40., 1.)])
def test_overlap_boundary_and_large_b_match_reference(config, region):
    config["study"].update(b_min_fm=region[0], b_max_fm=region[1], b_step_fm=region[2])
    cfg, case = resolved(config)
    r1, r2 = rotations()
    ref = DiffractionRefiner(cfg, case).calculate(r1, r2, return_nodes=True)
    kernel = ShapeScanFastV2Refiner(cfg, case)
    actual = kernel.calculate(r1, r2, return_nodes=True)
    a0, nodes = legacy_saved(ref, cfg["study"]["qmax"])
    assert_close(actual["A0"], a0)
    assert_close(actual["b_node_A0"], nodes)
    assert actual["nonoverlap_source_nodes"] > 0
    if region[0] == 36.:
        assert actual["nonoverlap_source_nodes"] == 2*len(kernel.b_host)
    else:
        assert 0 < actual["nonoverlap_source_nodes"] < 2*len(kernel.b_host)


@pytest.mark.parametrize("nonbinary", [False, True])
def test_full_spatial_amplitude_is_exactly_zero_outside_proven_support(config, nonbinary):
    if nonbinary:
        config["physics"]["grid_parameters"]["WS"]["L"] = 24.3
        config["physics"]["grid_parameters"]["T_prim"]["L"] = 24.3
        config["physics"]["grid_parameters"]["output"]["L"] = 128.7
        config["study"].update(b_min_fm=12.3, b_max_fm=14.3)
    cfg, case = resolved(config)
    kernel = ShapeScanFastV2Refiner(cfg, case)
    r1, r2 = (kernel._rotation(r) for r in rotations())
    old_base = kernel._prepare(r1, r2)
    old = kernel._prepare_y(old_base, 2, 0, 2)
    new = kernel._prepare_y_v2(kernel._prepare_v2(r1, r2), 2, 0, 2)
    model = kernel.model
    for bi, b in enumerate(kernel.b_host):
        model.b_imp = b
        for support in kernel._supports[bi]:
            target, source = support["target"], support["source"]
            fields = model._compute_amplitude_at_grid(target, old[f"t{target}"], old[f"thick{source}"],
                old[f"vx{source}"], old[f"vy{source}"], old[f"energy{source}"],
                getattr(model, f"Ep_{target}"), getattr(model, f"A_{source}"))
            selected = (slice(None), *support["selection"])
            cropped = kernel._amplitude_support(new, bi, support)
            for component, field in enumerate(fields):
                outside = field.clone()
                outside[selected] = 0
                assert torch.count_nonzero(outside) == 0
                torch.testing.assert_close(cropped[:, component], field[selected], rtol=3e-13,
                                           atol=3e-13*float(field.abs().max()))


def test_physical_longitudinal_units_and_cached_y_preparation_match_original(config):
    cfg, case = resolved(config)
    kernel = ShapeScanFastV2Refiner(cfg, case)
    r1, r2 = (kernel._rotation(r) for r in rotations())
    old_base, new_base = kernel._prepare(r1, r2), kernel._prepare_v2(r1, r2)
    for start, stop in ((0, 2), (2, 3)):
        old = kernel._prepare_y(old_base, 2, start, stop)
        new = kernel._prepare_y_v2(new_base, 2, start, stop)
        for nid in (1, 2):
            t, fields = new[f"t_channels{nid}"], new[f"field_channels{nid}"]
            reconstructed = {f"t{nid}": torch.complex(t[:, 0], t[:, 1]),
                f"vx{nid}": torch.complex(fields[:, 0], fields[:, 1]),
                f"vy{nid}": torch.complex(fields[:, 2], fields[:, 3])}
            for name, value in reconstructed.items():
                torch.testing.assert_close(value, old[name], rtol=3e-14,
                                           atol=3e-14*float(old[name].abs().max()))
            thick = new_base[f"thick{nid}"][:, None].expand(-1, stop-start, -1, -1).reshape_as(old[f"thick{nid}"])
            torch.testing.assert_close(thick, old[f"thick{nid}"], rtol=0, atol=0)


def test_probabilities_are_reused_across_y_and_nonoverlap_is_exact_one(config, monkeypatch):
    config["study"].update(b_min_fm=22., b_max_fm=26., b_step_fm=.5)
    cfg, case = resolved(config)
    kernel = ShapeScanFastV2Refiner(cfg, case)
    r1, r2 = (kernel._rotation(r) for r in rotations())
    base = kernel._prepare_v2(r1, r2)
    counts = []
    original = kernel.model._prob

    def count(*args, **kwargs):
        counts.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel.model, "_prob", count)
    for start, stop in ((0, 2), (2, 3)):
        prepared = kernel._prepare_y_v2(base, 2, start, stop)
        for bi, row in enumerate(kernel._supports):
            for support in row:
                probability = kernel._probability(prepared, bi, support)
                sampled, = kernel._sample_channels((base[f"thick{support['source']}"],),
                                                   support["source_thickness_grid"])
                expected = original(sampled, A_nuc=getattr(kernel.model, f"A_{support['source']}"))
                if support["source_overlap"]:
                    expected = expected[:, None].expand(-1, stop-start, -1, -1).reshape_as(probability)
                    torch.testing.assert_close(probability, expected, rtol=0, atol=0)
                else:
                    assert probability is None
                    assert torch.all(expected == 1)
    assert len(counts) == sum(s["source_overlap"] for row in kernel._supports for s in row)


def test_late_projection_production_and_diagnostic_agree_and_preserve_batches(config, monkeypatch):
    cfg, case = resolved(config)
    kernel = ShapeScanFastV2Refiner(cfg, case)
    r1, r2 = rotations()
    project, calls = kernel._project_intensity_vertices, []

    def count(value):
        calls.append(tuple(value.shape))
        return project(value)

    monkeypatch.setattr(kernel, "_project_intensity_vertices", count)
    production = kernel.calculate(r1, r2)
    assert len(calls) == 1
    calls.clear()
    diagnostic = kernel.calculate(r1, r2, return_nodes=True)
    assert len(calls) == len(kernel.b_host)*2+1
    np.testing.assert_array_equal(production["A0"], diagnostic["A0"])
    expected = np.einsum("lb,y,boysr->olsr", diagnostic["b_level_weights"],
                         diagnostic["y_weights"], diagnostic["b_node_A0"])
    assert_close(diagnostic["A0"], expected)
    singles = [kernel.calculate(a, b)["A0"] for a, b in zip(r1, r2)]
    assert_close(production["A0"], np.concatenate(singles))
    cfg = deepcopy(cfg); cfg["study"]["y_batch_size"] = 3
    one_y_batch = ShapeScanFastV2Refiner(cfg, case).calculate(r1, r2)
    assert_close(one_y_batch["A0"], production["A0"])


def test_spherical_source_cancellation_and_failed_node_restore_geometry(config, monkeypatch):
    cfg, case = resolved(config, spherical=True)
    kernel = ShapeScanFastV2Refiner(cfg, case)
    result = kernel.calculate(np.eye(3), np.eye(3), return_nodes=True)
    middle = np.argmin(abs(result["y_values"]))
    full, self_terms = result["b_node_A0"][:, 0, middle, :, 0].T
    assert np.all(self_terms > 0)
    assert np.all(full < 1e-6*self_terms)
    original_b = kernel.model.b_imp

    def fail(*_):
        raise RuntimeError("intentional node failure")

    monkeypatch.setattr(kernel, "_node_intensity", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        kernel.calculate(np.eye(3), np.eye(3))
    assert kernel.model.b_imp == original_b


def test_nonfinite_field_outside_support_is_not_hidden(config, monkeypatch):
    cfg, case = resolved(config)
    kernel = ShapeScanFastV2Refiner(cfg, case)
    original = kernel.model._compute_photon_field

    def corrupt(*args, **kwargs):
        x, y = original(*args, **kwargs)
        x[:, 0, 0] = complex(float("nan"), 0.)
        return x, y

    monkeypatch.setattr(kernel.model, "_compute_photon_field", corrupt)
    with pytest.raises(ValueError, match="nonfinite prepared"):
        kernel.calculate(np.eye(3), np.eye(3))


def test_production_support_keeps_quarter_shift_stencil_halo():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/shape_scan_a100.yaml").read_text())
    cfg["study"].update(b_min_fm=12., b_max_fm=12.5, b_step_fm=.5)
    cfg, case = resolved(cfg)
    kernel = ShapeScanFastV2Refiner(cfg, case)
    shapes = {item["shape"] for row in kernel._supports for item in row}
    assert (48, 48) in shapes and (49, 48) in shapes
    assert max(h*w for h, w in shapes) == 49*48


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable on this host")
def test_cuda_support_and_saved_a0_match_original(config):
    cfg, case = resolved(config)
    r1, r2 = rotations()
    legacy = DiffractionRefiner(cfg, case, device="cuda").calculate(r1, r2, return_nodes=True)
    actual = ShapeScanFastV2Refiner(cfg, case, device="cuda").calculate(r1, r2, return_nodes=True)
    a0, nodes = legacy_saved(legacy, cfg["study"]["qmax"])
    assert_close(actual["A0"], a0, rtol=2e-10)
    assert_close(actual["b_node_A0"], nodes, rtol=2e-10)
