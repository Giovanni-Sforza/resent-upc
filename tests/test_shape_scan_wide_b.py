"""Area, isolated photon field, and two-source interference for the wide b scan."""
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.special import k1

from upc_deformation.shape_scan_physics import build_cases
from upc_deformation.shape_scan_wide_b import DEFAULT_CUTOFFS, WideBRefiner, impact_grid


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(deformed=False):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/shape_scan_smoke.yaml").read_text())
    cfg["study"].update(b_step_fm=4., b_max_fm=20., n_radial=41, n_phi=16,
                        n_y=3 if deformed else 1)
    case = next(c for c in build_cases(cfg) if c["density_type"] == "none"
                and c["beta2"] == (.3 if deformed else 0.)
                and c["beta3"] == (.2 if deformed else 0.))
    return WideBRefiner(cfg, case, cutoffs=[20., 28.])


def test_all_cumulative_areas_and_cutoff_seams():
    nodes, weights = impact_grid(12., DEFAULT_CUTOFFS, .5, 1.)
    for upper, row in zip(DEFAULT_CUTOFFS, weights):
        selected = nodes <= upper
        grid = nodes[selected]
        assert grid[-1] == upper
        np.testing.assert_array_equal(row[~selected], 0.)
        assert row.sum() == pytest.approx(np.pi*(upper**2 - 12.**2), rel=1e-14)
        # A manufactured nonconstant integrand tests all endpoint half-weights.
        integrand = (1. + np.cos(grid/8.)) / grid**2
        integrand_times_area = 2*np.pi*grid*integrand
        expected = np.sum(np.diff(grid)*(integrand_times_area[:-1] + integrand_times_area[1:])/2)
        assert row[selected] @ integrand == pytest.approx(expected, rel=1e-14)
        standalone = impact_grid(12., [upper], .5, 1.)
        np.testing.assert_array_equal(standalone[0], grid)
        np.testing.assert_array_equal(standalone[1][0], row[selected])


def test_translations_on_actual_polar_angles_with_complex_cross_term():
    kernel = example()
    axis = kernel.model._p_full_q.numpy()
    ix = kernel.local._projection_vertices.numpy() // len(axis)
    kx = axis[ix]
    b = np.array([17., 451.])
    a = np.zeros((1, len(b), 2, len(kx)), dtype=np.complex128)
    c = np.zeros_like(a)
    a[:, :, 0] = 1.
    c[:, :, 0] = -.7 + 12j*kx
    a[:, :, 1] = .3j
    c[:, :, 1] = -.1j
    actual = kernel._radial_intensities(torch.from_numpy(a), torch.from_numpy(c),
                                        torch.from_numpy(b)).numpy()[0]

    phi = 2*np.pi*np.arange(kernel.local.n_phi)/kernel.local.n_phi
    x = kernel.stored_q[:, None]*np.cos(phi)
    # The analytic interference phase applies at the requested x, including
    # when the phase has varied by more than pi over a native-grid interval.
    right = np.searchsorted(axis, x)
    w = (x - axis[right-1])/(axis[right] - axis[right-1])
    for ib, separation in enumerate(b):
        translation = np.exp(-1j*x*separation/kernel.model.hbar_c)
        for branch in range(2):
            def intensity(i):
                first = 1.0
                second = -.7 + 12j*axis[i]
                if branch == 0:
                    return abs(first*translation + second)**2 + abs(.3j*translation - .1j)**2
                return abs(first)**2 + abs(second)**2 + .3**2 + .1**2

            expected = ((1-w)*intensity(right-1) + w*intensity(right)).mean(-1)/(2*np.pi)
            np.testing.assert_allclose(actual[ib, branch], expected, atol=2e-14, rtol=3e-13)


def test_spherical_exterior_field_and_midrapidity_dark_fringe():
    kernel = example()
    m = kernel.model
    rotation = kernel.local._rotation(np.eye(3))
    center = int(torch.argmin(kernel.patch.abs()))
    assert kernel.patch[center] == 0.
    for nid, sign in ((1, -1), (2, 1)):
        proton, total = m._compute_density_ws(nid, rotation, batch_size=1)
        _, _, field, energy = kernel._source_and_target(nid, proton, total, 0.)
        q = energy/getattr(m, f"gamma_{nid}")
        kappa = q/m.hbar_c
        factor = np.sqrt(4*m.alpha*getattr(m, f"gamma_{nid}")*getattr(m, f"Z_{nid}")**2/q)
        # Outside an isolated spherical charge the Coulomb field equals that
        # of a point charge. Its longitudinal Fourier component is kappa*K1.
        expected = -1j*sign*factor*kappa*k1(kappa*kernel.b_nodes)/(2*np.pi)
        value = field[0, :, :, center, center].numpy()
        np.testing.assert_allclose(value[:, 0]/expected, 1., rtol=.006, atol=0.)
        assert max(abs(value[:, 1]/expected)) < 1e-6
    levels = kernel.calculate(rotation, rotation)[0]
    assert np.all(levels[:, 1, 0] > 0)
    assert np.all(levels[:, 0, 0] < 1e-12*levels[:, 1, 0])
    assert np.all(np.max(levels[:, 0], axis=-1) > 0)


def test_mixed_deformation_pairs_batch_and_cutoff_order():
    kernel = example(deformed=True)
    c, s = np.cos(.74), np.sin(.74)
    tilted = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    first = np.stack((np.eye(3), tilted))
    second = np.stack((np.eye(3), tilted.T))
    batched = kernel.calculate(first, second)
    singles = np.concatenate([kernel.calculate(r1, r2) for r1, r2 in zip(first, second)])
    np.testing.assert_allclose(batched, singles, rtol=3e-12, atol=1e-12)
    assert np.all(batched[:, 1] >= batched[:, 0])
    assert not np.allclose(batched[0], batched[1], rtol=1e-4)
