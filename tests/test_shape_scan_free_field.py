import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j1, k1
import torch

from upc_deformation.shape_scan_free_field import FreePhotonStrip


def direct_lattice(x, y, axis, charge, kappa):
    dx, dy = np.meshgrid(x-axis, y-axis, indexing="ij")
    radius = np.hypot(dx, dy)
    safe = np.where(radius == 0, 1, radius)
    kernel = kappa*k1(kappa*safe)/safe
    return np.array([(charge*kernel*d).sum() for d in (dx, dy)]) * (-1j/(2*np.pi))


def bilinear_direct(x, y, axis, charge, kappa):
    h = axis[1]-axis[0]
    xy = (np.array([x, y])-axis[0])/h
    base = np.floor(xy)
    wx, wy = xy-base
    return sum((wx if i else 1-wx)*(wy if j else 1-wy)
               * direct_lattice(*(axis[0]+(base+[i,j])*h), axis, charge, kappa)
               for i in (0,1) for j in (0,1))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("sign", [1, -1])
def test_complex_extended_sources_off_axis_both_signs(device, sign):
    axis = np.arange(-2., 2., .5)
    patch = np.array([-.65, .3])
    b = np.array([0., .25, 2., 12., 451., 10000.])
    kappa = .045
    rng = np.random.default_rng(11)
    charge = rng.normal(size=(2, len(axis), len(axis))) + 1j*rng.normal(size=(2,len(axis),len(axis)))
    field = FreePhotonStrip(axis, kappa, b, patch, device).sample(charge, sign).cpu().numpy()
    for ib, offset in enumerate(b):
        for a in range(2):
            for i, x in enumerate(patch):
                for j, y in enumerate(patch):
                    expected = bilinear_direct(sign*offset+x, y, axis, charge[a], kappa)
                    # Test small far-tail fields relative to their own scale.
                    np.testing.assert_allclose(field[a,ib,:,i,j], expected, rtol=5e-12,
                                               atol=2e-13*np.max(np.abs(expected)))


def test_origin_symmetry_and_kernel_normalization():
    axis = np.arange(-2.,3.)
    charge = np.zeros((1,5,5))
    charge[0,2,2] = 1
    solver = FreePhotonStrip(axis, .1, [0,5,10000], [0])
    value = solver.sample(charge, 1)[0,:,:,0,0].cpu().numpy()
    np.testing.assert_allclose(value[0], np.zeros(2), atol=5e-16)
    np.testing.assert_allclose(value[1], [-1j*.1*k1(.5)/(2*np.pi), 0], atol=5e-16)
    np.testing.assert_array_equal(value[-1], np.zeros(2))


def test_smooth_source_hankel_reference_under_spatial_refinement():
    # The Fourier transform of the unit Gaussian charge is exp(-sigma² k²/2).
    # Analytic azimuth integration leaves an independent continuum reference.
    radius, sigma, kappa = 3., 1.5, .06
    radial = quad(lambda k: k*k*j1(k*radius)/(k*k+kappa*kappa)
                  * np.exp(-.5*(sigma*k)**2), 0, 14)[0]
    losses = []
    for h in (.5, .25):
        axis = np.arange(-8., 8.+h/2, h)
        charge = np.exp(-.5*(axis[:,None]**2+axis[None,:]**2)/sigma**2)
        charge /= charge.sum()
        field = FreePhotonStrip(axis, kappa, [radius], [0]).sample(charge[None], 1)
        computed = -2*np.pi*field[0,0,0,0,0].imag.item()
        losses.append(abs(computed/radial - 1))
    assert losses[1] < .002
    assert losses[1] < .35*losses[0]
