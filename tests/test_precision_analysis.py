import numpy as np
import pytest

from upc_deformation.precision_analysis import PixelMomentProjector, paired_ratio_difference
from upc_deformation.c2_convergence import project_orientation_moments


def example():
    r = np.linspace(0, .3, 75)
    axis = np.linspace(-.2, .2, 65)
    q = np.linspace(0, .17, 40)
    angles = np.arange(80)*2*np.pi/80
    rng = np.random.default_rng(8023)
    a0 = np.exp(-r**2/.014)[None, :]*(1+rng.uniform(-.1, .1, (8, len(r))))
    c = np.stack((a0, .2*a0*rng.normal(size=(8, 1)), .1*a0*rng.normal(size=(8, 1))), axis=1)
    return PixelMomentProjector(r, axis, q, angles), c


def test_sparse_projection_is_the_same_pixel_observable():
    projector, c = example()
    got = projector.project(c, chunk_size=3)
    expected = project_orientation_moments(c, projector.radii, projector.axis, projector.q, projector.angles)
    for key in got:
        np.testing.assert_allclose(got[key], expected[key], rtol=2e-12, atol=1e-15)
    assert max(projector.check_direct(c).values()) < 2e-12


def test_paired_covariance_cancels_identical_numerical_fluctuations():
    rng = np.random.default_rng(21)
    den = rng.uniform(1., 3., (32, 5))
    num = rng.normal(0, .1, (32, 5))
    same = paired_ratio_difference(num, den, num, den)
    np.testing.assert_array_equal(same['value'], 0)
    np.testing.assert_array_equal(same['orientation_se'], 0)
    shifted = paired_ratio_difference(num+.02*den, den, num, den)
    np.testing.assert_allclose(shifted['value'], .02, atol=1e-16)
    assert shifted['orientation_se'].max() < 1e-16


def test_projection_rejects_negative_angular_density():
    projector, c = example()
    c[:, 1] = 3*c[:, 0]
    with pytest.raises(ValueError, match='negative angular'):
        projector.project(c)


def test_projection_requires_radial_support_at_square_corners():
    with pytest.raises(ValueError, match='Cartesian corner'):
        PixelMomentProjector(np.linspace(0, .2, 32), np.linspace(-.2, .2, 33),
                             np.linspace(0, .17, 10), np.arange(32)*2*np.pi/32)
