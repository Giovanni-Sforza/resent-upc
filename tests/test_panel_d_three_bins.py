import numpy as np
import pytest

from upc_deformation.panel_d_three_bins import (
    _orientation_statistics, angular_curve, cartesian_bin_statistics,
    find_skin_boundaries, polar_bin_statistics,
)


def test_central_pixel_has_no_assigned_direction_and_annuli_partition_disk():
    axis = np.arange(-40, 41)/256
    image = np.ones((81, 81))
    image[40, 40] = 1000.
    edges = [0., .071, .113, .15]
    stats = cartesian_bin_statistics(image, axis, edges)
    np.testing.assert_allclose(stats["C2"], 0, atol=1e-15)
    np.testing.assert_allclose(stats["S2"], 0, atol=1e-15)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    disk = x*x+y*y <= edges[-1]**2
    assert stats["pixel_counts"].sum() == disk.sum()
    np.testing.assert_allclose(stats["yield"].sum(), image[disk].sum()*stats["pixel_area"])
    assert stats["yield_fraction"].sum() == pytest.approx(1.)


def test_polar_moments_use_yield_and_q_jacobian_not_uniform_c2_average():
    q = np.linspace(0, .4, 401)
    alpha = np.arange(720)*2*np.pi/720
    # J(q)=1+q, J*C2=q, J*S2=-q/2: the integrated primitives are linear.
    polar = (1+q[:, None])+2*q[:, None]*np.cos(2*alpha)-q[:, None]*np.sin(2*alpha)
    edges = np.array([0., .2, .4])
    stats = polar_bin_statistics(polar, q, alpha, edges)
    numerator = np.diff(edges**3)/3
    denominator = np.diff(edges**2)/2+np.diff(edges**3)/3
    np.testing.assert_allclose(stats["C2"], numerator/denominator, atol=1e-15)
    np.testing.assert_allclose(stats["S2"], -.5*numerator/denominator, atol=1e-15)
    uniform_mean = np.mean(q[q <= .2]/(1+q[q <= .2]))
    assert abs(stats["C2"][0]-uniform_mean) > .02


def test_cartesian_and_polar_representations_agree_for_smooth_anisotropy():
    axis = np.arange(-400, 401)*.001
    x, y = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(x, y)
    angle = np.arctan2(y, x)
    image = (1+radius)+2*radius*np.cos(2*angle)-radius*np.sin(2*angle)
    q = np.linspace(0, .4, 801)
    alpha = np.arange(720)*2*np.pi/720
    polar = (1+q[:, None])+2*q[:, None]*np.cos(2*alpha)-q[:, None]*np.sin(2*alpha)
    edges = [0., .15, .3, .4]
    cart = cartesian_bin_statistics(image, axis, edges)
    circ = polar_bin_statistics(polar, q, alpha, edges)
    np.testing.assert_allclose(cart["C2"], circ["C2"], atol=1e-4)
    np.testing.assert_allclose(cart["S2"], circ["S2"], atol=1e-4)
    np.testing.assert_allclose(cart["yield_fraction"], circ["yield_fraction"], atol=2e-4)


def test_boundary_selection_ignores_origin_and_requires_two_crossings():
    q = np.linspace(0, .17, 341)
    roots = find_skin_boundaries(q, q*(q-.10)*(q-.15))
    np.testing.assert_allclose(roots, [.10, .15], atol=1e-15)
    with pytest.raises(ValueError, match="expected two"):
        find_skin_boundaries(q, q-.1)
    alpha = np.arange(720)*2*np.pi/720
    with pytest.raises(ValueError, match="complete"):
        angular_curve(np.full((len(q), len(alpha)), np.nan), alpha)


def test_orientation_uncertainty_uses_ratio_of_means_and_reconstructs_pixels():
    import torch
    from upc_deformation.decay import decay_axis_image

    axis, q = np.linspace(-.2, .2, 65), np.linspace(0, .3, 151)
    coeff = np.zeros((4, 3, len(q)))
    coeff[:, 0] = np.array([1., 2., 3., 4.])[:, None]
    coeff[:, 1] = np.array([0., 1., 0., 0.])[:, None]
    images = decay_axis_image(torch.from_numpy(coeff), torch.from_numpy(q), torch.from_numpy(axis)).numpy()
    mean_image = images.mean(0)
    archived = np.zeros((4, 2, len(axis), len(axis)))
    archived[3, 0] = mean_image
    retained = np.zeros((4, 3, 2, 3, len(q)))
    retained[:, -1, 0] = coeff
    result = {"axis": axis, "radii": q, "images": archived,
              "orientation_coefficients": retained, "n_orientations": 4}
    edges = [0., .1, .15, .17]
    primary = cartesian_bin_statistics(mean_image, axis, edges)
    orientation = _orientation_statistics(result, edges, primary)
    individual = cartesian_bin_statistics(images, axis, edges)
    assert abs(individual["C2"].mean(0)[1]-primary["C2"][1]) > .005
    influence = (individual["C2_numerator"]-individual["yield"]*primary["C2"])/individual["yield"].mean(0)
    np.testing.assert_allclose(orientation["C2_orientation_se"], influence.std(0, ddof=1)/2, atol=1e-15)
    assert orientation["orientation_image_relative_closure"] < 1e-14
    result["images"][3, 0, 0, 0] += .01
    with pytest.raises(ValueError, match="does not close"):
        _orientation_statistics(result, edges, primary)
