"""Lepton relative transverse momentum at Born level, with full polar acceptance.

Neglect lepton mass and the qT/M transverse boost, as in decay.py. The input
Stokes vector is integrated over the chosen J/psi qT sample at fixed true b.
"""
from __future__ import annotations
import numpy as np


def polarization(stokes):
    s = np.asarray(stokes, dtype=float)
    if s.shape != (3,) or not np.isfinite(s).all() or s[0] <= 0:
        raise ValueError('Expected a finite integrated (S0,S1,S2) with S0>0')
    p = s[1:] / s[0]
    if np.linalg.norm(p) > 1 + 1e-6:
        raise ValueError('Invalid transverse polarization matrix')
    return p


def matrix_element_weight(stokes, k_axis, mass):
    """Dimensionless old image_P/(M² Sigma), without a phase-space factor.

    The square is retained here specifically to reproduce the old display;
    points outside kT<=M/2 do not belong to the on-shell decay phase space.
    """
    p1, p2 = polarization(stokes)
    x, y = np.meshgrid(k_axis, k_axis, indexing='ij')
    return 1 - ((1+p1)*x*x + (1-p1)*y*y + 2*p2*x*y) / (2*(mass/2)**2)


def azimuth_density(stokes, psi):
    """Normalized directed azimuth psi relative to b, on [0,2pi)."""
    p1, p2 = polarization(stokes)
    return (1 - .5*(p1*np.cos(2*psi) + p2*np.sin(2*psi))) / (2*np.pi)


def kt_bin_density(stokes, mass, bins=201, quadrature=64):
    """Pixel-averaged dP/dkx dky, including both signs of rest-frame kz.

    Integrate the 1/sqrt(k0²-kx²-ky²) edge analytically in y, then by
    Gauss-Legendre in x. This yields finite probability per pixel even on
    the kinematic boundary. Do not renormalize away quadrature errors.
    """
    p1, p2 = polarization(stokes)
    if mass <= 0 or bins < 3 or quadrature < 4:
        raise ValueError('Positive mass and sufficient bins/quadrature required')
    k0 = mass/2
    a, b, c = (1+p1)/2, (1-p1)/2, p2/2
    edges = np.linspace(-k0, k0, bins+1)
    axis = (edges[:-1]+edges[1:])/2
    z, w = np.polynomial.legendre.leggauss(quadrature)
    probability = np.empty((bins, bins))
    for i in range(bins):
        half = (edges[i+1]-edges[i])/2
        x = (axis[i]+half*z)[:, None]
        r2 = np.maximum(k0*k0-x*x, 0)
        r = np.sqrt(r2)

        def primitive(y):
            y = np.clip(y[None, :], -r, r)
            root = np.sqrt(np.maximum(r2-y*y, 0))
            arc = np.arcsin(np.clip(y/r, -1, 1))
            i2 = .5*(r2*arc-y*root)
            return (1-a*x*x/k0**2)*arc - b*i2/k0**2 + 2*c*x*root/k0**2

        integral_y = primitive(edges[1:])-primitive(edges[:-1])
        probability[i] = 3/(4*np.pi*k0)*half*(w @ integral_y)
    if probability.min() < -1e-12:
        raise FloatingPointError('Negative decay-bin probability')
    return axis, edges, np.maximum(probability, 0) / (edges[1]-edges[0])**2


def add_kt_arrays(result, mass, bins=201):
    stokes = result['fixed_b_integrated_stokes']
    axis, edges, density = kt_bin_density(stokes, mass, bins)
    _, _, coarse = kt_bin_density(stokes, mass, bins, quadrature=32)
    psi = np.linspace(0, 2*np.pi, 721)
    dx = edges[1]-edges[0]
    p1, p2 = polarization(stokes)
    result.update(kt_axis=axis, kt_edges=edges, kt_density=density,
                  kt_matrix_weight=matrix_element_weight(stokes, axis, mass),
                  kt_psi=psi, kt_azimuth_density=azimuth_density(stokes, psi))
    return {'integrated_stokes':np.asarray(stokes).tolist(),
            'linear_polarization':[float(p1), float(p2)],
            'kt_total_probability':float(density.sum()*dx*dx),
            'kt_quadrature_32_vs_64_l1':float(np.abs(density-coarse).sum()*dx*dx),
            'cos2_psi_relative_to_b':float(-p1/4),
            'perpendicular_axis_R2':float(p1/4),
            'perpendicular_axis_within_15deg_probability':float(1/6+p1/(4*np.pi))}
