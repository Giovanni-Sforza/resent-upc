"""Analytic J/psi leptonic decay analyzer in a measured transverse axis.

Born level, transverse polarization, massless leptons, full polar acceptance,
and negligible q_T/M boost. All arrays use physical (q_x,q_y) index order.
"""
from __future__ import annotations

import math
import torch
from .vendor.upc_forward import _interp2d_batched


def stokes_maps(mx: torch.Tensor, my: torch.Tensor) -> torch.Tensor:
    """Return (...,3,N,N): S0,S1,S2, formed BEFORE grid interpolation."""
    if mx.shape != my.shape or not mx.is_complex() or not my.is_complex():
        raise ValueError('Mx and My must be complex and have matching shapes')
    xx, yy = mx.abs().square(), my.abs().square()
    return torch.stack((xx + yy, xx - yy, 2 * (mx * my.conj()).real), dim=-3) / (2 * math.pi)


def lepton_azimuth_density(stokes: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """dP/(d²q dpsi); last Stokes dimension must be length 3."""
    if stokes.shape[-1] != 3:
        raise ValueError('Last dimension must contain S0,S1,S2')
    return (stokes[..., 0] - .5 * (stokes[..., 1] * torch.cos(2 * psi)
                                  + stokes[..., 2] * torch.sin(2 * psi))) / (2 * math.pi)


def polar_stokes(stokes: torch.Tensor, p_axis: torch.Tensor,
                 radii: torch.Tensor, n_phi: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample real bilinears on a circle; do not interpolate complex amplitudes."""
    if stokes.shape[-3] != 3 or stokes.shape[-2:] != (len(p_axis), len(p_axis)):
        raise ValueError('Expected (...,3,N,N) matching p_axis')
    if n_phi < 8 or n_phi % 4:
        raise ValueError('n_phi must be a multiple of four, at least 8')
    if radii[0] < 0 or radii[-1] > torch.minimum(-p_axis[0], p_axis[-1]):
        raise ValueError('Requested circles exceed the FFT grid')
    phi = torch.arange(n_phi, dtype=stokes.dtype, device=stokes.device) * (2 * math.pi / n_phi)
    coords = torch.stack((radii[:, None] * phi.cos(), radii[:, None] * phi.sin()), -1)
    leading = stokes.shape[:-3]
    flat = stokes.reshape(-1, *stokes.shape[-2:])
    queries = coords.reshape(1, -1, 2).expand(len(flat), -1, -1)
    sampled = _interp2d_batched(flat, queries, float(p_axis[0]), float(p_axis[-1]),
                              float(p_axis[0]), float(p_axis[-1]))
    return sampled.reshape(*leading, 3, len(radii), n_phi), phi


def decay_coefficients(stokes: torch.Tensor, p_axis: torch.Tensor,
                       radii: torch.Tensor, n_phi: int = 512) -> torch.Tensor:
    """Return (...,3,n_r) = A0,C,D for I=A0-.5(C cos2alpha+D sin2alpha)."""
    samples, phi = polar_stokes(stokes, p_axis, radii, n_phi)
    s0, s1, s2 = samples.unbind(-3)
    return torch.stack((s0.mean(-1),
                        (s1 * (2 * phi).cos() + s2 * (2 * phi).sin()).mean(-1),
                        (s1 * (2 * phi).sin() - s2 * (2 * phi).cos()).mean(-1)), -2)


def interpolate_radial(values: torch.Tensor, radii: torch.Tensor,
                       query: torch.Tensor) -> torch.Tensor:
    if len(radii) < 2 or not bool(torch.all(radii[1:] > radii[:-1])):
        raise ValueError('radii must be strictly increasing')
    if float(query.max()) > float(radii[-1]) + 1e-6 or float(query.min()) < float(radii[0]):
        raise ValueError('Radial query outside the computed range')
    index = torch.searchsorted(radii, query.contiguous()).clamp(1, len(radii) - 1)
    fraction = (query - radii[index - 1]) / (radii[index] - radii[index - 1])
    return values[..., index - 1] * (1 - fraction) + values[..., index] * fraction


def decay_axis_image(coefficients: torch.Tensor, radii: torch.Tensor,
                     axis: torch.Tensor, reference: str = 'lepton') -> torch.Tensor:
    """Measured-axis image; perpendicular reference is exactly a 90 degree change."""
    if reference not in ('lepton', 'perpendicular'):
        raise ValueError('reference must be lepton or perpendicular')
    x, y = torch.meshgrid(axis, axis, indexing='ij')
    r, alpha = (x.square() + y.square()).sqrt(), torch.atan2(y, x)
    a0, c, d = interpolate_radial(coefficients, radii, r).unbind(-3)
    sign = -1 if reference == 'lepton' else 1
    return a0 + sign * .5 * (c * (2 * alpha).cos() + d * (2 * alpha).sin())


def b_quadrature(b_values: torch.Tensor) -> torch.Tensor:
    """Unnormalized trapezoid 2pi b db weights, applied to intensities."""
    if b_values.ndim != 1 or len(b_values) < 2 or not bool(torch.all(b_values > 0)):
        raise ValueError('At least two positive b nodes required')
    gaps = b_values[1:] - b_values[:-1]
    if not bool(torch.all(gaps > 0)):
        raise ValueError('b nodes must be strictly increasing')
    widths = torch.cat((gaps[:1] / 2, (gaps[:-1] + gaps[1:]) / 2, gaps[-1:] / 2))
    return 2 * math.pi * b_values * widths


def normalize_in_disk(image: torch.Tensor, axis: torch.Tensor,
                      radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = torch.meshgrid(axis, axis, indexing='ij')
    disk = x.square() + y.square() <= radius ** 2
    total = (image * disk).sum(dim=(-2, -1)) * (axis[1] - axis[0]).square()
    if not bool(torch.all(total > 0)) or not bool(torch.isfinite(image).all()):
        raise ValueError('Invalid image yield')
    return image / total[..., None, None], total
