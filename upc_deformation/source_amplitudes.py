"""Separate two-source interference from the single-source diffraction terms.

This is a mechanism diagnostic, not an experimentally selectable event sample.
The ``self`` result retains coherent scattering within each nucleus and removes
only the cross term between the two photon-emission directions.  It is unrelated
to incoherent nuclear breakup or a Good--Walker fluctuation variance.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from .decay import stokes_maps
from .forward import UPCForwardModel


def source_stokes(
    model: UPCForwardModel,
    y_views: torch.Tensor,
    b_values: Sequence[float],
    R_1: torch.Tensor | None = None,
    R_2: torch.Tensor | None = None,
    *,
    return_incoherent: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Return full and source-self Stokes maps on the native FFT momentum grid.

    ``y_views`` has shape ``(orientation,n_y)`` and optional ``R_1``/``R_2``
    have shape ``(orientation,3,3)``, with the same rotation convention as the
    model.  Returned maps have shape ``(orientation,n_b,n_y,3,N,N)``; the last
    two axes are physical ``(q_x,q_y)`` and the final return value is their
    shared momentum axis in GeV.  Stokes order is ``S0,S1,S2``, normalized as
    in :func:`upc_deformation.decay.stokes_maps`.

    For the two complex transverse amplitudes ``M21`` and ``M12``, full is
    ``Stokes(M21 + M12)`` and self is ``Stokes(M21) + Stokes(M12)``.  Their
    difference isolates the interference bilinears, including polarization.
    No orientation, b, or rapidity average and no independent normalization
    is applied.  Form bilinears here before interpolation or ensemble sums.

    Woods--Saxon densities are constructed by the existing forward model;
    directional amplitudes come from its existing density-response API.  The
    API also calculates a legacy empirical incoherent comparison.  With
    ``return_incoherent=True``, return this as a fourth value on the output
    grid, shape ``(orientation,n_b,n_y,Nout,Nout)``; it is the sum of the two
    branch yields and is never included in the coherent Stokes maps.  Its
    momentum axis is ``model._p_axis_q``, not the returned native FFT axis.
    The calling analysis must state its assumed polarization and angular
    distribution if adding this empirical yield to decay observables.
    The model's b setting is restored by the existing response API.
    """
    if not isinstance(y_views, torch.Tensor) or y_views.ndim != 2:
        raise ValueError("y_views must be a (orientation,n_y) tensor")
    if not y_views.is_floating_point() or not y_views.numel():
        raise ValueError("y_views must be nonempty and floating point")
    if not bool(torch.isfinite(y_views).all()):
        raise ValueError("y_views must contain only finite rapidities")
    batch = y_views.shape[0]
    for name, rotation in (("R_1", R_1), ("R_2", R_2)):
        if rotation is not None:
            if rotation.shape != (batch, 3, 3):
                raise ValueError(f"{name} must have shape ({batch},3,3)")
            if not rotation.is_floating_point() or not bool(torch.isfinite(rotation).all()):
                raise ValueError(f"{name} must contain finite floating-point rotations")
    if any(model._nuclear_configs[i]["density_model"] != "woods_saxon" for i in (1, 2)):
        raise ValueError("source_stokes currently supports Woods--Saxon nuclei only")

    rho_p_1, rho_total_1 = model._compute_density_ws(nuclear_ID=1, R=R_1, batch_size=batch)
    rho_p_2, rho_total_2 = model._compute_density_ws(nuclear_ID=2, R=R_2, batch_size=batch)
    response = model.fixed_b_branch_response_multi_y_from_density(
        rho_p_1, rho_total_1, y_views, b_values,
        rho_p_2=rho_p_2, rho_total_2=rho_total_2,
    )
    m21x, m21y = response.amplitude_2_to_1_x, response.amplitude_2_to_1_y
    m12x, m12y = response.amplitude_1_to_2_x, response.amplitude_1_to_2_y
    full = stokes_maps(m21x + m12x, m21y + m12y)
    self_terms = stokes_maps(m21x, m21y) + stokes_maps(m12x, m12y)
    if return_incoherent:
        empirical = response.empirical_incoherent_2_to_1 + response.empirical_incoherent_1_to_2
        return full, self_terms, response.p_full_GeV, empirical
    return full, self_terms, response.p_full_GeV
