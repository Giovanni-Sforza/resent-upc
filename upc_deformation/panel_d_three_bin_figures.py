"""Plot direct panel-d C2 in three common momentum intervals.

The boundaries are the two sign reversals of the skin delta-C2 curve.  Every
structure uses those same boundaries.  The heatmap explains the bin choice;
the lower panels quantify absolute C2 and its change relative to the sphere.
Uncertainty bars, when supplied, describe the finite orientation integral.
They are not experimental errors or a numerical-convergence certificate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .panel_d_ratio_figures import _cmap, _style


CASES = ('spherical', 'beta2', 'beta3', 'skin')
COLORS = {'spherical': '#626262', 'beta2': '#26749d',
          'beta3': '#8670a4', 'skin': '#bc6937'}
MARKERS = {'spherical': 'o', 'beta2': 's', 'beta3': '^', 'skin': 'D'}


def _label(name: str, row: dict[str, Any]) -> str:
    meta = row['case']
    if name == 'spherical':
        return 'Spherical reference'
    if name == 'beta2':
        return rf'$\beta_2={meta["beta2"]:g}$'
    if name == 'beta3':
        return rf'$\beta_3={meta["beta3"]:g}$'
    return rf'$\Delta r_{{np}}={meta["skin_fm"]:g}$ fm'


def _panel(ax, letter: str, x: float = -.11):
    ax.text(x, 1.065, letter, transform=ax.transAxes,
            fontsize=10, weight='bold', va='bottom')


def _series(ax, data, names, key, error_key, scale=1.):
    offsets = np.linspace(-.24, .24, len(names))
    handles = []
    for offset, name in zip(offsets, names):
        row = data['cases'][name]
        values = np.asarray(row[key], dtype=float)
        errors = np.asarray(row[error_key], dtype=float)
        if values.shape != (3,) or errors.shape != (3,):
            raise ValueError(f'{name}: expected three bin values and uncertainties')
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(errors)):
            raise ValueError(f'{name}: nonfinite value or orientation uncertainty')
        if np.any(errors < 0):
            raise ValueError(f'{name}: negative orientation uncertainty')
        handles.append(ax.errorbar(
            np.arange(3)+offset, scale*values, yerr=scale*errors,
            color=COLORS[name], marker=MARKERS[name], linestyle='none',
            markersize=4.8, capsize=2.3, capthick=.8, elinewidth=.9,
            label=_label(name, row), zorder=3))
    return handles


def render_three_bins(data: dict[str, Any], destination: str | Path) -> dict[str, Any]:
    """Render a discussion figure from saved distributions and computed bins."""
    from matplotlib.colors import Normalize

    for name in CASES:
        if name not in data['cases']:
            raise ValueError(f'Missing {name}')
        expected = 32 if name in ('beta2', 'beta3') else 1
        if int(data['cases'][name]['n_orientations']) != expected:
            raise ValueError(f'{name}: expected original {expected}-orientation cache')
    edges = np.asarray(data['edges_GeV'], dtype=float)
    if edges.shape != (4,) or np.any(np.diff(edges) <= 0):
        raise ValueError('Expected four increasing common bin edges')
    radii = np.asarray(data['radii'], dtype=float)
    angles = np.asarray(data['angles'], dtype=float)
    ratios = np.asarray(data['skin_angular_polar'], dtype=float)
    valid = np.asarray(data['skin_valid_polar'], dtype=bool)
    if ratios.shape != (len(radii), len(angles)) or valid.shape != ratios.shape:
        raise ValueError('Skin angular array must have shape (radius, angle)')
    if not np.isclose(angles[0], 0) or np.any(np.diff(angles) <= 0) or angles[-1] >= 2*np.pi:
        raise ValueError('Expected full-circle angle samples in [0, 2pi)')
    percent = np.ma.array(100*(ratios-1), mask=~valid | ~np.isfinite(ratios))
    if not percent.count():
        raise ValueError('No valid skin angular values for display')
    peak = float(np.max(np.abs(percent.compressed())))
    # Include every valid heatmap value in a zero-centred colour scale.
    bound = max(.1, float(np.ceil(peak*10)/10))
    plt = _style()
    fig = plt.figure(figsize=(8.55, 6.50))
    grid = fig.add_gridspec(2, 2, left=.092, right=.962, bottom=.25, top=.88,
                           hspace=.68, wspace=.34, height_ratios=(.83, 1.))
    heat = fig.add_subplot(grid[0, :])
    absolute = fig.add_subplot(grid[1, 0])
    delta = fig.add_subplot(grid[1, 1])
    norm = Normalize(-bound, bound)
    closed_degrees = np.degrees(np.r_[angles, 2*np.pi])
    closed_values = np.ma.column_stack((percent, percent[:, 0]))
    mesh = heat.pcolormesh(radii, closed_degrees, closed_values.T,
                           shading='nearest', cmap=_cmap(plt), norm=norm,
                           rasterized=True)
    heat.set(xlim=(edges[0], edges[-1]), ylim=(0, 360), yticks=(0, 90, 180, 270, 360),
             xticks=(0, .05, .10, .15, .17),
             xlabel=r'$p_T$ (GeV/$c$)', ylabel=r'$\alpha$ (degrees)')
    heat.set_facecolor('#e4e4e4')
    heat.set_title('Skin: conditional angular change at fixed momentum',
                   fontsize=9, loc='left', pad=19)
    for edge in edges[1:-1]:
        heat.axvline(edge, color='#4d4d4d', linestyle=(0, (4, 3)), linewidth=1.0)
    for roman, center in zip(('I', 'II', 'III'), .5*(edges[:-1]+edges[1:])):
        heat.text(center, 1.025, roman, transform=heat.get_xaxis_transform(),
                  ha='center', va='bottom', fontsize=8.5, weight='bold')
    _panel(heat, 'a', x=-.04)
    colorbar = fig.colorbar(mesh, ax=heat, fraction=.025, pad=.016, aspect=20,
                           ticks=np.linspace(-bound, bound, 5))
    colorbar.set_label(r'$100(R_{\rm ang}-1)$ (%)', fontsize=8)

    handles = _series(absolute, data, CASES, 'C2', 'C2_orientation_se')
    _series(delta, data, CASES[1:], 'delta_C2', 'delta_C2_orientation_se', scale=1e3)
    ticklabels = [f'{roman}\n{left:.3f}–{right:.3f}'
                  for roman, left, right in zip(('I', 'II', 'III'), edges[:-1], edges[1:])]
    for ax in (absolute, delta):
        ax.set(xlim=(-.55, 2.55), xticks=np.arange(3), xticklabels=ticklabels,
               xlabel=r'Common $p_T$ interval (GeV/$c$)')
        ax.grid(axis='y', color='#dddddd', linewidth=.5, zorder=0)
        ax.margins(y=.16)
        ax.tick_params(axis='x', length=0, pad=7)
    absolute.set(ylabel=r'$C_2=\langle\cos 2\alpha\rangle$', title='Absolute angular moment')
    delta.set(ylabel=r'$10^3\,\Delta C_2$', title='Change from the spherical reference')
    delta.axhline(0, color='#929292', linewidth=.8, zorder=1)
    _panel(absolute, 'b')
    _panel(delta, 'c')
    fig.legend(handles, [_label(name, data['cases'][name]) for name in CASES],
               loc='upper center', bbox_to_anchor=(.52, .585), ncol=4,
               columnspacing=1.5, handletextpad=.4, fontsize=8)
    fig.suptitle('Angular moments in three momentum intervals', y=.977,
                 fontsize=11, weight='normal')
    fig.text(.092, .075,
             r'Panel d: $x\parallel k_T$, $b=12$–$60$ fm, $|y|<1$. '
             'Bin edges fixed by the two skin '+r'$\Delta C_2(p_T)$'+' zero crossings (exploratory).\n'
             'Bars: 1 SE from 32 orientation pairs for each deformation; sphere and skin are deterministic.\n'
             'Moments use the original panel-d intensities; the heatmap display mask does not select events.',
             fontsize=7.0, linespacing=1.5)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for extension in ('png', 'pdf', 'svg'):
        path = destination / f'three_bin_C2.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.08)
        files.append(str(path))
    plt.close(fig)
    return {
        'files': files,
        'backend': 'matplotlib',
        'editable_vector_text': True,
        'heatmap_scale_percent': [-bound, bound],
        'heatmap_max_valid_absolute_percent': peak,
        'clipped_heatmap_values': 0,
        'common_bin_edges_GeV': edges.tolist(),
        'bin_boundary_definition': 'Two noncentral zero crossings of the skin delta-C2 curve; all structures share the same bins.',
        'uncertainty_display': '1 standard error of the finite 32-pair orientation mean for pure deformations only; no experimental uncertainty or b-grid convergence encoded.',
        'source': 'Original saved Cartesian panel d; heatmap supplied by its polar conditional-angular decomposition.',
        'display_mask_applied_to_bin_moments': False,
    }
