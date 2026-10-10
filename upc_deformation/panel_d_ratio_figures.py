"""Display radial and conditional-angular factors of saved panel-d ratios.

Figure contract: reveal whole-ring redistribution and same-radius directionality
in the exact cached 32-orientation panel-d sample, without naming either factor
as a unique physical mechanism.  The two figures are image plates plus numeric
summaries, drawn solely with matplotlib.  They consume the numerical module's
linear arrays and masks; no forward model, response simulation, or image-derived
measurement occurs here.  Every display scale includes all valid displayed
extrema and is shared across the three structures within its factor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


CASES = ('beta2', 'beta3', 'skin')
COLORS = {'beta2': '#26749d', 'beta3': '#8670a4', 'skin': '#bc6937'}
GRAY = '#e4e4e4'


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 8,
        'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.linewidth': .7, 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.facecolor': 'white',
        'savefig.facecolor': 'white', 'lines.linewidth': 1.6,
    })
    return plt


def _save(fig, destination: Path, stem: str, plt):
    files = []
    for ext in ('png', 'pdf', 'svg'):
        path = destination / f'{stem}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.08)
        files.append(str(path))
    plt.close(fig)
    return files


def _case_label(row):
    meta = row['case']
    if meta['name'].startswith('beta2'):
        return rf'$\beta_2={meta["beta2"]:g}$'
    if meta['name'].startswith('beta3'):
        return rf'$\beta_3={meta["beta3"]:g}$'
    return rf'$\Delta r_{{np}}={meta["skin_fm"]:g}$ fm'


def _title(row):
    n = int(row['n_orientations'])
    return _case_label(row) + ('\n32 orientation pairs' if n == 32 else
                               '\nDeterministic spherical density' if n == 1 else
                               f'\n{n} orientation pairs')


def _panel(ax, letter, x=-.13):
    ax.text(x, 1.11, letter, transform=ax.transAxes, weight='bold', fontsize=10)


def _percent(values, mask=None):
    values = 100*(np.asarray(values, dtype=float)-1)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return np.ma.array(values, mask=~valid)


def _valid(data, row, domain):
    """Use the numerical module's masks rather than recover a mask from colors."""
    common = np.asarray(data['mask_'+domain], dtype=bool)
    local = row.get('valid_'+domain, row.get('mask_'+domain))
    return common if local is None else common & np.asarray(local, dtype=bool)


def _range(arrays):
    values = [np.asarray(a.compressed(), dtype=float) for a in arrays if a.count()]
    if not values:
        raise ValueError('No finite data inside the supplied display masks')
    peak = max(float(np.max(np.abs(a))) for a in values)
    if peak == 0:
        return 1., 0.
    # Round upward, never clip an observed valid extremum or select a quantile.
    magnitude = 10**np.floor(np.log10(peak))
    bound = float(np.ceil(peak / magnitude * 2) / 2 * magnitude)
    return bound, peak


def _cmap(plt):
    from matplotlib.colors import LinearSegmentedColormap
    # An odd-length lookup table puts exact white at the zero midpoint.
    cmap = LinearSegmentedColormap.from_list('ratio_blue_white_red',
                                             ['#2166ac', '#ffffff', '#b2182b'], N=257)
    cmap.set_bad(GRAY)
    return cmap


def _xy(ax, axis, values, norm, cmap, qmax, y_label):
    from matplotlib.patches import Circle
    artist = ax.imshow(values.T, extent=[axis[0], axis[-1], axis[0], axis[-1]],
                       origin='lower', cmap=cmap, norm=norm, interpolation='nearest')
    ax.add_patch(Circle((0, 0), qmax, fill=False, edgecolor='#777777',
                        linestyle=':', linewidth=.65))
    ax.set_aspect('equal')
    ax.set_xticks([-.15, 0, .15])
    ax.set_yticks([-.15, 0, .15])
    ax.set_xlabel(r'$q_x$ (GeV/$c$)')
    if y_label:
        ax.set_ylabel(r'$q_y$ (GeV/$c$)')
    else:
        ax.set_yticklabels([])
    ax.set_facecolor(GRAY)
    return artist


def _footer(data):
    cfg = data['config']
    return (
        r'Panel d: $x\parallel k_T$, $b=12$–$60$ fm, $\Delta b=2$ fm, $|y|<1$. '
        'Original 32-orientation sample for each pure deformation.\n'
        'Center estimates only; no orientation-error bars. High-q b-grid sensitivity is unresolved; algebraic closure is not physical convergence.\n'
        rf'Gray: $q_T>{cfg["qmax"]:g}$ GeV/$c$ or baseline below {100*cfg["floor_fraction"]:g}% of its peak '
        '(plus undefined factors); this is a display mask.\n'
        'Factors describe whole-ring and conditional angular changes; radial position alone does not identify interference or diffraction.'
    )


def _radial(data, destination, plt):
    from matplotlib.colors import Normalize
    axis = np.asarray(data['axis'], dtype=float)
    radii = np.asarray(data['radii'], dtype=float)
    qmax = float(data['config']['qmax'])
    maps = [_percent(data['cases'][name]['radial_map'], _valid(data, data['cases'][name], 'cart'))
            for name in CASES]
    curves = [_percent(data['cases'][name]['radial_ratio'], data['cases'][name].get('radial_valid'))
              for name in CASES]
    bound, peak = _range(maps+curves)
    norm, cmap = Normalize(-bound, bound), _cmap(plt)
    fig = plt.figure(figsize=(8.8, 6.6))
    grid = fig.add_gridspec(2, 3, left=.075, right=.90, bottom=.23, top=.83,
                           hspace=.48, wspace=.22, height_ratios=(1.0, .86))
    top = [fig.add_subplot(grid[0, i]) for i in range(3)]
    curve_ax = fig.add_subplot(grid[1, :])
    for i, (name, ax, values) in enumerate(zip(CASES, top, maps)):
        artist = _xy(ax, axis, values, norm, cmap, qmax, i == 0)
        ax.set_title(_title(data['cases'][name]), fontsize=8.5, pad=8)
        _panel(ax, chr(97+i))
    pos = top[-1].get_position()
    cax = fig.add_axes([.918, pos.y0, .014, pos.height])
    fig.colorbar(artist, cax=cax, label=r'$100(R_{\rm rad}-1)$ (%)',
                 ticks=np.linspace(-bound, bound, 5))
    for name, values in zip(CASES, curves):
        curve_ax.plot(radii, values, color=COLORS[name], label=_case_label(data['cases'][name]))
    curve_ax.axhline(0, color='#a6a6a6', linewidth=.7)
    curve_ax.set(xlim=(0, qmax), ylim=(-bound, bound),
                 xlabel=r'$q_T$ (GeV/$c$)', ylabel=r'$100(R_{\rm rad}-1)$ (%)')
    curve_ax.legend(loc='best', ncol=3)
    _panel(curve_ax, 'd', x=-.025)
    fig.suptitle('Radial redistribution in the panel-d structure ratios', y=.98, fontsize=11)
    fig.text(.49, .92,
             r'$R_{\rm rad}(q_T)=J_\theta(q_T)/J_0(q_T),\quad J(q_T)=\frac{1}{2\pi}\int_0^{2\pi} I(q_T,\alpha)\,d\alpha$',
             ha='center', fontsize=9)
    fig.text(.075, .034, _footer(data) + '\n'
             'The lower curves use complete-circle integrals and the same symmetric percentage scale as the maps.',
             fontsize=6.6, linespacing=1.4)
    return _save(fig, destination, 'radial_changes', plt), {
        'symmetric_color_limit_percent': bound, 'largest_valid_absolute_change_percent': peak,
        'clipped_values': 0, 'curve_and_map_y_scale_identical': True,
    }


def _angular(data, destination, plt):
    from matplotlib.colors import Normalize
    axis = np.asarray(data['axis'], dtype=float)
    radii = np.asarray(data['radii'], dtype=float)
    angles = np.asarray(data['angles'], dtype=float)
    qmax = float(data['config']['qmax'])
    if not np.isclose(angles[0], 0) or not np.all(np.diff(angles)>0) or angles[-1] >= 2*np.pi:
        raise ValueError('Expected sorted full-circle angles in radians, [0, 2pi)')
    maps = [_percent(data['cases'][name]['angular_map'], _valid(data, data['cases'][name], 'cart'))
            for name in CASES]
    polar = [_percent(data['cases'][name]['angular_polar'], _valid(data, data['cases'][name], 'polar'))
             for name in CASES]
    bound, peak = _range(maps+polar)
    norm, cmap = Normalize(-bound, bound), _cmap(plt)
    fig = plt.figure(figsize=(8.8, 7.25))
    grid = fig.add_gridspec(2, 3, left=.075, right=.90, bottom=.245, top=.84,
                           hspace=.45, wspace=.22, height_ratios=(1., 1.05))
    axs = np.array([[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)])
    closed_degrees = np.degrees(np.r_[angles, 2*np.pi])
    for col, name in enumerate(CASES):
        ax = axs[0, col]
        artist = _xy(ax, axis, maps[col], norm, cmap, qmax, col == 0)
        ax.set_title(_title(data['cases'][name]), fontsize=8.5, pad=8)
        _panel(ax, chr(97+col))
        ax = axs[1, col]
        # Repeat only the first angular sample at 360 degrees to close the
        # periodic display. No half-circle is dropped or reflected.
        values = np.ma.column_stack((polar[col], polar[col][:, 0]))
        ax.pcolormesh(radii, closed_degrees, values.T, shading='nearest',
                      cmap=cmap, norm=norm, rasterized=True)
        ax.set(xlim=(0, qmax), ylim=(0, 360), yticks=[0, 90, 180, 270, 360],
               xticks=[0, .05, .10, .15], xlabel=r'$q_T$ (GeV/$c$)')
        ax.set_facecolor(GRAY)
        if col == 0:
            ax.set_ylabel(r'$\alpha=\phi_q-\phi_k$ (degrees)')
        else:
            ax.set_yticklabels([])
        _panel(ax, chr(100+col))
    bottom, top = axs[1, -1].get_position().y0, axs[0, -1].get_position().y1
    cax = fig.add_axes([.918, bottom, .014, top-bottom])
    fig.colorbar(artist, cax=cax, label=r'$100(R_{\rm ang}-1)$ (%)',
                 ticks=np.linspace(-bound, bound, 5))
    fig.suptitle('Conditional angular redistribution at the same momentum radius', y=.982, fontsize=11)
    fig.text(.49, .925,
             r'$R_{\rm ang}(q_T,\alpha)=\dfrac{I_\theta(q_T,\alpha)/J_\theta(q_T)}{I_0(q_T,\alpha)/J_0(q_T)},\quad R_{\rm full}=R_{\rm rad}R_{\rm ang}$',
             ha='center', fontsize=9)
    fig.text(.075, .035, _footer(data) + '\n'
             'Lower panels show the complete 0–360 degree circle; each vertical column compares directions at one fixed radius.\n'
             'Angular and radial figures use separate symmetric scales. The same angular scale is shared by all six panels here.',
             fontsize=6.6, linespacing=1.4)
    return _save(fig, destination, 'angular_changes', plt), {
        'symmetric_color_limit_percent': bound, 'largest_valid_absolute_change_percent': peak,
        'clipped_values': 0, 'angular_display_degrees': [0, 360],
        'periodic_display_closure': 'First angular sample repeated at 360 degrees; no data reflection or half-circle omission.',
    }


def render_panel_d_ratios(data: dict[str, Any], destination: str | Path) -> dict[str, Any]:
    """Render two figures from the numerical decomposition's saved arrays."""
    for name in CASES:
        if name not in data['cases']:
            raise ValueError(f'Missing required case {name}')
        expected = 1 if name == 'skin' else 32
        if int(data['cases'][name]['n_orientations']) != expected:
            raise ValueError(f'{name}: this figure contract requires the original {expected}-orientation cache')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plt = _style()
    radial_files, radial_meta = _radial(data, destination, plt)
    angular_files, angular_meta = _angular(data, destination, plt)
    return {
        'files': radial_files+angular_files, 'radial': radial_meta, 'angular': angular_meta,
        'backend': 'matplotlib', 'editable_vector_text': True,
        'sample': 'Original panel-d cached sample; 32 orientation pairs per pure-deformation case.',
        'uncertainty_display': 'Central estimates only; no orientation uncertainty evaluated or shown.',
        'claim_boundary': 'An algebraic shape decomposition, not a unique separation of physical mechanisms or a convergence certificate.',
    }
