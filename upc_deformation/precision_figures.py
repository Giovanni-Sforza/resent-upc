"""Precision-campaign figures with counts/settings drawn from actual inputs.

Contract: quantitative grids distinguish the angular response, b-quadrature
changes and finite-orientation convergence. No plot asserts convergence from
sample count alone. Python/matplotlib; editable PDF/SVG and 300-dpi PNG.
"""
from pathlib import Path

import numpy as np

from .panel_d_ratio_figures import _style, _save

COLORS = {'spherical': '#656565', 'beta2': '#26749d', 'beta3': '#8670a4', 'skin': '#bc6937'}


def label(data, name):
    c = data['cases'][name]['case']
    if name == 'spherical': return 'Spherical'
    if name == 'skin': return rf'$\Delta r_{{np}}={c["skin_fm"]:g}$ fm'
    k = 2 if name == 'beta2' else 3
    return rf'$\beta_{k}={c[f"beta{k}"]:g}$'


def decorate(ax, ylabel, qmax, high=False):
    ax.set(xlabel=r'$p_T$ (GeV/$c$)', ylabel=ylabel, xlim=(.10 if high else 0, qmax))
    ax.grid(color='#e7e7e7', lw=.55)
    ax.axhline(0, color='#aaaaaa', lw=.65)


def footer(fig, data, extra):
    cfg = data['config']; s = cfg['study']
    n = data['cases']['beta2']['n_orientations']
    text = (rf'$n={n}$ pairs per deformation; $b={s["b_min_fm"]:g}$–${s["b_max_fm"]:g}$ fm, '
            rf'$\Delta b={data["b_steps_fm"][0]:g}$ fm; $|y|<{s["ymax"]:g}$; '
            + str(cfg.get('runtime', {}).get('dtype', '')) + '.\n' + extra)
    note = cfg.get('analysis', {}).get('figure_note', 'Ideal coherent signal; detector and model uncertainties are not included.')
    fig.text(.10, .025, text+'\n'+note, fontsize=7, va='bottom', linespacing=1.4)


def render_precision_figures(data, destination):
    destination = Path(destination); plt = _style()
    q = data['q']; m = q > 0; files = []
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6))
    fig.subplots_adjust(left=.10, right=.98, bottom=.25, top=.84, wspace=.29)
    for name, row in data['cases'].items():
        color = COLORS[name]; err = row['orientation_se']
        axes[0].plot(q[m], row['C2'][m], color=color, label=label(data, name), ls='--' if name == 'spherical' else '-')
        if name != 'spherical':
            axes[0].fill_between(q[m], (row['C2']-err)[m], (row['C2']+err)[m], color=color, alpha=.13, lw=0)
            axes[1].plot(q[m], 1e3*row['delta_C2'][m], color=color)
            axes[1].fill_between(q[m], 1e3*(row['delta_C2']-err)[m], 1e3*(row['delta_C2']+err)[m], color=color, alpha=.13, lw=0)
    decorate(axes[0], r'$C_2(p_T)$', q[-1]); decorate(axes[1], r'$10^3\,[C_2-C_{2,\rm sph}]$', q[-1], True)
    axes[0].set_title('a   Angular moment'); axes[1].set_title('b   Higher-momentum structural response')
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc='upper center', bbox_to_anchor=(.53, .98))
    footer(fig, data, 'Bands: pointwise orientation-integration 1 SE. Sphere and skin have no orientation sampling error.')
    files += _save(fig, destination, 'C2_precision', plt)

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 4.6), sharey=True)
    fig.subplots_adjust(left=.10, right=.98, bottom=.26, top=.82, wspace=.16)
    styles = ['-', '--', ':', '-.']
    for ax, name in zip(axes, ('beta2', 'beta3', 'skin')):
        for j, entry in enumerate(data['grid'][name]):
            delta = entry['C2']-data['grid']['spherical'][j]['C2']
            ax.plot(q[m], 1e3*delta[m], color=COLORS[name], ls=styles[j % len(styles)],
                    lw=1.7 if j == 0 else 1.1, label=rf'$\Delta b={entry["b_step_fm"]:g}$ fm')
        row = data['cases'][name]
        ax.fill_between(q[m], 1e3*(row['delta_C2']-row['orientation_se'])[m],
                        1e3*(row['delta_C2']+row['orientation_se'])[m], color=COLORS[name], alpha=.12, lw=0)
        decorate(ax, '', q[-1], True); ax.set_title(label(data, name))
    axes[0].set_ylabel(r'$10^3\,[C_2-C_{2,\rm sph}]$')
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=len(data['b_steps_fm']), loc='upper center', bbox_to_anchor=(.54, .975))
    footer(fig, data, 'All b grids share orientations and endpoints. Shading: finest-grid orientation 1 SE.\nGrid differences are diagnostics, not strict error bounds; paired differences are saved in the CSV.')
    files += _save(fig, destination, 'C2_b_grid_comparison', plt)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.5), sharex=True)
    fig.subplots_adjust(left=.10, right=.98, bottom=.19, top=.91, wspace=.30, hspace=.37)
    for i, name in enumerate(('beta2', 'beta3')):
        row = data['cases'][name]; base = data['cases']['spherical']['C2']; n = row['n_orientations']
        nested = row.get('nested', {str(n): row})
        for j, (count, e) in enumerate(nested.items()):
            last = int(count) == n
            axes[i, 0].plot(q[m], 1e3*(e['C2']-base)[m], label=f'n={count}',
                            color=COLORS[name] if last else plt.cm.Greys(.3+.5*j/max(1, len(nested)-1)), lw=1.7 if last else .85)
        for j, e in enumerate(row.get('halves', [row])):
            value, err = 1e3*(e['C2']-base), 1e3*e['orientation_se']
            color = COLORS[name] if j == 0 else '#666666'
            axes[i, 1].plot(q[m], value[m], color=color, ls='-' if j == 0 else '--', label=f'Half {j+1}: n={e["n_orientations"]}')
            axes[i, 1].fill_between(q[m], (value-err)[m], (value+err)[m], color=color, alpha=.12, lw=0)
        for j, title in enumerate(('Nested prefixes', 'Disjoint half samples')):
            decorate(axes[i, j], r'$10^3\,[C_2-C_{2,\rm sph}]$', q[-1], True)
            axes[i, j].set_title(label(data, name)+' — '+title)
            axes[i, j].legend(fontsize=6.5, ncol=2)
    footer(fig, data, 'Nested prefixes are correlated. Half-sample bands are each half\'s pointwise orientation 1 SE.')
    files += _save(fig, destination, 'C2_orientation_precision', plt)
    return files
