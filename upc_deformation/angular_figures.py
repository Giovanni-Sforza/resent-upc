"""Plot the finite-momentum angular bridge from saved numerical summaries.

Figure contract: test whether angular contrast outside the central momentum bin
retains the interference mechanism and a resolved nuclear-structure response.
Five quantitative grids separate mechanism, structural contrast, morphology,
track response, and orientation precision.  The figures do not identify C2 with
collective flow or infer a mechanism merely from a nonzero angular coefficient.
All drawing uses matplotlib; PDF/SVG text remains editable.  This module makes
no forward-model calls and does not infer unreported uncertainties.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


FAMILIES = ('beta2', 'beta3', 'skin')
FAMILY_COLORS = {'beta2': '#26749d', 'beta3': '#8670a4', 'skin': '#bc6937'}
BIN_COLORS = ('#31708e', '#619785', '#b58d4e', '#997a9b')
STAGE_COLORS = ('#6d7076', '#9b7447', '#327a8c')
STAGE_LABELS = ('Fixed impact parameter', 'Impact-parameter average',
                'Impact parameter + rapidity')


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'legend.fontsize': 7, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.linewidth': .7, 'axes.spines.top': False,
        'axes.spines.right': False, 'savefig.facecolor': 'white',
        'figure.facecolor': 'white', 'lines.linewidth': 1.4,
    })
    return plt


def _array(values):
    """JSON nulls become masked numerical positions, never zero data."""
    return np.asarray(values, dtype=float)


def _case_label(case):
    meta = case['case']
    if meta['name'].startswith('beta2'):
        return rf'$\beta_2={meta["beta2"]:g}$'
    if meta['name'].startswith('beta3'):
        return rf'$\beta_3={meta["beta3"]:g}$'
    if meta['name'].startswith('skin'):
        return rf'$\Delta r_{{np}}={meta["skin_fm"]:g}$ fm'
    return str(meta.get('label', meta['name']))


def _family_parameter(family):
    return 'skin_fm' if family == 'skin' else family


def _family_cases(analytic, family):
    key = _family_parameter(family)
    return sorted((v for k, v in analytic['cases'].items() if k.startswith(family)),
                  key=lambda v: v['case'][key])


def _endpoint(analytic, family):
    cases = _family_cases(analytic, family)
    if not cases:
        raise ValueError(f'Missing endpoint family: {family}')
    return cases[-1]


def _bin_labels(analytic):
    return [f'{lo:.2f}–{hi:.2f}' for lo, hi in analytic['bin_edges_GeV']]


def _panel(ax, letter):
    ax.text(-.10, 1.085, letter, transform=ax.transAxes, weight='bold', fontsize=10)


def _zero(ax):
    ax.axhline(0, color='#a6a6a6', lw=.7, zorder=0)


def _bin_axis(ax, labels):
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_xlim(-.45, len(labels)-.55)
    ax.set_xlabel(r'$q_T$ interval (GeV/$c$)')


def _selection(study):
    cfg = study.get('study', study)
    db = (float(cfg['b_max_fm']) - float(cfg['b_min_fm'])) / (int(cfg['b_nodes']) - 1)
    return (rf'Hard $b$ window: {cfg["b_min_fm"]:g}–{cfg["b_max_fm"]:g} fm, step {db:g} fm; '
            rf'$|y|<{cfg["ymax"]:g}$. No neutron-tag selection or calibrated acceptance.')


def _orientation_note(analytic):
    counts = []
    for family in FAMILIES:
        c = _endpoint(analytic, family)
        n = c.get('n_orientations')
        if n is not None:
            counts.append(f'{family}: {n}')
    return 'Orientation samples: ' + ', '.join(counts) + '.' if counts else ''


def _save(fig, destination, stem, plt):
    files = []
    for ext in ('png', 'pdf', 'svg'):
        path = destination / f'{stem}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.08)
        files.append(str(path))
    plt.close(fig)
    return files


def _finite_curve(branch, key='C2'):
    values = _array(branch['curves'][key]).copy()
    valid = np.asarray(branch['curves'].get('valid', np.isfinite(values)), dtype=bool)
    values[~valid] = np.nan
    return values


def _mechanism(analytic, study, destination, plt):
    from matplotlib.lines import Line2D
    q = _array(analytic['curve_radii_GeV'])
    base = analytic['cases']['spherical']
    stages = analytic['stage_names']
    final = stages[-1]
    bins = _array(analytic['bin_edges_GeV'])
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.9))
    fig.subplots_adjust(left=.10, right=.975, bottom=.21, top=.86,
                        wspace=.31, hspace=.47)
    for branch, ls, label in (('full', '-', 'Full'),
                             ('self', '--', 'Cross term removed')):
        axs[0, 0].plot(q, _finite_curve(base['stages'][final][branch]),
                       color='#343c45', ls=ls, label=label)
    axs[0, 0].set_title('Spherical angular correlation')
    axs[0, 0].set_ylabel(r'$C_2(q_T)=\langle\cos2\alpha\rangle$')
    axs[0, 0].legend(loc='best')
    for family in FAMILIES:
        case = _endpoint(analytic, family)
        data = case['stages'][final]['full']
        delta = _finite_curve(data) - _finite_curve(base['stages'][final]['full'])
        se = _array(data['curves']['C2_se'])
        color = FAMILY_COLORS[family]
        axs[0, 1].plot(q, 1000*delta, color=color, label=_case_label(case))
        axs[0, 1].fill_between(q, 1000*(delta-se), 1000*(delta+se),
                               color=color, alpha=.15, linewidth=0)
    axs[0, 1].set_title('Structure minus spherical, full intensity')
    axs[0, 1].set_ylabel(r'$\Delta C_2(q_T)$ ($10^{-3}$)')
    axs[0, 1].legend(loc='best')
    for stage, color, label in zip(stages, STAGE_COLORS, STAGE_LABELS):
        axs[1, 0].plot(q, _finite_curve(base['stages'][stage]['full']),
                       color=color, label=label)
    axs[1, 0].set_title('Geometry averaging, spherical baseline')
    axs[1, 0].set_ylabel(r'$C_2(q_T)$')
    axs[1, 0].legend(loc='best')
    x = np.arange(len(bins))
    frac = _array(base['stages'][final]['full']['yield_fraction'])
    bg = _array(base['background']['fraction_of_total'])
    axs[1, 1].bar(x-.15, 100*frac, width=.29, color='#547c91', label='Coherent bin / coherent disk')
    axs[1, 1].bar(x+.15, 100*bg, width=.29, color='#d7b28e',
                   label='Background / total in bin')
    axs[1, 1].set_title('Available yield and empirical background')
    axs[1, 1].set_ylabel('Fraction (%)')
    _bin_axis(axs[1, 1], _bin_labels(analytic))
    axs[1, 1].legend(loc='best', fontsize=6.6)
    # Reserve a band for the legend above both physical fractions.
    axs[1, 1].set_ylim(0, max(1., 100*max(float(frac.max()), float(bg.max())))*1.34)
    for ax in (axs[0, 0], axs[0, 1], axs[1, 0]):
        ax.axvspan(0, bins[0, 0], color='#999999', alpha=.10, lw=0)
        for edge in np.unique(bins):
            ax.axvline(edge, color='#bbbbbb', lw=.5, ls=':', zorder=0)
        _zero(ax)
        ax.set_xlim(0, bins[-1, 1])
        ax.set_xlabel(r'$q_T$ (GeV/$c$)')
    for i, ax in enumerate(axs.flat):
        _panel(ax, chr(97+i))
    fig.suptitle('Finite-momentum angular contrast: mechanism and structural response',
                 y=.97, fontsize=11)
    cfg = study.get('study', study)
    note = (r'$\alpha=\phi_q-\phi_k$; $k_T=(p_{T,+}-p_{T,-})/2$. '
            r'This lepton-axis correlation is not a collective-flow coefficient.' '\n'
            'Shading in b: orientation-integration SE; gray momentum strip: excluded from the four fixed bins.\n'
            rf'Fixed geometry in c: $b={cfg["fixed_b_fm"]:g}$ fm, $y=0$. '
            'Removing the cross term is a theoretical diagnostic, not an event selection.\n'
            'Background uses the saved empirical model; its fraction is not an acceptance-corrected experimental forecast.\n'
            + _orientation_note(analytic) + ' ' + _selection(study))
    fig.text(.10, .025, note, fontsize=6.8, linespacing=1.35)
    return _save(fig, destination, 'angular_mechanism', plt)


def _bin_response(analytic, study, destination, plt):
    from matplotlib.lines import Line2D
    labels = _bin_labels(analytic)
    x = np.arange(len(labels))
    stage = analytic['stage_names'][-1]
    fig, axs = plt.subplots(3, 2, figsize=(8.6, 8.1))
    fig.subplots_adjust(left=.10, right=.975, bottom=.17, top=.87,
                        wspace=.34, hspace=.54)
    for row, family in enumerate(FAMILIES):
        case = _endpoint(analytic, family)
        data = case['stages'][stage]
        color = FAMILY_COLORS[family]
        for branch, dx, ls, marker in (('full', -.06, '-', 'o'), ('self', .06, '--', 's')):
            delta = _array(data['contrast']['delta_'+branch])[:, 0]
            se = _array(data[branch]['se'])[:, 0]
            axs[row, 0].errorbar(x+dx, 1000*delta, yerr=1000*se,
                                  ls=ls, marker=marker, color=color, ms=4,
                                  capsize=2, elinewidth=.9)
        interaction = _array(data['contrast']['interaction'])[:, 0]
        interaction_se = _array(data['contrast']['interaction_se'])[:, 0]
        axs[row, 1].errorbar(x, 1000*interaction, yerr=1000*interaction_se,
                              color=color, marker='D', ms=3.5, capsize=2,
                              elinewidth=.9)
        for col in range(2):
            ax = axs[row, col]
            _zero(ax); _bin_axis(ax, labels)
            ax.set_title(_case_label(case), pad=9)
            ax.set_ylabel(r'$\Delta C_2$ ($10^{-3}$)' if col == 0 else
                          r'$\Delta C_2^{\rm full}-\Delta C_2^{\rm self}$ ($10^{-3}$)')
            ax.margins(y=.25)
            _panel(ax, chr(97+row*2+col))
    handles = [Line2D([0], [0], color='#444444', marker='o', label='Full intensity'),
               Line2D([0], [0], color='#444444', marker='s', ls='--', label='Cross term removed'),
               Line2D([0], [0], color='#444444', marker='D', label='Paired full–self response contrast')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .935),
               ncol=3, fontsize=7)
    fig.suptitle('Fixed-bin structural response and the role of the cross term', y=.98, fontsize=11)
    fig.text(.10, .025,
             'Changes are relative to the corresponding spherical branch; bars are orientation-integration SE.\n'
             'Right: paired full/self errors are propagated together; normalized-coefficient differences are not additive yield components.\n'
             'High-q b-grid sensitivity is unresolved; see the quadrature diagnostic.\n'
             r'The complementary-sector observable contains the same information: $\Delta\mathcal{A}_\phi=-(4/\pi)\Delta C_2$.' '\n'
             + _orientation_note(analytic) + ' ' + _selection(study),
             fontsize=6.8, linespacing=1.35)
    return _save(fig, destination, 'angular_bin_response', plt)


def _morphology_bridge(analytic, study, destination, plt):
    from matplotlib.lines import Line2D
    labels = _bin_labels(analytic)
    stage = analytic['stage_names'][-1]
    fig, axs = plt.subplots(2, 3, figsize=(9.0, 6.9))
    fig.subplots_adjust(left=.085, right=.98, bottom=.225, top=.86,
                        wspace=.43, hspace=.48)
    for col, family in enumerate(FAMILIES):
        cases = [analytic['cases']['spherical']] + _family_cases(analytic, family)
        xx = _array([c['bridge']['full']['delta_dark_fixed_mean'] for c in cases])
        xa = 1000*_array([c['bridge']['full']['delta_sqrt_area_halfmax'] for c in cases])
        for bi, (label, color) in enumerate(zip(labels, BIN_COLORS)):
            yy = _array([c['stages'][stage]['contrast']['delta_full'][bi][2] for c in cases])
            se = _array([c['stages'][stage]['full']['se'][bi][2] for c in cases])
            axs[0, col].errorbar(xx, 1000*yy, yerr=1000*se, color=color,
                                  marker='o', ms=3.4, capsize=2, elinewidth=.8,
                                  label=label)
        for branch, ls, marker in (('full', '-', 'o'), ('self', '--', 's')):
            yy = 1000*_array([c['bridge'][branch]['delta_mean_q'] for c in cases])
            se = 1000*_array([c['bridge'][branch]['mean_q_se'] for c in cases])
            axs[1, col].errorbar(xa, yy, yerr=se, color=FAMILY_COLORS[family],
                                  ls=ls, marker=marker, ms=3.5, capsize=2, elinewidth=.8)
        param = _family_parameter(family)
        values = [c['case'].get(param, 0) for c in cases]
        # Labels are attached to the diffraction branch, where only two curves
        # are present; duplicating labels over four angular curves obscures data.
        for i in range(1, len(cases)):
            y = 1000*cases[i]['bridge']['full']['delta_mean_q']
            axs[1, col].annotate(f'{values[i]:g}', (xa[i], y), xytext=(4, 5),
                                  textcoords='offset points', fontsize=6.8,
                                  color=FAMILY_COLORS[family])
        family_title = {'beta2': r'Vary $\beta_2$', 'beta3': r'Vary $\beta_3$',
                        'skin': r'Vary $\Delta r_{np}$ (fm)'}[family]
        axs[0, col].set_title(family_title)
        axs[0, col].set_xlabel(r'$\Delta$ fixed-dark-region mean' '\n' r'[(GeV/$c$)$^{-2}$]')
        axs[0, col].set_ylabel(r'$\Delta\mathcal{A}_\phi$ ($10^{-3}$)')
        axs[1, col].set_xlabel(r'$\Delta\sqrt{A_{1/2}}$ (MeV/$c$)')
        axs[1, col].set_ylabel(r'$\Delta\langle q_T\rangle$ (MeV/$c$)')
        for row in range(2):
            ax = axs[row, col]
            _zero(ax); ax.axvline(0, color='#bcbcbc', lw=.6)
            ax.margins(x=.22, y=.25)
            _panel(ax, chr(97+row*3+col))
    handles = [Line2D([0], [0], color=c, marker='o', label=lab)
               for lab, c in zip(labels, BIN_COLORS)]
    fig.legend(handles=handles, title=r'Angular bins (GeV/$c$), full intensity',
               loc='upper center', bbox_to_anchor=(.54, .952), ncol=4, fontsize=7,
               title_fontsize=7)
    fig.suptitle('Connecting image morphology to finite-momentum observables', y=.998, fontsize=11)
    fig.text(.085, .025,
             'Horizontal coordinates: known-b image features; dark mask is frozen from the spherical baseline. Horizontal errors are not quantified.\n'
             'Top: full angular-sector contrast after b,y averaging. Curves pass through zero, midpoint and endpoint; they only guide the eye.\n'
             'Bottom: whole-disk mean momentum, full (solid circles) and cross term removed (dashed squares); numbers give the varied parameter.\n'
             'All vertical bars: orientation-integration SE. A visible relation does not establish a shared mechanism or unique parameter extraction.\n'
             + _selection(study), fontsize=6.8, linespacing=1.35)
    return _save(fig, destination, 'angular_morphology_bridge', plt)


def _response_case(response, name):
    return next(c for c in response['cases'] if c['name'] == name)


def _scenario(case, fractional, angular):
    choices = [r for r in case['rows'] if np.isclose(r['fractional_sigma'], fractional)
               and np.isclose(r['angular_sigma'], angular)]
    if len(choices) != 1:
        raise ValueError(f'Expected one response ({fractional}, {angular}), found {len(choices)}')
    return choices[0]


def _response_values(row):
    # Prefer the exact-truth control variate when supplied; retain legacy
    # compatibility for raw response smoke summaries.
    d = row.get('truth_anchored', row)['vs_spherical']
    return (_array(d['delta']).reshape(-1, 4)[:, 0],
            _array(d['combined_sampling_se']).reshape(-1, 4)[:, 0])


def _detector(analytic, response, study, destination, plt):
    from matplotlib.lines import Line2D
    labels = _bin_labels(analytic)
    x = np.arange(len(labels))
    momenta = sorted({float(r['fractional_sigma']) for r in response['responses']
                      if np.isclose(r['angular_sigma'], 0)})
    angles = [r for r in response['responses'] if r['angular_sigma'] > 0]
    ncols = 2 if angles else 1
    fig, axs = plt.subplots(3, ncols, figsize=(8.6 if angles else 5.0, 8.0), squeeze=False)
    fig.subplots_adjust(left=.10 if angles else .17, right=.975, bottom=.17,
                        top=.86, wspace=.34, hspace=.54)
    colors = ['#343c45', '#397d9c', '#b47f45']
    markers = ['o', 's', '^']
    for ri, family in enumerate(FAMILIES):
        case = _endpoint(analytic, family)
        rsp = _response_case(response, case['case']['name'])
        for si, fraction in enumerate(momenta):
            delta, se = _response_values(_scenario(rsp, fraction, 0))
            axs[ri, 0].errorbar(x + (si-(len(momenta)-1)/2)*.11,
                                  1000*delta, yerr=1000*se, color=colors[si % len(colors)],
                                  marker=markers[si % len(markers)], ms=3.5, capsize=2,
                                  elinewidth=.8, label=rf'$\sigma_f={100*fraction:g}\%$')
        axs[ri, 0].set_title(_case_label(case) + ': momentum response')
        if angles:
            for ai, angular_scenario in enumerate(angles):
                fraction = float(angular_scenario['fractional_sigma'])
                angular = float(angular_scenario['angular_sigma'])
                for added, ls, marker in ((0, '-', 'o'), (angular, '--', 's')):
                    delta, se = _response_values(_scenario(rsp, fraction, added))
                    axs[ri, 1].errorbar(x + (-.06 if added == 0 else .06),
                                         1000*delta, yerr=1000*se, color=FAMILY_COLORS[family],
                                         marker=marker, ls=ls, ms=3.5, capsize=2,
                                         elinewidth=.8,
                                         label=rf'$\sigma_\phi={1000*added:g}$ mrad')
                axs[ri, 1].set_title(_case_label(case) + rf': $\sigma_f={100*fraction:g}\%$')
            if ri == 0:
                axs[ri, 1].legend(loc='best')
        for ci in range(ncols):
            ax = axs[ri, ci]
            _zero(ax); _bin_axis(ax, labels)
            ax.set_ylabel(r'$\Delta C_2$ ($10^{-3}$)')
            ax.margins(y=.25)
            _panel(ax, chr(97+ri*ncols+ci))
    handles, legends = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, legends, loc='upper center', bbox_to_anchor=(.53, .935), ncol=len(momenta))
    fig.suptitle('Track-response tests of the finite-momentum structural signal', y=.98, fontsize=11)
    anchored = all('truth_anchored' in r for c in response['cases'] for r in c['rows'])
    anchor_note = ('Exact zero-response anchor; paired MC uncertainty retained. Not detector calibration.\n'
                   if anchored else '')
    fig.text(.10 if angles else .17, .018,
             'Same reconstructed lepton pair defines momentum and reference axis; bins use reconstructed momentum.\n'
             + anchor_note
             + 'Bars combine paired response-MC and orientation-integration SE. These are numerical errors, not data errors.\n'
             + 'Uncalibrated smearing toys; coherent signal only. No acceptance model, detector calibration or systematic-error estimate.\n'
             + _selection(study), fontsize=6.8, linespacing=1.35)
    return _save(fig, destination, 'angular_detector_response', plt)


def _orientation_precision(analytic, study, destination, plt):
    labels = _bin_labels(analytic)
    stage = analytic['stage_names'][-1]
    baseline = _array(analytic['cases']['spherical']['stages'][stage]['full']['values'])[:, 0]
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.7))
    fig.subplots_adjust(left=.10, right=.975, bottom=.19, top=.85,
                        wspace=.30, hspace=.45)
    for bi, ax in enumerate(axs.flat):
        for fi, family in enumerate(('beta2', 'beta3')):
            case = _endpoint(analytic, family)
            prefixes = case['orientation_prefix']
            ns = sorted(int(k) for k in prefixes)
            values = _array([prefixes[str(n)]['full']['values'][bi][0] for n in ns])-baseline[bi]
            ses = _array([prefixes[str(n)]['full']['se'][bi][0] for n in ns])
            offset = .97 if fi == 0 else 1.03
            ax.errorbar(offset*np.asarray(ns), 1000*values, yerr=1000*ses,
                        color=FAMILY_COLORS[family], marker='o' if fi == 0 else 's',
                        ms=4, capsize=2, elinewidth=.9, label=_case_label(case))
            # Open rings flag a precision test rather than suppressing data.
            unresolved = np.abs(values) <= 3*ses
            ax.scatter(offset*np.asarray(ns)[unresolved], 1000*values[unresolved],
                       s=58, facecolors='none', edgecolors=FAMILY_COLORS[family],
                       linewidths=.75, zorder=5)
        _zero(ax)
        ax.set_xscale('log', base=2)
        alln = sorted({int(k) for f in ('beta2', 'beta3')
                       for k in _endpoint(analytic, f)['orientation_prefix']})
        ax.set_xticks(alln, [str(n) for n in alln])
        ax.set_xlim(min(alln)*.82, max(alln)*1.22)
        ax.set_xlabel('Independent orientation pairs, N')
        ax.set_ylabel(r'$\Delta C_2$ ($10^{-3}$)')
        ax.set_title(r'$q_T=$' + labels[bi] + r' GeV/$c$')
        ax.margins(y=.22)
        _panel(ax, chr(97+bi))
    handles, legends = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, legends, loc='upper center', bbox_to_anchor=(.54, .935), ncol=2)
    fig.suptitle('Orientation-integration precision for finite-momentum angular response', y=.98, fontsize=11)
    fig.text(.10, .025,
             'Points are nested orientation-prefix estimates of the full structural change; bars show one integration SE.\n'
             r'Open rings: $|\Delta C_2|\leq3\,\mathrm{SE}$, unresolved by this sampling criterion. Prefix estimates are correlated.' '\n'
             'This is a finite-sample stability check; agreement of prefixes does not establish full numerical convergence.\n'
             + _selection(study), fontsize=6.8, linespacing=1.35)
    return _save(fig, destination, 'angular_orientation_precision', plt)


def render_angular_figures(analytic: dict[str, Any], response: dict[str, Any] | None,
                           study: dict, destination: str | Path) -> dict[str, Any]:
    """Render saved analytic/response summaries; no forward calculations occur.

    ``analytic`` follows ``angular_observables.analyze_angular_results`` and
    ``response`` follows ``angular_response.run_angular_response``. The output
    lists every PNG/PDF/SVG path plus compact figure-integrity metadata.
    """
    if analytic['metric_names'] != ['C2', 'S2', 'Aphi', 'mean_q']:
        raise ValueError('Unexpected analytic metric ordering')
    if response and response['metric_names'] != analytic['metric_names']:
        raise ValueError('Response and analytic metric ordering differ')
    if response and not np.allclose(response['bin_edges_gev'], analytic['bin_edges_GeV']):
        raise ValueError('Response and analytic bins differ')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    plt = _style()
    files = []
    files += _mechanism(analytic, study, destination, plt)
    files += _bin_response(analytic, study, destination, plt)
    files += _morphology_bridge(analytic, study, destination, plt)
    if response:
        files += _detector(analytic, response, study, destination, plt)
    files += _orientation_precision(analytic, study, destination, plt)
    return {
        'files': files,
        'backend': 'matplotlib',
        'source_contract': 'Saved analytic and response JSON summaries; no image-derived measurements.',
        'geometry_scope': _selection(study),
        'bin_selection': 'Fixed four bins; no optimization using observed parameter responses.',
        'error_scope': 'Orientation SE or paired response-MC plus orientation SE, as stated per figure.',
        'morphology_horizontal_error': None,
        'editable_vector_text': True,
        'figure_width_inches': [8.6, 8.6, 9.0, 8.6, 8.6],
    }
