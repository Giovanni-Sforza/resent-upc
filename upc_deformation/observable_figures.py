"""Trace Fig. 7 image features into measured-axis UPC observables.

The figure contract is diagnostic: test how interference filling and diffraction
scale responses change with the reference axis and geometry averaging.  The
plots do not infer a mechanism from radial location or assume that the two
mechanisms remain separated.  Source intensities are
never extracted from display colors.  All figures are made with matplotlib and
exported as editable PDF/SVG plus PNG previews.

``render_observable_figures`` consumes the linear arrays and metric dictionaries
written by the observable-bridge computation.  Array order is physical (qx,qy),
stage, then branch (full, source self terms only).  Removing the interference
cross term is a theoretical diagnostic, not an experimentally selectable sample.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


COLORS = {
    "spherical": "#43484c", "beta2": "#26749d",
    "beta3": "#8670a4", "skin": "#bc6937",
}
PARAMETERS = (("beta2", r"$\beta_2$"), ("beta3", r"$\beta_3$"),
              ("skin", r"$\Delta r_{np}$ (fm)"))
STAGE_TITLES = ("Known impact-parameter axis", "Lepton reference axis",
                "Impact-parameter average", "Impact parameter + rapidity")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "legend.frameon": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "lines.linewidth": 1.7,
    })
    return plt


def _save(fig, destination: Path, stem: str, plt) -> list[str]:
    files = []
    for extension in ("png", "pdf", "svg"):
        path = destination / f"{stem}.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=.08)
        files.append(str(path))
    plt.close(fig)
    return files


def _study(config: dict) -> dict:
    return config.get("study", config.get("observable_study", config.get("comparison", config)))


def _case_name(result: dict) -> str:
    return str(result["case"]["name"])


def _case_label(result: dict) -> str:
    case = result["case"]
    name = _case_name(result)
    if name == "spherical":
        return "Spherical, no skin"
    if name.startswith("beta2"):
        return rf'$\beta_2={case["beta2"]:g}$'
    if name.startswith("beta3"):
        return rf'$\beta_3={case["beta3"]:g}$'
    if name.startswith("skin"):
        return rf'$\Delta r_{{np}}={case["skin_fm"]:g}$ fm'
    return str(case.get("label", name))


def _baseline(results: list[dict]) -> dict:
    named = [r for r in results if _case_name(r) == "spherical"]
    if len(named) != 1:
        raise ValueError("Exactly one spherical baseline is required")
    return named[0]


def _endpoints(results: list[dict]) -> list[dict]:
    selected = []
    for family, _ in PARAMETERS:
        choices = [r for r in results if _case_name(r).startswith(family)]
        key = "skin_fm" if family == "skin" else family
        if choices:
            selected.append(max(choices, key=lambda r: r["case"][key]))
    return selected


def _metric(result: dict, stage: int, branch: int, key: str) -> float:
    return float(result["metrics"][stage][branch][key])


def _stage_captions(cfg: dict) -> list[str]:
    fixed = cfg["fixed_b_fm"]
    blo, bhi = cfg["b_min_fm"], cfg["b_max_fm"]
    ymax = cfg["ymax"]
    return [rf'$b={fixed:g}$ fm, $y=0$; $x\parallel b$',
            rf'$b={fixed:g}$ fm, $y=0$; $x\parallel k_T$',
            rf'${blo:g}<b<{bhi:g}$ fm, $y=0$; $x\parallel k_T$',
            rf'${blo:g}<b<{bhi:g}$ fm, $|y|<{ymax:g}$; $x\parallel k_T$']


def _selection_label(cfg: dict) -> str:
    return (rf'Hard-window diagnostic: ${cfg["b_min_fm"]:g}<b<'
            rf'{cfg["b_max_fm"]:g}$ fm, $|y|<{cfg["ymax"]:g}$; '
            "no neutron-tag selection")


def _panel(ax, letter: str):
    ax.text(-.025, 1.075, letter, transform=ax.transAxes, weight="bold", fontsize=11)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapz(y, x))


def _curve(result: dict, branch: int, qmax: float):
    """Return density, radial probability and cumulative on the analysis disk."""
    radii = np.asarray(result["radii"], dtype=float)
    values = np.asarray(result["coefficients"][-1, branch], dtype=float)
    q = np.unique(np.r_[radii[(radii >= 0) & (radii < qmax)], qmax])
    a0, c, _ = np.stack([np.interp(q, radii, row) for row in values])
    raw = 2 * np.pi * q * a0
    total = _trapz(raw, q)
    if total <= 0:
        raise ValueError("Nonpositive radial analysis yield")
    pdf = raw / total
    cumulative = np.r_[0., np.cumsum((pdf[1:] + pdf[:-1]) * np.diff(q) / 2)]
    c2 = np.divide(-c, 4 * a0, out=np.full_like(c, np.nan),
                   where=a0 > 1e-8 * max(float(a0.max()), 1e-30))
    return q, a0, pdf, cumulative, c2


def _baseline_bridge(results, cfg, destination, plt):
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle
    base = _baseline(results)
    axis = np.asarray(base["axis"])
    ims = np.asarray(base["normalized_images"])[:, 0]
    vmax = float(ims.max())
    norm = LogNorm(vmax * 1e-4, vmax)
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    fig, axs = plt.subplots(1, 4, figsize=(13.4, 4.3))
    fig.subplots_adjust(left=.06, right=.92, bottom=.26, top=.79, wspace=.18)
    titles = _stage_captions(cfg)
    for i, ax in enumerate(axs):
        artist = ax.imshow(ims[i].T, extent=extent, origin="lower", cmap="magma",
                           norm=norm, interpolation="nearest")
        ax.contour(axis, axis, ims[i].T, levels=[.5 * ims[i].max()],
                   colors=["#8bd3ec"], linewidths=.8)
        ax.add_patch(Circle((0, 0), cfg.get("q0", .02), fill=False,
                            ec="white", lw=.85, ls="--"))
        ax.set_xticks([-.2, 0, .2]); ax.set_yticks([-.2, 0, .2])
        ax.set_xlabel(r"$q_x$ (GeV/$c$)")
        if i == 0:
            ax.set_ylabel(r"$q_y$ (GeV/$c$)")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{STAGE_TITLES[i]}\n{titles[i]}", fontsize=9, pad=10)
        _panel(ax, chr(97 + i))
    cax = fig.add_axes([.94, .26, .012, .53])
    fig.colorbar(artist, cax=cax, label=r"Unit-yield density [(GeV/$c$)$^{-2}$]")
    fig.suptitle("From Fig. 7-inspired image features to a lepton-defined distribution", x=.49, y=.97, fontsize=13)
    fig.text(.06, .09, "Spherical baseline. Cyan: half-maximum contour; white circle: fixed low-momentum candidate window.\n"
             + rf'Raw intensities are averaged first; each panel is then normalized in $q_T<{cfg.get("qmax", .17):g}$ GeV/$c$. '
             + "Common logarithmic scale; contour thresholds are panel specific.\n"
             + _selection_label(cfg), fontsize=8)
    return _save(fig, destination, "baseline_bridge", plt)


def _ratio_bridge(results, cfg, destination, plt):
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FuncFormatter
    from matplotlib.patches import Circle
    baseline = _baseline(results)
    endpoint = _endpoints(results)
    axis = np.asarray(baseline["axis"])
    x, y = np.meshgrid(axis, axis, indexing="ij")
    qmax = cfg.get("qmax", .17)
    floor = cfg.get("ratio_floor_fraction", .005)
    base = np.asarray(baseline["normalized_images"])[:, 0]
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    nrows = len(endpoint)
    fig, axs = plt.subplots(nrows, 4, figsize=(13.2, 3.0 * nrows + 1.4), squeeze=False)
    fig.subplots_adjust(left=.10, right=.87, bottom=.12, top=.89, wspace=.14, hspace=.31)
    diagnostics = {}
    for row, result in enumerate(endpoint):
        images = np.asarray(result["normalized_images"])[:, 0]
        valid = ((x*x+y*y)[None] <= qmax*qmax) & (base > floor * base.max(axis=(1, 2))[:, None, None])
        valid &= images > 0
        ratios = np.divide(images, base, out=np.full_like(images, np.nan), where=valid)
        factor = max(1.03, float(np.nanmax(ratios)), 1 / float(np.nanmin(ratios)))
        norm = LogNorm(1 / factor, factor)
        diagnostics[_case_name(result)] = {"ratio_scale_factor": factor, "floor_fraction": floor}
        for col, ax in enumerate(axs[row]):
            artist = ax.imshow(np.ma.masked_invalid(ratios[col].T), origin="lower",
                               extent=extent, cmap="RdBu_r", norm=norm, interpolation="nearest")
            ax.set_facecolor("#ededed")
            ax.add_patch(Circle((0, 0), cfg.get("q0", .02), fill=False,
                                ec="#222222", lw=.65, ls="--"))
            ax.set_xticks([-.2, 0, .2]); ax.set_yticks([-.2, 0, .2])
            if row == 0:
                ax.set_title(STAGE_TITLES[col] + "\n" + _stage_captions(cfg)[col], fontsize=9, pad=10)
            if row == nrows - 1:
                ax.set_xlabel(r"$q_x$ (GeV/$c$)")
            if col == 0:
                ax.set_ylabel(_case_label(result) + "\n" + r"$q_y$ (GeV/$c$)", labelpad=5)
            else:
                ax.set_yticklabels([])
            _panel(ax, chr(97 + 4 * row + col))
        pos = axs[row, -1].get_position()
        cax = fig.add_axes([.90, pos.y0, .012, pos.height])
        cb = fig.colorbar(artist, cax=cax, ticks=[1/factor, 1/np.sqrt(factor), 1, np.sqrt(factor), factor])
        cb.formatter = FuncFormatter(lambda v, _: f"{v:.3g}")
        cb.update_ticks(); cb.minorticks_off(); cb.set_label("Shape / spherical")
    fig.suptitle("Structure responses through the same sequence of projections", y=.975, fontsize=13)
    fig.text(.10, .035, "Each row shares one reciprocal color scale centered at 1; rows have different scales.\n"
             + rf'Gray: $q_T>{qmax:g}$ GeV/$c$, or baseline density below {100*floor:g}% of its panel peak. '
             + "This display mask does not define an observable.\n"
             + "Ratios compare separately unit-normalized full-interference distributions; dashed circle is the fixed low-momentum window.\n"
             + _selection_label(cfg), fontsize=8)
    return _save(fig, destination, "structure_bridge_ratios", plt), diagnostics


def _dark_region_audit(results, cfg, destination, plt):
    """Locate dark regions and compare the same raw full/self intensities.

    Radial location is not a mechanism classifier: a high-q dark region may
    contain a transverse arm of an interference fringe.  The division mask is
    solely a numerical display requirement and never removes observable pixels.
    """
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    base = _baseline(results)
    axis = np.asarray(base['axis'])
    qx, qy = np.meshgrid(axis, axis, indexing='ij')
    radius = np.hypot(qx, qy)
    disk = radius <= cfg['qmax']
    normalized = np.asarray(base['normalized_images'], dtype=float)
    raw = np.asarray(base['images'], dtype=float)
    images = normalized[:, 0]
    norm = LogNorm(float(images.max()) * 1e-4, float(images.max()))
    k_norm = Normalize(-1, 1)
    floor = float(cfg.get('ratio_floor_fraction', .005))
    tolerance = 1e-6
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    fig, axs = plt.subplots(2, 4, figsize=(14.2, 8.8))
    fig.subplots_adjust(left=.065, right=.88, bottom=.24, top=.84, wspace=.16, hspace=.32)
    diagnostics = {}
    captions = _stage_captions(cfg)
    for i in range(4):
        img = images[i]
        full_mask = disk & (img <= np.quantile(img[disk], .05))
        self_img = normalized[i, 1]
        self_mask = disk & (self_img <= np.quantile(self_img[disk], .05))
        full_raw, self_raw = raw[i]
        # Display the complete saved square so that upper/lower high-q arms
        # remain visible even when they lie outside the observable's qmax disk.
        # This changes neither the dark-percentile definition nor any metric.
        valid = np.isfinite(full_raw) & np.isfinite(self_raw)
        valid &= self_raw > floor * float(self_raw.max())
        contrast = np.divide(full_raw-self_raw, self_raw,
                             out=np.full_like(full_raw, np.nan), where=valid)
        values = contrast[valid]
        if not len(values):
            raise ValueError('No stable raw full/self pixels for the dark-region audit')
        if np.max(np.abs(values)) > 1 + tolerance:
            raise ValueError(f'Raw cross-term contrast outside [-1,1] at stage {i}: '
                             f'{values.min():.8g} to {values.max():.8g}')
        dark_count = int(full_mask.sum())
        unavailable_count = int((full_mask & ~valid).sum())
        stage = base.get('stage_names', ['b_axis', 'decay_fixed', 'decay_b', 'decay_by'])[i]
        diagnostics[stage] = {
            'raw_intensity_definition': '(I_full-I_self)/I_self',
            'full_dark_mask_pixel_count': dark_count,
            'self_dark_mask_pixel_count': int(self_mask.sum()),
            'full_dark_mask_K_unavailable_pixels': unavailable_count,
            'full_dark_mask_K_unavailable_fraction': unavailable_count / dark_count,
            'division_floor_fraction_of_self_peak': floor,
            'K_min_displayed': float(values.min()), 'K_max_displayed': float(values.max()),
            'K_bound_tolerance': tolerance,
            'K_display_domain': 'Entire saved qx,qy square; observable analysis disk unchanged.',
            'mask_interpretation': 'Division stability only; not a physical exclusion window.',
        }
        samples = {}
        for label, region in (
                ('outside_analysis_disk', radius > cfg['qmax']),
                ('upper_display_band_qy_gt_qmax', qy > cfg['qmax']),
                ('lower_display_band_qy_lt_minus_qmax', qy < -cfg['qmax'])):
            eligible = valid & region
            if not eligible.any():
                samples[label] = None
                continue
            # Data-derived coordinates are examples of the displayed contrast,
            # not a fitted or optimized experimental window.
            index = np.unravel_index(np.argmin(np.where(eligible, contrast, np.inf)), contrast.shape)
            samples[label] = {
                'selection': 'Minimum displayed K within this geometrical band.',
                'qx_GeV': float(qx[index]), 'qy_GeV': float(qy[index]),
                'q_GeV': float(radius[index]), 'K': float(contrast[index]),
                'self_over_stage_self_peak': float(self_raw[index] / self_raw.max()),
                'full_over_stage_full_peak': float(full_raw[index] / full_raw.max()),
            }
        diagnostics[stage]['high_q_diagnostic_samples'] = samples
        ax = axs[0, i]
        gray_artist = ax.imshow(img.T, origin='lower', extent=extent,
                               cmap='Greys_r', norm=norm, interpolation='nearest')
        ax.contour(axis, axis, full_mask.T.astype(float), levels=[.5],
                   colors=['#db3d1a'], linewidths=.9)
        ax.contour(axis, axis, self_mask.T.astype(float), levels=[.5],
                   colors=['#51b8ce'], linewidths=.9, linestyles='--')
        ax.add_patch(Circle((0, 0), cfg['q0'], ec='#d9d9d9', fill=False,
                            lw=.8, ls='--', alpha=.85))
        ax.set_title(STAGE_TITLES[i] + '\n' + captions[i], fontsize=8.5, pad=10)
        ax = axs[1, i]
        k_artist = ax.imshow(np.ma.masked_invalid(contrast.T), origin='lower', extent=extent,
                            cmap='RdBu_r', norm=k_norm, interpolation='nearest')
        ax.set_facecolor('#e6e6e6')
        ax.contour(axis, axis, full_mask.T.astype(float), levels=[.5],
                   colors=['#222222'], linewidths=.65)
        ax.add_patch(Circle((0, 0), cfg['q0'], ec='#777777', fill=False, lw=.7, ls='--', alpha=.65))
        ax.add_patch(Circle((0, 0), cfg['qmax'], ec='#666666', fill=False, lw=.8, ls=':', alpha=.85))
        ax.set_xlabel(r'$q_x$ (GeV/$c$)')
        ax.text(.5, -.31, f'K unavailable in dark mask: {unavailable_count}/{dark_count}',
                transform=ax.transAxes, ha='center', fontsize=7)
        for row in range(2):
            ax = axs[row, i]
            ax.set(xticks=[-.2, 0, .2], yticks=[-.2, 0, .2])
            if i == 0:
                ax.set_ylabel(('Full intensity\n' if row == 0 else 'Cross-term contrast\n')
                              + r'$q_y$ (GeV/$c$)')
            else:
                ax.set_yticklabels([])
            _panel(ax, chr(97 + 4*row + i))
    for row, artist, label in (
            (0, gray_artist, r'Unit-yield density [(GeV/$c$)$^{-2}$]'),
            (1, k_artist, r'$K=(I_{\rm full}-I_{\rm self})/I_{\rm self}$')):
        pos = axs[row, -1].get_position()
        cax = fig.add_axes([.915, pos.y0, .012, pos.height])
        fig.colorbar(artist, cax=cax, label=label)
    fig.suptitle('Dark-region locations and the two-source cross term', y=.97, fontsize=13)
    fig.legend(handles=[
        Line2D([0], [0], color='#db3d1a', label='Full: lowest-5% region'),
        Line2D([0], [0], color='#51b8ce', ls='--', label='Cross term removed: lowest-5% region'),
        Line2D([0], [0], color='#222222', label='Lower row: full dark-region boundary')],
        loc='lower center', bbox_to_anchor=(.50, .125), ncol=3, fontsize=8)
    fig.text(.065, .025,
             'Spherical diagnostic baseline; red/cyan lowest-5% contours are defined within the analysis disk separately for each branch and stage.\n'
             'Outer high-q dark regions can contain transverse arms of the interference fringe; radial location alone does not identify the mechanism.\n'
             + rf'K uses raw intensities over the entire saved square. Gray: self intensity below {100*floor:g}% of its stage peak; this is only a division-stability mask.' + '\n'
             + rf'Dotted circles: analysis boundary $q_{{\max}}={cfg["qmax"]:g}$ GeV/$c$; faint dashed circles: low-momentum candidate window, not a dark-region acceptance criterion.' + '\n'
             + 'The full-square K map is a mechanism diagnostic; it changes no observable window and is not a numerical replay of Fig. 7.\n'
             + _selection_label(cfg), fontsize=8)
    return _save(fig, destination, 'dark_region_audit', plt), diagnostics


def _radial_mechanism(results, cfg, destination, plt):
    from matplotlib.lines import Line2D
    base = _baseline(results)
    cases = [base] + _endpoints(results)
    qmax, q0 = cfg.get("qmax", .17), cfg.get("q0", .02)
    fig, axs = plt.subplots(2, 2, figsize=(11.3, 8.0))
    fig.subplots_adjust(left=.09, right=.97, bottom=.17, top=.86, wspace=.26, hspace=.36)
    for result in cases:
        name = _case_name(result)
        family = next((x for x in COLORS if name.startswith(x)), "spherical")
        color = COLORS[family]
        q, a0, pdf, cumulative, c2 = _curve(result, 0, qmax)
        _, self_a0, self_pdf, _, _ = _curve(result, 1, qmax)
        label = _case_label(result)
        axs[0, 0].plot(q, pdf, color=color, label=label)
        axs[0, 0].plot(q, self_pdf, color=color, ls="--", lw=1.25)
        keep = self_a0 > 1e-6 * self_a0.max()
        axs[0, 1].plot(q[keep], a0[keep] / self_a0[keep], color=color)
        axs[1, 0].plot(q, cumulative, color=color)
        axs[1, 1].plot(q, c2, color=color)
    axs[0, 0].set(ylabel=r"$dP/dq_T$ [(GeV/$c$)$^{-1}$]", title="Diffraction envelope and interference")
    axs[0, 0].legend(loc="upper right", fontsize=8)
    axs[0, 1].axhline(1, color="#999999", lw=.8, ls=":")
    axs[0, 1].set(ylabel=r"$I_{\rm full}(q_T)/I_{\rm self}(q_T)$", title="Interference contribution at fixed radial momentum")
    axs[1, 0].set(ylabel=r"$P(q_T<q)$", title="Fixed-window filling from the radial distribution", ylim=(0, 1.02))
    axs[1, 0].axvline(q0, color="#777777", lw=.8, ls=":")
    axs[1, 0].text(q0 + .002, .91, rf'$q_0={q0:g}$', fontsize=8)
    axs[1, 1].set(ylabel=r"$C_2(q_T)=\langle\cos 2\alpha\rangle$", title="Lepton-axis angular correlation")
    axs[1, 1].axhline(0, color="#999999", lw=.8, ls=":")
    for i, ax in enumerate(axs.flat):
        ax.set_xlabel(r"$q_T$ (GeV/$c$)")
        ax.set_xlim(0, qmax)
        _panel(ax, chr(97 + i))
    handles = [Line2D([0], [0], color="#333333", label="Full intensity"),
               Line2D([0], [0], color="#333333", ls="--", label="Cross term removed (diagnostic)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .085), ncol=2, frameon=False)
    fig.suptitle("Mechanism tests after impact-parameter and rapidity averaging", y=.97, fontsize=13)
    fig.text(.09, .025, "Top left: full and self-term spectra each have unit yield in the analysis disk. Top right: the raw-intensity ratio retains their relative yield.\n"
             + "Bottom row: full intensity only. Removing the cross term is a theoretical diagnostic; it is not an event selection.\n"
             + _selection_label(cfg), fontsize=8)
    return _save(fig, destination, "mechanism_radial", plt)


def _translation(results, cfg, destination, plt):
    from matplotlib.lines import Line2D
    base = _baseline(results)
    fig, axs = plt.subplots(2, 3, figsize=(13.0, 8.1))
    fig.subplots_adjust(left=.095, right=.975, bottom=.17, top=.87, wspace=.39, hspace=.49)
    dark_key = "dark_fixed_mean" if "dark_fixed_mean" in base["metrics"][0][0] else "fringe_low5"
    for col, (family, parameter_label) in enumerate(PARAMETERS):
        key = "skin_fm" if family == "skin" else family
        cases = [base] + sorted([r for r in results if _case_name(r).startswith(family)], key=lambda r: r["case"][key])
        color = COLORS[family]
        parameters = [float(r["case"].get(key, 0)) for r in cases]
        xdark = np.asarray([_metric(r, 0, 0, dark_key) for r in cases])
        xarea = np.sqrt([_metric(r, 0, 0, "area_halfmax") for r in cases])
        xdark -= xdark[0]
        xarea = 1000 * (xarea - xarea[0])
        for branch, ls, marker in ((0, "-", "o"), (1, "--", "s")):
            fill = np.asarray([_metric(r, 3, branch, "F_dip") for r in cases])
            mean = np.asarray([_metric(r, 3, branch, "mean_q") for r in cases])
            # The spherical baseline has no orientation variance.  The supplied
            # SE therefore also describes each structure-minus-baseline change.
            # Morphological x-axis uncertainties are not supplied and are not
            # inferred from these vertical error bars.
            fill_se = np.asarray([float(r["orientation_se"][3][branch][0])
                                  if "orientation_se" in r else 0. for r in cases])
            mean_se = np.asarray([float(r["orientation_se"][3][branch][1])
                                  if "orientation_se" in r else 0. for r in cases])
            axs[0, col].errorbar(xdark, 100 * (fill - fill[0]), yerr=100*fill_se,
                                 color=color, ls=ls, marker=marker, ms=4, capsize=2, elinewidth=1)
            axs[1, col].errorbar(xarea, 1000 * (mean - mean[0]), yerr=1000*mean_se,
                                 color=color, ls=ls, marker=marker, ms=4, capsize=2, elinewidth=1)
            if branch == 0:
                for row, (xx, yy) in enumerate(((xdark, 100*(fill-fill[0])), (xarea, 1000*(mean-mean[0])))):
                    for index in range(1, len(cases)):
                        axs[row, col].annotate(f"{parameters[index]:g}", (xx[index], yy[index]),
                                               xytext=(4, 6), textcoords="offset points", fontsize=8, color=color)
        for row in range(2):
            ax = axs[row, col]
            ax.axhline(0, color="#bcbcbc", lw=.65)
            ax.axvline(0, color="#bcbcbc", lw=.65)
            ax.margins(x=.22, y=.25)
            _panel(ax, chr(97 + row*3 + col))
        axs[0, col].set_title("Vary " + parameter_label, pad=11)
        axs[0, col].set_xlabel(r"$\Delta$ fixed-dark-region mean density" + "\n" + r"[(GeV/$c$)$^{-2}$]")
        axs[0, col].set_ylabel(r"$\Delta F_{\rm dip}$ (percentage points)")
        axs[1, col].set_xlabel(r"$\Delta\sqrt{A_{1/2}}$ (MeV/$c$)")
        axs[1, col].set_ylabel(r"$\Delta\langle q_T\rangle$ (MeV/$c$)")
    fig.suptitle("Testing links between Fig. 7-inspired features and candidate observables", y=.975, fontsize=13)
    handles = [Line2D([0], [0], color="#444444", marker="o", label="Full final distribution"),
               Line2D([0], [0], color="#444444", ls="--", marker="s", label="Final cross term removed (diagnostic)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .093), ncol=2, frameon=False)
    dark_note = "The dark-region mask is frozen from the spherical image." if dark_key == "dark_fixed_mean" else "Dark proxy: mean of the lowest 5% of pixels."
    fig.text(.095, .025, "Horizontal axes: features of the known-b image; vertical axes: changes in the final geometry-averaged observable, each relative to its spherical baseline.\n"
             + dark_note + " Half-maximum area is evaluated per image; labels give the varied parameter.\n"
             + "Vertical bars: orientation integration SE only; uncertainty of the morphological horizontal coordinates is not quantified.\n"
             + "Connections guide the eye; they do not establish a shared mechanism or unique parameter extraction. "
             + _selection_label(cfg), fontsize=8)
    return _save(fig, destination, "feature_translation", plt)


def _screening(results, cfg, destination, plt):
    base = _baseline(results)
    if not base.get("response"):
        return [], {}
    endpoints = [r for r in _endpoints(results) if r.get("response")]
    labels = [r"$\Delta F_{\rm dip}$ (percentage points)",
              r"$\Delta\langle q_T\rangle$ (MeV/$c$)", r"$\Delta C_2$"]
    factors = (100., 1000., 1.)
    fig, axs = plt.subplots(3, 2, figsize=(11.0, 10.2))
    fig.subplots_adjust(left=.12, right=.97, bottom=.16, top=.87, wspace=.28, hspace=.39)
    thresholds = {}
    unresolved = [[] for _ in range(3)]
    includes_orientation = any("delta_numerical_se" in s for r in endpoints for s in r["response"])
    baseline_response = {float(s["fractional_sigma"]): s for s in base["response"]}
    for result in endpoints:
        name = _case_name(result)
        color = COLORS[next(x for x in ("beta2", "beta3", "skin") if name.startswith(x))]
        scenarios = sorted(result["response"], key=lambda s: s["fractional_sigma"])
        sigma = 100 * np.asarray([s["fractional_sigma"] for s in scenarios])
        deltas = np.asarray([s["delta_vs_spherical"] for s in scenarios])
        # delta_numerical_se is a supplied TOTAL numerical error (for example
        # paired MC and paired orientation SE combined in quadrature).  Adding
        # delta_mc_se again would double count its Monte Carlo component.
        ses = np.asarray([s.get("delta_numerical_se", s["delta_mc_se"]) for s in scenarios])
        need = []
        for scenario, delta in zip(scenarios, deltas):
            b = baseline_response[float(scenario["fractional_sigma"])]
            variance = np.maximum(np.diag(scenario["event_covariance"]), np.diag(b["event_covariance"]))
            n = np.divide(9 * variance, delta**2, out=np.full(3, np.inf), where=np.abs(delta)>0)
            need.append(n)
        need = np.asarray(need)
        resolved = np.isfinite(deltas) & np.isfinite(ses) & (np.abs(deltas) > 3 * ses)
        need[~resolved] = np.nan
        thresholds[name] = {
            "fractional_sigma": (sigma / 100).tolist(),
            "events_3sigma": [[float(n) if np.isfinite(n) else None for n in row] for row in need],
            "numerically_resolved": resolved.tolist(),
            "numerical_se": ses.tolist(),
            "resolution_rule": "abs(delta) > 3 * numerical_se",
        }
        for row, factor in enumerate(factors):
            axs[row, 0].errorbar(sigma, factor*deltas[:, row], yerr=factor*ses[:, row],
                                 color=color, marker="o", ms=4, capsize=2, label=_case_label(result))
            axs[row, 1].plot(sigma, need[:, row], color=color, marker="o", ms=4)
            if not np.all(resolved[:, row]):
                missing = ", ".join(f"{v:g}" for v in sigma[~resolved[:, row]])
                unresolved[row].append((_case_label(result) + rf": $\sigma_f={missing}\%$", color))
    for row in range(3):
        axs[row, 0].axhline(0, color="#999999", lw=.8)
        axs[row, 0].set_ylabel(labels[row])
        axs[row, 1].set_yscale("log")
        axs[row, 1].set_ylabel(r"Events for nominal $3\sigma$ separation")
        for n in (1e4, 1e5, 1e6):
            axs[row, 1].axhline(n, color="#cccccc", lw=.6, ls=":")
        if unresolved[row]:
            # Reserve a text band above every finite event-count estimate;
            # unresolved-case notes must not obscure a resolved curve.
            lo, hi = axs[row, 1].get_ylim()
            reserved = min(.45, .18 + .085 * len(unresolved[row]))
            loglo, loghi = np.log10(lo), np.log10(hi)
            axs[row, 1].set_ylim(lo, 10 ** (loglo + (loghi-loglo)/(1-reserved)))
            axs[row, 1].text(.98, .97, "Numerically unresolved; no count quoted:",
                            transform=axs[row, 1].transAxes, ha="right", va="top", fontsize=7)
            for j, (label, color) in enumerate(unresolved[row]):
                axs[row, 1].text(.98, .88 - .085*j, label, color=color,
                                transform=axs[row, 1].transAxes, ha="right", va="top", fontsize=7)
        for col in range(2):
            ax = axs[row, col]
            ax.set_xlabel(r"Track momentum resolution parameter $\sigma_f$ (%)")
            _panel(ax, chr(97 + row*2 + col))
    axs[0, 0].set_title("Structure difference after the same track response")
    axs[0, 1].set_title("Conditional event-count requirement")
    handles, legend_labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=8, loc="upper center",
               bbox_to_anchor=(.54, .931), ncol=3)
    fig.suptitle("Track response and conditional statistical screening", y=.97, fontsize=13)
    numerical_note = ("Left error bars: sampling SE (paired MC and orientations); grid shifts are checked separately. These are not data errors."
                      if includes_orientation else
                      "Left error bars: paired Monte Carlo SE only; orientation uncertainty has not been included.")
    fig.text(.12, .019, numerical_note + "\n"
             + r"Right: $N_{3\sigma}=9\max(V_{\rm case},V_{\rm sph})/(\Delta O)^2$ for one sample against exact templates; $V$ is per-event variance."
             + r" Counts are suppressed when $|\Delta O|\leq3\sigma_{\rm numerical}$."
             + "\nCoherent signal only; background checks are tabulated separately. Toy response and counts omit detector acceptance and systematic uncertainties.\n"
             + _selection_label(cfg), fontsize=8)
    return _save(fig, destination, "response_screening", plt), thresholds


def render_observable_figures(results: list[dict[str, Any]], config: dict,
                              destination: str | Path) -> dict[str, Any]:
    """Render bridge, feature translation, mechanism and response diagnostics.

    Required result arrays: ``normalized_images[4,2,N,N]``,
    ``coefficients[3,2,3,nr]``, ``axis``, ``radii``, and metric dictionaries
    ``metrics[4][2]``.  Optional ``response`` rows add response screening.
    No physics calculation or source-data mutation is performed here.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if not results:
        raise ValueError("No results to plot")
    plt = _style()
    cfg = _study(config)
    files = _baseline_bridge(results, cfg, destination, plt)
    paths, dark_diagnostics = _dark_region_audit(results, cfg, destination, plt)
    files += paths
    paths, ratio_diagnostics = _ratio_bridge(results, cfg, destination, plt)
    files += paths
    files += _radial_mechanism(results, cfg, destination, plt)
    files += _translation(results, cfg, destination, plt)
    paths, screening = _screening(results, cfg, destination, plt)
    files += paths
    return {"files": files, "ratio_diagnostics": ratio_diagnostics,
            "dark_region_location_audit": dark_diagnostics,
            "conditional_event_requirements": screening}
