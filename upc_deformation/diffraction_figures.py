"""Exploratory plots of saved diffraction-analysis summaries, with no forward run."""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np


def _case_label(name, entry):
    case = entry.get("case", {})
    if name == "spherical":
        return "Spherical"
    parts = []
    for key, symbol in (("beta2", r"\beta_2"), ("beta3", r"\beta_3")):
        if case.get(key, 0):
            parts.append(rf"${symbol}={case[key]:g}$")
    if case.get("skin_fm", 0):
        # This is a literal model input, not an independently verified RMS label.
        parts.append(f"skin_fm = {case['skin_fm']:g} fm")
    return ", ".join(parts) or case.get("label", name)


def _mark_windows(ax, registry):
    colors = {"outer_slope": "#e9c46a", "fixed_012_014": "#6ab7bd"}
    for name, color in colors.items():
        window = registry["windows"].get(name, {})
        if window.get("status") == "registered":
            ax.axvspan(window["low"], window["high"], color=color, alpha=.16, zorder=0)
    ax.axvline(registry["qmax"], color=".45", lw=.8, ls=":")
    ax.set_xlim(0, registry["qmax"])
    ax.set_xlabel(r"$p_T$ (GeV/$c$)")
    ax.grid(alpha=.16)


def _window_note(registry):
    pieces = []
    for name, label in (("outer_slope", "Gold: frozen outer slope"),
                        ("fixed_012_014", "Teal: fixed comparison")):
        w = registry["windows"].get(name, {})
        if w.get("status") == "registered":
            pieces.append(f"{label} [{w['low']:.4f}, {w['high']:.4f}] GeV/c")
        else:
            pieces.append(f"{label} unavailable ({w.get('status', 'missing')})")
    return "; ".join(pieces)+"."


def _plot_estimate(ax, q, estimate, *, label, color, linestyle="-", scale=1.):
    values = np.asarray(estimate["value"])*scale
    errors = np.asarray(estimate["orientation_se"])*scale
    ax.plot(q, values, label=label, color=color, ls=linestyle, lw=1.7)
    if np.any(errors > 0):
        ax.fill_between(q, values-errors, values+errors, color=color, alpha=.12, linewidth=0)


def render_diffraction_figures(data, destination):
    """Write two PNG/PDF/SVG diagnostics and return six absolute file paths.

    Input is the compact analyze_diffraction_case/contrast_cases output.  Only
    saved summary arrays are read.  Peak-normalized shape ratios are descriptive
    curves with no invented error band for their fitted/selected normalization.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    registry, cases = data["registry"], data["cases"]
    if "spherical" not in cases:
        raise ValueError("plotting requires the spherical reference")
    sphere = cases["spherical"]
    names = list(cases)
    comparison_names = [name for name in names if name != "spherical" and name in data.get("contrasts", {})]
    palette = ["#555555", "#1676a0", "#8a6aa5", "#c46c32", "#508c63", "#b05278"]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(names)}
    files = []

    def save(fig, stem):
        for suffix in ("png", "pdf", "svg"):
            path = destination/f"{stem}.{suffix}"
            fig.savefig(path, dpi=180 if suffix == "png" else None)
            files.append(str(path))
        plt.close(fig)

    style = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
             "svg.fonttype": "none", "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8))
        ax = axes[0, 0]
        radial = sphere["radial"]
        r = np.asarray(radial["q"])
        d = np.asarray(radial["D"]["value"])
        j = np.asarray(radial["J"]["value"])
        peak = float(np.max(d))
        if peak <= 0:
            raise ValueError("spherical self density needs a positive peak")
        ax.plot(r, d/peak, color="black", ls="--", lw=1.8, label="Sphere D: source self")
        ax.plot(r, j/peak, color="#1676a0", lw=1.8, label="Sphere J: full")
        ax.set_title("a  Radial density on one momentum axis")
        ax.set_ylabel(r"Density / spherical $D_{\rm peak}$")
        ax.legend(fontsize=9)

        ax = axes[0, 1]
        for name, entry in cases.items():
            curves = entry["curves"]
            ax.plot(curves["q"], curves["full"]["C2"]["value"], color=colors[name],
                    lw=1.6, label=_case_label(name, entry))
            ax.plot(curves["q"], curves["self"]["C2"]["value"], color=colors[name], ls="--", lw=1.4)
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([Line2D([], [], color=".3", label="Full"),
                        Line2D([], [], color=".3", ls="--", label="Source self")])
        labels.extend(["Full", "Source self"])
        ax.legend(handles, labels, fontsize=8, ncol=2)
        ax.set_title("b  Cartesian angular moments")
        ax.set_ylabel(r"$C_2(p_T)$")
        ax.axhline(0, color=".5", lw=.7)

        ax = axes[1, 0]
        for name, entry in cases.items():
            rr = np.asarray(entry["radial"]["q"])
            dd = np.asarray(entry["radial"]["D"]["value"])
            if not np.array_equal(rr, r):
                raise ValueError("shape-ratio figures require identical saved radial grids")
            valid = (d > 0) & (np.max(dd) > 0)
            ratio = np.divide(dd/np.max(dd), d/peak, out=np.zeros_like(dd), where=valid)
            # Plot separate valid spans so an exact zero cannot be bridged by a
            # fabricated ratio or by a line across a physically undefined gap.
            indices = np.flatnonzero(valid)
            groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)
            for group_i, group in enumerate(groups):
                if len(group):
                    ax.plot(rr[group], ratio[group], color=colors[name], lw=1.7,
                            ls="--" if name == "spherical" else "-",
                            label=_case_label(name, entry) if group_i == 0 else None)
        ax.set_title("c  Single-source radial shape")
        ax.set_ylabel(r"$(D/D_{\rm peak})/(D_{\rm sph}/D_{\rm sph,peak})$")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        for name in comparison_names:
            _plot_estimate(ax, cases[name]["curves"]["q"],
                           data["contrasts"][name]["curves"]["double_difference_C2"],
                           color=colors[name], label=_case_label(name, cases[name]), scale=1e3)
        ax.set_title("d  Change of the full/self angular contrast")
        ax.set_ylabel(r"$10^3\,\Delta(C_{2,\rm full}-C_{2,\rm self})$")
        ax.axhline(0, color=".5", lw=.7)
        if comparison_names:
            ax.legend(fontsize=8)
        for ax in axes.flat:
            _mark_windows(ax, registry)
        fig.suptitle("Exploratory diffraction/angle diagnostics", y=.985, fontsize=14)
        fig.text(.055, .055, _window_note(registry), fontsize=8.5)
        fig.text(.055, .031, "Source: saved radial coefficients and Cartesian moment summaries. Bands in d: orientation-integration 1 SE only.", fontsize=8.5)
        fig.text(.055, .012, "Peak-normalized shape ratios have no uncertainty band. Full/self angular contrasts do not isolate an additive interference observable.", fontsize=8.5)
        fig.tight_layout(rect=(0, .08, 1, .96), h_pad=2.5, w_pad=2.)
        save(fig, "diffraction_same_axis")

        ncols = min(3, max(1, len(comparison_names)))
        nrows = max(1, math.ceil(len(comparison_names)/ncols))
        # A one-case smoke run still needs enough width for shared provenance
        # and uncertainty notes; otherwise figure-level text is cropped.
        fig, axes = plt.subplots(nrows, ncols, figsize=(max(11.5, 5*ncols), 4.3*nrows+1.05), squeeze=False)
        for i, ax in enumerate(axes.flat):
            if i >= len(comparison_names):
                ax.axis("off")
                if not comparison_names and i == 0:
                    ax.text(.5, .5, "No nonreference cases", ha="center", va="center")
                continue
            name = comparison_names[i]
            q = cases[name]["curves"]["q"]
            contrast = data["contrasts"][name]["curves"]
            for key, label, color, linestyle in (
                ("full_C2", r"$\Delta C_{2,\rm full}$", "#1676a0", "-"),
                ("self_C2", r"$\Delta C_{2,\rm self}$", "#c46c32", "--"),
                ("double_difference_C2", r"$\Delta(C_{2,\rm full}-C_{2,\rm self})$", "#8a6aa5", "-.")
            ):
                _plot_estimate(ax, q, contrast[key], color=color, label=label, linestyle=linestyle, scale=1e3)
            _mark_windows(ax, registry)
            ax.axhline(0, color=".5", lw=.7)
            ax.set_title(_case_label(name, cases[name]))
            ax.set_ylabel(r"$10^3\times$ change relative to sphere")
            ax.legend(fontsize=9)
        fig.suptitle("Structural changes with paired full/self orientation covariance", fontsize=14, y=.985)
        fig.text(.045, .075, _window_note(registry), fontsize=8.5)
        fig.text(.045, .045, "Bands: pointwise orientation-integration 1 SE; neighboring momenta are correlated. No detector, grid, or model error is included.", fontsize=8.5)
        fig.text(.045, .017, "Full/self subtraction is a nonlinear mechanism diagnostic. skin_fm labels denote the model input parameter.", fontsize=8.5)
        fig.tight_layout(rect=(0, .105, 1, .95), w_pad=2.)
        save(fig, "diffraction_structural_contrasts")
    return files
