"""Show panel-d angular moments and orientation-integration convergence.

Figure contract: compare the 512-pair central estimates with the original
32-pair curves and assess the finite-orientation integral.  The three figures
are quantitative plots of supplied calculations; no forward model, fitting,
smoothing, or event selection occurs here.  Shading is pointwise Monte Carlo
1 SE, not an experimental error band or a simultaneous confidence band.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .panel_d_ratio_figures import _style


FAMILIES = ("beta2", "beta3", "skin")
DEFORMATIONS = ("beta2", "beta3")
NESTED_COUNTS = (32, 64, 128, 256, 512)
COLORS = {"spherical": "#626262", "beta2": "#26749d",
          "beta3": "#8670a4", "skin": "#bc6937"}
STYLES = {"spherical": (0, (4, 2)), "beta2": "-",
          "beta3": (0, (5, 2, 1, 2)), "skin": (0, (2, 1))}


def _label(name, row):
    meta = row["case"]
    if name == "spherical":
        return "Spherical reference"
    if name == "beta2":
        return rf'$\beta_2={meta["beta2"]:g}$'
    if name == "beta3":
        return rf'$\beta_3={meta["beta3"]:g}$'
    return rf'$\Delta r_{{np}}={meta["skin_fm"]:g}$ fm'


def _array(row, key, q, *, nonnegative=False):
    arr = np.asarray(row[key], dtype=float)
    valid = q > 0
    if arr.shape != q.shape or not np.all(np.isfinite(arr[valid])):
        raise ValueError(f"{key}: expected a finite value at each positive momentum")
    if nonnegative and np.any(arr[valid] < 0):
        raise ValueError(f"{key}: standard errors must be nonnegative")
    return arr


def _axis(ax, q, *, ylabel=None):
    ax.set(xlim=(0, float(q[-1])), xlabel=r'$p_T$ (GeV/$c$)')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(color="#e5e5e5", linewidth=.55)
    ax.tick_params(direction="out", length=3, width=.7)


def _panel(ax, letter):
    ax.text(-.15, 1.055, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=10, va="bottom")


def _scope(data):
    cfg = data["config"]
    return (rf'$b={cfg["b_min_fm"]:g}$–${cfg["b_max_fm"]:g}$ fm, '
            rf'$\Delta b={cfg["b_step_fm"]:g}$ fm, $|y|<{cfg["ymax"]:g}$; '
            r'$x\parallel k_T$. The $p_T=0$ point is omitted.')


def _footer(fig, data, extra, *, x=.105, y=.033):
    fig.text(x, y, _scope(data) + "\n" + extra + "\n"
             "Pointwise orientation-integration errors only; detector and b-grid uncertainties are not included.",
             fontsize=7, va="bottom", linespacing=1.5)


def _save(fig, destination, stem, plt):
    files = []
    for ext in ("png", "pdf", "svg"):
        path = destination / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.08)
        files.append(str(path))
    plt.close(fig)
    return files


def _absolute(data, destination, plt):
    q = np.asarray(data["q"], dtype=float)
    valid = q > 0
    fig, ax = plt.subplots(figsize=(8.85, 4.9))
    fig.subplots_adjust(left=.105, right=.975, bottom=.235, top=.85)
    handles = []
    for name in (*FAMILIES, "spherical"):
        row = data["cases"][name]
        value = _array(row, "C2", q)
        error = _array(row, "orientation_se", q, nonnegative=True)
        if name in DEFORMATIONS:
            ax.fill_between(q[valid], (value-error)[valid], (value+error)[valid],
                            color=COLORS[name], alpha=.14, linewidth=0, zorder=2)
        line, = ax.plot(q[valid], value[valid], color=COLORS[name],
                       linestyle=STYLES[name], linewidth=1.7,
                       label=_label(name, row), zorder=3)
        handles.append(line)
    ax.axhline(0, color="#a0a0a0", linewidth=.75, zorder=1)
    _axis(ax, q, ylabel=r'$C_2(p_T)=\langle\cos 2\alpha\rangle_{p_T}$')
    ax.set_xticks(np.r_[np.arange(0, .161, .02), .17])
    ax.margins(y=.1)
    fig.suptitle("Angular moment with 512 orientation pairs", y=.985, fontsize=11)
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center",
               bbox_to_anchor=(.54, .935), ncol=4, fontsize=8, handlelength=2.8,
               columnspacing=1.5)
    _footer(fig, data,
            "Shading: pointwise 1 SE of the 512-pair mean for each deformation; sphere and skin are deterministic.")
    return _save(fig, destination, "C2_vs_pT_512", plt)


def _delta(data, destination, plt):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    q = np.asarray(data["q"], dtype=float)
    valid = q > 0
    base = _array(data["cases"]["spherical"], "C2", q)
    original_base = _array(data["cases"]["spherical"], "C2_original32", q)
    fig, axes = plt.subplots(1, 3, figsize=(8.85, 4.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.975, bottom=.25, top=.79, wspace=.18)
    extrema = []
    for letter, ax, name in zip("abc", axes, FAMILIES):
        row = data["cases"][name]
        value = 1e3 * (_array(row, "C2", q) - base)
        error = 1e3 * _array(row, "orientation_se", q, nonnegative=True)
        original = 1e3 * (_array(row, "C2_original32", q) - original_base)
        if name in DEFORMATIONS:
            ax.fill_between(q[valid], (value-error)[valid], (value+error)[valid],
                            color=COLORS[name], alpha=.15, linewidth=0, zorder=2)
            ax.plot(q[valid], original[valid], color="#858585", linestyle=(0, (4, 2)),
                    linewidth=1.1, zorder=2)
            extrema.extend([(value-error)[valid], (value+error)[valid], original[valid]])
        else:
            extrema.append(value[valid])
            ax.text(.97, .94, "Deterministic", color="#666666", fontsize=7,
                    va="top", ha="right", transform=ax.transAxes)
        ax.plot(q[valid], value[valid], color=COLORS[name], linewidth=1.7, zorder=3)
        ax.axhline(0, color="#a0a0a0", linewidth=.75, zorder=1)
        _axis(ax, q)
        ax.set(xticks=(0, .05, .10, .15), title=_label(name, row))
        _panel(ax, letter)
    lo = min(float(np.min(a)) for a in extrema)
    hi = max(float(np.max(a)) for a in extrema)
    margin = .1 * max(hi-lo, 1e-3)
    axes[0].set_ylim(lo-margin, hi+margin)
    axes[0].set_ylabel(r'$10^3\,[C_2(p_T)-C_{2,\rm sph}(p_T)]$')
    fig.suptitle("Angular response relative to the spherical reference", y=.985, fontsize=11)
    fig.legend([Line2D([], [], color="#555555", linewidth=1.7),
                Patch(facecolor="#777777", alpha=.16, edgecolor="none"),
                Line2D([], [], color="#858585", linestyle=(0, (4, 2)), linewidth=1.1)],
               ["512-pair mean", "512-pair pointwise 1 SE", "Original 32-pair mean"],
               loc="upper center", bbox_to_anchor=(.54, .924), ncol=3,
               fontsize=8, handlelength=2.8, columnspacing=1.5)
    _footer(fig, data,
            "All three panels share the same vertical scale. Original 32-pair curves are shown only for deformations.",
            x=.10)
    return _save(fig, destination, "delta_C2_vs_pT_512", plt)


def _convergence(data, destination, plt):
    from matplotlib.lines import Line2D
    q = np.asarray(data["q"], dtype=float)
    valid = q > 0
    base = _array(data["cases"]["spherical"], "C2", q)
    fig, axes = plt.subplots(2, 3, figsize=(8.85, 7.1), sharex=True)
    fig.subplots_adjust(left=.095, right=.975, bottom=.23, top=.81,
                        hspace=.40, wspace=.40)
    nested_style = {
        32: ("#a6a6a6", (0, (2, 2)), .95),
        64: ("#858585", (0, (4, 2)), 1.),
        128: ("#666666", (0, (5, 2, 1, 2)), 1.1),
        256: ("#484848", "-", 1.1),
    }
    for i, name in enumerate(DEFORMATIONS):
        row = data["cases"][name]
        color = COLORS[name]
        nested, se_axis, halves = axes[i]
        for count in NESTED_COUNTS:
            entry = row["nested"][str(count)]
            center = 1e3 * (_array(entry, "C2", q) - base)
            c, ls, lw = (color, "-", 1.8) if count == 512 else nested_style[count]
            nested.plot(q[valid], center[valid], color=c, linestyle=ls,
                        linewidth=lw, label=str(count), zorder=3 if count == 512 else 2)
        nested.axhline(0, color="#aaaaaa", linewidth=.7, zorder=1)
        _axis(nested, q, ylabel=r'$10^3\,\Delta C_2$')
        for count, c, ls in ((32, "#858585", (0, (4, 2))), (512, color, "-")):
            uncertainty = 1e3 * _array(row["nested"][str(count)], "orientation_se", q,
                                       nonnegative=True)
            positive = valid & (uncertainty > 0)
            se_axis.plot(q[positive], uncertainty[positive], color=c,
                         linestyle=ls, linewidth=1.5)
        expected = 1e3 * _array(row["nested"]["32"], "orientation_se", q,
                               nonnegative=True) / 4
        positive = valid & (expected > 0)
        se_axis.plot(q[positive], expected[positive], color="#b6b6b6",
                     linestyle=(0, (1, 2)), linewidth=1.0, zorder=1)
        _axis(se_axis, q, ylabel=r'$10^3\,\mathrm{SE}[C_2]$')
        se_axis.set_yscale("log")
        for half_index, half in enumerate(row["halves"]):
            center = 1e3 * (_array(half, "C2", q) - base)
            error = 1e3 * _array(half, "orientation_se", q, nonnegative=True)
            halfcolor = color if half_index == 0 else "#626262"
            linestyle = "-" if half_index == 0 else (0, (4, 2))
            halves.fill_between(q[valid], (center-error)[valid], (center+error)[valid],
                                color=halfcolor, alpha=.11, linewidth=0, zorder=1)
            halves.plot(q[valid], center[valid], color=halfcolor, linestyle=linestyle,
                        linewidth=1.5, zorder=3)
        halves.axhline(0, color="#aaaaaa", linewidth=.7, zorder=1)
        _axis(halves, q, ylabel=r'$10^3\,\Delta C_2$')
        for j, ax in enumerate(axes[i]):
            ax.set_xticks((0, .05, .10, .15))
            ax.margins(y=.13)
            _panel(ax, "abcdef"[3*i+j])
            ax.text(.99, 1.05, _label(name, row), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color=color)
            if i == 0:
                ax.set_xlabel("")
    for ax, title in zip(axes[0], ("Nested central estimates", "Pointwise integration SE",
                                   "Two disjoint 256-pair halves")):
        ax.set_title(title, fontsize=8.5, pad=31)
    nested_handles = [Line2D([], [], color=nested_style[n][0], linestyle=nested_style[n][1],
                              linewidth=1) for n in NESTED_COUNTS[:-1]]
    nested_handles.append(Line2D([], [], color="#333333", linewidth=1.8))
    fig.legend(nested_handles, [f"n = {n}" for n in NESTED_COUNTS],
               loc="upper center", bbox_to_anchor=(.53, .94), ncol=5,
               fontsize=7.5, handlelength=2.3, columnspacing=1.3)
    fig.suptitle("Convergence of the orientation integral", y=.985, fontsize=11)
    fig.text(.095, .114,
             "Middle column: 32 pairs (grey dashed), 512 pairs (colour), and SE(32)/4 (grey dotted; scaling guide).\n"
             "Right column: first half (colour solid), second half (grey dashed); shading is each half's pointwise 1 SE.",
             fontsize=7, linespacing=1.5)
    _footer(fig, data, "Nested estimates are correlated. Separate vertical scales are used in the diagnostic panels.",
            x=.095, y=.024)
    return _save(fig, destination, "C2_orientation_convergence", plt)


def render_c2_convergence(data: dict[str, Any], destination: str | Path) -> dict[str, Any]:
    """Render final 512-pair curves and nested/subsample diagnostic plots."""
    q = np.asarray(data["q"], dtype=float)
    if q.ndim != 1 or len(q) < 2 or np.any(np.diff(q) <= 0) or q[0] < 0:
        raise ValueError("Expected an increasing, nonnegative momentum grid")
    if int(data["config"]["n_orientations"]) != 512:
        raise ValueError("This final-curve figure requires 512 orientation pairs")
    for name in ("spherical", *FAMILIES):
        row = data["cases"][name]
        expected = 512 if name in DEFORMATIONS else 1
        if int(row["n_orientations"]) != expected:
            raise ValueError(f"{name}: expected {expected} orientations")
        _array(row, "C2", q)
        _array(row, "orientation_se", q, nonnegative=True)
        _array(row, "C2_original32", q)
        _array(row, "orientation_se_original32", q, nonnegative=True)
        if name in DEFORMATIONS:
            for count in NESTED_COUNTS:
                entry = row["nested"][str(count)]
                _array(entry, "C2", q)
                _array(entry, "orientation_se", q, nonnegative=True)
            if len(row["halves"]) != 2:
                raise ValueError("Exactly two 256-pair half estimates are required")
            for half in row["halves"]:
                _array(half, "C2", q)
                _array(half, "orientation_se", q, nonnegative=True)
            if not np.allclose(row["nested"]["512"]["C2"], row["C2"],
                               rtol=0, atol=1e-13, equal_nan=True):
                raise ValueError("Nested 512 mean differs from the final curve")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plt = _style()
    files = (_absolute(data, destination, plt) + _delta(data, destination, plt)
             + _convergence(data, destination, plt))
    return {
        "files": files,
        "backend": "matplotlib",
        "editable_vector_text": True,
        "origin_omitted": True,
        "uncertainty_display": "Pointwise orientation Monte Carlo 1 SE; not simultaneous or experimental bands.",
        "absolute_figure": "Three structures with 512-pair deformation SE bands and a deterministic spherical reference.",
        "delta_figure": "Shared vertical scale across three structures; original 32-pair deformation centers are grey dashed.",
        "convergence_figure": "Nested central estimates, SE contraction, and two disjoint 256-pair half estimates; nested curves are correlated.",
        "SE_scaling_guide": "SE(32)/sqrt(512/32)=SE(32)/4 is a reference, not an imposed result.",
        "preset_momentum_windows": False,
    }
