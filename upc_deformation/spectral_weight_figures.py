"""Draw spectrum-weight observables directly extracted from saved panel d.

Figure contract: show where spectral probability moves among three common
momentum annuli, then compare two ratios of their yields.  This quantitative
grid uses the same window normalization for every case; it neither optimizes
the boundaries nor adds a physical model.  Curves are central estimates.
Discrete error bars denote the finite orientation integral only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .panel_d_ratio_figures import _style


ENDPOINTS = ("spherical", "beta2", "beta3", "skin")
FAMILIES = ("beta2", "beta3", "skin")
COLORS = {"spherical": "#626262", "beta2": "#26749d",
          "beta3": "#8670a4", "skin": "#bc6937"}
MARKERS = {"spherical": "o", "beta2": "s", "beta3": "^", "skin": "D"}
LINESTYLES = {"spherical": (0, (4, 2)), "beta2": "-",
              "beta3": (0, (5, 2, 1, 2)), "skin": (0, (2, 1))}


def _family(name: str) -> str:
    return name.removesuffix("_mid")


def _label(name: str, row: dict[str, Any]) -> str:
    family = _family(name)
    meta = row["case"]
    if family == "spherical":
        return "Spherical reference"
    if family == "beta2":
        return rf'$\beta_2={meta["beta2"]:g}$'
    if family == "beta3":
        return rf'$\beta_3={meta["beta3"]:g}$'
    return rf'$\Delta r_{{np}}={meta["skin_fm"]:g}$ fm'


def _panel(ax, letter: str) -> None:
    ax.text(-.14, 1.065, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=10, va="bottom")


def _clean(ax, axis="y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=axis, which="major", color="#e3e3e3", linewidth=.55)
    ax.tick_params(direction="out", length=3, width=.7)


def _window_label(edges: np.ndarray) -> str:
    return rf'${edges[0]:.2f}<p_T<{edges[-1]:.2f}$ GeV/$c$'


def _interval_labels(edges: np.ndarray) -> list[str]:
    return [f"{label}\n{lo:.2f}–{hi:.2f}" for label, lo, hi in
            zip(("L", "M", "H"), edges[:-1], edges[1:])]


def _window_background(ax, edges: np.ndarray) -> None:
    for i, (label, lo, hi) in enumerate(zip(("L", "M", "H"), edges[:-1], edges[1:])):
        ax.axvspan(lo, hi, color=("#eef1f3" if i != 1 else "#fbfaf7"),
                   linewidth=0, zorder=0)
        ax.text((lo + hi) / 2, .96, label, transform=ax.get_xaxis_transform(),
                va="top", ha="center", fontsize=8, color="#565656", weight="bold")
    for edge in edges[1:-1]:
        ax.axvline(edge, linewidth=.7, color="#b7b7b7", linestyle=(0, (3, 3)), zorder=1)


def _error_array(row: dict[str, Any], field: str, size: int) -> np.ndarray:
    errors = np.asarray(row[field + "_orientation_se"], dtype=float)
    if errors.shape != (size,) or not np.all(np.isfinite(errors)) or np.any(errors < 0):
        raise ValueError(f"{field}: expected {size} finite, nonnegative orientation SEs")
    return errors


def _points(ax, data, names, key, scale=1.0):
    size = 2 if key.startswith("ratio") else 3
    offsets = np.linspace(-.23, .23, len(names))
    handles = []
    for offset, name in zip(offsets, names):
        row = data["cases"][name]
        values = np.asarray(row[key], dtype=float)
        if values.shape != (size,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name}.{key}: invalid values")
        errors = _error_array(row, key, size)
        handles.append(ax.errorbar(
            np.arange(size) + offset, scale * values, yerr=scale * errors,
            linestyle="none", marker=MARKERS[name], color=COLORS[name],
            markersize=5, markeredgewidth=.8, elinewidth=.9,
            capsize=2.4, capthick=.8, zorder=4,
            label=_label(name, row)))
    return handles


def _footer(fig, edges, y=.035):
    fig.text(.105, y,
             r'Panel d: $b=12$–$60$ fm (2 fm step), $|y|<1$; common window '
             + _window_label(edges) + ".\n"
             "Bars: 1 SE of the orientation integral (32 pairs for each deformation); sphere and skin are deterministic.\n"
             "Exploratory fixed intervals; no detector uncertainty or numerical-convergence error is included.",
             fontsize=7, linespacing=1.5, va="bottom")


def _save(fig, destination, stem, plt):
    files = []
    for ext in ("png", "pdf", "svg"):
        path = destination / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.08)
        files.append(str(path))
    plt.close(fig)
    return files


def _overview(data, destination, plt):
    edges = np.asarray(data["edges_GeV"], dtype=float)
    q = np.asarray(data["radii"], dtype=float)
    fig = plt.figure(figsize=(8.85, 7.15))
    grid = fig.add_gridspec(2, 2, left=.105, right=.97, bottom=.20, top=.87,
                           hspace=.49, wspace=.36)
    density, shape, fractions, relative = [fig.add_subplot(grid[i, j])
                                          for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    curve_handles = []
    for ax in (density, shape):
        _window_background(ax, edges)
        ax.set(xlim=(edges[0], edges[-1]), xlabel=r'$p_T$ (GeV/$c$)',
               xticks=edges)
        _clean(ax)
    for name in ENDPOINTS:
        row = data["cases"][name]
        density_curve = np.asarray(row["spectrum"], dtype=float)
        ratio_curve = np.asarray(row["spectrum_ratio_to_spherical"], dtype=float)
        if density_curve.shape != q.shape or ratio_curve.shape != q.shape:
            raise ValueError(f"{name}: spectrum and radius arrays differ")
        if not np.all(np.isfinite(density_curve)) or np.any(density_curve <= 0):
            raise ValueError(f"{name}: spectrum is not positive and finite")
        if not np.all(np.isfinite(ratio_curve)):
            raise ValueError(f"{name}: spectrum ratio is nonfinite")
        line, = density.plot(q, density_curve, color=COLORS[name],
                             linestyle=LINESTYLES[name], linewidth=1.55,
                             label=_label(name, row), zorder=3)
        curve_handles.append(line)
        shape.plot(q, ratio_curve, color=COLORS[name],
                   linestyle=LINESTYLES[name], linewidth=1.55, zorder=3)
    density.set(yscale="log", ylabel=r'$P(p_T)$  [$(\mathrm{GeV}/c)^{-1}$]',
                title="Momentum spectrum in the common window")
    shape.set(ylabel=r'$P(p_T)/P_0(p_T)$', title="Spectrum shape relative to the sphere")
    density.margins(y=.13)
    shape.margins(y=.16)

    _points(fractions, data, ENDPOINTS, "fractions", scale=100)
    _points(relative, data, FAMILIES, "fraction_relative_change", scale=100)
    for ax in (fractions, relative):
        ax.set(xticks=np.arange(3), xticklabels=_interval_labels(edges), xlim=(-.53, 2.53),
               xlabel=r'Common $p_T$ interval (GeV/$c$)')
        ax.tick_params(axis="x", length=0, pad=6)
        _clean(ax)
        ax.margins(y=.18)
    fraction_top = max(float(np.max(100 * (np.asarray(data["cases"][n]["fractions"])
                       + _error_array(data["cases"][n], "fractions", 3))))
                       for n in ENDPOINTS)
    fraction_top = max(5., float(5 * np.ceil(1.12 * fraction_top / 5)))
    fractions.set(ylim=(0, fraction_top), ylabel=r'Event fraction $100f_i$ (%)',
                  title="How many events lie in each interval?")
    relative.set(ylabel=r'$100(f_i/f_{i,0}-1)$ (%)',
                 title="Redistribution among intervals")
    relative.axhline(0, color="#8c8c8c", linewidth=.8, zorder=2)
    for ax, letter in zip((density, shape, fractions, relative), "abcd"):
        _panel(ax, letter)
    fig.suptitle("Spectrum weights in three momentum intervals", y=.985,
                 fontsize=11)
    fig.legend(curve_handles, [_label(n, data["cases"][n]) for n in ENDPOINTS],
               ncol=4, loc="upper center", bbox_to_anchor=(.54, .948),
               columnspacing=1.5, handlelength=2.8, fontsize=8)
    _footer(fig, edges)
    return _save(fig, destination, "spectral_weights", plt)


def _response(data, destination, plt):
    from matplotlib.lines import Line2D
    edges = np.asarray(data["edges_GeV"], dtype=float)
    fig = plt.figure(figsize=(8.85, 5.6))
    grid = fig.add_gridspec(1, 2, left=.105, right=.97, bottom=.33, top=.82,
                           wspace=.36)
    ratio, plane = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    _points(ratio, data, FAMILIES, "ratio_relative_change", scale=100)
    ratio.set(xticks=(0, 1), xticklabels=(r'$R_1=N_M/N_L$', r'$R_2=N_H/N_M$'),
              xlim=(-.5, 1.5), ylabel=r'$100(R_j/R_{j,0}-1)$ (%)',
              title="Two yield ratios")
    ratio.tick_params(axis="x", length=0, pad=8)
    ratio.axhline(0, color="#8c8c8c", linewidth=.8, zorder=2)
    ratio.margins(y=.2)
    _clean(ratio)

    plane.axhline(0, color="#aaaaaa", linewidth=.7, zorder=1)
    plane.axvline(0, color="#aaaaaa", linewidth=.7, zorder=1)
    parameter_notes = []
    for family in FAMILIES:
        names = ([family + "_mid"] if family + "_mid" in data["cases"] else []) + [family]
        points = np.asarray([[0., 0.]] + [100 * np.asarray(data["cases"][n]["ratio_relative_change"])
                                        for n in names])
        plane.plot(points[:, 0], points[:, 1], color=COLORS[family], linewidth=1,
                    alpha=.75, zorder=2)
        for name in names:
            row = data["cases"][name]
            point = 100 * np.asarray(row["ratio_relative_change"])
            error = 100 * _error_array(row, "ratio_relative_change", 2)
            middle = name.endswith("_mid")
            plane.errorbar(*point, xerr=error[0], yerr=error[1],
                           marker=MARKERS[family], linestyle="none", color=COLORS[family],
                           markerfacecolor="white" if middle else COLORS[family],
                           markeredgewidth=1, markersize=5 if middle else 6,
                           elinewidth=.8, capsize=2, capthick=.7, zorder=4)
        mid_note = _label(names[0], data["cases"][names[0]]) if len(names) == 2 else ""
        parameter_notes.append(mid_note)
    plane.plot(0, 0, marker="o", color=COLORS["spherical"], markersize=4.5, zorder=5)
    plane.set(xlabel=r'$100(R_1/R_{1,0}-1)$ (%)',
              ylabel=r'$100(R_2/R_{2,0}-1)$ (%)',
              title="Structural responses in the ratio plane")
    _clean(plane, axis="both")
    plane.margins(x=.22, y=.2)
    _panel(ratio, "a")
    _panel(plane, "b")
    fig.suptitle("Ratios of spectrum weights", y=.986, fontsize=11)
    handles = [Line2D([], [], color=COLORS[n], marker=MARKERS[n], linewidth=1,
                      label=_label(n, data["cases"][n])) for n in FAMILIES]
    fig.legend(handles, [_label(n, data["cases"][n]) for n in FAMILIES], ncol=3,
               loc="upper center", bbox_to_anchor=(.54, .947),
               columnspacing=2.6, handlelength=2, fontsize=8)
    fig.text(.105, .177, "Open markers: " + "; ".join(v for v in parameter_notes if v)
             + ".  Filled markers: legend values.\n"
             "The grey origin is the spherical reference; connecting segments are guides, not fitted parameter curves.",
             fontsize=7, linespacing=1.5)
    _footer(fig, edges, y=.032)
    return _save(fig, destination, "spectral_ratios", plt)


def render_spectral_weights(data: dict[str, Any], destination: str | Path) -> dict[str, Any]:
    """Render common-window spectra, interval fractions, and two yield ratios."""
    edges = np.asarray(data["edges_GeV"], dtype=float)
    if edges.shape != (4,) or np.any(np.diff(edges) <= 0):
        raise ValueError("Exactly three common, increasing intervals are required")
    for name in ENDPOINTS:
        if name not in data["cases"]:
            raise ValueError(f"Missing {name}")
    for name, row in data["cases"].items():
        expected = 32 if _family(name) in ("beta2", "beta3") else 1
        if int(row["n_orientations"]) != expected:
            raise ValueError(f"{name}: expected the original {expected}-orientation cache")
        fractions = np.asarray(row["fractions"], dtype=float)
        if fractions.shape != (3,) or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
            raise ValueError(f"{name}: invalid three-bin event fractions")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plt = _style()
    files = _overview(data, destination, plt) + _response(data, destination, plt)
    return {
        "files": files,
        "backend": "matplotlib",
        "editable_vector_text": True,
        "common_bin_edges_GeV": edges.tolist(),
        "curve_uncertainty_displayed": False,
        "uncertainty_display": "Marginal 1 SE of the finite orientation integral; no detector or numerical-convergence uncertainty.",
        "curves": "Four endpoint/reference curves with a common window normalization; central values only.",
        "response_plane": "Baseline, midpoint and endpoint per family; segments are guides, not fits.",
        "boundary_selection": "Common exploratory intervals supplied by the analysis configuration; no visual optimization.",
        "source": "Saved panel-d intensity arrays; annulus integrals provide event fractions and ratios.",
        "logarithmic_axes": ["spectral_weights:a:y"],
    }
