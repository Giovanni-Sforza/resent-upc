#!/usr/bin/env python3
"""Plot Q10 and K_tail with deformation-grid boxes and optional density ranges."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "upc_shape_figure_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import numpy as np
import pandas as pd


DENSITIES = ("halo", "skin", "none")
FEATURES = ("Q10_GeV", "K_tail")
BOX_COLORS = ("#73B6A2", "#E99572", "#94A2C3")
LINE_COLORS = ("#267CB3", "#EF8135", "#42934A", "#C8383A")
MARKERS = ("o", "s", "^", "D")
STYLES = ("-", "--", "-.", ":")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(source):
    metrics = source / "metrics.csv"
    manifest = json.loads((source / "report_manifest.json").read_text())
    if sha256(metrics) != manifest["output_sha256"]["metrics.csv"]:
        raise ValueError("metrics.csv does not match the returned report manifest")
    all_rows = pd.read_csv(metrics)
    finest = float(all_rows.b_step_fm.min())
    data = all_rows[(all_rows.branch == "full") & (all_rows.b_step_fm == finest)].copy()
    assert len(data) == 300 and set(data.density_id) == set(DENSITIES)
    assert not data.duplicated(["density_id", "beta2", "beta3"]).any()
    assert (data.status == "observed").all()
    assert np.isfinite(data[list(FEATURES)].to_numpy()).all()
    for density in DENSITIES:
        group = data[data.density_id == density]
        assert len(group) == 100 and group.beta2.nunique() == group.beta3.nunique() == 10
    sphere = data[(data.density_id == "none") & (data.beta2 == 0) & (data.beta3 == 0)]
    assert len(sphere) == 1
    references = {name: float(sphere.iloc[0][name]) for name in FEATURES}
    assert all(value > 0 for value in references.values())
    return data, references, finest


def make_source_tables(data, references):
    columns = ["case", "density_id", "beta2", "beta3", "branch", "b_step_fm", "n_orientations"]
    columns += list(FEATURES)
    source = data[columns].sort_values(["density_id", "beta2", "beta3"]).copy()
    for feature in FEATURES:
        source[feature + "_ratio"] = source[feature] / references[feature]
    beta3 = np.sort(data.beta3.unique())[[0, 3, 6, 9]]
    curves, boxes = [], []
    for feature in FEATURES:
        baseline = references[feature]
        for density in DENSITIES:
            values = data.loc[data.density_id == density, feature].to_numpy()
            q1, median, q3 = np.quantile(values, [.25, .5, .75])
            boxes.append(dict(feature=feature, density_id=density, n_grid_points=len(values),
                minimum=float(values.min()), q25=float(q1), median=float(median),
                q75=float(q3), maximum=float(values.max()), reference=baseline))
        for b3 in beta3:
            for b2, group in data[data.beta3 == b3].groupby("beta2", sort=True):
                values = group.set_index("density_id")[feature]
                assert set(values.index) == set(DENSITIES) and len(values) == 3
                center, low, high = float(values.mean()), float(values.min()), float(values.max())
                curves.append(dict(feature=feature, beta2=float(b2), beta3=float(b3),
                    n_variants=3, center=center, minimum=low, maximum=high,
                    error_lower=center-low, error_upper=high-center,
                    center_ratio=center/baseline, minimum_ratio=low/baseline,
                    maximum_ratio=high/baseline,
                    error_lower_ratio=(center-low)/baseline,
                    error_upper_ratio=(high-center)/baseline,
                    halo=float(values["halo"]), skin=float(values["skin"]),
                    none=float(values["none"]), reference=baseline))
    return source, pd.DataFrame(curves), pd.DataFrame(boxes), beta3


def draw_boxes(ax, boxes, feature, ratio=False):
    stats = []
    for density in DENSITIES:
        row = boxes[(boxes.feature == feature) & (boxes.density_id == density)].iloc[0]
        scale = row.reference if ratio else 1.
        stats.append(dict(label=density, med=row["median"]/scale, q1=row.q25/scale,
            q3=row.q75/scale, whislo=row.minimum/scale, whishi=row.maximum/scale,
            fliers=[]))
    artists = ax.bxp(stats, positions=[1, 2, 3], widths=.47, patch_artist=True,
        showfliers=False, manage_ticks=False,
        boxprops={"linewidth": .9, "edgecolor": "#555555"},
        medianprops={"linewidth": 1., "color": "#555555"},
        whiskerprops={"linewidth": .85, "color": "#555555"},
        capprops={"linewidth": .85, "color": "#555555"})
    for box, color in zip(artists["boxes"], BOX_COLORS):
        box.set_facecolor(color)
    ax.set_xlim(.5, 3.5)
    ax.set_xticks([1, 2, 3], ["halo type", "skin type", "none"])


def draw_curves(ax, curves, feature, beta3, ratio=False, show_ranges=True):
    suffix = "_ratio" if ratio else ""
    for b3, color, marker, style in zip(beta3, LINE_COLORS, MARKERS, STYLES):
        rows = curves[(curves.feature == feature) & (curves.beta3 == b3)].sort_values("beta2")
        # Range display does not change the means or the exact registered x.
        if show_ranges:
            ax.errorbar(rows.beta2, rows["center"+suffix],
                yerr=np.vstack([rows["error_lower"+suffix], rows["error_upper"+suffix]]),
                fmt="none", ecolor=matplotlib.colors.to_rgba(color, .48),
                elinewidth=.7 if ratio else .8, capsize=2.2, capthick=.7, zorder=2)
        ax.plot(rows.beta2, rows["center"+suffix], color=color, linestyle=style,
            marker=marker, markersize=3.6 if ratio else 4.5, linewidth=1.2,
            markeredgecolor="#222222", markeredgewidth=.35, zorder=4)
    ax.set_xlim(-.014, .314)
    ax.xaxis.set_major_locator(FixedLocator(np.arange(0, .301, .05)))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def style_axis(ax):
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   length=3.8, width=.8, pad=3)
    for spine in ax.spines.values():
        spine.set_linewidth(.85)


def build_figure(curves, boxes, beta3, *, show_ranges=True):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 9.5, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8.5, "legend.frameon": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": True, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(7.2, 7.35), dpi=160)
    outer = fig.add_gridspec(2, 1, left=.105, right=.983, bottom=.078,
                            top=.967, hspace=.32)
    handles = [Line2D([], [], color=c, marker=m, linestyle=s, linewidth=1.2,
        markersize=4.5, markeredgecolor="#222222", markeredgewidth=.35)
        for c, m, s in zip(LINE_COLORS, MARKERS, STYLES)]
    labels = ["0", r"$1/15$", r"$2/15$", "0.2"]
    axes, limits = {}, {}
    for row, feature in enumerate(FEATURES):
        grid = outer[row].subgridspec(2, 2, width_ratios=[.465, .535],
                                    height_ratios=[3., 1.], wspace=0, hspace=.035)
        left = fig.add_subplot(grid[0, 0])
        right = fig.add_subplot(grid[0, 1], sharey=left)
        left_ratio = fig.add_subplot(grid[1, 0], sharex=left)
        right_ratio = fig.add_subplot(grid[1, 1], sharex=right, sharey=left_ratio)
        for ax in (left, right, left_ratio, right_ratio):
            style_axis(ax)
        for ax in (right, right_ratio):
            ax.tick_params(labelleft=False)
        for ax in (left, right):
            ax.tick_params(labelbottom=False)
        draw_boxes(left, boxes, feature)
        draw_boxes(left_ratio, boxes, feature, ratio=True)
        draw_curves(right, curves, feature, beta3, show_ranges=show_ranges)
        draw_curves(right_ratio, curves, feature, beta3, ratio=True, show_ranges=show_ranges)
        if feature == "Q10_GeV":
            left.set_ylabel(r"$Q_{10}$ [GeV]")
            left.set_ylim(.1144, .11925)
            left.yaxis.set_major_locator(FixedLocator([.115, .116, .117, .118, .119]))
            left.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            left_ratio.set_ylim(.970, 1.004)
            left_ratio.yaxis.set_major_locator(FixedLocator([.97, .98, .99, 1.]))
            left_ratio.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        else:
            left.set_ylabel(r"$K_{\mathrm{tail}}$")
            left.set_ylim(.895, 1.165)
            left.yaxis.set_major_locator(FixedLocator([.90, .95, 1., 1.05, 1.10, 1.15]))
            left.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            left_ratio.set_ylim(.80, 1.015)
            left_ratio.yaxis.set_major_locator(FixedLocator([.8, .9, 1.]))
            left_ratio.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        for ax in (left_ratio, right_ratio):
            ax.axhline(1., color=".60", linestyle="--", linewidth=.8, zorder=1)
        left_ratio.set_ylabel("Ratio")
        left_ratio.set_xlabel("Neutron-skin type", labelpad=6)
        right_ratio.set_xlabel(r"Quadrupole deformation $\beta_2$", labelpad=6)
        right.legend(handles, labels, title=r"$\beta_3$", loc="upper right", ncol=2,
            columnspacing=.9, handlelength=2.2, handletextpad=.55, borderaxespad=.5,
            labelspacing=.32, title_fontsize=10)
        position = right.get_position()
        fig.text(position.x0, position.y1+.009, f"({chr(97+row)})",
                 fontsize=12, fontweight="bold", ha="center", va="bottom")
        axes[feature] = (left, right, left_ratio, right_ratio)
        limits[feature] = {"absolute": list(left.get_ylim()), "ratio": list(left_ratio.get_ylim())}
    fig.canvas.draw()
    return fig, axes, limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-density-ranges", action="store_true",
                        help="Show density-averaged curves without vertical min-max bars")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data, references, finest = load_source(args.analysis)
    source, curves, boxes, beta3 = make_source_tables(data, references)
    show_ranges = not args.no_density_ranges
    source.to_csv(args.output/"source_grid.csv", index=False, float_format="%.17g")
    curves.to_csv(args.output/"source_deformation_ranges.csv", index=False, float_format="%.17g")
    boxes.to_csv(args.output/"source_box_statistics.csv", index=False, float_format="%.17g")
    contract = {
        "conclusion": "Compare density-variant dependence and deformation dependence of Q10 and K_tail within the registered model grid.",
        "archetype": "quantitative grid", "backend": "Python/matplotlib",
        "layout": "Two stacked observables, each with left density boxes and right deformation curves, plus ratio strips; no inset.",
        "branch": "full", "b_step_fm": finest,
        "n_cases": len(data), "n_grid_points_per_density": 100,
        "n_orientation_pairs_deformed": 2048, "n_orientation_pairs_spherical": 1,
        "density_order": list(DENSITIES), "selected_beta3_indices": [0, 3, 6, 9],
        "selected_beta3": beta3.tolist(), "beta3_legend": ["0", "1/15", "2/15", "0.2"],
        "line_center": "unweighted arithmetic mean of none, skin, halo at the same beta2 and beta3",
        "density_ranges_shown": show_ranges,
        "vertical_bar": ("minimum to maximum of the same three density variants; asymmetric about the mean; not a statistical uncertainty"
                         if show_ranges else "not shown; density-specific values and ranges remain in the source tables"),
        "x_positions": "exact registered beta2 values, no offsets or interpolated parameter points",
        "box": "25th to 75th percentiles of 100 equally weighted registered deformation-grid values",
        "box_center": "median", "box_whiskers": "minimum and maximum of the 100 values",
        "ratio_reference": "none, beta2=beta3=0, same full branch and b grid",
        "reference_values": references,
        "uncertainty_note": ("Parameter-grid spread and density-variant ranges only. Orientation MC errors are not shown."
                             if show_ranges else "Left boxes show parameter-grid spread. Right panels show means only; density-variant ranges and orientation MC errors are not shown."),
        "source_sha256": {name: sha256(args.analysis/name) for name in ("metrics.csv", "report_manifest.json")},
        "figure_size_inches": [7.2, 7.35], "exports": ["PNG 300 dpi", "PDF TrueType fonts", "SVG editable text"],
        "reference_style": "Serif fonts, colored boxes, four line/marker encodings, shared scales and compact ratio strips.",
    }
    (args.output/"figure_contract.json").write_text(json.dumps(contract, indent=2)+"\n")
    fig, axes, limits = build_figure(curves, boxes, beta3, show_ranges=show_ranges)
    qa = {"source_cases": len(data), "deformation_points_per_observable": 40,
          "all_intervals_contain_center": True, "all_data_within_axes": True,
          "axes_count": len(fig.axes), "no_insets": len(fig.axes) == 8,
          "shared_scales_per_row": True, "limits": limits}
    from matplotlib.container import ErrorbarContainer
    counts = [sum(isinstance(item, ErrorbarContainer) for item in ax.containers)
              for row_axes in axes.values() for ax in (row_axes[1], row_axes[3])]
    assert counts == ([4]*4 if show_ranges else [0]*4)
    qa["right_panel_errorbar_container_counts"] = counts
    qa["density_ranges_shown"] = show_ranges
    for feature in FEATURES:
        group = curves[curves.feature == feature]
        assert (group.minimum <= group.center).all() and (group.center <= group.maximum).all()
        assert np.allclose(group.error_lower, group.center-group.minimum, rtol=0, atol=1e-15)
        assert np.allclose(group.error_upper, group.maximum-group.center, rtol=0, atol=1e-15)
        for scale in ("absolute", "ratio"):
            values = data[feature] / (references[feature] if scale == "ratio" else 1.)
            lo, hi = limits[feature][scale]
            assert lo <= values.min() <= values.max() <= hi
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output/f"q10_ktail_comparison.{extension}", dpi=300,
                    metadata={"Creator": "Matplotlib"} if extension != "png" else None)
    plt.close(fig)
    range_caption = ("Vertical bars span the minimum and maximum of those three values and are not statistical uncertainties. "
                     if show_ranges else "Only the mean curves are displayed; density-variant ranges and orientation Monte Carlo uncertainties are not shown. ")
    caption = (
        "Density and deformation dependence of Q10 and K_tail. (a) Q10; (b) K_tail. "
        "All results use the full spectrum and an impact-parameter integration step of 0.5 fm. "
        "Left: each box summarizes the 100 uniformly sampled (beta2, beta3) grid points for one "
        "density variant (halo, skin, none); boxes show the 25th and 75th percentiles, horizontal "
        "lines show medians, and whiskers span the minimum and maximum. "
        "Right: at each registered beta2, curves show the unweighted arithmetic mean over the "
        "three density variants at fixed beta3. "
        + range_caption + "The four beta3 values are "
        "0, 1/15, 2/15 and 0.2; fractions correspond to 0.0666667 and 0.1333333. "
        "Every lower panel shows the same quantity divided by its none spherical value "
        f"(Q10_0={references['Q10_GeV']:.12g} GeV; K_tail,0={references['K_tail']:.12g}). "
        "The dashed line marks this common reference, not the mean of the three spherical variants. "
        "Each deformed case uses 2048 orientation pairs; each spherical case uses one deterministic "
        "calculation. The spreads shown describe the registered model grid, not event or orientation "
        "sampling errors. Skin and halo are fixed Delta r_np=0.20 fm variants.\n"
    )
    (args.output/"caption.txt").write_text(caption)
    qa["output_sha256"] = {p.name: sha256(p) for p in args.output.glob("q10_ktail_comparison.*")}
    qa["source_unchanged"] = sha256(args.analysis/"metrics.csv") == contract["source_sha256"]["metrics.csv"]
    assert qa["source_unchanged"]
    (args.output/"numeric_qa.json").write_text(json.dumps(qa, indent=2)+"\n")
    shutil.copy2(__file__, args.output/Path(__file__).name)
    print(json.dumps({"output": str(args.output), "reference_values": references,
                      "beta3": beta3.tolist(), "axes": len(fig.axes)}, indent=2))


if __name__ == "__main__":
    main()
