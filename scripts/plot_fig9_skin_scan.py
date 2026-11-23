#!/usr/bin/env python3
"""Plot the fixed-diffuseness skin scan and the no-skin deformation scan of Fig. 9."""
import argparse
import csv
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


FEATURES = ["Q10_GeV", "K_tail"]
SKIN_GRID = np.arange(6) * 0.04
BETA2_GRID = np.linspace(0, 0.30, 10)
BETA3_CURVES = [
    (0, "0", "#267CB3", "o", "-"),
    (1 / 15, "$1/15$", "#EF8135", "s", "--"),
    (2 / 15, "$2/15$", "#42934A", "^", "-."),
    (0.20, "0.2", "#C8383A", "D", ":"),
]


class Point(NamedTuple):
    panel_column: str
    delta_r_np_fm: float
    beta2: float
    beta3: float
    Q10_GeV: float
    K_tail: float
    source: str


def load_points(reference_file, new_file):
    left, deformation = [], []
    endpoints = {"none": 0.0, "skin": 0.20}
    with reference_file.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["branch"] != "full" or float(row["b_step_fm"]) != 0.5:
                continue
            density = row["density_id"]
            beta2, beta3 = float(row["beta2"]), float(row["beta3"])
            q10, k_tail = float(row["Q10_GeV"]), float(row["K_tail"])
            if density in endpoints and beta2 == beta3 == 0:
                left.append(Point("skin", endpoints[density], beta2, beta3, q10, k_tail, "archived_grid"))
            if density == "none":
                deformation.append(Point("deformation", 0.0, beta2, beta3, q10, k_tail, "archived_grid"))
    with new_file.open(newline="") as stream:
        for row in csv.DictReader(stream):
            left.append(Point("skin", float(row["delta_r_np_fm"]), 0.0, 0.0,
                              float(row["Q10_GeV"]), float(row["K_tail"]), "new_spherical_scan"))
    left.sort(key=lambda p: p.delta_r_np_fm)
    if len(left) != len(SKIN_GRID) or not np.allclose(
        [p.delta_r_np_fm for p in left], SKIN_GRID, atol=1e-8, rtol=0
    ):
        raise ValueError("Fig. 9 requires all six skin thicknesses, including the archived 0 and 0.20 fm endpoints")

    points = list(left)
    for beta3, *_ in BETA3_CURVES:
        curve = sorted((p for p in deformation if abs(p.beta3 - beta3) <= 1e-10), key=lambda p: p.beta2)
        if len(curve) != len(BETA2_GRID) or not np.allclose(
            [p.beta2 for p in curve], BETA2_GRID, atol=1e-10, rtol=0
        ):
            raise ValueError(f"Incomplete no-skin deformation curve for beta3={beta3}")
        points.extend(curve)
    if not np.isfinite([(p.Q10_GeV, p.K_tail) for p in points]).all():
        raise ValueError("A plotted crossing or tail-slope ratio is undefined")
    return points


def draw_curve(main, ratio, x, values, baseline, **style):
    line = dict(linewidth=1.2, markeredgecolor="#222222", markeredgewidth=0.35, zorder=4, **style)
    main.plot(x, values, markersize=4.5, **line)
    ratio.plot(x, np.asarray(values) / baseline, markersize=3.6, **line)


def build_figure(points, sphere):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 9.5, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 0.85,
        "legend.fontsize": 8.5, "legend.frameon": False, "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.2, 7.35))
    rows = fig.add_gridspec(2, 1, left=0.105, right=0.983, bottom=0.078, top=0.967, hspace=0.32)
    left = [p for p in points if p.panel_column == "skin"]
    right = [p for p in points if p.panel_column == "deformation"]
    for row, feature in enumerate(FEATURES):
        grid = rows[row].subgridspec(2, 2, width_ratios=[0.465, 0.535],
                                    height_ratios=[3, 1], wspace=0, hspace=0.035)
        skin = fig.add_subplot(grid[0, 0])
        deformation = fig.add_subplot(grid[0, 1], sharey=skin)
        skin_ratio = fig.add_subplot(grid[1, 0], sharex=skin)
        deformation_ratio = fig.add_subplot(grid[1, 1], sharex=deformation, sharey=skin_ratio)

        baseline = getattr(sphere, feature)
        draw_curve(skin, skin_ratio, [p.delta_r_np_fm for p in left], [getattr(p, feature) for p in left],
                   baseline, color="#BD7150", marker="o", markerfacecolor="#E99572")
        for beta3, label, color, marker, style in BETA3_CURVES:
            curve = [p for p in right if abs(p.beta3 - beta3) <= 1e-10]
            draw_curve(deformation, deformation_ratio, [p.beta2 for p in curve],
                       [getattr(p, feature) for p in curve], baseline,
                       label=label, color=color, marker=marker, linestyle=style)

        for ax in (skin, deformation, skin_ratio, deformation_ratio):
            ax.tick_params(direction="in", top=True, right=True, length=3.8, width=0.8, pad=3)
        for ax in (deformation, deformation_ratio):
            ax.tick_params(labelleft=False)
        for ax in (skin, deformation):
            ax.tick_params(labelbottom=False)
        for ax in (skin_ratio, deformation_ratio):
            ax.axhline(1, color=".60", linestyle="--", linewidth=0.8, zorder=1)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        skin_ratio.set_xticks(SKIN_GRID)
        skin_ratio.set_xlim(-0.012, 0.214)
        deformation_ratio.set_xticks(np.arange(0, 0.301, 0.05))
        deformation_ratio.set_xlim(-0.014, 0.314)
        skin_ratio.set_xlabel(r"Neutron-skin thickness $\Delta r_{np}$ [fm]", labelpad=6)
        deformation_ratio.set_xlabel(r"Quadrupole deformation $\beta_2$", labelpad=6)
        skin_ratio.set_ylabel("Ratio")
        skin_ratio.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if feature == "Q10_GeV":
            skin.set_ylabel(r"$p_{T,10}$ [GeV]")
            skin.set_yticks([0.115, 0.116, 0.117, 0.118, 0.119])
            # Allow room above the no-skin curves for the two-row legend.
            skin.set_ylim(0.1144, 0.11985)
            skin.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            skin_ratio.set_yticks([0.97, 0.98, 0.99, 1])
            skin_ratio.set_ylim(0.97, 1.004)
        else:
            skin.set_ylabel(r"$K_{\mathrm{tail}}$")
            skin.set_yticks([0.90, 0.95, 1, 1.05, 1.10, 1.15])
            skin.set_ylim(0.895, 1.165)
            skin.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            skin_ratio.set_yticks([0.80, 0.90, 1])
            skin_ratio.set_ylim(0.80, 1.015)
        deformation.legend(title=r"$\beta_3$", loc="upper right", ncol=2, columnspacing=0.9,
                           handlelength=2.2, handletextpad=0.55, borderaxespad=0.5,
                           labelspacing=0.32, title_fontsize=10)
        pos = deformation.get_position()
        fig.text(pos.x0, pos.y1 + 0.009, f"({chr(97 + row)})", fontsize=12,
                 fontweight="bold", ha="center", va="bottom")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Archived Fig. 9 source-grid rows")
    parser.add_argument("--new-points", type=Path, required=True, help="Four new spherical skin spectra summarized as CSV")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = load_points(args.reference, args.new_points)
    sphere = next(p for p in points if p.panel_column == "skin" and p.delta_r_np_fm == 0)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "fig9_plot_data.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow((*Point._fields, "Q10_GeV_ratio", "K_tail_ratio"))
        for point in points:
            writer.writerow((*point, point.Q10_GeV / sphere.Q10_GeV, point.K_tail / sphere.K_tail))
    fig = build_figure(points, sphere)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(args.output / f"fig9_skin_deformation.{extension}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
