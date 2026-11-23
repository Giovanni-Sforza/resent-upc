#!/usr/bin/env python3
"""Plot the equal-skin-thickness profile ranges and deformation estimates of Fig. 9."""
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
PROFILE_FIELDS = ("profile", "target_delta_r_np_fm", "delta_r_np_fm", "R_ws_p_fm", "R_ws_n_fm",
                  "a_p_fm", "a_n_fm", "proton_rms_fm", "neutron_rms_fm", *FEATURES, "source")
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
    interval_kind: str
    Q10_GeV_low: float
    Q10_GeV_high: float
    K_tail_low: float
    K_tail_high: float


def load_skin_profiles(existing_file, new_file):
    grouped = {}
    for path, source in ((existing_file, "existing_spherical_scan"), (new_file, "new_spherical_scan")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                target = float(row["target_delta_r_np_fm"])
                profile = row["profile"]
                rows = grouped.setdefault(target, {})
                if profile in rows:
                    raise ValueError(f"Duplicate {profile} profile at {target} fm")
                if abs(float(row["delta_r_np_fm"]) - target) > 2e-9:
                    raise ValueError(f"Achieved RMS difference for {profile} differs from {target} fm")
                row["source"] = row.get("source", source)
                rows[profile] = row

    if set(grouped) != set(SKIN_GRID):
        raise ValueError(f"Expected six target skin thicknesses on {SKIN_GRID}")
    left, profiles = [], []
    for target in SKIN_GRID:
        rows = grouped[target]
        # All three prescriptions coincide with the one computed zero-skin sphere.
        profile_names = ("radius",) if target == 0 else ("radius", "mixed", "diffuseness")
        if set(rows) != set(profile_names):
            raise ValueError(f"Expected {profile_names} for skin thickness {target} fm, got {tuple(rows)}")
        central = rows["radius" if target == 0 else "mixed"]
        bounds = []
        for feature in FEATURES:
            values = [float(rows[name][feature]) for name in profile_names]
            if not np.isfinite(values).all():
                raise ValueError(f"Undefined {feature} at {target} fm")
            bounds.extend((min(values), max(values)))
        left.append(Point("skin", target, 0.0, 0.0, float(central["Q10_GeV"]), float(central["K_tail"]),
                          central["source"], "profile_range", *bounds))
        profiles.extend(rows[name] for name in profile_names)
    return left, profiles


def load_points(reference_file, existing_file, new_file):
    points, profiles = load_skin_profiles(existing_file, new_file)
    deformation = []
    with reference_file.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["branch"] != "full" or float(row["b_step_fm"]) != 0.5 or row["density_id"] != "none":
                continue
            beta2, beta3 = float(row["beta2"]), float(row["beta3"])
            q10, k_tail = float(row["Q10_GeV"]), float(row["K_tail"])
            q_se, k_se = (float(row[f"{f}_orientation_se"]) for f in FEATURES)
            if not np.isfinite([q10, k_tail, q_se, k_se]).all() or min(q_se, k_se) < 0:
                raise ValueError(f"Invalid observable or orientation standard error for {row['case']}")
            deformation.append(Point("deformation", 0.0, beta2, beta3, q10, k_tail, "archived_grid",
                                     "orientation_1se", q10-q_se, q10+q_se, k_tail-k_se, k_tail+k_se))
    for beta3, *_ in BETA3_CURVES:
        curve = sorted((p for p in deformation if abs(p.beta3 - beta3) <= 1e-10), key=lambda p: p.beta2)
        if len(curve) != len(BETA2_GRID) or not np.allclose(
            [p.beta2 for p in curve], BETA2_GRID, atol=1e-10, rtol=0
        ):
            raise ValueError(f"Incomplete no-skin deformation curve for beta3={beta3}")
        points.extend(curve)
    return points, profiles


def draw_curve(main, ratio, x, points, feature, baseline, **style):
    values = np.array([getattr(p, feature) for p in points])
    low = np.array([getattr(p, f"{feature}_low") for p in points])
    high = np.array([getattr(p, f"{feature}_high") for p in points])
    errors = np.stack((values - low, high - values))
    nonzero = high > low
    line = dict(linewidth=1.2, markeredgecolor="#222222", markeredgewidth=0.35, zorder=4, **style)
    for ax, scale, marker_size in ((main, 1.0, 4.5), (ratio, baseline, 3.6)):
        ax.errorbar(np.asarray(x)[nonzero], values[nonzero] / scale, yerr=errors[:, nonzero] / scale,
                    fmt="none", ecolor=style["color"],
                    elinewidth=0.8, capsize=2.4, capthick=0.8, zorder=3)
        ax.plot(x, values / scale, markersize=marker_size, **line)


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
        draw_curve(skin, skin_ratio, [p.delta_r_np_fm for p in left], left, feature,
                   baseline, color="#BD7150", marker="o", markerfacecolor="#E99572")
        for beta3, label, color, marker, style in BETA3_CURVES:
            curve = [p for p in right if abs(p.beta3 - beta3) <= 1e-10]
            draw_curve(deformation, deformation_ratio, [p.beta2 for p in curve], curve, feature, baseline,
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
            skin.text(0.04, 0.94, "Mixed profile\nRange across three profiles", transform=skin.transAxes,
                      va="top", fontsize=8.5)
            deformation.text(0.04, 0.94, r"Orientation sampling $\pm 1$ SE", transform=deformation.transAxes,
                             va="top", fontsize=8.5)
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
    parser.add_argument("--reference", type=Path, required=True, help="No-skin deformation grid with orientation SE")
    parser.add_argument("--existing-skin", type=Path, required=True, help="Seven reusable spherical profiles")
    parser.add_argument("--new-points", type=Path, required=True, help="Nine new diffuseness and mixed profiles")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points, profiles = load_points(args.reference, args.existing_skin, args.new_points)
    sphere = next(p for p in points if p.panel_column == "skin" and p.delta_r_np_fm == 0)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "fig9_plot_data.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow((*Point._fields, "Q10_GeV_ratio", "K_tail_ratio",
                         "Q10_GeV_ratio_low", "Q10_GeV_ratio_high", "K_tail_ratio_low", "K_tail_ratio_high"))
        for point in points:
            writer.writerow((*point, point.Q10_GeV / sphere.Q10_GeV, point.K_tail / sphere.K_tail,
                             point.Q10_GeV_low / sphere.Q10_GeV, point.Q10_GeV_high / sphere.Q10_GeV,
                             point.K_tail_low / sphere.K_tail, point.K_tail_high / sphere.K_tail))
    with (args.output / "fig9_skin_profiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(PROFILE_FIELDS)
        writer.writerows([row[field] for field in PROFILE_FIELDS] for row in profiles)
    fig = build_figure(points, sphere)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(args.output / f"fig9_skin_deformation.{extension}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
