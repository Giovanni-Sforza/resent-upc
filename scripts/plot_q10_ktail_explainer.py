#!/usr/bin/env python3
"""Explain pT,10 and K_tail using two saved, explicitly separated spherical spectra."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/mpl_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import numpy as np

from upc_deformation.shape_scan_analysis import extract_tail_features

CASE = "none__b2_000__b3_000"
MID, TAIL, INK = "#2166AC", "#B2182B", "#111111"


def load_spectra(fig8, wide):
    with np.load(fig8, allow_pickle=False) as data:
        index = int(np.argmin(data["b_steps_fm"]))
        branch = list(data["branch_names"]).index("full")
        q = data["q"]
        j = data[f"cases__{CASE}__mean_A0"][index, branch]
    yield "fig8_reference", q, j, dict(
        source=str(fig8), b_range_fm=[12, 60], ymax=1,
        field_method="archived fast_v2 finite-box field, matching current Fig. 8")
    with np.load(wide, allow_pickle=False) as data:
        index = list(data["cutoffs_fm"]).index(10000.)
        branch = list(data["branch_names"]).index("full")
        q, j = data["stored_q"], data["mean_A0"][index, branch]
    yield "wide_b_diagnostic", q, j, dict(
        source=str(wide), b_range_fm=[12, 10000], ymax=1,
        field_method="saved free-space field; separate diagnostic")


def draw(q, j, feature, meta, out, name):
    h = j / feature["peak_height"]
    q50, q10, q1 = (feature[k + "_GeV"] for k in ("Q50", "Q10", "Q1"))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 9,
        "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.linewidth": .7, "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(89 / 25.4, 73 / 25.4))
    fig.subplots_adjust(left=.17, right=.95, bottom=.18, top=.975)
    ax.semilogy(q*q, h, color=INK, lw=1.1, zorder=2)
    for crossing, level, label in (
        (q50, .5, r"$p_{T,50}^{2}$"),
        (q10, .1, r"$p_{T,10}^{2}$"), (q1, .01, r"$p_{T,1}^{2}$")
    ):
        ax.plot([0, crossing*crossing, crossing*crossing],
                [level, level, .0025], color=".62", lw=.5,
                ls=(0, (2, 3)), zorder=1)
        ax.plot(crossing*crossing, level, "o", ms=3.2,
                mfc="white", mec=INK, mew=.65, zorder=5)
        ax.annotate(label, (crossing*crossing, level), xytext=(3, 7),
                    textcoords="offset points", fontsize=9, color=INK)
    for lo, hi, low_h, high_h, color, style, label in (
        (q50, q10, .5, .1, MID, "--", r"$B_{\rm mid}$"),
        (q10, q1, .1, .01, TAIL, "-.", r"$B_{\rm tail}$")
    ):
        ax.plot([lo*lo, hi*hi], [low_h, high_h], color=color,
                lw=1, ls=style, zorder=4)
        ax.text((lo*lo+hi*hi)/2, np.sqrt(low_h*high_h)*1.7,
                label, color=color, fontsize=9, ha="center", va="bottom")
    ax.set(xlim=(0, .03), ylim=(.0025, 1.5),
           xlabel=r"$p_T^2\;[\mathrm{GeV}^2]$",
           ylabel="Peak-normalized yield")
    ax.set_xticks([0, .01, .02, .03], ["0", "0.01", "0.02", "0.03"])
    ax.set_yticks([1, .5, .1, .01], ["1", "0.5", "0.1", "0.01"])
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   width=.65, pad=3)
    ax.tick_params(which="major", length=3.5)
    ax.tick_params(which="minor", length=1.8)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(out / f"{name}.{extension}", dpi=400)
    plt.close(fig)


def process(args, out):
    contract = ROOT / "docs/q10_ktail_explainer_contract.json"
    shutil.copy2(contract, out / contract.name)
    shutil.copy2(ROOT / "docs/q10_ktail_explanation_proposal.tex", out / "section_IVB_proposal.tex")
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    rows, report = [], {}
    for name, q, j, meta in load_spectra(args.fig8, args.wide):
        f = extract_tail_features(q, j)
        if f["status"] != "observed":
            raise ValueError(f"{name}: thresholds are not all on the same observed branch")
        start = f["peak_node_index"]
        end = f["crossings"]["Q1"]["bracket_indices"][1]
        mid_prediction = .1*np.exp(-f["B_mid"]*(f["Q1_GeV"]**2-f["Q10_GeV"]**2))
        report[name] = dict(**meta, features=f,
            nuclear_parameters=dict(A=96, Z=40, R_p_fm=5.02, R_n_fm=5.02,
                                    a_p_fm=.46, a_n_fm=.46, beta2=0, beta3=0),
            peak_to_Q1_nonincreasing=bool(np.all(np.diff(j[start:end+1]) <= 0)),
            middle_slope_continuation_at_Q1_fraction=float(mid_prediction),
            actual_Q1_fraction=.01,
            middle_continuation_relative_excess_at_Q1=float(mid_prediction/.01-1))
        draw(q, j, f, meta, out, name)
        for qi, ji in zip(q, j):
            rows.append(dict(reference=name, pT_GeV=float(qi), pT2_GeV2=float(qi*qi),
                             J=float(ji), J_over_Jmax=float(ji/f["peak_height"])))
        print(name, {key: f[key] for key in ("Q50_GeV", "Q10_GeV", "Q1_GeV", "B_mid", "B_tail", "K_tail")}, flush=True)
    with (out / "source_spectra.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "captions.txt").write_text(
        "Fig. X. Construction of pT,10 and K_tail from the coherent radial spectrum "
        "in collisions of spherical 96Zr nuclei without a neutron skin, integrated over 12<b<60 fm and |y|<1. "
        "The spectrum is proportional to dN/d(pT^2) and normalized to its peak. "
        "Circles mark the outer 50%, 10%, and 1% peak-height crossings. "
        "The blue dashed and red dash-dotted secants indicate the 50%-10% and 10%-1% intervals. "
        "B_mid and B_tail are the corresponding natural-logarithmic decay rates; K_tail=B_tail/B_mid.\n\n"
        "Reference parameters: beta2=beta3=0, R_p=R_n=5.02 fm and a_p=a_n=0.46 fm, "
        "matching the registered scan underlying Fig. 8. No exponential fit is used. "
        "Numerical, experimental and model uncertainties are not displayed.\n\n"
        "The separate wide_b_diagnostic figure uses 12<b<10000 fm and a free-space photon field; "
        "all other caption definitions apply. Its values do not replace the current Fig. 8 reference.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig8", type=Path, default=ROOT / "outputs/shape_scan/review_n2048_fast_v2_20261027/analysis_n2048_fast_v2/radial_means.npz")
    parser.add_argument("--wide", type=Path, default=ROOT / f"outputs/shape_scan/b10000_20261111/representatives/cases/{CASE}/mean_A0.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S_q10_ktail_explainer"))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = "failed"
    try:
        process(args, out)
        status = "complete"
    except Exception:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  text=True, capture_output=True, check=True).stdout.strip()
        (out / "run_info.json").write_text(json.dumps(dict(
            status=status, command=[sys.executable, *sys.argv], git_commit=revision,
            sources=dict(fig8=str(args.fig8), wide=str(args.wide)),
            operation="Postprocess saved mean spectra only; no forward simulation.",
        ), indent=2) + "\n")
        package = ROOT / "outputs/_downloads" / f"{out.name}.tar.gz"
        package.parent.mkdir(parents=True, exist_ok=True)
        # The small dedicated run directory contains only this run's deliverables.
        with tarfile.open(package, "w:gz") as archive:
            archive.add(out, arcname=out.name)
        print(f"status={status}; download={package}", flush=True)


if __name__ == "__main__":
    main()
