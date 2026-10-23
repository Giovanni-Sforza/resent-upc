#!/usr/bin/env python3
"""Postprocess archived full J and C2: own-tail slope ratio and lobe balance.

All definitions use one image. Spherical comparisons are performed only after
extraction. No forward calculation, threshold optimization, or extrapolation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from analyze_single_image_shape import CASES, extract, sha256, write_csv
from analyze_single_image_angular_shape import negative_depth


def tail_shape(q, j, *, interpolation="linear", refine_peak=False):
    """50/10/1 percent crossings on the same descending branch after own peak."""
    q, j = np.asarray(q, float), np.asarray(j, float)
    result = extract(q, j, refine_peak=refine_peak)
    peak = result["peak_node_index"]
    rises = np.flatnonzero(np.diff(j)[peak:] > 0)
    end = peak+int(rises[0]) if len(rises) else len(q)-1
    result.update(descending_branch_end_q_GeV=float(q[end]),
                  interpolation=interpolation, crossings={})
    pchip = PchipInterpolator(q, j, extrapolate=False) if interpolation == "pchip" else None
    for name, fraction in (("Q50", .5), ("Q10", .1), ("Q1", .01)):
        threshold = fraction*result["peak_height"]
        candidates = [i for i in range(peak, end)
                      if j[i] >= threshold >= j[i+1] and j[i] > j[i+1]]
        if not candidates:
            result[name+"_GeV"] = None
            result["crossings"][name] = {"status": "not_observed_on_same_branch"}
            continue
        k = candidates[0]
        if interpolation == "pchip":
            crossing = brentq(lambda x: float(pchip(x))-threshold, q[k], q[k+1], xtol=1e-14)
        elif interpolation in ("log_linear", "log_q2"):
            weight = (np.log(threshold)-np.log(j[k]))/(np.log(j[k+1])-np.log(j[k]))
            crossing = (q[k]+weight*(q[k+1]-q[k]) if interpolation == "log_linear"
                        else np.sqrt(q[k]**2+weight*(q[k+1]**2-q[k]**2)))
        elif interpolation == "linear":
            crossing = q[k]+(threshold-j[k])*(q[k+1]-q[k])/(j[k+1]-j[k])
        else:
            raise ValueError(interpolation)
        result[name+"_GeV"] = float(crossing)
        result["crossings"][name] = {"status": "observed", "fraction": fraction,
            "bracket_q_GeV": q[k:k+2].tolist(), "bracket_density": j[k:k+2].tolist()}
    a, b, c = (result[x+"_GeV"] for x in ("Q50", "Q10", "Q1"))
    result.update(Bmid=None, Btail=None, Ktail=None, T=None)
    if all(x is not None for x in (a, b, c)):
        if not a < b < c:
            raise ValueError("unordered thresholds")
        bm, bt = np.log(5)/(b*b-a*a), np.log(10)/(c*c-b*b)
        result.update(Bmid=float(bm), Btail=float(bt), Ktail=float(bt/bm), T=float(b/a-1))
    return result


def lobe_balance(q, c2):
    q, c2 = np.asarray(q, float), np.asarray(c2, float)
    result = negative_depth(q, c2)
    start = result["negative_stop_index_exclusive"]
    while start < len(c2) and c2[start] == 0:
        start += 1
    if start >= len(c2) or c2[start] <= 0:
        raise ValueError("no following positive lobe")
    ends = np.flatnonzero(c2[start:] <= 0)
    if not len(ends):
        raise ValueError("positive lobe not fully observed; no endpoint maximum substituted")
    stop = start+int(ends[0])
    k = start+int(c2[start:stop].argmax())
    if not start < k < stop-1:
        raise ValueError("positive peak is not interior")
    vp, vm = float(c2[k]), result["Vminus"]
    result.update(Vplus=vp, qmax_GeV=float(q[k]), positive_index=k,
                  positive_start_index=start, positive_stop_index_exclusive=stop,
                  B2=float((vp-vm)/(vp+vm)))
    zeros = []
    for i in (start-1, stop-1):
        zeros.append(float(q[i]-c2[i]*(q[i+1]-q[i])/(c2[i+1]-c2[i])))
    result["positive_lobe_zeros_GeV"] = zeros
    x = q[k-1:k+2]-q[k]
    a, b, c = np.polyfit(x, c2[k-1:k+2], 2)
    vertex = -b/(2*a) if a < 0 else np.inf
    if x[0] < vertex < x[-1] and "parabolic_Vminus" in result:
        vp_fit = float(a*vertex**2+b*vertex+c)
        vm_fit = result["parabolic_Vminus"]
        result.update(parabolic_qmax_GeV=float(q[k]+vertex), parabolic_Vplus=vp_fit,
                      parabolic_B2=float((vp_fit-vm_fit)/(vp_fit+vm_fit)))
    return result


def angular_uncertainty(curve, covariance, result, seed, draws):
    """Covariance propagation, including reselection in both observed intervals.

Gaussian fluctuations of the archived mean are a CLT approximation, not an
orientation bootstrap or a detector uncertainty model. Lobe domains are held
at their observed endpoints; both interior extrema are reselected in each draw.
    """
    km, kp = result["index"], result["positive_index"]
    vp, vm = result["Vplus"], result["Vminus"]
    grad = np.array([2*vp, 2*vm])/(vp+vm)**2  # derivatives wrt signed C2[km,kp]
    cov2 = covariance[np.ix_([km, kp], [km, kp])]
    delta_se = float(np.sqrt(max(0, grad@cov2@grad)))
    gaps = {}
    for name, k, lo, hi, sign in (("negative", km, result["negative_start_index"],
            result["negative_stop_index_exclusive"], 1), ("positive", kp,
            result["positive_start_index"], result["positive_stop_index_exclusive"], -1)):
        idx = np.array([i for i in range(lo, hi) if i != k])
        variance = covariance[idx, idx]+covariance[k, k]-2*covariance[idx, k]
        if variance.min() < -1e-16:
            raise ValueError("negative correlated difference variance")
        use = variance > 0
        gaps[name] = float(np.min(sign*(curve[idx][use]-curve[k])/np.sqrt(variance[use]))) if use.any() else None
    out = {"B2_fixed_extrema_delta_se": delta_se, "minimum_competitor_gap_in_se": gaps}
    if not covariance.any():
        out.update(B2_gaussian_reselected_se=0., gaussian_mean_minus_plugin=0.,
                   positive_peak_switch_fraction=0., negative_peak_switch_fraction=0., draws=0)
        return out
    idx = np.r_[np.arange(result["negative_start_index"], result["negative_stop_index_exclusive"]),
                np.arange(result["positive_start_index"], result["positive_stop_index_exclusive"])]
    nminus = result["negative_stop_index_exclusive"]-result["negative_start_index"]
    cov = covariance[np.ix_(idx, idx)]
    eigenvalues, vectors = np.linalg.eigh((cov+cov.T)/2)
    if eigenvalues.min() < -1e-8*eigenvalues.max():
        raise ValueError("archived covariance not positive semidefinite")
    root = vectors*np.sqrt(np.maximum(eigenvalues, 0))
    rng = np.random.default_rng(seed)
    balances, switches_p, switches_m = [], 0, 0
    for offset in range(0, draws, 1000):
        count = min(1000, draws-offset)
        values = curve[idx]+rng.standard_normal((count, len(idx)))@root.T
        im, ip = np.argmin(values[:, :nminus], axis=1), np.argmax(values[:, nminus:], axis=1)
        vminus = -values[np.arange(count), im]
        vplus = values[np.arange(count), nminus+ip]
        if min(vminus.min(), vplus.min()) <= 0:
            raise ValueError("Gaussian draw changes lobe sign at extremum")
        balances.extend((vplus-vminus)/(vplus+vminus))
        switches_m += int(np.sum(idx[im] != km))
        switches_p += int(np.sum(idx[nminus+ip] != kp))
    balances = np.asarray(balances)
    out.update(B2_gaussian_reselected_se=float(balances.std(ddof=1)),
               gaussian_mean_minus_plugin=float(balances.mean()-result["B2"]),
               gaussian_quantiles_16_50_84=np.quantile(balances, [.16, .5, .84]).tolist(),
               positive_peak_switch_fraction=switches_p/draws,
               negative_peak_switch_fraction=switches_m/draws,
               draws=draws, seed=seed,
               covariance_min_eigenvalue=float(eigenvalues.min()))
    return out


def compare_rows(rows, group_key, metrics):
    for row in rows:
        sphere = next(x for x in rows if x["case"] == "spherical" and x[group_key] == row[group_key])
        for key in metrics:
            row["delta_"+key] = row[key]-sphere[key]
    return rows


def plot_results(curves, results, output):
    # Figure contract: compare the two single-image shape summaries and show
    # the newly measured outer tail after own-scale alignment. Diagnostic only.
    # Two quantitative panels, 183 x 99 mm, existing matplotlib workflow;
    # editable PDF/SVG, PNG, and tabulated source data. No K uncertainty available.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.right": False,
        "axes.spines.top": False, "legend.frameon": False})
    colors = dict(zip(CASES, ("#555555", "#3274A1", "#4B9385", "#C17C43")))
    labels = {"spherical": "Spherical", "beta2": r"$\beta_2=0.3$",
              "beta3": r"$\beta_3=0.2$", "skin": "Skin: 0.2 fm"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    fig.subplots_adjust(left=.085, right=.98, bottom=.28, top=.87, wspace=.34)
    for case in CASES:
        q, j = curves[case]
        t, a, u = (results[case][x] for x in ("tail", "angular", "uncertainty"))
        axes[0].semilogy(q/t["Q50_GeV"], j/t["peak_height"], color=colors[case], lw=1.3, label=labels[case])
        axes[0].plot(t["Q1_GeV"]/t["Q50_GeV"], .01, "o", color=colors[case], ms=3.5)
        axes[1].errorbar(t["Ktail"], a["B2"], yerr=u["B2_gaussian_reselected_se"],
                         color=colors[case], marker="o", ms=5, capsize=3)
        dx, dy = {"spherical": (-5, -14), "beta2": (7, -10),
                  "beta3": (7, -14), "skin": (-6, 8)}[case]
        axes[1].annotate(labels[case], (t["Ktail"], a["B2"]), xytext=(dx, dy),
            textcoords="offset points", ha="right" if dx < 0 else "left", color=colors[case], fontsize=8)
    for h in (.5, .1, .01):
        axes[0].axhline(h, color=".75", lw=.6, ls=":", zorder=0)
    axes[0].set(xlim=(.8, 2.3), ylim=(.003, .8), xlabel=r"$q_T/Q_{50}$",
                ylabel=r"$J(q_T)/J_{\rm peak}$", title="a  Outer shape after scale alignment")
    axes[0].legend(loc="lower left", fontsize=7.1)
    axes[1].set(xlim=(.99, 1.15), ylim=(-.519, -.442), xlabel=r"Outer-slope ratio $\mathcal{K}_{\rm tail}$",
                ylabel=r"Lobe balance $\mathcal{B}_2$", title="b  Single-image observables")
    axes[1].ticklabel_format(axis="both", style="plain", useOffset=False)
    axes[1].set_xticks([1.00, 1.05, 1.10, 1.15])
    axes[1].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    fig.text(.5, .07, "Full distributions; 32,768 orientation pairs for each deformation case.\n"
             "Vertical bars: Gaussian propagation of orientation MC covariance, with extrema reselected.\n"
             "K-tail uncertainty is unavailable. Detector and model uncertainties are not included.",
             ha="center", fontsize=7)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output/f"tail_lobes.{ext}", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gaussian-draws", type=int, default=20000)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new output directory")
    if args.gaussian_draws < 2:
        raise ValueError("at least two Gaussian draws required")
    manifest = json.loads((source/"analysis_manifest.json").read_text())
    summary = json.loads((source/"summary.json").read_text())
    names = ("summary_arrays.npz", "summary.json", "radial_density.csv", "curves.csv", "orientation_convergence.csv")
    hashes = {name: sha256(source/name) for name in names}
    for name in names:
        if hashes[name] != manifest["output_sha256"][name]:
            raise ValueError(f"input checksum mismatch: {name}")
    tables = {}
    for name in ("radial_density.csv", "curves.csv", "orientation_convergence.csv"):
        with (source/name).open() as f:
            tables[name] = list(csv.DictReader(f))
    results, curves, b_rows, variations, sources, self_rows = {}, {}, [], [], [], []
    with np.load(source/"summary_arrays.npz", allow_pickle=False) as z:
        qangle, radii = z["q"], z["radii"]
        for case_index, case in enumerate(CASES):
            q = z[f"cases__{case}__radial__q"]
            j = z[f"cases__{case}__radial__J__value"]
            d = z[f"cases__{case}__radial__D__value"]
            c2 = z[f"cases__{case}__curves__full__C2__value"]
            covariance = z[f"cases__{case}__curves__full__C2__covariance"]
            rows = [x for x in tables["radial_density.csv"] if x["case"] == case]
            np.testing.assert_array_equal(q, [float(x["q_GeV"]) for x in rows])
            np.testing.assert_array_equal(j, [float(x["J"]) for x in rows])
            np.testing.assert_array_equal(d, [float(x["D"]) for x in rows])
            rows = [x for x in tables["curves.csv"] if x["case"] == case]
            np.testing.assert_array_equal(qangle, [float(x["q_GeV"]) for x in rows])
            np.testing.assert_array_equal(c2, [float(x["C2_full"]) for x in rows])
            np.testing.assert_allclose(np.sqrt(np.maximum(0, np.diag(covariance))),
                                       [float(x["C2_full_se"]) for x in rows], rtol=1e-12, atol=1e-20)
            tail, angular = tail_shape(q, j), lobe_balance(qangle, c2)
            if tail["Ktail"] is None:
                raise ValueError(f"{case}: same-branch Q1 unavailable; no extrapolation performed")
            uncertainty = angular_uncertainty(c2, covariance, angular, 20261024+case_index, args.gaussian_draws)
            results[case] = {"tail": tail, "angular": angular, "uncertainty": uncertainty}
            curves[case] = (q, j)
            scaled = tail_shape(1.2*q, 7*j)
            np.testing.assert_allclose(scaled["Ktail"], tail["Ktail"], rtol=1e-13)
            np.testing.assert_allclose(scaled["Q1_GeV"], 1.2*tail["Q1_GeV"], rtol=1e-13)
            np.testing.assert_allclose(lobe_balance(1.2*qangle, 1.3*c2)["B2"], angular["B2"], rtol=1e-13)
            for method, options in (("pchip", {"interpolation": "pchip"}),
                    ("log_linear", {"interpolation": "log_linear"}), ("log_q2", {"interpolation": "log_q2"}),
                    ("parabolic_peak", {"refine_peak": True})):
                t = tail_shape(q, j, **options)
                variations.append({"case": case, "method": method, "Ktail": t["Ktail"],
                                   "Q1_GeV": t["Q1_GeV"], "B2": angular["parabolic_B2"] if method == "parabolic_peak" else angular["B2"]})
            coeff = z[f"cases__{case}__mean_coefficients"][0]
            use = (radii > 0) & (radii <= .17)
            native = lobe_balance(radii[use], -.25*coeff[1, use]/coeff[0, use])
            results[case]["native_angular_diagnostic"] = native
            for i, step in enumerate(summary["b_steps_fm"]):
                jt = j if i == 0 else j+z[f"b_grid__{case}__{i-1}__radial_coarse_minus_fine"][0]
                ct = c2 if i == 0 else c2+z[f"b_grid__{case}__{i-1}__C2_coarse_minus_fine__curves__full_C2__value"]
                t, a = tail_shape(q, jt), lobe_balance(qangle, ct)
                if t["Ktail"] is None:
                    raise ValueError(f"{case}, b={step}: Q1 not observed")
                b_rows.append({"case": case, "b_step_fm": step, **{k: t[k] for k in ("Q1_GeV", "Bmid", "Btail", "Ktail")},
                    **{k: a[k] for k in ("qmin_GeV", "qmax_GeV", "Vminus", "Vplus", "B2")}})
            ds = tail_shape(q, d)
            results[case]["self_D_mechanism_diagnostic"] = ds
            self_rows.append({"case": case, **{k: ds[k] for k in ("Q50_GeV", "Q10_GeV", "Q1_GeV", "Bmid", "Btail", "Ktail")}})
            for qi, ji in zip(q, j):
                sources.append({"case": case, "quantity": "J", "q_GeV": float(qi), "value": float(ji)})
            for qi, ci in zip(qangle, c2):
                sources.append({"case": case, "quantity": "C2", "q_GeV": float(qi), "value": float(ci)})
    # Two scientific limiting cases: exponential tail; no crossing before an upturn.
    qt = np.linspace(0, 1, 2001)
    np.testing.assert_allclose(tail_shape(qt, np.exp(-10*qt**2), interpolation="log_q2")["Ktail"], 1, atol=1e-13)
    censored = tail_shape([0, 1, 2, 3, 4, 5], [1, .4, .05, .04, .06, .001])
    assert censored["Q1_GeV"] is None and censored["Ktail"] is None
    compare_rows(b_rows, "b_step_fm", ("Ktail", "B2"))
    compare_rows(variations, "method", ("Ktail", "B2"))
    for row in b_rows:
        for key, section in (("Ktail", "tail"), ("B2", "angular")):
            fine_delta = results[row["case"]][section][key]-results["spherical"][section][key]
            row["coarse_minus_fine_delta_"+key] = row["delta_"+key]-fine_delta
    for row in self_rows:
        row["Ktail_change_percent"] = 100*(row["Ktail"]/self_rows[0]["Ktail"]-1)
    groups = {}
    for row in tables["orientation_convergence.csv"]:
        if row["section"] == "curves" and row["branch"] == "full":
            groups.setdefault((row["case"], row["sample"], int(row["n_orientations"])), []).append(row)
    prefixes = []
    for (case, sample, count), rows in groups.items():
        rows.sort(key=lambda x: float(x["q_or_window"]))
        a = lobe_balance([float(x["q_or_window"]) for x in rows], [float(x["C2"]) for x in rows])
        prefixes.append({"case": case, "sample": sample, "n_orientations": count,
                        **{key: a[key] for key in ("qmin_GeV", "qmax_GeV", "Vminus", "Vplus", "B2")}})
    metrics = []
    for case in CASES:
        t, a, u = (results[case][key] for key in ("tail", "angular", "uncertainty"))
        row = {"case": case, "n_orientations": summary["n_orientations"][case],
               **{key: t[key] for key in ("Q50_GeV", "Q10_GeV", "Q1_GeV", "Bmid", "Btail", "Ktail", "T")},
               **{key: a[key] for key in ("qmin_GeV", "qmax_GeV", "Vminus", "Vplus", "B2")},
               "B2_gaussian_orientation_se": u["B2_gaussian_reselected_se"], "Ktail_orientation_se": None}
        for key in ("Ktail", "Q1_GeV", "Bmid", "Btail", "T"):
            row[key+"_change_percent"] = 100*(t[key]/results["spherical"]["tail"][key]-1)
        for key in ("B2", "Vplus", "Vminus"):
            row["delta_"+key] = a[key]-results["spherical"]["angular"][key]
        metrics.append(row)
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("metrics.csv", metrics), ("b_grid_metrics.csv", b_rows),
            ("extraction_checks.csv", variations), ("orientation_prefixes.csv", prefixes),
            ("self_D_diagnostic.csv", self_rows), ("curve_source_data.csv", sources)):
        write_csv(output/filename, rows)
    (output/"details.json").write_text(json.dumps({"cases": results, "checks": {
        "input_hashes": "passed", "csv_npz_identity": "exact", "scale_invariance": "passed",
        "exponential_limit": "passed", "reject_crossing_after_upturn": "passed"},
        "uncertainty": "Gaussian covariance propagation describes orientation-integration uncertainty of archived C2 means only, with extrema reselected inside observed lobes. Not an event bootstrap or experimental error. J cross-q covariance is unavailable; no K-tail SE or K/B2 covariance is inferred."}, indent=2)+"\n")
    plot_results(curves, results, output)
    lines = ["# 单图外尾斜率比与角关联双瓣平衡", "",
        "仅对已有32768取向完整分布做后处理。每个案例单独提取，球形只用于事后比较。没有新正演或阈值优化。", "",
        "- Ktail = [ln(10)/(Q1²−Q10²)] / [ln(5)/(Q10²−Q50²)]。Q50/Q10/Q1是角平均二维密度J自身主峰后，同一连续下降支上50%/10%/1%交点；主值用保存节点峰高、线性J(q)插值。不是dN/dq。",
        "- B2 = (Vplus−Vminus)/(Vplus+Vminus)。Vminus为首个已观察负区间的内点谷深；Vplus为随后完整正瓣内点峰高。主值为保存q网格极值。负区间左零点未观测，正瓣两端过零均已观测。", "",
        "| case | Ktail | Ktail变化(%) | B2 | B2−球形 | B2取向SE* | Q1 (MeV/c) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in metrics:
        lines.append(f"| {r['case']} | {r['Ktail']:.9f} | {r['Ktail_change_percent']:+.4f} | {r['B2']:.9f} | {r['delta_B2']:+.9f} | {r['B2_gaussian_orientation_se']:.6g} | {1000*r['Q1_GeV']:.6f} |")
    lines += ["", "在当前参数点，Ktail保留β2、β3响应而明显压低skin交叉响应；B2的形变响应为负、skin响应为正。参数步长不同，百分比不是统一参数灵敏度，也不是实验显著性。", "",
        "## 物理解释", "",
        "skin使中段和尾段下降斜率近乎共同缩放；形变主要改变尾段的相对斜率。因此Ktail支持‘尺度与外尾形状’的区别。它并非纯干涉量：下面对模型内部source-self D独立提取同一定义，检查尾形响应是否也存在于单源部分。D仅用于机制诊断，不进入实验观测量定义。", "",
        "| case | self-D Ktail | 相对球形变化(%) |",
        "|---|---:|---:|"]
    for r in self_rows:
        lines.append(f"| {r['case']} | {r['Ktail']:.9f} | {r['Ktail_change_percent']:+.4f} |")
    lines += ["", "Ktail在D中也有明显形变响应，故不能把full J中Ktail的变化全部归因于双源干涉。非线性指标的full/self差不等于可测的纯干涉项。B2仍是完整C2(q)的形状摘要，不是新的独立谐波。", "",
        "## 数值核查与边界", "",
        "输入哈希和CSV/NPZ一致性、整体尺度/强度不变性、指数尾部极限、拒绝跨极小寻找Q1均通过。fine及b=1/2 fm下四case的三个交点均位于同一下降支；无需外推。PCHIP、logJ插值、三点抛物线峰高均列于extraction_checks.csv，不据此挑选定义。", "",
        "*B2误差：正峰附近竞争节点相关误差下仅约0.5–0.8 SE间隔，峰节点并不稳定；因此用存储的完整C2跨q协方差做高斯近似传播，每次在两个已观察区间重新选极值。它不是逐取向bootstrap，更不是实验置信区间。球形/skin的0只表示没有取向积分噪声。details.json同时保存固定节点delta近似、节点切换比例和原生径向表达诊断。", "",
        "没有full J跨q协方差或逐取向J，不能给Ktail取向SE、Ktail–B2联合误差、或Ktail取向收敛结论。1%是亮度阈值，不是1%事件数；外尾的背景与动量分辨率影响尚未评估。", "",
        "### b步长对结构差的影响", "",
        "Δ表示case减同网格球形；下表为粗网格Δ减fine(.5 fm)的Δ，不是误差界。", "",
        "| case | b步长(fm) | δΔKtail | δΔB2 |", "|---|---:|---:|---:|"]
    for r in b_rows:
        if r["case"] != "spherical" and r["b_step_fm"] != .5:
            lines.append(f"| {r['case']} | {r['b_step_fm']:g} | {r['coarse_minus_fine_delta_Ktail']:+.9f} | {r['coarse_minus_fine_delta_B2']:+.9f} |")
    lines += ["", "skin的Ktail残余很小，其相对b网格变化较大；不应把−0.37%当作高精度的普适污染率。", "",
        "B2绝对值的b步长依赖不可忽略：球形从步长0.5时的−0.46826，变为步长1时−0.47509、步长2时−0.50463。相同网格下结构差有所抵消，但β3的ΔB2仍变化约10%/32%，skin约6%/34%。粗步长比较不能证明0.5 fm已达到连续b极限。", "",
        "orientation_prefixes.csv记录各取向前缀与两半样本。β3的8192前缀B2约−0.46843，最终32768约−0.47204；两半16384分别约−0.47113、−0.47294。因此弱β3角响应的早期前缀不能视为收敛证据。", "",
        "![外尾与双瓣平衡](tail_lobes.png)", "",
        "图左为各案例自身峰高与Q50归一化后的外坡；右为绝对观测量。纵误差仅含上述取向积分高斯传播，Ktail无误差条不表示其误差为零。", "",
        "## 复现", "", "```bash",
        f"python3 scripts/analyze_single_image_tail_lobes.py --input '{args.input}' --output outputs/single_image_tail_lobes/replay_new_directory", "```", ""]
    (output/"report.md").write_text("\n".join(lines))
    code_files = (Path(__file__), Path(__file__).with_name("analyze_single_image_shape.py"),
                  Path(__file__).with_name("analyze_single_image_angular_shape.py"))
    metadata = {"source_analysis": str(source), "source_analysis_fingerprint": manifest["fingerprint"],
        "input_sha256": hashes, "code_sha256": {p.name: sha256(p) for p in code_files},
        "figure_contract": {"claim": "At these benchmark steps, Ktail suppresses the skin response and retains both deformation responses; B2 responds in opposite directions.",
            "archetype": "quantitative grid", "backend": "existing Python matplotlib", "exports": "PNG PDF SVG; editable text; source CSV",
            "uncertainty": "Vertical orientation-MC Gaussian propagation only; no K SE."},
        "output_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output/"manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
