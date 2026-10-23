#!/usr/bin/env python3
"""Single-image directional half-height radii and first negative C2 depth.

Postprocessing of archived full distributions only. Primary angular sectors
are the two complementary pairs |qx| >= |qy| and |qy| > |qx| (half-width 45 deg).
Each sector has its own peak; no spherical reference enters extraction.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from analyze_single_image_shape import CASES, extract, sha256, write_csv


def sectors(r, coefficients, half_width_deg=45, **kwargs):
    delta = np.deg2rad(half_width_deg)
    if not 0 < delta <= np.pi/4:
        raise ValueError("sector half width must lie in (0,45] degrees")
    a0, c, d = np.asarray(coefficients, float)
    factor = np.sin(2*delta)/(2*delta)
    parallel, perpendicular = a0-.5*factor*c, a0+.5*factor*c
    if min(parallel.min(), perpendicular.min()) < 0:
        raise ValueError("negative sector density")
    p, t = extract(r, parallel, **kwargs), extract(r, perpendicular, **kwargs)
    qp, qt = p["Q50_GeV"], t["Q50_GeV"]
    if qp is None or qt is None:
        raise ValueError("outer half-height not observed")
    return {"Qparallel_GeV": qp, "Qperpendicular_GeV": qt, "Qbar_GeV": (qp+qt)/2,
            "E50": (qp-qt)/(qp+qt), "parallel": p, "perpendicular": t,
            "half_width_deg": half_width_deg}, (parallel, perpendicular)


def negative_depth(q, c2):
    """Minimum inside the first observed connected negative interval."""
    q, c2 = np.asarray(q, float), np.asarray(c2, float)
    if q.shape != c2.shape or np.any(np.diff(q) <= 0) or not np.isfinite(c2).all():
        raise ValueError("invalid angular curve")
    starts = np.flatnonzero(c2 < 0)
    if not len(starts):
        raise ValueError("no observed negative interval")
    start = int(starts[0])
    ends = np.flatnonzero(c2[start:] >= 0)
    stop = start+int(ends[0]) if len(ends) else len(c2)
    index = start+int(c2[start:stop].argmin())
    if not start < index < stop-1:
        raise ValueError("negative minimum is not interior to observed interval")
    result = {"Vminus": float(-c2[index]), "qmin_GeV": float(q[index]), "index": index,
        "negative_start_index": start, "negative_stop_index_exclusive": stop,
        "left_zero_observed": start > 0, "right_zero_observed": stop < len(c2)}
    x = q[index-1:index+2]-q[index]
    a, b, c = np.polyfit(x, c2[index-1:index+2], 2)
    vertex = -b/(2*a) if a > 0 else np.inf
    if x[0] < vertex < x[-1]:
        result["parabolic_qmin_GeV"] = float(q[index]+vertex)
        result["parabolic_Vminus"] = float(-(a*vertex**2+b*vertex+c))
    return result


def grid_minimum_uncertainty(c2, covariance, feature):
    """Local delta SE, with correlated competition gaps on the saved grid."""
    k = feature["index"]
    indices = np.arange(feature["negative_start_index"], feature["negative_stop_index_exclusive"])
    indices = indices[indices != k]
    variance = covariance[indices, indices]+covariance[k, k]-2*covariance[indices, k]
    if np.min(variance) < -1e-16:
        raise ValueError("negative pairwise-difference variance")
    positive = variance > 0
    gaps = (c2[indices][positive]-c2[k])/np.sqrt(variance[positive])
    return {"Vminus_orientation_se_grid_delta": float(np.sqrt(max(0, covariance[k, k]))),
            "smallest_competing_node_gap_in_se": float(gaps.min()) if len(gaps) else None,
            "method": "local delta method at unique saved-grid minimum; excludes continuous minimum/grid/experimental uncertainty"}


def plot_metrics(results, output):
    # Existing Python/matplotlib workflow; quantitative comparison, not a
    # parameter-extraction claim. Points are self-referenced absolute metrics.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.right": False, "axes.spines.top": False})
    colors = dict(zip(CASES, ("#555555", "#3274A1", "#4B9385", "#C17C43")))
    labels = {"spherical": "Spherical", "beta2": r"$\beta_2=0.3$",
              "beta3": r"$\beta_3=0.2$", "skin": "Model skin: 0.2 fm"}
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    fig.subplots_adjust(left=.17, bottom=.26, right=.97, top=.88)
    offsets = {"spherical": (-8, -13), "beta2": (8, 3), "beta3": (-7, -13), "skin": (8, 3)}
    for case in CASES:
        result = results[case]
        x, y = result["E50"], result["Vminus"]
        ax.errorbar(x, y, yerr=result["Vminus_orientation_se_grid_delta"], marker="o",
                    ms=6, color=colors[case], capsize=3)
        dx, dy = offsets[case]
        ax.annotate(labels[case], (x, y), xytext=(dx, dy), textcoords="offset points",
                    color=colors[case], ha="right" if dx < 0 else "left", fontsize=8.5)
    ax.set(xlim=(.0291, .0312), ylim=(.06802, .06894),
        xlabel=r"Directional half-height difference $E_{50}$",
        ylabel=r"Negative angular depth $V_-=-\min C_2$",
        title="Both candidates respond to skin and deformation")
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.set_xticks([.0295, .0300, .0305, .0310])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    fig.text(.5, .06, "45-degree sector half-width; full distributions.\n"
        "Vertical bars: orientation MC SE at the stable saved-grid minimum.\n"
        "E50 uncertainty is unavailable; detector/model uncertainties are not included.",
        ha="center", fontsize=7.1)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output/f"angular_shape.{ext}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new output directory")
    manifest = json.loads((source/"analysis_manifest.json").read_text())
    summary = json.loads((source/"summary.json").read_text())
    names = ("summary_arrays.npz", "summary.json", "curves.csv", "orientation_convergence.csv")
    hashes = {name: sha256(source/name) for name in names}
    for name in names:
        if hashes[name] != manifest["output_sha256"][name]:
            raise ValueError(f"input checksum mismatch: {name}")
    with (source/"curves.csv").open() as f:
        angular_csv = list(csv.DictReader(f))
    results, detail, variations, profiles, b_rows = {}, {}, [], [], []
    with np.load(source/"summary_arrays.npz", allow_pickle=False) as z:
        r_all, q = z["radii"], z["q"]
        use = r_all <= .17
        r = r_all[use]
        for case in CASES:
            coeff_all = z[f"cases__{case}__mean_coefficients"][0]
            coeff = coeff_all[:, use]
            feature, (parallel, perpendicular) = sectors(r, coeff)
            curve = z[f"cases__{case}__curves__full__C2__value"]
            covariance = z[f"cases__{case}__curves__full__C2__covariance"]
            rows = [row for row in angular_csv if row["case"] == case]
            np.testing.assert_array_equal(q, [float(row["q_GeV"]) for row in rows])
            np.testing.assert_array_equal(curve, [float(row["C2_full"]) for row in rows])
            depth = negative_depth(q, curve)
            uncertainty = grid_minimum_uncertainty(curve, covariance, depth)
            np.testing.assert_allclose(np.sqrt(np.maximum(0, np.diag(covariance))),
                                       [float(row["C2_full_se"]) for row in rows], rtol=1e-12, atol=1e-20)
            results[case] = {key: feature[key] for key in ("Qparallel_GeV", "Qperpendicular_GeV", "Qbar_GeV", "E50")}
            results[case].update({key: depth[key] for key in ("Vminus", "qmin_GeV")})
            results[case].update(uncertainty)
            detail[case] = {"sectors": feature, "negative_interval": depth, "uncertainty": uncertainty}
            # Native angular expression is a separate representation diagnostic.
            analytic = -.25*np.interp(q, r_all, coeff_all[1])/np.interp(q, r_all, coeff_all[0])
            np.testing.assert_allclose(analytic, [float(row["C2_full_radial_analytic_diagnostic"]) for row in rows], atol=2e-16, rtol=1e-12)
            detail[case]["analytic_C2_on_curve_grid"] = negative_depth(q, analytic)
            detail[case]["native_C2"] = negative_depth(r[1:], -.25*coeff[1, 1:]/coeff[0, 1:])
            # Analytic sector integral checked by angular quadrature, including
            # both opposite sectors and the full sine coefficient.
            nodes, weights = np.polynomial.legendre.leggauss(64)
            for centers, expected in (((0, np.pi), parallel), ((np.pi/2, 3*np.pi/2), perpendicular)):
                values = np.zeros_like(r)
                for center in centers:
                    alpha = center+nodes*np.pi/4
                    intensity = coeff[0, :, None]-.5*(coeff[1, :, None]*np.cos(2*alpha)+coeff[2, :, None]*np.sin(2*alpha))
                    values += intensity@weights/4
                np.testing.assert_allclose(values, expected, rtol=1e-13, atol=1e-12)
            scaled, _ = sectors(1.2*r, 7*coeff)
            np.testing.assert_allclose(scaled["E50"], feature["E50"], rtol=1e-13)
            np.testing.assert_allclose(scaled["Qbar_GeV"], 1.2*feature["Qbar_GeV"], rtol=1e-13)
            # A q-independent modulation changes brightness but not own-peak widths.
            constant = np.stack((coeff[0], .2*coeff[0], .1*coeff[0]))
            constant_result, _ = sectors(r, constant)
            np.testing.assert_allclose(constant_result["E50"], 0, atol=1e-14)
            for label, width, options in (("halfwidth15", 15, {}), ("halfwidth30", 30, {}),
                    ("pchip_crossing", 45, {"interpolation": "pchip"}),
                    ("parabolic_peak", 45, {"refine_peak": True})):
                alternative, _ = sectors(r, coeff, width, **options)
                variations.append({"case": case, "method": label, "half_width_deg": width,
                    **{key: alternative[key] for key in ("Qbar_GeV", "E50")}})
            for qi, ip, it in zip(r, parallel, perpendicular):
                profiles.append({"case": case, "q_GeV": float(qi), "I_parallel": float(ip), "I_perpendicular": float(it)})
            for i, step in enumerate(summary["b_steps_fm"]):
                difference = 0 if i == 0 else z[f"b_grid__{case}__{i-1}__C2_coarse_minus_fine__curves__full_C2__value"]
                value = negative_depth(q, curve+difference)
                b_rows.append({"case": case, "b_step_fm": step, "Vminus": value["Vminus"], "qmin_GeV": value["qmin_GeV"]})
    for row in b_rows:
        sphere = next(x for x in b_rows if x["case"] == "spherical" and x["b_step_fm"] == row["b_step_fm"])
        row["delta_Vminus_vs_spherical"] = row["Vminus"]-sphere["Vminus"]
        fine_delta = results[row["case"]]["Vminus"]-results["spherical"]["Vminus"]
        row["structure_coarse_minus_fine"] = row["delta_Vminus_vs_spherical"]-fine_delta
        row["structure_change_percent"] = 100*row["structure_coarse_minus_fine"]/fine_delta if fine_delta else None
    for row in variations:
        sphere = next(x for x in variations if x["case"] == "spherical" and x["method"] == row["method"])
        row["delta_E50_vs_spherical"] = row["E50"]-sphere["E50"]
    with (source/"orientation_convergence.csv").open() as f:
        convergence = list(csv.DictReader(f))
    groups = {}
    for row in convergence:
        if row["section"] == "curves" and row["branch"] == "full":
            groups.setdefault((row["case"], row["sample"], int(row["n_orientations"])), []).append(row)
    prefix_rows = []
    for (case, sample, count), rows in groups.items():
        rows.sort(key=lambda x: float(x["q_or_window"]))
        values = negative_depth([float(x["q_or_window"]) for x in rows], [float(x["C2"]) for x in rows])
        prefix_rows.append({"case": case, "sample": sample, "n_orientations": count,
            "qmin_GeV": values["qmin_GeV"], "Vminus": values["Vminus"],
            "point_orientation_se_at_selected_node": float(rows[values["index"]]["orientation_se"])})
    metrics = []
    for case in CASES:
        result = results[case]
        row = {"case": case, "n_orientations": summary["n_orientations"][case],
            **{key: result[key] for key in ("Qparallel_GeV", "Qperpendicular_GeV", "Qbar_GeV", "E50", "Vminus", "qmin_GeV", "Vminus_orientation_se_grid_delta")},
            "E50_se": None}
        for key in ("Qbar_GeV", "E50", "Vminus"):
            row["delta_"+key] = result[key]-results["spherical"][key]
            row[key+"_change_percent"] = 100*(result[key]/results["spherical"][key]-1)
        metrics.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"metrics.csv", metrics)
    write_csv(output/"sector_and_interpolation_checks.csv", variations)
    write_csv(output/"sector_profiles.csv", profiles)
    write_csv(output/"negative_depth_b_grid.csv", b_rows)
    write_csv(output/"negative_depth_orientation_prefixes.csv", prefix_rows)
    (output/"details.json").write_text(json.dumps({"cases": detail, "definition": {
        "sectors": "Symmetric opposite sectors about lepton kT, half-width 45 deg, all angles covered by parallel and perpendicular pairs. Exact integral of full mean harmonic distribution; own peak for each profile.",
        "negative_depth": "First observed connected negative C2 interval; saved-grid internal minimum. The left zero is not observed; q=0 is excluded.",
        "uncertainty": "E50/Qbar covariance and b/prefix full coefficients unavailable. Vminus uses local delta MC SE at the strongly separated discrete minimum; no continuous-q, detector, background or model errors."},
        "checks": ["input hashes", "CSV/NPZ identity", "sector angular quadrature", "normalization and scale invariance", "constant modulation has E50=0"]}, indent=2)+"\n")
    plot_metrics(results, output)
    lines = ["# 单图方向性宽度与角关联负谷", "",
        "主结果来自已有32768取向完整分布，仅后处理。扇区半宽固定45度：平行与垂直两组各覆盖半个圆周。每条扇区曲线独立使用自身峰高。",
        "I_parallel=A0−C/pi，I_perpendicular=A0+C/pi；D的sin项在对称扇区平均中抵消。Qbar=(Qparallel+Qperpendicular)/2，E50=(Qparallel−Qperpendicular)/(Qparallel+Qperpendicular)。",
        "Vminus=-min C2，范围为本图首个已观测负区间。保存曲线左端已为负，不声称左零点已被观测。", "",
        "| 案例 | Qbar (MeV/c) | E50 | Vminus | ΔE50 (%) | ΔVminus (%) |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(f"| {row['case']} | {1000*row['Qbar_GeV']:.6f} | {row['E50']:.9f} | {row['Vminus']:.9f} | {row['E50_change_percent']:+.4f} | {row['Vminus_change_percent']:+.4f} |")
    lines += ["", "在这组参数点上，三类结构变化都降低E50、增加Vminus；E50保留β3响应，却没有压低中子皮响应。球形也有非零E50，因此它不是核本身的椭圆度。参数间变化幅度仅对应本次选定的参数步长。", "",
        "## 数值核查与不确定度", "",
        "输入哈希、CSV/NPZ一致性、扇区解析积分与64点角向求积、整体强度/尺度变化、恒定角调制下E50=0均通过。15/30/45度扇区及PCHIP/三点抛物线峰高为提取约定检查，不用于优化参数选择性。",
        "E50和Qbar缺少A0–C联合协方差、b层C均值及prefix系数，未给出统计误差或b/取向收敛结论。",
        "Vminus四个case的最低保存节点都在q=0.0275 GeV/c。β2/β3全协方差用于检查与首负区间其余节点的差距；局部delta误差只描述该离散网格极小值的取向积分误差。球形/skin是确定性计算，其零取向误差不等于无系统误差。",
        "完整b步长和prefix/half结果在对应CSV。图上的纵误差不包含有限q网格、探测器或模型误差；没有E50横误差条不表示横误差为零。", "",
        "| 案例 | Vminus取向SE（离散网格局部delta） | 最小竞争节点差距/SE |",
        "|---|---:|---:|"]
    for case in ("beta2", "beta3"):
        lines.append(f"| {case} | {results[case]['Vminus_orientation_se_grid_delta']:.6g} | {results[case]['smallest_competing_node_gap_in_se']:.2f} |")
    lines += ["", "15/30/45度扇区下三类变化的E50响应均为负；缩窄扇区会增强所有响应，没有实现中子皮消除。PCHIP/抛物线峰高检查也不改变响应方向。",
        "", "首负谷的连续位置不由共同的最低保存节点唯一决定。details.json另存三点抛物线、native系数角关联和同q网格解析角关联的诊断结果。", "",
        "| 案例 | b步长 (fm) | ΔV相对fine的变化 (%) |",
        "|---|---:|---:|"]
    for row in b_rows:
        if row["case"] != "spherical" and row["b_step_fm"] != summary["b_steps_fm"][0]:
            lines.append(f"| {row['case']} | {row['b_step_fm']:g} | {row['structure_change_percent']:+.3f} |")
    lines += ["", "这里ΔV=Vcase−Vsphere，fine为b步长0.5 fm；这些变化是数值诊断，不是严格误差界。", "",
        "![比较图](angular_shape.png)", "", "## 复现", "", "```bash",
        f"python3 scripts/analyze_single_image_angular_shape.py --input '{args.input}' --output outputs/single_image_angular_shape/replay_new_directory", "```", ""]
    (output/"report.md").write_text("\n".join(lines))
    metadata = {"source_analysis": str(source), "source_analysis_fingerprint": manifest["fingerprint"],
        "input_sha256": hashes, "code_sha256": {str(Path(__file__).name): sha256(__file__),
            "analyze_single_image_shape.py": sha256(Path(__file__).with_name("analyze_single_image_shape.py"))},
        "figure_contract": {"claim": "All three structural changes lower E50 and deepen the first negative C2 interval at these benchmarks.",
            "archetype": "quantitative scatter", "backend": "existing Python matplotlib workflow", "exports": "PNG PDF SVG; editable text; metrics.csv"},
        "output_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output/"manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
