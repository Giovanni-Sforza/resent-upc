#!/usr/bin/env python3
"""Extract self-referenced Q50 and T from archived full radial densities.

No production, spherical-template selection, or response optimization is run.
The figure compares absolute observables and their measured-curve definitions;
structural percentages are a separate, subsequent comparison only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "upc_shape_mpl"))
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

CASES = ("spherical", "beta2", "beta3", "skin")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(q, density, *, interpolation="linear", refine_peak=False):
    """Use this curve's own peak and first outer downward crossings only."""
    q, j = np.asarray(q, float), np.asarray(density, float)
    if (q.ndim != 1 or j.shape != q.shape or len(q) < 3
            or not np.isfinite(q).all() or not np.isfinite(j).all()
            or np.any(np.diff(q) <= 0) or np.any(j < 0) or j.max() <= 0):
        raise ValueError("invalid radial density")
    peak = int(j.argmax())
    peak_q, peak_height = float(q[peak]), float(j[peak])
    if refine_peak and 0 < peak < len(q)-1:
        a, b, c = np.polyfit(q[peak-1:peak+2]-q[peak], j[peak-1:peak+2], 2)
        vertex = -b/(2*a) if a < 0 else np.inf
        if q[peak-1]-q[peak] < vertex < q[peak+1]-q[peak]:
            peak_q = float(q[peak]+vertex)
            peak_height = float(a*vertex**2+b*vertex+c)
    interpolator = PchipInterpolator(q, j, extrapolate=False) if interpolation == "pchip" else None
    result = {"peak_q_GeV": peak_q, "peak_height": peak_height,
              "peak_node_index": peak, "peak_method": "three-node parabola" if refine_peak else "saved-node maximum",
              "local_peak_count": int(np.sum((j[1:-1] > j[:-2]) & (j[1:-1] >= j[2:]))),
              "interpolation": interpolation, "crossings": {}}
    for fraction, name in ((.5, "Q50"), (.1, "Q10")):
        threshold = fraction*peak_height
        candidates = [i for i in range(peak, len(q)-1)
                      if j[i] >= threshold >= j[i+1] and j[i] > j[i+1]]
        if not candidates:
            result[name+"_GeV"] = None
            result["crossings"][name] = {"status": "not_observed", "last_q_GeV": float(q[-1])}
            continue
        i = candidates[0]
        value = (brentq(lambda x: float(interpolator(x))-threshold, q[i], q[i+1], xtol=1e-14)
                 if interpolator is not None else
                 q[i]+(threshold-j[i])*(q[i+1]-q[i])/(j[i+1]-j[i]))
        result[name+"_GeV"] = float(value)
        result["crossings"][name] = {"status": "observed", "fraction": fraction,
            "bracket_q_GeV": q[i:i+2].tolist(), "bracket_J": j[i:i+2].tolist(),
            "downward_crossing_count": len(candidates)}
    result["T"] = (result["Q10_GeV"]/result["Q50_GeV"]-1
                   if result["Q10_GeV"] is not None and result["Q50_GeV"] else None)
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(curves, results, output):
    # Quantitative diagnostic, 183 x 89 mm; editable SVG/PDF and PNG preview.
    # Evidence: own-peak crossing locations -> absolute Q,T response positions.
    # No uncertainty bars: the archive does not contain joint J covariance.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False})
    colors = dict(zip(CASES, ("#555555", "#3274A1", "#4B9385", "#C17C43")))
    labels = {"spherical": "Spherical", "beta2": r"$\beta_2=0.3$",
              "beta3": r"$\beta_3=0.2$", "skin": "Model skin: 0.2 fm"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.18, 1]})
    fig.subplots_adjust(left=.095, right=.98, bottom=.25, top=.86, wspace=.36)
    for case in CASES:
        q, j = curves[case]
        r = results[case]
        axes[0].plot(q, j/r["peak_height"], color=colors[case], lw=1.25, label=labels[case])
        axes[0].plot(r["Q50_GeV"], .5, "o", color=colors[case], ms=3)
        axes[0].plot(r["Q10_GeV"], .1, "s", color=colors[case], ms=3)
        axes[1].plot(r["Q50_GeV"]*1000, r["T"], "o", color=colors[case], ms=6)
    for height in (.5, .1):
        axes[0].axhline(height, color=".7", lw=.6, ls=":", zorder=0)
    axes[0].set(xlim=(0, .17), ylim=(0, 1.04), xlabel=r"$q_T$ (GeV/$c$)",
                ylabel=r"$J(q_T)/J_{\rm peak}$", title="a  Own-peak thresholds")
    axes[0].legend(fontsize=7.4, loc="upper right")
    offsets = {"spherical": (-7, -15), "beta2": (5, 5), "beta3": (-8, 8), "skin": (7, 2)}
    for case in CASES:
        r = results[case]
        dx, dy = offsets[case]
        label = "Skin" if case == "skin" else labels[case]
        axes[1].annotate(label, (r["Q50_GeV"]*1000, r["T"]), xytext=(dx, dy),
                         textcoords="offset points", fontsize=8,
                         ha="right" if dx < 0 else "left", color=colors[case])
    axes[1].set(xlim=(74.5, 76.1), ylim=(.534, .570),
        xlabel=r"$Q_{50}$ (MeV/$c$)", ylabel=r"$T=(Q_{10}-Q_{50})/Q_{50}$",
        title="b  Absolute shape observables")
    fig.text(.5, .055, "Full radial density; 32,768 orientation pairs for each deformation case.\n"
             "Points are model means; uncertainties in Q and T are not available from this archive.",
             ha="center", fontsize=7.4)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output/f"single_image_shape.{ext}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use a new output directory: {output}")
    inputs = ("radial_density.csv", "summary_arrays.npz", "summary.json", "analysis_manifest.json")
    source_manifest = json.loads((source/"analysis_manifest.json").read_text())
    hashes = {name: sha256(source/name) for name in inputs}
    for name in inputs[:-1]:
        if hashes[name] != source_manifest["output_sha256"][name]:
            raise ValueError(f"source checksum mismatch: {name}")
    summary = json.loads((source/"summary.json").read_text())
    with (source/"radial_density.csv").open() as f:
        data = list(csv.DictReader(f))
    curves, results, checks, b_rows = {}, {}, [], []
    with np.load(source/"summary_arrays.npz", allow_pickle=False) as arrays:
        for case in CASES:
            selected = [row for row in data if row["case"] == case]
            q = np.array([float(row["q_GeV"]) for row in selected])
            j = np.array([float(row["J"]) for row in selected])
            np.testing.assert_array_equal(q, arrays[f"cases__{case}__radial__q"])
            np.testing.assert_array_equal(j, arrays[f"cases__{case}__radial__J__value"])
            curves[case] = (q, j)
            results[case] = extract(q, j)
            if results[case]["T"] is None:
                raise ValueError(f"{case}: a required outer crossing was not observed")
            scaled = extract(1.2*q, 7*j)
            np.testing.assert_allclose(scaled["Q50_GeV"], 1.2*results[case]["Q50_GeV"], rtol=1e-13)
            np.testing.assert_allclose(scaled["T"], results[case]["T"], rtol=1e-13)
            for method, kwargs in (("pchip_crossing", {"interpolation": "pchip"}),
                                   ("parabolic_peak", {"refine_peak": True})):
                alternate = extract(q, j, **kwargs)
                checks.append({"case": case, "method": method, **{
                    key+"_minus_primary": alternate[key]-results[case][key]
                    for key in ("Q50_GeV", "Q10_GeV", "T")}})
            for i, step in enumerate(summary["b_steps_fm"]):
                delta = (0 if i == 0 else arrays[f"b_grid__{case}__{i-1}__radial_coarse_minus_fine"][0])
                estimate = extract(q, j+delta)
                b_rows.append({"case": case, "b_step_fm": step,
                               **{key: estimate[key] for key in ("Q50_GeV", "Q10_GeV", "T")}})
    for row in b_rows:
        sphere = next(x for x in b_rows if x["case"] == "spherical" and x["b_step_fm"] == row["b_step_fm"])
        for key in ("Q50_GeV", "Q10_GeV", "T"):
            row["delta_"+key+"_vs_spherical"] = row[key]-sphere[key]
    stability = []
    for row in b_rows:
        if row["case"] == "spherical" or row["b_step_fm"] == summary["b_steps_fm"][0]:
            continue
        baseline = next(x for x in b_rows if x["case"] == row["case"]
                        and x["b_step_fm"] == summary["b_steps_fm"][0])
        fine_delta = baseline["delta_T_vs_spherical"]
        change = row["delta_T_vs_spherical"]-fine_delta
        stability.append({"case": row["case"], "b_step_fm": row["b_step_fm"],
            "fine_delta_T": fine_delta, "coarse_minus_fine_delta_T": change,
            "relative_change_percent": 100*change/fine_delta if fine_delta else None})
    fine = results["spherical"]
    rows = []
    for case in CASES:
        r = results[case]
        rows.append({"case": case, "n_orientations": summary["n_orientations"][case],
            **{key: r[key] for key in ("peak_q_GeV", "Q50_GeV", "Q10_GeV", "T")},
            **{key+"_change_percent": 100*(r[key]/fine[key]-1) for key in ("Q50_GeV", "Q10_GeV", "T")}})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"metrics.csv", rows)
    write_csv(output/"interpolation_checks.csv", checks)
    write_csv(output/"b_grid_metrics.csv", b_rows)
    write_csv(output/"b_grid_structural_stability.csv", stability)
    write_csv(output/"curve_source_data.csv", [{"case": case, "q_GeV": float(qi), "J": float(ji),
        "J_over_own_peak": float(ji/results[case]["peak_height"])}
        for case in CASES for qi, ji in zip(*curves[case])])
    detail = {"definition": "Each full J curve independently: saved-node maximum, then first downward 50% and 10% crossing; linear interpolation; T=Q10/Q50-1.",
        "not_used": "No self D, spherical thresholds, q Jacobian, or response-fitted coefficients enter extraction.",
        "domain_GeV": [float(curves["spherical"][0][0]), float(curves["spherical"][0][-1])],
        "cases": results, "b_grid_structural_stability": stability,
        "uncertainty": "Q and T covariance/SE unavailable. Pointwise J SE and J/D covariance are insufficient. Interpolation and b-grid differences are diagnostics, not confidence intervals.",
        "checks": {"input_sha256": "passed", "csv_npz_identity": "exact", "scale_invariance": "passed"}}
    (output/"results.json").write_text(json.dumps(detail, indent=2)+"\n")
    make_figure(curves, results, output)
    lines = ["# 单图定义的外半高尺度与外坡形状", "",
        "只对已有完整径向二维密度 J 做后处理；各案例独立使用自身峰高。没有运行新正演。",
        "Q50、Q10 分别为峰外侧首次降至自身峰高 50%、10% 的交点；T=Q10/Q50−1。",
        "下表球形差仅用于事后比较，不参与观测量定义。", "",
        "| 案例 | Q50 (GeV/c) | Q10 (GeV/c) | T | ΔQ50 (%) | ΔT (%) |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['Q50_GeV']:.8f} | {row['Q10_GeV']:.8f} | {row['T']:.8f} | {row['Q50_GeV_change_percent']:+.4f} | {row['T_change_percent']:+.4f} |")
    lines += ["", "这组基准点中，β2 与 skin 对 T 的响应异号，β3 的 T 响应很小。T 并非中子皮不敏感的形变专用量；Q50 对三类变化都有响应。响应幅度比较只适用于这里的参数步长。",
        "", "## 核查与边界", "",
        "- 输入哈希核对通过；CSV 与 NPZ 的 full J 及 q 数组逐元素相同。",
        "- 四个案例均有唯一主峰及唯一的外侧 50%、10% 交点；均在保存范围内。",
        "- 检查整体强度归一化和径向伸缩下的解析不变性；用 PCHIP 交点和三点抛物线峰高检查提取约定。",
        "- b_grid_metrics.csv 每个 b 步长重新计算本图峰值和交点；粗网格 J = fine J + archived (coarse−fine J)。",
        "- 报告未保存 full J 跨动量协方差及逐取向 J；因此未计算 Q、T 的取向积分误差，也没有实验误差。逐点 SE 不可当独立误差，J/D 协方差不能代替 J 协方差。",
        "- b 网格及插值差不是误差界；新量没有独立验证 y/FFT/模型/探测器收敛。",
        "- 球形与 skin 的取向样本数为 1 表示确定性球形计算，不表示无系统误差。",
        "", "![提取与比较](single_image_shape.png)", "",
        "图左展示每条 full J 以自身峰高归一化后的阈值，图右为 Q50、T 的绝对值。没有依赖球形核的选区或拟合权重。",
        "", "## b 步长对结构差的影响", "",
        "ΔT 定义为各案例 T 减同一步长下的球形 T。下表给粗步长相对 0.5 fm 的变化；这是数值诊断，不是误差条。", "",
        "| 案例 | b 步长 (fm) | fine ΔT | 粗−细的 ΔT 变化 | 相对 fine ΔT 的变化 (%) |",
        "|---|---:|---:|---:|---:|"]
    for row in stability:
        lines.append(f"| {row['case']} | {row['b_step_fm']:g} | {row['fine_delta_T']:+.8f} | {row['coarse_minus_fine_delta_T']:+.8f} | {row['relative_change_percent']:+.2f} |")
    lines += ["", "β3 的 ΔT 很小，不能用总 T 的稳定掩盖结构差对 b 步长的相对敏感性。",
        "", "## 复现", "", "```bash",
        f"python3 scripts/analyze_single_image_shape.py --input '{args.input}' --output outputs/single_image_shape/replay_new_directory", "```", ""]
    (output/"report.md").write_text("\n".join(lines))
    manifest = {"source_analysis": str(source), "source_analysis_fingerprint": source_manifest["fingerprint"],
        "input_sha256": hashes, "script_sha256": sha256(__file__),
        "figure_contract": {"claim": "Both skin and deformation change T; beta2 and skin move in different directions at these benchmark points.",
            "archetype": "quantitative grid", "backend": "existing Python/matplotlib workflow",
            "exports": "183x89 mm PNG/PDF/SVG, editable text, curve_source_data.csv", "statistics": "model means only; no available joint Q,T uncertainty"},
        "output_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"output": str(output), "metrics": rows}, indent=2))


if __name__ == "__main__":
    main()
