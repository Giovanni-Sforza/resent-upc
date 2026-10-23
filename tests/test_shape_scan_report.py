"""Small synthetic checkpoint reports; no physical forward calculation."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import upc_deformation.shape_scan_campaign as campaign

SPEC = importlib.util.spec_from_file_location("analyze_shape_scan", campaign.ROOT/"scripts/analyze_shape_scan.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class SyntheticRefiner:
    slope = 300.

    def __init__(self, cfg, case):
        self.cfg, self.case = cfg, case

    def calculate(self, r1, r2, return_nodes=False):
        s = self.cfg["study"]
        q = np.linspace(0., .3, s["n_radial"])
        slope = self.slope+20*r1[:, 0, 0]+10*r2[:, 0, 0]
        profile = np.exp(-slope[:, None]*q[None, :]**2)
        amplitude = 2. if self.case["density_type"] == "skin" else 1.
        coef = np.zeros((len(r1), 2, 3, len(q)))
        coef[:, 0, 0] = amplitude*profile
        coef[:, 1, 0] = 3*amplitude*profile
        levels = np.stack((coef, 1.01*coef, .98*coef))
        nb = s["b_nodes"]
        return {"branch_names": ["full", "self"], "coefficients": coef,
            "b_level_coefficients": levels, "radii": q, "axis": np.linspace(-.2, .2, 5),
            "p_full_axis": np.linspace(-1, 1, 32), "b_values": np.linspace(12, 60, nb),
            "b_weights": np.ones(nb), "b_steps_fm": np.array([12., 24., 48.]),
            "b_level_weights": np.ones((3, nb)), "y_values": np.zeros(1),
            "y_weights": np.ones(1), "elapsed_seconds": 0.}


@pytest.fixture
def run_config(tmp_path, monkeypatch):
    cfg = yaml.safe_load((campaign.ROOT/"configs/diffraction_observables_a100.yaml").read_text())
    cfg["scan"] = {"beta2_values": [0., .3], "beta3_values": [0.], "density_variants": [
        {"id": "none_a", "type": "none", "mode": "rms_matched", "R_p": 5.02, "a_p": .46, "delta_r_np_fm": 0.},
        {"id": "none_b", "type": "none", "mode": "rms_matched", "R_p": 5.1, "a_p": .46, "delta_r_np_fm": 0.},
        {"id": "skin", "type": "skin", "mode": "rms_matched", "R_p": 5.02, "a_p": .46, "delta_r_np_fm": .2}]}
    cfg["orientations"].update(n_orientations=8, block_size=4, import_bank=None, import_sha256=None)
    cfg["runtime"].update(device="cpu", batch_size=3, checkpoint_size=4, output=str(tmp_path/"run"))
    cfg["study"].update(n_radial=61, n_phi=8, b_step_fm=12., n_y=1, qmax=.173)
    cfg["analysis"] = {"bootstrap_draws": 16, "bootstrap_seed": 11, "bootstrap_block_size": 2,
                       "bootstrap_batch_size": 3, "prefixes": [2, 4, 8, 16]}
    file = tmp_path/"cfg.yaml"; file.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(campaign, "code_hashes", lambda: {"synthetic_kernel": "fixed"})
    SyntheticRefiner.slope = 300.

    def build(c, case):
        v = case["density_variant"]
        return SyntheticRefiner(c, case), {"id": v["id"], "type": v["type"], "mode": v["mode"],
            "R_ws_p_fm": v["R_p"], "R_ws_n_fm": v["R_p"]+.1, "a_p_fm": .46, "a_n_fm": .46,
            "delta_r_np_fm": v["delta_r_np_fm"], "proton_rms_fm": 4., "neutron_rms_fm": 4.2,
            "test_only": True}
    monkeypatch.setattr(campaign, "build_refiner", build)
    return campaign.load_config(file)


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_complete_report_joint_pairing_prefixes_and_hashes(run_config, tmp_path):
    cfg = run_config
    campaign.run_campaign(cfg)
    run, out = Path(cfg["runtime"]["output"]), tmp_path/"analysis"
    summary = report.analyze_run(run, out, n=4, no_plots=True)
    assert len(summary["analyzed_cases"]) == 6 and summary["prefixes"] == [2, 4]
    assert summary["nominal_qmax_GeV"] == .173
    assert summary["effective_saved_qmax_GeV"] == pytest.approx(.17)
    rows = read_rows(out/"metrics.csv")
    assert len(rows) == 36 and {row["branch"] for row in rows} == {"full", "self"}
    assert {row["b_level"] for row in rows} == {"0", "1", "2"}
    assert {row["n_orientations"] for row in rows} == {"1", "4"}
    assert {float(row["R_ws_p_fm"]) for row in rows} == {5.02, 5.1}
    assert np.shape(json.loads(rows[0]["covariance_json"])) == (len(report.FEATURE_NAMES), len(report.FEATURE_NAMES))
    assert summary["schema"] == "single_image_shape_scan_report_v2"
    assert summary["mean_pT_support_GeV"] == pytest.approx([0., .17])
    prefix = read_rows(out/"prefix_stability.csv")
    assert len(prefix) == 72 and {row["requested_prefix"] for row in prefix} == {"2", "4"}
    assert all(row["mean_pT_GeV"] != "" for row in prefix)
    sphere = [row for row in prefix if float(row["beta2"]) == 0]
    assert {row["n_orientations"] for row in sphere} == {"1"}
    paired = read_rows(out/"paired_response.csv")
    matched = [row for row in paired if row["contrast"] == "matched_beta_vs_none"
               and row["density_id"] == "skin" and row["beta2"] == "0.3"]
    assert {row["reference_density_id"] for row in matched} == {"none_a", "none_b"}
    # The fake skin differs only by intensity; shared orientation resampling
    # must cancel its fluctuations, unlike independent-SE quadrature.
    assert max(abs(float(row["difference"])) for row in matched) < 1e-10
    assert max(float(row["difference_orientation_se"]) for row in matched) < 1e-10
    assert max(float(row["ratio_orientation_se"]) for row in matched) < 1e-12
    name = summary["analyzed_cases"][1]
    compact = json.loads((out/"cases"/f"{name}.json").read_text())
    assert "replicate_features" not in compact and "mean_A0" not in compact
    assert compact["bootstrap"]["orientation_bank_id"] == summary["bootstrap_options"]["orientation_bank_id"]
    with np.load(out/"bootstrap_replicates.npz", allow_pickle=False) as z:
        reps = z[compact["replicate_npz_key"]]
        assert reps.shape == (3, 2, 16, len(report.FEATURE_NAMES))
        np.testing.assert_allclose(np.cov(reps[0, 0], rowvar=False), compact["channels"][0][0]["covariance"])
    with np.load(out/"radial_means.npz", allow_pickle=False) as z:
        assert z[compact["radial_mean_npz_key"]].shape == (3, 2, len(z["q"]))
    manifest = json.loads((out/"report_manifest.json").read_text())
    assert all(report.sha256(out/file) == digest for file, digest in manifest["output_sha256"].items())
    with pytest.raises(ValueError, match="nonempty"):
        report.analyze_run(run, out, no_plots=True)
    with pytest.raises(ValueError, match="exceeds"):
        report.analyze_run(run, tmp_path/"excess", n=9, no_plots=True)
    with pytest.raises(ValueError, match="divisible"):
        report.analyze_run(run, tmp_path/"remainder", n=3, no_plots=True)
    with pytest.raises(ValueError, match="deterministic deformed"):
        report.analyze_run(run, tmp_path/"singleton", n=1, no_plots=True)


def test_partial_only_includes_complete_cases_even_with_short_prefix(run_config, tmp_path):
    cfg = run_config
    campaign.run_campaign(cfg, max_new_pairs=3)
    run = Path(cfg["runtime"]["output"])
    with pytest.raises(ValueError, match="incomplete"):
        report.analyze_run(run, tmp_path/"refused", n=2, no_plots=True)
    assert not (tmp_path/"refused").exists()
    summary = report.analyze_run(run, tmp_path/"partial", n=2, allow_partial=True, no_plots=True)
    assert len(summary["analyzed_cases"]) == 1
    assert len(summary["missing_cases"]) == 5
    assert any(row["completed_pairs"] == 2 for row in summary["missing_cases"])
    assert {row["contrast"] for row in read_rows(tmp_path/"partial"/"paired_response.csv")} == {
        "full_vs_self", "b_refinement_vs_finest"}


def test_censoring_and_one_axis_plot_export(run_config, tmp_path):
    cfg = run_config
    SyntheticRefiner.slope = 120.
    campaign.run_campaign(cfg)
    out = tmp_path/"plots"
    summary = report.analyze_run(cfg["runtime"]["output"], out, bootstrap_draws=8)
    rows = read_rows(out/"metrics.csv")
    assert all(row["K_tail"] == "" and row["K_tail_orientation_se"] == "" for row in rows)
    assert all(row["Q10_GeV_marginal_orientation_se"] != "" and row["B_mid_marginal_orientation_se"] != ""
               and row["K_tail_marginal_orientation_se"] == ""
               and row["mean_pT_GeV_marginal_orientation_se"] != "" for row in rows)
    assert all(row["covariance_json"] == "null" and row["status"] == "censored" for row in rows)
    assert len(summary["figures"]) == 3*len(report.FEATURE_NAMES)
    for filename in summary["figures"]:
        assert (out/filename).stat().st_size > 1000
    assert "<text" in (out/"Q10_GeV_response.svg").read_text()
    assert summary["figure_qa"]["K_tail"]["colour_limits"] is None
    assert summary["figure_qa"]["mean_pT_GeV"]["observed_cells"] == 6
    paired = read_rows(out/"paired_response.csv")
    usable = [row for row in paired if row["feature"] != "K_tail"]
    assert usable and all(row["difference_marginal_orientation_se"] != ""
                          and row["ratio_marginal_orientation_se"] != "" for row in usable)
    assert all(row["difference_orientation_se"] == "" for row in paired)
    assert all(row["ratio_marginal_orientation_se"] == "" for row in paired if row["feature"] == "K_tail")


def test_bank_rotation_mismatch_rejected(run_config, tmp_path):
    cfg = run_config
    campaign.run_campaign(cfg)
    run = Path(cfg["runtime"]["output"])
    name = cfg["study"]["cases"][1]["name"]
    state_path = run/"cases"/name/"manifest.json"
    state = json.loads(state_path.read_text())
    filename = sorted(state["chunks"])[0]
    chunk_path = state_path.parent/filename
    with np.load(chunk_path, allow_pickle=False) as z:
        arrays = dict(z)
    arrays["rotation1"][0] = np.eye(3)
    campaign.atomic_npz(chunk_path, **arrays)
    state["chunks"][filename]["sha256"] = campaign.sha256(chunk_path)
    campaign.atomic_json(state_path, state)
    root = json.loads((run/"production_manifest.json").read_text())
    root["case_manifest_sha256"][name] = campaign.sha256(state_path)
    campaign.atomic_json(run/"production_manifest.json", root)
    with pytest.raises(ValueError, match="saved rotations"):
        report.analyze_run(run, tmp_path/"bad_rotation", no_plots=True)


def test_cli_no_plots_and_explicit_bootstrap_override(run_config, tmp_path):
    cfg = run_config
    campaign.run_campaign(cfg)
    out = tmp_path/"cli"
    report.main(["--run", cfg["runtime"]["output"], "--output", str(out),
                 "--n", "4", "--bootstrap-draws", "6", "--no-plots"])
    summary = json.loads((out/"summary.json").read_text())
    assert summary["bootstrap_options"]["bootstrap_draws"] == 6 and summary["figures"] == []


def test_no_sphere_or_none_is_explicitly_skipped(run_config, tmp_path):
    cfg = run_config
    cfg["scan"]["beta2_values"] = [.3]
    cfg["scan"]["density_variants"] = [cfg["scan"]["density_variants"][-1]]
    cfg["study"]["cases"] = campaign.build_cases(cfg)
    campaign.run_campaign(cfg)
    summary = report.analyze_run(cfg["runtime"]["output"], tmp_path/"no_reference", no_plots=True)
    assert any("sphere absent" in note for note in summary["comparison_notes"])
    assert any("none case absent" in note for note in summary["comparison_notes"])
