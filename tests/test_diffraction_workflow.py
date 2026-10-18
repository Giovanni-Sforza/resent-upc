"""End-to-end artifact contract on a deliberately coarse physical campaign."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import torch

from upc_deformation import diffraction_campaign as campaign
from upc_deformation.precision_validation import tiny_configuration


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def analyzer_module():
    path = campaign.ROOT / "scripts/analyze_diffraction_observables.py"
    spec = importlib.util.spec_from_file_location("diffraction_workflow_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_tiny_campaign_analysis_figures_and_compact_report_archive(tmp_path):
    run = tmp_path / "production"
    cfg = campaign.load_config(campaign.ROOT / "configs/diffraction_observables_a100.yaml", {
        "device": "cpu", "n_orientations": 4, "output": str(run), "batch_size": 2,
    })
    cfg = tiny_configuration(cfg)
    cfg["orientations"].update(import_bank=None, import_sha256=None, block_size=4)
    cfg["analysis"].update(q_points=35, angular_points=64, orientation_prefixes=[2, 4])
    production = campaign.run_campaign(cfg)
    assert production["status"] == "completed"
    assert production["npairs"] == {"spherical": 1, "beta2": 4, "beta3": 4, "skin": 1}
    for name, count in production["npairs"].items():
        path = run / f"{name}.npz"
        assert campaign.sha256(path) == production["output_sha256"][path.name]
        with np.load(path, allow_pickle=False) as saved:
            assert saved["branch_names"].tolist() == ["full", "self"]
            assert saved["orientation_coefficients"].shape == (count, 2, 3, 65)
            np.testing.assert_array_equal(saved["orientation_coefficients"], saved["b_level_coefficients"][0])

    analyzer = analyzer_module()
    summary = analyzer.run_analysis(run, figures=True, pack=True)
    out = run / "analysis_n4"
    assert summary["n_orientations"] == production["npairs"]
    assert summary["registry"]["definitions"]["density"] == "sphere self A0; no q Jacobian"
    manifest = json.loads((out / "analysis_manifest.json").read_text())
    assert manifest["status"] == "completed"
    for relative, digest in manifest["output_sha256"].items():
        assert campaign.sha256(out / relative) == digest
    for suffix in ("*.png", "*.pdf", "*.svg"):
        assert list(out.glob(suffix)), f"missing rendered figure format {suffix}"
    assert (out / "report.md").stat().st_size > 0

    with np.load(out / "summary_arrays.npz", allow_pickle=False) as saved:
        assert saved.files
        assert any(key.startswith("b_grid__") and key.endswith("covariance") for key in saved.files)
        for key in saved.files:
            assert not any(token in key for token in
                           ("_samples", "orientation_coefficients", "b_level_coefficients", "rotation_bank"))
            assert saved[key].ndim <= 3
        assert np.all(saved["q"] > 0)

    with (out / "radial_density.csv").open(newline="") as stream:
        radial = list(csv.DictReader(stream))
    assert {row["case"] for row in radial} == set(production["npairs"])
    for row in radial:
        d, j = float(row["D"]), float(row["J"])
        np.testing.assert_allclose(float(row["X_J_minus_D"]), j-d, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(float(row["dN_full_dq_2piqJ"]),
                                   2*np.pi*float(row["q_GeV"])*j, rtol=1e-14)
    with (out / "window_metrics.csv").open(newline="") as stream:
        windows = list(csv.DictReader(stream))
    assert windows
    for row in windows:
        for branch in ("full", "self"):
            np.testing.assert_allclose(float(row[f"C2_{branch}"]),
                float(row[f"N2_{branch}"])/float(row[f"yield_{branch}"]), rtol=1e-12, atol=1e-14)

    archive = out.with_name(out.name + "_reports.tar.gz")
    archive_manifest = json.loads(archive.with_suffix(archive.suffix + ".manifest.json").read_text())
    assert campaign.sha256(archive) == archive_manifest["archive_sha256"]
    assert archive.stat().st_size == archive_manifest["bytes"]
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        assert all(member.isfile() and member.name.startswith(out.name + "/") for member in members)
        assert len(members) == len(archive_manifest["files"])
        for member in members:
            relative = member.name.removeprefix(out.name + "/")
            assert hashlib.sha256(bundle.extractfile(member).read()).hexdigest() == archive_manifest["files"][relative]
            assert not any(token in relative for token in ("checkpoints/", "rotation_bank", "orientation_bank"))

    # The completed report is bound to the original production files, even
    # when analysis is directed to a fresh output directory.
    with (run / "skin.npz").open("ab") as stream:
        stream.write(b"intentional checksum corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        analyzer.run_analysis(run, output=tmp_path / "tampered_analysis", figures=False, pack=False)
