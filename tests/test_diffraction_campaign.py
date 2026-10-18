"""Production mechanics are tested with a cheap deterministic batched kernel."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

import upc_deformation.diffraction_campaign as campaign


@pytest.fixture
def config(tmp_path, monkeypatch):
    cfg = campaign.load_config(campaign.ROOT / "configs/panel_d_precision_a100.yaml",
        {"device": "cpu", "n_orientations": 4, "b_step_fm": 2., "fft_mode": "separate", "output": str(tmp_path / "run")})
    cfg["orientations"].update(import_bank=None, import_sha256=None, block_size=3)
    cfg["study"].update(n_radial=9, n_phi=8)
    monkeypatch.setattr(campaign, "code_hashes", lambda: {"fake-test-kernel": "immutable-v1"})
    return cfg


class FakeRefiner:
    calls = []
    fail_at = None

    def __init__(self, cfg, case):
        self.cfg, self.case = cfg, case

    def calculate(self, r1, r2, return_nodes=False):
        assert not return_nodes
        type(self).calls.append((self.case["name"], len(r1)))
        if len(type(self).calls) == type(self).fail_at: raise RuntimeError("simulated interruption")
        study = self.cfg["study"]; nr = study["n_radial"]
        coeff = np.ones((len(r1), 2, 3, nr)) * (1 + r1[:, 0, 0] + r2[:, 1, 1])[:, None, None, None]
        coeff[:, 1] *= 1.3
        coeff += self.case["beta2"] + self.case["beta3"] + self.case["skin_fm"]
        step = study["b_step_fm"]; nb = study["b_nodes"]; levels = np.stack((coeff, coeff*1.01, coeff*1.02))
        return {"branch_names": ["full", "self"], "coefficients": coeff, "b_level_coefficients": levels,
                "axis": np.linspace(-.2, .2, 5), "radii": np.linspace(0, .3, nr),
                "p_full_axis": np.linspace(-1, 1, 16), "b_values": np.linspace(12, 60, nb),
                "b_weights": np.ones(nb), "b_steps_fm": np.array([step, 2*step, 4*step]),
                "b_level_weights": np.ones((3, nb)), "y_values": np.linspace(-1, 1, study["n_y"]),
                "y_weights": np.ones(study["n_y"]), "elapsed_seconds": .001}


@pytest.fixture
def fake_kernel(monkeypatch):
    FakeRefiner.calls = []; FakeRefiner.fail_at = None
    monkeypatch.setattr(campaign, "build_refiner", lambda cfg, case: FakeRefiner(cfg, case))
    return FakeRefiner


def test_identity_excludes_scheduling_but_binds_scientific_changes(config):
    _, bank_meta = campaign.rotation_bank(config["orientations"], 4)
    _, first = campaign.scientific_identity(config, bank_meta)
    larger = deepcopy(config)
    larger["orientations"]["n_orientations"] = 16
    larger["runtime"].update(batch_size=4, output="elsewhere")
    larger["study"]["y_batch_size"] = 3
    assert campaign.scientific_identity(larger, bank_meta)[1] == first
    for section, key, value in [("runtime", "dtype", "float32"), ("runtime", "device", "cuda"),
                                ("study", "b_step_fm", 1.), ("study", "n_y", 9)]:
        changed = deepcopy(config); changed[section][key] = value
        assert campaign.scientific_identity(changed, bank_meta)[1] != first


def test_atomic_resume_batch_change_and_target_extension(config, fake_kernel):
    config["runtime"]["batch_size"] = 3
    first = campaign.run_campaign(config, max_new_pairs=3)
    assert first["status"] == "incomplete" and first["completed_pairs"] == 3
    out = Path(config["runtime"]["output"])
    old = {str(p): campaign.sha256(p) for p in (out / "checkpoints").glob("*/*.npz")}
    config["runtime"]["batch_size"] = 1
    done = campaign.run_campaign(config)
    assert done["status"] == "completed" and done["completed_pairs"] == 10
    assert done["npairs"] == {"spherical": 1, "beta2": 4, "beta3": 4, "skin": 1}
    for path, checksum in old.items(): assert campaign.sha256(path) == checksum
    all_old = {str(p): campaign.sha256(p) for p in (out / "checkpoints").glob("*/*.npz")}
    original_bank = campaign.sha256(out / done["bank_file"])
    config["orientations"]["n_orientations"] = 8
    config["runtime"]["batch_size"] = 3
    extended = campaign.run_campaign(config)
    assert extended["fingerprint"] == done["fingerprint"]
    assert extended["completed_pairs"] == 18 and extended["new_pairs_this_session"] == 8
    assert campaign.sha256(out / done["bank_file"]) == original_bank
    for path, checksum in all_old.items(): assert campaign.sha256(path) == checksum
    for name, n in extended["npairs"].items():
        with np.load(out / f"{name}.npz") as z:
            assert z["orientation_coefficients"].shape == (n, 2, 3, 9)
            assert z["b_level_coefficients"].shape == (3, n, 2, 3, 9)
            np.testing.assert_array_equal(z["orientation_coefficients"], z["b_level_coefficients"][0])
            assert z["rotation_bank_1"].shape == (n, 3, 3)
            assert z["branch_names"].tolist() == ["full", "self"]
            assert str(z["schema"]) == campaign.SCHEMA
        assert campaign.sha256(out / f"{name}.npz") == extended["output_sha256"][f"{name}.npz"]
    assert sum(n for _, n in fake_kernel.calls) == 18


def test_interruption_retains_completed_atomic_checkpoints(config, fake_kernel):
    fake_kernel.fail_at = 2
    with pytest.raises(RuntimeError, match="simulated"):
        campaign.run_campaign(config)
    out = Path(config["runtime"]["output"])
    manifest = json.loads((out / "production_manifest.json").read_text())
    assert manifest["status"] == "interrupted" and manifest["completed_pairs"] == 1
    path = out / "checkpoints/spherical/00000000.npz"; checksum = campaign.sha256(path)
    fake_kernel.fail_at = None
    assert campaign.run_campaign(config)["status"] == "completed"
    assert campaign.sha256(path) == checksum


def test_corrupted_checkpoint_and_changed_grid_are_rejected(config, fake_kernel):
    campaign.run_campaign(config, max_new_pairs=1)
    path = Path(config["runtime"]["output"]) / "checkpoints/spherical/00000000.npz"
    with np.load(path) as z: values = {k: z[k] for k in z.files}
    values["coefficients"][0, 0] += 1
    np.savez(path, **values)
    with pytest.raises(ValueError, match="data/rotation"):
        campaign.run_campaign(config)
    changed = deepcopy(config); changed["study"]["n_y"] = 9
    with pytest.raises(ValueError, match="identity changed"):
        campaign.run_campaign(changed)


def test_prepare_only_and_zero_budget_make_no_forward_calls(config, fake_kernel):
    prepared = campaign.run_campaign(config, prepare_only=True)
    assert prepared["status"] == "prepared" and prepared["completed_pairs"] == 0
    limited = campaign.run_campaign(config, max_new_pairs=0)
    assert limited["status"] == "incomplete" and limited["completed_pairs"] == 0
    assert not fake_kernel.calls
    assert prepared["environment"]["tf32_matmul"] is False
    assert prepared["environment"]["tf32_cudnn"] is False
    assert prepared["environment"]["amp_enabled"] is False


def test_cuda_unavailable_fails_without_cpu_fallback(config, monkeypatch, fake_kernel):
    monkeypatch.setattr(campaign.torch.cuda, "is_available", lambda: False)
    config["runtime"]["device"] = "cuda"
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        campaign.run_campaign(config, prepare_only=True)
    assert not fake_kernel.calls


def test_invalid_target_and_nondividing_b_step_fail_before_execution(config):
    config["orientations"]["n_orientations"] = 6
    with pytest.raises(ValueError, match="divisible"):
        campaign.validate_config(config)
    with pytest.raises(ValueError, match="divide"):
        campaign.load_config(campaign.ROOT / "configs/panel_d_precision_a100.yaml", {"b_step_fm": 1.3})


def test_real_kernel_one_pair_writes_compatible_checkpoint(config):
    # Interface smoke only: deliberately tiny grids, no production-accuracy claim.
    config["physics"]["grid_parameters"] = {"WS": {"N": 16, "L": 24.},
        "T_prim": {"N": 16, "L": 24.}, "output": {"N": 48, "L": 160.}}
    config["physics"]["output_settings"].update(safe_padding_factor=2, p_limit_q=.10, target_size_q=25)
    config["study"].update(b_step_fm=24., b_nodes=3, n_y=1, n_radial=31, n_phi=16, qmax=.08)
    result = campaign.run_campaign(config, max_new_pairs=1)
    assert result["status"] == "incomplete" and result["completed_pairs"] == 1
    out = Path(config["runtime"]["output"])
    with np.load(out / "checkpoints/spherical/00000000.npz") as row:
        assert row["coefficients"].shape == (2, 3, 31)
        assert row["b_level_coefficients"].shape == (2, 2, 3, 31)
        assert row["branch_names"].tolist() == ["full", "self"]
    with np.load(out / "grid.npz") as grid:
        np.testing.assert_array_equal(grid["b_values"], [12., 36., 60.])
        np.testing.assert_array_equal(grid["b_steps_fm"], [24., 48.])


def test_both_branches_share_orientations_and_deterministic_references(config, fake_kernel):
    campaign.run_campaign(config)
    out = Path(config["runtime"]["output"])
    with np.load(out / "beta2.npz") as beta2, np.load(out / "beta3.npz") as beta3:
        for name in ("rotation_bank_1", "rotation_bank_2"):
            np.testing.assert_array_equal(beta2[name], beta3[name])
    for case in ("spherical", "skin"):
        with np.load(out / f"{case}.npz") as z:
            assert z["orientation_coefficients"].shape[0] == 1
            np.testing.assert_array_equal(z["rotation_bank_1"], np.eye(3)[None])
            np.testing.assert_array_equal(z["rotation_bank_2"], np.eye(3)[None])


def test_separate_fft_and_legacy_schema_are_required(config, fake_kernel):
    wrong = deepcopy(config); wrong["runtime"]["fft_mode"] = "combined"
    with pytest.raises(ValueError, match="separate"):
        campaign.run_campaign(wrong)
    out = Path(config["runtime"]["output"]); out.mkdir()
    legacy = out / "production_manifest.json"
    legacy.write_text(json.dumps({"status": "completed", "fingerprint": "old"}))
    checksum = campaign.sha256(legacy)
    with pytest.raises(ValueError, match="different campaign schema"):
        campaign.run_campaign(config)
    assert campaign.sha256(legacy) == checksum
    assert not (out / "provenance.json").exists()


def test_branch_order_and_removed_registered_checkpoint_are_rejected(config, fake_kernel):
    campaign.run_campaign(config, max_new_pairs=1)
    out = Path(config["runtime"]["output"])
    path = out / "checkpoints/spherical/00000000.npz"
    original = path.read_bytes()
    with np.load(path) as z: values = {key: z[key] for key in z.files}
    values["branch_names"] = np.asarray(["self", "full"])
    np.savez(path, **values)
    with pytest.raises(ValueError, match="branch mismatch"):
        campaign.run_campaign(config)
    path.write_bytes(original)
    path.unlink()
    with pytest.raises(ValueError, match="checkpoint is missing"):
        campaign.run_campaign(config)


def test_analysis_window_and_branch_schema_bind_identity(config):
    _, metadata = campaign.rotation_bank(config["orientations"], 4)
    identity, original = campaign.scientific_identity(config, metadata)
    assert identity["schema"] == campaign.SCHEMA
    assert identity["branch_names"] == ["full", "self"]
    changed = deepcopy(config); changed["analysis"]["fixed_window_GeV"] = [.12, .14]
    assert campaign.scientific_identity(changed, metadata)[1] != original


def test_prepare_completed_run_validates_without_downgrading_or_new_session(config, fake_kernel):
    completed = campaign.run_campaign(config)
    out = Path(config["runtime"]["output"])
    scientific_files = [path for path in out.rglob("*") if path.is_file() and path.name != ".precision.lock"]
    hashes = {str(path.relative_to(out)): campaign.sha256(path) for path in scientific_files}
    calls = len(fake_kernel.calls)
    changed = deepcopy(config)
    changed["runtime"].update(batch_size=3, threads=2)
    inspected = campaign.run_campaign(changed, prepare_only=True)
    assert inspected == completed
    assert inspected["status"] == "completed"
    after = {str(path.relative_to(out)): campaign.sha256(path) for path in out.rglob("*")
             if path.is_file() and path.name != ".precision.lock"}
    assert after == hashes
    assert len(fake_kernel.calls) == calls
    repeated = campaign.run_campaign(changed)
    assert repeated == completed
    assert {str(path.relative_to(out)): campaign.sha256(path) for path in out.rglob("*")
            if path.is_file() and path.name != ".precision.lock"} == hashes
    assert len(fake_kernel.calls) == calls
    aggregate = out / "spherical.npz"
    original = aggregate.read_bytes()
    aggregate.write_bytes(original + b"corruption")
    with pytest.raises(ValueError, match="aggregate changed"):
        campaign.run_campaign(changed, prepare_only=True)
    aggregate.write_bytes(original)
    changed["orientations"]["n_orientations"] = 8
    extended = campaign.run_campaign(changed, prepare_only=True)
    assert extended["status"] == "prepared" and extended["completed_pairs"] == 10
    assert extended["npairs"]["beta2"] == 8
    assert "output_sha256" not in extended
    assert len(fake_kernel.calls) == calls
