"""Production mechanics are tested with a cheap deterministic batched kernel."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

import upc_deformation.precision_campaign as campaign


@pytest.fixture
def config(tmp_path, monkeypatch):
    cfg = campaign.load_config(campaign.ROOT / "configs/panel_d_precision_a100.yaml",
        {"device": "cpu", "n_orientations": 4, "b_step_fm": 2., "output": str(tmp_path / "run")})
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
        coeff = np.ones((len(r1), 3, nr)) * (1 + r1[:, 0, 0] + r2[:, 1, 1])[:, None, None]
        coeff += self.case["beta2"] + self.case["beta3"] + self.case["skin_fm"]
        step = study["b_step_fm"]; nb = study["b_nodes"]; levels = np.stack((coeff, coeff*1.01, coeff*1.02))
        return {"coefficients": coeff, "b_level_coefficients": levels,
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


def test_fresh_rotations_have_stable_prefix_independent_of_target():
    settings = {"seed": 1001, "block_size": 7, "import_bank": None}
    small, meta = campaign.rotation_bank(settings, 12)
    large, larger_meta = campaign.rotation_bank(settings, 40)
    assert meta == larger_meta
    for a, b in zip(small, large):
        np.testing.assert_array_equal(a, b[:12])
        np.testing.assert_allclose(a @ a.transpose(0, 2, 1), np.broadcast_to(np.eye(3), a.shape), atol=2e-15)
        np.testing.assert_allclose(np.linalg.det(a), 1, atol=2e-15)
    assert not np.array_equal(small[0], small[1])


def test_imported_prefix_preserves_values_and_extends_without_resampling(tmp_path):
    original, _ = campaign.rotation_bank({"seed": 9, "block_size": 4}, 8)
    path = tmp_path / "prefix.npz"
    np.savez(path, rotation_bank_1=original[0].astype(np.float32), rotation_bank_2=original[1].astype(np.float32))
    settings = {"seed": 1001, "block_size": 3, "import_bank": str(path), "import_sha256": campaign.sha256(path)}
    small, meta = campaign.rotation_bank(settings, 12)
    large, _ = campaign.rotation_bank(settings, 20)
    assert meta["imported_prefix_count"] == 8
    for j in (0, 1):
        np.testing.assert_array_equal(small[j][:8], original[j].astype(np.float32))
        np.testing.assert_array_equal(small[j], large[j][:12])
    with pytest.raises(ValueError, match="hash"):
        campaign.rotation_bank({**settings, "import_sha256": "wrong"}, 12)


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
            assert z["orientation_coefficients"].shape == (n, 3, 9)
            assert z["b_level_coefficients"].shape == (3, n, 3, 9)
            np.testing.assert_array_equal(z["orientation_coefficients"], z["b_level_coefficients"][0])
            assert z["rotation_bank_1"].shape == (n, 3, 3)
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


def test_output_lock_rejects_second_writer_and_releases_on_exception(tmp_path):
    with pytest.raises(ValueError, match="intentional"):
        with campaign.OutputLock(tmp_path):
            with pytest.raises(RuntimeError, match="Another process"):
                with campaign.OutputLock(tmp_path): pass
            raise ValueError("intentional")
    with campaign.OutputLock(tmp_path): pass


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
        assert row["coefficients"].shape == (3, 31)
        assert row["b_level_coefficients"].shape == (2, 3, 31)
    with np.load(out / "grid.npz") as grid:
        np.testing.assert_array_equal(grid["b_values"], [12., 36., 60.])
        np.testing.assert_array_equal(grid["b_steps_fm"], [24., 48.])
