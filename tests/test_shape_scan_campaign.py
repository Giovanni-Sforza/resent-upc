"""Resume/provenance and true-kernel adapter checks for the isolated shape scan."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import upc_deformation.shape_scan_campaign as scan


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    source = yaml.safe_load((scan.ROOT/"configs/diffraction_observables_a100.yaml").read_text())
    source["scan"] = {"beta2_values": [0., .3], "beta3_values": [0., .2],
        "density_variants": [{"id": "none", "type": "none", "mode": "rms_matched",
                               "R_p": 5.02, "a_p": .46, "delta_r_np_fm": 0.}]}
    source["orientations"].update(n_orientations=4, block_size=4, import_bank=None, import_sha256=None)
    source["runtime"].update(device="cpu", batch_size=2, checkpoint_size=3, output=str(tmp_path/"run"))
    source["study"].update(n_radial=19, n_phi=8, b_step_fm=24., n_y=1)
    path = tmp_path/"config.yaml"; path.write_text(yaml.safe_dump(source))
    monkeypatch.setattr(scan, "code_hashes", lambda: {"test": "fixed"})
    return scan.load_config(path)


class FakeRefiner:
    calls = []
    fail_at = None

    def __init__(self, cfg, case):
        self.cfg, self.case = cfg, case

    def calculate(self, r1, r2, return_nodes=False):
        type(self).calls.append((self.case["name"], len(r1)))
        if len(type(self).calls) == self.fail_at:
            raise RuntimeError("simulated kernel failure")
        s = self.cfg["study"]; nr, nb = s["n_radial"], s["b_nodes"]
        q = np.linspace(0, .3, nr)
        j = (3+r1[:, 0, 0]+.2*r2[:, 0, 0])[:, None]*np.exp(-200*q*q)
        coeff = np.zeros((len(r1), 2, 3, nr)); coeff[:, :, 0] = j[:, None]
        levels = np.stack([coeff, 1.01*coeff])
        return {"branch_names": ["full", "self"], "coefficients": coeff,
            "b_level_coefficients": levels, "radii": q, "axis": np.linspace(-.2, .2, 5),
            "p_full_axis": np.linspace(-1, 1, 32), "b_values": np.linspace(12, 60, nb),
            "b_weights": np.ones(nb), "b_steps_fm": np.array([24., 48.]),
            "b_level_weights": np.ones((2, nb)), "y_values": np.zeros(1),
            "y_weights": np.ones(1), "elapsed_seconds": .001}


@pytest.fixture
def fake(monkeypatch):
    FakeRefiner.calls = []; FakeRefiner.fail_at = None
    monkeypatch.setattr(scan, "build_refiner", lambda c, k: (FakeRefiner(c, k), {"test": True}))
    return FakeRefiner


def test_resume_extension_and_schedule_changes_preserve_chunks(cfg, fake):
    out = Path(cfg["runtime"]["output"])
    partial = scan.run_campaign(cfg, max_new_pairs=3)
    assert partial["completed_pairs"] == 3 and partial["status"] == "incomplete"
    frozen = {p: scan.sha256(p) for p in out.glob("cases/*/chunk_*.npz")}
    cfg["runtime"].update(batch_size=3, checkpoint_size=2)
    done = scan.run_campaign(cfg)
    assert done["status"] == "completed" and done["completed_pairs"] == 13
    for path, digest in frozen.items():
        assert scan.sha256(path) == digest
    for c in cfg["study"]["cases"]:
        q, steps, samples, _ = scan.load_case_samples(out, c["name"])
        expected = 1 if c["beta2"] == c["beta3"] == 0 else 4
        assert samples.shape == (expected, 2, 2, len(q)) and q[-1] <= .17
    frozen = {p: scan.sha256(p) for p in out.rglob("*") if p.is_file() and p.name != ".precision.lock"}
    assert scan.run_campaign(cfg, prepare_only=True) == done
    assert all(scan.sha256(p) == h for p, h in frozen.items())
    cfg["orientations"]["n_orientations"] = 8
    extended = scan.run_campaign(cfg)
    assert extended["completed_pairs"] == 25 and extended["new_pairs_this_session"] == 12
    assert extended["fingerprint"] == done["fingerprint"]
    for path, digest in frozen.items():
        if path.name.startswith(("chunk_", "orientation_banks_")):
            assert scan.sha256(path) == digest
    assert sum(n for _, n in fake.calls) == 25


def test_failure_retains_committed_chunks_and_resumes(cfg, fake):
    fake.fail_at = 3
    with pytest.raises(RuntimeError, match="simulated"):
        scan.run_campaign(cfg)
    out = Path(cfg["runtime"]["output"])
    assert json.loads((out/"production_manifest.json").read_text())["status"] == "interrupted"
    frozen = {p: scan.sha256(p) for p in out.glob("cases/*/chunk_*.npz")}
    fake.fail_at = None
    assert scan.run_campaign(cfg)["status"] == "completed"
    assert all(scan.sha256(p) == h for p, h in frozen.items())


def test_tamper_missing_and_changed_identity_rejected(cfg, fake):
    scan.run_campaign(cfg, max_new_pairs=1)
    out = Path(cfg["runtime"]["output"])
    path = next(out.glob("cases/*/chunk_*.npz")); original = path.read_bytes()
    with np.load(path) as z:
        values = dict(z)
    values["A0"][0, 0, 0, 0] += 1
    np.savez(path, **values)
    with pytest.raises(ValueError, match="data/rotation"):
        scan.run_campaign(cfg)
    path.write_bytes(original); path.unlink()
    with pytest.raises(ValueError, match="missing"):
        scan.run_campaign(cfg)
    changed = deepcopy(cfg); changed["study"]["n_y"] = 3
    with pytest.raises(ValueError, match="identity changed"):
        scan.run_campaign(changed)


def test_prepare_no_gpu_needed_and_no_cpu_fallback_on_run(cfg, fake, monkeypatch):
    cfg["runtime"]["device"] = "cuda"
    monkeypatch.setattr(scan.torch.cuda, "is_available", lambda: False)
    assert scan.run_campaign(cfg, prepare_only=True)["status"] == "prepared"
    assert not fake.calls
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        scan.run_campaign(cfg)


def test_selected_cases_and_zero_budget(cfg, fake):
    assert scan.run_campaign(cfg, max_new_pairs=0)["completed_pairs"] == 0
    chosen = cfg["study"]["cases"][-1]["name"]
    result = scan.run_campaign(cfg, selected_cases=[chosen])
    assert result["status"] == "incomplete" and result["completed_pairs"] == 4
    assert {name for name, _ in fake.calls} == {chosen}
    with pytest.raises(ValueError, match="unknown selected"):
        scan.run_campaign(cfg, selected_cases=["unknown"])


def test_identity_binds_grid_but_not_scheduling(cfg):
    _, bank = scan.rotation_bank(cfg["orientations"], 4)
    original = scan.scientific_identity(cfg, bank)[1]
    changed = deepcopy(cfg); changed["runtime"].update(batch_size=64, checkpoint_size=128)
    changed["study"]["y_batch_size"] = 5
    changed["orientations"]["n_orientations"] = 8
    assert scan.scientific_identity(changed, bank)[1] == original
    changed["study"]["qmax"] = .16
    assert scan.scientific_identity(changed, bank)[1] != original


def test_exact_prefix_and_committed_state_guard(cfg, fake):
    scan.run_campaign(cfg, max_new_pairs=3)
    out = Path(cfg["runtime"]["output"])
    name = cfg["study"]["cases"][1]["name"]
    for n in (0, -1, 1.5, True, 4):
        with pytest.raises(ValueError, match="prefix"):
            scan.load_case_samples(out, name, n=n)
    state_path = out/"cases"/name/"manifest.json"
    state = json.loads(state_path.read_text())
    state["density"] = {"test": False}
    scan.atomic_json(state_path, state)
    with pytest.raises(ValueError, match="committed case state changed"):
        scan.run_campaign(cfg)


def test_case_commit_ahead_of_root_is_recovered(cfg, fake, monkeypatch):
    out = Path(cfg["runtime"]["output"])
    scan.run_campaign(cfg, max_new_pairs=2)
    before = json.loads((out/"production_manifest.json").read_text())
    frozen = {p: scan.sha256(p) for p in out.glob("cases/*/chunk_*.npz")}
    writer = scan.atomic_json

    def fail_root(path, data):
        if path.name == "production_manifest.json" and data["completed_pairs"] > before["completed_pairs"]:
            raise RuntimeError("power loss before root commit")
        writer(path, data)

    monkeypatch.setattr(scan, "atomic_json", fail_root)
    with pytest.raises(RuntimeError, match="power loss"):
        scan.run_campaign(cfg)
    monkeypatch.setattr(scan, "atomic_json", writer)
    assert scan.run_campaign(cfg)["status"] == "completed"
    assert all(scan.sha256(path) == digest for path, digest in frozen.items())


def test_source_change_during_run_is_not_marked_completed(cfg, fake, monkeypatch):
    calls = []

    def changing_hashes():
        calls.append(None)
        return {"test": "fixed" if len(calls) == 1 else "changed"}

    monkeypatch.setattr(scan, "code_hashes", changing_hashes)
    with pytest.raises(ValueError, match="source files changed"):
        scan.run_campaign(cfg)
    manifest = json.loads((Path(cfg["runtime"]["output"])/"production_manifest.json").read_text())
    assert manifest["status"] == "interrupted"


def test_real_halo_kernel_adapter_and_saved_A0(cfg):
    cfg["scan"] = {"beta2_values": [0.], "beta3_values": [0.],
        "density_variants": [{"id": "halo", "type": "halo", "mode": "rms_matched",
                               "R_p": 5.02, "a_p": .46, "delta_r_np_fm": .2}]}
    cfg["study"]["cases"] = scan.build_cases(cfg)
    cfg["physics"]["grid_parameters"] = {"WS": {"N": 16, "L": 24.},
        "T_prim": {"N": 16, "L": 24.}, "output": {"N": 48, "L": 160.}}
    cfg["physics"]["output_settings"].update(safe_padding_factor=2, p_limit_q=.1, target_size_q=25)
    cfg["study"].update(n_radial=31, n_phi=16, qmax=.08)
    case = cfg["study"]["cases"][0]
    refiner, density = scan.build_refiner(cfg, case)
    expected = refiner.calculate(np.eye(3)[None], np.eye(3)[None])
    result = scan.run_campaign(cfg)
    assert result["status"] == "completed" and result["completed_pairs"] == 1
    q, _, a0, info = scan.load_case_samples(cfg["runtime"]["output"], case["name"])
    mask = expected["radii"] <= .08
    np.testing.assert_array_equal(a0, expected["b_level_coefficients"][:, :, :, 0, :][:, :, :, mask].transpose(1, 0, 2, 3))
    assert density["a_n_fm"] > density["a_p_fm"]
    assert info["density"]["R_ws_n_fm"] == pytest.approx(info["density"]["R_ws_p_fm"])
