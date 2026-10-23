"""Physical density contracts for scans; no forward amplitudes or production."""
from copy import deepcopy

import pytest

from upc_deformation.observables import case_physics, deformed_ws_rms
from upc_deformation.shape_scan_physics import build_cases, resolve_case


def _variant(kind="skin", mode="rms_matched", **values):
    item = {"id": kind, "type": kind, "mode": mode, "R_p": 5.02, "a_p": .46}
    if mode == "rms_matched":
        item["delta_r_np_fm"] = 0. if kind == "none" else .2
    else:
        item.update(R_n=5.3 if kind == "skin" else 5.02, a_n=.6 if kind == "halo" else .46)
    item.update(values)
    return item


def _config(*variants, beta2=None, beta3=None):
    return {
        "physics": {"nuclear_parameters": {"density_model": "woods_saxon", "Z": 40,
                    "A": 96, "N": 56, "R_ws_p": 5.02, "R_ws_n": 5.02,
                    "d_ws_p": .46, "d_ws_n": .46, "beta2": 0., "beta3": 0.}},
        "scan": {"beta2_values": [0.] if beta2 is None else beta2,
                 "beta3_values": [0.] if beta3 is None else beta3,
                 "density_variants": list(variants) or [_variant()]},
    }


def test_cartesian_product_retains_negative_beta_and_distinct_none_profiles():
    cfg = _config(_variant("none", id="none_a"), _variant("none", id="none_b", R_p=5.1),
                  _variant("halo"), beta2=[-.3, 0., .3], beta3=[0., .2])
    original = deepcopy(cfg)
    cases = build_cases(cfg)
    assert len(cases) == len({case["name"] for case in cases}) == 18
    assert cases[0]["beta2"] == -.3
    assert cases[0]["density_variant"]["R_p"] == 5.02
    assert cases[6]["density_variant"]["R_p"] == 5.1
    cases[0]["density_variant"]["R_p"] = 99
    assert cases[1]["density_variant"]["R_p"] == 5.02
    assert cfg == original


@pytest.mark.parametrize("beta2,beta3", [(0., 0.), (.3, .2), (-.3, .2)])
def test_rms_skin_matches_unchanged_legacy_point(beta2, beta3):
    cfg = _config(beta2=[beta2], beta3=[beta3]); original = deepcopy(cfg)
    case = build_cases(cfg)[0]
    resolved, adapted, info = resolve_case(cfg, case)
    expected, _ = case_physics(cfg, dict(case, skin_fm=.2))
    assert resolved["physics"] == expected
    assert adapted["skin_fm"] == .2
    assert info["a_n_fm"] == info["a_p_fm"] == .46
    assert info["R_ws_n_fm"] > info["R_ws_p_fm"]
    assert info["delta_r_np_fm"] == pytest.approx(.2, abs=2e-10)
    assert cfg == original and "skin_fm" not in case


@pytest.mark.parametrize("beta2,beta3", [(0., 0.), (.3, .2), (-.3, .2)])
def test_rms_halo_changes_diffuseness_and_preserves_radius(beta2, beta3):
    cfg = _config(_variant("halo"), beta2=[beta2], beta3=[beta3])
    resolved, adapted, info = resolve_case(cfg, build_cases(cfg)[0])
    nuc = resolved["physics"]["nuclear_parameters"]
    assert nuc["R_ws_n"] == pytest.approx(nuc["R_ws_p"], abs=2e-9)
    assert nuc["d_ws_n"] > nuc["d_ws_p"]
    assert adapted["skin_fm"] == .2
    assert info["delta_r_np_fm"] == pytest.approx(.2, abs=2e-10)


def test_none_keeps_profiles_equal_but_does_not_fix_proton_rms_across_deformation():
    cfg = _config(_variant("none"), beta2=[0., .3], beta3=[0., .2])
    infos = [resolve_case(cfg, case)[2] for case in build_cases(cfg)]
    for info in infos:
        assert info["R_ws_n_fm"] == pytest.approx(info["R_ws_p_fm"], abs=2e-9)
        assert info["a_n_fm"] == info["a_p_fm"]
        assert info["delta_r_np_fm"] == pytest.approx(0., abs=2e-9)
        assert info["center_of_mass_correction"] is False
        assert info["volume_preserving_radius_correction"] is False
        assert info["shared_proton_neutron_beta"] is True
        assert info["shared_proton_neutron_orientation"] is True
    assert abs(infos[-1]["proton_rms_fm"] - infos[0]["proton_rms_fm"]) > .01


@pytest.mark.parametrize("kind", ["none", "skin", "halo"])
def test_direct_ws_preserves_declared_parameters_at_combined_deformation(kind):
    variant = _variant(kind, "direct_ws")
    cfg = _config(variant, beta2=[-.3], beta3=[.2])
    resolved, adapted, info = resolve_case(cfg, build_cases(cfg)[0])
    nuc = resolved["physics"]["nuclear_parameters"]
    for source, target in [("R_p", "R_ws_p"), ("R_n", "R_ws_n"), ("a_p", "d_ws_p"), ("a_n", "d_ws_n")]:
        assert nuc[target] == pytest.approx(variant[source], abs=2e-9)
    expected_delta = (deformed_ws_rms(variant["R_n"], variant["a_n"], -.3, .2)
                      - deformed_ws_rms(variant["R_p"], variant["a_p"], -.3, .2))
    assert adapted["skin_fm"] == pytest.approx(expected_delta, abs=2e-10)
    assert info["delta_r_np_fm"] == pytest.approx(expected_delta, abs=2e-9)


def test_beta3_sign_preserves_origin_rms_and_resolved_halo_parameters():
    cfg = _config(_variant("halo"), beta2=[.3], beta3=[-.2, .2])
    a, b = [resolve_case(cfg, case)[2] for case in build_cases(cfg)]
    for key in ("proton_rms_fm", "neutron_rms_fm", "R_ws_n_fm", "a_n_fm"):
        assert a[key] == pytest.approx(b[key], abs=2e-11)


@pytest.mark.parametrize("variant", [
    _variant("none", delta_r_np_fm=.1), _variant("halo", delta_r_np_fm=0.),
    _variant("skin", delta_r_np_fm=-.1), _variant("none", "direct_ws", R_n=5.1),
    _variant("skin", "direct_ws", a_n=.5), _variant("halo", "direct_ws", R_n=5.1),
    _variant("halo", "direct_ws", a_n=.46), _variant("skin", R_p=0.),
    _variant("skin", a_p=float("nan")), _variant("skin", R_n=5.3),
    _variant("skin", id="../unsafe"), _variant("skin", type="unknown"),
])
def test_invalid_density_declarations_are_rejected(variant):
    with pytest.raises(ValueError):
        build_cases(_config(variant))


@pytest.mark.parametrize("values", [[], [0., 0.], [True], [float("inf")]])
def test_invalid_beta_axis_is_rejected(values):
    with pytest.raises(ValueError):
        build_cases(_config(beta2=values))


def test_duplicate_variant_ids_and_per_nucleus_overrides_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        build_cases(_config(_variant(), _variant()))
    cfg = _config(); cfg["physics"]["nuclear_1"] = {"beta2": .3}
    with pytest.raises(ValueError, match="per-nucleus"):
        resolve_case(cfg, build_cases(cfg)[0])


def test_direct_radius_outside_legacy_bracket_is_not_silently_changed():
    cfg = _config(_variant("skin", "direct_ws", R_n=12.))
    with pytest.raises(ValueError, match="solver bracket"):
        resolve_case(cfg, build_cases(cfg)[0])
